"""QCNet 라운드 하나 학습.

  학습 데이터 : init + 라운드 1..r 생성분 (gen_to_qcnet.py raw). 가중 복원추출, epoch 당 샘플 수 = init 크기
  샘플 가중치 : a 라운드에 들어온 데이터의 r 라운드 가중치 = max(alpha^(r-a), clip), init 은 a = 0 (GALTraj Sec 3.5)
               --gen_frac f > 0 이면 init 가중치 1, 생성분 가중치 합이 추출의 f 가 되도록 줄인다 (생성분끼리 비는 그대로)
               --gen_frac_exact 면 생성분이 f 보다 적을 때도 가중치를 올려 f 로 맞춘다
  pad 마스크  : 생성 장면에서 time-shift pad 스텝은 행이 없어 predict_mask 가 회귀 손실에서 뺀다
  시작        : --init_ckpt 의 가중치만 (optimizer 는 라운드마다 새로)
  lr          : Vanilla 와 같은 cosine (5e-4, T_max 300) 을 전체 epoch 기준으로 이어 쓴다 (epoch_offset + k)
  라운드 best : val_minFDE (K=6) 최소 → <out>/best.ckpt
               --select 'val_minADE+val_minADE_Worst3%' 처럼 주면 매 epoch 저장 후 val 지표 합이 최소인 epoch 을 best 로

모델은 build_model(a, args) 로 만든다. Ours 는 qcnet_round_ours.py 가 build_model 을 바꿔 부른다.
모델·옵티마이저 값은 ../hparams.py 기본값, --config 를 주면 그 json 의 model / optimizer 절.
"""
import math
import os
import sys
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from torch_geometric.loader import DataLoader

OURS = os.path.dirname(os.path.abspath(__file__))
QC = os.path.dirname(OURS)
sys.path.insert(0, QC)
from datasets import ArgoverseV2Dataset_ACL  # noqa: E402
from hparams import model_args  # noqa: E402
from predictors import QCNet  # noqa: E402
from transforms import TargetBuilder  # noqa: E402


def make_ds(root, split, raw, proc, transform):
    ds = ArgoverseV2Dataset_ACL(root, split, raw, proc, transform)
    if len(ds) == 0:   # 처음 처리하면 샘플 수가 처리 전 목록으로 잡힌다 → 한 번 더 (datamodule 의 prepare/setup 과 같음)
        ds = ArgoverseV2Dataset_ACL(root, split, raw, proc, transform)
    return ds


def with_round_lr(cls, offset, total):
    class Round(cls):
        def configure_optimizers(self):
            opts, _ = super().configure_optimizers()
            f = lambda k: 0.5 * (1 + math.cos(math.pi * min(offset + k, total) / total))  # noqa: E731
            return opts, [torch.optim.lr_scheduler.LambdaLR(opts[0], f)]
    Round.__name__ = cls.__name__
    return Round


class RoundData(pl.LightningDataModule):
    def __init__(self, train_parts, val_ds, weights, num_samples, batch_size, num_workers, seed):
        super().__init__()
        self.train_ds = ConcatDataset(train_parts)
        self.val_ds, self.w, self.n = val_ds, torch.as_tensor(weights, dtype=torch.double), num_samples
        self.bs, self.nw, self.seed = batch_size, num_workers, seed

    def train_dataloader(self):
        g = torch.Generator().manual_seed(self.seed + int(self.trainer.current_epoch if self.trainer else 0))
        sampler = WeightedRandomSampler(self.w, self.n, replacement=True, generator=g)
        return DataLoader(self.train_ds, batch_size=self.bs, sampler=sampler, num_workers=self.nw,
                          pin_memory=True, persistent_workers=self.nw > 0)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.bs, shuffle=False, num_workers=self.nw,
                          pin_memory=True, persistent_workers=self.nw > 0)


def build_model(a, args):
    return with_round_lr(QCNet, a.epoch_offset, a.total_epochs)(**args)


def parser():
    ap = ArgumentParser()
    ap.add_argument('--site', required=True, choices=['ss', 'ny'])
    ap.add_argument('--config', default=None, help='configs/*.json (model / optimizer / contrastive 절)')
    ap.add_argument('--data_root', default=os.path.join(QC, '..', 'data', 'qcnet'))
    ap.add_argument('--gen', default='', help='라운드:폴더 를 쉼표로 (폴더 안에 raw/, proc/)')
    ap.add_argument('--round', type=int, required=True)
    ap.add_argument('--alpha', type=float, default=0.7)
    ap.add_argument('--clip', type=float, default=0.5)
    ap.add_argument('--gen_frac', type=float, default=-1)
    ap.add_argument('--gen_frac_exact', action='store_true', help='생성분 비율이 gen_frac 보다 작아도 가중치를 올려 정확히 맞춘다')
    ap.add_argument('--init_ckpt', required=True)
    ap.add_argument('--epochs', type=int, required=True)
    ap.add_argument('--epoch_offset', type=int, default=200)
    ap.add_argument('--total_epochs', type=int, default=300)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--select', default='', help="라운드 best 기준 val 지표 합 (예: val_minADE+val_minADE_Worst3%%). 비우면 val_minFDE")
    return ap


def run(a, build=build_model):
    pl.seed_everything(a.seed, workers=True)
    root = os.path.abspath(os.path.join(a.data_root, a.site))
    proc = os.path.join(root, 'processed')
    os.makedirs(a.out_dir, exist_ok=True)
    tb = TargetBuilder(10, 30)
    parts = [make_ds(root, 'train', os.path.join(root, 'init', 'raw'), os.path.join(proc, 'init'), tb)]
    src = [0]
    for item in filter(None, a.gen.split(',')):
        q, d = item.split(':', 1)
        parts.append(make_ds(root, 'train', os.path.join(d, 'raw'), os.path.join(d, 'proc'), tb))
        src.append(int(q))
    w = np.concatenate([np.full(len(p), max(a.alpha ** (a.round - s), a.clip)) for p, s in zip(parts, src)])
    is_gen = np.concatenate([np.full(len(p), s > 0) for p, s in zip(parts, src)])
    if a.gen_frac > 0:
        w[~is_gen] = 1.0
        if is_gen.any():
            target = a.gen_frac / (1.0 - a.gen_frac) * (~is_gen).sum()
            if w[is_gen].sum() > target or a.gen_frac_exact:
                w[is_gen] *= target / w[is_gen].sum()
    n_init = len(parts[0])
    print(f'[round {a.round}] init {n_init:,} + 생성 {int(is_gen.sum()):,} 장면 | 생성분 추출 비율 '
          f'{w[is_gen].sum() / w.sum():.3f} | epoch {a.epoch_offset + 1}~{a.epoch_offset + a.epochs}', flush=True)
    val = make_ds(root, 'val', os.path.join(root, 'val', 'raw'), os.path.join(proc, 'val'), tb)
    dm = RoundData(parts, val, w, n_init, a.batch_size, a.num_workers, a.seed)

    model = build(a, model_args(a.config, a.out_dir, a.total_epochs))
    sd = torch.load(a.init_ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(sd.get('state_dict', sd))
    if a.select:
        ckpt = ModelCheckpoint(dirpath=a.out_dir, save_top_k=-1, every_n_epochs=1, filename='ep{epoch:03d}',
                               auto_insert_metric_name=False)
    else:
        ckpt = ModelCheckpoint(dirpath=a.out_dir, monitor='val_minFDE', save_top_k=1, mode='min', save_last=True,
                               filename='best')
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=a.epochs, deterministic=False,
                         reload_dataloaders_every_n_epochs=1,
                         callbacks=[ckpt, LearningRateMonitor(logging_interval='epoch')],
                         logger=CSVLogger(a.out_dir, name='logs'))
    trainer.fit(model, dm)
    if a.select:
        select_best(a, trainer)
        return
    with open(os.path.join(a.out_dir, 'best.txt'), 'w', encoding='utf-8') as f:
        f.write(f'{ckpt.best_model_path} val_minFDE {float(ckpt.best_model_score):.4f}\n')
    print(f'[round {a.round}] best val_minFDE {float(ckpt.best_model_score):.4f} -> {ckpt.best_model_path}', flush=True)


def select_best(a, trainer):
    import glob
    import shutil
    import pandas as pd
    keys = a.select.split('+')
    m = pd.read_csv(os.path.join(trainer.logger.log_dir, 'metrics.csv')).dropna(subset=keys)
    m = m.assign(score=m[keys].sum(axis=1))
    b = m.loc[m['score'].idxmin()]
    ep = int(b['epoch'])
    best = os.path.join(a.out_dir, 'best.ckpt')
    shutil.copy(os.path.join(a.out_dir, f'ep{ep:03d}.ckpt'), best)
    for p in glob.glob(os.path.join(a.out_dir, 'ep*.ckpt')):
        os.remove(p)
    with open(os.path.join(a.out_dir, 'best.txt'), 'w', encoding='utf-8') as f:
        f.write(f'{best} val_sel {b["score"]:.4f} epoch {a.epoch_offset + ep + 1} '
                + ' '.join(f'{k} {b[k]:.4f}' for k in keys) + '\n')
    print(f'[round {a.round}] best {a.select} {b["score"]:.4f} (epoch {a.epoch_offset + ep + 1}) -> {best}', flush=True)


if __name__ == '__main__':
    run(parser().parse_args())
