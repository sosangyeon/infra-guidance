"""QCNet Contrastive(CP) 학습 (단일 GPU, seed 1). ../train_qcnet_vanilla.py 와 같은 데이터·모델 인자·일정
(init 만, 300 epoch, cosine T_max 300, val_minFDE 로 best) 에 contrastive 항만 더한다.

  contrastive 값 (../hparams.py): λ 1, τ 0.5, θ_p 0.33, θ_n 0.50, 난이도 = manifest 의 qcnet_vanilla_cp_max
  --config 를 주면 그 json 의 model / optimizer / contrastive 절이 기본값을 덮어쓰고, 명령행 인자가 그 위에 온다.
  200 epoch 에 snapshot_2of3.ckpt (그 시점 모델), best_2of3.ckpt (0~200 val_minFDE best) 저장
  → best_2of3 = Ours (200~300) 시작점, 0~300 best = Contrastive(CP) 기준선.

사용
  python contrastive/train_qcnet_contrastive.py --site ss --config ../configs/qcnet_ss_ours13.json --out ../runs/qcnet/ss/contrastive
출력: <out>/ (기본 checkpoint/<site>_contrastive/)
"""
import os
import sys
from argparse import ArgumentParser

CON = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(CON)
sys.path.insert(0, HERE)
sys.path.insert(0, CON)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import hparams  # noqa: E402
from datamodules import ArgoverseV2DataModule_ACL  # noqa: E402
from qcnet_contrastive import QCNet_Contrastive  # noqa: E402


class TwoThirds(Callback):
    """0~at epoch 의 val_minFDE best 를 best_2of3.ckpt, epoch at 끝의 모델을 snapshot_2of3.ckpt 로 저장."""

    def __init__(self, out, at):
        self.out, self.at, self.best = out, at, float('inf')

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.current_epoch >= self.at:
            return
        v = trainer.callback_metrics.get('val_minFDE')
        if v is not None and float(v) < self.best:
            self.best = float(v)
            trainer.save_checkpoint(os.path.join(self.out, 'best_2of3.ckpt'))
            with open(os.path.join(self.out, 'best_2of3.txt'), 'w', encoding='utf-8') as f:
                f.write(f'epoch {trainer.current_epoch} val_minFDE {self.best:.4f}\n')

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch + 1 == self.at:
            trainer.save_checkpoint(os.path.join(self.out, 'snapshot_2of3.ckpt'))


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--site', required=True, choices=['ss', 'ny'])
    ap.add_argument('--config', default=None, help='configs/*.json (model / optimizer / contrastive 절)')
    ap.add_argument('--data_root', default=os.path.join(HERE, '..', 'data', 'qcnet'))
    ap.add_argument('--cp_col', default=None)
    ap.add_argument('--max_epochs', type=int, default=300)
    ap.add_argument('--snap_epoch', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--lambda_contrastive', type=float, default=None)
    ap.add_argument('--tau', type=float, default=None)
    ap.add_argument('--theta_p', type=float, default=None)
    ap.add_argument('--theta_n', type=float, default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    pl.seed_everything(a.seed, workers=True)

    root = os.path.abspath(os.path.join(a.data_root, a.site))
    proc = os.path.join(root, 'processed')
    out = a.out or os.path.join(HERE, 'checkpoint', f'{a.site}_contrastive')
    os.makedirs(out, exist_ok=True)
    args = dict(
        root=root, train_batch_size=a.batch_size, val_batch_size=a.batch_size, test_batch_size=a.batch_size,
        shuffle=True, num_workers=a.num_workers, pin_memory=True, persistent_workers=True,
        train_raw_dir=os.path.join(root, 'init', 'raw'), val_raw_dir=os.path.join(root, 'val', 'raw'),
        test_raw_dir=os.path.join(root, 'test', 'raw'),
        train_processed_dir=os.path.join(proc, 'init'), val_processed_dir=os.path.join(proc, 'val'),
        test_processed_dir=os.path.join(proc, 'test'),
        **hparams.model_args(a.config, out, a.max_epochs),
    )
    con = hparams.contrastive_args(a.config)
    con.update({k: getattr(a, k) for k in con if getattr(a, k) is not None})   # 명령행 인자가 우선
    model = QCNet_Contrastive(cp_table=os.path.abspath(os.path.join(HERE, '..', 'data', 'scenarios',
                                                                    f'manifest_{a.site}.csv')),
                              **con, **args)
    dm = ArgoverseV2DataModule_ACL(**args)
    ckpt = ModelCheckpoint(dirpath=out, monitor='val_minFDE', save_top_k=2, mode='min', save_last=True,
                           filename='{epoch:03d}-{val_minFDE:.4f}')
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=a.max_epochs, deterministic=False,
                         callbacks=[ckpt, TwoThirds(out, a.snap_epoch), LearningRateMonitor(logging_interval='epoch')],
                         logger=CSVLogger(out, name='logs'))
    trainer.fit(model, dm)
