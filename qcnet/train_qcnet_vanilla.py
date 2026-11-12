"""QCNet Vanilla 학습 (단일 GPU, seed 1).

  학습 = init, 검증 = val (best 선택: val_minFDE, K=6), test 는 학습 중에 쓰지 않는다 (datamodule 이 처리만 해 둔다).
  모델 인자 (hparams.py): 관측 10 / 예측 30, recurrent 3, 반경 150/50/50/150/150, time_span 10.
  lr 5e-4, weight_decay 1e-4 (QCNet 공식), cosine T_max = max_epochs.
  --config 를 주면 그 json 의 model / optimizer 절이 hparams.py 기본값을 덮어쓴다.

사용
  python train_qcnet_vanilla.py --site ss --config ../configs/qcnet_ss_ours13.json --out ../runs/qcnet/ss/vanilla
출력: <out>/ (기본 checkpoint/<site>_vanilla/) — Lightning 체크포인트 best 2 개 (epoch=NNN-val_minFDE=X.ckpt) + last.ckpt, 로그
  --stop_epoch 200 : cosine 은 max_epochs(300) 기준 그대로 두고 200 epoch 에서 멈추며 snapshot_2of3.ckpt 를 남긴다.
"""
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from datamodules import ArgoverseV2DataModule_ACL
from hparams import model_args
from predictors import QCNet

HERE = os.path.dirname(os.path.abspath(__file__))


class Snapshot(Callback):
    """epoch `at` 이 끝났을 때 (검증 포함) 모델을 snapshot_2of3.ckpt 로 저장."""

    def __init__(self, path, at):
        self.path, self.at = path, at

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch + 1 == self.at:
            trainer.save_checkpoint(self.path)

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--site', required=True, choices=['ss', 'ny'])
    ap.add_argument('--config', default=None, help='configs/*.json (model / optimizer 절)')
    ap.add_argument('--data_root', default=os.path.join(HERE, '..', 'data', 'qcnet'))
    ap.add_argument('--max_epochs', type=int, default=300)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default=None)
    ap.add_argument('--stop_epoch', type=int, default=None, help='여기서 멈추고 snapshot_2of3.ckpt 저장')
    ap.add_argument('--resume', action='store_true', help='out/last.ckpt 에서 이어서')
    a = ap.parse_args()
    pl.seed_everything(a.seed, workers=True)

    root = os.path.abspath(os.path.join(a.data_root, a.site))
    proc = os.path.join(root, 'processed')
    out = a.out or os.path.join(HERE, 'checkpoint', f'{a.site}_vanilla')
    os.makedirs(out, exist_ok=True)
    args = dict(
        root=root, train_batch_size=a.batch_size, val_batch_size=a.batch_size, test_batch_size=a.batch_size,
        shuffle=True, num_workers=a.num_workers, pin_memory=True, persistent_workers=True,
        train_raw_dir=os.path.join(root, 'init', 'raw'), val_raw_dir=os.path.join(root, 'val', 'raw'),
        test_raw_dir=os.path.join(root, 'test', 'raw'),
        train_processed_dir=os.path.join(proc, 'init'), val_processed_dir=os.path.join(proc, 'val'),
        test_processed_dir=os.path.join(proc, 'test'),
        **model_args(a.config, out, a.max_epochs),
    )
    model = QCNet(**args)
    dm = ArgoverseV2DataModule_ACL(**args)
    ckpt = ModelCheckpoint(dirpath=out, monitor='val_minFDE', save_top_k=2, mode='min', save_last=True,
                           filename='{epoch:03d}-{val_minFDE:.4f}')
    cbs = [ckpt, LearningRateMonitor(logging_interval='epoch')]
    if a.stop_epoch:
        cbs.append(Snapshot(os.path.join(out, 'snapshot_2of3.ckpt'), a.stop_epoch))
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=a.stop_epoch or a.max_epochs, deterministic=False,
                         callbacks=cbs,
                         logger=CSVLogger(out, name='logs'))
    last = os.path.join(out, 'last.ckpt')
    trainer.fit(model, dm, ckpt_path=last if a.resume and os.path.exists(last) else None)
