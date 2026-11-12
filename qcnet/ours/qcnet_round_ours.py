"""QCNet Ours 라운드 하나 = 라운드 학습 (qcnet_round.py) + CP contrastive (../contrastive/qcnet_contrastive.py).

  난이도 : init 장면 = manifest 의 qcnet_vanilla_cp_max, 생성 장면 = 같은 정의로 QCNet Vanilla top-1 예측의 CP
           (--gen_cp: 라운드별 <round>/gen_cp.csv, 열 scenario_id, cp)
  contrastive 값 : λ 1, τ 0.5, θ_p 0.33, θ_n 0.50 (Contrastive(CP) 학습과 같음, ../hparams.py / --config 의 contrastive 절)
"""
import os
import sys

import pandas as pd

OURS = os.path.dirname(os.path.abspath(__file__))
QC = os.path.dirname(OURS)
sys.path.insert(0, OURS)
sys.path.insert(0, QC)
sys.path.insert(0, os.path.join(QC, 'contrastive'))
import hparams  # noqa: E402
import qcnet_round as R  # noqa: E402
from qcnet_contrastive import QCNet_Contrastive  # noqa: E402


class QCNet_Contrastive_Gen(QCNet_Contrastive):
    gen_cp = []

    def on_fit_start(self):
        super().on_fit_start()
        for p in self.gen_cp:
            g = pd.read_csv(p, dtype={'scenario_id': str})
            self._cp.update(zip(g['scenario_id'], g['cp'].astype(float)))


def build(a, args):
    cls = R.with_round_lr(QCNet_Contrastive_Gen, a.epoch_offset, a.total_epochs)
    cls.gen_cp = [p for p in a.gen_cp.split(',') if p]
    return cls(cp_table=os.path.abspath(os.path.join(QC, '..', 'data', 'scenarios', f'manifest_{a.site}.csv')),
               **hparams.contrastive_args(a.config), **args)


if __name__ == '__main__':
    ap = R.parser()
    ap.add_argument('--gen_cp', default='', help='생성 장면 CP csv 를 쉼표로')
    R.run(ap.parse_args(), build)
