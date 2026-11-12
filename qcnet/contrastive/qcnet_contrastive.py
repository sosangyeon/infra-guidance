"""QCNet Contrastive(CP) — 연구실 QCNet_Contrastive_CP (predictors/qcnet_contrastive_cp.py) 를 그대로 쓰고,
난이도만 parquet 의 cp_max 열 대신 manifest 의 시나리오 CP 열에서 가져온다.

  난이도 = qcnet_vanilla_cp_max : init 만으로 학습한 QCNet Vanilla best 의 top-1 예측으로 계산한 시나리오 CP_max
           (data_prep/compute_cp.py --source pred --prefix qcnet_vanilla). 시나리오 값 하나를 그 시나리오의 모든 에이전트에 붙인다
           (TUTR Contrastive 와 같은 규칙).
"""
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predictors import QCNet_Contrastive_CP  # noqa: E402


class QCNet_Contrastive(QCNet_Contrastive_CP):

    def __init__(self, cp_table: str = None, cp_col: str = 'qcnet_vanilla_cp_max', **kwargs) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.cp_table, self.cp_col = cp_table, cp_col
        self._cp = None

    def on_fit_start(self):
        m = pd.read_csv(self.cp_table, dtype={'scenario_id': str})
        m = m[m[self.cp_col].notna()]
        self._cp = dict(zip(m['scenario_id'], m[self.cp_col].astype(float)))

    def training_step(self, data, batch_idx):
        cp = torch.tensor([self._cp[str(s)] for s in data['scenario_id']], dtype=torch.float, device=self.device)
        data['agent']['difficulty'] = cp[data['agent']['batch']]
        return super().training_step(data, batch_idx)
