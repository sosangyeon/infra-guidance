# -*- coding: utf-8 -*-
"""TrajectoryDataset + cp_max 전달.

cp_table(scenario_id 열이 있는 표, 예: data/scenarios/manifest_<site>.csv)의 cp_col 값을
시나리오의 모든 샘플에 붙인다 (연구실 merge_cp_max.py 와 같음: 시나리오 CP 하나를 그 시나리오 전체에).
배치 끝에 cp [B, 1] 이 붙는다. cp_col 을 안 주면 dataset.py 와 똑같이 동작한다.
"""
import numpy as np

from utils.dataset import TrajectoryDataset as _BaseDataset


class TrajectoryDataset(_BaseDataset):
    def __init__(self, *args, cp_table=None, cp_col=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not cp_col:
            return
        if self.meta is None:
            raise ValueError("cp 를 붙이려면 _meta.csv 가 필요함")
        meta = self.meta.merge(cp_table[["scenario_id", cp_col]], on="scenario_id", how="left")
        if meta[cp_col].isna().any():
            raise ValueError(f"{cp_col} 이 없는 시나리오 {meta[cp_col].isna().sum()} 샘플")
        self.meta = meta
        self.extra_cols = [cp_col]
        cp = meta[[cp_col]].to_numpy(dtype=np.float32)
        self.scenario_list = [tuple(s[:3]) + (c,) for s, c in zip(self.scenario_list, cp)]
