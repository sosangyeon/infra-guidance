"""Ours 학습셋: GalDataset(init + 라운드별 생성분) + 샘플별 CP 난이도.

샘플 = (hist, future, neighbor, fut_valid, cp[1])
  cp: init 은 manifest 의 cp_col (Vanilla 예측으로 계산한 시나리오 CP_max, contrastive 와 같은 값),
      생성분은 <gen pkl 이름>_cp.csv (gen_cp.py: 같은 Vanilla 로 생성 장면을 예측해 같은 식으로 계산).
  시나리오 CP 하나를 그 시나리오(생성 장면)의 모든 샘플에 붙인다 (연구실 merge_cp_max.py 와 같음).
배치 = GalDataset 배치 (ped, neis, mask, fut_valid) + cp [B, 1].
"""
import numpy as np
import pandas as pd
import torch

from gal_dataset import GalDataset


class OursDataset(GalDataset):
    def __init__(self, dataset_path, dataset_name, base_splits, gen_pkls=(), cp_table=None,
                 cp_col="vanilla_cp_max", **kwargs):
        super().__init__(dataset_path, dataset_name, base_splits, gen_pkls, **kwargs)
        tables = [cp_table[["scenario_id", cp_col]].rename(columns={cp_col: "cp"})]
        for _, path in gen_pkls:
            tables.append(pd.read_csv(path[:-len(".pkl")] + "_cp.csv")[["scenario_id", "cp"]])
        cp = self.meta[["scenario_id"]].merge(pd.concat(tables, ignore_index=True), on="scenario_id", how="left")
        if len(cp) != len(self.meta):
            raise ValueError("CP 표에 scenario_id 중복")
        if cp["cp"].isna().any():
            bad = cp.loc[cp["cp"].isna(), "scenario_id"].unique()[:5]
            raise ValueError(f"CP 가 없는 시나리오 {cp['cp'].isna().sum()} 샘플 (예: {list(bad)})")
        self.meta["cp"] = cp["cp"].to_numpy()
        vals = self.meta[["cp"]].to_numpy(dtype=np.float32)
        self.scenario_list = [tuple(s[:4]) + (c,) for s, c in zip(self.scenario_list, vals)]

    def coll_fn(self, scenario_list):
        out = super().coll_fn(scenario_list)
        cp = torch.tensor(np.stack([s[4] for s in scenario_list]), dtype=torch.float32)
        return out + (cp,)
