"""GAL 학습셋: 기존 분할(init) + 라운드별 LCSim 생성 pkl.

샘플 = (hist[10, 8], future[30, 8], neighbor[40, N, 8], fut_valid[30])
  fut_valid: 미래 프레임별 1/0. 생성분은 random time-window shift 로 뒤쪽 δt 프레임이 pad 라 0
  (GALTraj Eq.6 "zero-pad and mask"). 기존 데이터는 모두 1.
배치 = dataset.TrajectoryDataset.coll_fn 출력 + fut_valid [B, 30].

self.source: 샘플별 출처. 0 = 기존 분할, r = r 라운드에 추가된 생성분. 가중치 계산에 쓴다.
"""
import numpy as np
import pandas as pd
import torch

from dataset import TrajectoryDataset, _load_pickle


class GalDataset(TrajectoryDataset):
    def __init__(self, dataset_path, dataset_name, base_splits, gen_pkls=(), **kwargs):
        """gen_pkls: [(라운드 번호, pkl 경로), ...]. 각 pkl 옆에 <pkl 이름>_meta.csv."""
        super().__init__(dataset_path, dataset_name, base_splits, **kwargs)
        pred_len = self.scenario_list[0][1].shape[0]
        ones = np.ones(pred_len, dtype=np.float32)
        self.scenario_list = [tuple(s[:3]) + (ones,) for s in self.scenario_list]
        source = [np.zeros(len(self.scenario_list), dtype=np.int64)]
        metas = [self.meta.assign(source=0)] if self.meta is not None else []
        for r, path in gen_pkls:
            data = _load_pickle(path)
            meta = pd.read_csv(path[:-len(".pkl")] + "_meta.csv")
            if len(meta) != len(data):
                raise ValueError(f"{path}: meta {len(meta)} rows, pkl {len(data)} samples")
            self.scenario_list.extend(data)
            source.append(np.full(len(data), r, dtype=np.int64))
            metas.append(meta.assign(source=r))
        self.source = np.concatenate(source)
        if metas:
            self.meta = pd.concat(metas, ignore_index=True)
            self.meta["idx"] = np.arange(len(self.meta))

    def sample_weights(self, cur_round, alpha, clip):
        """GALTraj Sec 3.5: 라운드마다 기존 데이터 가중치 ×alpha (하한 clip), 이번 라운드 생성분 1.0.
        a 라운드에 들어온 데이터의 r 라운드 가중치 = max(alpha^(r-a), clip). 기존 분할은 a = 0."""
        age = cur_round - self.source
        if (age < 0).any():
            raise ValueError("이번 라운드보다 나중 라운드의 생성분이 섞여 있음")
        return np.maximum(alpha ** age, clip)

    def coll_fn(self, scenario_list):
        out = super().coll_fn(scenario_list)
        valid = torch.tensor(np.stack([s[3] for s in scenario_list]), dtype=torch.float32)
        return out + (valid,)
