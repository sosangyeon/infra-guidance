from torch.utils import data
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

from feature_utils import FULL_STATE_DIM, recompute_scalar_features_np


def _load_pickle(path):
    """Read a scenario pickle regardless of which numpy wrote it.

    numpy 2.0 moved numpy.core to numpy._core, and the module path is baked into the
    pickle. A file written under numpy>=2 therefore fails to load under numpy 1.x with
    "No module named 'numpy._core'". Aliasing the old layout under the new names lets
    either numpy read either file; nothing is monkey-patched unless the load fails.
    """
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as e:
            if "numpy._core" not in str(e):
                raise
        f.seek(0)
        import numpy.core as _c
        sys.modules.setdefault("numpy._core", _c)
        for sub in ("numeric", "multiarray", "umath", "_multiarray_umath",
                    "numerictypes", "_methods"):
            try:
                sys.modules.setdefault(
                    "numpy._core." + sub,
                    __import__("numpy.core." + sub, fromlist=[sub]))
            except Exception:
                pass
        return pickle.load(f)


class TrajectoryDataset(data.Dataset):
    """Dataset consuming a common 8-D pkl.

    The model always receives the full state. Feature selection is deliberately
    deferred to model.py through --input_mode.
    """

    def __init__(
        self,
        dataset_path,
        dataset_name,
        dataset_type,
        translation=False,
        rotation=False,
        scaling=False,
        obs_len=8,
        max_neis_num=50,
        dist_threshold=2,
        smooth=False,
        return_transform=False,
        sid_file=None,
        extra_cols=None,
    ):
        """dataset_type 은 분할 이름 하나 또는 목록이다 (예: ["init", "remain"]).
        목록이면 <name>_<type>.pkl 을 순서대로 이어 붙인다.

        각 pkl 옆에 <name>_<type>_meta.csv (data_prep/make_tutr_pkl.py) 가 있으면
        함께 읽어 self.meta 에 둔다 (샘플과 같은 순서).
          sid_file    scenario_id 한 줄에 하나인 텍스트 파일. 그 시나리오의 샘플만 남긴다.
          extra_cols  meta 의 열 이름 목록. 샘플마다 그 값을 4번째 원소로 붙이고,
                      coll_fn 은 배치 끝에 [B, len(extra_cols)] 텐서를 하나 더 돌려준다.
        """
        self.translation = translation
        self.rotation = rotation
        self.scaling = scaling
        self.obs_len = obs_len
        self.max_neis_num = max_neis_num
        self.dist_threshold = dist_threshold
        self.smooth = smooth
        self.return_transform = return_transform
        self.window_size = 3

        types = [dataset_type] if isinstance(dataset_type, str) else list(dataset_type)
        self.scenario_list, metas = [], []
        for t in types:
            pkl_path = os.path.join(dataset_path, f"{dataset_name}_{t}.pkl")
            data = _load_pickle(pkl_path)
            meta_path = os.path.join(dataset_path, f"{dataset_name}_{t}_meta.csv")
            if os.path.exists(meta_path):
                m = pd.read_csv(meta_path)
                if len(m) != len(data):
                    raise ValueError(f"{meta_path}: {len(m)} rows, pkl has {len(data)} samples")
                metas.append(m)
            self.scenario_list.extend(data)
        self.meta = None
        if metas and len(metas) == len(types):
            self.meta = pd.concat(metas, ignore_index=True)
            self.meta["idx"] = np.arange(len(self.meta))
        elif sid_file or extra_cols:
            raise ValueError("sid_file / extra_cols 를 쓰려면 모든 분할에 _meta.csv 가 있어야 함")

        if sid_file:
            with open(sid_file, encoding="utf-8") as f:
                keep = {line.strip() for line in f if line.strip()}
            sel = self.meta["scenario_id"].isin(keep).to_numpy()
            self.scenario_list = [s for s, k in zip(self.scenario_list, sel) if k]
            self.meta = self.meta[sel].reset_index(drop=True)
            self.meta["idx"] = np.arange(len(self.meta))

        self.extra_cols = list(extra_cols) if extra_cols else []
        if self.extra_cols:
            missing = [c for c in self.extra_cols if c not in self.meta.columns]
            if missing:
                raise ValueError(f"meta 에 없는 열: {missing}")
            extra = self.meta[self.extra_cols].to_numpy(dtype=np.float32)
            self.scenario_list = [tuple(s[:3]) + (e,) for s, e in zip(self.scenario_list, extra)]

        if not self.scenario_list:
            raise ValueError(f"No scenarios in {dataset_name}_{types}")

        hist_dim = self.scenario_list[0][0].shape[-1]
        fut_dim = self.scenario_list[0][1].shape[-1]
        if hist_dim != FULL_STATE_DIM or fut_dim != FULL_STATE_DIM:
            raise ValueError(
                "This pkl is not the new common 8-D format. Re-run get_data_pkl.py. "
                f"Expected hist/future dim={FULL_STATE_DIM}, got {hist_dim}/{fut_dim}."
            )

    @staticmethod
    def _apply_rotation(arr, rot_mat):
        """Apply row-vector rotation to position, velocity and acceleration vectors."""
        arr[..., 0:2] = np.matmul(arr[..., 0:2], rot_mat)
        arr[..., 2:4] = np.matmul(arr[..., 2:4], rot_mat)
        arr[..., 5:7] = np.matmul(arr[..., 5:7], rot_mat)
        return arr

    @staticmethod
    def _apply_scaling(arr, scale):
        # Scale all vector-valued quantities. Scalar magnitudes are recomputed later.
        arr[..., 0:2] *= scale
        arr[..., 2:4] *= scale
        arr[..., 5:7] *= scale
        return arr

    def coll_fn(self, scenario_list):
        ped_list, neis_list, neighbor_counts = [], [], []
        origins, rot_mats, scales = [], [], []
        extras = [s[3] for s in scenario_list] if self.extra_cols else None

        for ped_obs_traj, ped_pred_traj, neis_traj, *_ in scenario_list:
            ped_traj = np.concatenate((ped_obs_traj.copy(), ped_pred_traj.copy()), axis=0)
            neis_traj = neis_traj.copy().transpose(1, 0, 2)  # [N,T,8]

            # Original coordinate-system metadata for test-time restoration.
            origin = ped_traj[self.obs_len - 1, 0:2].copy()
            rot_mat = np.eye(2, dtype=np.float32)
            scale = np.float32(1.0)

            # Neighbor filtering uses only observation-position distance.
            if neis_traj.shape[0] > 0:
                distance = np.linalg.norm(
                    ped_traj[None, :, 0:2] - neis_traj[:, :, 0:2], axis=-1
                )
                mean_distance = np.mean(distance[:, :self.obs_len], axis=-1)
                neis_traj = neis_traj[mean_distance < self.dist_threshold]
                if self.max_neis_num is not None:
                    neis_traj = neis_traj[: self.max_neis_num]

            n = neis_traj.shape[0]

            if self.translation:
                ped_traj[:, 0:2] -= origin[None, :]
                if n:
                    neis_traj[:, :, 0:2] -= origin[None, None, :]

            if self.rotation:
                ref_vec = ped_traj[self.obs_len - 1, 0:2] - ped_traj[0, 0:2]
                angle = np.arctan2(ref_vec[1], ref_vec[0])
                c, s = np.cos(angle), np.sin(angle)
                # Row-vector convention. This maps the observed heading to +x.
                rot_mat = np.array([[c, -s], [s, c]], dtype=np.float32)
                ped_traj = self._apply_rotation(ped_traj, rot_mat)
                if n:
                    neis_traj = self._apply_rotation(neis_traj, rot_mat)

            if self.scaling:
                scale = np.float32(np.random.randn() * 0.05 + 1.0)
                ped_traj = self._apply_scaling(ped_traj, scale)
                if n:
                    neis_traj = self._apply_scaling(neis_traj, scale)

            # speed and acc are scalars. Never rotate them: recompute from transformed vectors.
            ped_traj = recompute_scalar_features_np(ped_traj)
            if n:
                neis_traj = recompute_scalar_features_np(neis_traj)

            ped_list.append(ped_traj.astype(np.float32))
            neis_list.append(neis_traj.astype(np.float32))
            neighbor_counts.append(n)
            origins.append(origin)
            rot_mats.append(rot_mat)
            scales.append(scale)

        max_neighbors = max(1, max(neighbor_counts))
        padded_neis, neis_masks = [], []
        total_steps = ped_list[0].shape[0]

        for neighbor, n in zip(neis_list, neighbor_counts):
            padded = np.zeros((max_neighbors, total_steps, FULL_STATE_DIM), dtype=np.float32)
            if n:
                padded[:n] = neighbor
            padded_neis.append(padded)

            # Decoder takes [B,K,N], built from first row in model.py. Keep [N,N] interface.
            mask = np.zeros((max_neighbors, max_neighbors), dtype=np.float32)
            mask[:n, :n] = 1.0
            neis_masks.append(mask)

        ped = torch.tensor(np.stack(ped_list), dtype=torch.float32)
        neis = torch.tensor(np.stack(padded_neis), dtype=torch.float32)
        neis_mask = torch.tensor(np.stack(neis_masks), dtype=torch.float32)
        tail = (torch.tensor(np.stack(extras), dtype=torch.float32),) if extras is not None else ()

        if not self.return_transform:
            return (ped, neis, neis_mask) + tail

        return (
            ped,
            neis,
            neis_mask,
            torch.tensor(np.stack(origins), dtype=torch.float32),
            torch.tensor(np.stack(rot_mats), dtype=torch.float32),
            torch.tensor(np.asarray(scales), dtype=torch.float32),
        ) + tail

    def __len__(self):
        return len(self.scenario_list)

    def __getitem__(self, item):
        return self.scenario_list[item]
