"""
lis_pca_gen.py — LIS 학습용 PCA 생성기 (pca_para_gen.py의 LIS 버전).

model.py의 `_reconstruct_traj`는 target과 무관하게 미래 궤적에 PCA 사영-역사영
스무딩을 적용한다(self.pca.forward → inverse). 따라서 module_vel 타깃이라도
PCA(pkl)가 필요하다. PCA는 미래 궤적을 (num_future_steps * 2) 차원으로 펴서 fit한다.
원본 pca_para_gen.py는 160차원(WOMD 80*2) 하드코딩이라 LIS(30*2=60)엔 못 쓴다.

사용:
    python motion_diff/scripts/lis_pca_gen.py \
        --data-dir <h5_out> --num-future-steps 30 --num-historical-steps 10 \
        --num-components 10 --out <h5_out>/lis_pca.pkl
"""

import argparse
import os
import pickle
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))  # motion_diff/

import torch
from dataset.lis_utils import LISMotionDataset
from sklearn.decomposition import PCA
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser("lis-pca-gen")
    p.add_argument("--data-dir", required=True, help="train.h5 위치")
    p.add_argument("--num-future-steps", type=int, default=30)
    p.add_argument("--num-historical-steps", type=int, default=10)
    p.add_argument("--num-components", type=int, default=10)
    p.add_argument("--out", required=True, help="pca.pkl 저장 경로")
    args = p.parse_args()

    dim = args.num_future_steps * 2  # LIS: 60
    ds = LISMotionDataset(
        root=args.data_dir,
        split="train",
        num_historical_steps=args.num_historical_steps,
        num_future_steps=args.num_future_steps,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    fit_data = []
    for data in tqdm(loader, desc="collect traj"):
        target = data["agent"]["target"]  # (A, num_future, 4)
        mask = (
            data["agent"]["valid"][:, args.num_historical_steps :]
            .squeeze(-1)
            .all(dim=-1)
        )
        traj = target[mask][..., :2].reshape(-1, dim)  # (A', 60)
        fit_data.append(traj)
    fit_data = torch.cat(fit_data, dim=0).numpy()
    print(f"[pca] fit data: {fit_data.shape} (dim={dim})")

    pca = PCA(n_components=args.num_components, whiten=True)
    pca.fit(fit_data)
    print(f"[pca] explained_variance_ratio_: {pca.explained_variance_ratio_}")
    print(f"[pca] cumulative: {pca.explained_variance_ratio_.cumsum()[-1]:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(pca, f)
    print(f"[pca] saved -> {args.out}")


if __name__ == "__main__":
    main()
