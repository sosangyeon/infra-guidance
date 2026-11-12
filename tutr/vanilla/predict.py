"""학습된 TUTR 로 분할 하나를 예측해 CSV 두 개로 저장한다.

  <out>_pred.csv     scenario_id, track_id, timestep(10~39), pred_x, pred_y
                     top-1(점수 최고) 후보, 원래 좌표계. compute_cp.py --source pred 입력.
  <out>_metrics.csv  meta 열 + ade1, fde1 (top-1), minade, minfde (top-K)
                     샘플별 오차. tail 마이닝(오차 상위 1%)에 쓴다.

샘플 순서는 <name>_<split>_meta.csv 와 같다 (shuffle 없음).

사용 예
  python predict.py --dataset_path ../../data/tutr/ss/ --dataset_name ss --split init,remain \
      --hp_config config/ig_tutr_hotel_ep200.py --run_dir ./checkpoint/<실험 폴더> \
      --input_mode xy_vxvy_axay --output_mode xy_vxvy --out ./pred/ss_vanilla_train
"""
import argparse
import importlib.util
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TrajectoryDataset
from feature_utils import (
    INPUT_MODE_TO_INDICES,
    OUTPUT_MODE_TO_INDICES,
    output_dim,
    restore_output_torch,
    select_output_features,
)
from model import TrajectoryModel


def load_config(path):
    if str(path).endswith(".json"):   # configs/*.json: model / train / contrastive 절을 속성으로 편다
        import json
        from types import SimpleNamespace
        with open(path, encoding="utf-8") as f:
            c = json.load(f)
        return SimpleNamespace(**{k: v for s in ("model", "train", "contrastive") for k, v in c.get(s, {}).items()})
    spec = importlib.util.spec_from_file_location("hp_config", path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--split", required=True, help="쉼표로 구분 (예: init,remain)")
    ap.add_argument("--hp_config", required=True)
    ap.add_argument("--run_dir", required=True, help="train.py 실험 폴더")
    ap.add_argument("--ckpt", default="best.pth")
    ap.add_argument("--input_mode", choices=list(INPUT_MODE_TO_INDICES), required=True)
    ap.add_argument("--output_mode", choices=list(OUTPUT_MODE_TO_INDICES), default="xy")
    ap.add_argument("--obs_len", type=int, default=10)
    ap.add_argument("--pred_len", type=int, default=30)
    ap.add_argument("--num_k", type=int, default=20)
    ap.add_argument("--num_works", type=int, default=0)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", required=True, help="출력 경로 앞부분")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = load_config(args.hp_config)
    splits = [s for s in args.split.split(",") if s]

    ds = TrajectoryDataset(args.dataset_path, args.dataset_name, splits, translation=True, rotation=True,
                           scaling=False, obs_len=args.obs_len, dist_threshold=hp.dist_threshold,
                           return_transform=True)
    if ds.meta is None:
        raise ValueError("meta CSV 가 필요함 (make_tutr_pkl.py 출력)")
    loader = DataLoader(ds, collate_fn=ds.coll_fn, batch_size=hp.batch_size, shuffle=False,
                        num_workers=args.num_works)

    with open(os.path.join(args.run_dir, "run_motion_modes.pkl"), "rb") as f:
        modes = torch.tensor(pickle.load(f), dtype=torch.float32, device=device)
    model = TrajectoryModel(args.input_mode, args.output_mode, args.obs_len, args.pred_len,
                            hp.model_hidden_dim, 2, [1, 1], 4, 2).to(device)
    model.load_state_dict(torch.load(os.path.join(args.run_dir, args.ckpt), map_location=device))
    model.eval()
    out_dim = output_dim(args.output_mode)

    top1, ade1, fde1, minade, minfde = [], [], [], [], []
    for batch in tqdm(loader, desc="Predict"):
        ped, neis, mask, origin, rot_mat, scale = [v.to(device) for v in batch]
        pred_flat, _ = model(ped[:, :args.obs_len], neis[:, :, :args.obs_len], modes, mask,
                             test=True, num_k=args.num_k)
        pred = pred_flat.reshape(pred_flat.shape[0], pred_flat.shape[1], args.pred_len, out_dim)
        pred_xy = restore_output_torch(pred, args.output_mode, origin, rot_mat, scale)[..., :2]
        gt_xy = restore_output_torch(select_output_features(ped[:, args.obs_len:], args.output_mode),
                                     args.output_mode, origin, rot_mat, scale)[..., :2]
        err = torch.norm(pred_xy - gt_xy.unsqueeze(1), dim=-1)          # [B, K, T]
        top1.append(pred_xy[:, 0].cpu().numpy())                         # topk 는 점수 내림차순
        ade1.append(err[:, 0].mean(-1).cpu().numpy())
        fde1.append(err[:, 0, -1].cpu().numpy())
        minade.append(err.mean(-1).min(-1).values.cpu().numpy())
        minfde.append(err[:, :, -1].min(-1).values.cpu().numpy())

    top1 = np.concatenate(top1)                                          # [N, T, 2]
    meta = ds.meta.copy()
    n, t = top1.shape[:2]
    pred_df = pd.DataFrame({
        "scenario_id": np.repeat(meta["scenario_id"].to_numpy(), t),
        "track_id": np.repeat(meta["track_id"].to_numpy(), t),
        "timestep": np.tile(np.arange(args.obs_len, args.obs_len + t), n),
        "pred_x": top1[..., 0].reshape(-1),
        "pred_y": top1[..., 1].reshape(-1),
    })
    meta["ade1"], meta["fde1"] = np.concatenate(ade1), np.concatenate(fde1)
    meta["minade"], meta["minfde"] = np.concatenate(minade), np.concatenate(minfde)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pred_df.to_csv(args.out + "_pred.csv", index=False)
    meta.to_csv(args.out + "_metrics.csv", index=False)
    print(f"{len(meta):,} 샘플 | ADE1 {meta['ade1'].mean():.4f} FDE1 {meta['fde1'].mean():.4f} | "
          f"minADE{args.num_k} {meta['minade'].mean():.4f} minFDE{args.num_k} {meta['minfde'].mean():.4f}")
    print(f"-> {args.out}_pred.csv, {args.out}_metrics.csv")


if __name__ == "__main__":
    main()
