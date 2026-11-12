"""Ours post-train 한 라운드 = GALTraj 식 라운드 학습 + CP contrastive 손실.

GALTraj 라운드와 다른 점:
  - 모델이 bottleneck 특징을 돌려주고 (model_contrative.py), 손실에 λ · contrastive 를 더한다
    (../contrastive/train_tutr_contrastive_cp.py 와 같은 식·값: config 의 λ, τ, θp, θn).
  - 샘플별 CP: init 은 --cp_table 의 --cp_col, 생성분은 <gen pkl>_cp.csv (ours_dataset.py, gen_cp.py).

라운드 학습:

기존 분할(init)과 지금까지의 생성분(gen_r1.pkl ...)을 합쳐, 시작 체크포인트에서 --epochs 만큼 학습한다.
  - 샘플 가중치: a 라운드에 들어온 데이터는 r 라운드에서 max(alpha^(r-a), clip). 기존 분할 a = 0,
    이번 라운드 생성분 1.0. alpha 0.7, clip 0.5 (GALTraj supplementary Sec 4).
  - epoch 당 샘플 수 고정 (--num_samples, 기본 = 기존 분할 크기). WeightedRandomSampler 복원추출.
  - 회귀 손실은 생성분의 pad 미래 프레임(fut_valid = 0)을 뺀다 (Eq.6).
  - 매 epoch val 로 채점해 이 라운드 안의 best 를 best.pth 로 저장 (ADE+FDE, K=1; train.py 와 같은 기준).
  - 모델 가중치만 이어받고 optimizer 는 새로 만든다 (lr 고정 1e-4, train.py 와 같음).
모션 모드는 시작 실험의 run_motion_modes.pkl 을 그대로 쓴다 (분류 헤드가 그 모드에 묶여 있음).

출력 (--out_dir): best.pth, last.pth, run_motion_modes.pkl (복사), log.txt
"""
import argparse
import os
import pickle
import random
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler

import train as base
from dataset import TrajectoryDataset
from feature_utils import (
    INPUT_MODE_TO_INDICES,
    OUTPUT_MODE_TO_INDICES,
    output_dim,
    parse_loss_weights,
    select_output_features,
)
from model_contrative import TrajectoryModel
from ours_dataset import OursDataset
from utils_contrastive import difficulty_contrastive_loss


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--hp_config", required=True)
    ap.add_argument("--base_splits", default="init")
    ap.add_argument("--val_split", default="val")
    ap.add_argument("--gen", default="", help="라운드:pkl 경로 를 쉼표로 (예: 1:r1/gen.pkl,2:r2/gen.pkl)")
    ap.add_argument("--round", type=int, required=True, help="이번 라운드 번호 (1부터)")
    ap.add_argument("--cp_table", required=True, help="scenario_id + cp 열 CSV (data/scenarios/manifest_<site>.csv)")
    ap.add_argument("--cp_col", default="vanilla_cp_max")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--num_samples", type=int, default=-1, help="epoch 당 샘플 수 (-1 = 기존 분할 크기)")
    ap.add_argument("--gen_frac", type=float, default=-1,
                    help=">0 이면 기존 분할 가중치를 1.0 으로 두고, 생성분 전체가 epoch 추출의 이 비율이 되도록 생성분 가중치를 줄인다 "
                         "(생성분끼리의 비는 max(alpha^(r-a), clip) 그대로). -1 = GALTraj 규칙 그대로")
    ap.add_argument("--init_ckpt", required=True, help="시작 가중치 (state_dict 또는 train.py 체크포인트)")
    ap.add_argument("--motion_modes", required=True, help="시작 실험의 run_motion_modes.pkl")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--epoch_offset", type=int, default=0, help="로그에 찍을 전체 epoch 시작값 (예: 200)")
    ap.add_argument("--total_epochs", type=int, default=300, help="로그 표시용")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--input_mode", choices=list(INPUT_MODE_TO_INDICES), default="xy_vxvy_axay")
    ap.add_argument("--output_mode", choices=list(OUTPUT_MODE_TO_INDICES), default="xy_vxvy")
    ap.add_argument("--loss_weights", default="1,1,1,1")
    ap.add_argument("--huber_beta", type=float, default=1.0)
    ap.add_argument("--cls_weight", type=float, default=1.0)
    ap.add_argument("--obs_len", type=int, default=10)
    ap.add_argument("--pred_len", type=int, default=30)
    ap.add_argument("--num_k", type=int, default=1)
    ap.add_argument("--num_works", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gpu", default="0")
    return ap.parse_args()


def masked_huber_loss(pred, target, valid, weights, beta):
    """feature_utils.weighted_huber_loss 와 같고, valid = 0 인 미래 프레임만 뺀다.
    pred/target [B, T, D], valid [B, T]. 모두 1 이면 weighted_huber_loss 와 같은 값."""
    el = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)        # [B, T, D]
    v = valid.unsqueeze(-1)
    per_channel = (el * v).sum(dim=(0, 1)) / v.sum().clamp(min=1.0)
    w = torch.as_tensor(weights, dtype=pred.dtype, device=pred.device)
    return (per_channel * w).sum() / w.sum()


def load_weights(path, device):
    ck = base.safe_torch_load(path, map_location=device)
    return ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    hp = base.load_config(args.hp_config)
    loss_weights = parse_loss_weights(args.loss_weights, args.output_mode)
    os.makedirs(args.out_dir, exist_ok=True)

    gen = []
    for item in [g for g in args.gen.split(",") if g]:
        r, path = item.split(":", 1)
        gen.append((int(r), path))
    base_splits = [s for s in args.base_splits.split(",") if s]

    train_ds = OursDataset(args.dataset_path, args.dataset_name, base_splits, gen,
                           cp_table=pd.read_csv(args.cp_table, usecols=["scenario_id", args.cp_col]),
                           cp_col=args.cp_col,
                           translation=True, rotation=True, scaling=True, obs_len=args.obs_len,
                           dist_threshold=hp.dist_threshold)
    val_ds = TrajectoryDataset(args.dataset_path, args.dataset_name, args.val_split,
                               translation=True, rotation=True, scaling=False, obs_len=args.obs_len,
                               dist_threshold=hp.dist_threshold)
    weights = train_ds.sample_weights(args.round, args.alpha, args.clip)
    if args.gen_frac > 0:
        is_gen = train_ds.source > 0
        weights = np.asarray(weights, dtype=np.float64).copy()
        weights[~is_gen] = 1.0
        if is_gen.any():
            g = weights[is_gen].sum()
            target = args.gen_frac / (1.0 - args.gen_frac) * (~is_gen).sum()   # 생성분 가중치 합 목표
            if g > target:   # 이미 비율보다 작으면 그대로 둔다
                weights[is_gen] *= target / g
    n_base = int((train_ds.source == 0).sum())
    num_samples = args.num_samples if args.num_samples > 0 else n_base
    val_loader = DataLoader(val_ds, collate_fn=val_ds.coll_fn, batch_size=hp.batch_size,
                            shuffle=False, num_workers=args.num_works)

    shutil.copyfile(args.motion_modes, os.path.join(args.out_dir, "run_motion_modes.pkl"))
    with open(args.motion_modes, "rb") as f:
        motion_modes = torch.tensor(pickle.load(f), dtype=torch.float32, device=device)

    model = TrajectoryModel(args.input_mode, args.output_mode, args.obs_len, args.pred_len,
                            hp.model_hidden_dim, 2, [1, 1], 4, 2).to(device)
    model.load_state_dict(load_weights(args.init_ckpt, device))
    optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=hp.lr)
    cls_criterion = torch.nn.CrossEntropyLoss()
    out_dim = output_dim(args.output_mode)

    with open(os.path.join(args.out_dir, "log.txt"), "a", encoding="utf-8") as log_file:
        log = lambda m: base.log_print(m, log_file)  # noqa: E731
        log(f"\n===== OURS ROUND {args.round} STARTED: {datetime.now():%Y-%m-%d %H:%M:%S} =====")
        log(f"{vars(args)}")
        parts = [f"{'base' if r == 0 else f'gen_r{r}'}={int((train_ds.source == r).sum())}"
                 f"(w={weights[train_ds.source == r][0]:.3f})" for r in np.unique(train_ds.source)]
        log(f"train samples: {' + '.join(parts)} | gen 추출 비율 {weights[train_ds.source > 0].sum() / weights.sum():.3f} | num_samples/epoch={num_samples} | "
            f"val={len(val_ds)} ({args.val_split}) | batch={hp.batch_size} | init={args.init_ckpt}")

        # 시작 가중치의 val 점수 (라운드 best 비교 기준은 아니고 기록용)
        ade, fde, *_ = base.evaluate(model, val_loader, motion_modes, device, args.obs_len, args.pred_len,
                                     args.output_mode, args.num_k, args.epoch_offset - 1, args.total_epochs)
        log(f"[start] K={args.num_k} ADE={ade:.4f} FDE={fde:.4f} ADE+FDE={ade + fde:.4f}")

        w = torch.as_tensor(weights, dtype=torch.double)
        best, best_epoch = float("inf"), -1
        for e in range(args.epochs):
            ep = args.epoch_offset + e
            sampler = WeightedRandomSampler(w, num_samples=num_samples, replacement=True)
            loader = DataLoader(train_ds, collate_fn=train_ds.coll_fn, batch_size=hp.batch_size,
                                sampler=sampler, num_workers=args.num_works,
                                pin_memory=(device.type == "cuda"))
            model.train()
            losses, regs, clss, contrs, n_gen = [], [], [], [], 0
            pbar = base.make_progress_bar(loader, f"E{ep + 1:03d}/{args.total_epochs} TRN", len(loader))
            try:
                for ped, neis, mask, valid, cp in pbar:
                    ped, neis, mask, valid, cp = (t.to(device, non_blocking=True)
                                                  for t in (ped, neis, mask, valid, cp))
                    gt_full = ped[:, args.obs_len:]
                    gt_out = select_output_features(gt_full, args.output_mode)
                    with torch.no_grad():
                        soft_label, closest = base.get_cls_label(gt_full[..., :2], motion_modes)
                    optimizer.zero_grad(set_to_none=True)
                    pred_flat, scores, bottleneck = model(ped[:, :args.obs_len], neis[:, :, :args.obs_len],
                                                          motion_modes, mask, closest, test=False, return_feat=True)
                    pred = pred_flat.reshape(pred_flat.shape[0], args.pred_len, out_dim)
                    reg = masked_huber_loss(pred, gt_out, valid, loss_weights, args.huber_beta)
                    cls = cls_criterion(scores, soft_label)
                    loss = reg + args.cls_weight * cls
                    if hp.lambda_contr > 0.0:
                        contr = difficulty_contrastive_loss(bottleneck, cp, tau=hp.contr_tau,
                                                            pos_delta=hp.contr_pos_delta,
                                                            neg_delta=hp.contr_neg_delta)
                        loss = loss + hp.lambda_contr * contr
                        contrs.append(contr.item())
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                    regs.append(reg.item())
                    clss.append(cls.item())
                    n_gen += int((valid.min(dim=1).values < 1).sum())
                    pbar.set_postfix_str(f"L={loss.item():.3f} avg={np.mean(losses):.3f}")
            finally:
                pbar.close()

            ade, fde, rmse_x, rmse_y, *_ = base.evaluate(model, val_loader, motion_modes, device, args.obs_len,
                                                         args.pred_len, args.output_mode, args.num_k, ep,
                                                         args.total_epochs)
            msg = []
            if ade + fde < best:
                best, best_epoch = ade + fde, ep
                torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pth"))
                msg.append("best.pth updated")
            torch.save(model.state_dict(), os.path.join(args.out_dir, "last.pth"))
            log(f"[Epoch {ep + 1:03d}/{args.total_epochs}] R{args.round} | Loss={np.mean(losses):.4f} "
                f"(reg={np.mean(regs):.4f}, cls={np.mean(clss):.4f}, contr={np.mean(contrs) if contrs else 0.0:.4f}) "
                f"masked_samples={n_gen} | "
                f"K={args.num_k} ADE={ade:.4f} FDE={fde:.4f} RMSE_X={rmse_x:.4f} RMSE_Y={rmse_y:.4f}"
                + (" | " + ", ".join(msg) if msg else ""))

        log(f"[Round {args.round} best {best_epoch + 1:03d}/{args.total_epochs}] ADE+FDE={best:.4f}")
        log(f"===== OURS ROUND {args.round} FINISHED: {datetime.now():%Y-%m-%d %H:%M:%S} =====")


if __name__ == "__main__":
    main()
