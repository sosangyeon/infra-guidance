import argparse
import importlib.util
import os

# 반드시 numpy / sklearn import 이전에 설정
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import pickle
import random
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset_cp import TrajectoryDataset
from utils.feature_utils import (
    INPUT_MODE_TO_INDICES,
    OUTPUT_MODE_TO_INDICES,
    output_dim,
    output_names,
    parse_loss_weights,
    select_output_features,
    weighted_huber_loss,
)
from utils.model_contrative import TrajectoryModel
from utils.utils_contrastive import difficulty_contrastive_loss
from utils.utils import get_motion_modes


# =========================================================
# Argument Parser
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="./dataset_small_stride_10/")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--hp_config", type=str, default="config/ig.py")

    # 학습/검증 분할. 기본값은 기존 동작(<name>_train.pkl 로 학습, <name>_test.pkl 로 검증).
    # LIS 데이터는 --train_splits init --val_split val 처럼 쓴다 (test 로 모델을 고르지 않게).
    parser.add_argument("--train_splits", type=str, default="train",
                        help="쉼표로 구분한 학습 분할 (예: init,remain)")
    parser.add_argument("--val_split", type=str, default="test",
                        help="학습 중 best 선택에 쓸 검증 분할 (예: val)")
    parser.add_argument("--train_sid_file", type=str, default=None,
                        help="학습 분할에서 이 파일의 scenario_id 만 남김 (한 줄에 하나)")

    # contrastive 난이도: 시나리오 표(cp_table)의 cp_col 값을 그 시나리오의 모든 학습 샘플에 붙인다.
    # 연구실 contrastive(CP) 는 Vanilla 예측 궤적으로 계산한 CP_max (data_prep/compute_cp.py --source pred).
    parser.add_argument("--cp_table", type=str, required=True,
                        help="scenario_id + cp 열이 있는 CSV (예: data/scenarios/manifest_ss.csv)")
    parser.add_argument("--cp_col", type=str, default="vanilla_cp_max")

    parser.add_argument("--lr_scaling", action="store_true", default=False)
    parser.add_argument("--num_works", type=int, default=4)

    parser.add_argument("--obs_len", type=int, default=10)
    parser.add_argument("--pred_len", type=int, default=30)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=str, default="0")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoint_ablation_small_stride_10/",
    )

    parser.add_argument(
        "--input_mode",
        choices=list(INPUT_MODE_TO_INDICES),
        default="xy_vxvy_axay",
    )

    parser.add_argument(
        "--output_mode",
        choices=list(OUTPUT_MODE_TO_INDICES),
        default="xy",
    )

    parser.add_argument(
        "--loss_weights",
        type=str,
        default=None,
        help=(
            "Comma-separated per-output Huber weights. "
            "E.g. xy: 1,1 | xy_speed: 1,1,0.25 | "
            "xy_vxvy: 1,1,0.5,0.5"
        ),
    )

    parser.add_argument(
        "--huber_beta",
        type=float,
        default=1.0,
        help="SmoothL1/Huber transition beta. Default: 1.0",
    )

    parser.add_argument(
        "--cls_weight",
        type=float,
        default=1.0,
        help="Weight multiplying CrossEntropy motion-mode classification loss.",
    )

    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional suffix to avoid overwriting runs with different loss weights.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size. Overrides the config, including its "
             "MIN_BATCHES_PER_EPOCH cap. Leave unset to let the config decide.",
    )

    parser.add_argument(
        "--num_k",
        type=int,
        default=1,
        help="Number of trajectory candidates used during validation.",
    )

    # =====================================================
    # Resume options
    # =====================================================
    parser.add_argument(
        "--resume",
        type=str,
        default="none",
        help=(
            "Resume training option. "
            "'none': start from scratch, "
            "'auto': load last.pth from current run_dir if exists, "
            "or give a checkpoint path directly."
        ),
    )

    return parser.parse_args()


# =========================================================
# Utility Functions
# =========================================================
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


def get_cls_label(gt_xy, motion_modes):
    """
    Motion mode label은 output_mode와 무관하게
    항상 future XY trajectory 기준으로 계산.
    """
    gt_flat = gt_xy.reshape(gt_xy.shape[0], -1).unsqueeze(1)

    mode_flat = motion_modes.reshape(
        motion_modes.shape[0],
        -1,
    ).unsqueeze(0)

    distance = torch.norm(gt_flat - mode_flat, dim=-1)

    soft_label = F.softmax(-distance, dim=-1)
    closest_mode_indices = torch.argmin(distance, dim=-1)

    return soft_label, closest_mode_indices


def log_print(message, log_file=None):
    """
    tqdm bar 출력 중에도 깨지지 않도록 메시지를 출력.
    Epoch 완료 결과는 log.txt에도 저장.
    """
    tqdm.write(message)

    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


def make_epoch_bar(start_epoch, total_epochs):
    """
    전체 학습 진행률과 전체 ETA를 보여주는 outer progress bar.
    resume 시 start_epoch부터 이어서 표시.
    """
    return tqdm(
        range(start_epoch, total_epochs),
        desc="TOTAL",
        total=total_epochs,
        initial=start_epoch,
        leave=True,
        position=0,
        dynamic_ncols=True,
        mininterval=1.0,
        maxinterval=3.0,
        smoothing=0.1,
        unit="epoch",
        bar_format=(
            "{desc} |{bar}| "
            "{n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        ),
    )


def make_progress_bar(iterable, desc, total):
    """
    train / validation 내부 batch 진행률을 보여주는 progress bar.

    position=1:
    TOTAL bar 아래 줄에서 표시되게 함.
    leave=False:
    epoch 내부 bar는 완료 후 남기지 않음.
    dynamic_ncols=True:
    터미널 현재 폭에 맞게 bar 길이를 자동 변경.
    """
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=False,
        position=1,
        dynamic_ncols=True,
        mininterval=1.0,
        maxinterval=3.0,
        smoothing=0.1,
        unit="batch",
        bar_format=(
            "{desc} |{bar}| "
            "{n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}{postfix}]"
        ),
    )


def safe_torch_load(path, map_location):
    """
    PyTorch 버전에 따라 torch.load의 weights_only 기본값 이슈가 있을 수 있어서
    우선 weights_only=False로 시도하고, 안 되면 일반 torch.load로 fallback.
    """
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def save_training_checkpoint(
    save_path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_ade_fde_score,
    best_ade_fde_epoch,
    best_rmse_score,
    best_rmse_epoch,
    args,
):
    """
    epoch:
        현재 완료된 epoch index.
        예: epoch=4이면 5번째 epoch까지 끝난 상태.
    next epoch부터 재시작하기 위해 start_epoch=epoch+1로 저장.
    """
    ckpt = {
        "epoch": epoch,
        "start_epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_ade_fde_score": best_ade_fde_score,
        "best_ade_fde_epoch": best_ade_fde_epoch,
        "best_rmse_score": best_rmse_score,
        "best_rmse_epoch": best_rmse_epoch,
        "args": vars(args),
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    torch.save(ckpt, save_path)


def resolve_resume_path(args, run_dir):
    """
    args.resume:
        none -> resume 안 함
        auto -> run_dir/last.pth 있으면 resume
        그 외 문자열 -> 해당 경로에서 resume
    """
    if args.resume is None:
        return None

    resume = str(args.resume).strip()

    if resume.lower() == "none":
        return None

    if resume.lower() == "auto":
        auto_path = os.path.join(run_dir, "last.pth")
        if os.path.exists(auto_path):
            return auto_path
        return None

    return resume


def load_training_checkpoint(
    resume_path,
    model,
    optimizer,
    scheduler,
    device,
):
    """
    checkpoint에서 model / optimizer / scheduler / best score / start_epoch 복구.
    """
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    ckpt = safe_torch_load(
        resume_path,
        map_location=device,
    )

    # 새 포맷: dict checkpoint
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])

        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if (
            scheduler is not None
            and "scheduler_state_dict" in ckpt
            and ckpt["scheduler_state_dict"] is not None
        ):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        start_epoch = int(ckpt.get("start_epoch", ckpt.get("epoch", -1) + 1))

        best_ade_fde_score = float(ckpt.get("best_ade_fde_score", float("inf")))
        best_ade_fde_epoch = int(ckpt.get("best_ade_fde_epoch", -1))

        best_rmse_score = float(ckpt.get("best_rmse_score", float("inf")))
        best_rmse_epoch = int(ckpt.get("best_rmse_epoch", -1))

        return (
            start_epoch,
            best_ade_fde_score,
            best_ade_fde_epoch,
            best_rmse_score,
            best_rmse_epoch,
        )

    # 예전 포맷: model.state_dict()만 저장된 best.pth 같은 경우
    # 이 경우 optimizer/scheduler/epoch 정보가 없어서 0 epoch부터 fine-tuning처럼 시작.
    model.load_state_dict(ckpt)

    return (
        0,
        float("inf"),
        -1,
        float("inf"),
        -1,
    )


# =========================================================
# Validation / Inference
# =========================================================
@torch.no_grad()
def evaluate(
    model,
    dataloader,
    motion_modes,
    device,
    obs_len,
    pred_len,
    output_mode,
    num_k,
    epoch,
    total_epochs,
):
    model.eval()

    out_dim = output_dim(output_mode)

    ade_sum = 0.0
    fde_sum = 0.0
    rmse_x_sum = 0.0
    rmse_y_sum = 0.0
    num_points = 0

    num_traj = 0
    inference_times = []

    val_pbar = make_progress_bar(
        iterable=dataloader,
        desc=f"E{epoch + 1:03d}/{total_epochs} VAL",
        total=len(dataloader),
    )

    try:
        for ped, neis, mask in val_pbar:
            ped = ped.to(device, non_blocking=True)
            neis = neis.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            ped_obs = ped[:, :obs_len, :]
            gt_full = ped[:, obs_len:, :]
            gt_xy = gt_full[..., :2]

            neis_obs = neis[:, :, :obs_len, :]

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            pred_flat, _ = model(
                ped_obs,
                neis_obs,
                motion_modes,
                mask,
                test=True,
                num_k=num_k,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            inference_times.append(time.perf_counter() - start)

            # [B, K, T * output_dim] -> [B, K, T, output_dim]
            pred = pred_flat.reshape(
                pred_flat.shape[0],
                pred_flat.shape[1],
                pred_len,
                out_dim,
            )

            # 평가는 항상 XY 기준
            pred_xy = pred[..., :2]

            # [B, K, T]
            error = torch.norm(
                pred_xy - gt_xy.unsqueeze(1),
                dim=-1,
            )

            # [B, K]
            ade_all = error.mean(dim=-1)
            fde_all = error[:, :, -1]

            # [B]
            min_ade = ade_all.min(dim=-1).values
            min_fde, min_fde_idx = fde_all.min(dim=-1)

            batch_idx = torch.arange(
                pred.shape[0],
                device=device,
            )

            # minFDE candidate 기준 RMSE
            best_pred_xy = pred_xy[
                batch_idx,
                min_fde_idx,
            ]

            diff_xy = best_pred_xy - gt_xy

            # 사양서 RMSE: 오차제곱을 표본·프레임에 걸쳐 모두 합산해 두고
            # 마지막에 한 번만 sqrt 한다 (test.py 와 동일 규약).
            ade_sum += min_ade.sum().item()
            fde_sum += min_fde.sum().item()
            rmse_x_sum += (diff_xy[..., 0] ** 2).sum().item()
            rmse_y_sum += (diff_xy[..., 1] ** 2).sum().item()
            num_points += diff_xy.shape[0] * diff_xy.shape[1]

            num_traj += pred.shape[0]

            current_ade = ade_sum / num_traj
            current_fde = fde_sum / num_traj

            val_pbar.set_postfix_str(
                f"A={current_ade:.3f} F={current_fde:.3f}"
            )

    finally:
        val_pbar.close()

    total_time = sum(inference_times)

    ade = ade_sum / num_traj
    fde = fde_sum / num_traj
    rmse_x = (rmse_x_sum / num_points) ** 0.5
    rmse_y = (rmse_y_sum / num_points) ** 0.5

    ms_per_sample = 1000.0 * total_time / num_traj

    return (
        ade,
        fde,
        rmse_x,
        rmse_y,
        total_time,
        ms_per_sample,
    )


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()

    if args.huber_beta <= 0:
        raise ValueError("--huber_beta must be > 0")

    if args.cls_weight < 0:
        raise ValueError("--cls_weight must be >= 0")

    loss_weights = parse_loss_weights(
        args.loss_weights,
        args.output_mode,
    )

    # CUDA 초기화 이전에 GPU 지정
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # -------------------------
    # Reproducibility
    # -------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # -------------------------
    # Hyperparameter config
    # -------------------------
    hp_config = load_config(args.hp_config)

    # -------------------------
    # Run directory
    # -------------------------
    train_splits = [s for s in args.train_splits.split(",") if s]
    train_tag = "+".join(train_splits)
    if args.train_sid_file:
        train_tag += "-" + os.path.splitext(os.path.basename(args.train_sid_file))[0]

    run_name = (
        f"{args.dataset_name}"
        + ("" if train_splits == ["train"] and not args.train_sid_file
           else f"__tr-{train_tag}__va-{args.val_split}")
        + f"__in-{args.input_mode}"
        f"__out-{args.output_mode}"
        f"__obs-{args.obs_len}"
        f"__pred-{args.pred_len}"
        f"--weight-{args.loss_weights}"
    )

    if args.run_tag:
        run_name += f"__{args.run_tag}"

    run_dir = os.path.join(
        args.checkpoint,
        run_name,
    )

    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(
        run_dir,
        "log.txt",
    )

    last_ckpt_path = os.path.join(run_dir, "last.pth")
    interrupted_ckpt_path = os.path.join(run_dir, "interrupted.pth")
    error_ckpt_path = os.path.join(run_dir, "error_checkpoint.pth")

    # -------------------------
    # Initial information
    # -------------------------
    print(
        f"\n===== TRAINING STARTED: "
        f"{datetime.now():%Y-%m-%d %H:%M:%S} ====="
    )

    print(
        f"device={device} | "
        f"input={args.input_mode} | "
        f"output={args.output_mode}"
    )

    print(
        f"Huber(beta={args.huber_beta}) "
        f"weights={dict(zip(output_names(args.output_mode), loss_weights))} | "
        f"cls_weight={args.cls_weight}"
    )

    print(f"checkpoint_dir={run_dir}")
    print(f"resume={args.resume}")

    # -------------------------
    # Dataset
    # -------------------------
    train_dataset = TrajectoryDataset(
        args.dataset_path,
        args.dataset_name,
        train_splits,
        translation=True,
        rotation=True,
        scaling=True,
        obs_len=args.obs_len,
        dist_threshold=hp_config.dist_threshold,
        return_transform=False,
        sid_file=args.train_sid_file,
        cp_table=pd.read_csv(args.cp_table, usecols=["scenario_id", args.cp_col]),
        cp_col=args.cp_col,
    )

    test_dataset = TrajectoryDataset(
        args.dataset_path,
        args.dataset_name,
        args.val_split,
        translation=True,
        rotation=True,
        scaling=False,
        obs_len=args.obs_len,
        dist_threshold=hp_config.dist_threshold,
        return_transform=False,
    )

    # -------------------------
    # DataLoader
    # -------------------------
    # --batch_size wins outright. Otherwise take the config value, shrinking it on small
    # datasets so that a fixed epoch count still means a comparable amount of training.
    # See MIN_BATCHES_PER_EPOCH in the config for what went wrong without this.
    # Datasets already above the floor are untouched.
    if args.batch_size:
        train_batch_size = args.batch_size
        batch_note = " (from --batch_size)"
    else:
        train_batch_size = hp_config.batch_size
        min_batches = getattr(hp_config, "MIN_BATCHES_PER_EPOCH", None)
        if min_batches:
            floor = getattr(hp_config, "MIN_BATCH_SIZE", 64)
            train_batch_size = max(floor, min(train_batch_size,
                                              len(train_dataset) // min_batches))
        batch_note = ("" if train_batch_size == hp_config.batch_size else
                      f" (config {hp_config.batch_size}, capped for dataset size)")

    train_loader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.coll_fn,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=args.num_works,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        collate_fn=test_dataset.coll_fn,
        batch_size=hp_config.batch_size,
        shuffle=False,
        num_workers=args.num_works,
        pin_memory=(device.type == "cuda"),
    )

    print(f"train samples={len(train_dataset)} ({train_tag})")
    print(f"val samples={len(test_dataset)} ({args.val_split})")
    print(f"train batch_size={train_batch_size}{batch_note}")
    print(f"train batches/epoch={len(train_loader)} "
          f"-> {len(train_loader) * hp_config.epoch:,} updates")
    print(f"val batches/epoch={len(test_loader)}")
    print(f"total epochs={hp_config.epoch}")

    # -------------------------
    # Motion modes
    # -------------------------
    # 모션 모드는 학습 집합마다 다르므로 실험 폴더(run_dir)에 둔다. 예전 실행처럼
    # <dataset_path>/<name>_motion_modes.pkl 만 있으면 그것을 쓴다 (기존 동작).
    legacy_modes_file = os.path.join(
        args.dataset_path,
        f"{args.dataset_name}_motion_modes.pkl",
    )
    motion_modes_file = os.path.join(run_dir, "run_motion_modes.pkl")
    if train_splits == ["train"] and not args.train_sid_file and os.path.exists(legacy_modes_file):
        motion_modes_file = legacy_modes_file

    if not os.path.exists(motion_modes_file):
        print("motion modes generating ...")

        motion_modes = get_motion_modes(
            train_dataset,
            args.obs_len,
            args.pred_len,
            hp_config.n_clusters,
            run_dir,
            "run",
            smooth_size=hp_config.smooth_size,
            random_rotation=hp_config.random_rotation,
            traj_seg=hp_config.traj_seg,
        )

    else:
        print("motion modes loading ...")

        with open(motion_modes_file, "rb") as f:
            motion_modes = pickle.load(f)

    motion_modes = torch.tensor(
        motion_modes,
        dtype=torch.float32,
        device=device,
    )

    # -------------------------
    # Model
    # -------------------------
    model = TrajectoryModel(
        args.input_mode,
        args.output_mode,
        args.obs_len,
        args.pred_len,
        hp_config.model_hidden_dim,
        2,
        [1, 1],
        4,
        2,
    ).to(device)

    optimizer = optim.AdamW(
        (
            p for p in model.parameters()
            if p.requires_grad
        ),
        lr=hp_config.lr,
    )

    cls_criterion = torch.nn.CrossEntropyLoss()

    if args.lr_scaling:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[270, 400],
            gamma=0.5,
        )
    else:
        scheduler = None

    # -------------------------
    # Resume
    # -------------------------
    start_epoch = 0

    # ADE + FDE 기준 best
    best_ade_fde_score = float("inf")
    best_ade_fde_epoch = -1

    # RMSE 기준 best
    # 여기서는 sqrt(RMSE_X^2 + RMSE_Y^2)를 기준으로 저장
    best_rmse_score = float("inf")
    best_rmse_epoch = -1

    resume_path = resolve_resume_path(args, run_dir)

    if resume_path is not None:
        print(f"[Resume] loading checkpoint: {resume_path}")

        (
            start_epoch,
            best_ade_fde_score,
            best_ade_fde_epoch,
            best_rmse_score,
            best_rmse_epoch,
        ) = load_training_checkpoint(
            resume_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        print(
            f"[Resume] start_epoch={start_epoch} | "
            f"best_ade_fde={best_ade_fde_score:.4f} "
            f"(epoch={best_ade_fde_epoch + 1}) | "
            f"best_rmse={best_rmse_score:.4f} "
            f"(epoch={best_rmse_epoch + 1})"
        )

    else:
        if str(args.resume).lower() == "auto":
            print("[Resume] auto requested, but last.pth not found. Start from scratch.")
        else:
            print("[Resume] disabled. Start from scratch.")

    if start_epoch >= hp_config.epoch:
        print(
            f"[Done] start_epoch={start_epoch} >= total_epochs={hp_config.epoch}. "
            f"Nothing to train."
        )
        return

    out_dim = output_dim(args.output_mode)

    # =====================================================
    # Training
    # =====================================================
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(
            f"\n===== TRAINING STARTED: "
            f"{datetime.now():%Y-%m-%d %H:%M:%S} =====\n"
        )

        log_file.write(f"{args}\n")

        log_file.write(
            f"output_names={output_names(args.output_mode)}, "
            f"loss_weights={loss_weights}, "
            f"huber_beta={args.huber_beta}, "
            f"cls_weight={args.cls_weight}\n"
        )

        log_file.write(
            f"train_samples={len(train_dataset)} ({train_tag}), "
            f"val_samples={len(test_dataset)} ({args.val_split}), "
            f"motion_modes={motion_modes_file}, "
            f"train_batches={len(train_loader)}, "
            f"test_batches={len(test_loader)}\n"
        )

        if resume_path is not None:
            log_file.write(
                f"[Resume] loaded={resume_path}, "
                f"start_epoch={start_epoch}, "
                f"best_ade_fde_score={best_ade_fde_score}, "
                f"best_ade_fde_epoch={best_ade_fde_epoch}, "
                f"best_rmse_score={best_rmse_score}, "
                f"best_rmse_epoch={best_rmse_epoch}\n"
            )
        else:
            log_file.write("[Resume] start from scratch\n")

        log_file.flush()

        epoch_pbar = make_epoch_bar(
            start_epoch=start_epoch,
            total_epochs=hp_config.epoch,
        )

        current_epoch_for_exception = start_epoch - 1

        try:
            for epoch in epoch_pbar:
                current_epoch_for_exception = epoch

                model.train()

                losses = []
                reg_losses = []
                cls_losses = []
                contr_losses = []

                current_lr = optimizer.param_groups[0]["lr"]

                # ---------------------------------------------
                # Training progress bar
                # ---------------------------------------------
                train_pbar = make_progress_bar(
                    iterable=train_loader,
                    desc=f"E{epoch + 1:03d}/{hp_config.epoch} TRN",
                    total=len(train_loader),
                )

                try:
                    for ped, neis, mask, cp in train_pbar:
                        ped = ped.to(device, non_blocking=True)
                        neis = neis.to(device, non_blocking=True)
                        mask = mask.to(device, non_blocking=True)

                        ped_obs = ped[:, :args.obs_len, :]
                        gt_full = ped[:, args.obs_len:, :]

                        # motion mode label은 XY 기준
                        gt_xy = gt_full[..., :2]

                        # regression loss target
                        gt_out = select_output_features(
                            gt_full,
                            args.output_mode,
                        )

                        neis_obs = neis[:, :, :args.obs_len, :]

                        with torch.no_grad():
                            soft_label, closest_mode_indices = get_cls_label(
                                gt_xy,
                                motion_modes,
                            )

                        optimizer.zero_grad(set_to_none=True)

                        pred_flat, scores, bottleneck = model(
                            ped_obs,
                            neis_obs,
                            motion_modes,
                            mask,
                            closest_mode_indices,
                            test=False,
                            return_feat=True,
                        )

                        # [B, pred_len * out_dim] -> [B, pred_len, out_dim]
                        pred = pred_flat.reshape(
                            pred_flat.shape[0],
                            args.pred_len,
                            out_dim,
                        )

                        reg_loss = weighted_huber_loss(
                            pred,
                            gt_out,
                            loss_weights,
                            beta=args.huber_beta,
                        )

                        cls_loss = cls_criterion(
                            scores,
                            soft_label,
                        )

                        loss = reg_loss + args.cls_weight * cls_loss
                        if hp_config.lambda_contr > 0.0:
                            contr_loss = difficulty_contrastive_loss(
                                bottleneck, cp.to(device, non_blocking=True),
                                tau=hp_config.contr_tau,
                                pos_delta=hp_config.contr_pos_delta,
                                neg_delta=hp_config.contr_neg_delta,
                            )
                            loss = loss + hp_config.lambda_contr * contr_loss
                            contr_losses.append(contr_loss.item())

                        loss.backward()
                        optimizer.step()

                        losses.append(loss.item())
                        reg_losses.append(reg_loss.item())
                        cls_losses.append(cls_loss.item())

                        train_pbar.set_postfix_str(
                            f"L={loss.item():.3f} avg={np.mean(losses):.3f}"
                        )

                finally:
                    train_pbar.close()

                # ---------------------------------------------
                # Validation / inference progress bar
                # ---------------------------------------------
                (
                    ade,
                    fde,
                    rmse_x,
                    rmse_y,
                    total_time,
                    ms_per_sample,
                ) = evaluate(
                    model=model,
                    dataloader=test_loader,
                    motion_modes=motion_modes,
                    device=device,
                    obs_len=args.obs_len,
                    pred_len=args.pred_len,
                    output_mode=args.output_mode,
                    num_k=args.num_k,
                    epoch=epoch,
                    total_epochs=hp_config.epoch,
                )

                if scheduler is not None:
                    scheduler.step()

                # ---------------------------------------------
                # Best model save
                # ---------------------------------------------
                ade_fde_score = ade + fde

                # RMSE_X + RMSE_Y를 단순합으로 쓰고 싶으면 아래 줄로 바꾸면 됨:
                # rmse_score = rmse_x + rmse_y
                rmse_score = (rmse_x ** 2 + rmse_y ** 2) ** 0.5

                updated_msgs = []

                # 1) ADE + FDE 기준 best 저장
                if ade_fde_score < best_ade_fde_score:
                    best_ade_fde_score = ade_fde_score
                    best_ade_fde_epoch = epoch

                    torch.save(
                        model.state_dict(),
                        os.path.join(run_dir, "best.pth"),
                    )

                    updated_msgs.append("best.pth updated")

                # 2) RMSE 기준 best 저장
                if rmse_score < best_rmse_score:
                    best_rmse_score = rmse_score
                    best_rmse_epoch = epoch

                    torch.save(
                        model.state_dict(),
                        os.path.join(run_dir, "best_rmse.pth"),
                    )

                    updated_msgs.append("best_rmse.pth updated")

                # ---------------------------------------------
                # Last checkpoint save
                # ---------------------------------------------
                save_training_checkpoint(
                    save_path=last_ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_ade_fde_score=best_ade_fde_score,
                    best_ade_fde_epoch=best_ade_fde_epoch,
                    best_rmse_score=best_rmse_score,
                    best_rmse_epoch=best_rmse_epoch,
                    args=args,
                )

                updated_msgs.append("last.pth saved")

                # GALTraj: 전체 epoch 의 2/3 지점 체크포인트 (GAL post-train 시작점)
                # best_2of3.pth = 0 ~ 2/3 구간의 best (이후 best.pth 는 전체 구간 best 로 계속 갱신됨)
                if epoch + 1 == round(hp_config.epoch * 2 / 3):
                    shutil.copyfile(last_ckpt_path, os.path.join(run_dir, "snapshot_2of3.pth"))
                    shutil.copyfile(os.path.join(run_dir, "best.pth"), os.path.join(run_dir, "best_2of3.pth"))
                    updated_msgs.append("snapshot_2of3.pth, best_2of3.pth saved")

                updated = ""
                if updated_msgs:
                    updated = " | " + ", ".join(updated_msgs)

                # ---------------------------------------------
                # TOTAL progress bar postfix
                # ---------------------------------------------
                epoch_pbar.set_postfix_str(
                    f"ADE={ade:.3f} "
                    f"FDE={fde:.3f} "
                    f"Best={best_ade_fde_score:.3f} "
                    f"RMSEBest={best_rmse_score:.3f}"
                )

                # ---------------------------------------------
                # Epoch summary
                # ---------------------------------------------
                line = (
                    f"[Epoch {epoch + 1:03d}/{hp_config.epoch}] "
                    f"LR={current_lr:.6e} | "
                    f"in={args.input_mode} "
                    f"out={args.output_mode} | "
                    f"Loss={np.mean(losses):.4f} "
                    f"(reg={np.mean(reg_losses):.4f}, "
                    f"cls={np.mean(cls_losses):.4f}, "
                    f"contr={np.mean(contr_losses) if contr_losses else 0.0:.4f}) | "
                    f"K={args.num_k} "
                    f"ADE={ade:.4f} "
                    f"FDE={fde:.4f} "
                    f"RMSE_X={rmse_x:.4f} "
                    f"RMSE_Y={rmse_y:.4f} "
                    f"RMSE_SCORE={rmse_score:.4f} | "
                    f"Infer={total_time:.2f}s "
                    f"({ms_per_sample:.3f} ms/sample)"
                    f"{updated}"
                )

                log_print(line, log_file)

        except KeyboardInterrupt:
            save_training_checkpoint(
                save_path=interrupted_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=current_epoch_for_exception,
                best_ade_fde_score=best_ade_fde_score,
                best_ade_fde_epoch=best_ade_fde_epoch,
                best_rmse_score=best_rmse_score,
                best_rmse_epoch=best_rmse_epoch,
                args=args,
            )

            log_print(
                f"[Interrupted] checkpoint saved: {interrupted_ckpt_path}",
                log_file,
            )
            raise

        except Exception as e:
            save_training_checkpoint(
                save_path=error_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=current_epoch_for_exception,
                best_ade_fde_score=best_ade_fde_score,
                best_ade_fde_epoch=best_ade_fde_epoch,
                best_rmse_score=best_rmse_score,
                best_rmse_epoch=best_rmse_epoch,
                args=args,
            )

            log_print(
                f"[Error] checkpoint saved: {error_ckpt_path}",
                log_file,
            )
            log_print(
                f"[Error] {repr(e)}",
                log_file,
            )
            raise

        finally:
            epoch_pbar.close()

        summary_ade_fde = (
            f"[Best ADE+FDE {best_ade_fde_epoch + 1:03d}/{hp_config.epoch}] "
            f"ADE+FDE={best_ade_fde_score:.4f}"
        )

        summary_rmse = (
            f"[Best RMSE {best_rmse_epoch + 1:03d}/{hp_config.epoch}] "
            f"RMSE_SCORE={best_rmse_score:.4f}"
        )

        log_print(summary_ade_fde, log_file)
        log_print(summary_rmse, log_file)

        log_file.write(
            f"===== TRAINING FINISHED: "
            f"{datetime.now():%Y-%m-%d %H:%M:%S} =====\n"
        )

        log_file.flush()


if __name__ == "__main__":
    main()