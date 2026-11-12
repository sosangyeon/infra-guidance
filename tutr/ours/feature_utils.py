# feature_utils.py
"""Common feature definitions and loss/coordinate helpers.

All pkl trajectories use one fixed 8-D state:
[x, y, vx, vy, speed, ax, ay, acc]

Input/output mode selection happens only after loading this common state.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


FEATURE_NAMES = ("x", "y", "vx", "vy", "speed", "ax", "ay", "acc")
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
FULL_STATE_DIM = len(FEATURE_NAMES)


INPUT_MODE_TO_INDICES = {
    "xy": [
        FEATURE_INDEX["x"],
        FEATURE_INDEX["y"],
    ],
    "xy_speed": [
        FEATURE_INDEX["x"],
        FEATURE_INDEX["y"],
        FEATURE_INDEX["speed"],
    ],
    "xy_vxvy": [
        FEATURE_INDEX["x"],
        FEATURE_INDEX["y"],
        FEATURE_INDEX["vx"],
        FEATURE_INDEX["vy"],
    ],
    "xy_vxvy_acc": [
        FEATURE_INDEX["x"],
        FEATURE_INDEX["y"],
        FEATURE_INDEX["vx"],
        FEATURE_INDEX["vy"],
        FEATURE_INDEX["acc"],
    ],
    "xy_vxvy_axay": [
        FEATURE_INDEX["x"],
        FEATURE_INDEX["y"],
        FEATURE_INDEX["vx"],
        FEATURE_INDEX["vy"],
        FEATURE_INDEX["ax"],
        FEATURE_INDEX["ay"],
    ],
}


OUTPUT_MODE_TO_INDICES = {
    "xy": [
        FEATURE_INDEX["x"],
        FEATURE_INDEX["y"],
    ],
    "xy_speed": [
        FEATURE_INDEX["x"],
        FEATURE_INDEX["y"],
        FEATURE_INDEX["speed"],
    ],
    "xy_vxvy": [
        FEATURE_INDEX["x"],
        FEATURE_INDEX["y"],
        FEATURE_INDEX["vx"],
        FEATURE_INDEX["vy"],
    ],
}


OUTPUT_MODE_TO_NAMES = {
    "xy": ("x", "y"),
    "xy_speed": ("x", "y", "speed"),
    "xy_vxvy": ("x", "y", "vx", "vy"),
}


def validate_input_mode(input_mode: str) -> None:
    if input_mode not in INPUT_MODE_TO_INDICES:
        raise ValueError(
            f"Unknown input_mode={input_mode!r}. "
            f"Choices: {', '.join(INPUT_MODE_TO_INDICES)}"
        )


def validate_output_mode(output_mode: str) -> None:
    if output_mode not in OUTPUT_MODE_TO_INDICES:
        raise ValueError(
            f"Unknown output_mode={output_mode!r}. "
            f"Choices: {', '.join(OUTPUT_MODE_TO_INDICES)}"
        )


def input_dim(input_mode: str) -> int:
    validate_input_mode(input_mode)
    return len(INPUT_MODE_TO_INDICES[input_mode])


def output_dim(output_mode: str) -> int:
    validate_output_mode(output_mode)
    return len(OUTPUT_MODE_TO_INDICES[output_mode])


def output_names(output_mode: str) -> Sequence[str]:
    validate_output_mode(output_mode)
    return OUTPUT_MODE_TO_NAMES[output_mode]


def select_input_features(traj, input_mode: str):
    """
    Select model input channels from [..., 8] NumPy/Torch trajectory.
    """
    validate_input_mode(input_mode)
    indices = INPUT_MODE_TO_INDICES[input_mode]

    if isinstance(traj, torch.Tensor):
        index = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=traj.device
        )
        return torch.index_select(
            traj,
            dim=-1,
            index=index
        )

    return traj[..., indices]


def select_output_features(traj, output_mode: str):
    """
    Select regression target channels from [..., 8] future trajectory.
    """
    validate_output_mode(output_mode)
    indices = OUTPUT_MODE_TO_INDICES[output_mode]

    if isinstance(traj, torch.Tensor):
        index = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=traj.device
        )
        return torch.index_select(
            traj,
            dim=-1,
            index=index
        )

    return traj[..., indices]


def parse_loss_weights(raw: str | None, output_mode: str) -> List[float]:
    """
    Parse comma-separated per-variable Huber weights.

    Examples:
        --loss_weights 1,1
        --loss_weights 1,1,0.25
        --loss_weights 1,1,0.5,0.5

    None means equal weight 1.0 for every output channel.
    """
    names = output_names(output_mode)

    if raw is None or raw.strip() == "":
        return [1.0] * len(names)

    try:
        weights = [
            float(token.strip())
            for token in raw.split(",")
        ]
    except ValueError as exc:
        raise ValueError(
            "--loss_weights must be comma-separated numeric values, "
            "e.g. 1,1,0.25"
        ) from exc

    if len(weights) != len(names):
        raise ValueError(
            f"output_mode={output_mode} requires {len(names)} weights "
            f"for {list(names)}, but got {len(weights)}: {weights}"
        )

    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError(
            "--loss_weights must be non-negative and have a positive sum."
        )

    return weights


def weighted_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Sequence[float],
    beta: float = 1.0
) -> torch.Tensor:
    """
    Channel-weighted, normalized SmoothL1 / Huber loss.

    pred/target: [..., D]

    loss = sum_d w_d * mean(Huber_d) / sum_d w_d

    Normalization makes weights relative importances rather than changing the
    overall regression-loss scale simply because more output channels exist.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shapes differ: {pred.shape} vs {target.shape}"
        )

    if pred.shape[-1] != len(weights):
        raise ValueError(
            f"Expected {pred.shape[-1]} weights, got {len(weights)}"
        )

    if beta <= 0:
        raise ValueError("Huber beta must be > 0")

    elementwise = F.smooth_l1_loss(
        pred,
        target,
        reduction="none",
        beta=beta
    )

    per_channel = elementwise.mean(
        dim=tuple(range(elementwise.ndim - 1))
    )

    weight_tensor = torch.as_tensor(
        weights,
        dtype=pred.dtype,
        device=pred.device
    )

    return (per_channel * weight_tensor).sum() / weight_tensor.sum()


def recompute_scalar_features_np(traj: np.ndarray) -> np.ndarray:
    """
    Recompute speed and acceleration magnitude after geometric transforms.

    traj: [..., 8]
          [x, y, vx, vy, speed, ax, ay, acc]
    """
    if traj.shape[-1] != FULL_STATE_DIM:
        raise ValueError(
            f"Expected last dimension {FULL_STATE_DIM}, got {traj.shape[-1]}"
        )

    traj[..., FEATURE_INDEX["speed"]] = np.sqrt(
        traj[..., FEATURE_INDEX["vx"]] ** 2
        + traj[..., FEATURE_INDEX["vy"]] ** 2
    )

    traj[..., FEATURE_INDEX["acc"]] = np.sqrt(
        traj[..., FEATURE_INDEX["ax"]] ** 2
        + traj[..., FEATURE_INDEX["ay"]] ** 2
    )

    return traj


def _prepare_scale(scale, values):
    """
    scale을 tensor로 변환하고 [B] 형태로 정리.
    """
    if not torch.is_tensor(scale):
        scale = torch.tensor(
            scale,
            dtype=values.dtype,
            device=values.device
        )

    scale = scale.to(
        device=values.device,
        dtype=values.dtype
    )

    return scale.reshape(-1)


def _scale_view(scale, values):
    """
    values:
        [B, T, C] 또는 [B, K, T, C]

    반환:
        scale이 values와 broadcast 가능한 shape.
    """
    scale = _prepare_scale(scale, values)

    view_shape = [values.shape[0]] + [1] * (values.ndim - 1)

    return scale.view(*view_shape)


def _prepare_origin_rot(origin, rot_mat, values):
    origin = origin.to(
        device=values.device,
        dtype=values.dtype
    )

    rot_mat = rot_mat.to(
        device=values.device,
        dtype=values.dtype
    )

    return origin, rot_mat


def inverse_transform_xy_torch(
    local_xy: torch.Tensor,
    origin: torch.Tensor,
    rot_mat: torch.Tensor,
    scale: torch.Tensor
) -> torch.Tensor:
    """
    Local 좌표계의 x,y를 원래 raw 좌표계로 복원.

    local_xy:
        [B, T, 2] 또는 [B, K, T, 2]

    origin:
        [B, 2]
        local 좌표계의 원점이었던 raw 좌표.
        보통 마지막 관측 위치.

    rot_mat:
        [B, 2, 2]
        raw -> local 변환에 사용했던 회전행렬.

    scale:
        [B] 또는 [B, 1]
        dataset augmentation에서 local 값에 곱했던 scale.

    변환 관계:
        local = ((raw - origin) @ rot_mat) * scale

    역변환:
        raw = (local / scale) @ rot_mat.T + origin
    """
    origin, rot_mat = _prepare_origin_rot(
        origin,
        rot_mat,
        local_xy
    )

    scale_ = _scale_view(
        scale,
        local_xy
    )

    local_unscaled = local_xy / scale_

    if local_xy.dim() == 3:
        origin_ = origin.unsqueeze(1)
        raw_xy = (
            torch.matmul(
                local_unscaled,
                rot_mat.transpose(-1, -2)
            )
            + origin_
        )

    elif local_xy.dim() == 4:
        origin_ = origin.unsqueeze(1).unsqueeze(1)
        rot_mat_ = rot_mat.unsqueeze(1)

        raw_xy = (
            torch.matmul(
                local_unscaled,
                rot_mat_.transpose(-1, -2)
            )
            + origin_
        )

    else:
        raise ValueError(
            f"Unsupported local_xy shape: {local_xy.shape}"
        )

    return raw_xy


def inverse_transform_vector_torch(
    vector: torch.Tensor,
    origin: torch.Tensor,
    rot_mat: torch.Tensor,
    scale: torch.Tensor
) -> torch.Tensor:
    """
    Local 좌표계의 velocity/acceleration vector를 raw 좌표계로 복원.

    vector:
        [B, T, 2] 또는 [B, K, T, 2]

    translation은 적용하지 않는다.

    변환 관계:
        local_vector = (raw_vector @ rot_mat) * scale

    역변환:
        raw_vector = (local_vector / scale) @ rot_mat.T
    """
    _, rot_mat = _prepare_origin_rot(
        origin,
        rot_mat,
        vector
    )

    scale_ = _scale_view(
        scale,
        vector
    )

    local_unscaled = vector / scale_

    if vector.dim() == 3:
        raw_vector = torch.matmul(
            local_unscaled,
            rot_mat.transpose(-1, -2)
        )

    elif vector.dim() == 4:
        rot_mat_ = rot_mat.unsqueeze(1)

        raw_vector = torch.matmul(
            local_unscaled,
            rot_mat_.transpose(-1, -2)
        )

    else:
        raise ValueError(
            f"Unsupported vector shape: {vector.shape}"
        )

    return raw_vector


def inverse_transform_scalar_torch(
    value: torch.Tensor,
    origin: torch.Tensor,
    rot_mat: torch.Tensor,
    scale: torch.Tensor
) -> torch.Tensor:
    """
    Local 좌표계의 scalar speed/acc magnitude를 raw 기준으로 복원.

    scalar는 translation/rotation과 무관하고 scale만 복원한다.

    변환 관계:
        local_scalar = raw_scalar * scale

    역변환:
        raw_scalar = local_scalar / scale
    """
    scale_ = _scale_view(
        scale,
        value
    )

    return value / scale_


def restore_output_torch(
    output_local: torch.Tensor,
    output_mode: str,
    origin: torch.Tensor,
    rot_mat: torch.Tensor,
    scale: torch.Tensor
) -> torch.Tensor:
    """
    output_local:
        [B, T, C] 또는 [B, K, T, C]

    output_mode:
        xy
        xy_speed
        xy_vxvy

    반환:
        raw coordinate/system 기준 output
    """
    validate_output_mode(output_mode)

    restored = output_local.clone()

    # 1. x, y 복원
    restored[..., :2] = inverse_transform_xy_torch(
        output_local[..., :2],
        origin,
        rot_mat,
        scale
    )

    # 2. 추가 output이 없으면 종료
    if output_mode == "xy":
        return restored

    # 3. speed 복원
    if output_mode == "xy_speed":
        restored[..., 2:3] = inverse_transform_scalar_torch(
            output_local[..., 2:3],
            origin,
            rot_mat,
            scale
        )

    # 4. vx, vy 복원
    elif output_mode == "xy_vxvy":
        restored[..., 2:4] = inverse_transform_vector_torch(
            output_local[..., 2:4],
            origin,
            rot_mat,
            scale
        )

    else:
        raise ValueError(
            f"Unsupported output_mode: {output_mode}"
        )

    return restored