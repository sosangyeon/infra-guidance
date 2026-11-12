# utils.py

import os
import pickle
import random

import numpy as np
import torch
from sklearn.cluster import KMeans


def translation(seq, refer_point_index):
    """
    seq: [N, T, 2]

    refer_point_index 위치를 원점으로 평행이동.
    예:
        refer_point_index = obs_len - 1
        -> 마지막 관측 위치를 원점으로 이동
    """
    return seq - seq[:, refer_point_index:refer_point_index + 1]


def get_rot_mats(theta_list):
    """
    theta_list: [N]

    row-vector convention:
        local = raw @ R

    R(theta) =
        [[cos, -sin],
         [sin,  cos]]

    이 행렬은 row-vector 기준에서 heading vector를 +x축으로 맞추는 데 사용 가능.
    """
    rot_mat_list = []

    for theta in theta_list:
        c = np.cos(theta)
        s = np.sin(theta)

        rot_mat_list.append(
            np.array(
                [
                    [c, -s],
                    [s,  c],
                ],
                dtype=np.float32
            )
        )

    return np.stack(rot_mat_list, axis=0)


def rotation(seq, refer_point_index=0):
    """
    기존 함수 유지용.

    seq: [N, T, 2]

    주의:
        translation(seq, obs_len - 1) 이후에 refer_point_index=0을 쓰면
        첫 관측점이 '과거 방향'을 가리키기 때문에 motion mode 방향이 반대로 잡힐 수 있음.

    새 motion mode 생성에서는 이 함수 대신
    rotation_by_observed_heading()을 사용.
    """
    angle = np.arctan2(
        seq[:, refer_point_index, 1],
        seq[:, refer_point_index, 0]
    )

    rot_mat = get_rot_mats(angle)
    rot_seq = np.matmul(seq, rot_mat)

    return rot_seq, rot_mat


def rotation_by_observed_heading(seq, obs_len):
    """
    관측 진행 방향 기준 회전.

    seq: [N, T, 2]
         이미 마지막 관측 위치가 원점으로 translation된 상태라고 가정.
         즉 seq[:, obs_len - 1] ~= [0, 0]

    진행 방향:
        first_obs -> last_obs

    translation 이후:
        first_obs = first_obs - last_obs
        last_obs  = [0, 0]

    따라서 heading vector:
        heading = last_obs - first_obs
                = seq[:, obs_len - 1] - seq[:, 0]

    이 heading을 local +x 방향으로 맞춤.
    """

    if seq.ndim != 3 or seq.shape[-1] != 2:
        raise ValueError(f"seq must have shape [N, T, 2], got {seq.shape}")

    if seq.shape[1] <= obs_len:
        raise ValueError(
            f"seq length must be greater than obs_len. "
            f"seq.shape={seq.shape}, obs_len={obs_len}"
        )

    heading = seq[:, obs_len - 1, :] - seq[:, 0, :]  # [N, 2]

    # 정지에 가까운 trajectory 방어
    heading_norm = np.linalg.norm(heading, axis=-1)
    eps = 1e-8

    # heading이 너무 작으면 기본 +x 방향으로 둠
    safe_heading = heading.copy()
    invalid = heading_norm < eps
    safe_heading[invalid] = np.array([1.0, 0.0], dtype=np.float32)

    angle = np.arctan2(
        safe_heading[:, 1],
        safe_heading[:, 0]
    )

    rot_mat = get_rot_mats(angle)
    rot_seq = np.matmul(seq, rot_mat)

    return rot_seq, rot_mat


def simple_moving_average(input, windows_size):
    """
    input: [N, T, 2]
    """
    x_len = input.shape[1]

    if windows_size is None or windows_size <= 1:
        return input

    if x_len <= windows_size:
        return input

    x_list = []
    keep_num = int(np.floor(windows_size / 2))

    for i in range(windows_size):
        x_list.append(input[:, i:x_len - windows_size + 1 + i])

    x = sum(x_list) / windows_size
    x = np.concatenate(
        (
            input[:, :keep_num],
            x,
            input[:, -keep_num:],
        ),
        axis=1
    )

    return x


def dy_random_rotation(seqs):
    """
    seqs: [N, T, 2]

    motion mode augmentation용 random rotation.
    """
    random_angle = -1 + 2 * np.random.rand(seqs.shape[0])
    random_angle = np.arcsin(random_angle)

    rot_mat = get_rot_mats(random_angle)
    rot_seq = np.matmul(seqs, rot_mat)

    return rot_seq


def kmeans_(seq, n_clusters=100):
    """
    seq: [N, pred_len, 2]

    KMeans로 대표 motion mode 추출.
    OpenBLAS thread 문제 방지를 위해 threadpool_limits 사용.
    """

    if seq.ndim != 3 or seq.shape[-1] != 2:
        raise ValueError(f"seq must have shape [N, T, 2], got {seq.shape}")

    if seq.shape[0] < n_clusters:
        raise ValueError(
            f"Number of trajectories ({seq.shape[0]}) is smaller than "
            f"n_clusters ({n_clusters}). Reduce n_clusters."
        )

    input_data = np.ascontiguousarray(
        seq.reshape(seq.shape[0], -1),
        dtype=np.float32
    )

    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            clf = KMeans(
                n_clusters=n_clusters,
                random_state=1,
                n_init=10
            ).fit(input_data)

    except ImportError:
        # threadpoolctl이 없으면 기존 방식으로 실행
        clf = KMeans(
            n_clusters=n_clusters,
            random_state=1,
            n_init=10
        ).fit(input_data)

    centers = clf.cluster_centers_
    centers = centers.reshape(centers.shape[0], -1, 2)

    return centers.astype(np.float32)


def trajectory_motion_modes(
    all_trajs,
    obs_len,
    n_units=120,
    smooth_size=3,
    random_rotation=False
):
    """
    Extract representative motion patterns using KMeans.

    all_trajs: [N, obs_len + pred_len, 2]
               이미 local coordinate로 변환된 trajectory.

    반환:
        motion_modes: [n_units, pred_len, 2]
    """

    valid_mask = np.array(
        [traj.shape[0] > obs_len for traj in all_trajs]
    )

    if not np.any(valid_mask):
        raise ValueError(
            f"[ERROR] No valid trajectories longer than obs_len={obs_len}."
        )

    all_trajs = all_trajs[valid_mask]

    clustering_input = []

    for traj in all_trajs:
        if traj.shape[0] <= obs_len:
            continue

        # future displacement
        # local coordinate에서 obs_len-1 위치는 원점이어야 함
        motion = traj[obs_len:] - traj[obs_len - 1]
        clustering_input.append(motion)

    if len(clustering_input) == 0:
        raise ValueError(
            "[ERROR] All trajectories are too short for motion mode extraction."
        )

    clustering_input = np.stack(clustering_input, axis=0).astype(np.float32)

    if smooth_size is not None and clustering_input.shape[1] > smooth_size:
        clustering_input = simple_moving_average(
            clustering_input,
            windows_size=smooth_size
        )

    if random_rotation:
        clustering_input = dy_random_rotation(clustering_input)

    motion_modes = kmeans_(
        clustering_input,
        n_clusters=n_units
    )

    return motion_modes


def get_motion_modes(
    dataset,
    obs_len,
    pred_len,
    n_clusters,
    dataset_path,
    dataset_name,
    smooth_size,
    random_rotation,
    traj_seg=False
):
    """
    dataset.scenario_list에서 motion mode 생성.

    scenario format:
        item[0]: observed trajectory, [obs_len, D]
        item[1]: future trajectory,   [pred_len, D]
        item[2]: neighbor trajectory

    여기서는 motion mode 생성을 위해 x,y만 사용.
    """

    trajs_list = []
    index1 = [0, 1, 2, 3, 4, 5]

    traj_scenarios = dataset.scenario_list

    for scenario_idx in range(len(traj_scenarios)):
        curr_ped_obs = traj_scenarios[scenario_idx][0][:, :2]
        curr_ped_pred = traj_scenarios[scenario_idx][1][:, :2]

        curr_traj = np.concatenate(
            (curr_ped_obs, curr_ped_pred),
            axis=0
        ).astype(np.float32)  # [T, 2]

        if curr_traj.shape[0] < obs_len + pred_len:
            continue

        if traj_seg:
            for start_idx in index1:
                seq = curr_traj[start_idx:start_idx + pred_len + 2]

                if seq.shape[0] <= 0:
                    continue

                required_len = obs_len + pred_len

                if seq.shape[0] < required_len:
                    pre_seq = np.repeat(
                        seq[0:1],
                        required_len - seq.shape[0],
                        axis=0
                    )
                    seq = np.concatenate((pre_seq, seq), axis=0)

                if seq.shape[0] == required_len:
                    trajs_list.append(seq.astype(np.float32))

        trajs_list.append(curr_traj)

    trajs_list = [
        traj for traj in trajs_list
        if traj.shape[0] > obs_len
    ]

    if len(trajs_list) == 0:
        raise ValueError(
            f"[ERROR] No trajectory has more than obs_len={obs_len} frames."
        )

    all_trajs = np.stack(trajs_list, axis=0).astype(np.float32)  # [N, T, 2]

    # 1. 마지막 관측 위치를 원점으로 이동
    all_trajs = translation(all_trajs, obs_len - 1)

    # 2. 관측 진행 방향 first_obs -> last_obs 기준으로 회전
    #    기존 rotation(all_trajs, 0)은 first_obs가 과거 방향이라
    #    motion mode가 반대로 잡힐 수 있음.
    all_trajs, _ = rotation_by_observed_heading(
        all_trajs,
        obs_len=obs_len
    )

    motion_modes = trajectory_motion_modes(
        all_trajs,
        obs_len,
        n_units=n_clusters,
        smooth_size=smooth_size,
        random_rotation=random_rotation
    )

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    save_path_file = os.path.join(
        dataset_path,
        f"{dataset_name}_motion_modes.pkl"
    )

    with open(save_path_file, "wb") as f:
        pickle.dump(motion_modes, f)

    print("Finished")
    print(f"Saved motion modes: {save_path_file}")
    print(f"motion_modes shape: {motion_modes.shape}")

    return motion_modes


def saving_motion_modes(
    dataloader,
    motion_modes,
    obs_len,
    dataset_path,
    dataset_name
):
    """
    motion mode와 가장 가까운 mode index / soft label 저장.

    주의:
        현재 메인 학습 코드에서는 보통 이 함수를 직접 쓰지 않고,
        train.py 안에서 get_cls_label()을 통해 batch마다 계산함.

    dataloader.seq_array가 있는 예전 코드 호환용 함수.
    """

    closest_mode_indices_list = []
    cls_soft_label_list = []

    traj_scenes = dataloader.seq_array

    for i in range(traj_scenes.shape[0]):
        curr_scene = traj_scenes[i]  # [N, T, 2]
        curr_traj = curr_scene[0:1]  # [1, T, 2]

        norm_curr_traj = translation(
            curr_traj,
            obs_len - 1
        )

        norm_curr_traj, _ = rotation_by_observed_heading(
            norm_curr_traj,
            obs_len=obs_len
        )

        norm_curr_traj = norm_curr_traj[:, obs_len:]
        norm_curr_traj = norm_curr_traj.reshape(1, -1)

        norm_curr_traj = np.repeat(
            norm_curr_traj,
            motion_modes.shape[0],
            axis=0
        )

        traj_units_ = motion_modes.reshape(
            motion_modes.shape[0],
            -1
        )

        distance = np.linalg.norm(
            norm_curr_traj - traj_units_,
            axis=-1
        )

        closest_unit_indices = np.argmin(distance)
        closest_unit_indices = np.expand_dims(
            closest_unit_indices,
            axis=0
        )

        closest_mode_indices_list.append(
            closest_unit_indices
        )
        cls_soft_label_list.append(
            -distance
        )

    closest_mode_indices_array = np.concatenate(
        closest_mode_indices_list,
        axis=0
    )

    cls_soft_label_array = np.stack(
        cls_soft_label_list,
        axis=0
    )

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    np.save(
        os.path.join(dataset_path, f"{dataset_name}_motion_modes.npy"),
        motion_modes
    )

    np.save(
        os.path.join(dataset_path, f"{dataset_name}_closest_mode_indices.npy"),
        closest_mode_indices_array
    )

    np.save(
        os.path.join(dataset_path, f"{dataset_name}_cls_soft_label.npy"),
        cls_soft_label_array
    )


def seed(seed: int):
    rand = seed is None

    if seed is None:
        seed = int.from_bytes(
            os.urandom(4),
            byteorder="big"
        )

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = not rand
    torch.backends.cudnn.benchmark = rand


def get_rng_state(device):
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state(device)
        if torch.cuda.is_available() and "cuda" in str(device)
        else None,
        np.random.get_state(),
        random.getstate(),
    )