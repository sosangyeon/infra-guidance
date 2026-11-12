"""
lis_vel_mean.py — LIS h5 데이터에서 VEL_MEAN(속도 정규화 상수)을 실측한다.

model.py의 target="module_vel"은 미래 궤적을 (속력모듈 - VEL_MEAN, 각도)로 표현한다.
여기서 VEL_MEAN = 미래 구간 step당 변위 크기 ‖pos[t]-pos[t-1]‖ 의 데이터셋 평균
(= 평균속력 m/s × step_interval). 회전/평행이동 불변이므로 global xyz로 계산.

도메인마다 다르므로(waymo 0.2417, argoverse 0.3741) LIS는 반드시 실측해
motion_diff/model.py 의 VEL_MEAN 상수에 넣어야 한다 (config 로는 주입되지 않음).

사용:
    python motion_diff/dataset/process/lis_vel_mean.py \
        --h5 /path/to/train.h5 --num-historical-steps 10
"""

from argparse import ArgumentParser

import h5py
import numpy as np


def measure_vel_mean(h5_path: str, num_historical_steps: int = 10) -> dict:
    disp = []  # future per-step displacement magnitudes (matches model.py exactly)
    with h5py.File(h5_path, "r") as hf:
        for k in hf.keys():
            ep = hf[k]
            xyz = ep["agent_xyz"][:]  # (N, T, 3)
            valid = ep["agent_valid"][:][..., 0]  # (N, T)
            T = xyz.shape[1]
            for i in range(xyz.shape[0]):
                for t in range(num_historical_steps, T):
                    if valid[i, t] and valid[i, t - 1]:
                        disp.append(np.linalg.norm(xyz[i, t, :2] - xyz[i, t - 1, :2]))
    disp = np.array(disp)
    return {
        "vel_mean": float(disp.mean()),
        "std": float(disp.std()),
        "n_samples": int(len(disp)),
        "mean_speed_mps": float(disp.mean() / 0.1),
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--h5", required=True, help="train.h5 경로")
    parser.add_argument("--num-historical-steps", type=int, default=10)
    args = parser.parse_args()
    r = measure_vel_mean(args.h5, args.num_historical_steps)
    print(f"VEL_MEAN = {r['vel_mean']:.4f}  (std={r['std']:.4f}, n={r['n_samples']})")
    print(f"  implied mean speed = {r['mean_speed_mps']:.2f} m/s")
    print(f"  -> set VEL_MEAN = {r['vel_mean']:.4f} in motion_diff/model.py")


if __name__ == "__main__":
    main()
