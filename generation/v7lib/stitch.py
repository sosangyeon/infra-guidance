"""stitch — 생성 궤적의 obs/future 이음새 보정. 원본 구현(stitch_aug.py) 그대로.

obs 끝 속도로 외삽한 위치와 future 첫 점의 차이를 future 앞 K 스텝에 cosine 램프로 감쇠 적용. obs 구간은 불변.
"""
import numpy as np

STITCH_K = 5


def stitch_traj(traj, obs, K=STITCH_K):
    traj = np.asarray(traj, dtype=np.float64).copy()
    if len(traj) < obs + 2 or obs < 2:
        return traj
    v_obs = traj[obs - 1] - traj[obs - 2]
    corr = traj[obs - 1] + v_obs - traj[obs]
    for k in range(K):
        t = obs + k
        if t >= len(traj):
            break
        traj[t] = traj[t] + corr * 0.5 * (1 + np.cos(np.pi * k / K))
    return traj


def seam_jump(traj, obs):
    traj = np.asarray(traj, dtype=np.float64)
    return float(np.linalg.norm(traj[obs] - 2 * traj[obs - 1] + traj[obs - 2]))
