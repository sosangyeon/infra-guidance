"""LCSim 생성 + time-shift 결과 CSV -> TUTR 학습 pkl + meta CSV (gal_dataset.GalDataset 입력).

입력: <gen_dir>/scenario_<sid>/scenario_<sid>_aug<NNN>.csv  (generation/postprocess_shift.py 출력)
      열: scenario_id, aug_id, seed, track_id, timestep, observed, extrap, position_x/y, heading, velocity_x/y
출력: <out>.pkl, <out>_meta.csv
  샘플 = (hist[10, 8], future[30, 8], neighbor[40, N, 8], fut_valid[30])  — make_tutr_pkl.py 와 같은 8채널
  meta = idx, scenario_id(<sid>_aug<NNN>), track_id, src_scenario_id, aug_id, n_agents, n_pad

규칙
  - 생성 장면(aug 파일) 하나 = 시나리오 하나. 차량마다 ego 샘플 하나, 나머지 차량이 이웃
    (make_tutr_pkl.scenario_samples 와 같음).
  - ax, ay 는 생성 궤적에 없어서 속도의 후방차분 / 0.1 s 로 만든다 (첫 프레임은 둘째 값).
    실측 Dataset_CSV 가속도와 표준편차가 같고(1.95 vs 1.95 m/s², SS 20250101) 상관 0.67.
  - extrap = 1 (time-shift pad) 인 미래 프레임은 fut_valid = 0. 관측 구간에 pad 가 있는 차량
    (원본 결측)은 빼고, 그 차량은 이웃으로도 넣지 않는다.
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
from feature_utils import FULL_STATE_DIM, recompute_scalar_features_np  # noqa: E402

OBS_LEN, HORIZON, DT = 10, 40, 0.1


def to_state(x, y, vx, vy):
    """[A, T] 배열들 -> [A, T, 8] (x, y, vx, vy, speed, ax, ay, acc)."""
    ax = np.diff(vx, axis=1, prepend=vx[:, :1]) / DT
    ay = np.diff(vy, axis=1, prepend=vy[:, :1]) / DT
    ax[:, 0], ay[:, 0] = ax[:, 1], ay[:, 1]
    out = np.zeros(x.shape + (FULL_STATE_DIM,), dtype=np.float32)
    out[..., 0], out[..., 1], out[..., 2], out[..., 3] = x, y, vx, vy
    out[..., 5], out[..., 6] = ax, ay
    return recompute_scalar_features_np(out).astype(np.float32)


def aug_samples(df):
    """aug CSV 하나 -> (samples, track_ids, n_pad 목록). 쓸 수 있는 차량이 없으면 빈 목록."""
    df = df.sort_values(["track_id", "timestep"])
    cnt = df.groupby("track_id")["timestep"].count()
    df = df[df["track_id"].isin(cnt.index[cnt == HORIZON])]
    hist_pad = df[(df["timestep"] < OBS_LEN) & (df["extrap"] == 1)]["track_id"].unique()
    df = df[~df["track_id"].isin(hist_pad)]
    tids = df["track_id"].unique()
    a = len(tids)
    if a == 0:
        return [], [], []
    arr = lambda c: df[c].to_numpy(dtype=np.float32).reshape(a, HORIZON)  # noqa: E731
    st = to_state(arr("position_x"), arr("position_y"), arr("velocity_x"), arr("velocity_y"))
    valid = 1.0 - arr("extrap")[:, OBS_LEN:]
    samples, n_pad = [], []
    for i in range(a):
        nei = np.delete(st, i, axis=0).transpose(1, 0, 2)
        samples.append((st[i, :OBS_LEN].copy(), st[i, OBS_LEN:].copy(), np.ascontiguousarray(nei),
                        valid[i].astype(np.float32)))
        n_pad.append(int((valid[i] == 0).sum()))
    return samples, tids, n_pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True, help="postprocess_shift.py 출력 폴더")
    ap.add_argument("--out", required=True, help="출력 경로 앞부분 (<out>.pkl, <out>_meta.csv)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.gen_dir, "scenario_*", "scenario_*_aug*.csv")))
    if not files:
        sys.exit(f"생성 CSV 없음: {args.gen_dir}")
    data, rows, dropped = [], [], 0
    for f in files:
        df = pd.read_csv(f)
        if df.empty:
            continue
        sid = str(df["scenario_id"].iloc[0])
        aug = int(df["aug_id"].iloc[0])
        samples, tids, n_pad = aug_samples(df)
        dropped += df["track_id"].nunique() - len(tids)
        data.extend(samples)
        rows.extend((f"{sid}_aug{aug:03d}", int(t), sid, aug, len(tids), p) for t, p in zip(tids, n_pad))

    meta = pd.DataFrame(rows, columns=["scenario_id", "track_id", "src_scenario_id", "aug_id",
                                       "n_agents", "n_pad"])
    meta.insert(0, "idx", np.arange(len(meta)))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".pkl", "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    meta.to_csv(args.out + "_meta.csv", index=False)
    print(f"[gen->pkl] {len(files)} aug 파일, 원본 {meta['src_scenario_id'].nunique()} 시나리오 -> "
          f"샘플 {len(data):,} (뺀 차량 {dropped}) | pad 있는 샘플 {(meta['n_pad'] > 0).mean():.1%}, "
          f"평균 pad {meta['n_pad'].mean():.2f} 프레임 -> {args.out}.pkl")


if __name__ == "__main__":
    main()
