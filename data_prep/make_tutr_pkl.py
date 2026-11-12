"""시나리오 parquet + manifest -> TUTR pkl (분할별) + meta CSV.

pkl 형식은 tutr/vanilla/dataset.py 가 읽는 형식이다.
  샘플 = (hist[10, 8], future[30, 8], neighbor[40, N, 8])
  8채널 = [x, y, vx, vy, speed, ax, ay, acc]  (ax, ay 는 Dataset_CSV 값 그대로)
시나리오 하나(40프레임, 차량 A대)에서 차량마다 ego 샘플을 하나씩 만들고, 나머지 A-1대를
이웃으로 넣는다. 이웃은 시나리오 안의 차량만 쓴다.

meta CSV 는 pkl 과 같은 순서로 샘플마다 한 줄:
  idx, scenario_id, track_id, date, split, n_agents
GAL 가중치, contrastive CP 등 방법별 값은 scenario_id/track_id 로 이 표에 붙인다.

출력 (<out>/<site>/)
  <site>_{init,remain,val,test}.pkl  + 같은 이름의 _meta.csv
  <site>_train.pkl                   init + remain (Vanilla 용, --no_train 이면 생략)

학습·검증에 쓸 분할은 train.py 의 --train_splits / --val_split 으로 고른다 (Vanilla: init / val).

사용 예
  python make_tutr_pkl.py --site ss
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCN = os.path.join(HERE, "..", "data", "scenarios")
DEFAULT_OUT = os.path.join(HERE, "..", "data", "tutr")
TUTR = os.path.join(HERE, "..", "tutr", "vanilla")

sys.path.insert(0, os.path.abspath(TUTR))
from feature_utils import FULL_STATE_DIM, recompute_scalar_features_np  # noqa: E402

OBS_LEN = 10
HORIZON = 40
SPLITS = ["init", "remain", "val", "test"]
STATE_COLS = ["position_x", "position_y", "velocity_x", "velocity_y",
              "acceleration_x", "acceleration_y"]


def to_state(arr6):
    """[..., 6] (x, y, vx, vy, ax, ay) -> [..., 8] (x, y, vx, vy, speed, ax, ay, acc)."""
    out = np.zeros(arr6.shape[:-1] + (FULL_STATE_DIM,), dtype=np.float32)
    out[..., 0:4] = arr6[..., 0:4]
    out[..., 5:7] = arr6[..., 4:6]
    return recompute_scalar_features_np(out).astype(np.float32)


def scenario_samples(g):
    """한 시나리오 DataFrame -> 샘플 목록과 track_id 목록."""
    g = g.sort_values(["track_id", "timestep"])
    tids = g["track_id"].unique()
    a = len(tids)
    st = to_state(g[STATE_COLS].to_numpy(dtype=np.float32).reshape(a, HORIZON, 6))  # [A, T, 8]
    samples = []
    for i in range(a):
        nei = np.delete(st, i, axis=0).transpose(1, 0, 2)  # [T, A-1, 8]
        samples.append((st[i, :OBS_LEN].copy(), st[i, OBS_LEN:].copy(), np.ascontiguousarray(nei)))
    return samples, tids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--scenarios", default=DEFAULT_SCN, help="make_scenarios.py 출력 폴더")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no_train", action="store_true", help="init+remain 합본(<site>_train.pkl)을 만들지 않음")
    args = ap.parse_args()

    man = pd.read_csv(os.path.join(args.scenarios, f"manifest_{args.site}.csv"), dtype={"date": str})
    if "split" not in man.columns:
        raise ValueError("manifest 에 split 열이 없음. split_dates.py 를 먼저 실행")
    split_of = dict(zip(man["scenario_id"], man["split"]))

    data = {s: [] for s in SPLITS}
    meta = {s: [] for s in SPLITS}
    files = sorted(glob.glob(os.path.join(args.scenarios, args.site, "*.parquet")))
    for k, f in enumerate(files):
        df = pd.read_parquet(f, columns=["scenario_id", "track_id", "timestep"] + STATE_COLS)
        for sid, g in df.groupby("scenario_id", sort=True):
            split = split_of[sid]
            samples, tids = scenario_samples(g)
            data[split].extend(samples)
            date = sid.split("_")[0]
            meta[split].extend((sid, int(t), date, split, len(tids)) for t in tids)
        sys.stdout.write(f"\r[{args.site}] {k + 1}/{len(files)} 일")
        sys.stdout.flush()
    print()

    out_dir = os.path.join(args.out, args.site)
    os.makedirs(out_dir, exist_ok=True)

    def save(name, samples, rows):
        with open(os.path.join(out_dir, f"{name}.pkl"), "wb") as fh:
            pickle.dump(samples, fh, protocol=pickle.HIGHEST_PROTOCOL)
        m = pd.DataFrame(rows, columns=["scenario_id", "track_id", "date", "split", "n_agents"])
        m.insert(0, "idx", np.arange(len(m)))
        m.to_csv(os.path.join(out_dir, f"{name}_meta.csv"), index=False)
        print(f"  {name}.pkl  샘플 {len(samples):,}  시나리오 {m['scenario_id'].nunique():,}")

    for s in SPLITS:
        save(f"{args.site}_{s}", data[s], meta[s])
    if not args.no_train:
        save(f"{args.site}_train", data["init"] + data["remain"], meta["init"] + meta["remain"])


if __name__ == "__main__":
    main()
