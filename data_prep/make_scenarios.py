"""Dataset_CSV(8컬럼) 일별 CSV -> 40프레임 시나리오 + manifest.

창 자르기 규칙은 연구실 Data_Processing/preprocessing_lis_sliding.py 와 같다.
  1) 파일 전체에서 mean|vx|, mean|vy| 가 둘 다 0.05 이하인 track 제거
  2) 시작 프레임 0, s, 2s, ... 마다 40프레임 창 (s = --window_step, 기본 1. 연구실은 10)
  3) 창 안에서 40프레임을 모두 가진 track 만
  4) 창 안에서 |mean vx|, |mean vy| 가 둘 다 0.05 이하인 track 제거
  5) 남은 track 이 MIN_AGENTS(2) 미만이면 창을 버림
연구실 코드에 없는 거리 조건 (--interact_dist, 기본 50 m. 0 이면 끔)
  6) 관측 구간(t=0~9) 평균 거리가 50 m 미만인 다른 track 이 하나도 없는 track 제거
  7) 다시 MIN_AGENTS 미만이면 창을 버림
  -> 모든 차량이 50 m 안에 이웃을 하나 이상 가진다. 50 m 는 TUTR dist_threshold,
     QCNet a2a_radius 와 같은 값이고, 거리 계산도 TUTR dataset.py 와 같다.
연구실 코드와 다른 점
  - 입력이 원본(ObjectID, Heading, Speed) 이 아니라 Dataset_CSV(frame_id,track_id,x,y,vx,vy,ax,ay)
  - heading 은 atan2(vy, vx). 속도가 0.5 m/s 미만인 프레임은 직전 heading 을 이어 쓴다
  - ax, ay 를 버리지 않고 acceleration_x, acceleration_y 로 남긴다
  - 시나리오마다 파일을 만들지 않고 날짜별 parquet 하나에 모은다 (scenario_id 열로 구분)

출력
  <out>/<site>/<YYYYMMDD>.parquet   시나리오 행 (연구실 AV2 유사 열 + 가속도)
  <out>/manifest_<site>.csv         scenario_id, date, start_frame, n_agents, max_step_m

사용 예
  python make_scenarios.py --site ss
  python make_scenarios.py --site ny
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(HERE, "..", "data", "Dataset_CSV")
DEFAULT_OUT = os.path.join(HERE, "..", "data", "scenarios")

INPUT_FRAME = 10
OUTPUT_FRAME = 30
WINDOW_STEP = 1                       # 연구실은 10. 2대 이상 창이 적어 1로 낮춤 (분할이 날짜 단위라 겹쳐도 누출 없음)
MIN_AGENTS = 2
STATIONARY_THRESH = 0.05
HEADING_MIN_SPEED = 0.5
INTERACT_DIST = 50.0

FILENAME_PATTERN = re.compile(r"(\d{8})_track\.csv$", re.IGNORECASE)
CITY = {"ss": "sesol", "ny": "namyang"}

OUT_COLS = [
    "observed", "track_id", "object_type", "object_category", "timestep",
    "position_x", "position_y", "heading", "velocity_x", "velocity_y",
    "acceleration_x", "acceleration_y",
    "scenario_id", "start_timestamp", "end_timestamp", "num_timestamps",
    "focal_track_id", "city", "Data_Num",
]


def track_heading(df):
    """atan2(vy, vx). 저속 프레임은 같은 track 의 직전 값(없으면 다음 값)으로 채운다."""
    hdg = np.arctan2(df["vy"].to_numpy(), df["vx"].to_numpy())
    slow = np.hypot(df["vx"].to_numpy(), df["vy"].to_numpy()) < HEADING_MIN_SPEED
    s = pd.Series(np.where(slow, np.nan, hdg), index=df.index)
    s = s.groupby(df["track_id"]).transform(lambda x: x.ffill().bfill())
    return s.fillna(0.0).to_numpy()


def remove_stationary_tracks(df):
    g = df.assign(avx=df["vx"].abs(), avy=df["vy"].abs()).groupby("track_id")[["avx", "avy"]].mean()
    stop = g[(g["avx"] <= STATIONARY_THRESH) & (g["avy"] <= STATIONARY_THRESH)].index
    return df[~df["track_id"].isin(stop)]


def window_tracks(df, frame_interval, window_step):
    """(start_frame, track_id) 목록: 창 [s, s+frame_interval) 를 빠짐없이 가진 track."""
    out = []
    for tid, f in df.groupby("track_id")["frame_id"]:
        f = np.sort(f.to_numpy())
        # 연속 구간 [a, b]
        cut = np.flatnonzero(np.diff(f) != 1)
        starts = np.r_[f[0], f[cut + 1]]
        ends = np.r_[f[cut], f[-1]]
        for a, b in zip(starts, ends):
            s0 = -(-a // window_step) * window_step
            s1 = b - frame_interval + 1
            if s1 >= s0:
                for s in range(s0, s1 + 1, window_step):
                    out.append((s, tid))
    return pd.DataFrame(out, columns=["start_frame", "track_id"])


def isolated_tracks(rows, interact_dist):
    """관측 구간 평균 거리 interact_dist 미만인 이웃이 없는 (start_frame, track_id) 목록."""
    obs = rows[rows["timestep"] < INPUT_FRAME].sort_values(["start_frame", "track_id", "timestep"])
    drop = []
    for s, g in obs.groupby("start_frame", sort=False):
        tids = g["track_id"].unique()
        xy = g[["x", "y"]].to_numpy().reshape(len(tids), INPUT_FRAME, 2)
        d = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1).mean(-1)  # [A, A]
        np.fill_diagonal(d, np.inf)
        drop.extend((s, t) for t in tids[d.min(1) >= interact_dist])
    return pd.DataFrame(drop, columns=["start_frame", "track_id"])


def process_day(path, date_str, data_num, city, window_step, min_agents, interact_dist):
    frame_interval = INPUT_FRAME + OUTPUT_FRAME
    df = pd.read_csv(path).sort_values(["track_id", "frame_id"]).reset_index(drop=True)
    df = remove_stationary_tracks(df).reset_index(drop=True)
    if df.empty:
        return None, None
    df["heading"] = track_heading(df)

    wt = window_tracks(df, frame_interval, window_step)
    if wt.empty:
        return None, None

    # 창 x track 행 펼치기
    off = np.arange(frame_interval)
    rows = pd.DataFrame({
        "start_frame": np.repeat(wt["start_frame"].to_numpy(), frame_interval),
        "track_id": np.repeat(wt["track_id"].to_numpy(), frame_interval),
        "timestep": np.tile(off, len(wt)),
    })
    rows["frame_id"] = rows["start_frame"] + rows["timestep"]
    rows = rows.merge(df, on=["track_id", "frame_id"], how="left", validate="many_to_one")

    # 창 안 정지 track 제거 (연구실: |mean v| 기준)
    key = ["start_frame", "track_id"]
    mvx = rows.groupby(key)["vx"].transform("mean").abs()
    mvy = rows.groupby(key)["vy"].transform("mean").abs()
    rows = rows[(mvx > STATIONARY_THRESH) | (mvy > STATIONARY_THRESH)]

    # 최소 차량 수
    n_agents = rows.groupby("start_frame")["track_id"].transform("nunique")
    rows = rows[n_agents >= min_agents].copy()
    if rows.empty:
        return None, None

    # 50 m 안에 이웃이 없는 차량 제거 후 다시 최소 차량 수
    if interact_dist > 0:
        iso = isolated_tracks(rows, interact_dist)
        if not iso.empty:
            m = rows.merge(iso.assign(_iso=True), on=key, how="left")["_iso"].notna().to_numpy()
            rows = rows[~m]
            n_agents = rows.groupby("start_frame")["track_id"].transform("nunique")
            rows = rows[n_agents >= min_agents].copy()
            if rows.empty:
                return None, None

    # 프레임 간 최대 이동거리 (순간이동 확인용, 필터는 하지 않음)
    rows = rows.sort_values(["start_frame", "track_id", "timestep"]).reset_index(drop=True)
    step = np.hypot(rows.groupby(key)["x"].diff(), rows.groupby(key)["y"].diff())
    rows["_step"] = step

    sid = date_str + "_" + rows["start_frame"].astype(str)
    out = pd.DataFrame({
        "observed": rows["timestep"] <= INPUT_FRAME - 1,
        "track_id": rows["track_id"].astype(int),
        "object_type": "VEHICLE",
        "object_category": 3,
        "timestep": rows["timestep"].astype(int),
        "position_x": rows["x"].astype(float),
        "position_y": rows["y"].astype(float),
        "heading": rows["heading"].astype(float),
        "velocity_x": rows["vx"].astype(float),
        "velocity_y": rows["vy"].astype(float),
        "acceleration_x": rows["ax"].astype(float),
        "acceleration_y": rows["ay"].astype(float),
        "scenario_id": sid,
        "start_timestamp": 0,
        "end_timestamp": 0,
        "num_timestamps": frame_interval,
        "focal_track_id": 0,
        "city": city,
        "Data_Num": data_num,
    })[OUT_COLS]

    man = rows.assign(scenario_id=sid).groupby("scenario_id").agg(
        start_frame=("start_frame", "first"),
        n_agents=("track_id", "nunique"),
        max_step_m=("_step", "max"),
    ).reset_index()
    man.insert(1, "date", date_str)
    man = man.sort_values("start_frame").reset_index(drop=True)
    return out, man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=sorted(CITY))
    ap.add_argument("--src", default=DEFAULT_SRC, help="Dataset_CSV 폴더")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--window_step", type=int, default=WINDOW_STEP)
    ap.add_argument("--min_agents", type=int, default=MIN_AGENTS)
    ap.add_argument("--interact_dist", type=float, default=INTERACT_DIST,
                    help="관측 구간 평균 거리가 이 값 미만인 이웃이 없는 차량 제거 (m). 0 이면 끔 = 연구실 규칙")
    ap.add_argument("--limit", type=int, default=0, help="앞에서부터 N일만 (시험용)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, f"{args.site}_bench_o", "*", "*_track.csv")),
                   key=lambda p: os.path.basename(p))
    if not files:
        raise FileNotFoundError(f"{args.src}/{args.site}_bench_o/*/*_track.csv 없음")
    if args.limit:
        files = files[:args.limit]

    out_dir = os.path.join(args.out, args.site)
    os.makedirs(out_dir, exist_ok=True)

    mans = []
    for i, path in enumerate(files):
        date_str = FILENAME_PATTERN.search(os.path.basename(path)).group(1)
        out, man = process_day(path, date_str, f"{i:04}", CITY[args.site],
                               args.window_step, args.min_agents, args.interact_dist)
        if out is None:
            print(f"[{date_str}] 시나리오 없음", flush=True)
            continue
        out.to_parquet(os.path.join(out_dir, f"{date_str}.parquet"), index=False)
        mans.append(man)
        print(f"[{date_str}] 시나리오 {len(man):,}  차량-샘플 {int(man['n_agents'].sum()):,}", flush=True)

    manifest = pd.concat(mans, ignore_index=True)
    mpath = os.path.join(args.out, f"manifest_{args.site}.csv")
    manifest.to_csv(mpath, index=False)
    print(f"\n{args.site}: {len(files)}일, 시나리오 {len(manifest):,}, "
          f"차량-샘플 {int(manifest['n_agents'].sum()):,} -> {mpath}")


if __name__ == "__main__":
    main()
