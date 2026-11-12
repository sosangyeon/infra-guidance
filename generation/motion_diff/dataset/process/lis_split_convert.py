"""
lis_split_convert.py — SS/NY Data_Split_s4 시나리오 CSV -> LCSim(motion_diff) train.h5 / val.h5.

lis_convert.py 는 K-City(Argoverse2 형식) CSV 를 받는다
(timestep, position_x/y, velocity_x/y, heading, scenario_id ...).
SS/NY 분할 CSV 는 백본 형식이라 열이 다르다:

    frame_id, track_id, x, y, vx, vy, length, width, agent_type, ax, ay

이 스크립트는 그 차이만 메우고, h5 스키마와 지도 디코딩은 lis_utils 를 그대로 쓴다.

- 분할: r0 와 같은 날짜 분할을 그대로 쓴다. <split-root>/<site>/train -> train.h5,
  <site>/val -> val.h5. **test 폴더는 읽지 않는다** (생성모델에 test 정보 누출 방지).
- timestep: 파일 안의 frame_id - min(frame_id) (파일 = 40 프레임 창).
- heading: CSV 에 없으므로 위치 중심차분(±2 스텝) 변위 방향으로 만든다
  (_track_heading 참고; 센서 vx, vy 는 일부 프레임에서 방향이 뒤집혀 있어 쓰지 않음).
- shape: CSV 의 length/width 사용 (높이 1.5 m 고정; lis_utils 는 CSV 에 크기가 없어 상수).
- 좌표: 지도(<SITE>_map.json)와 같은 좌표계임을 확인함 (샘플 점 100% 주행가능영역 안).

사용:
    python motion_diff/dataset/process/lis_split_convert.py \
        --split-root J:/Dataset_CSV_year/Data_Split_s4 --site ss \
        --map-path ../data/maps/SS_map.json --out-dir ./h5_out_ss

출력: <out-dir>/train.h5, <out-dir>/val.h5, <out-dir>/manifest.json
"""

import glob
import json
import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lis_utils import VEHICLE_TYPE_CODE, decode_lis_map

NUM_STEPS = 40
HEIGHT = 1.5
HEADING_HALF_WIN = 2     # heading 용 중심차분 반폭 [step] (±0.2 s)
HEADING_MIN_DISP = 0.5   # m; 중심차분 변위가 이보다 짧으면 방향이 지터라 이웃 프레임 heading 으로 채움
HEADING_MIN_SPEED = 0.5  # m/s; 트랙 전체가 정지에 가까울 때만 쓰는 속도 방향의 하한
SPLITS = ("train", "val")  # test 는 의도적으로 제외


def _track_heading(vx, vy, x, y):
    """위치 변위 방향을 heading 으로 쓴다.

    샘플 검증에서 atan2(vy, vx) 는 프레임 ~1% 가 실제 진행방향과 반대(센서 속도
    부호 뒤집힘)였고, 1스텝 변위는 위치 지터로 흔들렸다. 그래서 ±HEADING_HALF_WIN
    스텝 중심차분 변위가 HEADING_MIN_DISP 이상인 프레임은 그 방향을, 나머지는 가장
    가까운 그런 프레임의 값을 쓴다. 트랙 전체가 거의 정지면 속도 방향(주행 프레임
    한정), 그것도 없으면 0.
    """
    n = len(x)
    lo = np.clip(np.arange(n) - HEADING_HALF_WIN, 0, n - 1)
    hi = np.clip(np.arange(n) + HEADING_HALF_WIN, 0, n - 1)
    dx, dy = x[hi] - x[lo], y[hi] - y[lo]
    h = np.arctan2(dy, dx)
    ok = np.hypot(dx, dy) >= HEADING_MIN_DISP
    if not ok.any():
        ok = np.hypot(vx, vy) >= HEADING_MIN_SPEED
        h = np.arctan2(vy, vx)
        if not ok.any():
            return np.zeros(n)
    if ok.all():
        return h
    idx = np.flatnonzero(ok)
    nearest = idx[np.abs(np.arange(n)[:, None] - idx[None, :]).argmin(axis=1)]
    return h[nearest]


def decode_split_csv(csv_path, num_steps=NUM_STEPS):
    df = pd.read_csv(csv_path)
    df = df.sort_values(["track_id", "frame_id"])
    f0 = int(df["frame_id"].min())
    span = int(df["frame_id"].max()) - f0 + 1
    if span != num_steps:
        raise ValueError(f"frame span {span} != {num_steps}")

    track_ids = list(dict.fromkeys(df["track_id"].tolist()))
    n = len(track_ids)
    agent_xyz = np.zeros((n, num_steps, 3), dtype=np.float32)
    agent_vel = np.zeros((n, num_steps, 2), dtype=np.float32)
    agent_heading = np.zeros((n, num_steps, 1), dtype=np.float32)
    agent_shape = np.zeros((n, num_steps, 3), dtype=np.float32)
    agent_valid = np.zeros((n, num_steps, 1), dtype=bool)
    agent_type = np.full((n,), VEHICLE_TYPE_CODE, dtype=np.int8)

    for i, (_, g) in enumerate(df.groupby("track_id", sort=False)):
        t = g["frame_id"].to_numpy() - f0
        x, y = g["x"].to_numpy(), g["y"].to_numpy()
        vx, vy = g["vx"].to_numpy(), g["vy"].to_numpy()
        agent_xyz[i, t, 0], agent_xyz[i, t, 1] = x, y
        agent_vel[i, t, 0], agent_vel[i, t, 1] = vx, vy
        agent_heading[i, t, 0] = _track_heading(vx, vy, x, y)
        agent_shape[i, t, 0] = g["length"].to_numpy()
        agent_shape[i, t, 1] = g["width"].to_numpy()
        agent_shape[i, t, 2] = HEIGHT
        agent_valid[i, t, 0] = True

    return {
        "agent_xyz": agent_xyz,
        "agent_vel": agent_vel,
        "agent_heading": agent_heading,
        "agent_shape": agent_shape,
        "agent_valid": agent_valid,
        "agent_type": agent_type,
        "scenario_id": os.path.splitext(os.path.basename(csv_path))[0],
        "sdc_index": 0,
        "tracks_to_predict": np.arange(n, dtype=np.int32),
    }


def write_split(h5path, csvs, roadgraph):
    """lis_convert.write_split 와 같은 h5 구조."""
    stats = dict(scenarios=0, agents=0, skipped=[])
    with h5py.File(h5path, "w") as hf:
        for csv_path in tqdm(csvs, desc=os.path.basename(h5path)):
            try:
                agents = decode_split_csv(csv_path)
            except Exception as e:
                stats["skipped"].append(f"{os.path.basename(csv_path)}: {e}")
                continue
            ep = hf.create_group(str(stats["scenarios"]))
            for key, value in roadgraph.items():
                ep.create_dataset(key, data=value)
            for key in ["agent_xyz", "agent_vel", "agent_heading", "agent_shape", "agent_valid", "agent_type"]:
                ep.create_dataset(key, data=agents[key])
            ep.attrs["id"] = agents["scenario_id"]
            ep.attrs["sdc_index"] = agents["sdc_index"]
            ep.attrs["tracks_to_predict"] = agents["tracks_to_predict"]
            stats["scenarios"] += 1
            stats["agents"] += len(agents["agent_type"])
    print(f"[done] {h5path}: {stats['scenarios']} scenarios, {stats['agents']} agents, "
          f"{len(stats['skipped'])} skipped")
    return stats


def main():
    p = ArgumentParser()
    p.add_argument("--split-root", required=True, help="Data_Split_s4 (아래 <site>/train, <site>/val)")
    p.add_argument("--site", required=True, choices=["ss", "ny"])
    p.add_argument("--map-path", required=True, help="<SITE>_map.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=0, help="split 당 최대 시나리오 수 (0 = 전체; 시험용)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    roadgraph = decode_lis_map(args.map_path)
    print(f"[map] {args.map_path}: {len(np.unique(roadgraph['roadgraph_id']))} polylines, "
          f"{roadgraph['roadgraph_xyz'].shape[0]} points")

    manifest = dict(site=args.site, split_root=os.path.abspath(args.split_root),
                    map_path=os.path.abspath(args.map_path), splits={})
    for split in SPLITS:
        src = os.path.join(args.split_root, args.site, split)
        csvs = sorted(glob.glob(os.path.join(src, "scenario_*.csv")))
        if not csvs:
            sys.exit(f"no scenario_*.csv under {src}")
        if args.limit:
            csvs = csvs[: args.limit]
        print(f"[data] {args.site}/{split}: {len(csvs)} scenario csv")
        st = write_split(os.path.join(args.out_dir, f"{split}.h5"), csvs, roadgraph)
        dates = sorted({os.path.basename(c).split("_")[1] for c in csvs})
        manifest["splits"][split] = dict(source=os.path.abspath(src), csv=len(csvs), dates=dates, **st)

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
