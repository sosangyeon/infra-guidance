"""
parquet_split_convert.py — 시나리오 parquet(data/scenarios) -> LCSim(motion_diff) train.h5 / val.h5.

입력이 날짜별 parquet(Argoverse 형식 열:
scenario_id, track_id, timestep, position_x/y, heading, velocity_x/y ...)이고,
분할은 manifest_<site>.csv 의 split 열을 따른다.

- train.h5 = init 분할만 (예측 모델과 같은 학습 데이터), val.h5 = val 분할.
  remain(마이닝 풀)과 test 는 읽지 않는다.
- heading, velocity 는 parquet 값 그대로 (QCNet 입력·CP 계산과 같은 값).
- 차량 크기는 parquet 에 없어 lis_utils.DEFAULT_SHAPE 상수 (decode_lis_agents 와 같음).
- 지도 경계: 기본은 drivable-area 외곽선(ring). 생성 시 lcsim lis_convert.build_map 이 ring 지도를
  만들기 때문에 학습 h5 도 ring 으로 맞춘다 (lis_utils.py 주석: lis_big_rg 와 같은 조합).
  --perlane 이면 차로별 경계선.

사용:
    python motion_diff/dataset/process/parquet_split_convert.py --site ss \
        --scenarios ../../../data/scenarios --map-path ../data/maps/SS_map.json --out-dir ./h5_ss
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
import lis_utils  # noqa: E402
from lis_utils import DEFAULT_SHAPE, VEHICLE_TYPE_CODE, decode_lis_map  # noqa: E402

NUM_STEPS = 40
SPLITS = {"train": "init", "val": "val"}  # h5 이름 -> manifest split. remain, test 는 제외
COLS = ["scenario_id", "track_id", "timestep", "position_x", "position_y",
        "heading", "velocity_x", "velocity_y"]


def decode_group(g):
    """한 시나리오 DataFrame -> decode_lis_agents 와 같은 dict."""
    g = g.sort_values(["track_id", "timestep"])
    tids = list(dict.fromkeys(g["track_id"].tolist()))
    n = len(tids)
    row = {t: i for i, t in enumerate(tids)}
    i = g["track_id"].map(row).to_numpy()
    t = g["timestep"].to_numpy().astype(int)
    agent_xyz = np.zeros((n, NUM_STEPS, 3), dtype=np.float32)
    agent_vel = np.zeros((n, NUM_STEPS, 2), dtype=np.float32)
    agent_heading = np.zeros((n, NUM_STEPS, 1), dtype=np.float32)
    agent_valid = np.zeros((n, NUM_STEPS, 1), dtype=bool)
    agent_xyz[i, t, 0] = g["position_x"].to_numpy()
    agent_xyz[i, t, 1] = g["position_y"].to_numpy()
    agent_vel[i, t, 0] = g["velocity_x"].to_numpy()
    agent_vel[i, t, 1] = g["velocity_y"].to_numpy()
    agent_heading[i, t, 0] = g["heading"].to_numpy()
    agent_valid[i, t, 0] = True
    return {
        "agent_xyz": agent_xyz,
        "agent_vel": agent_vel,
        "agent_heading": agent_heading,
        "agent_shape": np.tile(np.array(DEFAULT_SHAPE, dtype=np.float32), (n, NUM_STEPS, 1)),
        "agent_valid": agent_valid,
        "agent_type": np.full((n,), VEHICLE_TYPE_CODE, dtype=np.int8),
        "sdc_index": 0,
        "tracks_to_predict": np.arange(n, dtype=np.int32),
    }


def main():
    p = ArgumentParser()
    p.add_argument("--site", required=True, choices=["ss", "ny"])
    p.add_argument("--scenarios", required=True, help="data/scenarios (manifest_<site>.csv, <site>/*.parquet)")
    p.add_argument("--map-path", required=True, help="<SITE>_map.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--perlane", action="store_true", help="지도 경계를 차로별 경계선으로 (기본: ring)")
    p.add_argument("--limit", type=int, default=0, help="split 당 최대 시나리오 수 (0 = 전체; 시험용)")
    args = p.parse_args()

    lis_utils.USE_DRIVABLE_RING = not args.perlane
    os.makedirs(args.out_dir, exist_ok=True)
    roadgraph = decode_lis_map(args.map_path)
    print(f"[map] {args.map_path}: {len(np.unique(roadgraph['roadgraph_id']))} polylines, "
          f"{roadgraph['roadgraph_xyz'].shape[0]} points, ring={lis_utils.USE_DRIVABLE_RING}")

    man = pd.read_csv(os.path.join(args.scenarios, f"manifest_{args.site}.csv"), dtype={"date": str})
    files = sorted(glob.glob(os.path.join(args.scenarios, args.site, "*.parquet")))
    manifest = dict(site=args.site, scenarios=os.path.abspath(args.scenarios),
                    map_path=os.path.abspath(args.map_path), ring=lis_utils.USE_DRIVABLE_RING, splits={})
    for h5name, split in SPLITS.items():
        keep = man.loc[man["split"] == split, ["scenario_id", "date"]]
        if args.limit:
            keep = keep.head(args.limit)
        sids, dates = set(keep["scenario_id"]), sorted(keep["date"].unique())
        stats = dict(split=split, scenarios=0, agents=0)
        h5path = os.path.join(args.out_dir, f"{h5name}.h5")
        with h5py.File(h5path, "w") as hf:
            for f in tqdm([f for f in files if os.path.basename(f)[:-8] in dates], desc=h5name):
                df = pd.read_parquet(f, columns=COLS)
                df = df[df["scenario_id"].isin(sids)]
                for sid, g in df.groupby("scenario_id", sort=True):
                    agents = decode_group(g)
                    ep = hf.create_group(str(stats["scenarios"]))
                    for key, value in roadgraph.items():
                        ep.create_dataset(key, data=value)
                    for key in ["agent_xyz", "agent_vel", "agent_heading", "agent_shape",
                                "agent_valid", "agent_type"]:
                        ep.create_dataset(key, data=agents[key])
                    ep.attrs["id"] = sid
                    ep.attrs["sdc_index"] = agents["sdc_index"]
                    ep.attrs["tracks_to_predict"] = agents["tracks_to_predict"]
                    stats["scenarios"] += 1
                    stats["agents"] += len(agents["agent_type"])
        if stats["scenarios"] != len(sids):
            sys.exit(f"{split}: manifest {len(sids)} 시나리오 중 {stats['scenarios']} 개만 찾음")
        print(f"[done] {h5path}: {split} {stats['scenarios']} scenarios, {stats['agents']} agents")
        manifest["splits"][h5name] = dict(dates=dates, **stats)

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
