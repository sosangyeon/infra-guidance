"""
lis_convert.py — LIS(K-City, Argoverse2-style) 데이터셋 -> WOMD h5 변환 스크립트.

`process.py`(Waymo TFRecord -> h5)의 LIS 버전. 학습 파이프라인(womd.py)이 읽는
train.h5 / val.h5 를 생성한다. 각 시나리오 = h5 group, 안에 roadgraph_* + agent_* 데이터셋과
id/sdc_index/tracks_to_predict attrs 를 담는다.

LIS map(SS_map.json)은 1개를 모든 시나리오가 공유하므로, 디코딩 1회 후 각 group에 복사한다.

사용 예:
    python motion_diff/dataset/process/lis_convert.py \
        --data-dir /path/to/Data/LIS \
        --map-path /path/to/Data/SS_map.json \
        --out-dir /path/to/output \
        --val-ratio 0.1

출력: <out-dir>/train.h5, <out-dir>/val.h5
"""

import glob
import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lis_utils import decode_lis_agents, decode_lis_map


def write_split(h5path, scenario_csvs, roadgraph, num_steps):
    """Write one h5 file (train or val). Shares the same roadgraph across scenarios."""
    with h5py.File(h5path, "w") as hf:
        idx = 0
        for csv_path in tqdm(scenario_csvs, desc=os.path.basename(h5path)):
            try:
                agents = decode_lis_agents(csv_path, num_steps=num_steps)
            except Exception as e:
                print(f"[skip] {csv_path}: {e}")
                continue
            ep = hf.create_group(str(idx))
            # roadgraph (shared, copied per scenario to match womd.py read pattern)
            for key, value in roadgraph.items():
                ep.create_dataset(key, data=value)
            # agent
            for key in [
                "agent_xyz",
                "agent_vel",
                "agent_heading",
                "agent_shape",
                "agent_valid",
                "agent_type",
            ]:
                ep.create_dataset(key, data=agents[key])
            # global attrs (womd.py reads: id, sdc_index, tracks_to_predict)
            ep.attrs["id"] = agents["scenario_id"]
            ep.attrs["sdc_index"] = agents["sdc_index"]
            ep.attrs["tracks_to_predict"] = agents["tracks_to_predict"]
            idx += 1
    print(f"[done] {h5path}: {idx} scenarios")
    return idx


def main():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="LIS dir (scenario_*.csv 하위)")
    parser.add_argument("--map-path", required=True, help="SS_map.json 경로")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) decode shared map once
    roadgraph = decode_lis_map(args.map_path)
    print(
        f"[map] {len(np.unique(roadgraph['roadgraph_id']))} polylines, "
        f"{roadgraph['roadgraph_xyz'].shape[0]} points"
    )

    # 2) collect scenario csvs
    csvs = sorted(glob.glob(os.path.join(args.data_dir, "**", "scenario_*.csv"), recursive=True))
    if not csvs:
        csvs = sorted(glob.glob(os.path.join(args.data_dir, "scenario_*.csv")))
    print(f"[data] {len(csvs)} scenarios found")

    # 3) deterministic train/val split
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(csvs))
    n_val = int(len(csvs) * args.val_ratio)
    val_idx = set(perm[:n_val].tolist())
    train_csvs = [c for i, c in enumerate(csvs) if i not in val_idx]
    val_csvs = [c for i, c in enumerate(csvs) if i in val_idx]
    print(f"[split] train={len(train_csvs)} val={len(val_csvs)}")

    # 4) write
    write_split(os.path.join(args.out_dir, "train.h5"), train_csvs, roadgraph, args.num_steps)
    write_split(os.path.join(args.out_dir, "val.h5"), val_csvs, roadgraph, args.num_steps)


if __name__ == "__main__":
    main()
