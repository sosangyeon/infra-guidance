"""
map_convert.py — LIS(SS_map.json, Argoverse2 vector map) -> WOMD roadgraph 스키마 변환 스크립트.

학습 파이프라인(womd.py -> map_encoder)이 기대하는 roadgraph_* 4개 필드를 생성한다.
변환 로직은 `lis_utils.decode_lis_map`을 사용하며, 이 스크립트는 CLI/저장/시각화를 담당한다.

사용 예:
    # 요약만 출력
    python motion_diff/dataset/process/map_convert.py --map-path /path/to/SS_map.json

    # npz로 저장 + 시각화 png 저장
    python motion_diff/dataset/process/map_convert.py \
        --map-path /path/to/SS_map.json --out roadgraph.npz --viz roadgraph.png

출력 npz 키: roadgraph_xyz (P,3), roadgraph_dir (P,3), roadgraph_type (P,), roadgraph_id (P,)
타입 코드: lane=2 (SURFACE_STREET), boundary=15 (ROAD_EDGE_BOUNDARY)
"""

import os
import sys
from argparse import ArgumentParser

import numpy as np

# 같은 디렉토리의 lis_utils 사용 (torch 의존성 없음)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lis_utils import decode_lis_map


def visualize(rg: dict, out_path: str):
    """변환된 roadgraph를 시각화해 형상/방향을 검증."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz, typ, ids = rg["roadgraph_xyz"], rg["roadgraph_type"], rg["roadgraph_id"]
    fig, ax = plt.subplots(figsize=(9, 9))
    for pid in np.unique(ids):
        m = ids == pid
        t = typ[m][0]
        is_lane = t == 2
        ax.plot(
            xyz[m, 0],
            xyz[m, 1],
            c="blue" if is_lane else "gray",
            lw=1.5 if is_lane else 0.5,
            ls="-" if is_lane else "--",
        )
    lane = typ == 2
    if lane.sum() > 0:
        idx = np.arange(0, lane.sum(), max(1, lane.sum() // 30))
        ax.quiver(
            xyz[lane][idx, 0],
            xyz[lane][idx, 1],
            rg["roadgraph_dir"][lane][idx, 0],
            rg["roadgraph_dir"][lane][idx, 1],
            color="red",
            scale=30,
            width=0.003,
        )
    ax.set_aspect("equal")
    ax.set_title("Converted roadgraph (blue=lane t=2, gray=boundary t=15, red=dir)")
    plt.savefig(out_path, dpi=90, bbox_inches="tight")
    print(f"[viz] saved {out_path}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--map-path", required=True, help="SS_map.json 경로")
    parser.add_argument("--out", default=None, help="변환 결과 저장 경로(.npz)")
    parser.add_argument("--viz", default=None, help="시각화 저장 경로(.png)")
    args = parser.parse_args()

    rg = decode_lis_map(args.map_path)

    # 요약
    print("=== map_convert: LIS -> roadgraph ===")
    for k, v in rg.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")
    print("  num polylines:", len(np.unique(rg["roadgraph_id"])))
    codes, counts = np.unique(rg["roadgraph_type"], return_counts=True)
    print("  type counts:", {int(c): int(n) for c, n in zip(codes, counts)})

    if args.out:
        np.savez(args.out, **rg)
        print(f"[out] saved {args.out}")
    if args.viz:
        visualize(rg, args.viz)


if __name__ == "__main__":
    main()
