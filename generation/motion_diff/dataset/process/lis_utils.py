"""
LIS (Argoverse2-style, K-City) data decoding utilities.

Mirrors the output schema of `process/utils.py:decode_roadgraph_feature`
so the rest of the training pipeline (womd.py -> map_encoder) works unchanged.

LIS map (`SS_map.json`) is an Argoverse2 vector map:
    {drivable_areas, lane_segments, pedestrian_crossings}
Each lane_segment has: centerline / left_lane_boundary / right_lane_boundary
(lists of {x, y, z}), lane_type, *_lane_mark_type, predecessors/successors, ...

Type mapping (LIS -> WOMD polyline_type code, see process/utils.py):
    lane centerline (lane_type=VEHICLE) -> 2  (TYPE_SURFACE_STREET)
    left/right boundary (mark_type=UNKNOWN) -> 15 (TYPE_ROAD_EDGE_BOUNDARY)
"""

import csv
import json
from collections import defaultdict

import numpy as np

# LIS -> WOMD polyline_type codes
LANE_TYPE_CODE = 2  # TYPE_SURFACE_STREET
BOUNDARY_TYPE_CODE = 15  # TYPE_ROAD_EDGE_BOUNDARY

# whether to include left/right lane boundaries as separate polylines
INCLUDE_BOUNDARY = True
# 경계 표현: True 면 per-lane 경계선 대신 drivable-area 외곽선(CCW) 한 줄을 type-15 로.
# [실험결론] 학습엔 per-lane(False)이 더 나음(big 11.2% vs big_rg 13.3% off-road).
# drivable-ring은 시뮬 off-road지표/guidance 교정용으로만 의미 → 학습용은 False 유지.
# 주의(분포 정합, docs/code_changes.md D항): 여기(학습 h5)는 per-lane, 반면 시뮬 map.pb는
#   ring(lis_convert.build_map 기본). per-lane 학습모델(lis_big)을 ring 맵 시뮬에 쓰면 OOD →
#   guidance(ring) 쓸 땐 ring 학습모델(lis_big_rg)과 짝짓는 게 정합적.
USE_DRIVABLE_RING = False


def build_drivable_rings(segments):
    """lane centerline 들을 차선 반폭만큼 buffer→union→조각별 외곽선(CCW) → [[(x,y), ...], ...].
    양방향 차선을 합치면 중앙선이 면 내부로 흡수돼 외곽 경계만 남는다(방향 일관).
    shapely 없으면 None 반환(호출부가 per-lane 경계로 폴백)."""
    try:
        from shapely.geometry import LineString, Point, Polygon
        from shapely.ops import unary_union
        from shapely.geometry.polygon import orient
    except ImportError:
        return None
    import statistics

    widths = []
    for seg in segments:
        if not (seg.get("left_lane_boundary") and seg.get("right_lane_boundary")):
            continue
        lb = LineString([(p["x"], p["y"]) for p in seg["left_lane_boundary"]])
        rb = LineString([(p["x"], p["y"]) for p in seg["right_lane_boundary"]])
        for p in seg["centerline"]:
            pt = Point(p["x"], p["y"])
            widths.append(lb.distance(pt) + rb.distance(pt))
    if not widths:
        return None
    half = statistics.median(widths) / 2.0
    corridors = [
        LineString([(p["x"], p["y"]) for p in seg["centerline"]]).buffer(half, cap_style=2)
        for seg in segments
        if len(seg["centerline"]) >= 2
    ]
    # 차로 경계(left/right boundary)로 만든 다각형도 합친다. 반폭 buffer 만 쓰면 NY 에서 실제
    #   궤적 점의 10 % 가 영역 밖이었고, 합치면 NY 3.0 %, SS 1.3 % (data/scenarios 5,000 점 표본).
    lane_polys = []
    for seg in segments:
        if seg.get("left_lane_boundary") and seg.get("right_lane_boundary"):
            pg = Polygon([(p["x"], p["y"]) for p in seg["left_lane_boundary"]]
                         + [(p["x"], p["y"]) for p in seg["right_lane_boundary"]][::-1])
            lane_polys.append(pg if pg.is_valid else pg.buffer(0))
    area = unary_union(corridors + lane_polys)
    # 조각을 모두 쓴다 (lis_convert.build_drivable_rings 와 같은 규칙; 이유는 그쪽 주석).
    parts = list(area.geoms) if area.geom_type == "MultiPolygon" else [area]
    return [list(orient(Polygon(g.exterior), sign=1.0).exterior.coords)[:-1]
            for g in sorted(parts, key=lambda g: g.area, reverse=True)]

# --- agent constants ---
NUM_STEPS = 40  # LIS scenario length (0..39), observed 0..9 = history
VEHICLE_TYPE_CODE = 1  # WOMD object_type: 1 = vehicle
# CSV has no bounding-box size -> standard passenger car (length, width, height) [m]
DEFAULT_SHAPE = (4.5, 2.0, 1.5)


def get_polyline_dir(polyline: np.ndarray) -> np.ndarray:
    """Per-point direction unit vectors (identical to process/utils.py)."""
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(
        np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1e9
    )
    return polyline_dir


def _pts_to_xyz(pts) -> np.ndarray:
    """[{x,y,z}, ...] -> (N, 3) float32."""
    return np.array([[p["x"], p["y"], p["z"]] for p in pts], dtype=np.float32)


def decode_lis_map(map_path_or_dict) -> dict:
    """Decode a LIS SS_map.json into the WOMD roadgraph h5 schema.

    Args:
        map_path_or_dict: path to SS_map.json or the already-loaded dict.

    Returns:
        dict with keys:
            roadgraph_xyz  (P, 3) float32
            roadgraph_dir  (P, 3) float32
            roadgraph_type (P,)   int8
            roadgraph_id   (P,)   int32   (running polyline index, unique per polyline)
    """
    if isinstance(map_path_or_dict, (str, bytes)):
        with open(map_path_or_dict, "r", encoding="utf-8") as f:
            m = json.load(f)
    else:
        m = map_path_or_dict

    roadgraph_xyz, roadgraph_dir, roadgraph_type, roadgraph_id = [], [], [], []
    poly_idx = 0  # running id so that polylines split correctly by id

    def add_polyline(pts, type_code):
        nonlocal poly_idx
        if pts is None or len(pts) < 2:
            return
        xyz = _pts_to_xyz(pts)
        dir = get_polyline_dir(xyz)
        n = xyz.shape[0]
        roadgraph_xyz.append(xyz)
        roadgraph_dir.append(dir)
        roadgraph_type.append(np.full(n, type_code, dtype=np.int8))
        roadgraph_id.append(np.full(n, poly_idx, dtype=np.int32))
        poly_idx += 1

    rings = (
        build_drivable_rings(list(m["lane_segments"].values()))
        if USE_DRIVABLE_RING
        else None
    )
    for seg in m["lane_segments"].values():
        add_polyline(seg.get("centerline"), LANE_TYPE_CODE)
        if rings is None and INCLUDE_BOUNDARY:
            # 폴백: drivable_ring 못 만들면 기존 per-lane 경계선 사용
            add_polyline(seg.get("left_lane_boundary"), BOUNDARY_TYPE_CODE)
            add_polyline(seg.get("right_lane_boundary"), BOUNDARY_TYPE_CODE)
    for ring in rings or []:
        # drivable-area 외곽선(조각마다 하나)을 방향 일관 경계(type 15)로 추가
        add_polyline(
            [{"x": float(x), "y": float(y), "z": 0.0} for x, y in ring],
            BOUNDARY_TYPE_CODE,
        )

    roadgraph_xyz = np.concatenate(roadgraph_xyz, axis=0)
    roadgraph_dir = np.concatenate(roadgraph_dir, axis=0)
    roadgraph_type = np.concatenate(roadgraph_type, axis=0)
    roadgraph_id = np.concatenate(roadgraph_id, axis=0)
    assert (
        roadgraph_xyz.shape[0]
        == roadgraph_dir.shape[0]
        == roadgraph_type.shape[0]
        == roadgraph_id.shape[0]
    )
    return {
        "roadgraph_xyz": roadgraph_xyz,
        "roadgraph_dir": roadgraph_dir,
        "roadgraph_type": roadgraph_type,
        "roadgraph_id": roadgraph_id,
    }


def decode_lis_agents(csv_path: str, num_steps: int = NUM_STEPS) -> dict:
    """Decode a LIS scenario CSV into the WOMD agent h5 schema.

    LIS CSV columns: observed, track_id, object_type, object_category, timestep,
    position_x, position_y, heading, velocity_x, velocity_y, scenario_id,
    start_timestamp, end_timestamp, num_timestamps, focal_track_id, city,
    Data_Num, Maneuver

    Every track spans 40 steps fully (verified over all scenarios), all VEHICLE,
    observed=True for steps 0..9 (history) and False for 10..39 (future).

    Returns:
        dict with:
            agent_xyz     (N, T, 3) float32   (z=0; LIS is 2D)
            agent_vel     (N, T, 2) float32
            agent_heading (N, T, 1) float32
            agent_shape   (N, T, 3) float32   (constant DEFAULT_SHAPE; CSV has none)
            agent_valid   (N, T, 1) bool
            agent_type    (N,)      int8       (1 = vehicle)
            scenario_id   str
            sdc_index     int                  (0 = first track; focal_track_id is broken)
            tracks_to_predict  (N,) int        (all agents are object_category=3 / focal)
    """
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))

    # group rows by track, preserving first-seen order (track order in file)
    by_track = defaultdict(list)
    for r in rows:
        by_track[r["track_id"]].append(r)
    track_ids = list(by_track.keys())
    n = len(track_ids)

    agent_xyz = np.zeros((n, num_steps, 3), dtype=np.float32)
    agent_vel = np.zeros((n, num_steps, 2), dtype=np.float32)
    agent_heading = np.zeros((n, num_steps, 1), dtype=np.float32)
    agent_shape = np.tile(
        np.array(DEFAULT_SHAPE, dtype=np.float32), (n, num_steps, 1)
    )
    agent_valid = np.zeros((n, num_steps, 1), dtype=bool)
    agent_type = np.full((n,), VEHICLE_TYPE_CODE, dtype=np.int8)

    for i, tid in enumerate(track_ids):
        for r in by_track[tid]:
            t = int(r["timestep"])
            if t < 0 or t >= num_steps:
                continue
            agent_xyz[i, t, 0] = float(r["position_x"])
            agent_xyz[i, t, 1] = float(r["position_y"])
            # z stays 0 (LIS is 2D / flat)
            agent_vel[i, t, 0] = float(r["velocity_x"])
            agent_vel[i, t, 1] = float(r["velocity_y"])
            agent_heading[i, t, 0] = float(r["heading"])
            agent_valid[i, t, 0] = True

    scenario_id = rows[0]["scenario_id"]
    # focal_track_id in LIS is a broken placeholder ('0'); not used for training.
    # Use index 0 as sdc and treat all (object_category=3) agents as predictable.
    sdc_index = 0
    tracks_to_predict = np.arange(n, dtype=np.int32)

    return {
        "agent_xyz": agent_xyz,
        "agent_vel": agent_vel,
        "agent_heading": agent_heading,
        "agent_shape": agent_shape,
        "agent_valid": agent_valid,
        "agent_type": agent_type,
        "scenario_id": scenario_id,
        "sdc_index": sdc_index,
        "tracks_to_predict": tracks_to_predict,
    }


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "SS_map.json"
    rg = decode_lis_map(path)
    print("=== decode_lis_map output ===")
    for k, v in rg.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")
    print("  num polylines:", len(np.unique(rg["roadgraph_id"])))
    print("  type counts:", dict(zip(*np.unique(rg["roadgraph_type"], return_counts=True))))
    print("  xyz x-range:", rg["roadgraph_xyz"][:, 0].min(), "~", rg["roadgraph_xyz"][:, 0].max())
    print("  dir norm (non-zero rows mean):",
          np.linalg.norm(rg["roadgraph_dir"], axis=1)[1:].mean())
