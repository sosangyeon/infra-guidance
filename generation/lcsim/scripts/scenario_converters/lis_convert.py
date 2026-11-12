"""
lis_convert.py — LIS(K-City, CSV + SS_map.json) -> lcsim 시뮬레이션 protobuf(map.pb / agents.pb).

waymo_convert.py 의 LIS 버전. 엔진(lcsim.engine.Engine)이 data_dir 에서 읽는
map.pb(junction+lanes) 와 agents.pb(agent trips/states) 를 생성한다.

- map: SS_map.json 의 lane_segments -> 하나의 junction 안에 lanes
       (centerline -> center_line, predecessors/successors -> LaneConnection)
- agents: scenario CSV 의 track 별 궤적 -> agent home/schedule/trip/agent_states

사용:
    python lcsim/scripts/scenario_converters/lis_convert.py \
        --scenario-csv /path/Data/LIS/<id>/scenario_<id>.csv \
        --map-json /path/Data/SS_map.json \
        --out-dir /path/sim_data/<id>
"""

import csv
import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np

import sys
# allow running from repo root or anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import json

from lcsim.entity.gen.agent import agent_pb2, trip_pb2
from lcsim.entity.gen.geo import geo_pb2
from lcsim.entity.gen.map import map_pb2

TIME_INTERVAL = 0.1
JUNCTION_ID = 300000000
LANE_TYPE_SURFACE = 2  # center_line_type (consistent with training mapping)
DEFAULT_LANE_SPEED = 16.67  # m/s (~60 km/h); CSV/map 에 제한속도 없음
DEFAULT_LENGTH = 4.5
DEFAULT_WIDTH = 2.0


def default_attr() -> agent_pb2.AgentAttribute:
    attr = agent_pb2.AgentAttribute()
    attr.max_speed = 40
    attr.max_acceleration = 5
    attr.usual_acceleration = 2
    attr.max_braking_acceleration = -5
    attr.usual_braking_acceleration = -2
    return attr


def _polyline_from_pts(pts):
    """[{x,y,z}, ...] -> (map_pb2.Polyline, length)."""
    poly = map_pb2.Polyline()
    length = 0.0
    prev = None
    for p in pts:
        if prev is not None:
            length += np.hypot(p["x"] - prev["x"], p["y"] - prev["y"])
        poly.nodes.append(geo_pb2.XYPosition(x=p["x"], y=p["y"]))
        prev = p
    return poly, length


def build_drivable_rings(segments):
    """lane_segments -> drivable-area 외곽선 목록 (조각마다 하나, CCW 정렬, (x,y) 리스트).

    각 centerline 을 차선 반폭만큼 buffer 해 corridor 를 만들고 union →
    조각마다 외곽선을 CCW 로 정렬. 이 외곽선을 방향 일관된
    road_edge 로 쓰면 NoOffRoad 의 signed-distance 부호(도로안=음수)가
    올바르게 나온다. (per-lane 경계선을 방향없이 넣던 기존 방식의 부호 오류 수정)

    shapely 가 없으면 None 을 반환 → 호출부가 기존(경계선) 방식으로 폴백.
    """
    try:
        from shapely.geometry import LineString, Point, Polygon
        from shapely.ops import unary_union
        from shapely.geometry.polygon import orient
    except ImportError:
        return None
    import statistics

    # 차선 반폭 추정: centerline 점에서 좌/우 경계까지 거리(합)의 중앙값/2
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
    # 서로 떨어진 조각을 모두 쓴다. 예전엔 가장 큰 조각만 남겼는데, NY 지도는 차로가 8 조각이라
    #   가장 큰 조각(433 m²)이 전체(1,854 m²)의 일부뿐 -> 실제 궤적 점 92 % 가 "도로 밖"이 되어
    #   no-offroad guidance 가 정상 주행 차량을 끌어당겼다(생성 차량 감속). SS 는 2 조각(1,920 + 25 m²).
    #   compute_signed_distance_to_drivable_area 가 road_edge(id)마다 다각형으로 보고 홀짝 판정하므로
    #   조각마다 road_edge 하나씩 넣으면 된다.
    parts = list(area.geoms) if area.geom_type == "MultiPolygon" else [area]
    rings = []
    for g in sorted(parts, key=lambda g: g.area, reverse=True):
        g = orient(Polygon(g.exterior), sign=1.0)   # 내부 구멍 제거, CCW (외곽은 반시계 → 내부=왼쪽)
        rings.append(list(g.exterior.coords)[:-1])  # 닫힘 중복점 제거
    return rings


def build_map(map_json, force_perlane: bool = False) -> map_pb2.Map:
    with open(map_json, "r", encoding="utf-8") as f:
        m = json.load(f)
    map_pb = map_pb2.Map()
    junction = map_pb.junctions.add()
    junction.id = JUNCTION_ID
    segments = list(m["lane_segments"].values())
    # drivable-area 외곽선(CCW) — 방향 일관 road_edge 로 off-road 부호 교정. (현재 표준)
    # [deprecated] force_perlane=True 는 옛 per-lane 경계선 방식(닫힌 면 없음 → guidance 부호 깨짐).
    #   per-lane roadgraph로 학습한 모델 비교용으로만 남겨둠. 관련 코드: legacy/ 참고.
    drivable_rings = None if force_perlane else build_drivable_rings(segments)
    for seg in segments:
        _id = int(seg["id"])
        junction.lane_ids.append(_id)
        lane_pb = map_pb.lanes.add()
        lane_pb.id = _id
        lane_pb.center_line_type = LANE_TYPE_SURFACE
        lane_pb.max_speed = DEFAULT_LANE_SPEED
        center, length = _polyline_from_pts(seg["centerline"])
        lane_pb.center_line.CopyFrom(center)
        lane_pb.length = length
        # left / right border lines (for offroad/structure)
        if seg.get("left_lane_boundary"):
            lb, _ = _polyline_from_pts(seg["left_lane_boundary"])
            lane_pb.left_border_line.CopyFrom(lb)
        if seg.get("right_lane_boundary"):
            rb, _ = _polyline_from_pts(seg["right_lane_boundary"])
            lane_pb.right_border_line.CopyFrom(rb)
        # junction road_edges: drivable_ring 사용 시엔 아래에서 한 번만 추가.
        # 폴백(shapely 없음)일 때만 per-lane 경계선을 edge 로 추가(방향무관·부호부정확).
        if drivable_rings is None:
            for bkey in ("left_lane_boundary", "right_lane_boundary"):
                if seg.get(bkey):
                    edge = junction.road_edges.add()
                    edge.type = 1  # ROAD_EDGE_BOUNDARY
                    for p in seg[bkey]:
                        edge.nodes.append(geo_pb2.XYPosition(x=p["x"], y=p["y"]))
        # topology: predecessors / successors are lane ids in LIS
        lane_pb.predecessors.extend(
            [map_pb2.LaneConnection(id=int(p), type=map_pb2.LANE_CONNECTION_TYPE_HEAD)
             for p in seg.get("predecessors", [])]
        )
        lane_pb.successors.extend(
            [map_pb2.LaneConnection(id=int(s), type=map_pb2.LANE_CONNECTION_TYPE_TAIL)
             for s in seg.get("successors", [])]
        )
        lane_pb.parent_id = junction.id
    # drivable-area 외곽선을 조각마다 road_edge 하나로 추가 (CCW → 부호 일관).
    # 엔진(junction.py)이 노드 순서로 dir 을 계산하므로 CCW 순서만 지키면 됨.
    for ring in drivable_rings or []:
        edge = junction.road_edges.add()
        edge.type = 1  # ROAD_EDGE_BOUNDARY (엔진에서 type 15 로 매핑)
        for x, y in ring:
            edge.nodes.append(geo_pb2.XYPosition(x=float(x), y=float(y)))
    return map_pb


def build_agents(scenario_csv) -> agent_pb2.Agents:
    with open(scenario_csv, "r") as f:
        rows = list(csv.DictReader(f))
    by_track = defaultdict(list)
    for r in rows:
        by_track[r["track_id"]].append(r)

    agents_pb = agent_pb2.Agents()
    agents_pb.ads_index = 0  # focal_track_id 가 깨져있어 첫 track 을 ego 로
    for tid, trows in by_track.items():
        trows.sort(key=lambda r: int(r["timestep"]))
        pb = agents_pb.agents.add()
        pb.id = int(tid)
        attr = default_attr()
        attr.type = agent_pb2.AGENT_TYPE_VEHICLE
        attr.length = DEFAULT_LENGTH
        attr.width = DEFAULT_WIDTH
        pb.attribute.CopyFrom(attr)
        # home = 첫 위치
        x0, y0 = float(trows[0]["position_x"]), float(trows[0]["position_y"])
        pb.home.CopyFrom(geo_pb2.Position(xy_position=geo_pb2.XYPosition(x=x0, y=y0)))
        # schedule + trip
        schedule = pb.schedules.add()
        schedule.loop_count = 1
        schedule.departure_time = int(trows[0]["timestep"]) * TIME_INTERVAL
        trip = schedule.trips.add()
        trip.mode = trip_pb2.TRIP_MODE_DRIVE_ONLY
        xe, ye = float(trows[-1]["position_x"]), float(trows[-1]["position_y"])
        trip.end.CopyFrom(geo_pb2.Position(xy_position=geo_pb2.XYPosition(x=xe, y=ye)))
        for r in trows:
            st = trip.agent_states.add()
            st.position.CopyFrom(
                geo_pb2.XYPosition(x=float(r["position_x"]), y=float(r["position_y"]))
            )
            st.velocity_x = float(r["velocity_x"])
            st.velocity_y = float(r["velocity_y"])
            st.heading = float(r["heading"])
    return agents_pb


def main():
    p = ArgumentParser()
    p.add_argument("--scenario-csv", required=True)
    p.add_argument("--map-json", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    map_pb = build_map(args.map_json)
    agents_pb = build_agents(args.scenario_csv)
    with open(os.path.join(args.out_dir, "map.pb"), "wb") as f:
        f.write(map_pb.SerializeToString())
    with open(os.path.join(args.out_dir, "agents.pb"), "wb") as f:
        f.write(agents_pb.SerializeToString())
    print(f"[map] lanes={len(map_pb.lanes)} junction={JUNCTION_ID}")
    print(f"[agents] n={len(agents_pb.agents)} ads_index={agents_pb.ads_index} "
          f"steps={len(agents_pb.agents[0].schedules[0].trips[0].agent_states)}")
    print(f"[out] {args.out_dir}/map.pb, agents.pb")


if __name__ == "__main__":
    main()
