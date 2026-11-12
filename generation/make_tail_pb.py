# -*- coding: utf-8 -*-
"""tail 시나리오(parquet) -> LCSim 입력 <out_root>/<sid>/{map.pb, agents.pb}.

시나리오 parquet(data/scenarios/<site>/<date>.parquet)은
LIS CSV 와 열이 같아서, 시나리오마다 CSV 로 덤프한 뒤 lis_convert.build_agents 를 그대로 쓴다.
지도는 사이트 지도 한 벌을 build_map(ring) 으로 한 번 만들어 모든 시나리오 폴더에 쓴다.

사용
  python make_tail_pb.py --site ss --tail_json <round>/tail.json --out_root <round>/pb
"""
import argparse
import json
import os
import sys
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lcsim", "scripts", "scenario_converters"))
from lis_convert import build_agents, build_map  # noqa: E402


ROOT = os.path.abspath(os.path.join(HERE, ".."))   # 저장소 루트
SCN_DIR = os.path.join(ROOT, "data", "scenarios")
MAP_DIR = os.path.join(ROOT, "data", "maps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=["ss", "ny"])
    ap.add_argument("--tail_json", required=True, help='{"<sid>": [tail track_id, ...]}')
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--scenarios", default=SCN_DIR)
    ap.add_argument("--map_json", default=None, help="기본: data/maps/<SITE>_map.json")
    args = ap.parse_args()

    tail = json.load(open(args.tail_json, encoding="utf-8"))
    map_json = args.map_json or os.path.join(MAP_DIR, f"{args.site.upper()}_map.json")
    map_bytes = build_map(map_json).SerializeToString()

    by_date = {}
    for sid in tail:
        by_date.setdefault(sid.split("_")[0], []).append(sid)
    done = 0
    with tempfile.TemporaryDirectory() as tmp:
        for date, sids in sorted(by_date.items()):
            df = pd.read_parquet(os.path.join(args.scenarios, args.site, f"{date}.parquet"))
            for sid in sids:
                g = df[df["scenario_id"] == sid]
                if g.empty:
                    sys.exit(f"{sid}: {date}.parquet 에 없음")
                csv_path = os.path.join(tmp, f"scenario_{sid}.csv")
                g.to_csv(csv_path, index=False)
                od = os.path.join(args.out_root, sid)
                os.makedirs(od, exist_ok=True)
                with open(os.path.join(od, "map.pb"), "wb") as f:
                    f.write(map_bytes)
                with open(os.path.join(od, "agents.pb"), "wb") as f:
                    f.write(build_agents(csv_path).SerializeToString())
                missing = set(tail[sid]) - set(g["track_id"].astype(int))
                if missing:
                    sys.exit(f"{sid}: tail track {sorted(missing)} 이 시나리오에 없음")
                done += 1
    print(f"[pb] {done} 시나리오 -> {args.out_root} (지도 {map_json})")


if __name__ == "__main__":
    main()
