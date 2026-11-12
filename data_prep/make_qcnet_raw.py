"""시나리오 parquet + manifest -> QCNet(AV2 형식) raw 폴더 (분할별).

AV2 시나리오 폴더 구조에서 지도 위치만 다르다.
  <out>/<site>/<split>/raw/<sid>/scenario_<sid>.parquet
  <out>/<site>/<split>/raw/log_map_archive_<site>.json   분할당 지도 한 벌 (av2 가 이름에서 log_id 를 읽음)
AV2 는 지도 json(0.5 MB)을 시나리오마다 두는데, 6만 개면 30 GB 가 넘는다. 그래서 지도를 raw_dir 에
한 벌만 두고, qcnet/datasets/argoverse_v2_dataset*.py 가 시나리오 폴더에 지도가 없으면
raw_dir/log_map_archive_*.json 을 읽는다 (있으면 AV2 방식 그대로).

입력 지도(data/maps/<SITE>_map.json)는 drivable_areas 에 "id" 가 없어 av2 ArgoverseStaticMap.from_json 이
KeyError 를 낸다. 딕셔너리 키를 id 로 넣어 쓴다 (나머지는 그대로).

parquet 열은 make_scenarios.py 출력 그대로다 (heading rad, object_type VEHICLE,
object_category 3, city). acceleration_x/y 는 QCNet 이 읽지 않는 추가 열이다.

split
  init / remain / val / test   manifest 의 split 그대로
  train                        init + remain (Vanilla 용, --no_train 이면 생략)

사용 예
  python make_qcnet_raw.py --site ss
"""

import argparse
import glob
import json
import os
import shutil
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCN = os.path.join(HERE, "..", "data", "scenarios")
DEFAULT_OUT = os.path.join(HERE, "..", "data", "qcnet")
DEFAULT_MAP_DIR = os.path.join(HERE, "..", "data", "maps")

MAP_FILE = {"ss": "SS_map.json", "ny": "NY_map.json"}
SPLITS = ["init", "remain", "val", "test"]


def write_map(src, dst):
    with open(src) as f:
        m = json.load(f)
    for k, da in m["drivable_areas"].items():
        da.setdefault("id", int(k))
    with open(dst, "w") as f:
        json.dump(m, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=sorted(MAP_FILE))
    ap.add_argument("--scenarios", default=DEFAULT_SCN, help="make_scenarios.py 출력 폴더")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--map", default=None, help="지도 json (기본: data/maps/<SITE>_map.json)")
    ap.add_argument("--no_train", action="store_true", help="init+remain 합본 train/ 을 만들지 않음")
    ap.add_argument("--map_only", action="store_true", help="parquet 는 두고 분할 지도만 다시 씀")
    args = ap.parse_args()

    map_src = args.map or os.path.join(DEFAULT_MAP_DIR, MAP_FILE[args.site])
    site_dir = os.path.join(args.out, args.site)
    outs = SPLITS + ([] if args.no_train else ["train"])
    map_name = f"log_map_archive_{args.site}.json"

    if args.map_only:
        for s in outs:
            raw = os.path.join(site_dir, s, "raw")
            for old in glob.glob(os.path.join(raw, "log_map_archive*.json")):
                os.remove(old)
            write_map(map_src, os.path.join(raw, map_name))
        print(f"[{args.site}] 지도 다시 씀: {', '.join(outs)}")
        return

    man = pd.read_csv(os.path.join(args.scenarios, f"manifest_{args.site}.csv"), dtype={"date": str})
    if "split" not in man.columns:
        raise ValueError("manifest 에 split 열이 없음. split_dates.py 를 먼저 실행")
    split_of = dict(zip(man["scenario_id"], man["split"]))

    for s in outs:
        d = os.path.join(site_dir, s)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(os.path.join(d, "raw"))
        write_map(map_src, os.path.join(d, "raw", map_name))

    counts = {s: 0 for s in SPLITS}
    files = sorted(glob.glob(os.path.join(args.scenarios, args.site, "*.parquet")))
    for k, f in enumerate(files):
        df = pd.read_parquet(f)
        for sid, g in df.groupby("scenario_id", sort=True):
            split = split_of[sid]
            g = g.sort_values(["track_id", "timestep"])
            targets = [split] + (["train"] if not args.no_train and split in ("init", "remain") else [])
            for t in targets:
                sdir = os.path.join(site_dir, t, "raw", sid)
                os.makedirs(sdir)
                g.to_parquet(os.path.join(sdir, f"scenario_{sid}.parquet"), index=False)
            counts[split] += 1
        sys.stdout.write(f"\r[{args.site}] {k + 1}/{len(files)} 일")
        sys.stdout.flush()
    print()

    for s in SPLITS:
        print(f"  {s:<6} 시나리오 {counts[s]:,}")
    if not args.no_train:
        print(f"  train  시나리오 {counts['init'] + counts['remain']:,} (init + remain)")


if __name__ == "__main__":
    main()
