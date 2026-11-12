"""run_lcsim_site.py — 사이트(SS/NY)별 LCSim(motion_diff) 학습 데이터 준비 + 학습.

단계 (각 단계 산출물이 이미 있으면 건너뜀; --force 로 다시):
  1. convert : data/scenarios 의 init -> h5_<site>/train.h5, val -> val.h5  (remain, test 미사용)
  2. velmean : train.h5 의 미래 step 변위 평균 -> h5_<site>/vel_mean.json
  3. pca     : train.h5 미래 궤적 PCA(10성분, 60차원) -> h5_<site>/pca.pkl
  4. config  : configs/lis_config.yml 기반 -> configs/lis_<site>_config.yml
  5. train   : train_md.py -> runs/lcsim_<site>/logs/lis_<site>_hist10_fut30/

실행 위치: 이 파일이 있는 generation 폴더 (train_md.py 가 상대경로 import 를 쓴다).
설정 파일의 data_dir, pca_path 도 generation 기준 상대경로로 쓴다.

    python run_lcsim_site.py --site ss
    python run_lcsim_site.py --site ny
    python run_lcsim_site.py --site ss ... --steps convert,velmean,pca,config   # 학습 빼고
"""

import argparse
import json
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


ROOT = os.path.abspath(os.path.join(HERE, ".."))   # 저장소 루트
SCN_DIR = os.path.join(ROOT, "data", "scenarios")
MAP_DIR = os.path.join(ROOT, "data", "maps")
STEPS = ["convert", "velmean", "pca", "config", "train"]


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=["ss", "ny"])
    ap.add_argument("--scenarios", default=SCN_DIR, help="make_scenarios.py 출력 (manifest_<site>.csv)")
    ap.add_argument("--map-dir", default=MAP_DIR)
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--epochs", type=int, default=None, help="기본: lis_config.yml 값(200)")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--val-interval", type=int, default=None, help="기본: lis_config.yml 값(1)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    steps = args.steps.split(",")
    py = sys.executable
    site, SITE = args.site, args.site.upper()
    h5dir = os.path.join(HERE, f"h5_{site}")
    map_path = os.path.join(args.map_dir, f"{SITE}_map.json")
    cfg_path = os.path.join(HERE, "motion_diff", "configs", f"lis_{site}_config.yml")
    vel_path = os.path.join(h5dir, "vel_mean.json")
    pca_path = os.path.join(h5dir, "pca.pkl")

    if "convert" in steps and (args.force or not os.path.exists(os.path.join(h5dir, "val.h5"))):
        run([py, "motion_diff/dataset/process/parquet_split_convert.py", "--scenarios", args.scenarios,
             "--site", site, "--map-path", map_path, "--out-dir", h5dir])

    if "velmean" in steps and (args.force or not os.path.exists(vel_path)):
        sys.path.insert(0, os.path.join(HERE, "motion_diff", "dataset", "process"))
        from lis_vel_mean import measure_vel_mean
        r = measure_vel_mean(os.path.join(h5dir, "train.h5"), 10)
        json.dump(r, open(vel_path, "w"), indent=1)
        print(f"[velmean] {site}: {r}")

    if "pca" in steps and (args.force or not os.path.exists(pca_path)):
        run([py, "motion_diff/scripts/lis_pca_gen.py", "--data-dir", h5dir, "--num-future-steps", "30",
             "--num-historical-steps", "10", "--num-components", "10", "--out", pca_path])

    if "config" in steps:
        c = yaml.safe_load(open(os.path.join(HERE, "motion_diff", "configs", "lis_config.yml"), encoding="utf-8"))
        c["title"] = f"motion-diff@lis_{site}"
        c["version"] = f"lis_{site}_hist10_fut30"
        c["dataset"]["data_dir"] = f"h5_{site}"            # generation 기준 상대경로
        c["model"]["pca_path"] = f"h5_{site}/pca.pkl"
        c["model"]["vel_mean"] = round(json.load(open(vel_path))["vel_mean"], 4)
        if args.epochs is not None:
            c["trainer"]["max_epochs"] = args.epochs
        if args.val_interval is not None:
            c["trainer"]["val_interval"] = args.val_interval
        if args.num_workers is not None:
            c["dataset"]["num_workers"] = args.num_workers
        yaml.safe_dump(c, open(cfg_path, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
        print(f"[config] {cfg_path}: vel_mean={c['model']['vel_mean']} epochs={c['trainer']['max_epochs']}")

    if "train" in steps:
        run([py, "train_md.py", "--config", cfg_path, "--save", os.path.join("runs", f"lcsim_{site}")])


if __name__ == "__main__":
    main()
