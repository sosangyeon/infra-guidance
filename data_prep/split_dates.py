"""manifest_<site>.csv 에 split 열(init / remain / val / test)을 붙인다.

날짜 단위로 나눈다. 창 간격이 40프레임보다 작으면 이웃한 창이 겹치므로, 같은 날의
시나리오가 서로 다른 분할에 들어가면 거의 같은 장면이 train 과 test 에 함께 들어간다.

날짜를 시간 순서대로 놓고, 시나리오 수 누적 비율이 init 0.2 / remain 0.2 / val 0.1 /
test 0.5 가 되도록 채운다 (연구실 train 0.4 / val 0.1 / test 0.5 에서 train 을 반으로).
각 날짜는 그 날 시나리오 누적 구간의 중간점이 들어가는 분할로 간다.

출력
  manifest_<site>.csv          split 열 추가 (덮어씀)
  split_dates_<site>.csv       date, split, n_scenarios, n_agents

사용 예
  python split_dates.py --site ss
  python split_dates.py --site ny --ratios 0.2 0.2 0.1 0.5
"""

import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(HERE, "..", "data", "scenarios")

SPLITS = ["init", "remain", "val", "test"]


def assign_dates(per_day, ratios, weight):
    w = per_day[weight].to_numpy(dtype=float)
    mid = (np.cumsum(w) - w / 2) / w.sum()
    edges = np.cumsum(ratios)[:-1]
    idx = np.searchsorted(edges, mid, side="right")
    return [SPLITS[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--dir", default=DEFAULT_DIR, help="manifest 가 있는 폴더")
    ap.add_argument("--ratios", type=float, nargs=4, default=[0.2, 0.2, 0.1, 0.5],
                    metavar=("INIT", "REMAIN", "VAL", "TEST"))
    ap.add_argument("--weight", choices=["n_scenarios", "n_agents"], default="n_scenarios",
                    help="비율을 맞출 기준 (시나리오 수 / 차량-샘플 수)")
    args = ap.parse_args()

    ratios = np.asarray(args.ratios, dtype=float)
    ratios = ratios / ratios.sum()

    mpath = os.path.join(args.dir, f"manifest_{args.site}.csv")
    man = pd.read_csv(mpath, dtype={"date": str})

    per_day = man.groupby("date").agg(
        n_scenarios=("scenario_id", "size"),
        n_agents=("n_agents", "sum"),
    ).reset_index().sort_values("date").reset_index(drop=True)
    per_day["split"] = assign_dates(per_day, ratios, args.weight)

    man = man.drop(columns=["split"], errors="ignore").merge(per_day[["date", "split"]], on="date")
    man.to_csv(mpath, index=False)
    per_day[["date", "split", "n_scenarios", "n_agents"]].to_csv(
        os.path.join(args.dir, f"split_dates_{args.site}.csv"), index=False)

    tot_s, tot_a = per_day["n_scenarios"].sum(), per_day["n_agents"].sum()
    print(f"{args.site}: 기준 {args.weight}, 목표 " + " / ".join(f"{s} {r:.2f}" for s, r in zip(SPLITS, ratios)))
    print(f"{'split':<7} {'날짜':>23} {'일수':>4} {'시나리오':>10} {'비율':>6} {'차량-샘플':>11} {'비율':>6}")
    for s in SPLITS:
        d = per_day[per_day["split"] == s]
        if d.empty:
            print(f"{s:<7} (없음)")
            continue
        print(f"{s:<7} {d['date'].iloc[0]}~{d['date'].iloc[-1]:>14} {len(d):>4} "
              f"{d['n_scenarios'].sum():>10,} {d['n_scenarios'].sum() / tot_s:>6.3f} "
              f"{d['n_agents'].sum():>11,} {d['n_agents'].sum() / tot_a:>6.3f}")


if __name__ == "__main__":
    main()
