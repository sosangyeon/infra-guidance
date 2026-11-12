"""생성 장면의 CP 난이도 -> <gen pkl 이름>_cp.csv (scenario_id, cp).

init 의 vanilla_cp_max 와 같은 정의로 만든다:
  Vanilla(init 만 학습, best.pth) 의 top-1 예측 궤적(predict.py 출력 _pred.csv)으로,
  data_prep/compute_cp.py 의 scenario_cp (heading = 예측 xy 의 np.gradient) 를 계산.
차량이 1대뿐인 생성 장면은 쌍이 없어 0 (연구실 get_data_pkl_cp.py: CP 없는 시나리오 = 0).

사용
  python gen_cp.py --pred_csv <round>/gen_r1_vanilla_pred.csv --out <round>/gen_r1_cp.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _bisci_root(start):
    p = start
    while not os.path.isdir(os.path.join(p, "data_prep")):
        if os.path.dirname(p) == p:
            raise FileNotFoundError(f"{start} 위에 data_prep 이 없음")
        p = os.path.dirname(p)
    return p


sys.path.insert(0, os.path.join(_bisci_root(HERE), "data_prep"))
from compute_cp import pred_arrays, scenario_cp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--meta_csv", required=True, help="생성 pkl 의 _meta.csv (모든 장면이 표에 들어가게)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pred = pd.read_csv(args.pred_csv)
    rows = []
    for sid, g in pred.groupby("scenario_id", sort=False):
        x, y, h = pred_arrays(g)
        cp, t = scenario_cp(x, y, h, np.sort(g["timestep"].unique()))
        rows.append((sid, cp, t))
    res = pd.DataFrame(rows, columns=["scenario_id", "cp", "cp_timestep"])
    scenes = pd.read_csv(args.meta_csv)[["scenario_id"]].drop_duplicates()
    res = scenes.merge(res, on="scenario_id", how="left")
    n_single = int(res["cp"].isna().sum())
    res["cp"] = res["cp"].fillna(0.0)
    res.to_csv(args.out, index=False)
    print(f"[gen cp] {len(res)} 장면 (차량 1대 -> 0: {n_single}) | 평균 {res['cp'].mean():.4f}, "
          f">0.33 비율 {(res['cp'] > 0.33).mean():.3f} -> {args.out}")


if __name__ == "__main__":
    main()
