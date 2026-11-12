"""연구실 논문(Table 3-1) 지표로 평가.

  minADE_K, minFDE_K  K 개 후보 중 최소 (predict.py --num_k K 의 metrics.csv)
  MR_FDE (%)          minFDE_K > 2 m 인 샘플 비율 (연구실 val_tutr.py)
  TOP p% (m)          minADE_K 내림차순 상위 int(n * p/100) 개 샘플의 minADE 평균 (연구실 val_tutr.py)
  MR_CP (%)           pred CP_max > 평가 분할 GT CP_max 의 전체 최댓값 인 시나리오 비율
                      (연구실 eval_cp_max.py; pred CP 는 top-1 예측 궤적)

사용 예
  python eval_lab.py --site ss --metrics pred/ss_vanilla_test_k6_metrics.csv --pred_cp vanilla_cp_max
"""

import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCN = os.path.join(HERE, "..", "data", "scenarios")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--scenarios", default=DEFAULT_SCN)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--pred_cp", required=True, help="manifest 의 예측 CP 열 (예: vanilla_cp_max)")
    ap.add_argument("--gt_cp", default="gt_cp_max")
    ap.add_argument("--top_pct", type=float, nargs="+", default=[1, 5])
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    m = pd.read_csv(args.metrics)
    man = pd.read_csv(os.path.join(args.scenarios, f"manifest_{args.site}.csv"))
    man = man[man["scenario_id"].isin(m["scenario_id"].unique())]

    n = len(m)
    srt = np.sort(m["minade"].to_numpy())[::-1]
    gt_max = man[args.gt_cp].max()
    n_exceed = int((man[args.pred_cp] > gt_max).sum())
    row = {"name": args.name or os.path.basename(args.metrics), "samples": n, "scenarios": len(man),
           "minADE": m["minade"].mean(), "minFDE": m["minfde"].mean(),
           "MR_FDE(%)": (m["minfde"] > 2.0).mean() * 100}
    for p in args.top_pct:
        row[f"TOP{p:g}%"] = srt[:int(n * p / 100)].mean()
    row["GT_CP_max"] = gt_max
    row["CP_events"] = n_exceed
    row["MR_CP(%)"] = n_exceed / len(man) * 100
    print(pd.DataFrame([row]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
