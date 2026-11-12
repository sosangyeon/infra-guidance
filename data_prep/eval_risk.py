"""샘플별 오차(predict.py 의 *_metrics.csv) -> 전체 / 위험 부분집합 지표 + 신뢰구간.

위험 부분집합: manifest 의 gt_cp_max > 기준 인 시나리오의 샘플.
사건: 창 간격 1 이라 사건 하나가 겹치는 창 여러 개로 들어간다. 같은 날짜에서
start_frame 이 40프레임 넘게 떨어지면 다른 사건으로 센다 (위험 창끼리만 묶음).
신뢰구간: 군집 부트스트랩. 전체는 날짜, 위험 부분집합은 사건 단위로 다시 뽑는다
(같은 사건의 창들은 독립이 아니므로 창 단위로 뽑으면 구간이 너무 좁아진다).

--compare 를 주면 두 결과(같은 분할, 같은 샘플)의 차이(A - B)를 짝지은 부트스트랩으로 낸다.

사용 예
  python eval_risk.py --site ny --metrics pred/ny_vanilla_test_metrics.csv
  python eval_risk.py --site ny --metrics A_metrics.csv --compare B_metrics.csv --out res.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCN = os.path.join(HERE, "..", "data", "scenarios")
METRICS = ["ade1", "fde1", "minade", "minfde", "mr"]
MR_THRESH = 2.0
EVENT_GAP = 40


def assign_events(man, cp_col, th):
    """위험 창(cp > th)에 사건 번호. 위험이 아닌 창은 -1."""
    ev = pd.Series(-1, index=man.index)
    r = man[man[cp_col] > th].sort_values(["date", "start_frame"])
    if len(r):
        new = r.groupby("date")["start_frame"].diff().isna() | (r.groupby("date")["start_frame"].diff() > EVENT_GAP)
        ev.loc[r.index] = new.cumsum().to_numpy() - 1
    return ev


def cluster_boot(values, clusters, n_boot, rng):
    """values [N, M], clusters [N] -> 군집을 복원추출한 평균의 [n_boot, M]."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums = np.zeros((len(uniq), values.shape[1]))
    np.add.at(sums, inv, values)
    cnt = np.bincount(inv, minlength=len(uniq)).astype(float)
    pick = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    return sums[pick].sum(1) / cnt[pick].sum(1)[:, None]


def summarize(df, cols, cluster_col, n_boot, rng):
    v = df[cols].to_numpy(dtype=float)
    boot = cluster_boot(v, df[cluster_col].to_numpy(), n_boot, rng)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return v.mean(0), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--scenarios", default=DEFAULT_SCN)
    ap.add_argument("--metrics", required=True, help="predict.py *_metrics.csv")
    ap.add_argument("--compare", default=None, help="비교할 다른 *_metrics.csv (차이 = metrics - compare)")
    ap.add_argument("--cp_col", default="gt_cp_max")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    ap.add_argument("--top_pct", type=float, nargs="*", default=[],
                    help="평가 시나리오 중 cp 상위 p%% 부분집합 (예: 1 5). 지정하면 --thresholds 대신 씀")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="결과 CSV")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    man = pd.read_csv(os.path.join(args.scenarios, f"manifest_{args.site}.csv"), dtype={"date": str})
    a = pd.read_csv(args.metrics)
    a["mr"] = (a["minfde"] > MR_THRESH).astype(float)
    if args.compare:
        b = pd.read_csv(args.compare)
        b["mr"] = (b["minfde"] > MR_THRESH).astype(float)
        if not (a[["scenario_id", "track_id"]].to_numpy() == b[["scenario_id", "track_id"]].to_numpy()).all():
            raise ValueError("두 metrics 의 샘플 순서가 다름")
        for m in METRICS:
            a[f"d_{m}"] = a[m] - b[m]
    cols = [f"d_{m}" for m in METRICS] if args.compare else METRICS

    if args.top_pct:
        # 평가 분할에 있는 시나리오끼리 cp 순위. 상위 p% 경계값 이상을 위험으로 본다.
        cp = man.loc[man["scenario_id"].isin(a["scenario_id"].unique()), args.cp_col]
        groups = [("all", 0.0)] + [(f"top{p:g}%", float(np.percentile(cp, 100 - p)) - 1e-12) for p in args.top_pct]
        for name, th in groups[1:]:
            print(f"{name}: {args.cp_col} >= {th + 1e-12:.4f}")
    else:
        groups = [("all", 0.0)] + [(f"cp>{t:g}", t) for t in args.thresholds]
    rows = []
    for name, th in groups:
        if name == "all":
            sub = a.assign(_cl=a["date"].astype(str))
            n_cl, cl_name = sub["_cl"].nunique(), "날짜"
        else:
            ev = assign_events(man, args.cp_col, th)
            ev = pd.Series(ev.to_numpy(), index=man["scenario_id"])
            sub = a.assign(_cl=a["scenario_id"].map(ev))
            sub = sub[sub["_cl"] >= 0]
            n_cl, cl_name = sub["_cl"].nunique(), "사건"
        if sub.empty:
            print(f"{name:<8} 샘플 없음")
            continue
        mean, lo, hi = summarize(sub, cols, "_cl", args.n_boot, rng)
        row = {"subset": name, "samples": len(sub), "scenarios": sub["scenario_id"].nunique(),
               "clusters": n_cl, "cluster_unit": cl_name}
        for c, m_, l_, h_ in zip(cols, mean, lo, hi):
            row[c], row[f"{c}_lo"], row[f"{c}_hi"] = m_, l_, h_
        rows.append(row)
        body = "  ".join(f"{c} {m_:.3f} [{l_:.3f}, {h_:.3f}]" for c, m_, l_, h_ in zip(cols, mean, lo, hi))
        print(f"{name:<8} 샘플 {len(sub):>6,} / {cl_name} {n_cl:>3}  {body}")

    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
