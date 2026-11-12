"""시나리오별 CP(충돌 확률) 최대값 -> manifest 열.

공식은 연구실 Evaluation/calculate_cp_max.py 와 같다 (그 파일 __main__ 의 설정값).
  - 예측 구간(timestep >= 10)의 같은 시점마다 차량 쌍 (a, b) 의 CP 를 계산
  - a 의 heading 좌표계로 b 의 상대 위치, 두 차량 사각형(4.7 x 1.8 m)이 겹치는 범위를
    가우시안(σ² = σ0² + α (t - 10), σ0 = 0.8, α = 0.05) 로 적분한 확률
  - max(p_ab, p_ba) 를 시점·쌍에 걸쳐 최대 -> cp_max, 그 시점 -> cp_timestep
연구실 코드를 행 단위 파이썬 반복에서 numpy 배열 연산으로 바꿨을 뿐 값은 같다
(--check N 으로 연구실 함수와 직접 비교).

source
  gt    시나리오 parquet 의 실제 궤적과 heading 열       -> gt_cp_max, gt_cp_timestep
  pred  예측 CSV (scenario_id, track_id, timestep, pred_x, pred_y)
        heading 은 연구실처럼 예측 xy 의 np.gradient 로 계산 -> pred_cp_max, pred_cp_timestep
        (연구실 contrastive 의 difficulty 는 Vanilla 예측으로 계산한 pred_cp_max 다)

사용 예
  python compute_cp.py --site ss --source gt
  python compute_cp.py --site ss --source pred --pred_csv <예측.csv> --prefix vanilla
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCN = os.path.join(HERE, "..", "data", "scenarios")

OBS = 10
T = 40
LEN, WID = 4.7, 1.8
SIGMA0, ALPHA = 0.8, 0.05


def cp_pair(xa, ya, ha, xb, yb, hb, sig):
    """[..., P] 배열. a 기준 b 의 CP (연구실 _calc_cp_pair 와 같은 식)."""
    dx, dy = xb - xa, yb - ya
    c, s = np.cos(ha), np.sin(ha)
    x_rel = c * dx + s * dy
    y_rel = -s * dx + c * dy
    yaw = hb - ha
    hx = 0.5 * (LEN * np.abs(np.cos(yaw)) + WID * np.abs(np.sin(yaw)))
    hy = 0.5 * (LEN * np.abs(np.sin(yaw)) + WID * np.abs(np.cos(yaw)))
    xi, xf = -LEN / 2 - hx, LEN / 2 + hx
    yi, yf = -WID / 2 - hy, WID / 2 + hy
    px = ndtr((xf - x_rel) / sig) - ndtr((xi - x_rel) / sig)
    py = ndtr((yf - y_rel) / sig) - ndtr((yi - y_rel) / sig)
    return np.clip(px * py, 0, 1)


def scenario_cp(x, y, h, t_steps):
    """x, y, h: [A, P] (P = 예측 시점 수). t_steps: [P] 절대 timestep. -> (cp_max, t_max)."""
    a = x.shape[0]
    if a < 2:
        return np.nan, -1
    ia, ib = np.triu_indices(a, 1)
    sig = np.sqrt(SIGMA0 ** 2 + ALPHA * (t_steps - OBS))[None, :]
    p_ab = cp_pair(x[ia], y[ia], h[ia], x[ib], y[ib], h[ib], sig)
    p_ba = cp_pair(x[ib], y[ib], h[ib], x[ia], y[ia], h[ia], sig)
    p = np.maximum(p_ab, p_ba)                      # [pairs, P]
    # 연구실: 쌍마다 시점 최대 -> 그 중 최대 쌍. argmax 는 첫 번째 최대 시점.
    pair_max = p.max(1)
    k = int(np.argmax(pair_max))
    return float(pair_max[k]), int(t_steps[int(np.argmax(p[k]))])


def gt_arrays(g):
    g = g[g["timestep"] >= OBS].sort_values(["track_id", "timestep"])
    a = g["track_id"].nunique()
    shape = (a, T - OBS)
    return (g["position_x"].to_numpy().reshape(shape), g["position_y"].to_numpy().reshape(shape),
            g["heading"].to_numpy().reshape(shape))


def pred_arrays(g):
    g = g.sort_values(["track_id", "timestep"])
    a = g["track_id"].nunique()
    shape = (a, -1)
    x = g["pred_x"].to_numpy().reshape(shape)
    y = g["pred_y"].to_numpy().reshape(shape)
    h = np.arctan2(np.gradient(y, axis=1), np.gradient(x, axis=1))
    return x, y, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--scenarios", default=DEFAULT_SCN)
    ap.add_argument("--source", choices=["gt", "pred"], required=True)
    ap.add_argument("--pred_csv", default=None, help="--source pred 일 때 예측 CSV")
    ap.add_argument("--prefix", default=None, help="열 이름 앞부분 (기본: gt / pred)")
    ap.add_argument("--check", type=int, default=0, help="연구실 함수와 N 개 시나리오 비교 후 종료")
    ap.add_argument("--lab_py", default=None, help="--check 용 연구실 calculate_cp_max.py 경로")
    args = ap.parse_args()

    prefix = args.prefix or args.source
    mpath = os.path.join(args.scenarios, f"manifest_{args.site}.csv")
    man = pd.read_csv(mpath, dtype={"date": str})

    rows = []
    if args.source == "gt":
        files = sorted(glob.glob(os.path.join(args.scenarios, args.site, "*.parquet")))
        cols = ["scenario_id", "track_id", "timestep", "position_x", "position_y", "heading"]
        for k, f in enumerate(files):
            df = pd.read_parquet(f, columns=cols)
            for sid, g in df.groupby("scenario_id", sort=False):
                x, y, h = gt_arrays(g)
                rows.append((sid,) + scenario_cp(x, y, h, np.arange(OBS, T)))
            sys.stdout.write(f"\r[{args.site}] {k + 1}/{len(files)} 일")
            sys.stdout.flush()
            if args.check and len(rows) >= args.check:
                break
        print()
    else:
        pred = pd.read_csv(args.pred_csv)
        for sid, g in pred.groupby("scenario_id", sort=False):
            x, y, h = pred_arrays(g)
            ts = np.sort(g["timestep"].unique())
            rows.append((sid,) + scenario_cp(x, y, h, ts))

    res = pd.DataFrame(rows, columns=["scenario_id", f"{prefix}_cp_max", f"{prefix}_cp_timestep"])

    if args.check:
        check_against_lab(args, res.head(args.check))
        return

    man = man.drop(columns=[c for c in res.columns if c != "scenario_id" and c in man.columns])
    man = man.merge(res, on="scenario_id", how="left")
    man.to_csv(mpath, index=False)
    v = man[f"{prefix}_cp_max"]
    print(f"{args.site}: {prefix}_cp_max {v.notna().sum():,}/{len(man):,} 시나리오, "
          f"평균 {v.mean():.4f}, 중앙 {v.median():.4f}, >0.5 비율 {(v > 0.5).mean():.3f} -> {mpath}")


def check_against_lab(args, res):
    """연구실 compute_cp_core 로 같은 시나리오를 계산해 비교 (gt 만)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("lab_cp", args.lab_py)
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)
    files = sorted(glob.glob(os.path.join(args.scenarios, args.site, "*.parquet")))
    df = pd.concat(pd.read_parquet(f) for f in files[:3])
    worst = 0.0
    for _, r in res.iterrows():
        g = df[df["scenario_id"] == r["scenario_id"]].rename(columns={"position_x": "x", "position_y": "y"})
        g = g[g["timestep"] >= OBS]
        cp, t = lab.compute_cp_core(g, LEN, WID, SIGMA0, SIGMA0, ALPHA, ALPHA, OBS, heading_col="heading")
        worst = max(worst, abs(cp - r.iloc[1]))
        assert t == r.iloc[2] or abs(cp - r.iloc[1]) < 1e-9, (r["scenario_id"], cp, t, r.iloc[1], r.iloc[2])
    print(f"연구실 함수 대조 {len(res)} 시나리오: 최대 차이 {worst:.2e}")


if __name__ == "__main__":
    main()
