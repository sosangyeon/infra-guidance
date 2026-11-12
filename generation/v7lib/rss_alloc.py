"""RSS 난이도 기반 생성 개수 재분배 (v7 C3). 원본 구현(rss_alloc.py) 의
rss_lon_lat · allocate 식을 그대로 쓰고, 입력만 우리 형식(tail.json + 시나리오 parquet)으로 바꿨다.

  시나리오 난이도 = 그 안 tail 차량들의 (R_lon + R_lat) 평균 (미래 30 스텝, 원본처럼 world x/y 를 종/횡으로 사용)
  총량 N = 시나리오 수 × base 고정, n_i ∝ 1 + γ·d̃_i (d̃ = min-max 정규화), floor ≤ n_i ≤ cap, largest-remainder.
  최대 난이도 < 0.01 이면 균등 배분 (원본의 uniform_fallback).

사용: python -m v7lib.rss_alloc --site ss --tail_json <round>/tail.json --out <round>/alloc.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

OBS, T = 10, 40
RHO, A_MAX, B_MIN, B_MAX = 0.5, 5.0, 3.0, 8.0


def rss_lon_lat(ep, ev, npz, nvz):
    dx = np.abs(npz[..., 0] - ep[None, :, 0])
    dy = np.abs(npz[..., 1] - ep[None, :, 1])
    vr = np.abs(ev[:, 0])[None]
    vfn = np.abs(nvz[..., 0])
    d_safe = np.maximum(0.0, vr * RHO + 0.5 * A_MAX * RHO ** 2
                        + (vr + RHO * A_MAX) ** 2 / (2 * B_MIN) - vfn ** 2 / (2 * B_MAX))
    lon = np.maximum(0.0, 1.0 - (dx + 1e-3) / (d_safe + 1e-3))
    vlat = np.abs(ev[:, 1][None] - nvz[..., 1])
    dl = vlat * RHO + 0.5 * A_MAX * RHO ** 2 + 0.5
    lat = np.maximum(0.0, 1.0 - (dy + 1e-3) / (dl + 1e-3))
    v = ~np.isnan(npz).any(2)
    lon = np.where(v, lon, 0.0)
    lat = np.where(v, lat, 0.0)
    return float(lon.max(0).mean()), float(lat.max(0).mean())


def to_heading_frame(ep, ev, npz, nvz):
    """tail 차량의 진행방향 좌표계로 회전 (논문 '도로 좌표계': 종방향 = 진행방향).
    heading = atan2(vy, vx); 속도 0.5 m/s 미만 시점은 직전 heading 유지 (make_scenarios.py 와 같은 규칙).
    반환값은 rss_lon_lat 의 입력 형식 그대로 (ep' = 0, ev' = 회전 속도, npz' = 회전 상대위치, nvz' = 회전 속도)."""
    T = ep.shape[0]
    psi = np.arctan2(ev[:, 1], ev[:, 0])
    spd = np.linalg.norm(ev, axis=1)
    for t in range(T):
        if not np.isfinite(spd[t]) or spd[t] < 0.5:
            psi[t] = psi[t - 1] if t > 0 else (psi[np.isfinite(spd) & (spd >= 0.5)][0]
                                               if np.any(np.isfinite(spd) & (spd >= 0.5)) else 0.0)
    c, s = np.cos(psi), np.sin(psi)
    R = np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -2)      # [T,2,2], world -> ego
    rot = lambda v: np.einsum("tij,...tj->...ti", R, v)                  # noqa: E731
    return np.zeros_like(ep), rot(ev), rot(npz - ep[None]), rot(nvz)


def allocate(diff, base=6, gamma=2.0, cap=12, floor=1):
    M = len(diff); N = M * base
    diff = np.asarray(diff, dtype=np.float64)
    fallback = bool(diff.max() < 1e-2)
    dt = np.zeros_like(diff) if fallback else (diff - diff.min()) / (diff.max() - diff.min() + 1e-9)
    w = 1.0 + gamma * dt
    raw = N * w / w.sum()
    n = np.clip(np.floor(raw).astype(int), floor, cap)
    order = np.argsort(-(raw - np.floor(raw)))
    i = 0
    while n.sum() < N and i < 100 * M:
        k = order[i % M]
        if n[k] < cap:
            n[k] += 1
        i += 1
    i = 0
    while n.sum() > N and i < 100 * M:
        k = int(np.argmax(np.where(n > floor, n, -1)))
        if n[k] > floor:
            n[k] -= 1
        i += 1
    return n, fallback, dt


def scene_difficulty(scn_root, site, tail, frame="world"):
    dates = {}
    for sid in tail:
        dates.setdefault(sid.split("_")[0], []).append(sid)
    out = {}
    for d, sids in dates.items():
        df = pd.read_parquet(os.path.join(scn_root, site, f"{d}.parquet"))
        df = df[df["scenario_id"].isin(sids)]
        for sid, g in df.groupby("scenario_id"):
            P, V = {}, {}
            for tid, tr in g.groupby("track_id"):
                ts = tr["timestep"].to_numpy().astype(int)
                pp = np.full((T, 2), np.nan); vv = np.full((T, 2), np.nan)
                pp[ts] = tr[["position_x", "position_y"]].to_numpy()
                vv[ts] = tr[["velocity_x", "velocity_y"]].to_numpy()
                P[int(tid)] = pp; V[int(tid)] = vv
            ros, rls = [], []
            for tid in map(int, tail[sid]):
                o = [a for a in P if a != tid]
                if not o:
                    ros.append(0.0); rls.append(0.0); continue
                args_ = (P[tid][OBS:], V[tid][OBS:], np.array([P[a][OBS:] for a in o]),
                         np.array([V[a][OBS:] for a in o]))
                if frame == "heading":
                    args_ = to_heading_frame(*args_)
                ro, rl = rss_lon_lat(*args_)
                ros.append(ro); rls.append(rl)
            out[sid] = float(np.mean(ros) + np.mean(rls))
    missing = set(tail) - set(out)
    if missing:
        raise SystemExit(f"[rss_alloc] tail 시나리오 {len(missing)} 개를 parquet 에서 못 찾음 (예 {sorted(missing)[:3]})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--scenarios", required=True, help="data/scenarios")
    ap.add_argument("--tail_json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", type=int, default=6)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--cap", type=int, default=12)
    ap.add_argument("--floor", type=int, default=1)
    ap.add_argument("--frame", default="world", choices=["world", "heading"],
                    help="world: 원본 v7 (world x/y 를 종/횡으로). heading: tail 차량 진행방향 좌표계 (논문의 도로 좌표계)")
    ap.add_argument("--crit", default="rss", choices=["rss", "minade"],
                    help="난이도 기준. rss = R_lon+R_lat (원본 v7). minade = 마이닝 예측의 tail 차량 minADE 평균 "
                         "(ours13: 꼬리 장면은 상호작용보다 단일 차량 기동이 어려워 RSS 가 거의 0 이므로 예측 오차로 배분)")
    ap.add_argument("--metrics_csv", default=None, help="--crit minade 일 때 mine_<pool>_metrics.csv")
    a = ap.parse_args()
    tail = json.load(open(a.tail_json, encoding="utf-8"))
    if a.crit == "minade":
        if not a.metrics_csv:
            raise SystemExit("[rss_alloc] --crit minade 에는 --metrics_csv 가 필요하다")
        m = pd.read_csv(a.metrics_csv)
        diff = {}
        for sid, tids in tail.items():
            g = m[(m["scenario_id"] == sid) & (m["track_id"].isin([int(t) for t in tids]))]
            if g.empty:
                raise SystemExit(f"[rss_alloc] metrics 에 tail 차량 없음: {sid} {tids}")
            diff[sid] = float(g["minade"].mean())
    else:
        diff = scene_difficulty(a.scenarios, a.site, tail, a.frame)
    sids = sorted(diff)
    n, fb, dt = allocate([diff[s] for s in sids], a.base, a.gamma, a.cap, a.floor)
    json.dump({s: int(k) for s, k in zip(sids, n)}, open(a.out, "w", encoding="utf-8"), indent=1)
    d = np.array([diff[s] for s in sids])
    meta = {"M": len(sids), "N": int(len(sids) * a.base), "sum": int(n.sum()), "min": int(n.min()), "max": int(n.max()),
            "base": a.base, "gamma": a.gamma, "cap": a.cap, "floor": a.floor, "uniform_fallback": fb,
            "rss_max": float(d.max()), "rss_median": float(np.median(d)), "rss_zero_frac": float((d < 1e-6).mean()),
            "frame": a.frame, "crit": a.crit}
    json.dump(meta, open(os.path.splitext(a.out)[0] + "_meta.json", "w"), indent=1)
    print(f"[rss_alloc] crit={a.crit} frame={a.frame} 시나리오 {len(sids)} | Σn={n.sum()} (= {len(sids)}×{a.base}) 범위[{n.min()},{n.max()}] | "
          f"난이도 max {d.max():.3f} 중앙 {np.median(d):.3f} 0 비율 {(d < 1e-6).mean():.1%} | 균등폴백 {fb} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
