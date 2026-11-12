"""v7 C2 기동 어휘 빌더 (사이트별 1회). 출력: vocab/c2_lcsim_v6_<site>.pt

어휘 = c1_v6 인수 (M, L, V; L 에 부호 있음) + 구간 라벨 + ProtoNCE 마진. 정의는 v7lib/maneuver.py 에 있다.

  1. 표본  : init 시나리오의 차량 중 40 프레임이 모두 있는 차량의 미래 30 스텝 (world xy)
  2. φ     : canonicalize_future → phi_c2 = (M, L, V)
  3. 정지  : 경로길이 < stop_path (= 0.01 × 비영 경로길이 중앙값, c1_v6.fit_vocab 과 같음) 는 제외
  4. 라벨  : c1_v6.fit_vocab 과 같은 구간 경계 — M, V 는 3구간(q.33/.67), L 은 5구간(±q|L|.60/.90)
            → 45 조합 중 실제로 나온 조합이 클러스터 (K = 쓰인 조합 수)
  5. MLP   : ProtoNCE (3→64→64→16, τ 0.1, 200 epoch) → 클러스터 평균 임베딩 = 프로토타입
  6. 저장  : phi_transform.save_vocab 과 같은 키 (schema 4, 프로브, sens, grad_w=σ/ŝ, in_dim 3)
            min_speed = stop_path (가이던스의 in_dim 3 경로는 이것을 경로길이 임계로 쓴다)
            margin    = GT gap(cos(c*) − 최고 오답 cos) 의 q10 (참고값; 생성은 configs 의 membership.margin 0.05 고정)
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v7lib import maneuver as mv  # noqa: E402

OBS, T = 10, 40


def load_futures(site, split):
    scn = os.path.join(HERE, "..", "data", "scenarios")
    man = pd.read_csv(os.path.join(scn, f"manifest_{site}.csv"), dtype={"date": str})
    man = man[man["split"] == split]
    futs = []
    for date, g in man.groupby("date"):
        df = pd.read_parquet(os.path.join(scn, site, f"{date}.parquet"),
                             columns=["scenario_id", "track_id", "timestep", "position_x", "position_y"])
        df = df[df["scenario_id"].isin(set(g["scenario_id"]))]
        df = df[(df["timestep"] >= 0) & (df["timestep"] < T)]
        cnt = df.groupby(["scenario_id", "track_id"])["timestep"].transform("count")
        df = df[cnt == T].sort_values(["scenario_id", "track_id", "timestep"])
        xy = df[["position_x", "position_y"]].to_numpy(np.float32).reshape(-1, T, 2)
        futs.append(xy[:, OBS:])
    return torch.from_numpy(np.concatenate(futs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=["ss", "ny"])
    ap.add_argument("--split", default="init")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    out = args.out or os.path.join(HERE, "vocab", f"c2_lcsim_v6_{args.site}.pt")

    fut = load_futures(args.site, args.split)
    can = mv.canonicalize_future(fut.double()).float()
    S = mv.smoother(can.shape[1])
    v = mv.factor_values(can, S)
    nz = v["_path"][v["_path"] > 0]
    stop_path = 0.01 * float(nz.median())
    moving = v["_path"] >= stop_path
    print(f"[vocab] {args.site} {args.split}: 차량 {len(fut):,} | 정지(path<{stop_path:.4f} m) {int((~moving).sum()):,} 제외")

    q = lambda x, p: float(x.quantile(p))  # noqa: E731
    vm = {f: v[f][moving] for f in mv.C2_FACTORS}
    a = vm["L"].abs()
    cuts = {"M": torch.tensor([q(vm["M"], .33), q(vm["M"], .67)]),
            "L": torch.tensor([-q(a, .90), -q(a, .60), q(a, .60), q(a, .90)]),
            "V": torch.tensor([q(vm["V"], .33), q(vm["V"], .67)])}
    b = {f: torch.bucketize(vm[f], cuts[f]) for f in mv.C2_FACTORS}
    combo = (b["M"] * 5 + b["L"]) * 3 + b["V"]
    used = torch.unique(combo)
    remap = {int(c): i for i, c in enumerate(used)}
    labels = np.array([remap[int(c)] for c in combo], dtype=np.int64)
    K = len(used)
    print(f"[vocab] 구간 조합 K={K}/45 | 크기 min {np.bincount(labels).min()} 중앙 {int(np.median(np.bincount(labels)))} "
          f"max {np.bincount(labels).max()}")

    rep = mv.phi_c2(can[moving])
    mean, std = rep.mean(0), rep.std(0).clamp(min=1e-6)
    mlp, protos = mv.train_protonce(rep, mean, std, labels, K, epochs=args.epochs)
    with torch.no_grad():
        cos = mv.embed(mlp, rep, mean, std) @ protos.t()
        y = torch.as_tensor(labels)
        pos = cos[torch.arange(len(y)), y]
        neg = cos.clone(); neg[torch.arange(len(y)), y] = -2.0
        g = pos - neg.max(1).values
    print(f"[vocab] GT 자기 클러스터 유지율 {(cos.argmax(1) == y).float().mean():.3f} | gap q10 {g.quantile(0.10):.4f} "
          f"중앙 {g.median():.4f}")
    sens = mv.phi_c2_sensitivity(can[moving][: min(4096, int(moving.sum()))])
    probe_in, probe_emb = mv.make_probe(mlp, rep, mean, std)
    ck = {"mlp": mlp.state_dict(), "mean": mean.float(), "std": std.float(), "dp": None,
          "used": used.numpy(), "K": K, "protos": protos.float(), "means_std": None,
          "labels": torch.as_tensor(labels), "probe_in": probe_in, "probe_emb": probe_emb,
          "meta": {"frame": "lcsim", "dataset": args.site, "source": f"scenarios/{args.split}",
                   "N": int(moving.sum()), "builder": os.path.basename(__file__),
                   "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "cuts": {f: cuts[f].tolist() for f in cuts}},
          "schema": mv.VOCAB_SCHEMA, "min_speed": stop_path, "stop_path": stop_path,
          "n_stationary": int((~moving).sum()), "sens": sens, "grad_w": mv.grad_weights(std, sens).float(),
          "in_dim": 3, "margin": float(g.quantile(0.10)), "margin_q": 0.10}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(ck, out)
    mv.load_vocab(out)   # 왕복 검증
    print(f"[vocab] 저장·검증 OK → {out} | grad_w={[round(float(x), 3) for x in ck['grad_w']]}")


if __name__ == "__main__":
    main()
