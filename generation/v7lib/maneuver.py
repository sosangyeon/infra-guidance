"""v7 C2 기동 어휘 — 정의·임베딩·로드.

원본 구현의 파일별 대응
  - smoother, canon          : vocab_c1.py
  - factor_values, _gap      : c1_v6.py   (M, L, T, V 인수 중 C2 는 M, L, V)
  - phi_c2                   : c1_v6_adapter.py
  - canonicalize_future, make_mlp, standardize, embed, reweight_grad, train_protonce,
    grad_weights, make_probe, verify, load : phi_transform.py (schema 4, in_dim 3)
수식은 바꾸지 않았다. 원본의 sys.path 조작·git/md5 기록만 뺐다.

φ_C2 = (M, L, V)  (정준 프레임 = 첫 점 원점, 첫 스텝 heading 을 +x)
  M = 프레임당 평균 이동,  L = 부호 있는 횡변위 / 경로길이 (좌 +, 우 −),  V = 마지막 스텝 이동 − 첫 스텝 이동
  (궤적은 3차 다항 사영으로 매끄럽게 한 뒤 계산)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

POLY_DEG = 3
C2_FACTORS = ("M", "L", "V")
VOCAB_SCHEMA = 4
PROBE_N = 16
PROBE_ATOL = 1e-4
INF = 1e9


# ── vocab_c1.py ──
def smoother(n: int, deg: int = POLY_DEG, dtype=torch.float32) -> torch.Tensor:
    t = torch.linspace(-1, 1, n, dtype=torch.float64)
    V = torch.stack([t ** k for k in range(deg + 1)], 1)
    return (V @ torch.linalg.pinv(V)).to(dtype)


def canon(traj: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,bjc->bic", S.to(traj), traj)


# ── c1_v6.py ──
def factor_values(traj: torch.Tensor, S: torch.Tensor) -> dict:
    c = canon(traj, S)
    d = c[:, 1:] - c[:, :-1]
    sp = d.norm(dim=-1)
    path = sp.sum(1)
    return {"_path": path,
            "M": sp.mean(1),
            "L": (c[:, -1, 1] - c[:, 0, 1]) / path.clamp(min=1e-3),
            "V": sp[:, -1] - sp[:, 0]}


def gap(s: torch.Tensor, b: torch.Tensor, cuts: torch.Tensor, scale: float) -> torch.Tensor:
    lo = torch.where(b > 0, cuts.to(s)[(b - 1).clamp(min=0)], torch.full_like(s, -INF))
    hi = torch.where(b < len(cuts), cuts.to(s)[b.clamp(max=len(cuts) - 1)], torch.full_like(s, INF))
    return torch.minimum(s - lo, hi - s) / scale


# ── c1_v6_adapter.py ──
_SM: dict = {}


def phi_c2(traj_canon: torch.Tensor) -> torch.Tensor:
    """정준 프레임 궤적 [N,T,2] → φ [N,3] = (M, L, V). 미분 가능."""
    T = int(traj_canon.shape[1])
    if T not in _SM:
        _SM[T] = smoother(T)
    v = factor_values(traj_canon, _SM[T].to(traj_canon))
    return torch.stack([v[f] for f in C2_FACTORS], dim=1)


def phi_c2_sensitivity(traj: torch.Tensor) -> torch.Tensor:
    x = traj.detach().float().clone().requires_grad_(True)
    out = []
    for i in range(len(C2_FACTORS)):
        if x.grad is not None:
            x.grad = None
        phi_c2(x)[:, i].sum().backward(retain_graph=(i < len(C2_FACTORS) - 1))
        out.append(float(x.grad.flatten(1).norm(dim=1).mean()))
    s = torch.tensor(out, dtype=torch.float32)
    if not torch.isfinite(s).all() or (s <= 0).any():
        raise SystemExit(f"[C2v6] 민감도 비정상 {s.tolist()}")
    return s


# ── phi_transform.py ──
def canonicalize_future(traj: torch.Tensor) -> torch.Tensor:
    """world future [A,T,2] → 정준 프레임: 원점=첫 점, 첫 스텝 heading 을 +x 로 회전."""
    v0 = traj[:, 1] - traj[:, 0]
    th = torch.atan2(v0[:, 1], v0[:, 0] + 1e-9)
    c, s = th.cos(), th.sin()
    R = torch.stack([torch.stack([c, s], -1), torch.stack([-s, c], -1)], -1)
    return torch.einsum("atj,aji->ati", traj - traj[:, :1], R)


def make_mlp(in_dim: int = 3) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 16))


def standardize(rep, mean, std):
    return (rep - mean.to(rep)) / std.to(rep)


def reweight_grad(rep, w):
    if w is None:
        return rep
    w = w.to(rep)
    return rep.detach() + (rep - rep.detach()) * w


def embed(mlp, rep_raw, mean, std, grad_w=None):
    return F.normalize(mlp(standardize(reweight_grad(rep_raw, grad_w), mean, std)), dim=1)


def grad_weights(sigma: torch.Tensor, sens: torch.Tensor) -> torch.Tensor:
    w = sigma.float() / sens.float()
    return w / torch.exp(torch.log(w).mean())


def _class_means(e, y, K):
    protos = torch.stack([e[y == k].mean(0) if (y == k).any() else torch.zeros(e.shape[1], device=e.device)
                          for k in range(K)])
    return F.normalize(protos, dim=1)


def train_protonce(rep_raw, mean, std, labels, K, device="cpu", epochs=200, temp=0.1, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    mlp = make_mlp(int(rep_raw.shape[1])).to(device)
    X = rep_raw.float().to(device)
    y = torch.as_tensor(labels, device=device, dtype=torch.long)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    for ep in range(epochs):
        mlp.train(); opt.zero_grad()
        e = embed(mlp, X, mean.to(device), std.to(device))
        loss = F.cross_entropy((e @ _class_means(e, y, K).t()) / temp, y)
        loss.backward(); opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"[vocab] ProtoNCE ep{ep} loss={loss.item():.4f}", flush=True)
    mlp.eval()
    with torch.no_grad():
        protos = _class_means(embed(mlp, X, mean.to(device), std.to(device)), y, K)
    mlp = mlp.cpu()
    for p in mlp.parameters():
        p.requires_grad_(False)
    return mlp, protos.cpu()


@torch.no_grad()
def make_probe(mlp, rep_raw, mean, std, n=PROBE_N):
    idx = torch.linspace(0, len(rep_raw) - 1, steps=min(n, len(rep_raw))).long()
    probe_in = rep_raw[idx].float().cpu().clone()
    return probe_in, embed(mlp.cpu(), probe_in, mean.cpu(), std.cpu()).cpu().clone()


def load_vocab(path, device="cpu"):
    """어휘 로드 + 프로브 검증 (phi_transform.load_vocab/verify_vocab 와 같은 검사). 반환 (mlp, ck)."""
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    if ck.get("schema") != VOCAB_SCHEMA or "probe_in" not in ck:
        raise SystemExit(f"[vocab] 스키마 불일치/프로브 없음: {path}")
    mlp = make_mlp(int(ck.get("in_dim", 3)))
    mlp.load_state_dict(ck["mlp"]); mlp.eval()
    for p in mlp.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        err = (embed(mlp, ck["probe_in"], ck["mean"], ck["std"]) - ck["probe_emb"]).abs().max().item()
    if err > PROBE_ATOL:
        raise SystemExit(f"[vocab] 프로브 불일치 max|Δ|={err:.2e}: {path}")
    if (grad_weights(ck["std"], ck["sens"]) - ck["grad_w"]).abs().max() > 1e-5:
        raise SystemExit(f"[vocab] grad_w 가 σ/ŝ 와 불일치: {path}")
    K = int(ck["K"])
    if tuple(ck["protos"].shape) != (K, 16):
        raise SystemExit(f"[vocab] K 불일치: K={K} protos={tuple(ck['protos'].shape)}")
    return mlp.to(device), ck
