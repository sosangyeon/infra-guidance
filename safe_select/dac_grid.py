# -*- coding: utf-8 -*-
"""DAC (주행가능영역 준수) 격자 엔진.

PDM 원본 판정 규칙을 그대로 쓰되, 폴리곤 점-포함 판정을 격자 조회로 바꾼다.
  원본  pdm_scorer._calculate_ego_area
        batch_nondrivable_area_mask = (corners_in_polygon.sum(-2) > 0).sum(-1) < 4
        -> 차량 박스 네 모서리 중 하나라도 어떤 폴리곤에도 없으면 그 프레임 위반
        _calculate_drivable_area_compliance
        -> 한 프레임이라도 위반이면 그 궤적 점수 0

격자로 바꾼 이유
  폴리곤 점-포함 비용 = 점 수 x 폴리곤 꼭짓점 수.
  버퍼링된 주행가능영역은 꼭짓점이 2,000개가 넘어 궤적당 3,200점 판정이 병목이었다.
  0.10 m 칸에 안/밖 1비트를 한 번 구워 두면 조회가 배열 인덱싱 한 번으로 끝난다.
  (폴리곤 직접 판정과의 불일치: 점 0.043~0.107%, 모드 0.217~0.617%)

여유 DA_MG = 0.75 m = 차량 폭 1.50 m 의 절반
  지도 경계의 절반 이상이 보간값이고 교차로 안에는 차선 도색 자체가 없어
  여유 0 이면 정상 주행 GT 조차 8.4% 가 도로이탈로 찍힌다.
  여유는 차량 제원에서 정한다 — 차체가 경계를 넘어야 이탈로 본다는 뜻이다.
  (이전 판은 GT 오탐률이 0.78% 가 되도록 0.80 m 를 골랐으나, 데이터셋을 보고
   고른 값이라 폐기했다. 차이는 0.05 m 다.)
  지도 생성 시 이미 바깥 마진 0.35 m 가 들어가 있으므로
  차로 폴리곤 합집합 기준 실효 여유는 약 1.10 m 다.

정지 임계 STAT_TH = 2.40 m = 차량 길이 4.80 m 의 절반
  관측 10 프레임 이동이 차체 길이의 절반에도 못 미치면 진행방향이 정의되지 않는다.
"""
import json
import os

import numpy as np
import shapely.geometry as G
from matplotlib.path import Path

DA_MG = 0.75          # 주행가능영역 여유 [m] = 차량 폭 / 2
DA_CELL = 0.10        # 격자 칸 크기 [m]
PAD = 3.0             # 격자 여백 [m]
SIMPLIFY = 0.02       # 폴리곤 단순화 허용오차 [m]
RJ = dict(join_style=1, cap_style=1)

VEH_W, VEH_L = 1.50, 4.80     # 차량 폭 x 길이 [m]
STAT_TH = 2.40                # 정지 판정 [m] = 차량 길이 / 2. 이하면 판정 생략
DIR_SPAN = 2.0                # 진행방향을 재는 전방 거리 [m]

MAPDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "maps")


def _paths(g):
    gs = [g] if g.geom_type == "Polygon" else list(g.geoms)
    return [(Path(np.array(q.exterior.coords)),
             [Path(np.array(h.coords)) for h in q.interiors]) for q in gs]


def _inside(P, Q):
    m = np.zeros(len(Q), bool)
    for ext, hol in P:
        a = ext.contains_points(Q)
        for h in hol:
            a &= ~h.contains_points(Q)
        m |= a
    return m


def traj_dirs(P):
    """궤적 각 점의 진행방향 단위벡터. 전방 DIR_SPAN 만큼의 현을 쓴다.

    궤적 끝에서는 전방 참조가 소진돼 방향이 0 이 되므로 직전 유효 방향으로 채운다.
    """
    seg = np.linalg.norm(np.diff(P, axis=1), axis=-1)
    s = np.concatenate([np.zeros((len(P), 1)), np.cumsum(seg, axis=1)], 1)
    hi = np.clip((s[:, None, :] <= (s + DIR_SPAN)[:, :, None]).sum(2) - 1,
                 0, P.shape[1] - 1)
    d = np.take_along_axis(P, hi[..., None], 1) - P
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    u = d / np.where(n < 1e-9, 1e-9, n)
    bad = n[..., 0] < 1e-9
    for t in range(1, P.shape[1]):
        u[bad[:, t], t] = u[bad[:, t], t - 1]
    for t in range(P.shape[1] - 2, -1, -1):
        m = bad[:, t] & (np.linalg.norm(u[:, t], axis=-1) < 1e-9)
        u[m, t] = u[m, t + 1]
    return u


def boxes(P):
    """궤적 점을 중심으로 하는 차량 박스 네 모서리 (n, T, 4, 2).

    순서는 PDM BBCoordsIndex 와 같은 FRONT_LEFT, REAR_LEFT, REAR_RIGHT, FRONT_RIGHT.
    """
    u = traj_dirs(P).reshape(-1, 2)
    nrm = np.stack([-u[:, 1], u[:, 0]], 1)
    c = P.reshape(-1, 2)
    hw, hl = VEH_W / 2, VEH_L / 2
    B = np.stack([c + hl * u + hw * nrm, c + hl * u - hw * nrm,
                  c - hl * u - hw * nrm, c - hl * u + hw * nrm], 1)
    return B.reshape(P.shape[0], P.shape[1], 4, 2)


class DrivableGrid:
    """지점 하나의 주행가능영역 격자. 만들 때 한 번 굽고 이후 조회만 한다."""

    def __init__(self, site, mapdir=None, margin=DA_MG, verbose=False):
        path = os.path.join(mapdir or MAPDIR, "%s_map.json" % site.upper())
        J = json.load(open(path, encoding="utf-8"))
        ps = []
        for v in J["drivable_areas"].values():
            e = np.array([[p["x"], p["y"]] for p in v["area_boundary"]])
            h = [np.array([[p["x"], p["y"]] for p in x]) for x in v.get("holes", [])]
            g = G.Polygon(e, h)
            ps.append(g if g.is_valid else g.buffer(0))
        u = ps[0] if len(ps) == 1 else G.MultiPolygon(ps)
        da = u.buffer(margin, **RJ).simplify(SIMPLIFY)
        b = da.bounds
        self.x0, self.y0 = b[0] - PAD, b[1] - PAD
        self.nx = int((b[2] - b[0] + 2 * PAD) / DA_CELL) + 1
        self.ny = int((b[3] - b[1] + 2 * PAD) / DA_CELL) + 1
        gx = self.x0 + (np.arange(self.nx) + .5) * DA_CELL
        gy = self.y0 + (np.arange(self.ny) + .5) * DA_CELL
        GX, GY = np.meshgrid(gx, gy)
        self.G = _inside(_paths(da), np.stack([GX.ravel(), GY.ravel()], 1)) \
            .reshape(self.ny, self.nx)
        self.site, self.margin = site.upper(), margin
        if verbose:
            print("[%s] 격자 %d x %d · %.1f MB" %
                  (self.site, self.nx, self.ny, self.G.nbytes / 1e6))

    def inside(self, Q):
        """점 (m, 2) -> 주행가능영역 안이면 True. 격자 밖은 False(=위반)."""
        ix = np.clip(((Q[:, 0] - self.x0) / DA_CELL).astype(np.int32), 0, self.nx - 1)
        iy = np.clip(((Q[:, 1] - self.y0) / DA_CELL).astype(np.int32), 0, self.ny - 1)
        return self.G[iy, ix]


def dac_violation(grid, obs, preds, stat_th=STAT_TH, chunk=200000):
    """모드별 DAC 위반 여부.

    obs   (n, To, 2)  관측
    preds (n, K, Tp, 2) 후보 궤적
    반환  (n, K) bool — True 면 그 모드는 도로를 이탈한다

    관측 구간을 앞에 붙여 진행방향을 안정적으로 구한 뒤 박스를 세운다.
    정지 궤적(관측 이동 <= stat_th)은 진행방향이 정의되지 않으므로 판정하지 않는다.
    """
    n, K, Tp = preds.shape[0], preds.shape[1], preds.shape[2]
    moving = np.linalg.norm(obs[:, -1] - obs[:, 0], axis=1) > stat_th
    out = np.zeros((n, K), bool)
    step = max(1, chunk // K)
    for a in range(0, n, step):
        b = min(n, a + step)
        E = np.concatenate([np.repeat(obs[a:b, None], K, 1), preds[a:b]], 2)
        E = E.reshape(-1, E.shape[2], 2)
        BX = boxes(E).reshape(-1, 4, 2)
        degen = np.abs(BX[:, 0] - BX[:, 2]).sum(1) < 1e-9      # 방향 0 인 퇴화 박스
        ins = grid.inside(BX.reshape(-1, 2)).reshape(-1, 4)
        miss = np.where(degen, 0, 4 - ins.sum(1)).reshape(b - a, K, -1)
        out[a:b] = (miss >= 1).any(-1)
    return out & moving[:, None]


def select(probs, bad):
    """살아남은 모드 중 확률 1위. 전부 탈락하면 원래 확률 1위(fallback).

    반환 (선택 인덱스 (n,), 전멸 여부 (n,))
    """
    top = probs.argmax(1)
    surv = ~bad
    has = surv.any(1)
    return np.where(has, np.where(surv, probs, -1.0).argmax(1), top), ~has


def apply_dac(site, obs, preds, probs, mapdir=None, margin=DA_MG, stat_th=STAT_TH):
    """지점 하나에 DAC 필터를 적용해 모드를 하나 고른다.

    반환 dict(sel, bad, wipe, grid)
    """
    grid = DrivableGrid(site, mapdir=mapdir, margin=margin)
    bad = dac_violation(grid, obs, preds, stat_th=stat_th)
    sel, wipe = select(probs, bad)
    return dict(sel=sel, bad=bad, wipe=wipe, grid=grid)
