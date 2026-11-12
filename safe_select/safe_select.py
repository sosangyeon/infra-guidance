"""안전 궤적 선택 (C3) — 주행가능영역 게이트 + 이웃 예측 충돌 페널티로 보고 집합 K 개를 고른다.

  1) 게이트  : 후보가 주행가능영역을 여유 margin 이상 벗어나면 제거 (dac_grid.py 의 박스 판정,
               관측 이동이 짧은 정지 차량은 판정 안 함). 전부 제거되면 게이트를 건너뛴다.
  2) 점수    : score(k) = s_k − λ_c · CP_k
               s_k  = 모델 확률, CP_k = 같은 장면 이웃(t0 거리 < d_th) 의 top-1 예측과의 양방향 CP 의 시점·이웃 최대
               (data_prep/compute_cp.py 의 cp_pair, heading 은 예측 xy 의 gradient — MR_CP 계산과 같은 식)
  3) 선택    : 점수 순으로 min(K, |C|) 개. 첫 번째가 top-1 (MR_CP 계산에 쓰임).
  λ_c 는 val 에서만 고른다: val minADE 가 λ_c = 0 (게이트만) 보다 나빠지지 않는 λ_c 중 val MR_CP 사건 수 최소
  (같으면 작은 λ_c). 정답은 지표 계산에만 쓴다.

출력: <out>_pred.csv (top-1, compute_cp.py 입력 형식), <out>_metrics.csv (meta + ade1, fde1, minade, minfde)
사용
  python safe_select.py --site ny --val <val.npz> --test <test.npz> --out <prefix>
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'data_prep'))
from dac_grid import DrivableGrid, dac_violation  # noqa: E402
from compute_cp import cp_pair, SIGMA0, ALPHA  # noqa: E402

SIG = np.sqrt(SIGMA0 ** 2 + ALPHA * np.arange(30))   # timestep 10..39


def grid_for(site, margin):
    d = os.path.join(HERE, '_maps')
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, f'{site.upper()}_map.json')
    if not os.path.exists(dst):
        shutil.copy(os.path.join(ROOT, 'data', 'qcnet', site, 'test', 'raw', f'log_map_archive_{site}.json'), dst)
    return DrivableGrid(site, mapdir=d, margin=margin)


def heading(xy):
    g = np.gradient(xy, axis=-2)
    return np.arctan2(g[..., 1], g[..., 0])


def cp_to_neighbors(D, d_th):
    """후보마다 CP_k [M, N]. 이웃 = 같은 장면 다른 샘플, top-1 예측 사용."""
    sid, cand, obs = D['sid'], D['cand'], D['obs']
    M, N = cand.shape[:2]
    out = np.zeros((M, N), np.float32)
    top1 = cand[:, 0]
    h_c = heading(cand)
    h_t = heading(top1)
    idx = pd.Series(np.arange(M)).groupby(sid).apply(list)
    for rows in idx:
        if len(rows) < 2:
            continue
        rows = np.array(rows)
        for a in rows:
            nb = rows[(rows != a) & (np.linalg.norm(obs[rows, -1] - obs[a, -1], axis=1) < d_th)]
            if len(nb) == 0:
                continue
            xa, ya, ha = cand[a, :, None, :, 0], cand[a, :, None, :, 1], h_c[a][:, None]      # [N,1,T]
            xb, yb, hb = top1[nb][None, :, :, 0], top1[nb][None, :, :, 1], h_t[nb][None]      # [1,J,T]
            p = np.maximum(cp_pair(xa, ya, ha, xb, yb, hb, SIG), cp_pair(xb, yb, hb, xa, ya, ha, SIG))
            out[a] = p.max((1, 2))
    return out


def select(D, legal, cp, lam, K, fill):
    """fill=False: 통과 후보만 min(K, |C|) 개. fill=True: 통과 후보를 먼저, 모자라면 위반 후보로 K 개를 채운다."""
    score = D['score'] - lam * cp
    key = np.where(legal, score, score - 1e6 if fill else -np.inf)
    order = np.argsort(-key, axis=1, kind='stable')[:, :K]
    n_ok = np.full(len(order), K) if fill else np.minimum(legal.sum(1), K)
    keep = np.arange(K)[None] < n_ok[:, None]
    return order, keep


def metrics(D, order, keep):
    ar = np.arange(len(order))[:, None]
    err = np.linalg.norm(D['cand'][ar, order] - D['gt'][:, None], axis=-1)          # [M,K,T]
    ade = np.where(keep, err.mean(-1), np.inf)
    fde = np.where(keep, err[..., -1], np.inf)
    return ade.min(1), fde.min(1), ade[:, 0], fde[:, 0]


def mr_cp(D, order, thr):
    top = D['cand'][np.arange(len(order)), order[:, 0]]
    h = heading(top)
    ev = 0
    sc = pd.Series(np.arange(len(top))).groupby(D['sid']).apply(list)
    for rows in sc:
        if len(rows) < 2:
            continue
        rows = np.array(rows)
        ia, ib = np.triu_indices(len(rows), 1)
        a, b = rows[ia], rows[ib]
        p = np.maximum(cp_pair(top[a, :, 0], top[a, :, 1], h[a], top[b, :, 0], top[b, :, 1], h[b], SIG[None]),
                       cp_pair(top[b, :, 0], top[b, :, 1], h[b], top[a, :, 0], top[a, :, 1], h[a], SIG[None]))
        ev += p.max() > thr
    return ev, len(sc)


def five(mm, ev, nsc):
    minade, minfde, ade1, fde1 = mm
    n1, n5 = max(1, int(len(minade) * 0.01)), max(1, int(len(minade) * 0.05))
    s = np.sort(minade)
    return dict(top1=s[-n1:].mean(), top5=s[-n5:].mean(), var999=np.quantile(minade, 0.999),
                fpr5=(minfde > 5).mean() * 100, mr_cp=ev / nsc * 100, ev=int(ev), minade=minade.mean(),
                minfde=minfde.mean(), ade1=ade1.mean(), fde1=fde1.mean())


def load(p):
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def gt_thr(site, split):
    """test 는 test 정답 최댓값 (표의 MR_CP 와 같음). val 은 test 를 뺀 분할(init, remain, val) 정답 최댓값."""
    m = pd.read_csv(os.path.join(ROOT, 'data', 'scenarios', f'manifest_{site}.csv'), dtype={'scenario_id': str})
    sp = ['test'] if split == 'test' else ['init', 'remain', 'val']
    return m.loc[m['split'].isin(sp), 'gt_cp_max'].max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--K', type=int, default=6)
    ap.add_argument('--margin', type=float, default=3.5, help='주행가능영역 여유 δm [m]')
    ap.add_argument('--d_th', type=float, default=50.0)
    ap.add_argument('--lambdas', default='0,0.25,0.5,1,2,4,8')
    a = ap.parse_args()
    grid = grid_for(a.site, a.margin)
    lams = [float(v) for v in a.lambdas.split(',')]
    res = {}
    for split, path in (('val', a.val), ('test', a.test)):
        D = load(path)
        bad = dac_violation(grid, D['obs'], D['cand'])
        legal = ~bad
        legal[~legal.any(1)] = True                          # 전부 위반이면 게이트 우회
        gt_bad = dac_violation(grid, D['obs'], D['gt'][:, None])[:, 0].mean() * 100
        cp = cp_to_neighbors(D, a.d_th)
        thr = gt_thr(a.site, split)
        base = metrics(D, np.tile(np.arange(a.K), (len(D['sid']), 1)), np.ones((len(D['sid']), a.K), bool))
        row = {'topK': five(base, *mr_cp(D, np.tile(np.arange(a.K), (len(D['sid']), 1)), thr))}
        sel = {}
        for fill in (False, True):
            for lam in lams:
                o, k = select(D, legal, cp, lam, a.K, fill)
                mm = metrics(D, o, k)
                row[(fill, lam)] = five(mm, *mr_cp(D, o, thr))
                sel[(fill, lam)] = (o, k, mm)
        res[split] = (D, row, sel)
        print(f'[{a.site} {split}] 후보 {D["cand"].shape[1]}, 게이트 위반 후보 {bad.mean() * 100:.2f}%, '
              f'GT 위반 {gt_bad:.2f}%, 게이트 전멸 {(~(~bad).any(1)).mean() * 100:.2f}%, CP 임계 {thr:.4f}')
        for k_, v in row.items():
            print(f'  {str(k_):>6}  TOP1 {v["top1"]:.3f}  VaR {v["var999"]:.3f}  FPR5 {v["fpr5"]:.3f}%  '
                  f'MR_CP {v["mr_cp"]:.3f}% ({v["ev"]})  minADE {v["minade"]:.4f}')
    vrow = res['val'][1]
    # 채우기 방식: val TOP1 이 낮은 쪽 (λ_c = 0 기준). λ_c: val minADE 가 λ_c = 0 보다 나빠지지 않는 것 중 val MR_CP 최소
    fill = min((True, False), key=lambda f: vrow[(f, 0.0)]['top1'])   # 같으면 K 개를 채우는 쪽
    ok = [l_ for l_ in lams if vrow[(fill, l_)]['minade'] <= vrow[(fill, 0.0)]['minade'] + 1e-9]
    lam = min(ok, key=lambda l_: (vrow[(fill, l_)]['ev'], l_))
    D, trow, sel = res['test']
    o, k, (minade, minfde, ade1, fde1) = sel[(fill, lam)]
    v = trow[(fill, lam)]
    print(f'[{a.site}] val 로 고른 fill = {fill}, λ_c = {lam} -> test TOP1 {v["top1"]:.3f} VaR {v["var999"]:.3f} '
          f'FPR5 {v["fpr5"]:.3f}% MR_CP {v["mr_cp"]:.3f}% ({v["ev"]}) minADE {v["minade"]:.3f}')
    meta = pd.read_csv(os.path.join(ROOT, 'data', 'tutr', a.site, f'{a.site}_test_meta.csv'), dtype={'scenario_id': str})
    assert (meta['scenario_id'].to_numpy() == D['sid']).all() and (meta['track_id'].to_numpy() == D['tid']).all()
    meta['ade1'], meta['fde1'], meta['minade'], meta['minfde'] = ade1, fde1, minade, minfde
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    meta.to_csv(a.out + '_metrics.csv', index=False)
    top = D['cand'][np.arange(len(o)), o[:, 0]]
    pd.DataFrame({'scenario_id': np.repeat(D['sid'], 30), 'track_id': np.repeat(D['tid'], 30),
                  'timestep': np.tile(np.arange(10, 40), len(o)),
                  'pred_x': top[..., 0].reshape(-1), 'pred_y': top[..., 1].reshape(-1)}).to_csv(a.out + '_pred.csv', index=False)
    json.dump({'fill': fill, 'lambda_c': lam, 'margin': a.margin, 'K': a.K,
               'val': {str(k_): v_ for k_, v_ in vrow.items()}, 'test': {str(k_): v_ for k_, v_ in trow.items()}},
              open(a.out + '_summary.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
