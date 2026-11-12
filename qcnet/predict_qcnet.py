"""QCNet test 예측을 TUTR 과 같은 형식으로 내보낸다 (compute_cp.py / eval_lab.py 입력).

  <out>_pred.csv     scenario_id, track_id, timestep (10..39), pred_x, pred_y  — 확률 최고 모드, 전역 좌표
  <out>_metrics.csv  TUTR test meta 열 + ade1, fde1 (top-1), minade, minfde (K=6, ADE/FDE 각각 최소)
  평가 샘플 = TUTR test meta 의 (scenario_id, track_id) 집합 (모두 있어야 한다).

사용
  python predict_qcnet.py --site ss --ckpt checkpoint/ss_vanilla/<best>.ckpt --out <dir>/ss_qcnet_test_k6
  --ckpt 에 폴더를 주면 그 폴더의 val_minFDE 최소 ckpt (epoch=NNN-val_minFDE=X.ckpt).
  --raw_dir/--processed_dir/--all_agents : 생성 장면처럼 meta 가 없는 raw 폴더 — 모든 에이전트의 top-1 예측만 _pred.csv 로
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from datasets import ArgoverseV2Dataset_ACL  # noqa: E402
from predictors import QCNet  # noqa: E402

ROOT = os.path.dirname(HERE)
OBS, PRED = 10, 30


def resolve_ckpt(p):
    if os.path.isdir(p):
        c = [(float(re.search(r'val_minFDE=(\d+\.\d+)', f).group(1)), f)
             for f in glob.glob(os.path.join(p, 'epoch=*-val_minFDE=*.ckpt'))]
        return min(c)[1]
    return p


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', required=True, choices=['ss', 'ny'])
    ap.add_argument('--ckpt', required=True, help='ckpt 파일, 또는 run 폴더 (val_minFDE best)')
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--raw_dir', default=None)
    ap.add_argument('--processed_dir', default=None)
    ap.add_argument('--all_agents', action='store_true', help='meta 없이 모든 에이전트 (_pred.csv 만)')
    a = ap.parse_args()
    dev = torch.device('cuda')
    ck = resolve_ckpt(a.ckpt)
    model = QCNet.load_from_checkpoint(ck, map_location=dev)
    model.eval()
    print(f'[predict] {ck}', flush=True)
    root = os.path.join(ROOT, 'data', 'qcnet', a.site)
    mk = lambda: ArgoverseV2Dataset_ACL(root, 'test', a.raw_dir or os.path.join(root, a.split, 'raw'),  # noqa: E731
                                        a.processed_dir or os.path.join(root, 'processed', a.split), None)
    ds = mk()
    if len(ds) == 0:   # 처음 처리한 경우 샘플 수가 처리 전 목록으로 잡혀 0 이다 (datamodule 처럼 한 번 더 만든다)
        ds = mk()
    if len(ds) == 0:
        raise SystemExit(f'{a.raw_dir or os.path.join(root, a.split, "raw")} 에 시나리오가 없음: python run.py --stage prep 을 먼저 돌리세요')

    keys, best1, errs = [], [], []
    for data in DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=4):
        for sid, ids in zip(data['scenario_id'], data['agent']['id']):
            keys += [(str(sid), int(t)) for t in ids]
        data = data.to(dev)
        data['agent']['av_index'] += data['agent']['ptr'][:-1]
        pred = model(data)
        loc = pred['loc_refine_pos'][..., :2]                                   # [N, K, T, 2] 로컬
        origin = data['agent']['position'][:, OBS - 1, :2]
        th = data['agent']['heading'][:, OBS - 1]
        c, s = th.cos(), th.sin()
        rot = torch.stack([torch.stack([c, s], -1), torch.stack([-s, c], -1)], -2)   # test_step 과 같은 역회전
        glob_ = torch.einsum('nktd,nde->nkte', loc, rot) + origin[:, None, None]
        pi = F.softmax(pred['pi'], -1)
        gt = data['agent']['position'][:, OBS:, :2]
        err = torch.norm(glob_ - gt[:, None], dim=-1)                           # [N, K, T]
        top = pi.argmax(-1)
        ar = torch.arange(len(top), device=dev)
        best1.append(glob_[ar, top].cpu().numpy())
        errs.append(torch.stack([err[ar, top].mean(-1), err[ar, top, -1],
                                 err.mean(-1).min(-1).values, err[:, :, -1].min(-1).values], 1).cpu().numpy())
    best1, errs = np.concatenate(best1), np.concatenate(errs)
    pos = {k: i for i, k in enumerate(keys)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.all_agents:
        kk = np.array(keys, dtype=object)
        pd.DataFrame({'scenario_id': np.repeat(kk[:, 0], PRED), 'track_id': np.repeat(kk[:, 1].astype(int), PRED),
                      'timestep': np.tile(np.arange(OBS, OBS + PRED), len(keys)),
                      'pred_x': best1[..., 0].reshape(-1), 'pred_y': best1[..., 1].reshape(-1)}).to_csv(
            a.out + '_pred.csv', index=False)
        print(f'{len(keys):,} 에이전트 / {len(set(kk[:, 0])):,} 장면 -> {a.out}_pred.csv', flush=True)
        return

    meta = pd.read_csv(os.path.join(ROOT, 'data', 'tutr', a.site, f'{a.site}_{a.split}_meta.csv'),
                       dtype={'scenario_id': str, 'date': str})
    rows = np.array([pos.get((s, int(t)), -1) for s, t in zip(meta['scenario_id'], meta['track_id'])])
    if (rows < 0).any():
        raise SystemExit(f'QCNet 예측에 없는 meta 샘플 {(rows < 0).sum()} 개')
    meta[['ade1', 'fde1', 'minade', 'minfde']] = errs[rows]
    meta.to_csv(a.out + '_metrics.csv', index=False)
    tr = best1[rows]                                                           # [M, T, 2]
    pr = pd.DataFrame({'scenario_id': np.repeat(meta['scenario_id'].to_numpy(), PRED),
                       'track_id': np.repeat(meta['track_id'].to_numpy(), PRED),
                       'timestep': np.tile(np.arange(OBS, OBS + PRED), len(meta)),
                       'pred_x': tr[..., 0].reshape(-1), 'pred_y': tr[..., 1].reshape(-1)})
    pr.to_csv(a.out + '_pred.csv', index=False)
    print(f'{len(meta):,} 샘플 (QCNet 에이전트 {len(keys):,}) | ADE1 {meta.ade1.mean():.4f} FDE1 {meta.fde1.mean():.4f} | '
          f'minADE6 {meta.minade.mean():.4f} minFDE6 {meta.minfde.mean():.4f}', flush=True)


if __name__ == '__main__':
    main()
