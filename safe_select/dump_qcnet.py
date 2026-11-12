"""QCNet 후보 궤적 덤프 (안전 궤적 선택 C3 입력). 출력 형식은 dump_tutr.py 와 같다 (N = 6, score = softmax(pi)).

사용
  python dump_qcnet.py --site ny --split test --ckpt <ckpt> --out <npz>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QC = os.path.join(ROOT, 'qcnet')
sys.path.insert(0, QC)
from datasets import ArgoverseV2Dataset_ACL  # noqa: E402
from predictors import QCNet  # noqa: E402

OBS = 10


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = torch.device('cuda')
    model = QCNet.load_from_checkpoint(a.ckpt, map_location=dev)
    model.eval()
    root = os.path.join(ROOT, 'data', 'qcnet', a.site)
    mk = lambda: ArgoverseV2Dataset_ACL(root, 'test', os.path.join(root, a.split, 'raw'),  # noqa: E731
                                        os.path.join(root, 'processed', a.split), None)
    ds = mk()
    if len(ds) == 0:
        ds = mk()
    keys, obs_l, gt_l, cand_l, sc_l = [], [], [], [], []
    for data in DataLoader(ds, batch_size=32, shuffle=False, num_workers=4):
        for sid, ids in zip(data['scenario_id'], data['agent']['id']):
            keys += [(str(sid), int(t)) for t in ids]
        data = data.to(dev)
        data['agent']['av_index'] += data['agent']['ptr'][:-1]
        pred = model(data)
        loc = pred['loc_refine_pos'][..., :2]
        origin = data['agent']['position'][:, OBS - 1, :2]
        th = data['agent']['heading'][:, OBS - 1]
        c, s = th.cos(), th.sin()
        rot = torch.stack([torch.stack([c, s], -1), torch.stack([-s, c], -1)], -2)
        glob_ = torch.einsum('nktd,nde->nkte', loc, rot) + origin[:, None, None]
        pi = F.softmax(pred['pi'], -1)
        order = pi.argsort(-1, descending=True)
        ar = torch.arange(len(pi), device=dev)[:, None]
        cand_l.append(glob_[ar, order].cpu().numpy()); sc_l.append(pi[ar, order].cpu().numpy())
        obs_l.append(data['agent']['position'][:, :OBS, :2].cpu().numpy())
        gt_l.append(data['agent']['position'][:, OBS:, :2].cpu().numpy())
    pos = {k: i for i, k in enumerate(keys)}
    meta = pd.read_csv(os.path.join(ROOT, 'data', 'tutr', a.site, f'{a.site}_{a.split}_meta.csv'),
                       dtype={'scenario_id': str})
    rows = np.array([pos[(s, int(t))] for s, t in zip(meta['scenario_id'], meta['track_id'])])
    cat = lambda L: np.concatenate(L)[rows].astype(np.float32)  # noqa: E731
    np.savez_compressed(a.out, sid=meta['scenario_id'].to_numpy().astype(str), tid=meta['track_id'].to_numpy(),
                        obs=cat(obs_l), gt=cat(gt_l), cand=cat(cand_l), score=cat(sc_l))
    print(f'{a.site} {a.split}: {len(meta):,} 샘플, 후보 6 -> {a.out}')


if __name__ == '__main__':
    main()
