"""TUTR 후보 궤적 덤프 (안전 궤적 선택 C3 입력).

분할 하나를 예측해 샘플마다 점수 상위 N 개 후보와 점수를 원래 좌표계로 저장한다.
  <out>.npz  sid, tid (meta 순서), obs [M,10,2], gt [M,30,2], cand [M,N,30,2], score [M,N] (전체 모드 softmax 중 상위 N, 내림차순)

사용
  python dump_tutr.py --site ny --split test --hp_config <cfg> --run_dir <round3 폴더> --out <npz>
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TG = os.path.join(ROOT, 'tutr', 'ours')
sys.path.insert(0, TG)
from dataset import TrajectoryDataset  # noqa: E402
from feature_utils import output_dim, restore_output_torch, select_output_features  # noqa: E402
from model import TrajectoryModel  # noqa: E402
from predict import load_config  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--hp_config', required=True)
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--ckpt', default='best.pth')
    ap.add_argument('--num_k', type=int, default=20)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = torch.device('cuda')
    hp = load_config(os.path.join(TG, a.hp_config) if not os.path.isabs(a.hp_config) else a.hp_config)
    ds = TrajectoryDataset(os.path.join(ROOT, 'data', 'tutr', a.site), a.site, [a.split], translation=True,
                           rotation=True, scaling=False, obs_len=10, dist_threshold=hp.dist_threshold,
                           return_transform=True)
    loader = DataLoader(ds, collate_fn=ds.coll_fn, batch_size=hp.batch_size, shuffle=False)
    with open(os.path.join(a.run_dir, 'run_motion_modes.pkl'), 'rb') as f:
        modes = torch.tensor(pickle.load(f), dtype=torch.float32, device=dev)
    om = 'xy_vxvy'
    model = TrajectoryModel('xy_vxvy_axay', om, 10, 30, hp.model_hidden_dim, 2, [1, 1], 4, 2).to(dev)
    ck = torch.load(os.path.join(a.run_dir, a.ckpt), map_location=dev, weights_only=False)
    r = model.load_state_dict(ck['model_state_dict'] if 'model_state_dict' in ck else ck, strict=False)
    if r.missing_keys:
        raise SystemExit(f'가중치에 없는 키: {r.missing_keys[:5]}')
    if r.unexpected_keys:
        print('예측에 안 쓰는 키 무시:', r.unexpected_keys[:5])
    model.eval()
    obs_l, gt_l, cand_l, sc_l = [], [], [], []
    for batch in loader:
        ped, neis, mask, origin, rot, scale = [v.to(dev) for v in batch]
        pred_flat, scores = model(ped[:, :10], neis[:, :, :10], modes, mask, test=True, num_k=a.num_k)
        pred = pred_flat.reshape(pred_flat.shape[0], pred_flat.shape[1], 30, output_dim(om))
        cand = restore_output_torch(pred, om, origin, rot, scale)[..., :2]
        gt = restore_output_torch(select_output_features(ped[:, 10:], om), om, origin, rot, scale)[..., :2]
        obs = restore_output_torch(select_output_features(ped[:, :10], om), om, origin, rot, scale)[..., :2]
        top = torch.topk(torch.softmax(scores, -1), k=pred.shape[1], dim=-1).values   # 전체 모드 softmax, predict.py 와 같은 순서
        obs_l.append(obs.cpu().numpy()); gt_l.append(gt.cpu().numpy()); cand_l.append(cand.cpu().numpy())
        sc_l.append(top.cpu().numpy())
    m = ds.meta
    np.savez_compressed(a.out, sid=m['scenario_id'].astype(str).to_numpy(), tid=m['track_id'].to_numpy(),
                        obs=np.concatenate(obs_l).astype(np.float32), gt=np.concatenate(gt_l).astype(np.float32),
                        cand=np.concatenate(cand_l).astype(np.float32), score=np.concatenate(sc_l).astype(np.float32))
    print(f'{a.site} {a.split}: {len(m):,} 샘플, 후보 {a.num_k} -> {a.out}')


if __name__ == '__main__':
    main()
