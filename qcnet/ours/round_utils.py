"""QCNet Ours 라운드에서 쓰는 공통 함수 (run_qcnet_ours.py 가 부른다)."""
import glob
import json
import math
import os
import subprocess
import sys

import pandas as pd

OURS = os.path.dirname(os.path.abspath(__file__))
QC = os.path.dirname(OURS)
ROOT = os.path.dirname(QC)                      # 저장소 루트
SCN = os.path.join(ROOT, 'data', 'scenarios')
GEN = os.path.join(ROOT, 'generation')
PY = sys.executable


def run(cmd, cwd=QC, env=None, log=None):
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    e = dict(os.environ, PYTHONIOENCODING='utf-8', **(env or {}))
    if log:
        with open(log, 'a', encoding='utf-8') as f:
            subprocess.run([str(c) for c in cmd], cwd=cwd, env=e, check=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run([str(c) for c in cmd], cwd=cwd, env=e, check=True)


def lcsim_ckpt(site):
    """generation/runs/lcsim_<site> 에 다시 학습한 것이 있으면 그것, 없으면 weights/lcsim/<site>."""
    c = sorted(glob.glob(os.path.join(GEN, 'runs', f'lcsim_{site}', 'logs', '*', 'checkpoints', '*.ckpt')),
               key=os.path.getmtime)
    c = c or sorted(glob.glob(os.path.join(ROOT, 'weights', 'lcsim', site, '*.ckpt')))
    if not c:
        sys.exit(f'LCSim 체크포인트 없음 ({site}): python download_weights.py 로 받거나 '
                 f'python run.py --config <json> --stage lcsim 으로 학습하세요')
    return c[-1]


def mine(site, r, rd, out_root, ckpt, top_percent=1.0, pool='remain'):
    """샘플 단위 마이닝: 이전 라운드 tail 시나리오 제외, 차량별 minADE@6 상위 p %."""
    out = os.path.join(rd, 'tail.json')
    if os.path.exists(out):
        return
    pred = os.path.join(rd, f'mine_{pool}')
    if not os.path.exists(pred + '_metrics.csv'):
        run([PY, 'predict_qcnet.py', '--site', site, '--split', pool, '--ckpt', ckpt, '--out', pred])
    m = pd.read_csv(pred + '_metrics.csv', dtype={'scenario_id': str})
    used = set()
    for q in range(1, r):
        used |= set(json.load(open(os.path.join(out_root, f'round{q}', 'tail.json'), encoding='utf-8')))
    p = m[~m['scenario_id'].isin(used)]
    n = max(1, math.ceil(len(p) * top_percent / 100))
    top = p.nlargest(n, 'minade')
    tail = {sid: sorted(int(t) for t in g['track_id']) for sid, g in top.groupby('scenario_id')}
    json.dump(tail, open(out, 'w', encoding='utf-8'), indent=1)
    print(f'[mine r{r}] 풀 {len(p):,} 샘플 (제외 {len(used)} 시나리오) -> tail {n} 차량 / {len(tail)} 시나리오, '
          f'minADE6 >= {top["minade"].min():.3f} (풀 평균 {p["minade"].mean():.3f})', flush=True)


def to_qcnet(site, rd):
    if not os.path.exists(os.path.join(rd, 'qc', 'raw')):
        run([PY, os.path.join(OURS, 'gen_to_qcnet.py'), '--site', site, '--gen_dir', os.path.join(rd, 'gen_shift'),
             '--out', os.path.join(rd, 'qc', 'raw')])


def round_best(rd):
    """qcnet_round.py 가 쓴 best.txt 의 체크포인트 경로."""
    with open(os.path.join(rd, 'best.txt'), encoding='utf-8') as f:
        return f.read().split(' val_')[0].strip()
