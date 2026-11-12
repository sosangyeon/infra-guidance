"""QCNet Ours — Contrastive(CP) 의 0~200 best 에서 3 라운드(33/33/34 epoch) contrastive + 생성 데이터 post-train.

라운드 r (1, 2, 3):
  1. mine   : 현재 모델로 remain 예측. 차량 단위 (mine_vehicle): 차량마다 최악 window 하나, 이전 라운드 차량 제외,
              예산은 샘플 1 % 개수. (--mine_unit sample 이면 샘플 단위 상위 1 %)
  2. pb     : tail 시나리오 -> LCSim 입력 (generation/make_tail_pb.py)
  3. alloc  : 시나리오별 생성 개수 배분 (generation/v7lib/rss_alloc.py), 총량 = 시나리오수 × num_aug
              ours13 은 마이닝 minADE 기준 (--alloc_crit minade), 평균 12, 6~24
  4. gen    : LCSim 생성 (generation/generate_v7.py) — 물리(no_offroad + repeller, head 만) + membership + anchor-pull,
              λ_rel 0.6, λ_tail 0.25, stitch, plan 1 회. ours13 은 membership 의 tail 목표가 push (--tail_mode push)
  5. shift  : time-window shift + pad (generation/postprocess_shift.py)
  6. qc     : 생성 CSV -> QCNet raw (gen_to_qcnet.py)
  7. cp     : QCNet Vanilla best 로 생성 장면 예측 -> 장면 CP (init 의 qcnet_vanilla_cp_max 와 같은 정의, tutr/ours/gen_cp.py)
  8. train  : init + 생성분(1..r), 생성분 추출 비율 gen_frac 0.5 (정확히 맞춤), contrastive (qcnet_round_ours.py)
              라운드 best = val minADE + val Worst3% 가 최소인 epoch
현재 모델: 1 라운드는 시작 모델(Contrastive(CP) 의 best_2of3.ckpt = 0~200 best), 이후는 직전 라운드 best.
최종 모델 = 마지막 라운드 best (<out_root>/round3/best.ckpt).

TUTR 과 다른 점: 마이닝 K = 6 (QCNet 출력 모드 수), 차량 단위 마이닝, gen_frac 0.5, 라운드 best 기준.

보통은 저장소 루트의 run.py 가 configs/qcnet_<site>_ours13.json 값을 아래 인자로 바꿔 부른다.
  python run.py --config configs/qcnet_ny_ours13.json --stage ours
"""
import argparse
import json
import math
import os
import sys

import pandas as pd

OURS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OURS)
import round_utils as G  # noqa: E402

GEN_CP = os.path.join(G.ROOT, 'tutr', 'ours', 'gen_cp.py')


def gen_cp(site, rd, vanilla_ckpt):
    out = os.path.join(rd, 'gen_cp.csv')
    if os.path.exists(out):
        return out
    raw = os.path.join(rd, 'qc', 'raw')
    pred = os.path.join(rd, 'gen_vanilla')
    if not os.path.exists(pred + '_pred.csv'):
        G.run([G.PY, 'predict_qcnet.py', '--site', site, '--ckpt', vanilla_ckpt,
               '--raw_dir', raw, '--processed_dir', os.path.join(rd, 'qc', 'proc'), '--all_agents', '--out', pred])
    meta = os.path.join(rd, 'gen_scenes.csv')
    pd.DataFrame({'scenario_id': sorted(d for d in os.listdir(raw) if os.path.isdir(os.path.join(raw, d)))}).to_csv(meta, index=False)
    G.run([G.PY, GEN_CP, '--pred_csv', pred + '_pred.csv', '--meta_csv', meta, '--out', out])
    return out


def mine_vehicle(site, r, rd, out_root, ckpt, top_percent=1.0, pool='remain'):
    """차량 단위 마이닝. sliding window 때문에 같은 차량이 연속 시나리오에 여러 번 들어가
    샘플 단위 상위 1 % 가 차량 1~6 대에 몰린다 (NY). 예산(샘플 1 % 개수)은 그대로 두고
    차량(날짜, track_id)마다 minADE@6 가 가장 큰 window 하나만, 이전 라운드에서 뽑힌 차량은 빼고 고른다."""
    out = os.path.join(rd, 'tail.json')
    if os.path.exists(out):
        return
    pred = os.path.join(rd, f'mine_{pool}')
    if not os.path.exists(pred + '_metrics.csv'):
        G.run([G.PY, 'predict_qcnet.py', '--site', site, '--split', pool, '--ckpt', ckpt, '--out', pred])
    m = pd.read_csv(pred + '_metrics.csv', dtype={'scenario_id': str})
    m['vehicle'] = m['scenario_id'].str.split('_').str[0] + '_' + m['track_id'].astype(str)
    used = set()
    for q in range(1, r):
        for sid, tids in json.load(open(os.path.join(out_root, f'round{q}', 'tail.json'), encoding='utf-8')).items():
            used |= {f'{sid.split("_")[0]}_{t}' for t in tids}
    p = m[~m['vehicle'].isin(used)]
    n = max(1, math.ceil(len(m) * top_percent / 100))
    top = p.sort_values('minade', ascending=False).drop_duplicates('vehicle').head(n)
    tail = {sid: sorted(int(t) for t in g['track_id']) for sid, g in top.groupby('scenario_id')}
    json.dump(tail, open(out, 'w', encoding='utf-8'), indent=1)
    print(f'[mine r{r}] 차량 단위: 풀 {len(p):,} 샘플 / {p["vehicle"].nunique():,} 차량 (제외 {len(used)} 차량) -> '
          f'tail {n} 차량 / {len(tail)} 시나리오, minADE6 >= {top["minade"].min():.3f} (풀 평균 {p["minade"].mean():.3f})', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', required=True, choices=['ss', 'ny'])
    ap.add_argument('--config', default=None, help='model / optimizer / contrastive 절이 있는 설정 (configs/*.json)')
    ap.add_argument('--start_ckpt', required=True, help='Contrastive(CP) 의 best_2of3.ckpt (0~200 best)')
    ap.add_argument('--vanilla_ckpt', required=True, help='생성 장면 CP 계산용 Vanilla ckpt, 또는 그 실험 폴더')
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--round_epochs', default='33,33,34')
    ap.add_argument('--epoch_offset', type=int, default=200)
    ap.add_argument('--total_epochs', type=int, default=300)
    ap.add_argument('--alpha', type=float, default=0.7)
    ap.add_argument('--clip', type=float, default=0.5)
    ap.add_argument('--gen_frac', type=float, default=0.5, help='생성분 추출 비율 (정확히 맞춤)')
    ap.add_argument('--select', default='val_minADE+val_minADE_Worst3%', help='라운드 best 기준 (qcnet_round.py --select)')
    ap.add_argument('--top_percent', type=float, default=1.0)
    ap.add_argument('--mine_unit', default='vehicle', choices=['vehicle', 'sample'],
                    help='vehicle: 차량마다 최악 window 하나, 이전 라운드 차량 제외 / sample: 샘플 단위')
    ap.add_argument('--num_aug', type=int, default=12, help='시나리오당 평균 생성 수 (배분 총량)')
    ap.add_argument('--alloc_gamma', type=float, default=2.0)
    ap.add_argument('--alloc_cap', type=int, default=24)
    ap.add_argument('--alloc_floor', type=int, default=2)
    ap.add_argument('--rss_frame', default='world', choices=['world', 'heading'])
    ap.add_argument('--alloc_crit', default='rss', choices=['rss', 'minade'],
                    help='생성 개수 배분 기준. minade = 마이닝 minADE (ours13)')
    ap.add_argument('--tail_mode', default='keep', choices=['keep', 'push'],
                    help='membership 의 tail 목표. push = 이웃 희귀 클러스터로 밀기 (ours13)')
    ap.add_argument('--push_alt', type=int, default=3)
    ap.add_argument('--lambda_rel', type=float, default=0.6)
    ap.add_argument('--lambda_tail', type=float, default=0.25)
    ap.add_argument('--vocab', default=None, help='기본: generation/vocab/c2_lcsim_v6_<site>.pt')
    ap.add_argument('--lcsim_ckpt', default=None, help='기본: generation/runs/lcsim_<site> 의 마지막 ckpt, 없으면 weights/lcsim/<site>')
    ap.add_argument('--lcsim_cfg', default=None, help='기본: generation/motion_diff/configs/lis_<site>_config.yml')
    ap.add_argument('--gen_env', default=None, help='생성 환경변수를 덮어쓸 JSON 파일 ({이름: 값})')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    out_root = os.path.abspath(a.out_root)
    start = os.path.abspath(a.start_ckpt)
    vanilla = os.path.abspath(a.vanilla_ckpt)
    epochs = [int(e) for e in a.round_epochs.split(',')]
    vocab = a.vocab or os.path.join(G.GEN, 'vocab', f'c2_lcsim_v6_{a.site}.pt')
    if not os.path.exists(vocab):
        sys.exit(f'C2 어휘 없음: {vocab} (generation/build_c2_vocab.py --site {a.site})')
    cfg = a.lcsim_cfg or os.path.join(G.GEN, 'motion_diff', 'configs', f'lis_{a.site}_config.yml')
    ckpt = a.lcsim_ckpt or G.lcsim_ckpt(a.site)
    gen_env = {
        'GUID_ON': '1', 'RG_PLAN_INTERVAL': '30', 'RG_V7_VOCAB': os.path.abspath(vocab),
        'RG_LAMBDA_REL': str(a.lambda_rel), 'RG_STITCH': '1',
        'RG_V7_PHYS': '1', 'RG_GUID_HEAD_ONLY': '1', 'REPELLER_RADIUS': '2.5',
        'NO_OFFROAD_EPS': '0.0', 'RG_MODULES': 'no_offroad_exact,repeller_exact', 'RG_TAIL_REALISM': '0',
        'LCSIM_CKPT': os.path.abspath(ckpt), 'LCSIM_MCFG': os.path.abspath(cfg),
        'RG_V7_TAIL_MODE': a.tail_mode, 'RG_V7_PUSH_ALT': str(a.push_alt),
    }
    if a.gen_env:
        gen_env.update({k: str(v) for k, v in json.load(open(a.gen_env, encoding='utf-8')).items()})
    print(f'[run_qcnet_ours] {a.site} | start {start} | gen_frac {a.gen_frac} | membership({a.tail_mode}) | '
          f'alloc {a.alloc_crit} base {a.num_aug} [{a.alloc_floor},{a.alloc_cap}] frame {a.rss_frame} | LCSim {ckpt}', flush=True)
    cur = start
    cps = []
    for r in range(1, len(epochs) + 1):
        rd = os.path.join(out_root, f'round{r}')
        os.makedirs(rd, exist_ok=True)
        if os.path.exists(os.path.join(rd, 'done')):
            cur = G.round_best(rd)
            cps.append(os.path.join(rd, 'gen_cp.csv'))
            print(f'[round {r}] 이미 끝남 -> {cur}', flush=True)
            continue
        if a.mine_unit == 'vehicle':
            mine_vehicle(a.site, r, rd, out_root, cur, a.top_percent)
        else:
            G.mine(a.site, r, rd, out_root, cur, a.top_percent)
        if not os.path.exists(os.path.join(rd, 'pb')):
            G.run([G.PY, 'make_tail_pb.py', '--site', a.site, '--tail_json', os.path.join(rd, 'tail.json'),
                   '--out_root', os.path.join(rd, 'pb')], cwd=G.GEN)
        alloc = os.path.join(rd, 'alloc.json')
        if not os.path.exists(alloc):
            G.run([G.PY, '-m', 'v7lib.rss_alloc', '--site', a.site, '--scenarios', G.SCN,
                   '--tail_json', os.path.join(rd, 'tail.json'), '--out', alloc, '--base', a.num_aug,
                   '--gamma', a.alloc_gamma, '--cap', a.alloc_cap, '--floor', a.alloc_floor,
                   '--frame', a.rss_frame, '--crit', a.alloc_crit,
                   '--metrics_csv', os.path.join(rd, 'mine_remain_metrics.csv')], cwd=G.GEN)
        env = dict(gen_env, LCSIM_DATA=rd, RG_PB_ORIG=os.path.join(rd, 'pb'),
                   RG_TAIL_JSON=os.path.join(rd, 'tail.json'), RG_ALLOC_JSON=alloc)
        if not os.path.exists(os.path.join(rd, 'gen.done')):
            G.run([G.PY, 'generate_v7.py', '0', 'gen', str(a.lambda_tail)], cwd=G.GEN, env=env, log=os.path.join(rd, 'gen.log'))
            open(os.path.join(rd, 'gen.done'), 'w').close()
        if not os.path.exists(os.path.join(rd, 'shift.done')):
            G.run([G.PY, 'postprocess_shift.py', 'gen', 'gen_shift'], cwd=G.GEN, env=env, log=os.path.join(rd, 'gen.log'))
            open(os.path.join(rd, 'shift.done'), 'w').close()
        G.to_qcnet(a.site, rd)
        cps.append(gen_cp(a.site, rd, vanilla))
        gen = ','.join(f'{q}:{os.path.join(out_root, f"round{q}", "qc")}' for q in range(1, r + 1))
        G.run([G.PY, os.path.join(OURS, 'qcnet_round_ours.py'), '--site', a.site, '--gen', gen, '--round', r,
               '--gen_cp', ','.join(cps), '--gen_frac', a.gen_frac, '--gen_frac_exact', '--select', a.select,
               '--alpha', a.alpha, '--clip', a.clip,
               '--init_ckpt', cur, '--epochs', epochs[r - 1], '--epoch_offset', a.epoch_offset + sum(epochs[:r - 1]),
               '--total_epochs', a.total_epochs, '--out_dir', rd, '--batch_size', a.batch_size,
               '--num_workers', a.num_workers, '--seed', a.seed]
              + (['--config', os.path.abspath(a.config)] if a.config else []))
        open(os.path.join(rd, 'done'), 'w').close()
        cur = G.round_best(rd)
    print(f'[run_qcnet_ours] 최종 모델: {cur}', flush=True)


if __name__ == '__main__':
    main()
