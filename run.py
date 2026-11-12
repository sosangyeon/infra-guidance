"""단계별 실행기. configs/*.json 하나가 실험 하나 (백본 × 사이트) 의 값을 모두 담는다.

  python run.py --config configs/tutr_ss_ours13.json --stage eval                 # 받은 가중치로 평가만
  python run.py --config configs/tutr_ss_ours13.json --stage prep,vanilla,vanilla_cp,contrastive,lcsim,ours,eval

단계 (--stage, 쉼표로 여러 개)
  prep         data/Dataset_CSV -> 시나리오 -> 날짜 분할 -> GT CP -> TUTR pkl -> QCNet raw   (사이트당 한 번, 두 백본 공용)
  vanilla      Vanilla 학습 (init, 300 epoch)
  vanilla_cp   Vanilla 의 top-1 예측으로 시나리오 CP 를 계산해 manifest 에 넣는다 (contrastive 의 난이도)
  contrastive  Contrastive(CP) 학습 (init, 300 epoch). 0~200 epoch best 가 Ours 의 시작점
  lcsim        LCSim 확산 모델 학습 + C2 기동 어휘 (사이트당 한 번, 두 백본 공용)
  ours         Ours 3 라운드 (200~300 epoch): tail 마이닝 -> 생성 -> contrastive + 생성 데이터 학습
  eval         test 평가 (--arm vanilla | contrastive | ours | ours13, 기본 ours)

가중치 위치: runs/<backbone>/<site>/ 에 직접 학습한 것이 있으면 그것, 없으면 weights/ 의 배포 가중치를 쓴다.
  ours   = 논문 결과표의 가중치. 배포 가중치만 있다 (weights/).
  ours13 = ours 단계 (이 저장소의 생성 코드) 로 학습한 가중치. runs/ 에 다시 학습한 것이 있으면 그것을 쓴다.
출력은 모두 runs/<backbone>/<site>/ 아래에 쓴다 (compute_cp.py 는 data/scenarios/manifest_<site>.csv 에 CP 열을 더한다).
"""
import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
STAGES = ['prep', 'vanilla', 'vanilla_cp', 'contrastive', 'lcsim', 'ours', 'eval']
ARMS = ['vanilla', 'contrastive', 'ours', 'ours13']

for _s in (sys.stdout, sys.stderr):   # 한글을 못 쓰는 콘솔 코드페이지에서도 멈추지 않게
    _s.reconfigure(errors='replace')


def P(*a):
    return os.path.join(ROOT, *a)


def sh(cmd, cwd=ROOT):
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, env=dict(os.environ, PYTHONIOENCODING='utf-8'), check=True)


def manifest(site):
    return P('data', 'scenarios', f'manifest_{site}.csv')


def missing(what):
    sys.exit(f'{what} 없음: python download_weights.py 로 받거나 앞 단계를 먼저 돌리세요 (python run.py --help)')


def gen_env(c, out_root):
    """generation 절 -> 생성 스크립트 환경변수 (<out_root>/gen_env.json)."""
    g = c['generation']
    m, p = g['membership'], g['physics']
    env = {
        'RG_PLAN_INTERVAL': g['plan_interval'], 'RG_STITCH': int(g['stitch']), 'RG_LAMBDA_REL': g['lambda_rel'],
        'RG_V7_TAIL_MODE': g['tail_mode'], 'RG_V7_PUSH_ALT': g['push_alt'],
        'RG_V7_MARGIN': m['margin'], 'RG_V7_ALPHA': m['alpha'], 'RG_V7_NUMSTEP': m['num_steps'],
        'RG_V7_CLIP': m['clip'], 'RG_V7_ANCHOR_W': m['anchor_w'], 'RG_V7_PULL_SCOPE': m['pull_scope'],
        'RG_V7_STEPS': m['guidance_steps'], 'RG_V7_DENOISED': int(m['denoised']), 'RG_V7_SCOPE': m['scope'],
        'RG_MODULES': p['modules'], 'RG_GUID_HEAD_ONLY': int(p['head_only']),
        'REPELLER_RADIUS': p['repeller_radius'], 'NO_OFFROAD_EPS': p['no_offroad_eps'],
    }
    os.makedirs(out_root, exist_ok=True)
    path = os.path.join(out_root, 'gen_env.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({k: str(v) for k, v in env.items()}, f, indent=1)
    return path


def round_args(c):
    """두 백본의 Ours 실행기가 같이 받는 인자."""
    r, al, g = c['rounds'], c['allocation'], c['generation']
    return ['--round_epochs', ','.join(str(e) for e in r['round_epochs']), '--epoch_offset', r['epoch_offset'],
            '--total_epochs', r['total_epochs'], '--alpha', r['alpha'], '--clip', r['clip'], '--gen_frac', r['gen_frac'],
            '--num_aug', al['num_aug'], '--rss_frame', al['frame'], '--alloc_crit', al['crit'],
            '--tail_mode', g['tail_mode'], '--push_alt', g['push_alt'], '--lambda_rel', g['lambda_rel'],
            '--lambda_tail', g['lambda_tail'], '--seed', c['seed']]


# ---------------------------------------------------------------- 공용 단계
def prep(c, cfg, a):
    site, dp = c['site'], P('data_prep')
    sh([PY, 'make_scenarios.py', '--site', site], cwd=dp)
    sh([PY, 'split_dates.py', '--site', site], cwd=dp)
    sh([PY, 'compute_cp.py', '--site', site, '--source', 'gt'], cwd=dp)
    sh([PY, 'make_tutr_pkl.py', '--site', site], cwd=dp)
    sh([PY, 'make_qcnet_raw.py', '--site', site], cwd=dp)


def lcsim(c, cfg, a):
    gen = P('generation')
    sh([PY, 'run_lcsim_site.py', '--site', c['site']], cwd=gen)
    sh([PY, 'build_c2_vocab.py', '--site', c['site']], cwd=gen)


def pred_cp(site, pred_csv, prefix):
    sh([PY, 'compute_cp.py', '--site', site, '--source', 'pred', '--pred_csv', pred_csv, '--prefix', prefix],
       cwd=P('data_prep'))


def eval_lab(site, metrics, prefix, name):
    sh([PY, 'eval_lab.py', '--site', site, '--metrics', metrics, '--pred_cp', f'{prefix}_cp_max', '--name', name],
       cwd=P('data_prep'))


# ---------------------------------------------------------------- TUTR
def tutr_dir(c, arm):
    """실험 폴더 (best.pth, run_motion_modes.pkl). runs/ 가 먼저, 없으면 weights/."""
    site = c['site']
    if arm == 'ours':                                      # 배포 가중치만 있다
        cand = []
    elif arm == 'ours13':
        cand = [P('runs', 'tutr', site, 'ours13', f"round{len(c['rounds']['round_epochs'])}")]
    else:
        cand = sorted(os.path.dirname(p) for p in glob.glob(P('runs', 'tutr', site, arm, '*', 'best.pth')))
    cand = [d for d in cand if os.path.exists(os.path.join(d, 'best.pth'))] + [P('weights', 'tutr', site, arm)]
    if not os.path.exists(os.path.join(cand[0], 'best.pth')):
        missing(f'TUTR {site} {arm} 가중치 ({cand[0]})')
    return cand[0]


def tutr_io(c, cfg):
    d = c['data']
    return ['--dataset_path', P('data', 'tutr', c['site']), '--dataset_name', c['site'], '--hp_config', cfg,
            '--input_mode', d['input_mode'], '--output_mode', d['output_mode'],
            '--obs_len', d['obs_len'], '--pred_len', d['pred_len']]


def tutr_train(c, cfg, arm):
    script = P('tutr', 'vanilla', 'train.py') if arm == 'vanilla' else \
        P('tutr', 'contrastive', 'train_tutr_contrastive_cp.py')
    cmd = [PY, script] + tutr_io(c, cfg) + [
        '--train_splits', 'init', '--val_split', 'val', '--checkpoint', P('runs', 'tutr', c['site'], arm),
        '--loss_weights', c['train']['loss_weights'], '--num_works', c['train']['num_workers'],
        '--seed', c['seed'], '--resume', 'auto']
    if arm == 'contrastive':
        cmd += ['--cp_table', manifest(c['site']), '--cp_col', c['contrastive']['cp_col']]
    sh(cmd, cwd=os.path.dirname(script))


def tutr_vanilla_cp(c, cfg, a):
    out = P('runs', 'tutr', c['site'], 'vanilla_cp', 'all')
    sh([PY, 'predict.py'] + tutr_io(c, cfg) + ['--split', 'init,remain,val,test', '--num_k', 1,
                                               '--run_dir', tutr_dir(c, 'vanilla'), '--out', out],
       cwd=P('tutr', 'vanilla'))
    pred_cp(c['site'], out + '_pred.csv', c['contrastive']['cp_col'].removesuffix('_cp_max'))


def tutr_ours(c, cfg, a):
    site, mi, al = c['site'], c['mining'], c['allocation']
    out = P('runs', 'tutr', site, 'ours13')
    sh([PY, 'run_ours.py', '--site', site, '--hp_config', cfg,
        '--start_run_dir', tutr_dir(c, 'contrastive'), '--start_ckpt', 'best_2of3.pth',
        '--vanilla_run_dir', tutr_dir(c, 'vanilla'), '--cp_col', c['contrastive']['cp_col'], '--out_root', out,
        '--pool_split', mi['pool_split'], '--top_percent', mi['top_percent'], '--mine_k', mi['mine_k'],
        '--rss_gamma', al['gamma'], '--rss_cap', al['cap'], '--rss_floor', al['floor'],
        '--input_mode', c['data']['input_mode'], '--output_mode', c['data']['output_mode'],
        '--loss_weights', c['train']['loss_weights'], '--gen_env', gen_env(c, out)] + round_args(c),
       cwd=P('tutr', 'ours'))


def tutr_eval(c, cfg, a):
    site, arm, e = c['site'], a.arm, c['eval']
    rd = tutr_dir(c, arm)
    out = P('runs', 'tutr', site, 'eval', arm)
    pre = os.path.join(out, f"test_k{e['num_k']}")
    prefix = f'eval_tutr_{arm}'
    print(f'[eval] TUTR {site} {arm}: {rd}', flush=True)
    sh([PY, 'predict.py'] + tutr_io(c, cfg) + ['--split', 'test', '--num_k', e['num_k'], '--run_dir', rd, '--out', pre],
       cwd=P('tutr', 'ours'))
    pred_cp(site, pre + '_pred.csv', prefix)
    eval_lab(site, pre + '_metrics.csv', prefix, f'TUTR {site} {arm} (top-{e["num_k"]})')
    s = e['safe_select']                                   # 안전 궤적 선택 (C3): 후보 20 개에서 6 개
    for split in ('val', 'test'):
        sh([PY, 'dump_tutr.py', '--site', site, '--split', split, '--hp_config', cfg, '--run_dir', rd,
            '--num_k', s['candidates'], '--out', os.path.join(out, f'cand_{split}.npz')], cwd=P('safe_select'))
    sh([PY, 'safe_select.py', '--site', site, '--val', os.path.join(out, 'cand_val.npz'),
        '--test', os.path.join(out, 'cand_test.npz'), '--out', os.path.join(out, 'c3'),
        '--K', s['K'], '--margin', s['margin'], '--d_th', s['d_th'], '--lambdas', s['lambdas']], cwd=P('safe_select'))


# ---------------------------------------------------------------- QCNet
def qcnet_ckpt(c, arm, start=False):
    """체크포인트 파일, 또는 val_minFDE best 를 고를 실험 폴더. start = Contrastive(CP) 의 0~200 best."""
    site = c['site']
    run = P('runs', 'qcnet', site, arm)
    if arm == 'ours':                                      # 배포 가중치만 있다
        own, w = '', 'ours.ckpt'
    elif arm == 'ours13':
        own, w = os.path.join(run, f"round{len(c['rounds']['round_epochs'])}", 'best.ckpt'), 'ours13.ckpt'
    elif start:
        own, w = os.path.join(run, 'best_2of3.ckpt'), 'contrastive_best_2of3.ckpt'
    else:
        own, w = (run if glob.glob(os.path.join(run, 'epoch=*-val_minFDE=*.ckpt')) else ''), f'{arm}.ckpt'
    if own and os.path.exists(own):
        return own
    w = P('weights', 'qcnet', site, w)
    if not os.path.exists(w):
        missing(f'QCNet {site} {arm} 가중치 ({w})')
    return w


def qcnet_train(c, cfg, arm):
    t = c['train']
    script = 'train_qcnet_vanilla.py' if arm == 'vanilla' else os.path.join('contrastive', 'train_qcnet_contrastive.py')
    cmd = [PY, script, '--site', c['site'], '--config', cfg, '--max_epochs', t['max_epochs'],
           '--batch_size', t['batch_size'], '--num_workers', t['num_workers'], '--seed', c['seed'],
           '--out', P('runs', 'qcnet', c['site'], arm)]
    if arm == 'contrastive':
        cmd += ['--snap_epoch', c['contrastive']['snap_epoch']]
    sh(cmd, cwd=P('qcnet'))


def qcnet_vanilla_cp(c, cfg, a):
    out = P('runs', 'qcnet', c['site'], 'vanilla_cp', 'init')
    sh([PY, 'predict_qcnet.py', '--site', c['site'], '--split', 'init', '--ckpt', qcnet_ckpt(c, 'vanilla'),
        '--out', out], cwd=P('qcnet'))
    pred_cp(c['site'], out + '_pred.csv', c['contrastive']['cp_col'].removesuffix('_cp_max'))


def qcnet_ours(c, cfg, a):
    site, r, mi, al = c['site'], c['rounds'], c['mining'], c['allocation']
    out = P('runs', 'qcnet', site, 'ours13')
    sh([PY, os.path.join('ours', 'run_qcnet_ours.py'), '--site', site, '--config', cfg,
        '--start_ckpt', qcnet_ckpt(c, 'contrastive', start=True), '--vanilla_ckpt', qcnet_ckpt(c, 'vanilla'),
        '--out_root', out, '--select', r['select'], '--mine_unit', mi['unit'], '--top_percent', mi['top_percent'],
        '--alloc_gamma', al['gamma'], '--alloc_cap', al['cap'], '--alloc_floor', al['floor'],
        '--batch_size', r['batch_size'], '--num_workers', r['num_workers'], '--gen_env', gen_env(c, out)]
       + round_args(c), cwd=P('qcnet'))


def qcnet_eval(c, cfg, a):
    site, arm = c['site'], a.arm
    ck = qcnet_ckpt(c, arm)
    pre = P('runs', 'qcnet', site, 'eval', arm, f"test_k{c['eval']['num_k']}")
    prefix = f'eval_qcnet_{arm}'
    print(f'[eval] QCNet {site} {arm}: {ck}', flush=True)
    sh([PY, 'predict_qcnet.py', '--site', site, '--split', 'test', '--ckpt', ck, '--out', pre], cwd=P('qcnet'))
    pred_cp(site, pre + '_pred.csv', prefix)
    eval_lab(site, pre + '_metrics.csv', prefix, f'QCNet {site} {arm} (top-{c["eval"]["num_k"]})')


RUN = {
    'tutr': {'prep': prep, 'lcsim': lcsim, 'vanilla_cp': tutr_vanilla_cp, 'ours': tutr_ours, 'eval': tutr_eval,
             'vanilla': lambda c, cfg, a: tutr_train(c, cfg, 'vanilla'),
             'contrastive': lambda c, cfg, a: tutr_train(c, cfg, 'contrastive')},
    'qcnet': {'prep': prep, 'lcsim': lcsim, 'vanilla_cp': qcnet_vanilla_cp, 'ours': qcnet_ours, 'eval': qcnet_eval,
              'vanilla': lambda c, cfg, a: qcnet_train(c, cfg, 'vanilla'),
              'contrastive': lambda c, cfg, a: qcnet_train(c, cfg, 'contrastive')},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='configs/<backbone>_<site>_ours13.json')
    ap.add_argument('--stage', required=True, help='쉼표로 여러 개: ' + ', '.join(STAGES))
    ap.add_argument('--arm', default='ours', choices=ARMS, help='eval 에서 평가할 모델')
    a = ap.parse_args()
    cfg = os.path.abspath(a.config)
    with open(cfg, encoding='utf-8') as f:
        c = json.load(f)
    stages = a.stage.split(',')
    bad = [s for s in stages if s not in STAGES]
    if bad:
        sys.exit(f'모르는 단계 {bad}. 가능: {STAGES}')
    for s in stages:
        print(f'===== {c["backbone"]} {c["site"]} : {s} =====', flush=True)
        RUN[c['backbone']][s](c, cfg, a)


if __name__ == '__main__':
    main()
