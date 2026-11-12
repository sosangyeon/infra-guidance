"""Ours — Contrastive(CP) 의 0~200 best 에서 3 라운드(33/33/34 epoch) contrastive + 생성 데이터 post-train.

라운드 r (1, 2, 3):
  1. mine   : 현재 모델로 remain 예측, 이전 라운드 tail 시나리오 제외, 차량별 minADE@K 상위 p % -> tail.json
  2. pb     : tail 시나리오 -> LCSim 입력 (generation/make_tail_pb.py)
  3. alloc  : 시나리오별 생성 개수 배분 (generation/v7lib/rss_alloc.py), 총량 = 시나리오수 × num_aug
              ours13 은 마이닝 minADE 기준 (--alloc_crit minade), 평균 12, 6~24
  4. gen    : LCSim 생성 (generation/generate_v7.py). λ_tail 0.25 / λ_rel 0.6 / λ_head 1.0, plan 1 회(30 스텝), stitch
              가이던스 = 물리(no_offroad_exact + repeller_exact, head 만) + membership + anchor-pull(모든 차량)
              ours13 은 membership 의 tail 목표가 push (--tail_mode push: GT 와 가까운 비-GT 프로토타입으로 민다)
  5. shift  : random time-window shift + pad 마스크 (generation/postprocess_shift.py)
  6. pkl    : 생성 CSV -> gen_r<r>.pkl
  7. cp     : Vanilla(init 만 학습, best.pth) 로 생성 장면 예측 -> gen_r<r>_cp.csv (init 의 vanilla_cp_max 와 같은 식, gen_cp.py)
  8. train  : init + 생성분(1..r) 으로 E_r epoch (train_ours_round.py: 라운드 학습 + CP contrastive)
              샘플 가중치 max(0.7^(r−a), 0.5). --gen_frac f > 0 이면 생성분 가중치 합 비율을 f 로 고정
              라운드 best = round<r>/best.pth
현재 모델: 1 라운드는 시작 모델(Contrastive(CP) 실험의 best_2of3.pth = 0~200 best), 이후는 직전 라운드 best.
최종 모델 = 마지막 라운드 best.

생성기(LCSim)는 generation/runs/lcsim_<site> 에 다시 학습한 것이 있으면 그것을, 없으면 weights/lcsim/<site> 의 것을 쓴다.

보통은 저장소 루트의 run.py 가 configs/tutr_<site>_ours13.json 값을 아래 인자로 바꿔 부른다.
  python run.py --config configs/tutr_ss_ours13.json --stage ours
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # 저장소 루트
GEN = os.path.join(ROOT, "generation")
DATA_TUTR = os.path.join(ROOT, "data", "tutr")
SCN = os.path.join(ROOT, "data", "scenarios")


def run(cmd, cwd=HERE, env=None, log=None):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    e = dict(os.environ, PYTHONIOENCODING="utf-8", **(env or {}))
    if log:
        with open(log, "a", encoding="utf-8") as f:
            subprocess.run([str(c) for c in cmd], cwd=cwd, env=e, check=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run([str(c) for c in cmd], cwd=cwd, env=e, check=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=["ss", "ny"])
    ap.add_argument("--hp_config", required=True, help="모델·학습·contrastive 값이 있는 설정 (configs/*.json)")
    ap.add_argument("--start_run_dir", required=True, help="Contrastive(CP) 실험 폴더 (run_motion_modes.pkl 위치)")
    ap.add_argument("--start_ckpt", default="best_2of3.pth", help="start_run_dir 기준 (0~200 best)")
    ap.add_argument("--vanilla_run_dir", required=True, help="생성 장면 CP 계산용 Vanilla 실험 폴더")
    ap.add_argument("--vanilla_ckpt", default="best.pth")
    ap.add_argument("--vanilla_hp_config", default=None, help="기본: --hp_config (모델 크기·이웃 반경 같음)")
    ap.add_argument("--cp_col", default="vanilla_cp_max")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--round_epochs", default="33,33,34")
    ap.add_argument("--epoch_offset", type=int, default=200)
    ap.add_argument("--total_epochs", type=int, default=300)
    ap.add_argument("--pool_split", default="remain")
    ap.add_argument("--top_percent", type=float, default=1.0, help="tail = 차량별 minADE 상위 p %%")
    ap.add_argument("--mine_k", type=int, default=20)
    ap.add_argument("--num_aug", type=int, default=6, help="시나리오당 평균 생성 수 (배분 총량)")
    ap.add_argument("--rss_gamma", type=float, default=2.0)
    ap.add_argument("--rss_cap", type=int, default=12)
    ap.add_argument("--rss_floor", type=int, default=1)
    ap.add_argument("--rss_frame", default="world", choices=["world", "heading"],
                    help="RSS 종/횡 좌표계. world = 지도 좌표, heading = tail 차량 진행방향")
    ap.add_argument("--gen_frac", type=float, default=-1,
                    help="> 0 이면 라운드 학습에 train_ours_round --gen_frac 적용 (생성분 가중치 합 비율 고정)")
    ap.add_argument("--alloc_crit", default="rss", choices=["rss", "minade"],
                    help="생성 개수 배분 기준. rss = RSS 난이도, minade = 마이닝 minADE (ours13)")
    ap.add_argument("--tail_mode", default="keep", choices=["keep", "push"],
                    help="membership 의 tail 목표. keep = GT 클러스터 보존, push = 이웃 희귀 클러스터로 밀기 (ours13)")
    ap.add_argument("--push_alt", type=int, default=3, help="push 후보 수 (GT 와 가까운 비-GT 프로토타입)")
    ap.add_argument("--lambda_rel", type=float, default=0.6)
    ap.add_argument("--vocab", default=None, help="기본: generation/vocab/c2_lcsim_v6_<site>.pt")
    ap.add_argument("--lambda_tail", type=float, default=0.25)
    ap.add_argument("--lcsim_ckpt", default=None, help="기본: generation/runs/lcsim_<site> 의 마지막 ckpt, 없으면 weights/lcsim/<site>")
    ap.add_argument("--lcsim_cfg", default=None, help="기본: generation/motion_diff/configs/lis_<site>_config.yml")
    ap.add_argument("--gen_env", default=None, help="생성 환경변수를 덮어쓸 JSON 파일 ({이름: 값})")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--input_mode", default="xy_vxvy_axay")
    ap.add_argument("--output_mode", default="xy_vxvy")
    ap.add_argument("--loss_weights", default="1,1,1,1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rounds", default=None, help="돌릴 라운드 (기본: 전부, 예 1,2)")
    return ap.parse_args()


def lcsim_ckpt(site):
    c = sorted(glob.glob(os.path.join(GEN, "runs", f"lcsim_{site}", "logs", "*", "checkpoints", "*.ckpt")),
               key=os.path.getmtime)
    c = c or sorted(glob.glob(os.path.join(ROOT, "weights", "lcsim", site, "*.ckpt")))
    if not c:
        sys.exit(f"LCSim 체크포인트 없음 ({site}): python download_weights.py 로 받거나 "
                 f"python run.py --config <json> --stage lcsim 으로 학습하세요")
    return c[-1]


def mine(args, r, rd, cur_run_dir, cur_ckpt):
    out = os.path.join(rd, "tail.json")
    if os.path.exists(out):
        return
    pred = os.path.join(rd, f"mine_{args.pool_split}")
    if not os.path.exists(pred + "_metrics.csv"):
        run([sys.executable, "predict.py", "--dataset_path", os.path.join(DATA_TUTR, args.site),
             "--dataset_name", args.site, "--split", args.pool_split, "--hp_config", args.hp_config,
             "--run_dir", cur_run_dir, "--ckpt", cur_ckpt, "--input_mode", args.input_mode,
             "--output_mode", args.output_mode, "--num_k", args.mine_k, "--out", pred])
    m = pd.read_csv(pred + "_metrics.csv")
    used = set()
    for q in range(1, r):
        used |= set(json.load(open(os.path.join(args.out_root, f"round{q}", "tail.json"), encoding="utf-8")))
    pool = m[~m["scenario_id"].isin(used)]
    n = max(1, math.ceil(len(pool) * args.top_percent / 100))
    top = pool.nlargest(n, "minade")
    tail = {sid: sorted(int(t) for t in g["track_id"]) for sid, g in top.groupby("scenario_id")}
    json.dump(tail, open(out, "w", encoding="utf-8"), indent=1)
    print(f"[mine r{r}] 풀 {len(pool):,} 샘플 (제외 {len(used)} 시나리오) -> tail {n} 차량 / {len(tail)} 시나리오, "
          f"minADE{args.mine_k} >= {top['minade'].min():.3f} (풀 평균 {pool['minade'].mean():.3f})", flush=True)


def main():
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)
    epochs = [int(e) for e in args.round_epochs.split(",")]
    modes = os.path.join(args.start_run_dir, "run_motion_modes.pkl")
    ckpt = args.lcsim_ckpt or lcsim_ckpt(args.site)
    cfg = args.lcsim_cfg or os.path.join(GEN, "motion_diff", "configs", f"lis_{args.site}_config.yml")
    vocab = args.vocab or os.path.join(GEN, "vocab", f"c2_lcsim_v6_{args.site}.pt")
    if not os.path.exists(vocab):
        sys.exit(f"C2 어휘 없음: {vocab} (generation/build_c2_vocab.py --site {args.site})")
    gen_env = {
        "GUID_ON": "1", "RG_PLAN_INTERVAL": "30", "RG_V7_VOCAB": os.path.abspath(vocab),
        "RG_LAMBDA_REL": str(args.lambda_rel), "RG_STITCH": "1",
        "RG_V7_PHYS": "1", "RG_GUID_HEAD_ONLY": "1", "REPELLER_RADIUS": "2.5",
        "NO_OFFROAD_EPS": "0.0", "RG_MODULES": "no_offroad_exact,repeller_exact", "RG_TAIL_REALISM": "0",
        "LCSIM_CKPT": os.path.abspath(ckpt), "LCSIM_MCFG": os.path.abspath(cfg),
        "RG_V7_TAIL_MODE": args.tail_mode, "RG_V7_PUSH_ALT": str(args.push_alt),
    }
    if args.gen_env:
        gen_env.update({k: str(v) for k, v in json.load(open(args.gen_env, encoding="utf-8")).items()})
    print(f"[run_ours] {args.site} | vocab {vocab} | 물리+membership({args.tail_mode}), time shift, λ_rel {args.lambda_rel} | "
          f"alloc {args.alloc_crit} base {args.num_aug} [{args.rss_floor},{args.rss_cap}] frame {args.rss_frame} | gen_frac {args.gen_frac}", flush=True)
    print(f"[run_ours] {args.site} | start {args.start_run_dir}/{args.start_ckpt} | LCSim {ckpt}", flush=True)

    for r in ([int(x) for x in args.rounds.split(",")] if args.rounds else range(1, len(epochs) + 1)):
        rd = os.path.abspath(os.path.join(args.out_root, f"round{r}"))
        os.makedirs(rd, exist_ok=True)
        if r == 1:
            cur_run_dir, cur_ckpt = args.start_run_dir, args.start_ckpt
        else:
            cur_run_dir, cur_ckpt = os.path.join(args.out_root, f"round{r - 1}"), "best.pth"
        if os.path.exists(os.path.join(rd, "best.pth")) and os.path.exists(os.path.join(rd, "done")):
            print(f"[round {r}] 이미 끝남", flush=True)
            continue

        mine(args, r, rd, cur_run_dir, cur_ckpt)
        if not os.path.exists(os.path.join(rd, "pb")):
            run([sys.executable, "make_tail_pb.py", "--site", args.site, "--tail_json", os.path.join(rd, "tail.json"),
                 "--out_root", os.path.join(rd, "pb")], cwd=GEN)
        alloc = os.path.join(rd, "alloc.json")
        if not os.path.exists(alloc):
            run([sys.executable, "-m", "v7lib.rss_alloc", "--site", args.site, "--scenarios", SCN,
                 "--tail_json", os.path.join(rd, "tail.json"), "--out", alloc, "--base", args.num_aug,
                 "--gamma", args.rss_gamma, "--cap", args.rss_cap, "--floor", args.rss_floor,
                 "--frame", args.rss_frame, "--crit", args.alloc_crit,
                 "--metrics_csv", os.path.join(rd, f"mine_{args.pool_split}_metrics.csv")], cwd=GEN)
        env = dict(gen_env, LCSIM_DATA=rd, RG_PB_ORIG=os.path.join(rd, "pb"),
                   RG_TAIL_JSON=os.path.join(rd, "tail.json"), RG_ALLOC_JSON=alloc)
        if not os.path.exists(os.path.join(rd, "gen.done")):
            run([sys.executable, "generate_v7.py", "0", "gen", str(args.lambda_tail)], cwd=GEN, env=env,
                log=os.path.join(rd, "gen.log"))
            open(os.path.join(rd, "gen.done"), "w").close()
        if not os.path.exists(os.path.join(rd, "shift.done")):
            run([sys.executable, "postprocess_shift.py", "gen", "gen_shift"], cwd=GEN, env=env,
                log=os.path.join(rd, "gen.log"))
            open(os.path.join(rd, "shift.done"), "w").close()
        gen_pkl = os.path.join(rd, f"gen_r{r}")
        if not os.path.exists(gen_pkl + ".pkl"):
            run([sys.executable, "gen_to_pkl.py", "--gen_dir", os.path.join(rd, "gen_shift"), "--out", gen_pkl])
        if not os.path.exists(gen_pkl + "_cp.csv"):
            vpred = gen_pkl + "_vanilla"
            if not os.path.exists(vpred + "_pred.csv"):
                run([sys.executable, "predict.py", "--dataset_path", rd, "--dataset_name", "gen", "--split", f"r{r}",
                     "--hp_config", args.vanilla_hp_config or args.hp_config, "--run_dir", args.vanilla_run_dir,
                     "--ckpt", args.vanilla_ckpt, "--input_mode", args.input_mode, "--output_mode", args.output_mode,
                     "--num_k", 1, "--out", vpred])
            run([sys.executable, "gen_cp.py", "--pred_csv", vpred + "_pred.csv", "--meta_csv", gen_pkl + "_meta.csv",
                 "--out", gen_pkl + "_cp.csv"])

        gen = ",".join(f"{q}:{os.path.abspath(os.path.join(args.out_root, f'round{q}', f'gen_r{q}.pkl'))}"
                       for q in range(1, r + 1))
        run([sys.executable, "train_ours_round.py", "--dataset_path", os.path.join(DATA_TUTR, args.site),
             "--dataset_name", args.site, "--hp_config", args.hp_config, "--base_splits", "init",
             "--val_split", "val", "--gen", gen, "--round", r,
             "--cp_table", os.path.join(SCN, f"manifest_{args.site}.csv"), "--cp_col", args.cp_col,
             "--alpha", args.alpha, "--clip", args.clip,
             "--init_ckpt", os.path.join(cur_run_dir, cur_ckpt), "--motion_modes", modes, "--epochs", epochs[r - 1],
             "--epoch_offset", args.epoch_offset + sum(epochs[:r - 1]), "--total_epochs", args.total_epochs,
             "--out_dir", rd, "--input_mode", args.input_mode, "--output_mode", args.output_mode,
             "--loss_weights", args.loss_weights, "--seed", args.seed]
            + (["--gen_frac", args.gen_frac] if args.gen_frac > 0 else []))
        open(os.path.join(rd, "done"), "w").close()

    print(f"[run_ours] 최종 모델: {os.path.join(args.out_root, f'round{len(epochs)}', 'best.pth')}", flush=True)


if __name__ == "__main__":
    main()
