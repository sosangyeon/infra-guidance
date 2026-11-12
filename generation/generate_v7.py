"""LCSim 생성: tail 시나리오마다 정해진 개수만큼 롤아웃해 생성본마다 CSV 하나를 쓴다.

  - real guidance : tail 차량 (RG_TAIL_JSON) 은 noised-GT 에서 시작 (λ_tail), 나머지 차량은 순수 노이즈에서 시작
  - 가이던스      : RG_V7_VOCAB 를 주면 엔진이 v7lib.membership_guidance (membership + anchor-pull) 를 쓰고,
                    RG_V7_PHYS=1 이면 물리 가이던스 (RG_MODULES: no-offroad, repeller) 도 함께 건다
  - 생성 개수     : RG_ALLOC_JSON (시나리오 -> 개수) 이 있으면 그 개수, 없으면 RG_NUM_AUG
  - stitch        : 저장 전에 모든 차량의 obs/future 이음새를 v7lib.stitch.stitch_traj 로 보정하고 그 구간 속도·heading 을 다시 계산
  - time shift 는 여기서 하지 않는다 (postprocess_shift.py). 출력의 extrap 열은 0

사용 (generation 폴더에서)
  python generate_v7.py <시나리오 수, 0 = 전체> <출력 폴더 이름> <λ_tail>
환경변수는 tutr/ours/run_ours.py, qcnet/ours/run_qcnet_ours.py 가 넣어 준다:
  LCSIM_DATA (라운드 폴더), RG_PB_ORIG (make_tail_pb.py 출력), RG_TAIL_JSON, RG_ALLOC_JSON, LCSIM_CKPT, LCSIM_MCFG, RG_V7_*

출력: <LCSIM_DATA>/<출력 폴더>/scenario_<sid>/scenario_<sid>_aug<NNN>.csv
컬럼: scenario_id, aug_id, seed, track_id, timestep, observed, extrap,
      position_x, position_y, heading, velocity_x, velocity_y
"""
import sys, os, csv, time, yaml, math, numpy as np, torch
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg")

DATA = os.environ.get("LCSIM_DATA", "").strip()
if not DATA:
    sys.exit("환경변수 LCSIM_DATA 에 생성 데이터 폴더를 지정하세요.")
# tail 시나리오 pb (make_tail_pb.py 출력)
PB_ORIG = os.environ.get("RG_PB_ORIG", "").strip() or \
    os.path.join(DATA, "sim_data", "_gen_pb_orig")     # 준비된 tail 시나리오 pb
# tail 차량 목록 (시나리오 -> track_id 목록)
TAIL_JSON = os.environ.get("RG_TAIL_JSON", "").strip() or \
    os.path.join(DATA, "tail_dict_per_scenario.json")
# 확산 모델 체크포인트 (사이트별: weights/lcsim/<site>/*.ckpt 또는 run_lcsim_site.py 로 학습한 것)
CKPT = os.environ.get("LCSIM_CKPT", "").strip()
if not CKPT:
    sys.exit("환경변수 LCSIM_CKPT 에 LCSim 체크포인트를 지정하세요.")
# 확산 모델 설정: 사이트별 학습(run_lcsim_site.py)은 vel_mean·pca 가 달라 LCSIM_MCFG 로 그 설정을 준다
MCFG = os.environ.get("LCSIM_MCFG", "").strip() or "motion_diff/configs/lis_config.yml"
N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 0     # 0 = 전체
TAG = sys.argv[2] if len(sys.argv) > 2 else "gen_traj_rg_csv"
LAMBDA_TAIL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
OUT_DIR = os.path.join(DATA, TAG)
# 시나리오당 생성 수 (RG_ALLOC_JSON 이 없을 때)
SEEDS = list(range(int(os.environ.get("RG_NUM_AUG", "6"))))
HIST = 10                      # observed = timestep < HIST (LIS num_historical=10)
# 횡 다양성 주입: RG_LAT_OFFSETS="0,-0.8,0.8,..." 설정 시 aug별 LAT_OFFSET 을 순환 적용
#   (RG_MODULES=onlane_offset 과 함께 사용; Engine 이 rollout 마다 새로 생성되므로 env 로 전달)
_OFFS = os.environ.get("RG_LAT_OFFSETS", "").strip()
LAT_OFFSETS = [float(v) for v in _OFFS.split(",")] if _OFFS else None
# 분기(route) 다양성: RG_BRANCH=1 이면 aug별 RG_BRANCH_AUG 를 설정(engine 이 분기 배정).
BRANCH = os.environ.get("RG_BRANCH", "0") == "1"
# 씬 필터: RG_SIDS_JSON=<json path> 이면 그 json 의 키에 있는 씬만 생성(예: branch_dict.json)
SIDS_JSON = os.environ.get("RG_SIDS_JSON", "").strip()

os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, "lcsim/scripts/scenario_converters")
from lcsim.engine import Engine

import json
tail_dict = json.load(open(TAIL_JSON, "r", encoding="utf-8"))
sids = sorted(s for s in tail_dict if os.path.isdir(os.path.join(PB_ORIG, s)))
if SIDS_JSON:
    _keep = set(json.load(open(SIDS_JSON, encoding="utf-8")))
    sids = [s for s in sids if s in _keep]
if N_SAMPLES > 0:
    sids = sids[:N_SAMPLES]
print(f"[gen-rg] {len(sids)} tail samples x {len(SEEDS)} runs "
      f"(lambda_tail={LAMBDA_TAIL}) -> {OUT_DIR}", flush=True)

cfg = yaml.safe_load(open("lcsim/config/example.yml", encoding="utf-8"))
cfg["input"].update(model_path=CKPT, model_config_path=MCFG)
cfg["control"]["step"]["total"] = 40
_GUID = os.environ.get("GUID_ON", "1") == "1"   # GUID_ON=0 이면 gradient guidance off(다양성↑)
_PLAN_INT = int(os.environ.get("RG_PLAN_INTERVAL", "10"))  # 30=플랜1회 전체실행(분기용)
cfg["control"].update(enable_poly_emb=True, enable_plan=True, plan_init_step=10,
                      plan_interval=_PLAN_INT, device="cuda", guidance=_GUID)  # best
# tail-score(TailResidual)용 외부 예측궤적 입력(선택). 파일/컬럼 없으면 engine 이 자동 no-op.
# 환경변수 PRED_DIR 로 오버라이드.
PRED_DIR = os.environ.get("PRED_DIR", os.path.join(DATA, "pred"))
cfg["control"]["real_guidance"] = {
    "tail_json": TAIL_JSON, "lambda_tail": LAMBDA_TAIL, "pred_dir": PRED_DIR,
}
T = cfg["control"]["step"]["total"]
HEADER = ["scenario_id", "aug_id", "seed", "track_id", "timestep", "observed", "extrap",
          "position_x", "position_y", "heading", "velocity_x", "velocity_y"]
from v7lib.stitch import stitch_traj, STITCH_K
ALLOC_JSON = os.environ.get("RG_ALLOC_JSON", "").strip()
ALLOC = {str(k): int(v) for k, v in json.load(open(ALLOC_JSON, encoding="utf-8")).items()} if ALLOC_JSON else None
if ALLOC is not None:
    _miss = [s for s in sids if s not in ALLOC]
    if _miss:
        sys.exit(f"[v7] alloc.json 에 없는 시나리오 {len(_miss)} 개 (예 {_miss[:3]})")
STITCH = os.environ.get("RG_STITCH", "1") == "1"
cfg["control"]["real_guidance"]["lambda_rel"] = float(os.environ.get("RG_LAMBDA_REL", "0.6"))


def stitch_recs(recs):
    """recs [(tid,t,x,y,h,vx,vy)] -> 40 프레임이 다 있는 차량에 stitch, 바뀐 구간(HIST-1..HIST+K) 의 속도·heading 재계산."""
    if not STITCH:
        return recs
    by = {}
    for r in recs:
        by.setdefault(r[0], {})[r[1]] = list(r)
    out = []
    for tid, rows in by.items():
        if len(rows) == T and all(t in rows for t in range(T)):
            p = np.array([[rows[t][2], rows[t][3]] for t in range(T)])
            q = stitch_traj(p, obs=HIST)
            for t in range(HIST, HIST + STITCH_K + 1):
                d = (q[t] - q[t - 1]) / 0.1
                rows[t][2], rows[t][3] = q[t, 0], q[t, 1]
                rows[t][5], rows[t][6] = d[0], d[1]
                if np.hypot(*d) > 1e-3:
                    rows[t][4] = math.atan2(d[1], d[0])
        out.extend(tuple(rows[t]) for t in sorted(rows))
    return out


# DiversityRepel(§5): RG_MODULES 에 div_repel 포함 시 aug 간 실행궤적을 축적해
# 다음 aug 의 Engine 에 전달(aug0=자연 샘플, aug1은 aug0 에서 반발, ...).
_DIVR = "div_repel" in os.environ.get("RG_MODULES", "")


def rollout(data_dir, seed, div_prev=None):
    c = {**cfg, "input": {**cfg["input"], "data_dir": data_dir}}
    torch.manual_seed(seed); eng = Engine(c); torch.manual_seed(seed)
    if div_prev:
        eng._div_prev = div_prev
    rec = []
    for t in range(T):
        for a in eng.agent_manager.agents.values():
            rt = a.runtime
            if rt is None or rt.xy is None or not a.dynamic_flag:
                continue
            xy = np.asarray(rt.xy, dtype=float); h = float(rt.heading); v = float(rt.v)
            if xy.shape != (2,) or not (np.isfinite(xy).all() and math.isfinite(h) and math.isfinite(v)):
                continue
            rec.append((a.id, t, xy[0], xy[1], h, v * math.cos(h), v * math.sin(h)))
        if t < T - 1:
            eng.step()
    return rec


ok = fail = 0; t0 = time.time()
for n, sid in enumerate(sids):
    try:
        d = os.path.join(PB_ORIG, sid)
        sdir = os.path.join(OUT_DIR, f"scenario_{sid}")   # 시나리오별 폴더
        os.makedirs(sdir, exist_ok=True)
        nrows = 0
        div_prev = {}                                     # agent_id -> [ (T,2) per prev aug ]
        seeds = list(range(ALLOC[sid])) if ALLOC is not None else SEEDS
        for aug_id, seed in enumerate(seeds):             # 증강본(seed)당 파일 1개
            if LAT_OFFSETS:
                os.environ["LAT_OFFSET"] = str(LAT_OFFSETS[aug_id % len(LAT_OFFSETS)])
            if BRANCH:
                os.environ["RG_BRANCH_AUG"] = str(aug_id)
            os.environ["RG_V7_AUG_ID"] = str(aug_id)      # membership push 모드가 목표 클러스터를 번갈아 고르는 데 씀
            out_csv = os.path.join(sdir, f"scenario_{sid}_aug{aug_id:03d}.csv")
            recs = stitch_recs(rollout(d, seed, div_prev if (_DIVR and div_prev) else None))
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(HEADER)
                for (tid, t, x, y, h, vx, vy) in recs:
                    w.writerow([sid, aug_id, seed, tid, t, int(t < HIST), 0,
                                f"{x:.4f}", f"{y:.4f}", f"{h:.5f}", f"{vx:.4f}", f"{vy:.4f}"])
                    nrows += 1
            if _DIVR:                                     # 이번 aug 실행궤적 축적
                by_tid = {}
                for (tid, t, x, y, *_rest) in recs:
                    by_tid.setdefault(tid, {})[t] = (x, y)
                for tid, pts in by_tid.items():
                    arr = np.full((T, 2), np.nan, dtype=np.float32)
                    for t, xy in pts.items():
                        arr[t] = xy
                    fin = np.where(np.isfinite(arr[:, 0]))[0]
                    if len(fin) == 0:
                        continue
                    arr[: fin[0]] = arr[fin[0]]           # 결측은 최근접 값으로
                    arr[fin[-1] + 1:] = arr[fin[-1]]
                    for t in range(fin[0] + 1, fin[-1] + 1):
                        if not np.isfinite(arr[t, 0]):
                            arr[t] = arr[t - 1]
                    div_prev.setdefault(tid, []).append(arr)
        ok += 1
        print(f"[{n+1}/{len(sids)}] {sid} -> {len(seeds)} aug files, {nrows} rows "
              f"({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        fail += 1
        print(f"[{n+1}/{len(sids)}] {sid} FAIL {str(e)[:90]}", flush=True)

print(f"\n=== DONE gen-rg: {ok} files, {fail} failed -> {OUT_DIR} ({time.time()-t0:.0f}s) ===", flush=True)
# 실패한 시나리오가 있으면 0 이 아닌 코드로 끝낸다 (모델 로드 실패가 조용히 넘어가지 않게)
sys.exit(1 if fail else 0)
