"""후처리(논문충실판): random time-window shift **만** — GALTraj Sec 3.5 / Eq.5-6.

`_postproc_bcts.py` 와의 차이 = 논문에 없는 2건을 제거한 것:
  ① 미래 구간 자전거모델 재적분  제거  (논문·보충자료에 bicycle/kinematic 언급 자체가 없음)
  ② horizon 끝 δt 스텝 EXTEND     제거  (논문 요구는 zero-pad + 마스킹)

근거(감사): δt=0 증강본만 골라 잰 자전거 보정량이 종점 0.68~3.44 m 로
tail-score 가 만든 다양성(1.84 m)보다 커서, 학습 데이터가 **확산 출력이 아니라 적분기
출력**이 되어 ablation 자체가 무효가 된다.

동작:
  δt 를 그대로 샘플링(같은 시드 → `_postproc_bcts.py` 와 **동일한 δt** 배정)한 뒤
  t' = t - δt 로 윈도우만 민다. t'+δt >= T 인 뒤쪽 δt 스텝은 원본 상태가 없으므로
  **pad 행을 만들고 `extrap=1` 로 표시**한다 → `convert_to_tutr.py` 가 extrap 컬럼을 보존하고
  `dataloader.py`(228/271/279) 가 `future_valid` 마스크로 바꿔 회귀 loss 에서 제외한다.
  = Eq.6 의 "zero-pad **and** mask" 두 가지를 모두 구현.

  ※ 검증 워크플로 blocker(4건) 반영. 처음엔 pad 행을 **생략**했는데,
    TUTR `Dataloader.load`(dataloader.py:252-254) 가 40프레임 미만 파일을 통째로 버려
    (`return None`, 로그도 예외도 없음) **δt>0 인 증강본 93.3% 가 조용히 소실**됐다.
    실측 검증 60개 중 54개 소실 확인. δt=0 확률은 1-exp(-1/29/0.5)≈6.7% 뿐이다.

  pad 값은 **마지막 유효 상태 유지**(속도 0). 논문 문구는 "zero-pad" 이나 그것은 에이전트
  기준 정규화 좌표에서의 0 을 뜻한다. 여기 CSV 는 **절대 월드 좌표**라 문자 그대로 0 을 넣으면
  K-City 지도 원점(수백 m 밖)이 되어 논문 의도와 오히려 멀어진다. 마스킹이 정상 동작하면
  값은 정의상 무의미하고, 만에 하나 마스킹이 새더라도 오차가 유계로 남는 fail-safe 선택이다.

사용: python scratch/eval/postprocess_shift.py <gen_folder> [out_folder] [dt|rand]

"""
import csv, os, sys, glob, zlib, collections
import numpy as np

DATA = os.environ.get("LCSIM_DATA", ".")   # 생성 데이터 폴더. 미지정이면 현재 폴더 기준
HIST, T = 10, 40
TF = T - HIST                          # future length 30
SHIFT_MULT = TF / 2                    # = 15 (논문 T/2)

folder = sys.argv[1] if len(sys.argv) > 1 else "gen_traj_rg_band"
IN = folder if os.path.isabs(folder) else os.path.join(DATA, folder)
_out = sys.argv[2] if len(sys.argv) > 2 else IN + "_shift"
OUT = _out if os.path.isabs(_out) else os.path.join(DATA, _out)
_dtarg = sys.argv[3] if len(sys.argv) > 3 else "rand"
FIXED_DT = None if _dtarg.lower() == "rand" else int(_dtarg)
os.makedirs(OUT, exist_ok=True)
HEADER = ["scenario_id", "aug_id", "seed", "track_id", "timestep", "observed", "extrap",
          "position_x", "position_y", "heading", "velocity_x", "velocity_y"]


def sample_dt(sid, aug):
    """scene(aug)별 random time-window shift δt. 시드 규칙은 _postproc_bcts.py 와 동일."""
    if FIXED_DT is not None:
        return FIXED_DT
    seed = (zlib.crc32(sid.encode()) + int(aug)) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    raw = rng.exponential(scale=0.5)
    norm = raw / (raw + 1.0)
    return int(round(norm * SHIFT_MULT))


def timeshift(rows, sid, aug, seed, dt):
    """윈도우만 δt 앞으로. 뒤쪽 δt 스텝은 pad 행 + extrap=1 (Eq.6 zero-pad and mask).

    길이는 항상 정확히 T(=40) 프레임 — TUTR Dataloader 의 완전트랙 요구를 만족시켜야
    파일이 통째로 버려지지 않는다(위 docstring 의 blocker 참조).
    """
    byt = {int(r["timestep"]): r for r in rows}
    tid = rows[0]["track_id"]
    out, last = [], None
    for tp in range(T):
        s = byt.get(tp + dt)
        if s is not None:
            last = s
            out.append({"scenario_id": sid, "aug_id": aug, "seed": seed,
                        "track_id": s["track_id"], "timestep": tp,
                        "observed": int(tp < HIST), "extrap": 0,
                        "position_x": s["position_x"], "position_y": s["position_y"],
                        "heading": s["heading"], "velocity_x": s["velocity_x"],
                        "velocity_y": s["velocity_y"]})
        else:                          # tp + dt >= T (또는 원본 결측) → pad + mask
            out.append({"scenario_id": sid, "aug_id": aug, "seed": seed,
                        "track_id": last["track_id"] if last else tid, "timestep": tp,
                        "observed": int(tp < HIST), "extrap": 1,
                        "position_x": last["position_x"] if last else "0.0000",
                        "position_y": last["position_y"] if last else "0.0000",
                        "heading": last["heading"] if last else "0.00000",
                        "velocity_x": "0.0000", "velocity_y": "0.0000"})
    return out


sdirs = sorted(d for d in glob.glob(os.path.join(IN, "scenario_*")) if os.path.isdir(d))
mode = f"fixed δt={FIXED_DT}" if FIXED_DT is not None else f"random exp(0.5)·{SHIFT_MULT:.0f}"
print(f"[shift] {len(sdirs)} scenes  shift={mode}  (자전거 재적분/EXTEND 없음)  -> {OUT}", flush=True)
dts = []
n_empty = n_pad = n_row = 0
for d in sdirs:
    sid = os.path.basename(d).replace("scenario_", "")
    so = os.path.join(OUT, f"scenario_{sid}"); os.makedirs(so, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(d, "scenario_*aug*.csv"))):
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        if not rows:                   # 헤더만 있는 빈 CSV — 건너뛴다(IndexError 방지)
            print(f"  [skip] 빈 파일 {os.path.basename(f)}", flush=True)
            n_empty += 1
            continue
        by = collections.defaultdict(list)
        for r in rows:
            by[r["track_id"]].append(r)
        aug = rows[0]["aug_id"]; seed = rows[0]["seed"]
        dt = sample_dt(sid, aug); dts.append(dt)
        out_rows = []
        for tid, tr in by.items():
            out_rows += timeshift(tr, sid, aug, seed, dt)
        out_rows.sort(key=lambda r: (int(r["track_id"]), int(r["timestep"])))
        n_row += len(out_rows); n_pad += sum(r["extrap"] for r in out_rows)
        with open(os.path.join(so, os.path.basename(f)), "w", newline="", encoding="utf-8") as fo:
            w = csv.DictWriter(fo, fieldnames=HEADER); w.writeheader(); w.writerows(out_rows)
if not dts:
    print(f"[warn] 처리한 aug CSV 가 0개다 — 입력 경로 확인: {IN}", flush=True)
    sys.exit(1)
dts = np.array(dts)
print(f"[done] -> {OUT}  | δt 분포: mean={dts.mean():.2f} min={dts.min()} max={dts.max()} "
      f"(0={int((dts==0).sum())}개)  | 유효 미래길이 = {TF}-δt", flush=True)
print(f"[check] 행 {n_row} (pad/extrap=1 {n_pad} = {100*n_pad/n_row:.1f}%) · 빈 파일 {n_empty}개 · "
      f"트랙당 프레임은 항상 {T} (TUTR 완전트랙 요구)", flush=True)
