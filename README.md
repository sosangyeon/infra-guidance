# infra-guidance

인프라 센서로 수집한 교차로 궤적 데이터 LIS-SS, LIS-NY 에서 차량 궤적 예측 모델의 long-tail 오차를 줄이는 코드이다.
예측 백본은 TUTR 와 QCNet 두 가지이고, 아래 세 가지 학습 방법을 같은 조건으로 비교한다.

| 이름 | 내용 |
|---|---|
| Vanilla | 백본을 init 분할로 300 epoch 학습 |
| Contrastive(CP) | Vanilla 예측의 충돌 확률 (CP) 을 난이도로 쓰는 contrastive 손실을 더해 300 epoch 학습 |
| Ours | Contrastive(CP) 의 0~200 epoch best 에서 시작해 100 epoch 을 3 라운드 (33, 33, 34) 로 이어서 학습. 라운드마다 remain 분할에서 오차가 큰 tail 장면을 찾고, 확산 모델 LCSim 에 membership 가이던스를 걸어 그 장면을 다시 생성해 학습에 더한다 |

## 방법

Ours 는 CP contrastive 학습, membership 가이던스, RSS 재분배, 안전 궤적 선택의 네 요소로 이루어진다.

### CP contrastive

Vanilla 의 top-1 예측으로 같은 장면 이웃 차량과의 충돌 확률 (CP) 을 계산해 샘플마다 난이도로 붙인다.
학습할 때는 배치 안에서 CP 차이가 문턱보다 작은 샘플 쌍을 양성, 큰 쌍을 음성으로 삼는 InfoNCE 형 contrastive 손실을 예측 손실에 더한다.
위험도가 비슷한 장면끼리 표현이 모이므로 충돌 위험이 큰 장면을 따로 구분해 배우게 된다.
코드는 `tutr/contrastive/`, `qcnet/contrastive/` 이고 값은 설정 파일의 `contrastive` 절에 있다.

### Membership 가이던스

학습셋 궤적을 k-means 로 묶은 기동 프로토타입을 쓴다. 차량마다 목표 프로토타입을 정하고, 생성 궤적의 임베딩이 목표 프로토타입에 다른 프로토타입보다 여유 δ 이상 가깝도록 하는 hinge 손실의 gradient 를 확산 샘플링의 마지막 단계에 건다.
주변 차량의 목표는 GT 가 속한 프로토타입이고, 여유를 어긴 차량에는 GT 쪽으로 당기는 anchor 항을 더해 원래 기동을 지킨다.
tail 차량의 목표는 GT 와 가까운 프로토타입 가운데 학습셋에서 드문 것으로 바꿔 (`tail_mode: push`), 실제와 비슷하지만 드문 기동을 만든다.
도로 이탈 방지와 차량 간 거리 유지의 물리 가이던스를 함께 건다.
코드는 `generation/v7lib/membership_guidance.py` 이고 값은 `generation` 절에 있다.

### RSS 재분배

tail 장면마다 생성 개수를 다르게 준다. tail 차량과 이웃 차량 사이의 RSS 안전거리 위반 정도를 tail 차량의 진행방향 좌표계에서 종방향과 횡방향으로 계산해 장면 난이도로 삼는다.
총 생성 개수는 장면 수의 12 배로 고정하고, 장면별 개수를 1 + γ·(정규화한 난이도) 에 비례해 최소 6 개, 최대 24 개 사이로 나눈다.
코드는 `generation/v7lib/rss_alloc.py` 이고 값은 `allocation` 절에 있다. 배포한 설정 파일은 `crit: minade` 로 마이닝 minADE 에 비례해 나누며, `crit: rss` 로 바꾸면 RSS 난이도로 나눈다.

### 안전 궤적 선택

예측 후보 20 개 가운데 주행가능영역을 여유 3.5 m 이상 벗어난 후보를 지우고, 모델 확률에서 이웃 예측과의 CP 를 λ_c 배 뺀 점수 순으로 6 개를 고른다.
λ_c 와 후보 채우기 여부는 val 분할에서만 고르고, 정답은 지표 계산에만 쓴다. TUTR 평가에 적용한다.
코드는 `safe_select/` 이고 값은 `eval` 절에 있다.

입력은 관측 10 프레임 (1 초), 출력은 예측 30 프레임 (3 초) 이고 10 Hz 이다. 모든 학습은 seed 1 이고 모델 선택은 val 분할로만 한다.

## 결과

test 분할, K = 6 이다. 저장소의 가중치로 `--stage eval` 을 돌리면 아래 숫자가 나온다. 모델은 `--arm` 으로 고른다.

TUTR 는 안전 궤적 선택 (`safe_select/`, 후보 20 개에서 6 개) 을 거친 값이다.

| 데이터 | 방법 | Top1% [m] | VaR99.9 [m] | FPR5 [%] | minADE [m] | minFDE [m] |
|---|---|---|---|---|---|---|
| LIS-SS | Vanilla | 3.158 | 3.851 | 2.517 | 0.630 | 1.372 |
| LIS-SS | Contrastive(CP) | 3.120 | 3.716 | 2.492 | 0.625 | 1.359 |
| LIS-SS | Ours (`ours`) | 3.084 | 3.677 | 2.400 | 0.619 | 1.344 |
| LIS-SS | Ours (`ours13`) | 3.169 | 3.783 | 2.597 | 0.628 | 1.365 |
| LIS-NY | Vanilla | 4.674 | 5.054 | 8.146 | 0.970 | 2.208 |
| LIS-NY | Contrastive(CP) | 4.747 | 5.074 | 9.003 | 0.977 | 2.235 |
| LIS-NY | Ours (`ours`) | 4.241 | 4.855 | 7.262 | 0.913 | 2.099 |
| LIS-NY | Ours (`ours13`) | 4.082 | 4.688 | 6.860 | 0.901 | 2.060 |

QCNet 은 모델이 낸 6 개를 그대로 평가한 값이다.

| 데이터 | 방법 | Top1% [m] | minADE [m] | minFDE [m] |
|---|---|---|---|---|
| LIS-SS | Vanilla | 2.799 | 0.455 | 0.860 |
| LIS-SS | Contrastive(CP) | 2.816 | 0.466 | 0.876 |
| LIS-SS | Ours (`ours`) | 2.523 | 0.436 | 0.795 |
| LIS-SS | Ours (`ours13`) | 2.669 | 0.450 | 0.838 |
| LIS-NY | Vanilla | 6.811 | 1.008 | 2.242 |
| LIS-NY | Contrastive(CP) | 7.162 | 1.007 | 2.255 |
| LIS-NY | Ours (`ours`) | 5.979 | 0.897 | 1.969 |
| LIS-NY | Ours (`ours13`) | 6.484 | 0.908 | 1.987 |

지표의 뜻은 다음과 같다.

| 지표 | 뜻 |
|---|---|
| Top1% | minADE 가 큰 순서로 상위 1 % 샘플의 minADE 평균 |
| VaR99.9 | minADE 의 99.9 % 분위수 |
| FPR5 | minFDE 가 5 m 를 넘는 샘플의 비율 |
| minADE, minFDE | K 개 후보 중 최소 평균 오차, 최소 최종 오차의 전체 평균 |

## 설치

Python 3.11, CUDA 11.8, Windows 10 에서 확인했다.

```
conda create -n infra python=3.11
conda activate infra
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
pip install torch_geometric==2.5.3
pip install -r requirements.txt
```

## 데이터

이 저장소에는 데이터가 없다. 데이터를 읽고 나누고 변환하는 코드만 있다.
원본 궤적 CSV 와 지도 json 을 `data/` 아래에 놓는 방법, 열 이름, 분할 규칙, 분할별 개수는 [data/README.md](data/README.md) 에 적었다.
원본을 놓은 뒤 아래 한 줄이 두 백본의 입력을 모두 만든다. 사이트마다 한 번만 돌리면 된다.

```
python run.py --config configs/tutr_ss_ours13.json --stage prep
python run.py --config configs/tutr_ny_ours13.json --stage prep
```

## 가중치

TUTR 가중치와 LCSim 의 PCA 파일은 작아서 `weights/` 에 들어 있다.
QCNet 과 LCSim 체크포인트는 Release 에 있고 아래 명령으로 받는다.

```
python download_weights.py                 # 전부
python download_weights.py --site ny       # 한 사이트만
python download_weights.py --only lcsim    # TUTR 만 쓸 때
```

| 위치 | 내용 |
|---|---|
| `weights/tutr/<site>/vanilla/best.pth` | Vanilla |
| `weights/tutr/<site>/contrastive/best.pth` | Contrastive(CP), 0~300 epoch best |
| `weights/tutr/<site>/contrastive/best_2of3.pth` | Contrastive(CP), 0~200 epoch best. Ours 의 시작점 |
| `weights/tutr/<site>/ours/best.pth` | Ours. 논문 결과표를 재현하는 가중치 |
| `weights/tutr/<site>/ours13/best.pth` | Ours. 이 저장소의 `ours` 단계로 다시 학습한 가중치 |
| `weights/tutr/<site>/<방법>/run_motion_modes.pkl` | 그 모델이 쓰는 motion mode (k-means 중심) |
| `weights/qcnet/<site>/{vanilla,contrastive,contrastive_best_2of3,ours,ours13}.ckpt` | QCNet, 뜻은 TUTR 와 같다 |
| `weights/lcsim/<site>/lcsim_<site>.ckpt`, `pca.pkl` | 생성에 쓰는 LCSim 확산 모델 |

## 실행

모든 단계는 `run.py` 하나로 돌린다. 설정 파일 하나가 백본 하나와 사이트 하나의 실험이다.

```
python run.py --config configs/<backbone>_<site>_ours13.json --stage <단계>[,<단계>...] [--arm vanilla|contrastive|ours|ours13]
```

받은 가중치로 평가만 하려면 `prep` 과 `eval` 만 돌리면 된다.

```
python run.py --config configs/tutr_ny_ours13.json --stage eval                    # Ours (ours)
python run.py --config configs/tutr_ny_ours13.json --stage eval --arm ours13
python run.py --config configs/tutr_ny_ours13.json --stage eval --arm vanilla
python run.py --config configs/qcnet_ny_ours13.json --stage eval --arm contrastive
```

처음부터 다시 학습하려면 단계를 순서대로 준다.

```
python run.py --config configs/tutr_ny_ours13.json --stage prep,vanilla,vanilla_cp,contrastive,lcsim,ours,eval
```

| 단계 | 하는 일 | 출력 |
|---|---|---|
| `prep` | 원본 CSV 를 40 프레임 시나리오로 자르고, 날짜로 분할하고, GT 의 CP 를 계산하고, TUTR 와 QCNet 입력을 만든다 | `data/scenarios`, `data/tutr`, `data/qcnet` |
| `vanilla` | Vanilla 학습 | `runs/<backbone>/<site>/vanilla` |
| `vanilla_cp` | Vanilla 의 top-1 예측으로 장면별 CP 를 계산해 manifest 에 넣는다. Contrastive 의 난이도이다 | `manifest_<site>.csv` 의 `*_cp_max` 열 |
| `contrastive` | Contrastive(CP) 학습. 200 epoch 시점에 0~200 best 를 따로 저장한다 | `runs/<backbone>/<site>/contrastive` |
| `lcsim` | LCSim 확산 모델 학습과 기동 어휘 만들기. 받은 LCSim 가중치를 쓰면 건너뛴다 | `generation/runs/lcsim_<site>`, `generation/vocab` |
| `ours` | 3 라운드 반복. tail 마이닝, 생성, 시간 창 이동, 생성 장면 CP 계산, 학습 | `runs/<backbone>/<site>/ours13/round<r>` |
| `eval` | test 예측, CP 계산, 지표 출력. TUTR 는 안전 궤적 선택까지 한다 | `runs/<backbone>/<site>/eval/<arm>` |

`runs/` 에 직접 학습한 모델이 있으면 그것을 쓰고, 없으면 `weights/` 의 것을 쓴다.
그래서 `ours` 단계만 따로 돌리면 받은 Vanilla, Contrastive(CP), LCSim 가중치에서 Ours 학습만 다시 한다. 그 결과는 `--arm ours13` 으로 평가한다.
`eval` 과 `vanilla_cp` 는 `data/scenarios/manifest_<site>.csv` 에 CP 열을 더한다.

## 설정 파일

`configs/` 의 json 네 개가 논문 실험의 모든 값을 담는다. 값을 바꾸려면 json 을 복사해 고치고 `--config` 로 준다.

| 절 | 내용 |
|---|---|
| `data`, `model`, `train`, `optimizer` | 입출력 길이, 백본 크기, 학습률, 배치, epoch |
| `contrastive` | contrastive 손실의 가중치, 온도, 양성과 음성의 CP 차이 문턱, 난이도로 쓰는 manifest 열 |
| `rounds` | 라운드별 epoch, 시작 epoch, 생성 데이터 가중치 감쇠 (`alpha`, `clip`), 한 epoch 에서 생성 데이터가 차지하는 비율 (`gen_frac`) |
| `mining` | tail 을 찾는 분할과 비율. 차량별 minADE 상위 1 % |
| `allocation` | 장면별 생성 개수. 평균 12, 최소 6, 최대 24. `crit` 이 `rss` 면 RSS 난이도, `minade` 면 마이닝 minADE 에 비례 |
| `generation` | LCSim 가이던스. tail 차량과 주변 차량의 세기 (`lambda_tail`, `lambda_rel`), tail 목표 (`tail_mode: push`), membership 가이던스, 물리 가이던스 (도로 이탈 방지, 차량 간 거리) |
| `eval` | K, 안전 궤적 선택의 후보 수와 주행가능영역 여유, λ_c 후보 |

TUTR 와 QCNet 의 Ours 는 `allocation`, `generation` 절이 같다. 다른 값은 `rounds.gen_frac` 으로 TUTR 0.1 (상한), QCNet 0.5 이다.

## 폴더

```
run.py                  단계별 실행기
download_weights.py     Release 의 큰 가중치 받기
configs/                실험 설정 json
data/                   데이터를 놓는 자리 (README 만 들어 있다)
data_prep/              시나리오 자르기, 날짜 분할, CP 계산, TUTR / QCNet 입력 만들기, 지표
tutr/vanilla/           TUTR Vanilla 학습과 예측
tutr/contrastive/       TUTR Contrastive(CP) 학습
tutr/ours/              TUTR Ours 라운드 (run_ours.py)
qcnet/                  QCNet 모델, Vanilla 학습, 예측
qcnet/contrastive/      QCNet Contrastive(CP) 학습
qcnet/ours/             QCNet Ours 라운드 (run_qcnet_ours.py)
generation/             LCSim 학습 (run_lcsim_site.py), 가이던스 생성 (generate_v7.py, v7lib/), 시간 창 이동
safe_select/            안전 궤적 선택 (주행가능영역 게이트와 CP 페널티)
weights/                가중치
```

## 출처

이 저장소는 아래 공개 코드를 고쳐서 쓴다. 각 코드의 라이선스를 따른다.

| 부분 | 원본 |
|---|---|
| `tutr/` | TUTR (Shi et al., ICCV 2023), https://github.com/lssiair/TUTR |
| `qcnet/` | QCNet (Zhou et al., CVPR 2023), https://github.com/ZikangZhou/QCNet, Apache-2.0 (`qcnet/LICENSE`) |
| `generation/lcsim`, `generation/motion_diff` | LCSim (Zhang et al., 2024), https://github.com/tsinghua-fib-lab/LCSim, MIT (`generation/LICENSE_LCSim`) |
| 라운드 학습 절차 | GALTraj (Park et al., ICCV 2025) 의 tail 마이닝, 생성, 이어 학습 절차를 따랐다 |
