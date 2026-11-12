# data/

이 저장소에는 데이터가 들어 있지 않다. 원본 궤적과 지도를 아래 위치에 놓으면 나머지는 코드가 만든다.

## 직접 놓아야 하는 것

```
data/
  Dataset_CSV/
    ss_bench_o/{train,test}/*_YYYYMMDD_track.csv     LIS-SS 일별 궤적
    ny_bench_o/{train,test}/*_YYYYMMDD_track.csv     LIS-NY 일별 궤적
  maps/
    SS_map.json
    NY_map.json
```

궤적 CSV 는 하루에 한 파일이고 10 Hz 이다. 파일 이름 끝의 `YYYYMMDD_track.csv` 에서 날짜를 읽는다.
`train`, `test` 폴더 구분은 쓰지 않는다. 분할은 `split_dates.py` 가 날짜로 다시 나눈다.

| 열 | 뜻 |
|---|---|
| frame_id | 프레임 번호 (0 부터, 0.1 초 간격) |
| track_id | 차량 번호 |
| x, y | 위치 [m] |
| vx, vy | 속도 [m/s] |
| ax, ay | 가속도 [m/s²] |

지도는 Argoverse 2 의 `log_map_archive` 와 같은 형식의 json 이다 (`drivable_areas`, `lane_segments`, `pedestrian_crossings`).

## 코드가 만드는 것

`python run.py --config configs/<backbone>_<site>_ours13.json --stage prep` 한 번이면 아래가 모두 생긴다.

```
data/
  scenarios/
    <site>/<YYYYMMDD>.parquet        40 프레임 시나리오 (관측 10 + 예측 30)        make_scenarios.py
    manifest_<site>.csv              시나리오 목록, split, CP 열                   make_scenarios.py, split_dates.py, compute_cp.py
    split_dates_<site>.csv           날짜별 분할                                  split_dates.py
  tutr/<site>/<site>_<split>.pkl     TUTR 입력                                    make_tutr_pkl.py
  tutr/<site>/<site>_<split>_meta.csv
  qcnet/<site>/<split>/raw/          QCNet 입력 (Argoverse 2 형식) 과 분할별 지도   make_qcnet_raw.py
  qcnet/<site>/processed/<split>/    QCNet 이 처음 읽을 때 만드는 캐시
```

분할은 날짜 단위이다. 날짜를 시간 순서로 놓고 시나리오 수 누적 비율이 init 0.2, remain 0.2, val 0.1, test 0.5 가 되게 채운다.
init 은 처음 학습, remain 은 tail 마이닝 풀, val 은 모델 선택, test 는 최종 보고에만 쓴다.
같은 원본이면 아래 숫자가 그대로 나온다.

| 사이트 | 분할 | 날짜 | 시나리오 | 평가 샘플 (차량) |
|---|---|---|---|---|
| SS | init | 20250101 ~ 20250124 | 12,512 | 25,338 |
| SS | remain | 20250125 ~ 20250213 | 13,163 | 26,802 |
| SS | val | 20250214 ~ 20250215 | 5,264 | 10,808 |
| SS | test | 20250216 ~ 20250331 | 32,292 | 65,721 |
| NY | init | 20250101 ~ 20250122 | 708 | 1,416 |
| NY | remain | 20250123 ~ 20250205 | 784 | 1,621 |
| NY | val | 20250209 ~ 20250213 | 441 | 882 |
| NY | test | 20250214 ~ 20250329 | 1,853 | 3,732 |

`manifest_<site>.csv` 에는 단계가 진행되면서 열이 더해진다.
`gt_cp_max` 는 prep 단계, `vanilla_cp_max` 와 `qcnet_vanilla_cp_max` 는 vanilla_cp 단계, `eval_<backbone>_<arm>_cp_max` 는 eval 단계가 쓴다.
