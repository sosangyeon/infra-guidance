"""LCSim 생성 + time-shift 결과 CSV -> QCNet raw 폴더 (data_prep/make_qcnet_raw.py 와 같은 구조).

입력: <gen_dir>/scenario_<sid>/scenario_<sid>_aug<NNN>.csv  (generation/postprocess_shift.py 출력)
      열: scenario_id, aug_id, seed, track_id, timestep, observed, extrap, position_x/y, heading, velocity_x/y
출력: <out>/<sid>_aug<NNN>/scenario_<sid>_aug<NNN>.parquet, <out>/log_map_archive_<site>.json (init raw 의 지도 복사)

규칙 (TUTR gen_to_pkl.py 와 같게)
  - 생성 장면(aug 파일) 하나 = 시나리오 하나.
  - extrap = 1 (time-shift pad) 인 미래 프레임은 행을 뺀다 → QCNet predict_mask 가 그 스텝을 회귀 손실에서 뺀다
    (GALTraj Eq.5-6 pad 마스크와 같은 효과).
  - 관측 구간에 pad 가 있는 차량(원본 결측)은 통째로 뺀다.
  - 열은 make_scenarios.py parquet 과 같게: object_type VEHICLE, object_category 3, city 는 사이트 값,
    가속도는 속도 후방차분 / 0.1 s (QCNet 은 읽지 않음).
"""
import argparse
import glob
import os
import shutil

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OBS_LEN, HORIZON, DT = 10, 40, 0.1
CITY = {'ss': 'sesol', 'ny': 'namyang'}


def convert(df, sid, site):
    df = df.sort_values(['track_id', 'timestep'])
    hist_pad = df[(df['timestep'] < OBS_LEN) & (df['extrap'] == 1)]['track_id'].unique()
    df = df[~df['track_id'].isin(hist_pad) & (df['extrap'] == 0)].copy()
    if df.empty:
        return None
    df['velocity_x'] = df['velocity_x'].astype(float)
    df['velocity_y'] = df['velocity_y'].astype(float)
    g = df.groupby('track_id')
    df['acceleration_x'] = g['velocity_x'].diff().fillna(0.0) / DT
    df['acceleration_y'] = g['velocity_y'].diff().fillna(0.0) / DT
    out = pd.DataFrame({
        'observed': df['timestep'].to_numpy() < OBS_LEN,
        'track_id': df['track_id'].astype(np.int32).to_numpy(),
        'object_type': 'VEHICLE',
        'object_category': 3,
        'timestep': df['timestep'].astype(np.int32).to_numpy(),
        'position_x': df['position_x'].astype(float).to_numpy(),
        'position_y': df['position_y'].astype(float).to_numpy(),
        'heading': df['heading'].astype(float).to_numpy(),
        'velocity_x': df['velocity_x'].to_numpy(),
        'velocity_y': df['velocity_y'].to_numpy(),
        'acceleration_x': df['acceleration_x'].to_numpy(),
        'acceleration_y': df['acceleration_y'].to_numpy(),
        'scenario_id': sid,
        'start_timestamp': 0, 'end_timestamp': 0, 'num_timestamps': HORIZON, 'focal_track_id': 0,
        'city': CITY[site], 'Data_Num': '0000',
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', required=True, choices=sorted(CITY))
    ap.add_argument('--gen_dir', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out)
    m = os.path.join(ROOT, 'data', 'qcnet', a.site, 'init', 'raw', f'log_map_archive_{a.site}.json')
    shutil.copyfile(m, os.path.join(a.out, os.path.basename(m)))
    n_scn = n_agent = n_drop = 0
    for f in sorted(glob.glob(os.path.join(a.gen_dir, 'scenario_*', 'scenario_*_aug*.csv'))):
        sid = os.path.basename(f)[len('scenario_'):-len('.csv')]
        out = convert(pd.read_csv(f), sid, a.site)
        if out is None:
            n_drop += 1
            continue
        d = os.path.join(a.out, sid)
        os.makedirs(d)
        out.to_parquet(os.path.join(d, f'scenario_{sid}.parquet'), index=False)
        n_scn += 1
        n_agent += out['track_id'].nunique()
    print(f'[gen_to_qcnet] {a.site}: 장면 {n_scn:,} (에이전트 {n_agent:,}, 빈 장면 제외 {n_drop}) -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
