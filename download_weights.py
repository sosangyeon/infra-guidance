"""큰 가중치 (QCNet, LCSim) 를 GitHub Release 에서 받아 weights/ 에 놓는다.

  python download_weights.py                  # 전부
  python download_weights.py --site ny        # 한 사이트만
  python download_weights.py --only lcsim     # TUTR 만 쓸 때 (생성 모델만)

TUTR 가중치와 LCSim 의 pca.pkl 은 크기가 작아 저장소에 들어 있다.
"""
import argparse
import os
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
URL = 'https://github.com/sosangyeon/infra-guidance/releases/download/v1.0/'
QCNET = ['vanilla', 'contrastive', 'contrastive_best_2of3', 'ours', 'ours13']

for _s in (sys.stdout, sys.stderr):   # 한글을 못 쓰는 콘솔 코드페이지에서도 멈추지 않게
    _s.reconfigure(errors='replace')


def files(site):
    """(Release 파일 이름, 저장 위치)"""
    out = [(f'qcnet_{site}_{n}.ckpt', os.path.join('weights', 'qcnet', site, f'{n}.ckpt')) for n in QCNET]
    return out + [(f'lcsim_{site}.ckpt', os.path.join('weights', 'lcsim', site, f'lcsim_{site}.ckpt'))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--site', default='ss,ny')
    ap.add_argument('--only', default=None, choices=['qcnet', 'lcsim'])
    ap.add_argument('--force', action='store_true', help='이미 있어도 다시 받는다')
    a = ap.parse_args()
    failed = []
    for site in a.site.split(','):
        for name, rel in files(site):
            if a.only and not name.startswith(a.only):
                continue
            dst = os.path.join(ROOT, rel)
            if os.path.exists(dst) and not a.force:
                print(f'있음   {rel}')
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                urllib.request.urlretrieve(URL + name, dst + '.part')
            except urllib.error.URLError as e:
                failed.append(name)
                print(f'실패   {name}: {e}')
                continue
            os.replace(dst + '.part', dst)
            print(f'받음   {rel}  ({os.path.getsize(dst) / 1e6:.1f} MB)')
    if failed:
        sys.exit(f'받지 못한 파일 {len(failed)} 개: {failed}')


if __name__ == '__main__':
    main()
