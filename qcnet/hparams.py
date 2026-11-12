"""QCNet 모델·옵티마이저·contrastive 값.

아래 기본값이 논문 실험 값이다. configs/qcnet_<site>_ours13.json 의 model / optimizer / contrastive 절을
--config 로 주면 그 값이 기본값을 덮어쓴다 (학습 스크립트 세 개가 같은 함수를 쓴다).
"""
import json

MODEL = dict(
    dataset='argoverse_v2_ACL', input_dim=2, hidden_dim=128, output_dim=2, output_head=False,
    num_historical_steps=10, num_future_steps=30, num_modes=6, num_recurrent_steps=3,
    num_freq_bands=64, num_map_layers=1, num_agent_layers=2, num_dec_layers=2, num_heads=8,
    head_dim=16, dropout=0.1, pl2pl_radius=150, time_span=10, pl2a_radius=50, a2a_radius=50,
    num_t2m_steps=30, pl2m_radius=150, a2m_radius=150)
OPTIMIZER = dict(lr=5e-4, weight_decay=1e-4)
CONTRASTIVE = dict(cp_col='qcnet_vanilla_cp_max', lambda_contrastive=1.0, tau=0.5, theta_p=0.33, theta_n=0.50)


def load(path):
    if not path:
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def model_args(config, out, total):
    """QCNet(**args) 인자. cosine 주기 T_max = 전체 epoch."""
    c = load(config)
    return dict(MODEL, **c.get('model', {}), **dict(OPTIMIZER, **c.get('optimizer', {})), T_max=total,
                submission_dir=out, submission_file_name='submission')


def contrastive_args(config):
    """QCNet_Contrastive 의 cp_col, lambda_contrastive, tau, theta_p, theta_n."""
    c = load(config).get('contrastive', {})
    return dict(CONTRASTIVE, **{k: v for k, v in c.items() if k in CONTRASTIVE})
