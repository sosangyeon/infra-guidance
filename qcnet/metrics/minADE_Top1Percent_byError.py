import torch
from torchmetrics import Metric

class minADE_Top1Percent_byError(Metric):
    """
    에폭 전체에서 minADE 값 기준 상위 K% (가장 큰 에러) 샘플들의 평균 ADE 계산.
    난이도(difficulty)와 무관.
    """
    full_state_update = False

    def __init__(self, max_guesses: int = 6, percentile: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.max_guesses = max_guesses
        self.percentile = percentile
        self.add_state("all_ade", default=[], dist_reduce_fx=None)

    def update(self, pred, target, prob=None, valid_mask=None, **_):
        """
        pred: [N, M, T, 2]   예측 궤적
        target: [N, T, 2]    GT 궤적
        """
        # 1️⃣ 각 모드별 ADE 계산
        displacement_error = torch.norm(pred - target.unsqueeze(1), dim=-1)  # [N, M, T]
        ade_all_modes = displacement_error.mean(dim=-1)  # [N, M]
        min_ade_for_each_agent = ade_all_modes.min(dim=-1)[0]  # [N]

        # 2️⃣ 누적 저장
        self.all_ade.append(min_ade_for_each_agent.detach().cpu())

    def compute(self):
        """
        에폭 전체에서 가장 큰 ADE 상위 K%의 평균을 계산
        """
        import math
        if len(self.all_ade) == 0:
            return torch.tensor(0.0, device=self.device)

        all_ade = torch.cat(self.all_ade)  # [Total_N]
        N = all_ade.numel()
        p = self.percentile / 100.0
        k = max(1, math.ceil(N * p))

        # 🔹 ADE가 큰 순서대로 상위 k개 선택
        _, topk_idx = torch.topk(all_ade, k, largest=True)
        return all_ade[topk_idx].mean().to(self.device)
