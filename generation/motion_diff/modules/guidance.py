import math
import os
from typing import Mapping

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from ..metrics import geometry, roadgraph
from .tail_score_net import TailScoreNet


def wrap_angle(
    angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi
) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)


class Repeller(nn.Module):
    def __init__(
        self,
        epsilon=1e-6,
        alpha=0.05,
        num_step=10,
        radius=6,
        beta=0.05,
    ):
        super(Repeller, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_step = num_step
        self.radius = radius  # desired minimum distance between agents
        self.beta = beta  # clipping parameter

    def loss(
        self,
        x: torch.Tensor,
        diffuser,
        data: HeteroData,
    ):
        # reconstruct the trajectory
        traj = diffuser._reconstruct_traj(data, x)  # (agent_num, time_step, 2)
        shape = data["agent"]["shape"][:, 0, :2]
        # compute pairwise distance between agents at each time step
        dist = torch.norm(traj.unsqueeze(1) - traj.unsqueeze(0), dim=-1, p=2)
        # compute loss
        a = torch.nn.functional.relu(
            (1 - dist / self.radius)
            * (1 - torch.eye(dist.shape[0], device=dist.device)).unsqueeze(-1)
        )
        return torch.sum(a) / (torch.sum(a > 0) + self.epsilon)


class NoOffRoad(nn.Module):
    def __init__(self, epsilon=1e-6, alpha=0.05, num_step=10, radius=1.0, beta=0.05):
        super(NoOffRoad, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_step = num_step
        self.radius = radius  # desired minimum distance between agents
        self.beta = beta  # clipping parameter

    def loss(
        self,
        x: torch.Tensor,
        diffuser,
        data: HeteroData,
    ):
        # reconstruct the trajectory
        traj = diffuser._reconstruct_traj(data, x)  # (agent_num, time_step, 2)
        valid_mask = (
            data["agent"]["valid"][:, diffuser.num_historical_steps - 1 :]
            .squeeze(-1)
            .all(dim=-1)
        )
        vehicle_mask = data["agent"]["type"] == 1
        agent_mask = valid_mask & vehicle_mask
        if not torch.any(agent_mask):
            return torch.zeros((), device=x.device, dtype=x.dtype)
        traj = traj[agent_mask].reshape(-1, 2)
        roadgraph_points = torch.cat(
            [
                data["roadgraph_points"]["xyz"],
                data["roadgraph_points"]["dir"],
                data["roadgraph_points"]["ids"].unsqueeze(-1),
                data["roadgraph_points"]["type"].unsqueeze(-1),
            ],
            dim=-1,
        ).contiguous()
        # Use polygonal drivable-area distance when available. This is robust to
        # arbitrary polyline orientation in reconstructed AV2/LIS maps.
        signed_dis = roadgraph.compute_signed_distance_to_drivable_area(
            query_points=traj, roadgraph_points=roadgraph_points
        )
        # compute loss
        a = torch.nn.functional.relu(self.radius + signed_dis)
        return torch.sum(a) / (torch.sum(a > 0) + self.epsilon)


class RepellerPaperExact(nn.Module):
    """GALTraj 논문(ICCV 2025 supplementary Sec 5, Eq.4) 그대로:
        C_repeller(y_n, y_m) = sum_t max(0, eps_rep - ||p_n_t - p_m_t||)
    위 `Repeller`는 (1 - dist/radius) 형태(=eps_rep=radius일 때 1/radius 스칼라배)에 위반개수로
    나눈 평균(=스칼라 나눗셈)을 썼다. `RealGuidance.forward`가 grad_norm=True(기본, 프로덕션 설정)
    로 모듈별 그래디언트를 항상 단위벡터 정규화하므로 두 형태의 실제 guidance 방향은 이미 동일했지만
    (양의 스칼라배는 정규화 후 사라짐), 수식 자체를 논문과 literal하게 맞추기 위해 별도 클래스로
    분리한다. eps_rep 수치는 논문에 없어 기존 `REPELLER_RADIUS` 선례를 그대로 승계."""

    def __init__(self, eps_rep=2.5):
        super().__init__()
        self.eps_rep = eps_rep

    def loss(self, x: torch.Tensor, diffuser, data: HeteroData):
        traj = diffuser._reconstruct_traj(data, x)  # (agent_num, time_step, 2)
        dist = torch.norm(traj.unsqueeze(1) - traj.unsqueeze(0), dim=-1, p=2)  # (N, N, T)
        offdiag = (1 - torch.eye(dist.shape[0], device=dist.device)).unsqueeze(-1)
        a = torch.nn.functional.relu(self.eps_rep - dist) * offdiag
        return a.sum()


class NoOffRoadPaperExact(nn.Module):
    """GALTraj 논문(ICCV 2025 supplementary Sec 5, Eq.3) 그대로:
        C_no-off-road(y_n) = sum_t max(0, d_road(p_n_t) - eps_off)
    d_road(p) = 도로(주행가능영역) 밖으로 벗어난 거리(안쪽이면 0). eps_off = 허용오차(논문 수치
    미제공, 기본 0 = 벗어나는 즉시 패널티).

    위 `NoOffRoad`는 `relu(radius + signed_dis)` 형태였다 — `signed_dis`는 도로 안쪽이 음수인데,
    `signed_dis > -radius`(=아직 도로 안쪽이지만 경계에서 radius 이내)에서도 패널티가 발동해
    "실제로 도로를 벗어나기 전부터 중앙 쪽으로 미리 미는" 선제적 코리도어였다. 이는 스칼라배 차이가
    아니라 **패널티가 발동하는 조건 자체가 다른**(hinge의 꺾이는 지점이 다른) 진짜 함수 형태 차이라
    grad_norm 정규화로도 상쇄되지 않는다. 아래는 논문 그대로 "실제로 도로를 벗어난 거리"에 대해서만
    패널티를 준다(d_road=relu(signed_dis), 즉 안쪽이면 정확히 0)."""

    def __init__(self, eps_off=0.0):
        super().__init__()
        self.eps_off = eps_off

    def loss(self, x: torch.Tensor, diffuser, data: HeteroData):
        traj = diffuser._reconstruct_traj(data, x)  # (agent_num, time_step, 2)
        valid_mask = (
            data["agent"]["valid"][:, diffuser.num_historical_steps - 1:]
            .squeeze(-1)
            .all(dim=-1)
        )
        vehicle_mask = data["agent"]["type"] == 1
        agent_mask = valid_mask & vehicle_mask
        if not torch.any(agent_mask):
            return torch.zeros((), device=x.device, dtype=x.dtype)
        traj = traj[agent_mask].reshape(-1, 2)
        roadgraph_points = torch.cat(
            [
                data["roadgraph_points"]["xyz"],
                data["roadgraph_points"]["dir"],
                data["roadgraph_points"]["ids"].unsqueeze(-1),
                data["roadgraph_points"]["type"].unsqueeze(-1),
            ],
            dim=-1,
        ).contiguous()
        signed_dis = roadgraph.compute_signed_distance_to_drivable_area(
            query_points=traj, roadgraph_points=roadgraph_points
        )
        # signed_dis: 음수=도로 안쪽, 양수=도로 밖(roadgraph.py 확인).
        d_road = torch.nn.functional.relu(signed_dis)  # 안쪽=0, 밖=벗어난 거리
        a = torch.nn.functional.relu(d_road - self.eps_off)
        return a.sum()


class OnLane(nn.Module):
    """진행방향이 일치하는(같은 방향) centerline 후보만 대상으로 '거리 + heading 정렬'
    soft cost 를 최소화 → 자기차로/진행방향 유지, 중앙선 침범(반대차선·역주행) 억제.

    [개선] 기존 hard-gate(angle<0.2 & dist<5) 방식은 인접/반대차선 centerline
    이 잘못 선택되어 침범을 못 줄였음(절제: 침범 +2.1%). 같은 방향(cos>cos_thresh)만 남기고
    거리에 heading 비용을 더한 soft cost 로 바꿔 침범을 실제로 감소시킴(절제: off-road
    6.9→3.7%, 침범 11.7→6.7% 동시 개선, 단독으로 최적 J). 옛 파라미터(angle_thresh/
    dist_thresh)는 하위호환 위해 attribute 로만 유지(미사용).
    """

    def __init__(
        self, epsilon=1e-6, alpha=0.05, num_step=10, radius=1.0, beta=0.05,
        angle_thresh=0.2, dist_thresh=5.0, min_speed=1e-3,
        cos_thresh=0.3, w_head=2.0, cap=10.0,
    ):
        super(OnLane, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_step = num_step
        self.radius = radius
        self.beta = beta
        self.angle_thresh = angle_thresh  # (구버전, 미사용)
        self.dist_thresh = dist_thresh    # (구버전, 미사용)
        self.min_speed = min_speed        # 이보다 느린 점은 heading 미정 → 손실 제외
        self.cos_thresh = cos_thresh      # 같은 방향으로 볼 cos(heading차) 하한(>0.3≈±72°)
        self.w_head = w_head              # heading 정렬 비용 가중
        self.cap = cap                    # 단일점 비용 상한(폭주 방지)

    def loss(
        self,
        x: torch.Tensor,
        diffuser,
        data: HeteroData,
    ):
        xy = diffuser._reconstruct_traj(data, x)  # (agent_num, time_step, 2)
        dir_xy = torch.cat(
            [xy[:, 1:, :] - xy[:, :-1, :], xy[:, -1:, :] - xy[:, -2:-1, :]], dim=1
        )
        q_xy = xy.reshape(-1, 2)
        q_dir = dir_xy.reshape(-1, 2)
        types = data["roadgraph_points"]["type"]
        centerline_index = (types == 1) | (types == 2)
        cl_xy = data["roadgraph_points"]["xyz"][centerline_index][:, :2]
        cl_dir = data["roadgraph_points"]["dir"][centerline_index][:, :2]
        if cl_xy.shape[0] == 0:
            return q_xy.new_zeros(())
        q_ang = torch.atan2(q_dir[:, 1], q_dir[:, 0])
        cl_ang = torch.atan2(cl_dir[:, 1], cl_dir[:, 0])
        ang_diff = wrap_angle(q_ang[:, None] - cl_ang)        # (Q, C) 부호 유지
        same_dir = torch.cos(ang_diff) > self.cos_thresh      # 같은 방향 후보만
        # 거리(미분안전: clamp + sqrt(.+eps)) + heading 정렬 비용
        diff = (q_xy[:, None, :] - cl_xy[None, :, :]).clamp(-1e4, 1e4)
        distance = torch.sqrt((diff ** 2).sum(-1) + self.epsilon)
        cost = distance + self.w_head * torch.abs(ang_diff)
        # 미매칭(반대/인접차선)은 큰 값으로 → torch.where 의 0*inf=nan 회피
        big = torch.full_like(cost, 1e6)
        cost = torch.where(same_dir, cost, big)
        min_cost = torch.min(cost, dim=1)[0].clamp(max=self.cap)  # (Q,)
        # 정지/저속 점은 heading 미정 → 제외 (atan2(0,0) nan 방지)
        speed = torch.norm(q_dir, dim=-1)
        valid = speed > self.min_speed
        if not bool(valid.any()):
            return q_xy.new_zeros(())
        return min_cost[valid].mean()


class OnLaneBand(OnLane):
    """차로 코리도어 유지(deadband): 같은방향 차로 중심선에서 band(m) 이내면 0(자유),
    밖이면 (거리-band)만 패널티 → 중앙선에 달라붙지 않되 차로 이탈/침범은 억제.
    onlane(중심으로 끌어당김)과 달리 차로 안 자연스러운 횡변위 허용. band=BAND_M(env)."""

    def __init__(self, band=1.2, **kw):
        super().__init__(**kw)
        self.band = band

    def loss(self, x, diffuser, data: HeteroData):
        xy = diffuser._reconstruct_traj(data, x)
        dir_xy = torch.cat([xy[:, 1:] - xy[:, :-1], xy[:, -1:] - xy[:, -2:-1]], dim=1)
        q_xy = xy.reshape(-1, 2)
        q_dir = dir_xy.reshape(-1, 2)
        types = data["roadgraph_points"]["type"]
        idx = (types == 1) | (types == 2)
        cl_xy = data["roadgraph_points"]["xyz"][idx][:, :2]
        cl_dir = data["roadgraph_points"]["dir"][idx][:, :2]
        if cl_xy.shape[0] == 0:
            return q_xy.new_zeros(())
        q_ang = torch.atan2(q_dir[:, 1], q_dir[:, 0])
        cl_ang = torch.atan2(cl_dir[:, 1], cl_dir[:, 0])
        ang_diff = wrap_angle(q_ang[:, None] - cl_ang)
        same_dir = torch.cos(ang_diff) > self.cos_thresh
        diff = (q_xy[:, None, :] - cl_xy[None, :, :]).clamp(-1e4, 1e4)
        distance = torch.sqrt((diff ** 2).sum(-1) + self.epsilon)
        big = torch.full_like(distance, 1e6)
        d_same = torch.where(same_dir, distance, big)
        min_d = torch.min(d_same, dim=1)[0]
        cost = torch.nn.functional.relu(min_d - self.band).clamp(max=self.cap)
        speed = torch.norm(q_dir, dim=-1)
        valid = (speed > self.min_speed) & (min_d < 1e5)
        if not bool(valid.any()):
            return q_xy.new_zeros(())
        return cost[valid].mean()


class OnLaneOffset(OnLaneBand):
    """head 횡방향 다양성 '주입'(허용이 아니라 당김): 같은방향 최근접 centerline 기준
    부호있는 횡거리(왼쪽+)를 목표 offset(m) 으로 끌어당긴다 + band 밖 barrier 유지.
    prior 가 횡방향으로 붕괴된 경우(offset 을 aug/rollout 별로 다르게) lateral 모드를
    강제로 분기시키는 용도. offset=LAT_OFFSET(env), 강도 w_off=LAT_W(env)."""

    def __init__(self, band=1.2, offset=0.0, w_off=1.0, **kw):
        super().__init__(band=band, **kw)
        self.offset = offset
        self.w_off = w_off

    def loss(self, x, diffuser, data: HeteroData):
        xy = diffuser._reconstruct_traj(data, x)
        dir_xy = torch.cat([xy[:, 1:] - xy[:, :-1], xy[:, -1:] - xy[:, -2:-1]], dim=1)
        q_xy = xy.reshape(-1, 2)
        q_dir = dir_xy.reshape(-1, 2)
        types = data["roadgraph_points"]["type"]
        idx = (types == 1) | (types == 2)
        cl_xy = data["roadgraph_points"]["xyz"][idx][:, :2]
        cl_dir = data["roadgraph_points"]["dir"][idx][:, :2]
        if cl_xy.shape[0] == 0:
            return q_xy.new_zeros(())
        q_ang = torch.atan2(q_dir[:, 1], q_dir[:, 0])
        cl_ang = torch.atan2(cl_dir[:, 1], cl_dir[:, 0])
        ang_diff = wrap_angle(q_ang[:, None] - cl_ang)
        same_dir = torch.cos(ang_diff) > self.cos_thresh
        diff = (q_xy[:, None, :] - cl_xy[None, :, :]).clamp(-1e4, 1e4)
        distance = torch.sqrt((diff ** 2).sum(-1) + self.epsilon)
        big = torch.full_like(distance, 1e6)
        d_same = torch.where(same_dir, distance, big)
        min_d, min_i = torch.min(d_same, dim=1)
        # 부호있는 횡거리: 최근접 centerline 진행방향의 왼쪽 법선 성분
        c_xy = cl_xy[min_i]
        c_dir = cl_dir[min_i]
        c_dir = c_dir / (c_dir.norm(dim=-1, keepdim=True) + 1e-9)
        rel = q_xy - c_xy
        signed_lat = c_dir[:, 0] * rel[:, 1] - c_dir[:, 1] * rel[:, 0]
        cost = (
            torch.nn.functional.relu(min_d - self.band)
            + self.w_off * torch.abs(signed_lat - self.offset)
        ).clamp(max=self.cap)
        speed = torch.norm(q_dir, dim=-1)
        valid = (speed > self.min_speed) & (min_d < 1e5)
        if not bool(valid.any()):
            return q_xy.new_zeros(())
        return cost[valid].mean()


class Velocity(nn.Module):

    DESIRED_SPEED = 4.0  # desired speed [m/0.1s]
    VEL_MEAN = 0.2417  # mean of velocity [m/0.1s]

    def __init__(self, epsilon=1e-6, alpha=0.05, num_step=10, beta=0.05):
        super(Velocity, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_step = num_step
        self.beta = beta  # clipping parameter

    def loss(
        self,
        x: torch.Tensor,
        diffuser,
        data: HeteroData,
    ):
        module = torch.clip(x[..., 0] + self.VEL_MEAN, min=0)
        diff = torch.nn.functional.relu(self.DESIRED_SPEED - module)
        return torch.mean(diff)


class Acceleration(nn.Module):

    ACC_MAX = 3.0

    def __init__(self, epsilon=1e-6, alpha=0.05, num_step=10, beta=0.05):
        super(Acceleration, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_step = num_step

    def loss(
        self,
        x: torch.Tensor,
        diffuser,
        data: HeteroData,
    ):
        vel_module = torch.clip(x[..., 0] + Velocity.VEL_MEAN, min=0)
        acc = vel_module[..., 1:] - vel_module[..., :-1]
        loss = torch.nn.functional.relu(acc.abs() - self.ACC_MAX)
        return torch.mean(loss)


class GoalPoint(nn.Module):

    def __init__(self, epsilon=1e-6, alpha=0.05, num_step=10, beta=0.05):
        super(GoalPoint, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_step = num_step

    def loss(
        self,
        x: torch.Tensor,
        diffuser,
        data: HeteroData,
        goal_point: torch.Tensor,
        mask: torch.Tensor,
    ):
        xy = diffuser._reconstruct_traj(data, x)  # (agent_num, time_step, 2)
        dis = torch.norm((xy - goal_point), dim=-1, p=2) * mask
        return torch.mean(dis)


class TailResidual(nn.Module):
    """tail-score(학습 없음). residual = GT − pred(외부 예측모델 출력) 을 궤적공간 vector
    field 로 정의하고 ⟨traj, residual⟩ 을 키워(=손실 −⟨⟩) 생성궤적을 '예측이 틀린 방향'
    으로 민다. 아주대 김찬미(2026.02) 논문 식 (4.5)의 c·g(GALTraj 확장, GALTraj 자체엔
    없는 개념)를 학습 없이 구현한 TP_GAL TailResidual 을 이 repo(모드 차원 없음,
    traj=(N,nfut,2))로 포팅. 최신 LCSim-gal(div_repel 등 7월 갱신분) 동기화 후 재적용.

    엔진이 tail 에이전트에 대해 첫 plan 에서 채우는
      data["agent"]["tail_residual"]       (N, nfut, 2)  world, GT−pred (x 무관 상수)
      data["agent"]["tail_residual_valid"] (N, nfut)     pred 가 있는 step 만 True
    가 있을 때만 동작. 없으면 autograd 그래프 보존하며 0 (자동 no-op).
    residual 은 tail 에이전트에만 비0 → 손실이 자연히 tail-only (별도 마스크 불필요)."""

    def __init__(self, epsilon: float = 1e-6):
        super(TailResidual, self).__init__()
        self.epsilon = epsilon

    def loss(self, x: torch.Tensor, diffuser, data: HeteroData):
        ag = data["agent"]
        if "tail_residual" not in ag:
            return x.sum() * 0.0
        traj = diffuser._reconstruct_traj(data, x)        # (N, nfut, 2) world, 미분가능
        residual = ag["tail_residual"].to(traj.dtype)     # (N, nfut, 2)  GT−pred
        if residual.shape != traj.shape:
            return x.sum() * 0.0
        if "tail_residual_valid" in ag:
            v = ag["tail_residual_valid"].to(torch.bool)  # (N, nfut)
        else:
            v = residual.abs().sum(dim=-1) > self.epsilon
        if not bool(v.any()):
            return x.sum() * 0.0
        inner = (traj * residual).sum(dim=-1)             # (N, nfut) = ⟨traj, residual⟩
        return -inner[v].mean()                            # 부호 −: residual 방향으로 push


class GTLeash(nn.Module):
    """tail 에이전트의 GT 이탈 상한(힌지). 학위논문 §4.3 요구를 식 (4.5) 와 공존시킨다.

    배경 — 논문 내부 모순:
      · §4.3(식 4.2): "Tail agent 는 ... GT 궤적을 크게 벗어나지 않는 범위에서
        **제한적인 변동성만** 허용"  → tail 을 붙잡으라
      · 식 (4.5): tail-score gradient 로 tail 을 **밀어낸다**
      실측(shift 전·extrap 제외): GT 대비 최대이탈
        ⑥(밀지 않음) 2.745m / ⑦(식 4.5) 3.570m — 미는 쪽이 논문 요구를 덜 지킨다.

    설계 — 힌지이지 페널티가 아니다:
        ℓ = Σ_t max(0, ‖x_t − x^GT_t‖ − d_max)²
      d_max 이내에서는 gradient 가 정확히 0 이라 **tail_score 의 다양성을 건드리지 않고**,
      초과분만 되당긴다. c 를 낮추는 것(모든 곳에서 균일하게 약화)과 다르다.

    d_max 는 임의 상수가 아니라 데이터 기준: 실주행 최대이탈 3.41 m (env GT_LEASH_DMAX).

    엔진이 채우는 data["agent"]["tail_gt"] / ["tail_gt_valid"] 가 있을 때만 동작.
    없으면 autograd 그래프를 보존한 채 0 (자동 no-op).
    """

    def __init__(self, d_max: float = 3.41, epsilon: float = 1e-6):
        super(GTLeash, self).__init__()
        self.d_max = float(d_max)
        self.epsilon = epsilon

    def loss(self, x: torch.Tensor, diffuser, data: HeteroData):
        ag = data["agent"]
        if "tail_gt" not in ag:
            return x.sum() * 0.0
        traj = diffuser._reconstruct_traj(data, x)          # (N, nfut, 2) world, 미분가능
        gt = ag["tail_gt"].to(traj.dtype)
        if gt.shape != traj.shape:
            return x.sum() * 0.0
        v = ag["tail_gt_valid"].to(torch.bool) if "tail_gt_valid" in ag else \
            (gt.abs().sum(dim=-1) > self.epsilon)
        if not bool(v.any()):
            return x.sum() * 0.0
        dist = (traj - gt).norm(dim=-1)                     # (N, nfut)
        over = torch.clamp(dist - self.d_max, min=0.0)      # 힌지: 상한 이내는 0
        return (over[v] ** 2).mean()


class TailScore(nn.Module):
    """Learned tail-score guidance.

    A small network predicts tail-likeness from agent-centric observed history and
    generated future. Guidance maximizes this score for tail agents only.
    """

    def __init__(self, ckpt_path: str | None = None, epsilon: float = 1e-6):
        super(TailScore, self).__init__()
        self.epsilon = epsilon
        self.enabled = False
        self.net: TailScoreNet | None = None
        ckpt_path = ckpt_path or os.environ.get("TAIL_SCORE_CKPT", "").strip()
        if ckpt_path and os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            kwargs = ckpt.get("model_kwargs", {})
            self.net = TailScoreNet(**kwargs)
            self.net.load_state_dict(ckpt["state_dict"])
            self.net.eval()
            for p in self.net.parameters():
                p.requires_grad_(False)
            self.enabled = True

    def loss(self, x: torch.Tensor, diffuser, data: HeteroData):
        if not self.enabled or self.net is None:
            return x.sum() * 0.0
        ag = data["agent"]
        if "tail_score_valid" not in ag:
            return x.sum() * 0.0
        mask = ag["tail_score_valid"].to(torch.bool)
        if not bool(mask.any()):
            return x.sum() * 0.0
        traj = diffuser._reconstruct_traj(data, x)  # (N, nfut, 2) world
        # [좌표계 수정] tail-score net 은 **TUTR TrajectoryDataset 규약**으로 학습됐다.
        #   (02_tutr/dataset.py:46-57 — translation=True, rotation=True)
        #     origin = hist[obs_len-1]                       (마지막 관측점)
        #     ref    = (hist - origin)[0]                    (평행이동 후 **궤적 시작점**)
        #     angle  = atan2(ref.y, ref.x)
        #     rot    = [[cos, -sin], [sin, cos]] ,  v @ rot
        #   결과적으로 hist[0] 이 +x 축 위에 놓이고(실측 100.0%, |y| 평균 0.0000)
        #   차량은 **-x 방향으로 진행**한다(미래 종점이 -x 인 비율 99.2%, x 평균 -18.84).
        #
        #   기존 구현은 ag["heading"] 을 기준으로 회전해 **진행 방향을 +x** 로 놓았고,
        #   회전행렬 부호도 반대([[c,s],[-s,c]])였다. 즉 학습 분포와 약 180도 뒤집힌 입력을
        #   네트워크에 먹이고 있었다 → ρ=0.49 가 주입 시점에 전혀 전이되지 않고 gradient 가
        #   무의미한 방향을 가리킨다. (tail_score_vlf.pt 로 생성한 구 산출물도 동일 결함)
        time_step = diffuser.num_historical_steps - 1
        hist = ag["xyz"][:, : diffuser.num_historical_steps, :2].to(traj.dtype)
        origin = ag["xyz"][:, time_step, :2].to(traj.dtype)
        hist_c = hist - origin.unsqueeze(1)
        fut_c = traj - origin.unsqueeze(1)
        ref = hist_c[:, 0]                                  # 궤적 시작점(평행이동 후)
        angle = torch.atan2(ref[:, 1], ref[:, 0])
        cos, sin = angle.cos(), angle.sin()
        rot_t = angle.new_zeros((len(angle), 2, 2))
        rot_t[:, 0, 0] = cos
        rot_t[:, 0, 1] = -sin
        rot_t[:, 1, 0] = sin
        rot_t[:, 1, 1] = cos
        hist_ag = torch.bmm(hist_c, rot_t)
        fut_ag = torch.bmm(fut_c, rot_t)
        # cur_speed 도 학습 시 정의와 일치시킨다: ||hist[-1]-hist[-2]|| / 0.1 (유한차분).
        #   norm 은 회전불변이라 좌표계 영향은 없으나, ag["velocity"] 필드는 학습에 쓰인
        #   유한차분과 값이 다를 수 있어 분기를 제거한다(학습: build_labels.py 참조).
        cur_speed = (
            (ag["xyz"][:, time_step, :2] - ag["xyz"][:, time_step - 1, :2]).norm(dim=-1)
            / 0.1
        ).to(traj.dtype)
        net = self.net.to(device=x.device, dtype=traj.dtype)
        score = net(hist_ag, fut_ag, cur_speed)

        # ── 논문 문구에 맞춘 상한(cap) ──────────────────────────────
        #   학위논문 식 (4.5) 설명: "tail-score 함수의 gradient 를 역확산 단계에 주입함으로써,
        #   **데이터 분포를 유지하면서도** 예측이 어려운 tail 영역에 대한 궤적 샘플을
        #   선택적으로 증강" / "c 는 ... tail sample **주변에서의 분포를 강화**하는 역할".
        #
        #   기존 구현은 -score.mean() = **상한 없는 최대화**였다. 그런데 real-guidance 가
        #   앵커로 삼는 tail GT 자체가 이미 minADE 백분위 89.0%(중앙 95.0%)이므로,
        #   그 위로 더 밀면 관측 분포 **밖**으로 나간다 — 논문의 "분포를 유지" 조건 위반.
        #   실측 귀결: GT 이탈 ⑥ 2.745m → ⑦ 3.570m, 50% 가 실주행 최대이탈(3.41m) 초과,
        #             입력 대비 미래분산 +36% → 예측모델에 라벨 잡음(Top1% +0.149, p<0.0001).
        #
        #   수정: **GT tail 이 달성한 score 를 상한**으로 둔다. 그 아래에서는 논문대로 강화하고,
        #   도달하면 gradient 가 0 이 되어 더 밀지 않는다("분포 유지").
        #     loss = -mean( min(score, s_GT) )
        #   s_GT 는 엔진이 채우는 tail GT 미래로 같은 net 을 태워 구한다(상수 취급, detach).
        #   TAIL_SCORE_CAP=0 이면 기존 무한 최대화(구 동작) — ablation 용.
        if os.environ.get("TAIL_SCORE_CAP", "1") == "1" and "tail_gt" in ag:
            gt_w = ag["tail_gt"].to(traj.dtype)                     # (N, nfut, 2) world
            if gt_w.shape == traj.shape:
                gt_c = gt_w - origin.unsqueeze(1)
                gt_ag = torch.bmm(gt_c, rot_t)
                with torch.no_grad():
                    s_gt = net(hist_ag, gt_ag, cur_speed)           # (N,) 상수
                score = torch.minimum(score, s_gt)
        return -score[mask].mean()


class DiversityRepel(nn.Module):
    """같은 에이전트의 이전 증강본(aug) 궤적에서 반발 → aug 간 다양성 유도 (설계 메모).

    repeller(타 에이전트 반발)의 쌍대: C_div = mean relu(1 - dist/sigma).
    이전 aug 궤적과 sigma(m) 이내로 겹치는 시점만 벌점 → 모델+차로제약이 회피 방향 결정.
    - self.prev: (N, P, T, 2) world 좌표, self.prev_valid: (N, P) — 엔진이 plan 마다 설정.
      None 이면 no-op (aug0 = 자연 샘플).
    - tail/relevant 는 RealGuidance.guidance_mask 로 자동 제외(head 만 반발).
    """

    def __init__(self, sigma=None):
        super(DiversityRepel, self).__init__()
        self.sigma = (
            float(os.environ.get("DIV_REPEL_SIGMA", 1.5)) if sigma is None else sigma
        )
        self.prev = None
        self.prev_valid = None

    def loss(self, x, diffuser, data: HeteroData):
        if self.prev is None or self.prev.numel() == 0:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        traj = diffuser._reconstruct_traj(data, x)  # (N, T, 2) world
        prev = self.prev.to(device=traj.device, dtype=traj.dtype)  # (N, P, T, 2)
        if prev.shape[0] != traj.shape[0]:  # 노드수 불일치(에이전트 등장/퇴장) → skip
            return torch.zeros((), device=x.device, dtype=x.dtype)
        T = min(traj.shape[1], prev.shape[2])
        dist = torch.norm(traj[:, None, :T] - prev[:, :, :T], dim=-1)  # (N, P, T)
        a = torch.nn.functional.relu(1.0 - dist / self.sigma)
        if self.prev_valid is not None:
            a = a * self.prev_valid.to(device=a.device, dtype=a.dtype)[:, :, None]
        return a.sum() / (a.gt(0).sum() + 1e-6)


class RealGuidance(nn.Module):
    """[최적 구성] lis_big_rg(ring) 기준 모듈 절제·재설계 결과 onlane 단독이 최적.
    OFF 대비 off-road 6.9→3.7%, 중앙선침범 11.7→6.7% 동시 개선(충돌 0, J 최소).
      - repeller(factor10)는 lane 무시 횡방향 밀림으로 off-road 악화 → 제외
      - no_offroad 는 도로 안쪽(=중앙선)으로 밀어 침범 폭증 → 제외
    repeller/no_offroad 는 필요 시(혼잡·강한 off-road 억제) 아래 dict 에 다시 추가 가능.
    grad_norm=True: 모듈별 그래디언트를 정규화해 합산(여러 모듈 켤 때 스케일 지배 방지).
    """

    def __init__(self, alpha=0.5, num_step=10, beta=0.1, guidance_steps=18,
                 grad_norm=True):
        super(RealGuidance, self).__init__()
        self.alpha = alpha
        self.num_step = num_step
        self.beta = beta
        self.guidance_steps = guidance_steps
        self.grad_norm = grad_norm
        # per-agent 마스크: shape (N,1,1) float, 1=guidance 적용 / 0=미적용(예: tail).
        # 엔진이 plan 마다 설정. None 이면 전 에이전트 적용(기존 동작).
        self.guidance_mask = None
        # 활성 모듈은 환경변수 RG_MODULES(쉼표구분)로 선택 가능. 기본=onlane(프로덕션 최적).
        #   예) RG_MODULES=no_offroad        -> off-road 단독
        #       RG_MODULES=no_offroad,repeller
        _factory = {
            "onlane": OnLane,        # 같은방향 centerline soft-cost (중심선 끌어당김)
            "onlane_band": lambda: OnLaneBand(band=float(os.environ.get("BAND_M", 1.2))),
            "onlane_offset": lambda: OnLaneOffset(
                band=float(os.environ.get("BAND_M", 1.2)),
                offset=float(os.environ.get("LAT_OFFSET", 0.0)),
                w_off=float(os.environ.get("LAT_W", 1.0)),
            ),  # 횡 offset 목표로 당김(aug별 다양성 주입)
            "no_offroad": NoOffRoad,  # drivable-area 이탈 억제
            "repeller": lambda: Repeller(
                radius=float(os.environ.get("REPELLER_RADIUS", 2.5))
            ),  # 충돌 방지 (LIS 좁은 도로는 radius 축소 검토)
            # GALTraj 논문(supplementary Eq.3/4) literal 재현용 -- .
            "no_offroad_exact": lambda: NoOffRoadPaperExact(
                eps_off=float(os.environ.get("NO_OFFROAD_EPS", 0.0))
            ),
            "repeller_exact": lambda: RepellerPaperExact(
                eps_rep=float(os.environ.get("REPELLER_RADIUS", 2.5))
            ),
            "div_repel": DiversityRepel,  # 이전 aug 궤적 반발(다양성, §5)
            "velocity": Velocity,
            # tail-score(학습없음, 아주대 논문 확장): GT−pred residual. tail 에이전트에만
            # 발화(엔진이 채움). 예) RG_MODULES=onlane_band,tail_residual
            "tail_residual": TailResidual,
            # 학습된 tail-score network 기반 guidance. ckpt 는 env TAIL_SCORE_CKPT 로 지정.
            "tail_score": TailScore,
            # tail GT 이탈 상한(힌지). 학위논문 §4.3("GT 를 크게 벗어나지 않는 범위")을
            # 식 (4.5) 의 push 와 공존시킨다. d_max 이내는 gradient 0 → 다양성 보존.
            "gt_leash": lambda: GTLeash(d_max=float(os.environ.get("GT_LEASH_DMAX", 3.41))),
        }
        _names = [s.strip() for s in os.environ.get("RG_MODULES", "onlane").split(",") if s.strip()]
        self.guidance_modules = nn.ModuleDict({n: _factory[n]() for n in _names})
        self.guidance_factors = {
            # onlane 1.0→0.5: factor 1.0은 궤적이 길어질수록 누적 감속
            #   (속도 GT대비 0.85, 시간↑일수록 브레이크)을 유발. factor 탐색(12scene) 결과
            #   off-road는 factor에 거의 무관(~7.7%)하나 속도는 0.5에서 0.94로 회복(=OFF 0.93,
            #   감속 제거). 0.35~0.5 plateau 중 차선유지 마진 큰 0.5 채택.
            # 환경변수 ONLANE_FACTOR 로 오버라이드 가능(미설정 시 0.5=프로덕션 기본).
            "onlane": float(os.environ.get("ONLANE_FACTOR", 0.5)),
            "onlane_band": float(os.environ.get("ONLANE_FACTOR", 0.5)),
            "onlane_offset": float(os.environ.get("ONLANE_FACTOR", 0.5)),
            "repeller": float(os.environ.get("REPELLER_FACTOR", 0.4)),
            "no_offroad": float(os.environ.get("NO_OFFROAD_FACTOR", 1.0)),
            "no_offroad_exact": float(os.environ.get("NO_OFFROAD_FACTOR", 1.0)),
            "repeller_exact": float(os.environ.get("REPELLER_FACTOR", 0.4)),
            "div_repel": float(os.environ.get("DIV_REPEL_FACTOR", 0.5)),
            "velocity": 1.0,
            # 아주대 논문 식 (4.5) 의 c (guidance scale). grad_norm 으로 방향만 쓰이므로 상대 가중.
            "tail_residual": float(os.environ.get("TAIL_SCORE_FACTOR", 1.0)),
            "tail_score": float(os.environ.get("TAIL_SCORE_FACTOR", 1.0)),
            # 이탈 상한. tail_score(=미는 힘)와 같은 스케일에서 시작해 게이트로 조정한다.
            "gt_leash": float(os.environ.get("GT_LEASH_FACTOR", 1.0)),
        }
        # tail 에이전트에 guidance_mask(=tail 제외)를 적용하지 '않을' 모듈 집합.
        #   tail_residual/tail_score 는 항상(=tail 을 미는 게 목적).
        #
        # tail 에 realism 을 거는 조건을 "tail push 모듈이 켜져 있을 때"로 한정.
        #
        # GALTraj §3.4.2 는 gradient guidance 를 head 에만 거는 이유를 명시한다:
        #   "While tail and relevant agents are guided by real constraints, head agents are
        #    not directly constrained, which may result in the violation of traffic rules."
        # 즉 tail 의 realism 은 real guidance 앵커(λ_tail=0.25, Eq.3)가 보장하므로 gradient
        # guidance 가 불필요하다는 것 — head-only(RG_GUID_HEAD_ONLY=1)는 논문 그대로다.
        #
        # 그런데 tail_score/tail_residual(학위논문 식 4.5 확장)은 tail 을 앵커시키면서 동시에
        # '밀어낸다'. 논문이 상정하지 않은 경우이고, 미는 순간 위 전제("앵커되어 있으므로
        # 안전")가 무효가 되어 논문 자신의 논리상 tail 에도 규칙 준수 제약이 필요해진다.
        # 실측: 구 동작(tail realism 없음)으로 생성한 tail-score 세트는 144 시나리오 중
        #   114(79%)가 실 주행 최대이탈(3.41m) 초과, 최대 32.09m — §3.4 가 경고한
        #   "unrealistic training samples ... degrading overall performance" 그대로.
        # 또한 구 RG_TAIL_REALISM=1 경로에는 paper-exact 변형(*_exact)이 누락돼 있어
        #   F10 재현 설정(no_offroad_exact,repeller_exact)에서는 켜도 무효였다. → 함께 포함.
        #
        # tail push 가 없는 순정 재현(RG_MODULES 에 tail_* 없음)은 논문 그대로 head-only 유지.
        # RG_TAIL_REALISM 으로 강제 가능: 1=항상 적용, 0=항상 미적용(구 동작 재현·대조용).
        # gt_leash 는 tail 을 붙잡는 항이므로 tail 에 반드시 걸려야 한다(tail push 와 짝).
        self._tail_unmasked = {"tail_residual", "tail_score", "gt_leash"}
        _pushing_tail = bool(self._tail_unmasked & set(self.guidance_modules))
        _realism = os.environ.get("RG_TAIL_REALISM", "auto")
        if _realism == "1" or (_realism == "auto" and _pushing_tail):
            self._tail_unmasked |= {"onlane_band", "onlane", "no_offroad",
                                    "no_offroad_exact", "repeller_exact"}

    @torch.enable_grad()
    def forward(
        self,
        x: torch.Tensor,
        diffuser,
        data: HeteroData,
        sigma: float = None,
    ):
        """
        x: (agent_num, time_step, 2) output of the model
        his_states: historical states of the agents
        diffuser: the diffuser module
        """
        x = x.clone().detach().requires_grad_(True)
        x_0 = x.clone().detach()
        x_safe = x_0  # 마지막으로 유한했던 결과 (NaN 발생 시 폴백)
        # RG_SIGMA_SCALE=1: 논문 Eq.4의 Σ^k 스케일링 근사 — 보정 강도를 현재 스텝
        # 노이즈 σ에 비례(w=σ/(σ+1), 유계·단조)시켜 고노이즈에서 강하게, 수렴부에서
        # 약하게. 기본 off = 기존 고정 α (grad_norm 정규화와 병행).
        alpha_eff = self.alpha
        if sigma is not None and os.environ.get("RG_SIGMA_SCALE", "0") == "1":
            alpha_eff = self.alpha * (sigma / (sigma + 1.0))
        for _ in range(self.num_step):
            if self.grad_norm:
                # 모듈별 그래디언트를 ‖·‖ 정규화 후 factor 가중 합산.
                grad = torch.zeros_like(x)
                bad = False
                for name, module in self.guidance_modules.items():
                    li = module.loss(x, diffuser, data)
                    if not torch.isfinite(li):
                        bad = True
                        break
                    try:
                        gi = torch.autograd.grad(li, x, retain_graph=True)[0]
                    except RuntimeError:
                        continue  # 상수 loss → 이 모듈은 건너뜀
                    if gi is None or not torch.isfinite(gi).all():
                        continue
                    gi_n = self.guidance_factors[name] * gi / (gi.norm() + 1e-6)
                    # per-module 마스크: _tail_unmasked(기본 tail_residual, RG_TAIL_REALISM=1 시
                    #   onlane_band/no_offroad 포함) 는 tail 에도 작동해야 하므로 guidance_mask
                    #   (=tail 제외) 미적용. 나머지는 head/relevant 만.
                    if (
                        name not in self._tail_unmasked
                        and self.guidance_mask is not None
                        and self.guidance_mask.shape[0] == gi_n.shape[0]
                    ):
                        gi_n = gi_n * self.guidance_mask.to(gi_n.dtype)
                    grad = grad + gi_n
                if bad:
                    break
            else:
                loss = 0
                for name, module in self.guidance_modules.items():
                    loss += module.loss(x, diffuser, data) * self.guidance_factors[name]
                # 안전장치: loss/grad 가 비정상이면 보정 중단 → 마지막 유한값 반환.
                # (고노이즈 스텝의 거대한 궤적 등에서 드물게 NaN 발생 → 궤적 오염 방지)
                if not torch.isfinite(loss):
                    break
                try:
                    grad = torch.autograd.grad(loss, x)[0]
                except RuntimeError:
                    break  # loss 가 x 에 의존 안 함(상수) → 보정 불가
                if grad is None or not torch.isfinite(grad).all():
                    break
            if not torch.isfinite(grad).all():
                break
            # per-agent 마스크: 지정된 에이전트(예: tail)는 guidance 미적용(grad 0)
            #   grad_norm 경로는 위에서 모듈별로 이미 적용(tail_residual 등 면제)했으므로 건너뜀.
            if (
                not self.grad_norm
                and self.guidance_mask is not None
                and self.guidance_mask.shape[0] == grad.shape[0]
            ):
                grad = grad * self.guidance_mask.to(grad.dtype)
            x = x - alpha_eff * grad
            delta = torch.clip(x - x_0, -self.beta, self.beta)
            x = x_0 + delta
            x_safe = x.detach()
        return x_safe
