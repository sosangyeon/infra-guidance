"""v7 membership + anchor-pull 가이던스를 우리 LCSim 엔진(lcsim/engine)에 꽂는 판본.

원본: _lcsim_patched/motion_diff/modules/within_cluster_guidance.py (WithinClusterGuidance)
      + anchor_pull_guidance.py (AnchorPullGuidance). 손실 식은 그대로다.

  c*_i       = argmax_k cos(e(φ(GT 미래_i)), p_k)   (정지 = K, 가이던스 0)
  J_mem_i    = relu(δ − (cos(e_i, p_c*) − max_{k≠c*} cos(e_i, p_k)))
  J_anchor_i = mean_t ‖x̂₀_i(t) − GT_i(t)‖²  ·  1[gap_i < δ]   (pull_scope = violators)
  J          = mean_{i∈대상} (J_mem_i + w · J_anchor_i)          (β_div = 0: 다양성 반발 끔, v7 설정)
  denoised   : x̂₀ = D(x, σ) 에서 정규화 gradient 로 num_step 번 내려가고, x 에 clip(x̂₀' − x̂₀, ±anchor) 를 더한다

v7 설정값 (method.yaml c2 + overrides_port_v7_tutr.yaml): δ 0.05, α 0.1, numstep 10, anchor(clip) 0.8,
  denoised true, membership_scope all, anchor_pull 1.0 / violators, β_div 0.

우리 엔진과 다른 점
  - 엔진이 ids_list 순서로 batch 를 만들므로, 생성 스크립트가 아니라 엔진이 plan 마다
    set_scene(ids, gt_future, scene_enc) 을 부른다. GT 미래·c* 는 첫 plan(plan_init_step) 에서 정한다.
  - 우리 샘플러는 18 스텝이라, v7 의 "50 스텝 중 마지막 18 스텝"과 σ 구간을 맞춰
    guidance_steps 기본 7 (18 스텝 중 마지막 7 스텝)을 쓴다.

tail_mode (ours13 에서 추가)
  keep : 원본 v7. 모든 차량의 c* = GT 클러스터 (기동 보존).
  push : tail 차량만 c* 를 GT 가 아닌 "이웃 클러스터" 로 바꿔 기동을 옆 칸으로 민다. 이웃 차량은 keep 그대로.
         후보 = GT 임베딩과 cos 가 큰 순서로 push_alt 개의 비-GT 프로토타입. 그중 학습셋 빈도(vocab labels)가
         낮은 순으로 두 개를 골라 aug_id(RG_V7_AUG_ID) 짝/홀로 번갈아 쓴다 → 한 장면 안에서 두 방향의 희귀 기동.
         push 대상에는 anchor pull(GT 로 당김)을 걸지 않는다 (목표와 모순). 정지(c*=K) 차량은 그대로 둔다.
  이유: test 꼬리(상위 1 %)는 회전·가감속이 큰 단일 차량 기동이고(ours12 분석), 기동을 보존하는 생성은 그 꼬리에
        새 표본을 주지 못했다. 옆 칸 기동은 GT 와 가깝지만 모델이 덜 본 쪽이다.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from v7lib import maneuver as mv


class MembershipAnchorGuidance(nn.Module):
    def __init__(self, vocab_path, margin=0.05, alpha=0.1, num_step=10, anchor=0.8,
                 anchor_w=1.0, pull_scope="violators", guidance_steps=7, denoised=True, scope="all",
                 tail_mode="keep", push_alt=3):
        super().__init__()
        self.mlp, ck = mv.load_vocab(vocab_path)
        self.register_buffer("protos", F.normalize(ck["protos"].float(), dim=1))
        self.register_buffer("mean", ck["mean"].float())
        self.register_buffer("std", ck["std"].float())
        self.register_buffer("grad_w", ck["grad_w"].float())
        self.K = int(ck["K"])
        self.stop_path = float(ck["min_speed"])
        self.margin = float(margin) if margin is not None and margin >= 0 else float(ck["margin"])
        self.alpha, self.num_step, self.anchor = alpha, num_step, anchor
        self.anchor_w, self.pull_scope = float(anchor_w), pull_scope
        self.guidance_steps, self.denoised, self.scope = int(guidance_steps), bool(denoised), scope
        self.tail_mode, self.push_alt = str(tail_mode), int(push_alt)
        # 클러스터별 학습셋 빈도 (push 후보 중 희귀한 쪽을 고르는 데 씀)
        lab = torch.as_tensor(ck["labels"]).long() if "labels" in ck else torch.zeros(0, dtype=torch.long)
        self.register_buffer("freq", torch.bincount(lab[lab < self.K], minlength=self.K).float())
        self.guidance_mask = None          # 엔진이 넣지만 쓰지 않는다 (scope 로 정함)
        self.c_star = self.gt_future = self.target_mask = self.push_mask = self.scene_enc = None
        self._c_by_id, self._gt_by_id, self._push_by_id = {}, {}, {}
        self.stats = []
        print(f"[v7-guid] vocab {vocab_path} K={self.K} δ={self.margin} α={alpha} n={num_step} clip={anchor} "
              f"w={anchor_w}/{pull_scope} steps={guidance_steps} denoised={denoised} scope={scope} "
              f"tail_mode={self.tail_mode}/{self.push_alt}", flush=True)

    @torch.no_grad()
    def _push_targets(self, fut, is_tail):
        """tail 차량의 목표 클러스터: GT 임베딩과 가까운 비-GT 프로토타입 push_alt 개 중 학습 빈도가 낮은 두 개를
        aug_id 짝/홀로 번갈아 고른다. 반환 (c*, pushed) — 정지·비-tail 은 GT 클러스터 그대로, pushed False."""
        can = mv.canonicalize_future(fut.float())
        cos = mv.embed(self.mlp.to(can.device), mv.phi_c2(can), self.mean, self.std) @ self.protos.t()
        stat = (can[:, 1:] - can[:, :-1]).norm(dim=-1).sum(1) < self.stop_path
        gt = cos.argmax(1)
        aug_id = int(os.environ.get("RG_V7_AUG_ID", "0"))
        c = torch.where(stat, torch.full_like(gt, self.K), gt).clone()
        pushed = torch.zeros_like(stat)
        for j in range(len(fut)):
            if not is_tail[j] or bool(stat[j]):
                continue
            order = cos[j].argsort(descending=True)
            cand = order[order != gt[j]][: max(1, self.push_alt)]
            cand = cand[self.freq[cand].argsort()]              # 희귀한 순
            c[j] = cand[aug_id % min(2, len(cand))]
            pushed[j] = True
        return c, pushed

    # ── 엔진 연결 ──
    @torch.no_grad()
    def assign(self, fut):
        can = mv.canonicalize_future(fut.float())
        c = (mv.embed(self.mlp.to(can.device), mv.phi_c2(can), self.mean, self.std) @ self.protos.t()).argmax(1)
        stat = (can[:, 1:] - can[:, :-1]).norm(dim=-1).sum(1) < self.stop_path
        return torch.where(stat, torch.full_like(c, self.K), c)

    def set_scene(self, ids, gt_future, tail_ids, scene_enc, first_plan):
        """ids: batch 순서 agent id, gt_future [A,T,2] world (첫 plan 에서만 GT 로 의미 있음)."""
        dev = self.protos.device
        if first_plan:
            if self.tail_mode == "push":
                c, pushed = self._push_targets(gt_future.to(dev), [i in tail_ids for i in ids])
            else:
                c = self.assign(gt_future.to(dev)); pushed = torch.zeros(len(ids), dtype=torch.bool, device=dev)
            for j, i in enumerate(ids):
                self._c_by_id[i] = int(c[j]); self._gt_by_id[i] = gt_future[j].detach().to(dev)
                self._push_by_id[i] = bool(pushed[j])
        self.c_star = torch.tensor([self._c_by_id.get(i, self.K) for i in ids], device=dev)
        self.push_mask = torch.tensor([self._push_by_id.get(i, False) for i in ids], device=dev)
        self.gt_future = torch.stack([self._gt_by_id.get(i, gt_future[j].to(dev)) for j, i in enumerate(ids)])
        self.target_mask = (torch.ones(len(ids), dtype=torch.bool, device=dev) if self.scope == "all"
                            else torch.tensor([i in tail_ids for i in ids], device=dev))
        self.scene_enc = scene_enc

    # ── 손실 (AnchorPullGuidance._loss 와 같은 정의, num_modes = 1) ──
    def _loss(self, x, diffuser, data, record=False):
        traj = diffuser._reconstruct_traj(data, x)
        e = mv.embed(self.mlp.to(traj.device), mv.phi_c2(mv.canonicalize_future(traj.float())),
                     self.mean, self.std, grad_w=self.grad_w)
        cosk = e @ self.protos.t()
        A = cosk.shape[0]
        ar = torch.arange(A, device=cosk.device)
        cs = self.c_star
        stat = cs >= self.K
        cs_safe = torch.where(stat, torch.zeros_like(cs), cs)
        pos = cosk[ar, cs_safe]
        neg = cosk.clone(); neg[ar, cs_safe] = -2.0; neg = neg.max(1).values
        gap = pos - neg
        loss = F.relu(self.margin - gap) * (~stat).float()
        push = self.push_mask if self.push_mask is not None else torch.zeros_like(stat)
        if self.anchor_w > 0:
            J_anchor = ((traj.float() - self.gt_future.float()) ** 2).sum(-1).mean(-1)
            pull = (~stat) if self.pull_scope == "all" else ((gap.detach() < self.margin) & (~stat))
            pull = pull & ~push                      # push 대상은 GT 로 당기지 않는다
            loss = loss + self.anchor_w * J_anchor * pull.float()
        m = self.target_mask.float()
        if record:
            with torch.no_grad():
                mv_ = (m > 0) & ~stat
                rec = {"n": int(mv_.sum()), "retention": float((gap[mv_] > 0).float().mean()) if mv_.any() else float("nan"),
                       "viol": float((gap[mv_] < self.margin).float().mean()) if mv_.any() else float("nan")}
                if push.any():
                    rec["push_n"] = int(push.sum()); rec["push_reach"] = float((gap[push] > 0).float().mean())
                self.stats.append(rec)
        return (loss * m).sum() / m.sum().clamp(min=1.0)

    @torch.enable_grad()
    def forward(self, x, diffuser, data, sigma=None):
        if self.c_star is None or self.c_star.shape[0] != x.shape[0]:
            return x
        if self.denoised and self.scene_enc is not None and sigma is not None:
            s = torch.as_tensor(max(float(sigma), 0.02), device=x.device)
            with torch.no_grad():
                xh0 = diffuser._edm_precondition(data, x, s, self.scene_enc).detach().to(x.dtype)
            if not torch.isfinite(xh0).all():
                return x
            xh = xh0.clone().requires_grad_(True)
            for it in range(self.num_step):
                loss = self._loss(xh, diffuser, data, record=(it == 0))
                grad = torch.nan_to_num(torch.autograd.grad(loss, xh)[0], 0.0, 0.0, 0.0)
                gn = grad.flatten(1).norm(dim=1).clamp(min=1e-6)
                xh = torch.nan_to_num((xh - self.alpha * grad / gn.view(-1, *([1] * (grad.dim() - 1)))).detach(),
                                      0.0, 0.0, 0.0).requires_grad_(True)
            nudge = torch.nan_to_num(torch.clip(xh.detach() - xh0, -self.anchor, self.anchor), 0.0, 0.0, 0.0)
            return x + nudge
        x0 = x.clone().detach()
        xg = x0.clone().requires_grad_(True)
        for it in range(self.num_step):
            loss = self._loss(xg, diffuser, data, record=(it == 0))
            grad = torch.nan_to_num(torch.autograd.grad(loss, xg)[0], 0.0, 0.0, 0.0)
            gn = grad.flatten(1).norm(dim=1).clamp(min=1e-6)
            delta = torch.clip((xg - self.alpha * grad / gn.view(-1, *([1] * (grad.dim() - 1)))) - x0,
                               -self.anchor, self.anchor)
            xg = (x0 + delta).detach().requires_grad_(True)
        return xg.detach()


class PhysicsMembershipGuidance(nn.Module):
    """물리(RealGuidance: no_offroad_exact + repeller_exact, head 만) + membership/anchor 를 같은 샘플러에 함께 건다.

    샘플러는 i >= num_steps − guidance_steps 인 스텝마다 한 번 부른다 (model.py).
    guidance_steps = max(물리 18, membership 7) 로 두고, 한 번의 샘플링 안에서 몇 번째 호출인지 세어
      - 물리      : 마지막 phys.guidance_steps 스텝
      - membership: 마지막 mem.guidance_steps 스텝 (v7 과 같음)
    에 적용한다. 같은 스텝에서는 물리 → membership 순서. σ 가 직전 호출보다 커지면 새 샘플링으로 보고 카운터를 0 으로 되돌린다.
    엔진이 넣는 guidance_mask(head 만)는 물리에만, set_scene 은 membership 에만 전달된다.
    """

    def __init__(self, phys, mem):
        super().__init__()
        self.phys, self.mem = phys, mem
        self.guidance_steps = max(int(phys.guidance_steps), int(mem.guidance_steps))
        self._i, self._last_sigma = 0, None
        print(f"[v7-guid] 물리+membership: 물리 마지막 {phys.guidance_steps} 스텝, "
              f"membership 마지막 {mem.guidance_steps} 스텝", flush=True)

    @property
    def guidance_mask(self):
        return self.phys.guidance_mask

    @guidance_mask.setter
    def guidance_mask(self, m):
        if "phys" in self._modules:
            self.phys.guidance_mask = m

    @property
    def stats(self):
        return self.mem.stats

    def set_scene(self, *a, **k):
        return self.mem.set_scene(*a, **k)

    def forward(self, x, diffuser, data, sigma=None):
        if sigma is not None and self._last_sigma is not None and sigma > self._last_sigma:
            self._i = 0
        self._last_sigma = sigma
        left = self.guidance_steps - self._i          # 이 호출을 포함해 남은 가이던스 스텝 수
        self._i = (self._i + 1) % self.guidance_steps
        if left <= self.phys.guidance_steps:
            x = self.phys(x, diffuser, data, sigma=sigma)
        if left <= self.mem.guidance_steps:
            x = self.mem(x, diffuser, data, sigma=sigma)
        return x
