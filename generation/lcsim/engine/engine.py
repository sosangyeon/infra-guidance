import csv
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch_geometric.data import Batch

from motion_diff.model import MotionDiff
from motion_diff.modules.guidance import RealGuidance

from ..entity import Action
from ..entity.agent.manager import AgentManager
from ..entity.gen.agent import agent_pb2
from ..entity.gen.map import map_pb2
from ..entity.junction.manager import JunctionManager
from ..entity.lane.manager import LaneManager
from ..entity.road.manager import RoadManager
from ..entity.utils import polygon


class Engine:
    """
    Simulation engine.
    """

    # config
    config: dict
    # managers
    road_manager: RoadManager
    lane_manager: LaneManager
    junction_manager: JunctionManager
    agent_manager: AgentManager
    # deep models
    motion_diffuser: MotionDiff
    guidance: RealGuidance = None
    # clock
    start_time: float
    total_steps: int
    step_interval: float
    current_time: float
    current_step: int
    # control
    enable_poly_emb: bool
    enable_plan: bool
    plan_interval: int
    plan_init_step: int
    device: str
    allow_new_obj: bool
    reactive: bool
    no_static: bool

    def __init__(self, config: dict):
        """init simulation engine from config file.

        Args:
            config (dict): config file.
        """
        self.config = config

        # init clock
        step_config = config["control"]["step"]
        self.start_time = self.current_time = step_config["start"]
        self.total_steps = step_config["total"]
        self.step_interval = step_config["interval"]
        self.current_step = 0

        # control
        control_config = config["control"]
        self.enable_poly_emb = control_config["enable_poly_emb"]
        self.enable_plan = control_config["enable_plan"]
        if self.enable_plan:
            assert self.enable_poly_emb
        self.plan_interval = control_config["plan_interval"]
        self.plan_init_step = control_config["plan_init_step"]
        self.device = control_config["device"]
        self.allow_new_obj = control_config["allow_new_obj"]
        self.reactive = control_config["reactive"]
        self.no_static = control_config["no_static"]

        # reward
        self.reward_config = config["reward"]

        # init entity managers
        input_config = config["input"]
        data_dir = input_config["data_dir"]
        path_map, path_agents = os.path.join(data_dir, "map.pb"), os.path.join(
            data_dir, "agents.pb"
        )
        map_pb = map_pb2.Map()
        map_pb.ParseFromString(open(path_map, "rb").read())
        self.lane_manager = LaneManager(map_pb.lanes)
        self.road_manager = RoadManager(map_pb.roads, self.lane_manager)
        self.junction_manager = JunctionManager(
            map_pb.junctions, self.lane_manager, self
        )
        agents_pb = agent_pb2.Agents()
        agents_pb.ParseFromString(open(path_agents, "rb").read())
        self.agent_manager = AgentManager(agents_pb, self)

        # real guidance (SDEdit noised-GT start) — optional, off by default.
        # config["control"]["real_guidance"] = {
        #     "tail_json": "<path to tail_dict_per_scenario.json>",
        #     "lambda_tail": 0.25,   # optional, default 0.25
        # }
        #     "lambda_tail": 0.25,   # optional, default 0.25
        #     "lambda_rel": 0.6,     # optional, default 0.6 (relevant agents)
        #     "relevant": True,      # optional, default True (attention-based)
        # }
        rg_cfg = control_config.get("real_guidance", None)
        self.real_guidance_enabled = bool(rg_cfg)
        self.tail_ids = set()
        self.lambda_tail = 0.25
        self.lambda_rel = 0.6
        self.relevant_enabled = True
        if self.real_guidance_enabled:
            self.lambda_tail = float(rg_cfg.get("lambda_tail", 0.25))
            self.lambda_rel = float(rg_cfg.get("lambda_rel", 0.6))
            self.relevant_enabled = bool(rg_cfg.get("relevant", True))
            sid = os.path.basename(os.path.normpath(data_dir))
            tail_dict = json.load(open(rg_cfg["tail_json"], "r", encoding="utf-8"))
            self.tail_ids = set(int(t) for t in tail_dict.get(sid, []))

            # tail-score(TailResidual)용 외부 예측궤적 pred 로드(선택, GAL-ours 계보).
            #   real_guidance.pred_dir/<sid>/scenario_<sid>.csv 의 pred_x_minade/pred_y_minade
            #   (world) 를 track_id·timestep 별로 적재 → _build_real_guidance 에서 residual=GT−pred.
            #   미지정/파일없음/컬럼없음 이면 self.pred_traj={} → tail_residual 자동 no-op.
            self.pred_traj = {}  # {track_id: {timestep: (px, py)}}
            pred_dir = rg_cfg.get("pred_dir", None)
            if pred_dir:
                pcsv = os.path.join(pred_dir, sid, f"scenario_{sid}.csv")
                if os.path.isfile(pcsv):
                    with open(pcsv, newline="", encoding="utf-8") as _f:
                        for row in csv.DictReader(_f):
                            px, py = row.get("pred_x_minade", ""), row.get("pred_y_minade", "")
                            if px in ("", "nan", "NaN") or py in ("", "nan", "NaN"):
                                continue
                            try:
                                tid = int(float(row["track_id"]))
                                ts = int(float(row["timestep"]))
                                self.pred_traj.setdefault(tid, {})[ts] = (float(px), float(py))
                            except (ValueError, KeyError, TypeError):
                                continue

        # 분기(route) 다양성: head 를 "배정된 분기의 합성 참조궤적"에 real guidance 로
        # 앵커링(반사실적 경로 조건화). env 제어(Engine 이 rollout=aug 마다 새로 생성됨):
        #   RG_BRANCH=1            활성화 (real_guidance 켜진 경우만)
        #   RG_BRANCH_AUG=<k>      분기 배정 인덱스(회전각 정렬 후 k % n 선택)
        #   RG_LAMBDA_ROUTE=0.5    앵커 강도(작을수록 참조에 밀착)
        self.branch_enabled = (
            os.environ.get("RG_BRANCH", "0") == "1" and self.real_guidance_enabled
        )
        self.branch_aug = int(os.environ.get("RG_BRANCH_AUG", "0"))
        self.lambda_route = float(os.environ.get("RG_LAMBDA_ROUTE", "0.5"))
        if self.branch_enabled:
            self._branch_cl = {
                l.id: np.array([(n.x, n.y) for n in l.center_line.nodes])
                for l in map_pb.lanes
            }
            self._branch_succ = {l.id: [s.id for s in l.successors] for l in map_pb.lanes}
            # rollout 내 배정 고정(sticky): agent_id -> 선택한 분기 lane id.
            # replan 마다 재배정하면 배정이 뒤집혀(차로매칭/정렬 변동) 지그재그 실행이 됨.
            self._branch_choice: Dict[int, int] = {}

        # init deep models
        try:
            model_path = input_config["model_path"]
            model_config_path = input_config["model_config_path"]
            model_config = yaml.safe_load(open(model_config_path, "r", encoding="utf-8"))
            self.motion_diffuser = MotionDiff.load_from_checkpoint(
                checkpoint_path=model_path,
                map_location=self.device,
                cfg=model_config,
            )
            self.motion_diffuser.eval()
            for para in self.motion_diffuser.parameters():
                para.requires_grad = False
            if control_config["guidance"]:
                self.guidance = RealGuidance()
                # RG_V7_VOCAB 가 있으면 물리 가이던스 대신 membership + anchor-pull
                if os.environ.get("RG_V7_VOCAB", "").strip():
                    from v7lib.membership_guidance import MembershipAnchorGuidance
                    self.guidance = MembershipAnchorGuidance(
                        os.environ["RG_V7_VOCAB"],
                        margin=float(os.environ.get("RG_V7_MARGIN", "0.05")),
                        alpha=float(os.environ.get("RG_V7_ALPHA", "0.1")),
                        num_step=int(os.environ.get("RG_V7_NUMSTEP", "10")),
                        anchor=float(os.environ.get("RG_V7_CLIP", "0.8")),
                        anchor_w=float(os.environ.get("RG_V7_ANCHOR_W", "1.0")),
                        pull_scope=os.environ.get("RG_V7_PULL_SCOPE", "violators"),
                        guidance_steps=int(os.environ.get("RG_V7_STEPS", "7")),
                        denoised=os.environ.get("RG_V7_DENOISED", "1") == "1",
                        scope=os.environ.get("RG_V7_SCOPE", "all"),
                        tail_mode=os.environ.get("RG_V7_TAIL_MODE", "keep"),      # ours13: push = tail 기동을 이웃 클러스터로
                        push_alt=int(os.environ.get("RG_V7_PUSH_ALT", "3")),
                    ).to(self.device)
                    # hybrid (RG_V7_PHYS=1): 물리 가이던스(RG_MODULES, head 만)도 함께 건다
                    if os.environ.get("RG_V7_PHYS", "0") == "1":
                        from v7lib.membership_guidance import PhysicsMembershipGuidance
                        self.guidance = PhysicsMembershipGuidance(
                            RealGuidance().to(self.device), self.guidance
                        ).to(self.device)
        except Exception as e:
            print(f"Failed to load deep models: {e}")
            self.motion_diffuser = None

        if self.enable_poly_emb:
            # init polyline embeddings for lanes, roads and junctions
            self.lane_manager.init_polyline_emb(self.motion_diffuser)
            self.road_manager.init_polyline_emb(self.motion_diffuser)
            self.junction_manager.init_polyline_emb(self.motion_diffuser)
            self.junction_manager.init_roadgraph_data()

        # prepare
        self.agent_manager.check_departure_and_arrival(self.current_time)
        self.lane_manager.prepare()
        self.agent_manager.prepare_obs()

    def reset(self):
        """
        Reset the simulation engine to the initial state.
        """
        self.current_time = self.start_time
        self.current_step = 0
        # reset managers
        self.agent_manager.reset()
        self.lane_manager.reset()
        # prepare
        self.agent_manager.check_departure_and_arrival(self.current_time)
        self.agent_manager.prepare_obs()

    def step(self, actions: Dict[int, Action] = {}):
        """Step the simulation engine.

        Args:
            actions (Dict[int, Action]):  outer actions, agents provided won't be update by engine itself.

        Returns:
            Dict[int, Observation]: observations for agents.
        """
        self.update(actions)
        self.prepare()
        # update clock
        self.current_time += self.step_interval
        self.current_step += 1

    def get_step_return(self, agent_id: int):
        """
        Get the return of the current step, [obs, reward, terminated, truncated, info].
        """
        agent = self.agent_manager.get_agent_by_id(agent_id)
        assert agent is not None
        if not agent.runtime.is_finished():
            data_list, ids_list = [], []
            for junc in self.junction_manager.junctions.values():
                hetero = junc.get_hetero_data()
                if hetero is None:
                    continue
                data_list.append(hetero[0])
                ids_list.extend(hetero[1])
            if len(data_list) == 0:
                return
            batch = Batch.from_data_list(data_list).to(self.device)
            scene_enc = self.motion_diffuser.agent_encoder(
                batch, batch["roadgraph"]["poly_emb"]
            )["agent"]
            ads_index = ids_list.index(self.agent_manager.ads_id)
            scene_emb = scene_enc[ads_index, -1].detach().cpu().numpy()
            route = self.agent_manager.get_ads().get_route()
            obs = {
                "scene_emb": scene_emb,
                "route": route,
            }
            torch.cuda.empty_cache()
        else:
            obs = {
                "scene_emb": np.zeros(128),
                "route": np.zeros((10, 2)),
            }
        rewards, terminated = self._compute_rewards(
            self.reward_config.keys(), self.agent_manager.ads_id
        )
        reward = sum([rewards[key] * coef for key, coef in self.reward_config.items()])
        truncated = (
            self.current_step >= self.total_steps
            or self.agent_manager.get_ads().runtime.is_finished()
        )
        return obs, reward, terminated, truncated, {"rewards": rewards}

    def prepare(self):
        """
        Prepare stage: check departure and arrival, prepare observation for agents.
        """
        # prepare
        self.agent_manager.check_departure_and_arrival(
            self.current_time, self.allow_new_obj
        )
        self.lane_manager.prepare()
        self.agent_manager.prepare_obs()
        # plan trajectory
        if (
            self.enable_plan
            and self.current_step >= self.plan_init_step
            and (self.current_step - self.plan_init_step) % self.plan_interval == 0
        ):
            self._plan_trajectory()

    def update(self, actions: Dict[int, Action] = {}):
        """
        Update stage: get action for each agent, update agent status.
        """
        self.agent_manager.update(self.step_interval, actions)

    def render(self) -> np.ndarray:
        """
        Render the current simulation state, return the rendered image.
        """
        config = self.config["render"]
        if config["center_type"] == "agent":
            return self.agent_manager.render(config)
        elif config["center_type"] == "junction":
            return self.junction_manager.render(config)
        else:
            raise ValueError(f"Unknown center type {config['center_type']}")

    def release_deep_models(self):
        """
        Release deep models for saving memory.
        """
        self.motion_diffuser = None
        torch.cuda.empty_cache()

    def load_deep_models(self):
        """
        Load deep models.
        """
        model_path = self.config["input"]["model_path"]
        model_config_path = self.config["input"]["model_config_path"]
        model_config = yaml.safe_load(open(model_config_path, "r", encoding="utf-8"))
        self.motion_diffuser = MotionDiff.load_from_checkpoint(
            checkpoint_path=model_path,
            map_location=self.device,
            cfg=model_config,
        )
        self.motion_diffuser.eval()
        for para in self.motion_diffuser.parameters():
            para.requires_grad = False

    def _relevant_nodes(self, batch, scene_enc, tail_nodes: List[int]) -> set:
        """
        Identify *relevant* agents (paper Sec 3.4.1): agents a tail attends to
        above 1/|N_j| in the decoder's agent→future (a2m) attention, evaluated
        near the last denoising step (low sigma) with a **noised-GT future**
        query so attention reflects realistic future interaction.
        Requires batch["agent"]["target"] to be populated first.

        NOTE(검증됨): 논문의 future-future relevance 는 *별도 학습한 cross-agent 모듈* 필요.
        본 LCSim 의 `m2m` 은 구조적으로 self-only(mask_dst (N,1)→dense_to_sparse 가
        N개의 1×1 배치로 해석 → self-loop 만)라 cross-agent 가 아님. 즉 이 코드베이스에서
        에이전트 상호작용을 담는 것은 `a2m`. 재학습 없이 쓸 수 있는 cross-agent 신호는
        a2m 뿐이라 이를 미래 query 로 평가해 근사한다. (순정 future-future = 추후 Path B)
        """
        model = self.motion_diffuser
        dec = model.diff_decoder
        a2m_layers = dec.a2m_attn_layers  # agent(history) → mode(future) attention
        for layer in a2m_layers:
            layer.reset_capture()
            layer.capture_attn = True
        try:
            gt = model._get_training_targets(batch).to(torch.float32)  # (N, F, out_dim)
            out_dim = model.output_dim + model.output_head
            if gt.shape[-1] < out_dim:
                pad = out_dim - gt.shape[-1]
                gt = torch.cat([gt, gt.new_zeros(*gt.shape[:-1], pad)], dim=-1)
            # last-denoising-step proxy: low sigma + noised-GT future query
            sigma_small = 0.02
            x = gt + torch.randn_like(gt) * sigma_small
            sigma = torch.tensor(sigma_small, device=self.device)
            with torch.no_grad():
                model._edm_precondition(batch, x, sigma, scene_enc)
            edge = dec.last_edge_index_a2m  # (2, E): src=row0, dst=row1
            attns = [l.get_captured_attn() for l in a2m_layers]
            attns = [a for a in attns if a is not None]
            if edge is None or not attns or edge.shape[1] != attns[0].shape[0]:
                return set()
            aw = torch.stack(attns, dim=0).mean(dim=0)  # (E,) mean over a2m layers
        finally:
            for layer in a2m_layers:
                layer.capture_attn = False
                layer.reset_capture()

        src, dst = edge[0], edge[1]
        # in-degree (|N_j|, neighbors) per destination node
        indeg = torch.zeros(batch["agent"]["num_nodes"], device=self.device)
        indeg.scatter_add_(0, dst, torch.ones_like(dst, dtype=indeg.dtype))
        tail_set = set(tail_nodes)
        relevant = set()
        for e in range(dst.shape[0]):
            j = int(dst[e])
            i = int(src[e])
            if j in tail_set and i not in tail_set and indeg[j] > 0:
                if float(aw[e]) > 1.0 / float(indeg[j]):
                    relevant.add(i)
        return relevant

    def _branch_chains(self, lid: int, idx: int, need: float) -> List[List[int]]:
        """현재 차로 idx 이후 successor 체인 열거(need[m] 도달까지). 각 체인=분기 후보."""
        chains = []

        def walk(chain, remain):
            nxt = [s for s in self._branch_succ.get(chain[-1], []) if s in self._branch_cl]
            if remain <= 0 or not nxt:
                chains.append(chain)
                return
            for s in nxt:
                p = self._branch_cl[s]
                seg = (
                    float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
                    if len(p) > 1
                    else 0.0
                )
                walk(chain + [s], remain - seg)

        p0 = self._branch_cl[lid][idx:]
        rem0 = (
            float(np.linalg.norm(np.diff(p0, axis=0), axis=1).sum()) if len(p0) > 1 else 0.0
        )
        walk([lid], need - rem0)
        return chains

    def _branch_reference(
        self, agent_id: int, xy, heading: float, v: float, nfut: int, dt: float = 0.1
    ):
        """분기 후보>=2 면 RG_BRANCH_AUG 로 분기를 골라 등속 합성 참조궤적 (nfut,2)[world]
        를 반환. 후보<2(분기 없음/도달 불가) 또는 저속이면 None(기존 head 동작 유지).
        단 이미 배정된 에이전트(sticky)는 후보 1개여도 계속 앵커링(회전 관철)."""
        if v < 0.5:
            return None
        hv = np.array([math.cos(heading), math.sin(heading)])
        best = (1e9, None, None)
        for lid, pts in self._branch_cl.items():
            if len(pts) < 2:
                continue
            dd = np.linalg.norm(pts - xy, axis=1)
            i = int(dd.argmin())
            j = min(i + 1, len(pts) - 1)
            k = max(j - 1, 0)
            d = pts[j] - pts[k]
            n = np.linalg.norm(d)
            if n < 1e-9 or (d / n) @ hv < 0.5:  # 같은 방향 차로만
                continue
            if dd[i] < best[0]:
                best = (float(dd[i]), lid, i)
        _, lid, idx = best
        _dbg = os.environ.get("RG_BRANCH_DEBUG", "0") == "1"
        if lid is None:
            if _dbg:
                print(f"[branch] xy={xy.round(1)} h={heading:.2f} v={v:.1f} -> lane match FAIL")
            return None
        # 참조 속도: 순간속도가 낮으면 분기 통과가 윈도우 밖으로 밀리므로 하한 적용
        # (RG_ROUTE_VMIN, 기본 3.0 m/s). 앵커가 속도 프로파일도 함께 조건화하므로
        # "가속해서 분기를 통과하는" 참조가 됨(속도 다양성과 자연 결합).
        v_ref = max(v, float(os.environ.get("RG_ROUTE_VMIN", "3.0")))
        need = v_ref * (nfut * dt) * 1.2
        chains = self._branch_chains(lid, idx, need)
        if _dbg:
            print(f"[branch] xy={xy.round(1)} v={v:.1f} vref={v_ref:.1f} lane={lid} "
                  f"d={best[0]:.2f} idx={idx} need={need:.1f} chains={len(chains)} aug={self.branch_aug}")
        chosen = self._branch_choice.get(agent_id)
        if len(chains) < 2 and chosen is None:
            return None

        def turn_of(chain):  # 체인 종점 방향 - 시작 방향 (좌+/우-), 정렬용
            p_end = self._branch_cl[chain[-1]]
            j = len(p_end) - 1
            d1 = p_end[j] - p_end[max(j - 1, 0)]
            a1 = math.atan2(d1[1], d1[0])
            a0 = math.atan2(hv[1], hv[0])
            return (a1 - a0 + math.pi) % (2 * math.pi) - math.pi

        if chosen is not None:
            # sticky: 배정된 분기 lane 을 포함하는(또는 그 lane 에서 이미 진행 중인) 체인 유지
            chain = next(
                (c for c in chains if chosen in c or c[0] == chosen), chains[0]
            )
        else:
            chains.sort(key=turn_of)
            chain = chains[self.branch_aug % len(chains)]
            after = [c for c in chain if c != lid]
            self._branch_choice[agent_id] = after[0] if after else lid
        pts = np.vstack(
            [xy[None, :], self._branch_cl[lid][idx:]]
            + [self._branch_cl[c] for c in chain[1:]]
        )
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        # 속도 램프(가속 상한 RG_ROUTE_ACC, 기본 2.5 m/s^2): 순간 속도 점프 방지
        acc = float(os.environ.get("RG_ROUTE_ACC", "2.5"))
        speeds = np.minimum(v_ref, max(v, 0.5) + acc * dt * np.arange(1, nfut + 1))
        tgt = np.clip(np.cumsum(speeds * dt), 0.0, s[-1])
        return np.stack(
            [np.interp(tgt, s, pts[:, 0]), np.interp(tgt, s, pts[:, 1])], axis=1
        )

    def _build_real_guidance(self, batch, ids_list: List[int], scene_enc) -> torch.Tensor:
        """
        Build the per-agent real-guidance lambda tensor and populate
        ``batch["agent"]["target"]`` (agent-centric GT future) so that the
        sampler can seed the reverse process from noised ground-truth.

        - tail agents (id in self.tail_ids)             -> lambda_tail (close to GT)
        - relevant agents (future-future m2m attn>1/N) -> lambda_rel
        - head agents (everything else)                 -> 1.0 (pure noise, unchanged)
        """
        model = self.motion_diffuser
        nhist = model.num_historical_steps
        nfut = model.num_future_steps
        device = self.device

        # ── 1) agent-centric GT future → target 필드 (relevant 판정에 noised-GT 필요) ──
        xyz = batch["agent"]["xyz"]  # (N, hist, 3)
        origin = xyz[:, nhist - 1, :2]  # (N, 2)
        theta = batch["agent"]["heading"][:, nhist - 1].squeeze(-1)  # (N,)
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(len(theta), 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos  # forward rotation (world -> agent-centric)

        future = torch.zeros(len(ids_list), nfut, 2, device=device)
        for j, _id in enumerate(ids_list):
            a = self.agent_manager.get_agent_by_id(_id)
            if a is None or a.ref_trajectory is None:
                continue
            idx = a.ref_trajectory.index
            seg = torch.as_tensor(
                a.ref_trajectory.xy[idx : idx + nfut], dtype=torch.float32, device=device
            )
            n = seg.shape[0]
            if n == 0:
                continue
            future[j, :n] = seg
            if n < nfut:  # pad by repeating the last GT point
                future[j, n:] = seg[-1]
        # ── 1b) 분기 다양성: head 의 target 을 GT 대신 "배정 분기의 합성 참조"로 대체 ──
        branch_nodes = set()
        if self.branch_enabled:
            v_now = (
                (xyz[:, nhist - 1, :2] - xyz[:, nhist - 2, :2]).norm(dim=-1) / 0.1
            )  # (N,) world m/s
            origin_np = origin.detach().cpu().numpy()
            theta_np = theta.detach().cpu().numpy()
            v_np = v_now.detach().cpu().numpy()
            for j, _id in enumerate(ids_list):
                if _id in self.tail_ids:  # tail 은 GT 앵커링 유지
                    continue
                ref = self._branch_reference(
                    _id, origin_np[j], float(theta_np[j]), float(v_np[j]), nfut
                )
                if ref is not None:
                    future[j] = torch.as_tensor(
                        ref, dtype=torch.float32, device=device
                    )
                    branch_nodes.add(j)
                    self._dbg_branch_ref = (j, _id, ref[-1].copy())  # debug 용

        # ── 1b-2) tail GT (world) 저장 — gt_leash 가이던스용 ──
        #   학위논문 §4.3: "Tail agent 는 ... GT 궤적을 크게 벗어나지 않는 범위에서
        #   제한적인 변동성만 허용". 그런데 식 (4.5) 의 tail_score 는 tail 을 밀어낸다.
        #   두 요구가 충돌하므로, 미는 항과 같은 에이전트에 이탈 상한을 건다(GTLeash).
        #   tail_residual 과 달리 pred_traj 유무와 무관하게 항상 채운다.
        if self.current_step == self.plan_init_step:
            gt_t = torch.zeros(len(ids_list), nfut, 2, device=device)
            gt_v = torch.zeros(len(ids_list), nfut, dtype=torch.bool, device=device)
            for j, _id in enumerate(ids_list):
                if _id in self.tail_ids:
                    gt_t[j] = future[j, :nfut, :2]
                    gt_v[j] = True
            batch["agent"]["tail_gt"] = gt_t
            batch["agent"]["tail_gt_valid"] = gt_v

        # ── 1c) tail-score residual (GT − pred), world frame, tail 에이전트만 ──
        #   (아주대 논문 식 4.5 확장) 첫 plan(current_step==plan_init_step)에서만: 이때
        #   ref_trajectory 가 아직 원본 GT 이고 future window(plan_init_step..+nfut)가 pred
        #   호라이즌과 정렬됨(이후 plan 은 ref 가 생성궤적으로 교체되어 정렬이 깨짐).
        #   future[j] = GT future world (위에서 계산). pred 는 동일 timestep 으로 매칭.
        if self.pred_traj and self.current_step == self.plan_init_step:
            resid = torch.zeros(len(ids_list), nfut, 2, device=device)
            rvalid = torch.zeros(len(ids_list), nfut, dtype=torch.bool, device=device)
            for j, _id in enumerate(ids_list):
                if _id not in self.tail_ids:
                    continue
                pmap = self.pred_traj.get(int(_id))
                if not pmap:
                    continue
                for k in range(nfut):
                    p = pmap.get(self.plan_init_step + k)
                    if p is None:
                        continue
                    resid[j, k, 0] = future[j, k, 0] - p[0]
                    resid[j, k, 1] = future[j, k, 1] - p[1]
                    rvalid[j, k] = True
            batch["agent"]["tail_residual"] = resid
            batch["agent"]["tail_residual_valid"] = rvalid
            if os.environ.get("TAIL_RESIDUAL_DEBUG", "0") == "1":
                na = int((rvalid.any(dim=1)).sum())
                print(f"[tail_residual] step={self.current_step} "
                      f"tail_agents_with_pred={na}/{len(self.tail_ids)} "
                      f"valid_steps={int(rvalid.sum())}", flush=True)

        self._v7_future = future.detach()   # v7 가이던스용 GT 미래 (world, 첫 plan 에서만 원본 GT)
        target_xy = torch.bmm(future - origin.unsqueeze(1), rot_mat)  # (N, nfut, 2)
        target = origin.new_zeros(len(ids_list), nfut, 4)
        target[..., :2] = target_xy
        target[..., 3] = torch.atan2(target_xy[..., 1], target_xy[..., 0])
        batch["agent"]["target"] = target

        # ── 2) tail / relevant 분류 ──
        tail_nodes = [j for j, _id in enumerate(ids_list) if _id in self.tail_ids]
        relevant_nodes = (
            self._relevant_nodes(batch, scene_enc, tail_nodes)
            if (self.relevant_enabled and tail_nodes)
            else set()
        )

        # ── 3) per-agent λ ── (branch 가 relevant 보다 우선: 분기 조건화가 목적일 때
        #     그 에이전트의 GT 유사성 유지(λ_rel)는 의미가 없어짐)
        lam = torch.ones(len(ids_list), device=device)  # default head
        for j, _id in enumerate(ids_list):
            if _id in self.tail_ids:
                lam[j] = self.lambda_tail
            elif j in branch_nodes:
                lam[j] = self.lambda_route
            elif j in relevant_nodes:
                lam[j] = self.lambda_rel
        return lam

    def _plan_trajectory(self):
        """
        Generate planned trajectory for agents.
        """
        data_list, ids_list = [], []
        for junc in self.junction_manager.junctions.values():
            hetero = junc.get_hetero_data()
            if hetero is None:
                continue
            data_list.append(hetero[0])
            ids_list.extend(hetero[1])
        if len(data_list) == 0:
            return
        batch = Batch.from_data_list(data_list).to(self.device)
        scene_enc = self.motion_diffuser.agent_encoder(
            batch, batch["roadgraph"]["poly_emb"]
        )
        real_guidance = (
            self._build_real_guidance(batch, ids_list, scene_enc)
            if self.real_guidance_enabled
            else None
        )
        # tail 에이전트는 gradient guidance 제외(real guidance 앵커링만). head/relevant만 적용.
        # RG_GUID_HEAD_ONLY=1 이면 논문 정합(편차② 해소): relevant(λ_rel)도 제외하고 head(λ=1)만 적용.
        if self.guidance is not None:
            if real_guidance is None:
                self.guidance.guidance_mask = None
            elif os.environ.get("RG_GUID_HEAD_ONLY", "0") == "1":
                self.guidance.guidance_mask = (
                    (real_guidance > 1.0 - 1e-6).view(-1, 1, 1).float()
                )
            else:
                self.guidance.guidance_mask = (
                    (torch.abs(real_guidance - self.lambda_tail) > 1e-6).view(-1, 1, 1).float()
                )
            # Learned tail-score guidance uses its own tail-only mask rather than the
            # global guidance_mask(head/relevant). Keep this per-agent flag on batch.
            batch["agent"]["tail_score_valid"] = torch.as_tensor(
                [_id in self.tail_ids for _id in ids_list],
                dtype=torch.bool,
                device=self.device,
            )
        # v7 membership 가이던스: batch 순서 id · GT 미래 · scene_enc 전달 (c* 는 첫 plan 에서 정함)
        if self.guidance is not None and hasattr(self.guidance, "set_scene"):
            if not self.real_guidance_enabled:
                raise SystemExit("[v7] membership 가이던스에는 real_guidance(tail_json) 설정이 필요하다")
            self.guidance.set_scene(list(ids_list), self._v7_future, self.tail_ids, scene_enc,
                                    first_plan=(self.current_step == self.plan_init_step))
        # DiversityRepel(§5): 이전 aug 궤적(world, (40,2))을 현재 plan 윈도우
        # [current_step, +nfut) 로 잘라 모듈에 주입. 생성 스크립트가 eng._div_prev
        # (agent_id -> list[np.ndarray (T_total,2)]) 를 설정했을 때만 활성.
        if self.guidance is not None and "div_repel" in getattr(
            self.guidance, "guidance_modules", {}
        ):
            div = self.guidance.guidance_modules["div_repel"]
            prev_dict = getattr(self, "_div_prev", None)
            if prev_dict:
                nfut = self.motion_diffuser.num_future_steps
                cs = self.current_step
                P = max(len(v) for v in prev_dict.values())
                prev = torch.zeros(len(ids_list), P, nfut, 2)
                valid = torch.zeros(len(ids_list), P, dtype=torch.bool)
                for j, _id in enumerate(ids_list):
                    for p, tr in enumerate(prev_dict.get(_id, [])):
                        seg = tr[cs : cs + nfut]
                        if len(seg) == 0:
                            continue
                        seg_t = torch.as_tensor(seg, dtype=torch.float32)
                        prev[j, p, : len(seg_t)] = seg_t
                        if len(seg_t) < nfut:  # 마지막 점 반복 패딩
                            prev[j, p, len(seg_t):] = seg_t[-1]
                        valid[j, p] = True
                div.prev, div.prev_valid = prev, valid
            else:
                div.prev = div.prev_valid = None
        sample = self.motion_diffuser.sampling(
            data=batch,
            scene_enc=scene_enc,
            show_progress=False,
            # eps_scaler=0.999,
            guide_fn=self.guidance,
            real_guidance=real_guidance,
            # EDM 샘플러 stochasticity (기본 0/1.0 = 기존 결정론적 ODE와 동일)
            S_churn=float(os.environ.get("RG_SCHURN", "0")),
            S_noise=float(os.environ.get("RG_SNOISE", "1.0")),
        )
        trajs, headings, vels = self.motion_diffuser._reconstruct_traj(
            data=batch, sample=sample, with_heading_and_vel=True
        )  # (num_agents, num_steps, 2)
        assert trajs.shape[0] == len(ids_list)
        trajs = trajs.detach().cpu().numpy()
        headings = headings.detach().cpu().numpy()
        vels = vels.detach().cpu().numpy()
        if os.environ.get("RG_BRANCH_DEBUG", "0") == "1" and getattr(self, "_dbg_branch_ref", None):
            j, _bid, ref_end = self._dbg_branch_ref
            print(f"[branch-out] agent {_bid} ref_end={ref_end.round(1)} "
                  f"plan_end={trajs[j][-1].round(1)}  (diff={np.linalg.norm(trajs[j][-1]-ref_end):.2f}m)")
            self._dbg_branch_ref = None
        for i, _id in enumerate(ids_list):
            a = self.agent_manager.get_agent_by_id(_id)
            assert a is not None
            if a.dynamic_flag:
                a.set_ref_trajectory(trajs[i], vels[i], headings[i])
        # free memory cache
        torch.cuda.empty_cache()

    def _compute_rewards(
        self, keys: List[str], agent_id: int
    ) -> Tuple[Dict[str, float], bool]:
        """
        Compute reward for target agent. Return reward and whether the episode is terminated.
        """
        rewards = {key: 0.0 for key in keys}
        terminated = False
        agent = self.agent_manager.get_agent_by_id(agent_id)
        assert agent is not None
        # dense reward for moving forward & smooth penalty
        his_states = agent.history_states
        if his_states.valid.sum() <= 1:
            # no history states
            rewards["forward"] = 0.0
            rewards["smooth"] = 0.0
            rewards["route_progress"] = 0.0
        else:
            last_pos = his_states.xyz[-2, :2]
            cur_pos = his_states.xyz[-1, :2]
            last_heading = his_states.heading[-2]
            cur_heading = his_states.heading[-1]
            cur_v = np.linalg.norm(his_states.vel[-1])
            # project the vector of last_pos -> cur_pos to the heading direction
            last_d, last_s, _ = polygon.frenet_projection(
                last_pos, agent.ref_trajectory.xy
            )
            cur_d, cur_s, _ = polygon.frenet_projection(
                cur_pos, agent.ref_trajectory.xy
            )
            rewards["forward"] = (cur_s - last_s) - (cur_d - last_d)
            rewards["route_progress"] = cur_s / polygon.get_length(
                agent.ref_trajectory.xy
            )
            # smooth penalty
            steer = polygon.wrap_angle(cur_heading - last_heading) / np.linalg.norm(
                cur_pos - last_pos
            )
            steer = np.clip(
                steer, -Action.BicycleAction.MAX_STEER, Action.BicycleAction.MAX_STEER
            )
            rewards["smooth"] = min(0, float(1 / cur_v - abs(steer)))
        # collision penalty
        collision = self.agent_manager.check_collision(agent_id)
        rewards["collision"] = -1.0 if collision else 0.0
        # offroad penalty
        offroad = agent.is_offroad()
        rewards["offroad"] = -1.0 if offroad else 0.0
        # destination reward
        if agent.runtime.is_finished():
            dest = agent.ref_trajectory.xy[-1]
            cur_xy = agent.runtime.xy
            dis = np.linalg.norm(dest - cur_xy)
            rewards["destination"] = 10 if dis < 2.5 else -5
        # check termination
        terminated = collision or offroad
        return rewards, terminated
