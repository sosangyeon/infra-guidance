import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from filterpy.kalman import KalmanFilter
from torch_geometric.data import Batch, HeteroData
from itertools import chain, compress
from pathlib import Path
from typing import Optional

# ----------------------------------------------------------------
# Internal imports (원본 QCNet 구성요소)
# ----------------------------------------------------------------
from losses import MixtureNLLLoss, NLLLoss
from metrics import Brier, MR, minADE, minAHE, minFDE, minFHE
from metrics.minADE_Top1Percent_byError import minADE_Top1Percent_byError
from modules import QCNetDecoder, QCNetEncoder
try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object

# ================================================================
class QCNet_Contrastive_CP(pl.LightningModule):
    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 num_recurrent_steps: int,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_dec_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 submission_dir: str,
                 submission_file_name: str,
                 lambda_contrastive: float,  
                 tau: float,
                 theta_p: float,
                 theta_n: float,
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.lambda_contrastive = lambda_contrastive
        self.tau = tau
        self.theta_p = theta_p
        self.theta_n = theta_n

        # ------------------------------------------------------------
        # Encoder / Decoder
        # ------------------------------------------------------------
        self.encoder = QCNetEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_map_layers=num_map_layers,
            num_agent_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.decoder = QCNetDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            output_head=output_head,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            num_modes=num_modes,
            num_recurrent_steps=num_recurrent_steps,
            num_t2m_steps=num_t2m_steps,
            pl2m_radius=pl2m_radius,
            a2m_radius=a2m_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_dec_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

        # ------------------------------------------------------------
        # Losses & Metrics
        # ------------------------------------------------------------
        self.reg_loss = NLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                reduction='none')
        self.cls_loss = MixtureNLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                       reduction='none')

        self.Brier = Brier(max_guesses=6)
        self.minADE = minADE(max_guesses=6)
        self.minAHE = minAHE(max_guesses=6)
        self.minFDE = minFDE(max_guesses=6)
        self.minFHE = minFHE(max_guesses=6)
        self.MR = MR(max_guesses=6)        
        self.minADE_Worst_1 = minADE_Top1Percent_byError(max_guesses=6, percentile=1.0)
        self.minADE_Worst_2 = minADE_Top1Percent_byError(max_guesses=6, percentile=2.0)
        self.minADE_Worst_3 = minADE_Top1Percent_byError(max_guesses=6, percentile=3.0)
        self.test_predictions = dict()

    # ---------------------------------------------------------------------
    # 🧩 Contrastive Loss 정의
    # ---------------------------------------------------------------------
    def contrastive_loss(self, z, difficulty):
        tau = self.tau
        theta_p = self.theta_p
        theta_n = self.theta_n

        sim = torch.matmul(z, z.T) / tau
        diff = (difficulty.unsqueeze(1) - difficulty.unsqueeze(0)).abs()
        pos_mask = (diff < theta_p) & (~torch.eye(len(diff), dtype=bool, device=diff.device))
        neg_mask = (diff > theta_n)

        exp_sim = torch.exp(sim)
        pos_exp = exp_sim * pos_mask
        denom = exp_sim * (pos_mask | neg_mask)
        loss = -torch.log((pos_exp.sum(1) + 1e-6) / (denom.sum(1) + 1e-6))
        return loss.mean()

    # ---------------------------------------------------------------------
    def forward(self, data: HeteroData, return_feature=False):
        scene_enc = self.encoder(data)
        pred = self.decoder(data, scene_enc)
        if return_feature:
            return pred, scene_enc
        else:
            return pred


    # ---------------------------------------------------------------------
    def training_step(self, data, batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]
        pred, scene_enc = self(data, return_feature=True)


        # ---------------------- 기존 QCNet 손실 ----------------------
        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim],
                        data['agent']['target'][..., -1:]], dim=-1)
        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) *
                   reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        reg_loss_propose = self.reg_loss(traj_propose_best,
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()
        reg_loss_refine = self.reg_loss(traj_refine_best,
                                        gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()
        cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim + self.output_head],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)


        # ---------------------- 🔹 난이도 (Dataset에서 직접 가져오기) ----------------------
        if 'difficulty' not in data['agent']:
            raise ValueError("❌ data['agent']['difficulty']가 없습니다. Dataset에서 난이도 값을 추가해주세요.")
        difficulty = data['agent']['difficulty'].float().to(self.device)

        # ---------------------- 🔹 Contrastive Loss ----------------------

        if isinstance(scene_enc, dict):
            z = scene_enc['x_a']    # 🚗 agent feature 사용
        else:
            z = scene_enc

        z = F.normalize(z.mean(dim=1), dim=-1)  # 각 에이전트 평균, feature 정규화
        contrastive_loss = self.contrastive_loss(z, difficulty)
        self.log("train_reg_loss_propose", reg_loss_propose, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log("train_reg_loss_refine", reg_loss_refine, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log("train_cls_loss", cls_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log("train_contrastive_loss", contrastive_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)

        loss = reg_loss_propose + reg_loss_refine + cls_loss + self.lambda_contrastive*contrastive_loss
        self.log("train_total_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)

        return loss
    # ---------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]


    def validation_step(self,
                        data,
                        batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]
        pred = self(data)
        # print(f"keys of original prediction : {pred.keys()}")
        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)

        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
        gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        reg_loss_propose = self.reg_loss(traj_propose_best,
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()
        reg_loss_refine = self.reg_loss(traj_refine_best,
                                        gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()
        cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim + self.output_head],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        # self.log('val_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
        #          sync_dist=True)
        # self.log('val_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
        #          sync_dist=True)
        # self.log('val_cls_loss', cls_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        if self.dataset == 'argoverse_v2' or self.dataset == 'argoverse_v2_ACL':
            eval_mask = data['agent']['category'] == 3
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        valid_mask_eval = reg_mask[eval_mask]
        traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head]
        

        if not self.output_head:
            traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
                                                     traj_eval[..., :2]], dim=-2)
            motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
            head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
            traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)
        pi_eval = F.softmax(pi[eval_mask], dim=-1)
        gt_eval = gt[eval_mask]
        
        # ---------------------------------------------------------------------------------
        # jsa add
        origin = data['agent']['position'][:, self.num_historical_steps - 1] 
        theta = data['agent']['heading'][:, self.num_historical_steps - 1]
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(data['agent']['num_nodes'], 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        inv_rot_mat = rot_mat.transpose(1, 2)   
        
        traj_eval_global = torch.zeros(traj_eval.shape)
        for i in range(6):
            traj_eval_global[:,i,:,:2] = torch.bmm(traj_eval[:,i,:,:2], inv_rot_mat) + origin[:, :2].unsqueeze(1)
        traj_eval_global = traj_eval_global.to(traj_eval.device)
        # jsa add
        # ---------------------- 🔹 [NEW] Long Tail Metrics 업데이트 및 로깅 ----------------------
        self.minADE_Worst_1.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim])
        self.minADE_Worst_2.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim])
        self.minADE_Worst_3.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim])
        self.log('val_minADE_Worst1%', self.minADE_Worst_1, on_epoch=True, prog_bar=True)
        self.log('val_minADE_Worst2%', self.minADE_Worst_2, on_epoch=True, prog_bar=True)
        self.log('val_minADE_Worst3%', self.minADE_Worst_3, on_epoch=True, prog_bar=True)

        # --------------------------------------------------------------------------------------
        self.Brier.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                          valid_mask=valid_mask_eval)
        self.minADE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)
        self.minAHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        self.minFDE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)
        self.minFHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        self.MR.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                       valid_mask=valid_mask_eval)
        # self.log('val_Brier', self.Brier, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        # self.log('val_minAHE', self.minAHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        # self.log('val_minFHE', self.minFHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_MR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))


    def test_step(self,
                  data,
                  batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        pred = self(data)
        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        if self.dataset == 'argoverse_v2' or self.dataset == 'argoverse_v2_ACL':
            eval_mask = data['agent']['category'] == 3
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
        theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        traj_eval = torch.matmul(traj_refine[eval_mask, :, :, :2],
                                 rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        pi_eval = F.softmax(pi[eval_mask], dim=-1)

        traj_eval = traj_eval.cpu().numpy()
        pi_eval = pi_eval.cpu().numpy()
        
        if self.dataset == 'argoverse_v2' or self.dataset == 'argoverse_v2_ACL':
            eval_id = list(compress(list(chain(*data['agent']['id'])), eval_mask))
            if isinstance(data, Batch):
                for i in range(data.num_graphs):
                    self.test_predictions[data['scenario_id'][i]] = (pi_eval[i], {eval_id[i]: traj_eval[i]})
            else:
                self.test_predictions[data['scenario_id']] = (pi_eval[0], {eval_id[0]: traj_eval[0]})
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        
        #0609수정###
        import os
        save_dir = '/media/user/purple/cum/active_bench/bench_random_pet7/base/test_추가2/contrasive/version_129_33_50_1'
        os.makedirs(save_dir, exist_ok=True)

        if isinstance(data, Batch):
            cnt = 0  # 전체 eval_id 인덱스를 따라가며 카운트
            for i in range(data.num_graphs):
                scenario_id = data['scenario_id'][i]
                save_path = os.path.join(save_dir, f"{scenario_id}.txt")

                with open(save_path, 'w') as f:
                    f.write(f"# Scenario ID: {scenario_id}\n\n")

                    # 이 그래프에 해당하는 에이전트 수 확인
                    num_agents = data['agent']['ptr'][i + 1] - data['agent']['ptr'][i]
                    for j in range(num_agents):
                        if not eval_mask[cnt]:
                            cnt += 1
                            continue
                        agent_id = eval_id[cnt]
                        f.write(f"# Agent ID: {agent_id}\n")
                        f.write(f"# Mode Probabilities (pi): {pi_eval[cnt].tolist()}\n")
                        for mode_idx, traj in enumerate(traj_eval[cnt]):
                            f.write(f"## Mode {mode_idx} (Prob: {pi_eval[cnt][mode_idx]:.4f})\n")
                            for t, (x, y) in enumerate(traj):
                                f.write(f"t={t}, x={x:.3f}, y={y:.3f}\n")
                            f.write("\n")
                        cnt += 1
        else:
            scenario_id = data['scenario_id']
            save_path = os.path.join(save_dir, f"{scenario_id}.txt")
            with open(save_path, 'w') as f:
                f.write(f"# Scenario ID: {scenario_id}\n\n")
                for i, agent_id in enumerate(eval_id):
                    f.write(f"# Agent ID: {agent_id}\n")
                    f.write(f"# Mode Probabilities (pi): {pi_eval[i].tolist()}\n")
                    for mode_idx, traj in enumerate(traj_eval[i]):
                        f.write(f"## Mode {mode_idx} (Prob: {pi_eval[i][mode_idx]:.4f})\n")
                        for t, (x, y) in enumerate(traj):
                            f.write(f"t={t}, x={x:.3f}, y={y:.3f}\n")
                        f.write("\n")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('QCNet')
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--output_head', action='store_true')
        parser.add_argument('--num_historical_steps', type=int, required=True)
        parser.add_argument('--num_future_steps', type=int, required=True)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--num_recurrent_steps', type=int, required=True)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--num_map_layers', type=int, default=1)
        parser.add_argument('--num_agent_layers', type=int, default=2)
        parser.add_argument('--num_dec_layers', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--pl2pl_radius', type=float, required=True)
        parser.add_argument('--time_span', type=int, default=None)
        parser.add_argument('--pl2a_radius', type=float, required=True)
        parser.add_argument('--a2a_radius', type=float, required=True)
        parser.add_argument('--num_t2m_steps', type=int, default=None)
        parser.add_argument('--pl2m_radius', type=float, required=True)
        parser.add_argument('--a2m_radius', type=float, required=True)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        parser.add_argument('--submission_dir', type=str, default='./')
        parser.add_argument('--submission_file_name', type=str, default='submission')
        parser.add_argument("--lambda_contrastive", type=float, default=None)
        parser.add_argument("--tau", type=float, default=None)
        parser.add_argument("--theta_p", type=float, default=None)
        parser.add_argument("--theta_n", type=float, default=None)
        return parent_parser
