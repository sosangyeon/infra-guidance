# -*- coding: utf-8 -*-
"""TrajectoryModel + bottleneck 특징 반환.

bottleneck = 정답에 가장 가까운 모션 모드의 특징 ped_feat[b, closest_mode] (상호작용 전).
가중치 구조는 model.py 와 같다 (파라미터 추가 없음).
"""
import torch

from model import TrajectoryModel as _BaseModel


class TrajectoryModel(_BaseModel):
    def forward(self, ped_obs_full, neis_obs_full, motion_modes, mask,
                closest_mode_indices=None, test=False, num_k=1, return_feat=False):
        if test or not return_feat:
            return super().forward(ped_obs_full, neis_obs_full, motion_modes, mask,
                                   closest_mode_indices, test=test, num_k=num_k)
        B = ped_obs_full.shape[0]
        ped_feat = self._encode_modes(ped_obs_full, motion_modes)
        scores = self.cls_head(ped_feat).squeeze(-1)
        batch_idx = torch.arange(B, device=ped_obs_full.device)
        bottleneck = ped_feat[batch_idx, closest_mode_indices.long()]          # [B, E]
        int_feat = self.spatial_interaction(bottleneck.unsqueeze(1), neis_obs_full, mask)
        return self.reg_head(int_feat.squeeze(1)), scores, bottleneck
