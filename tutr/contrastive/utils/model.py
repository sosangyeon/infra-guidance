import torch
import torch.nn as nn

from utils.feature_utils import (
    input_dim,
    output_dim,
    select_input_features,
    validate_input_mode,
    validate_output_mode,
)
from utils.transformer_encoder import Encoder
from utils.transformer_decoder import Decoder


class TrajectoryModel(nn.Module):
    """Input-feature / output-target ablation model.

    Common pkl state: [x,y,vx,vy,speed,ax,ay,acc].
    Motion-mode classification remains future XY-only for every experiment.
    Regression output is selected by output_mode.
    """

    def __init__(
        self,
        input_mode,
        output_mode,
        obs_len,
        pred_len,
        embed_size,
        enc_num_layers,
        int_num_layers_list,
        heads,
        forward_expansion,
    ):
        super().__init__()
        validate_input_mode(input_mode)
        validate_output_mode(output_mode)
        self.input_mode = input_mode
        self.output_mode = output_mode
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.output_dim = output_dim(output_mode)

        # Candidate future motion modes contain XY only.
        self.pos_embedding = nn.Linear(2 * self.seq_len, embed_size)
        self.vel_embedding = None
        self.speed_embedding = None
        self.acc_embedding = None
        fusion_count = 1

        if input_mode in {"xy_vxvy", "xy_vxvy_acc", "xy_vxvy_axay"}:
            self.vel_embedding = nn.Linear(2 * obs_len, embed_size)
            fusion_count += 1
        elif input_mode == "xy_speed":
            self.speed_embedding = nn.Linear(obs_len, embed_size)
            fusion_count += 1

        if input_mode == "xy_vxvy_acc":
            self.acc_embedding = nn.Linear(obs_len, embed_size)
            fusion_count += 1
        elif input_mode == "xy_vxvy_axay":
            self.acc_embedding = nn.Linear(2 * obs_len, embed_size)
            fusion_count += 1

        self.fusion = nn.Linear(embed_size * fusion_count, embed_size)
        self.mode_encoder = Encoder(embed_size, enc_num_layers, heads, forward_expansion, islinear=True)
        self.cls_head = nn.Linear(embed_size, 1)

        self.nei_embedding = nn.Linear(input_dim(input_mode) * obs_len, embed_size)
        self.social_decoder = Decoder(
            embed_size, int_num_layers_list[1], heads, forward_expansion, islinear=False
        )
        self.reg_head = nn.Linear(embed_size, self.output_dim * pred_len)

    def spatial_interaction(self, ped_feat, neis_obs_full, mask):
        neis_obs = select_input_features(neis_obs_full, self.input_mode)
        neis_flat = neis_obs.reshape(neis_obs.shape[0], neis_obs.shape[1], -1)
        nei_embeddings = self.nei_embedding(neis_flat)
        social_mask = mask[:, 0:1].repeat(1, ped_feat.shape[1], 1)
        return self.social_decoder(ped_feat, nei_embeddings, social_mask)

    def _encode_modes(self, ped_obs_full, motion_modes):
        B = ped_obs_full.shape[0]
        K = motion_modes.shape[0]

        obs_xy = ped_obs_full[..., 0:2].unsqueeze(1).expand(-1, K, -1, -1)
        future_xy = motion_modes.unsqueeze(0).expand(B, -1, -1, -1)
        pos_input = torch.cat((obs_xy, future_xy), dim=2).reshape(B, K, -1)
        embeddings = [self.pos_embedding(pos_input)]

        if self.vel_embedding is not None:
            vel = ped_obs_full[..., 2:4].reshape(B, -1).unsqueeze(1).expand(-1, K, -1)
            embeddings.append(self.vel_embedding(vel))

        if self.speed_embedding is not None:
            speed = ped_obs_full[..., 4:5].reshape(B, -1).unsqueeze(1).expand(-1, K, -1)
            embeddings.append(self.speed_embedding(speed))

        if self.input_mode == "xy_vxvy_acc":
            acc = ped_obs_full[..., 7:8].reshape(B, -1).unsqueeze(1).expand(-1, K, -1)
            embeddings.append(self.acc_embedding(acc))
        elif self.input_mode == "xy_vxvy_axay":
            acc = ped_obs_full[..., 5:7].reshape(B, -1).unsqueeze(1).expand(-1, K, -1)
            embeddings.append(self.acc_embedding(acc))

        return self.mode_encoder(self.fusion(torch.cat(embeddings, dim=-1)))

    def forward(self, ped_obs_full, neis_obs_full, motion_modes, mask,
                closest_mode_indices=None, test=False, num_k=1):
        B = ped_obs_full.shape[0]
        device = ped_obs_full.device
        ped_feat = self._encode_modes(ped_obs_full, motion_modes)
        scores = self.cls_head(ped_feat).squeeze(-1)

        if not test:
            if closest_mode_indices is None:
                raise ValueError("closest_mode_indices is required during training.")
            batch_idx = torch.arange(B, device=device)
            chosen_feat = ped_feat[batch_idx, closest_mode_indices.long()].unsqueeze(1)
            int_feat = self.spatial_interaction(chosen_feat, neis_obs_full, mask)
            return self.reg_head(int_feat.squeeze(1)), scores

        num_k = min(num_k, scores.shape[1])
        top_k = torch.topk(scores, k=num_k, dim=-1).indices
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_k).reshape(-1)
        mode_idx = top_k.reshape(-1)
        top_k_feat = ped_feat[batch_idx, mode_idx].reshape(B, num_k, -1)
        int_feat = self.spatial_interaction(top_k_feat, neis_obs_full, mask)
        return self.reg_head(int_feat), scores
