from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def _mlp(dims: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class TailScoreNet(nn.Module):
    """Small tail-score regressor for guidance.

    Inputs are agent-centric observed history and candidate future trajectory.
    Output is a scalar tail-likeness score; larger means harder / more tail-like.
    """

    def __init__(
        self,
        hist_steps: int = 10,
        fut_steps: int = 30,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        map_dim: int = 0,
    ) -> None:
        super().__init__()
        self.hist_steps = hist_steps
        self.fut_steps = fut_steps
        self.map_dim = map_dim
        hist_dim = hist_steps * 2
        fut_dim = fut_steps * 2
        self.hist_proj = _mlp([hist_dim, hidden_dim, hidden_dim], dropout)
        self.fut_proj = _mlp([fut_dim, hidden_dim, hidden_dim], dropout)
        # v2: 지도 특징 optional. map_dim=0 이면 구조·수치 완전 동일(하위호환).
        self.map_proj = _mlp([map_dim, hidden_dim, hidden_dim], dropout) if map_dim > 0 else None
        extra = hidden_dim if map_dim > 0 else 0
        self.head = _mlp([hidden_dim * 2 + 1 + extra, hidden_dim, hidden_dim // 2, 1], dropout)

    def forward(
        self,
        hist_xy: torch.Tensor,
        fut_xy: torch.Tensor,
        cur_speed: torch.Tensor,
        map_feat: torch.Tensor = None,
    ) -> torch.Tensor:
        if hist_xy.ndim != 3 or fut_xy.ndim != 3:
            raise ValueError("hist_xy and fut_xy must be (N,T,2)")
        hist_feat = self.hist_proj(hist_xy.reshape(hist_xy.shape[0], -1))
        fut_feat = self.fut_proj(fut_xy.reshape(fut_xy.shape[0], -1))
        speed_feat = cur_speed.reshape(-1, 1)
        parts = [hist_feat, fut_feat, speed_feat]
        if self.map_proj is not None:
            if map_feat is None:
                raise ValueError("map_dim>0 인 net 은 map_feat 가 필요하다")
            parts.append(self.map_proj(map_feat))
        score = self.head(torch.cat(parts, dim=-1))
        return score.squeeze(-1)

    def checkpoint_payload(self) -> Dict[str, float | int]:
        return {
            "hist_steps": self.hist_steps,
            "fut_steps": self.fut_steps,
            "map_dim": self.map_dim,
        }
