"""
FractalAttention — интегрированное фрактальное внимание для UnifiedTransformer.

Архитектура:
  12 голов = 4 уровня × 3 масштаба
  
  Level 0 (heads 0-2):  symbol→symbol,   масштабы 1, 2, 4
  Level 1 (heads 3-5):  word→word,       масштабы 1, 2, 4
  Level 2 (heads 6-8):  sentence→sentence, масштабы 1, 2, 4
  Level 3 (heads 9-11): domain→domain,    масштабы 1, 2, 4

Каждая голова:
  attention[i][j] = softmax(QK^T/√d + manifold_bias[i][j] + level_bias[i][j])
  manifold_bias[i][j] = exp(-||coord_i - coord_j||² / (2 * scale²))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from loguru import logger


class FractalAttention(nn.Module):
    """Фрактальное внимание — num_levels × scales_per_level голов."""
    
    def __init__(self, d_model: int = 64, num_levels: int = 4, scales_per_level: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_levels = num_levels
        self.scales_per_level = scales_per_level
        self.total_heads = num_levels * scales_per_level
        self.head_dim = d_model // self.total_heads
        
        if self.head_dim < 2:
            self.total_heads = d_model // 2
            self.head_dim = 2
        self.heads_per_level = self.total_heads // num_levels
        
        self.W_Q = nn.Linear(d_model, self.total_heads * self.head_dim, bias=False)
        self.W_K = nn.Linear(d_model, self.total_heads * self.head_dim, bias=False)
        self.W_V = nn.Linear(d_model, self.total_heads * self.head_dim, bias=False)
        self.W_O = nn.Linear(self.total_heads * self.head_dim, d_model, bias=False)
        
        self.level_bias = nn.Parameter(torch.zeros(num_levels, 1, 1))
        
        # Scales: 1, 2, 4, 8 for 4-per-level, or 1, 2, 4 for 3-per-level
        default_scales = [1.0, 2.0, 4.0, 8.0][:scales_per_level]
        self.scales = nn.Parameter(torch.tensor(default_scales).repeat(num_levels))
        
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Linear(32, num_levels), nn.Sigmoid(),
        )
        
        self.coord_proj = nn.Linear(d_model, 2, bias=False)

    def compute_manifold_bias(
        self,
        x: torch.Tensor,           # [B, L, D]
        scale: float,
    ) -> torch.Tensor:
        """
        Manifold bias: близкие в координатном пространстве → сильнее внимание.
        bias[i][j] = exp(-||proj(x_i) - proj(x_j)||² / (2 * scale²))
        """
        B, L, D = x.shape
        proj = self.coord_proj(x)  # [B, L, 2]
        diffs = proj.unsqueeze(2) - proj.unsqueeze(1)  # [B, L, L, 2]
        dists = torch.norm(diffs, dim=-1)  # [B, L, L]
        sigma = scale * 0.5
        bias = torch.exp(-dists ** 2 / (2 * sigma ** 2 + 1e-8))
        return bias

    def forward(
        self,
        x: torch.Tensor,           # [B, L, D]
        return_level_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Фрактальный forward pass.

        Returns: (output, level_weights)
        """
        B, L, D = x.shape
        H = self.total_heads
        Dh = self.head_dim

        # 1. Q, K, V проекции
        q = self.W_Q(x).view(B, L, H, Dh).transpose(1, 2)  # [B, H, L, Dh]
        k = self.W_K(x).view(B, L, H, Dh).transpose(1, 2)
        v = self.W_V(x).view(B, L, H, Dh).transpose(1, 2)

        # 2. Gate: насколько активен каждый уровень
        context = x.mean(dim=1)  # [B, D]
        level_gates = self.gate_net(context)  # [B, num_levels]

        # 3. Attention с манифолд-байесом для каждого масштаба
        outputs = []
        gate_sum = 0.0

        for h in range(H):
            level = h // self.heads_per_level
            scale = self.scales[h]

            # QK^T
            attn_scores = torch.matmul(
                q[:, h], k[:, h].transpose(-2, -1)
            ) / (Dh ** 0.5)  # [B, L, L]

            # Manifold bias (зависит от масштаба)
            manifold_bias = self.compute_manifold_bias(x, scale.item())

            # Level bias
            level_b = self.level_bias[level]
            attn_scores = attn_scores + manifold_bias + level_b

            # Causal mask
            causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
            attn_scores = attn_scores.masked_fill(causal, float('-inf'))

            # Softmax
            attn_weights = F.softmax(attn_scores, dim=-1)

            # Apply attention
            head_out = torch.matmul(attn_weights, v[:, h])  # [B, L, Dh]

            # Weight by level gate
            gate_w = level_gates[:, level].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            outputs.append(head_out * gate_w)
            gate_sum = gate_sum + gate_w.mean()

        # Merge heads
        out = torch.stack(outputs, dim=1)  # [B, H, L, Dh]
        out = out.transpose(1, 2).contiguous().view(B, L, H * Dh)  # [B, L, D]
        out = self.W_O(out)

        if gate_sum > 0:
            out = out / (gate_sum / H + 1e-8)

        if return_level_weights:
            return out, level_gates
        return out, None

    def summary(self) -> str:
        return f"FractalAttentionV2(H={self.total_heads}, levels={self.num_levels}, head_dim={self.head_dim})"
