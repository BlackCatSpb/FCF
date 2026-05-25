"""
EVA — UnifiedMultidimensionalTransformer v2.

64-dim координатное пространство, 16 голов (4×4), 3 слоя.
RoPE, RMSNorm, Pre-norm, SwiGLU. ~141K параметров.
"""

import torch, torch.nn as nn, torch.nn.functional as F, math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


# ============================================================
# RoPE — Rotary Position Embeddings
# ============================================================

class RoPE(nn.Module):
    """Rotary Position Embedding — кодирует позицию поворотом вектора."""
    
    def __init__(self, dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        positions = torch.arange(L, device=x.device).float()
        freqs = self.freqs.to(x.device)
        angles = torch.outer(positions, freqs)  # [L, D/2]
        
        cos = angles.cos().unsqueeze(0).unsqueeze(2)  # [1, L, 1, D/2]
        sin = angles.sin().unsqueeze(0).unsqueeze(2)
        
        x_reshaped = x.view(B, L, -1, 2)  # [B, L, D/2, 2]
        x0, x1 = x_reshaped[..., 0], x_reshaped[..., 1]
        
        rotated = torch.stack([
            x0 * cos.squeeze(2) - x1 * sin.squeeze(2),
            x1 * cos.squeeze(2) + x0 * sin.squeeze(2),
        ], dim=-1)
        
        return rotated.view(B, L, D)


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA style)."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ============================================================
# 1. CoordinateEmbedding — символ → ℝ⁶⁴ (без lookup)
# ============================================================

class CoordinateEmbedding(nn.Module):
    """Координатный эмбеддинг. Позиции из TopologicalField MDS."""
    
    def __init__(self, vocab_size: int = 157, coord_dim: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.coord_dim = coord_dim
        self.register_buffer("coordinates", torch.randn(vocab_size, coord_dim) * 0.02)
        self.scale = nn.Parameter(torch.ones(1))
    
    def set_coordinates(self, coords: torch.Tensor):
        assert coords.shape[0] == self.vocab_size
        self.coordinates.copy_(coords[:, :self.coord_dim])
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        ids = token_ids.clamp(0, self.vocab_size - 1)
        return self.coordinates[ids] * self.scale


# ============================================================
# 2. CoordinateDecoder — ℝ⁶⁴ → 157 символов
# ============================================================

class CoordinateDecoder(nn.Module):
    """Обучаемый линейный классификатор + nearest neighbor."""
    
    def __init__(self, coord_embedding: CoordinateEmbedding):
        super().__init__()
        self.embed = coord_embedding
        self.vocab_size = coord_embedding.vocab_size
        self.coord_dim = coord_embedding.coord_dim
        
        self.linear = nn.Linear(self.coord_dim, self.vocab_size)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.nn_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x) / self.temperature.clamp(min=0.1)
        
        coords = self.embed.coordinates
        diffs = x.unsqueeze(2) - coords.unsqueeze(0).unsqueeze(0)
        dists = torch.norm(diffs, dim=-1)
        nn_logits = -dists * self.nn_weight
        
        return logits + nn_logits
    
    def decode_to_ids(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=-1)


# ============================================================
# 3. SwiGLU FFN
# ============================================================

class SwiGLUFFN(nn.Module):
    """Gated FFN: (SiLU(x·W_gate) ⊙ (x·W_up)) · W_down"""
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.W_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.W_up = nn.Linear(dim, hidden_dim, bias=False)
        self.W_down = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.W_gate(x))
        up = self.W_up(x)
        return self.W_down(gate * up)


# ============================================================
# 4. TransformerBlock (Pre-norm, FractalAttention, SwiGLU)
# ============================================================

class TransformerBlock(nn.Module):
    """Один блок: Pre-norm Attention + Pre-norm FFN."""
    
    def __init__(self, dim: int, num_levels: int = 4, scales_per_level: int = 4,
                 d_ff: int = 128):
        super().__init__()
        self.dim = dim
        
        from .fractal_v2 import FractalAttention
        self.attention = FractalAttention(
            d_model=dim,
            num_levels=num_levels,
            scales_per_level=scales_per_level,
        )
        
        self.norm_attn = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, d_ff)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm Attention
        attn_out, _ = self.attention(self.norm_attn(x))
        x = x + attn_out
        
        # Pre-norm FFN
        x = x + self.ffn(self.norm_ffn(x))
        
        return x


# ============================================================
# 5. UnifiedMultidimensionalTransformer
# ============================================================

class UnifiedMultidimensionalTransformer(nn.Module):
    """Координатный трансформер: 64-dim, 16 голов (4×4), 3 слоя, ~141K params."""
    
    def __init__(
        self,
        vocab_size: int = 157,
        coord_dim: int = 64,
        num_heads: int = 16,
        num_levels: int = 4,
        scales_per_level: int = 4,
        num_layers: int = 3,
        d_ff: int = 128,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.coord_dim = coord_dim
        self.num_layers = num_layers
        
        self.embed = CoordinateEmbedding(vocab_size, coord_dim)
        self.rope = RoPE(coord_dim, max_seq_len)
        self.decoder = CoordinateDecoder(self.embed)
        
        self.layers = nn.ModuleList([
            TransformerBlock(coord_dim, num_levels, scales_per_level, d_ff)
            for _ in range(num_layers)
        ])
        
        self.norm_final = RMSNorm(coord_dim)
    
    def set_symbol_coordinates(self, coords: torch.Tensor):
        self.embed.set_coordinates(coords)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L = token_ids.shape
        
        x = self.embed(token_ids)
        x = self.rope(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm_final(x)
        
        scores = None
        if return_scores:
            scores = self.decoder.forward(x)
        
        return x, scores
    
    def summary(self) -> str:
        params = sum(p.numel() for p in self.parameters())
        return (f"UnifiedTransformer(dim={self.coord_dim}, heads={self.num_layers*4*4}, "
                f"layers={self.num_layers}, params={params:,})")
