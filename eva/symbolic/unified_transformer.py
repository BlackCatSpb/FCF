"""
EVA — UnifiedMultidimensionalTransformer v2.

64-dim координатное пространство, 16 голов (4×4), 3 слоя.
RoPE, RMSNorm, Pre-norm, SwiGLU. ~141K параметров.
"""

import torch, torch.nn as nn, torch.nn.functional as F, math
import numpy as np
from typing import List, Tuple, Optional, Dict
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
    """Subspace Hierarchical Softmax (SubHSM) + nearest neighbor."""
    
    def __init__(self, coord_embedding: CoordinateEmbedding):
        super().__init__()
        self.embed = coord_embedding
        self.vocab_size = coord_embedding.vocab_size
        self.coord_dim = coord_embedding.coord_dim
        
        self.linear = nn.Linear(self.coord_dim, self.vocab_size)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.nn_weight = nn.Parameter(torch.tensor(0.1))
        
        # --- SubHSM: 4 группы по ~40 токенов ---
        n_groups = 4
        group_size = (self.vocab_size + n_groups - 1) // n_groups
        self.group_size = group_size
        
        # Register buffer: group_id and local_id for each vocab token
        group_ids = torch.zeros(self.vocab_size, dtype=torch.long)
        local_ids = torch.zeros(self.vocab_size, dtype=torch.long)
        for i in range(self.vocab_size):
            group_ids[i] = i // group_size
            local_ids[i] = i % group_size
        self.register_buffer('group_ids', group_ids)
        self.register_buffer('local_ids', local_ids)
        
        self.group_classifier = nn.Linear(self.coord_dim, n_groups)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Full logits
        logits = self.linear(x) / self.temperature.clamp(min=0.1)
        
        # Nearest neighbor branch
        coords = self.embed.coordinates
        diffs = x.unsqueeze(2) - coords.unsqueeze(0).unsqueeze(0)
        dists = torch.norm(diffs, dim=-1)
        nn_logits = -dists * self.nn_weight
        
        full_logits = logits + nn_logits
        
        # --- SubHSM auxiliary logits ---
        group_logits = self.group_classifier(x)  # [B, L, 4]
        full_logits = full_logits + group_logits[:, :, :1] * 0  # no-op, just for group aux
        
        return full_logits
    
    def forward_subhsm(self, x: torch.Tensor):
        """Returns (full_logits, group_logits) for combined loss."""
        B, L, D = x.shape
        full_logits = self.forward(x)
        
        # Aggregate per-group log-sum-exp
        gs = self.group_size
        n_groups = 4
        group_logits = torch.stack([
            full_logits[:, :, g*gs:(g+1)*gs].logsumexp(dim=-1)
            for g in range(n_groups)
        ], dim=-1)  # [B, L, 4]
        
        return full_logits, group_logits
    
    def decode_to_ids(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        # Fast hierarchical decoding: top group → top token in group
        group_logits = self.group_classifier(x)  # [B, L, 4]
        best_group = group_logits.argmax(dim=-1)  # [B, L]
        gs = self.group_size
        B, L = best_group.shape
        result = torch.zeros(B, L, dtype=torch.long, device=x.device)
        for b in range(B):
            for i in range(L):
                g = best_group[b, i].item()
                start = g * gs
                end = min(start + gs, self.vocab_size)
                local_logits = logits[b, i, start:end]
                result[b, i] = start + local_logits.argmax()
        return result


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
    """Один блок: Pre-norm AdaptiveAttention + Pre-norm FFN."""
    
    def __init__(self, dim: int, max_levels: int = 8, total_heads: int = 32, d_ff: int = 128):
        super().__init__()
        self.dim = dim
        
        from .adaptive_fractal import AdaptiveFractalAttention
        self.attention = AdaptiveFractalAttention(
            d_model=dim, max_levels=max_levels, total_heads=total_heads,
        )
        
        self.norm_attn = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, d_ff)
    
    def forward(self, x: torch.Tensor):
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
    """Координатный трансформер: 128-dim, адаптивные уровни, ~1.6M params."""
    
    def __init__(self, vocab_size=161, coord_dim=128, max_levels=8, total_heads=32,
                 num_layers=6, d_ff=128, max_seq_len=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.coord_dim = coord_dim
        self.num_layers = num_layers
        
        self.embed = CoordinateEmbedding(vocab_size, coord_dim)
        self.rope = RoPE(coord_dim, max_seq_len)
        self.decoder = CoordinateDecoder(self.embed)
        
        # Multi-subspace projections (agent Phase 3)
        from .subspace_coords import MultiSubspaceEmbedding
        self.subspace = MultiSubspaceEmbedding(vocab_size, coord_dim, sym_dim=32)
        
        from .fractal_conv import HybridFractalBlock
        from .static_topology import StaticTopologyLayer
        self.topology = StaticTopologyLayer(vocab_size, coord_dim)
        self.layers = nn.ModuleList([
            HybridFractalBlock(coord_dim, max_levels, total_heads, d_ff, self.topology)
            for _ in range(num_layers)
        ])
        
        self.norm_final = RMSNorm(coord_dim)
        
        from .subspace_coords import WordWeightEncoder
        self.word_weight = WordWeightEncoder(coord_dim)
        
        from .adaptive_fractal import ConnectionCoordinateHead
        self.connection_head = ConnectionCoordinateHead(coord_dim)
    
    def set_symbol_coordinates(self, coords: torch.Tensor):
        self.embed.set_coordinates(coords)
        self.subspace.set_coordinates(coords)
    
    def forward(self, token_ids, return_scores=False, return_weights=False):
        B, L = token_ids.shape
        
        # Base symbol embedding
        x = self.embed(token_ids)
        
        # Multi-subspace: enrich with word/connection/sentence projections
        sym = self.subspace(token_ids)  # [B, L, 32]
        # Pad sym to full dim and add to base
        x = x + F.pad(sym, (0, self.coord_dim - 32))
        
        x = self.rope(x)
        
        # Coordinate Residual Stream — сквозной поток координат
        coord_stream = torch.zeros_like(x)
        
        for layer in self.layers:
            x, coord_stream = layer(x, token_ids=token_ids, coord_stream=coord_stream)
        
        x = self.norm_final(x)
        
        # Word weights
        weights, w_shift = self.word_weight(x)
        
        scores = None
        if return_scores:
            scores = self.decoder.forward(x + w_shift * 0.1)
        
        if return_weights:
            return x, scores, weights
        return x, scores
    
    def summary(self) -> str:
        params = sum(p.numel() for p in self.parameters())
        return f"UnifiedTransformer(dim={self.coord_dim}, layers={self.num_layers}, adaptive, params={params:,})"
