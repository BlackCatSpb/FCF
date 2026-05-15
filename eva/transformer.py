"""
TransformerBlock — базовый строительный блок PrimordialLayer.

Содержит:
- Causal Multi-Head Self-Attention
- SwiGLU Feed-Forward Network
- RMSNorm (Pre-Norm архитектура)
- Xavier/Glorot инициализацию всех весов
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.cat([freqs, freqs], dim=-1)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs: torch.Tensor):
    cos = freqs[:, :xq.shape[-1]].cos().unsqueeze(0).unsqueeze(0)
    sin = freqs[:, :xq.shape[-1]].sin().unsqueeze(0).unsqueeze(0)

    def rotate(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    return (xq * cos + rotate(xq) * sin), (xk * cos + rotate(xk) * sin)


class CausalSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(self.head_dim, max_seq_len),
            persistent=False,
        )

        self._k_cache = None
        self._v_cache = None

        self._init_weights()

    def _init_weights(self):
        for module in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(module.weight)

    def reset_cache(self):
        self._k_cache = None
        self._v_cache = None

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None,
                use_cache: bool = False):
        B, T, C = x.shape

        q = self.W_Q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_K(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_V(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        total_T = T
        offset = 0

        if use_cache and self._k_cache is not None:
            offset = self._k_cache.shape[2]
            total_T = offset + T
            k = torch.cat([self._k_cache, k], dim=2)
            v = torch.cat([self._v_cache, v], dim=2)

        freqs_q = self.rope_freqs[offset:total_T, :self.head_dim].to(x.device)
        freqs_k = self.rope_freqs[:total_T, :self.head_dim].to(x.device)

        q, _ = apply_rotary_emb(q, torch.zeros_like(q), freqs_q)
        if use_cache and self._k_cache is not None:
            k_full = k.clone()
            k_new = k[:, :, offset:, :]
            _, k_new_rope = apply_rotary_emb(torch.zeros_like(k_new), k_new, freqs_k[offset:, :])
            k_full[:, :, offset:, :] = k_new_rope
            k = k_full
        else:
            _, k = apply_rotary_emb(torch.zeros_like(k), k, freqs_k)

        if use_cache:
            self._k_cache = k.detach()
            self._v_cache = v.detach()

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal_mask = torch.triu(
            torch.ones(T, total_T, device=x.device, dtype=torch.bool),
            diagonal=1 + offset,
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_O(attn_output)


class SwiGLUFFN(nn.Module):

    def __init__(self, d_model: int, ff_mult: int = 4):
        super().__init__()
        hidden_dim = d_model * ff_mult
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int = 2560,
        num_heads: int = 32,
        ff_mult: int = 4,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.attention = CausalSelfAttention(d_model, num_heads, max_seq_len)
        self.ffn = SwiGLUFFN(d_model, ff_mult)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), attention_mask, use_cache=use_cache)
        x = x + self.ffn(self.norm2(x))
        return x

    def get_kv_state(self, x: torch.Tensor) -> tuple:
        x_norm = self.norm1(x)
        B, T, C = x_norm.shape

        k = (
            self.attention.W_K(x_norm)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.attention.W_V(x_norm)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        last_k = k[:, :, -1, :].detach().cpu().numpy()
        last_v = v[:, :, -1, :].detach().cpu().numpy()
        return last_k, last_v
