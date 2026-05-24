"""
UnifiedMultidimensionalTransformer — единое ℝ¹² пространство.

Символ ≡ позиция в ℝ¹². Не индекс.
Трансформер навигирует по координатам, а не выбирает из словаря.
FractalAttention на 4 уровнях × 3 масштаба = 12 голов.

Архитектура:
  Text → CharVocab → координаты из TopologicalField → ℝ¹²
       → FractalAttention (12 голов) → предсказанная координата
       → nearest_symbol(координата) → output символ

GPU-native: 12-dim × 12 heads = 144 ops/head — идеально для 2000+ ядер.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger


# ============================================================
# 1. CoordinateEmbedding — символ → ℝ¹² (без lookup)
# ============================================================

class CoordinateEmbedding(nn.Module):
    """
    Координатный эмбеддинг.
    НЕ nn.Embedding. Каждый символ имеет фиксированную позицию в ℝ¹².
    Позиции загружаются из TopologicalField (MDS на affinity).
    """

    def __init__(self, vocab_size: int = 156, coord_dim: int = 12):
        super().__init__()
        self.vocab_size = vocab_size
        self.coord_dim = coord_dim

        # Координаты символов — загружаются извне, не обучаются
        self.register_buffer("coordinates", torch.randn(vocab_size, coord_dim) * 0.1)

        # Масштабный фактор (обучаемый)
        self.scale = nn.Parameter(torch.ones(1))

    def set_coordinates(self, coords: torch.Tensor):
        """Загрузить координаты из TopologicalField MDS."""
        assert coords.shape[0] == self.vocab_size, f"Expected {self.vocab_size}, got {coords.shape[0]}"
        self.coordinates.copy_(coords[:, :self.coord_dim])

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: [B, L] индексы символов
        returns: [B, L, coord_dim] координаты в ℝ¹²
        """
        # Clamp to valid range
        ids = token_ids.clamp(0, self.vocab_size - 1)
        return self.coordinates[ids] * self.scale


class CoordinateDecoder(nn.Module):
    """
    Декодер: ℝ¹² → 156 символов.
    Обучаемый линейный классификатор + nearest neighbor fallback.
    """

    def __init__(self, coord_embedding: CoordinateEmbedding):
        super().__init__()
        self.embed = coord_embedding
        self.vocab_size = coord_embedding.vocab_size
        self.coord_dim = coord_embedding.coord_dim

        # Обучаемый линейный классификатор (12 → 156)
        self.linear = nn.Linear(coord_embedding.coord_dim, coord_embedding.vocab_size)

        # Температура для sharpen распределения
        self.temperature = nn.Parameter(torch.tensor(1.0))
        # Learnable weight for nearest-neighbor signal
        self.nn_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, coord_dim] предсказанные координаты
        returns: [B, L, vocab_size] логиты
        """
        # Основной путь: обучаемый линейный слой
        logits = self.linear(x) / self.temperature.clamp(min=0.1)

        # Nearest-neighbor: расстояние до каждого символа
        coords = self.embed.coordinates  # [V, D]
        diffs = x.unsqueeze(2) - coords.unsqueeze(0).unsqueeze(0)  # [B, L, V, D]
        dists = torch.norm(diffs, dim=-1)  # [B, L, V]
        nn_logits = -dists * self.nn_weight  # scaled by learnable weight

        return logits + nn_logits

    def decode_to_ids(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D] → [B, L] индексы ближайших символов."""
        scores = self.forward(x)  # [B, L, V]
        return torch.argmax(scores, dim=-1)

    def decode_to_text(self, x: torch.Tensor, char_vocab) -> List[str]:
        """Декодировать в читаемый текст."""
        ids = self.decode_to_ids(x)
        results = []
        for b in range(ids.shape[0]):
            text = char_vocab.decode(ids[b].tolist())
            results.append(text)
        return results


# ============================================================
# 2. UnifiedMultidimensionalTransformer
# ============================================================

class UnifiedMultidimensionalTransformer(nn.Module):
    """
    Трансформер в едином ℝ¹² пространстве.

    Отличие от PrimordialLayer:
    - Нет nn.Embedding → CoordinateEmbedding
    - Нет lm_head → CoordinateDecoder
    - Вместо CausalSelfAttention → FractalAttention
    - d_model = 12 (не 256)
    """

    def __init__(
        self,
        vocab_size: int = 156,
        coord_dim: int = 12,
        num_heads: int = 12,       # 4 уровня × 3 масштаба
        num_levels: int = 4,
        max_scale: int = 3,
        ff_mult: int = 4,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.coord_dim = coord_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.max_scale = max_scale

        # Координатный эмбеддинг (вместо nn.Embedding)
        self.embed = CoordinateEmbedding(vocab_size, coord_dim)

        # NOTE: FractalAttention available via fractal_attention.py for future use
        # self.attention = FractalAttentionMask(...) — disabled, unused
        self.attention = None

        # Координатный декодер (вместо lm_head)
        self.decoder = CoordinateDecoder(self.embed)

        # Multi-head attention projections (learnable Q, K, V, O)
        self.attn_heads = 4  # 4 heads × 3D = 12D
        self.W_Q = nn.Linear(coord_dim, coord_dim, bias=False)
        self.W_K = nn.Linear(coord_dim, coord_dim, bias=False)
        self.W_V = nn.Linear(coord_dim, coord_dim, bias=False)
        self.W_O = nn.Linear(coord_dim, coord_dim, bias=False)
        hidden_dim = coord_dim * ff_mult
        self.ffn = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, coord_dim),
        )

        # Normalization
        self.norm1 = nn.LayerNorm(coord_dim)
        self.norm2 = nn.LayerNorm(coord_dim)

        # Position encoding (RoPE-like, но для ℝ¹²)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, coord_dim) * 0.02)

    def set_symbol_coordinates(self, coords: torch.Tensor):
        """Загрузить координаты символов из MDS."""
        self.embed.set_coordinates(coords)

    def forward(
        self,
        token_ids: torch.Tensor,  # [B, L]
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Returns: (координаты [B, L, coord_dim], scores [B, L, V] если return_scores)
        """
        B, L = token_ids.shape
        device = token_ids.device

        # 1. Токены → координаты в ℝ¹²
        x = self.embed(token_ids)  # [B, L, coord_dim]

        # 2. Добавляем позиционное кодирование
        pos = self.pos_encoding[:, :L, :].to(device)
        x = x + pos

        # 3. Multi-head self-attention with learnable projections
        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)
        H = self.attn_heads
        D = self.coord_dim
        # [B, L, D] → [B, H, L, D/H]
        q = q.view(B, L, H, D // H).transpose(1, 2)
        k = k.view(B, L, H, D // H).transpose(1, 2)
        v = v.view(B, L, H, D // H).transpose(1, 2)

        scale = (D // H) ** 0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        causal_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        # Merge heads back: [B, H, L, D/H] → [B, L, D]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = self.W_O(attn_out)
        x = self.norm1(x + attn_out)

        # 4. FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # 5. Декодер: координаты → scores (расстояния до символов)
        scores = None
        if return_scores:
            scores = self.decoder.forward(x)

        return x, scores

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_new: int = 128,
        temperature: float = 0.7,
    ) -> List[int]:
        """
        Генерация: навигация по координатному пространству.
        """
        device = next(self.parameters()).device
        generated = list(prompt_ids)
        context = list(prompt_ids)

        for _ in range(max_new):
            # Forward pass
            inp = torch.tensor([context[-512:]], dtype=torch.long, device=device)
            coords, scores = self.forward(inp, return_scores=True)

            # Scores для последней позиции
            last_scores = scores[0, -1] / max(temperature, 0.1)  # [V]
            probs = F.softmax(last_scores, dim=-1)

            # Sample
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            context.append(next_token)

        return generated

    # compute_loss removed — loss is defined in train_full_pipeline.py
    # (KL divergence + coordinate MSE)

    def summary(self) -> str:
        params = sum(p.numel() for p in self.parameters())
        return f"UnifiedTransformer(dim={self.coord_dim}, heads={self.num_heads}, params={params:,})"
