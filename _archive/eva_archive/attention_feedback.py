"""
AttentionFeedback — метаданные модифицируют attention трансформера.

Замыкает петлю обратной связи:

  Transformer → attention → метаданные (contradictions, grammar, topology)
       ↑                                                    |
       └────────────── feedback (modify attention) ←────────┘

Как работает:
1. После forward pass извлекаем attention_matrix
2. Символьные метаданные вычисляют "поправки" к attention
3. Поправки применяются к attention ПЕРЕД следующим forward pass
   через модификацию позиционных эмбеддингов или bias к attention scores

Типы поправок:
- Contradiction feedback: ослабить attention к запрещённым связям
- Grammar feedback: усилить attention к валидным паттернам
- Topology feedback: направить attention в сторону плотных областей многообразия
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from loguru import logger


class AttentionFeedback(nn.Module):
    """
    Модифицирует attention трансформера на основе символьных метаданных.

    Attention bias добавляется к attention scores перед softmax:
      new_attention = softmax(QK^T/√d + bias)
    """

    def __init__(
        self,
        potential_field,
        contradiction_filter,
        grammar,
        topological_field,
        feedback_strength: float = 0.1,
    ):
        super().__init__()

        self.potential_field = potential_field
        self.contradiction_filter = contradiction_filter
        self.grammar = grammar
        self.topological_field = topological_field
        self.feedback_strength = feedback_strength

        # Обучаемые коэффициенты для разных типов feedback
        self.contra_scale = nn.Parameter(torch.tensor(-1.0))   # negative → suppress
        self.grammar_scale = nn.Parameter(torch.tensor(1.0))   # positive → boost
        self.topo_scale = nn.Parameter(torch.tensor(0.5))

    def compute_attention_bias(
        self,
        symbol_indices: torch.Tensor,  # [B, L]
    ) -> Optional[torch.Tensor]:
        """
        Вычислить bias матрицу [B, L, L] для модификации attention.

        bias[i,j] > 0 → усилить внимание между i и j
        bias[i,j] < 0 → ослабить внимание между i и j
        """
        B, L = symbol_indices.shape
        device = symbol_indices.device
        bias = torch.zeros(B, L, L, device=device)

        for b in range(B):
            for i in range(L):
                si = symbol_indices[b, i].item()
                if si >= self.potential_field.vocab_size:
                    continue

                for j in range(L):
                    if i == j:
                        continue
                    sj = symbol_indices[b, j].item()
                    if sj >= self.potential_field.vocab_size:
                        continue

                    # 1. Противоречивый feedback
                    context = symbol_indices[b, :i+1].tolist()
                    forbidden, conf, _ = self.contradiction_filter.is_forbidden(context, sj)
                    if forbidden:
                        bias[b, i, j] += self.contra_scale * conf

                    # 2. Грамматический feedback
                    if j == i + 1:
                        for _, pats in self.grammar.patterns.items():
                            for ph, pattern in pats.items():
                                if pattern.length >= 2:
                                    pat_symbols = pattern.symbol_indices[:pattern.length]
                                    if pat_symbols[0] == si and pat_symbols[1] == sj:
                                        bias[b, i, j] += self.grammar_scale * pattern.coherence_score

                    # 3. Топологический feedback
                    if si in self.topological_field.points and sj in self.topological_field.points:
                        ci = self.topological_field.points[si].coordinates
                        cj = self.topological_field.points[sj].coordinates
                        dist = np.linalg.norm(ci - cj)
                        proximity = 1.0 / (1.0 + dist)
                        bias[b, i, j] += self.topo_scale * proximity

        return bias * self.feedback_strength

    def apply_to_layer(
        self,
        layer,
        symbol_indices: torch.Tensor,
    ):
        """
        Применить attention bias к конкретному слою трансформера.

        Модифицирует attention модуль слоя, добавляя bias к attention scores.
        """
        bias = self.compute_attention_bias(symbol_indices)
        if bias is None:
            return

        attention_mod = layer.transformer.attention

        original_forward = attention_mod.forward

        def modified_forward(x, mask=None, use_cache=False):
            B, T, C = x.shape

            q = attention_mod.W_Q(x)
            k = attention_mod.W_K(x)
            v = attention_mod.W_V(x)

            # Reshape для multi-head
            head_dim = C // attention_mod.num_heads
            q = q.view(B, T, attention_mod.num_heads, head_dim).transpose(1, 2)
            k = k.view(B, T, attention_mod.num_heads, head_dim).transpose(1, 2)
            v = v.view(B, T, attention_mod.num_heads, head_dim).transpose(1, 2)

            # Attention scores
            scale = head_dim ** -0.5
            attn_scores = (q @ k.transpose(-2, -1)) * scale

            # ПРИМЕНЯЕМ BIAS
            if bias is not None and bias.shape[1] == T and bias.shape[2] == T:
                attn_scores = attn_scores + bias.unsqueeze(1)

            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

            attn_weights = torch.softmax(attn_scores, dim=-1)
            attention_mod.last_attention = attn_weights

            out = attn_weights @ v
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            out = attention_mod.W_O(out)

            return out

        attention_mod.forward = modified_forward

    def summary(self) -> str:
        return (
            f"AttentionFeedback(strength={self.feedback_strength}, "
            f"contra={self.contra_scale.item():.2f}, "
            f"grammar={self.grammar_scale.item():.2f})"
        )
