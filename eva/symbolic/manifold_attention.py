"""
ManifoldAttention — многослойное иерархическое внимание в многообразии.

Заменяет классическое внимание softmax(QK^T) на топологически-обоснованное:
  attention[i][j] = combine(
    dot_product(Q_i, K_j),       # классический сигнал
    manifold_proximity(i, j),    # близость в 3D-координатах
    domain_affinity(i, j),       # сила меж-доменной связи
    hierarchy_level(i, j)        # кросс-уровневое внимание
  )

Уровни внимания:
  L0: символ→символ (локальное)
  L1: символ→слово (полу-локальное)  
  L2: слово→фраза (глобальное)
  L3: домен→домен (контекстное)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from loguru import logger


class ManifoldAttention(nn.Module):
    """
    Многослойное внимание, управляемое топологией многообразия.

    Комбинирует 4 сигнала:
    1. dot_product (классический attention score)
    2. manifold_proximity (близость в координатном пространстве)
    3. domain_affinity (сила меж-доменной связи)
    4. hierarchy_bias (вес связи на разных уровнях иерархии)

    Веса сигналов — обучаемые параметры.
    """

    def __init__(self, d_model: int = 256, num_heads: int = 8, coord_dim: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.coord_dim = coord_dim

        # Обучаемые веса для комбинирования сигналов
        self.w_dot = nn.Parameter(torch.tensor(1.0))
        self.w_manifold = nn.Parameter(torch.tensor(0.3))
        self.w_domain = nn.Parameter(torch.tensor(0.2))
        self.w_hierarchy = nn.Parameter(torch.tensor(0.1))

        # Проекция координат в пространство attention
        self.coord_proj = nn.Linear(coord_dim, num_heads, bias=False)

        # Иерархические уровни (0=символ,1=слово,2=фраза,3=предложение)
        self.level_embedding = nn.Parameter(torch.randn(4, num_heads))

    def compute_manifold_bias(
        self,
        coordinates: torch.Tensor,     # [L, coord_dim]
        domain_ids: Optional[torch.Tensor] = None,  # [L]
        hierarchy_levels: Optional[torch.Tensor] = None,  # [L]
        domain_affinity_matrix: Optional[torch.Tensor] = None,  # [L, L]
    ) -> torch.Tensor:
        """
        Вычислить bias матрицу на основе топологии многообразия.

        Returns: bias [L, L] — добавляется к attention scores перед softmax.
        """
        L = coordinates.shape[0]
        device = coordinates.device

        bias = torch.zeros(L, L, device=device)

        # 1. Manifold proximity: насколько близки точки в 3D
        coord_proj = self.coord_proj(coordinates)  # [L, num_heads]
        # Pairwise distance в projected space
        diffs = coord_proj.unsqueeze(0) - coord_proj.unsqueeze(1)  # [L, L, H]
        manifold_sim = -torch.norm(diffs, dim=-1)  # [L, L] — отрицательное расстояние
        manifold_sim = (manifold_sim - manifold_sim.mean()) / (manifold_sim.std() + 1e-8)
        bias += self.w_manifold * manifold_sim

        # 2. Domain affinity: символы в одном домене получают boost
        if domain_ids is not None:
            same_domain = (domain_ids.unsqueeze(0) == domain_ids.unsqueeze(1)).float()
            bias += self.w_domain * same_domain

        # 3. Предзаданная доменная аффинность (из KnowledgeBase)
        if domain_affinity_matrix is not None:
            bias += self.w_domain * 0.5 * domain_affinity_matrix

        # 4. Иерархический bias
        if hierarchy_levels is not None:
            level_diff = (hierarchy_levels.unsqueeze(0) - hierarchy_levels.unsqueeze(1)).abs()
            # Близкие уровни → больший вес
            level_bias = -level_diff.float() * 0.5
            bias += self.w_hierarchy * level_bias

        return bias

    def forward(
        self,
        query: torch.Tensor,    # [B, L, d_model]
        key: torch.Tensor,      # [B, L, d_model]
        value: torch.Tensor,    # [B, L, d_model]
        coordinates: Optional[torch.Tensor] = None,  # [L, coord_dim]
        domain_ids: Optional[torch.Tensor] = None,
        hierarchy_levels: Optional[torch.Tensor] = None,
        domain_affinity: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Multi-layer manifold-aware attention.

        Если координаты не переданы — работает как обычный attention.
        """
        B, L, _ = query.shape
        H = self.num_heads
        D = self.head_dim

        # Reshape для multi-head
        q = query.view(B, L, H, D).transpose(1, 2)  # [B, H, L, D]
        k = key.view(B, L, H, D).transpose(1, 2)
        v = value.view(B, L, H, D).transpose(1, 2)

        # Стандартные attention scores
        scale = D ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, L, L]

        # Добавляем топологический bias
        if coordinates is not None:
            manifold_bias = self.compute_manifold_bias(
                coordinates, domain_ids, hierarchy_levels, domain_affinity
            )
            # Расширяем bias до [B, H, L, L]
            manifold_bias = manifold_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
            attn_scores = attn_scores + self.w_dot * manifold_bias

        # Маска
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Output
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)

        return out


class MultiScaleAttentionStack(nn.Module):
    """
    Стек многослойного внимания на разных масштабах.

    Уровень 0: символ→символ (полное разрешение, L токенов)
    Уровень 1: символ→группа (среднее, L//2 токенов)
    Уровень 2: группа→слово (низкое, L//4 токенов)

    Каждый уровень использует ManifoldAttention с координатами
    соответствующего масштаба.
    """

    def __init__(self, d_model: int = 256, num_heads: int = 8, num_levels: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_levels = num_levels

        self.attention_layers = nn.ModuleList([
            ManifoldAttention(d_model, num_heads) for _ in range(num_levels)
        ])

        # Веса уровней
        self.level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)

    def forward(
        self,
        x: torch.Tensor,  # [B, L, d_model]
        coordinates: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Multi-scale forward pass.

        Level 0: полное разрешение
        Level 1: pooling ×2
        Level 2: pooling ×4
        """
        B, L, D = x.shape
        outputs = []

        for level in range(self.num_levels):
            scale = 2 ** level
            if L // scale < 1:
                break

            # Pooling для этого уровня (если нужно)
            if scale > 1:
                # Average pooling по временной оси
                x_pooled = x[:, :L - (L % scale), :].view(B, L // scale, scale, D).mean(dim=2)
                coords_pooled = coordinates[:L - (L % scale), :].view(L // scale, scale, -1).mean(dim=1) if coordinates is not None else None
                doms_pooled = domain_ids[:L - (L % scale)].view(L // scale, scale).mode(dim=1).values if domain_ids is not None else None
            else:
                x_pooled = x
                coords_pooled = coordinates
                doms_pooled = domain_ids

            # Attention на этом уровне
            L_scaled = x_pooled.shape[1]
            out = self.attention_layers[level](
                x_pooled, x_pooled, x_pooled,
                coordinates=coords_pooled,
                domain_ids=doms_pooled,
            )

            # Upsample обратно
            if scale > 1:
                out = out.unsqueeze(2).expand(-1, -1, scale, -1).contiguous().view(B, L_scaled * scale, D)
                out = out[:, :L, :]

            outputs.append(out)

        # Взвешенная сумма уровней
        weights = F.softmax(self.level_weights, dim=0)
        result = torch.zeros_like(x)
        for level, out in enumerate(outputs):
            if out.shape[1] == L:
                result += weights[level] * out

        return result


class CoordinateProjector:
    """
    Проецирует символы из 3D-многообразия на 2D для визуализации.

    Использует PCA на координатах для получения главных компонент.
    """

    def __init__(self, potential_field, topological_field):
        self.pf = potential_field
        self.topo = topological_field

    def project_to_2d(self, symbol_indices: List[int]) -> np.ndarray:
        """Спроецировать символы на 2D через их 3D-координаты."""
        coords_3d = []
        labels = []
        for si in symbol_indices:
            if si < self.topo.coordinates.shape[0]:
                coords_3d.append(self.topo.coordinates[si].cpu().numpy())
                labels.append(si)

        if not coords_3d:
            return np.zeros((0, 2))

        coords = np.array(coords_3d)
        # PCA до 2D
        centered = coords - coords.mean(axis=0)
        if centered.shape[0] > 2:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            proj_2d = centered @ Vt[:2].T
        else:
            proj_2d = centered[:, :2]

        return proj_2d

    def project_domains_to_2d(
        self, domain_ids: List[int], knowledge_base
    ) -> Dict[int, np.ndarray]:
        """Спроецировать домены на 2D."""
        result = {}
        for did in domain_ids:
            if did in knowledge_base.domains:
                symbols = knowledge_base.domains[did].symbol_indices[:50]
                result[did] = self.project_to_2d(symbols)
        return result

    def summary(self) -> str:
        return f"CoordinateProjector(dims={self.topo.coord_dim})"
