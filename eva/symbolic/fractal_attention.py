"""
FractalAttention — динамические головы в многомерном пространстве.

Принципы:
1. Головы НЕ фиксированы. Они разворачиваются из контекста.
2. Каждый уровень иерархии → своя система координат (вложенная).
3. Маски — проекции между координатными системами уровней.
4. Вычисления — в едином многомерном пространстве (прямое произведение координат).

Фрактальность:
  coord(domain) → ∂coord/∂symbol = проекция на символьный уровень
  coord(sentence) = ∫ coord(words) по словам предложения
  coord(word) = Σ w_i · coord(symbol_i) / Σ w_i

Маска уровня N = attention в координатном пространстве уровня N.
Кросс-уровневая маска = проекция координат уровня N на уровень M.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class FractalHead:
    """Динамическая голова — разворачивается из контекста."""
    level: int                          # 0=symbol,1=word,2=sentence,3=domain
    scale: int                          # радиус внимания (1,2,4,8...)
    active: bool = True                 # активна ли в текущем контексте
    coord_weight: float = 1.0           # вес координатного сигнала
    attention_weight: float = 1.0       # вес attention-сигнала
    gate: float = 0.5                   # насколько доверять этой голове


class FractalCoordinateSystem:
    """
    Многомерное координатное пространство.

    Уровни вложены: domain ⊃ sentence ⊃ word ⊃ symbol.
    Каждый уровень — своё подпространство.
    Общее пространство = прямое произведение подпространств.
    """

    def __init__(self, dim_per_level: int = 8, num_levels: int = 4):
        self.dim_per_level = dim_per_level
        self.num_levels = num_levels
        self.total_dim = dim_per_level * num_levels

        # Координаты для каждого уровня (заполняются из данных)
        self.coords: Dict[int, Dict[int, np.ndarray]] = {
            level: {} for level in range(num_levels)
        }  # level → {entity_id → np.ndarray[dim_per_level]}

        # Проекционные матрицы: уровень M → уровень N
        self.projections: Dict[Tuple[int, int], np.ndarray] = {}

    def set_coordinates(self, level: int, entity_id: int, coords: np.ndarray):
        """Установить координаты сущности на уровне."""
        self.coords[level][entity_id] = coords[:self.dim_per_level]

    def get_coordinates(self, level: int, entity_id: int) -> np.ndarray:
        """Получить координаты сущности."""
        return self.coords[level].get(entity_id, np.zeros(self.dim_per_level))

    def project(self, from_level: int, to_level: int, coords: np.ndarray) -> np.ndarray:
        """Спроецировать координаты с уровня from на уровень to."""
        key = (from_level, to_level)
        if key not in self.projections:
            # Инициализируем случайной проекцией (обучается через backprop)
            self.projections[key] = np.random.randn(self.dim_per_level, self.dim_per_level) * 0.1
        return coords @ self.projections[key]

    def to_full_space(self, level: int, entity_id: int) -> np.ndarray:
        """
        Развернуть сущность в полное координатное пространство.

        entity на уровне level → вектор [total_dim]:
          [0...dim-1]           = координаты уровня level
          [dim...2dim-1]        = проекция на уровень level+1
          ...
        """
        result = np.zeros(self.total_dim)
        entity_coords = self.get_coordinates(level, entity_id)

        for target_level in range(self.num_levels):
            if target_level == level:
                projected = entity_coords
            else:
                projected = self.project(level, target_level, entity_coords)

            offset = target_level * self.dim_per_level
            result[offset:offset + self.dim_per_level] = projected

        return result


class FractalAttentionMask(nn.Module):
    """
    Фрактальная маска внимания: multi-level, multi-scale, в многомерном пространстве.

    Маска[k] для уровня k строится из координатных проекций.
    Внимание между сущностями i,j на уровне k:
      attention[i,j] = f(||proj(coord_i) - proj(coord_j)||, scale(k))
    """

    def __init__(
        self,
        d_model: int = 256,
        num_base_heads: int = 8,
        max_scale: int = 8,
        coord_dim_per_level: int = 8,
        num_levels: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_base_heads = num_base_heads
        self.max_scale = max_scale
        self.num_levels = num_levels

        # Координатная система
        self.space = FractalCoordinateSystem(coord_dim_per_level, num_levels)

        # Обучаемые проекции для каждой головы
        self.head_projections = nn.Parameter(
            torch.randn(num_base_heads, d_model, coord_dim_per_level) * 0.1
        )

        # Gate-сеть: определяет, сколько голов каждого уровня нужно
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, num_levels * max_scale),
        )

        # Масштабные веса
        self.scale_weights = nn.Parameter(torch.ones(max_scale))

    def compute_heads(self, context_vector: torch.Tensor) -> List[FractalHead]:
        """
        Развернуть головы из контекста.

        Gate-сеть решает:
        - Сколько голов нужно для каждого уровня
        - Какие масштабы активны
        """
        gate_logits = self.gate_net(context_vector)  # [num_levels * max_scale]
        gate_logits = gate_logits.view(self.num_levels, self.max_scale)

        heads = []
        for level in range(self.num_levels):
            for scale_idx in range(self.max_scale):
                scale = 2 ** scale_idx
                if scale > self.max_scale:
                    continue

                gate_val = torch.sigmoid(gate_logits[level, scale_idx]).item()
                active = gate_val > 0.3  # Порог активации

                head = FractalHead(
                    level=level,
                    scale=scale,
                    active=active,
                    gate=gate_val,
                    coord_weight=self.scale_weights[scale_idx].item(),
                )
                heads.append(head)

        return heads

    def build_mask(
        self,
        level: int,
        entity_coords: torch.Tensor,     # [N, coord_dim]
        entity_ids: List[int],
        scale: int,
    ) -> torch.Tensor:
        """
        Построить маску внимания для уровня.

        Маска[i,j] = exp(-||coords[i] - coords[j]||² / (2 * scale²))

        Чем ближе в координатном пространстве — тем сильнее внимание.
        Масштаб определяет радиус: scale=1 → только соседи, scale=8 → весь уровень.
        """
        N = entity_coords.shape[0]
        # Pairwise расстояния в координатном пространстве
        diffs = entity_coords.unsqueeze(0) - entity_coords.unsqueeze(1)  # [N, N, D]
        dists = torch.norm(diffs, dim=-1)  # [N, N]

        # Гауссово ядро с масштабом
        sigma = scale * 0.5
        mask = torch.exp(-dists ** 2 / (2 * sigma ** 2))

        # Диагональ = 0 (не attend to self)
        mask.fill_diagonal_(0)

        return mask

    def multi_level_attention(
        self,
        x: torch.Tensor,  # [B, L, d_model]
        symbol_ids: torch.Tensor,  # [B, L]
        word_boundaries: Optional[List[List[int]]] = None,
    ) -> torch.Tensor:
        """
        Многоуровневое внимание с фрактальными масками.

        Для каждого уровня:
        1. Развернуть координаты сущностей
        2. Построить маску через расстояния в координатном пространстве
        3. Применить attention
        4. Спроецировать результат обратно на символьный уровень
        """
        B, L, D = x.shape
        device = x.device

        # Контекстный вектор (усреднение по последовательности)
        context_vec = x.mean(dim=1).mean(dim=0)  # [d_model]

        # Развернуть головы из контекста
        heads = self.compute_heads(context_vec)
        active_heads = [h for h in heads if h.active]

        if not active_heads:
            return x  # Нет активных голов → возвращаем как есть

        # Обновить координаты символов в пространстве
        for i in range(L):
            sym = symbol_ids[0, i].item()
            coords = self.head_projections[0] @ x[0, i]  # проекция на координаты
            self.space.set_coordinates(0, sym, coords.detach().cpu().numpy())

        outputs = []
        weights = []

        for head in active_heads:
            level = head.level
            scale = head.scale

            # Получить координаты для этого уровня
            if level == 0:  # Символьный уровень
                entity_coords = torch.stack([
                    torch.tensor(
                        self.space.get_coordinates(0, symbol_ids[0, i].item()),
                        device=device, dtype=torch.float32
                    )
                    for i in range(L)
                ])
                mask = self.build_mask(level, entity_coords, [], scale)
                # Attention с маской
                attn_out = F.scaled_dot_product_attention(
                    x, x, x, attn_mask=mask.unsqueeze(0).unsqueeze(0)
                )

            elif level >= 1 and word_boundaries:
                # Word/sentence/domain level: группируем по границам
                groups = self._group_by_level(level, symbol_ids[0].tolist(), word_boundaries)
                if groups:
                    group_tensors = []
                    group_masks = []
                    for g_symbols in groups:
                        if not g_symbols: continue
                        g_coords = torch.stack([
                            torch.tensor(
                                self.space.to_full_space(0, s),
                                device=device, dtype=torch.float32
                            )
                            for s in g_symbols
                        ])
                        group_tensors.append(g_coords)
                        group_masks.append(self.build_mask(level, g_coords, [], scale))

                    if group_tensors and group_masks:
                        # Attention внутри каждой группы
                        group_outputs = []
                        for gt, gm in zip(group_tensors, group_masks):
                            if gt.shape[0] > 1:
                                go = F.scaled_dot_product_attention(
                                    gt.unsqueeze(0), gt.unsqueeze(0), gt.unsqueeze(0),
                                    attn_mask=gm.unsqueeze(0).unsqueeze(0)
                                )
                                group_outputs.append(go.mean(dim=1))
                        if group_outputs:
                            attn_out = torch.cat(group_outputs, dim=1)
                            if attn_out.shape[1] < L:
                                attn_out = F.pad(attn_out, (0, 0, 0, L - attn_out.shape[1]))
                            elif attn_out.shape[1] > L:
                                attn_out = attn_out[:, :L, :]
                        else:
                            attn_out = x
                    else:
                        attn_out = x
                else:
                    attn_out = x
            else:
                attn_out = x

            outputs.append(attn_out * head.gate)
            weights.append(head.gate)

        if outputs:
            total_weight = sum(weights)
            result = sum(o * w / total_weight for o, w in zip(outputs, weights))
            return result

        return x

    def _group_by_level(self, level: int, symbol_ids: List[int],
                        word_boundaries: Optional[List[List[int]]]) -> List[List[int]]:
        """Сгруппировать символы по уровню."""
        if level == 1 and word_boundaries:
            return word_boundaries  # Слова → группы символов

        if level == 2 and word_boundaries:
            # Предложения → группы слов (по 3-5 слов)
            chunks = []
            chunk = []
            for w in word_boundaries:
                chunk.extend(w)
                if len(chunk) >= 15:  # ~3-5 слов
                    chunks.append(chunk)
                    chunk = []
            if chunk: chunks.append(chunk)
            return chunks

        if level == 3:
            # Домен → все символы
            return [symbol_ids]

        return []

    def summary(self) -> str:
        return f"FractalAttention(base_heads={self.num_base_heads}, levels={self.num_levels})"
