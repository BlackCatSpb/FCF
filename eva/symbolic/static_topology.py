"""
StaticTopologyLayer — статическая матрица топологии вычислений.

Объединяет координаты, потенциалы и запреты в единый тензор [V, V, 3]:
- Канал 0: affinity (сила связи i→j)
- Канал 1: potential barrier (высота барьера между i и j)
- Канал 2: forbidden (0=разрешено, 1=запрещено)

Fast Path: кэш успешных траекторий для быстрой генерации.
"""

import torch, torch.nn as nn, numpy as np
from typing import Tuple, Optional


class StaticTopologyLayer(nn.Module):
    """
    Precomputed topology matrix [V, V, 3] — read-only bias для attention.
    
    Интегрируется в HybridFractalBlock перед Gate Merge.
    """
    
    def __init__(self, vocab_size=160, coord_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.coord_dim = coord_dim
        
        # Топологическая матрица [V, V, 3]
        self.register_buffer('topology', torch.zeros(vocab_size, vocab_size, 3))
        
        # Fast Path cache: центроид → список успешных траекторий
        self.register_buffer('fast_path_keys', torch.zeros(0, coord_dim))
        self.fast_path_values = []  # List of trajectory continuations
        
        # Обучаемый projection: топология → bias для attention
        self.topo_proj = nn.Linear(3, 1, bias=False)
    
    def build_topology(self, affinity, potential_func, contradiction_filter, coords):
        """
        Построить статическую матрицу из существующих компонентов.
        
        Args:
            affinity: [V, V] — матрица аффинности
            potential_func: V(z) — скалярный потенциал
            contradiction_filter: фильтр запретов
            coords: [V, D] — координаты символов
        """
        V = self.vocab_size
        
        with torch.no_grad():
            for i in range(V):
                for j in range(V):
                    if i == j: continue
                    
                    # Канал 0: affinity (нормализованная)
                    self.topology[i, j, 0] = affinity[i, j].item()
                    
                    # Канал 1: potential barrier между i и j
                    za = coords[i:i+1]
                    zb = coords[j:j+1]
                    mid = (za + zb) / 2
                    v_i = potential_func(za).item()
                    v_j = potential_func(zb).item()
                    v_mid = potential_func(mid).item()
                    barrier = max(0, v_mid - max(v_i, v_j))
                    self.topology[i, j, 1] = float(barrier)
                    
                    # Канал 2: forbidden
                    if hasattr(contradiction_filter, 'forbidden_mask') and contradiction_filter.forbidden_mask is not None:
                        self.topology[i, j, 2] = float(contradiction_filter.forbidden_mask[i, j])
    
    def get_topology_bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Преобразовать топологию в bias для attention scores.
        
        Args:
            token_ids: [B, L] — текущие ID токенов
        
        Returns:
            topology_bias: [B, L, L] — bias для добавления к QK^T
        """
        B, L = token_ids.shape
        device = token_ids.device
        
        ids = token_ids.clamp(0, self.vocab_size - 1)
        
        # Извлечь топологию для всех пар токенов
        topo_pairs = self.topology[ids.unsqueeze(-1), ids.unsqueeze(1)]  # [B, L, L, 3]
        
        # Проекция 3 каналов → 1 скаляр
        bias = self.topo_proj(topo_pairs).squeeze(-1)  # [B, L, L]
        
        # Инвертировать запреты: forbidden → -inf
        forbidden = topo_pairs[..., 2]  # [B, L, L]
        bias = bias - forbidden * 1e9
        
        return bias
    
    def get_optimal_direction(self, token_ids: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Найти оптимальное направление движения из текущей позиции.
        
        Returns: direction [B, L, D] — вектор следующего шага
        """
        B, L = token_ids.shape
        ids = token_ids.clamp(0, self.vocab_size - 1)
        
        # Для каждого токена найти топ-1 продолжение по affinity
        directions = torch.zeros(B, L, self.coord_dim, device=token_ids.device)
        
        for b in range(B):
            for i in range(L):
                curr = ids[b, i].item()
                # Найти самый вероятный следующий токен (исключая запреты)
                aff = self.topology[curr, :, 0].clone()
                forb = self.topology[curr, :, 1] > 0.5  # barrier > 0.5 → трудно
                aff[forb] = -1e9
                best_next = aff.argmax().item()
                
                # Направление = координата_next - координата_curr
                directions[b, i] = coords[best_next] - coords[curr]
        
        return directions
    
    def cache_fast_path(self, query_centroid: torch.Tensor, continuation_ids: list):
        """
        Кэшировать успешную траекторию для Fast Path генерации.
        
        Args:
            query_centroid: [D] — центроид запроса
            continuation_ids: [L] — успешное продолжение
        """
        self.fast_path_values.append(continuation_ids)
        if len(self.fast_path_keys) == 0:
            self.fast_path_keys = query_centroid.unsqueeze(0)
        else:
            self.fast_path_keys = torch.cat([self.fast_path_keys, query_centroid.unsqueeze(0)])
    
    def fast_path_lookup(self, query_centroid: torch.Tensor, top_k=5, threshold=0.1):
        """
        Fast Path: найти кэшированную траекторию по центроиду запроса.
        
        Returns: список продолжений или None если нет близких
        """
        if len(self.fast_path_keys) == 0:
            return None
        
        dists = torch.norm(self.fast_path_keys - query_centroid, dim=-1)
        min_dist, min_idx = dists.min(dim=0)
        
        if min_dist < threshold:
            return [self.fast_path_values[min_idx.item()]]
        
        # Top-K ближайших
        _, top_indices = dists.topk(min(top_k, len(dists)), largest=False)
        return [self.fast_path_values[i.item()] for i in top_indices]
    
    def build_from_store(self, trajectory_store):
        """Построить Fast Path кэш из TrajectoryStore."""
        if trajectory_store.total_stored == 0:
            return
        
        for i in range(trajectory_store.total_stored):
            centroid = trajectory_store.centroids[i]
            ids = trajectory_store.ids_list[i]
            if len(ids) > 5:
                self.cache_fast_path(
                    torch.tensor(centroid),
                    ids[5:]  # продолжение после первых 5 токенов
                )
