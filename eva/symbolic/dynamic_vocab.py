"""
Dynamic Coordinate Expansion + Latent Code Operators.

1. Dynamic expansion: add new symbol → interpolate coordinate from context.
2. Latent codes: synthesize new trajectories via State Algebra operators.
"""

import torch, numpy as np
from typing import List, Dict, Tuple


class DynamicVocabExpander:
    """
    Добавляет новые символы в словарь, вычисляя координаты 
    через контекстную близость к существующим символам.
    """
    
    def __init__(self, char_vocab, coords):
        self.cv = char_vocab
        self.coords = coords  # [V, D]
    
    def add_symbol(self, char: str, context_chars: List[str] = None) -> int:
        """
        Добавить новый символ и вычислить его координату.
        
        Args:
            char: новый символ
            context_chars: список контекстных символов для интерполяции
        
        Returns: ID нового символа
        """
        if char in self.cv._char_to_idx:
            return self.cv._char_to_idx[char]
        
        # Назначить новый ID
        new_id = self.cv.vocab_size
        self.cv._char_to_idx[char] = new_id
        self.cv._idx_to_char[new_id] = char
        self.cv.vocab_size = len(self.cv._char_to_idx)
        
        # Вычислить координату через интерполяцию
        if context_chars:
            ctx_ids = [self.cv._char_to_idx.get(c, 1) for c in context_chars if c in self.cv._char_to_idx]
            if ctx_ids:
                new_coord = self.coords[ctx_ids].mean(dim=0)
            else:
                new_coord = torch.randn(self.coords.shape[1]) * 0.02
        else:
            new_coord = torch.randn(self.coords.shape[1]) * 0.02
        
        # Нормализовать
        new_coord = new_coord / new_coord.norm().clamp(min=1e-8)
        
        # Расширить координатный тензор
        self.coords = torch.cat([self.coords, new_coord.unsqueeze(0)])
        
        return new_id
    
    def add_from_corpus(self, text: str):
        """Добавить все уникальные символы из текста."""
        added = 0
        for ch in set(text):
            if ch not in self.cv._char_to_idx:
                self.add_symbol(ch)
                added += 1
        return added


class LatentCodeOperator:
    """
    Операторы над траекториями как латентными кодами.
    
    Операции:
    - add: сложение двух траекторий (объединение смыслов)
    - interpolate: плавный переход от A к B
    - project: проецировать траекторию на подпространство
    - blend: смешать N траекторий с весами
    """
    
    def __init__(self, trajectory_store, coords):
        self.store = trajectory_store
        self.coords = coords
    
    def add(self, traj_a: np.ndarray, traj_b: np.ndarray) -> np.ndarray:
        """Сложение траекторий: поэлементная сумма + нормализация."""
        min_len = min(len(traj_a), len(traj_b))
        a = traj_a[:min_len]
        b = traj_b[:min_len]
        
        # Сложение в координатном пространстве
        result = a + b
        
        # Нормализация
        norms = np.linalg.norm(result, axis=1, keepdims=True) + 1e-8
        return result / norms
    
    def interpolate(self, traj_a: np.ndarray, traj_b: np.ndarray, 
                     steps: int = 5) -> List[np.ndarray]:
        """Интерполяция: плавный переход от A к B через N шагов."""
        min_len = min(len(traj_a), len(traj_b))
        a = traj_a[:min_len]
        b = traj_b[:min_len]
        
        result = []
        for i in range(steps):
            t = i / (steps - 1)
            interp = (1 - t) * a + t * b
            norms = np.linalg.norm(interp, axis=1, keepdims=True) + 1e-8
            result.append(interp / norms)
        
        return result
    
    def project(self, traj: np.ndarray, subspace_dims: Tuple[int, int]) -> np.ndarray:
        """Проекция траектории на подпространство [start:end]."""
        projected = np.zeros_like(traj)
        projected[:, subspace_dims[0]:subspace_dims[1]] = traj[:, subspace_dims[0]:subspace_dims[1]]
        norms = np.linalg.norm(projected, axis=1, keepdims=True) + 1e-8
        return projected / norms
    
    def blend(self, trajectories: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
        """Смешать N траекторий с весами."""
        if not trajectories:
            return np.zeros((0, self.coords.shape[1]))
        
        if weights is None:
            weights = [1.0 / len(trajectories)] * len(trajectories)
        
        min_len = min(len(t) for t in trajectories)
        
        result = np.zeros((min_len, self.coords.shape[1]))
        for traj, w in zip(trajectories, weights):
            result += w * traj[:min_len]
        
        norms = np.linalg.norm(result, axis=1, keepdims=True) + 1e-8
        return result / norms
    
    def synthesize(self, query_ids: List[int], top_k: int = 5) -> np.ndarray:
        """
        Синтезировать новую траекторию из ближайших в TrajectoryStore.
        
        Алгоритм:
        1. Найти top-K похожих траекторий
        2. Blend с весами обратно пропорционально расстоянию
        3. Применить additive noise для вариации
        """
        if self.store.total_stored < top_k:
            return None
        
        # Поиск похожих
        query_traj = self.coords[query_ids].cpu().numpy()
        similar = self.store.find_similar(query_traj, top_k=top_k)
        
        if not similar:
            return None
        
        # Веса обратно пропорциональны расстоянию
        dists = np.array([s['distance'] for s in similar])
        weights = 1.0 / (dists + 1e-8)
        weights = weights / weights.sum()
        
        # Blend
        trajs = [s['trajectory'] for s in similar]
        synthesized = self.blend(trajs, weights.tolist())
        
        # Additive noise для уникальности
        noise = np.random.randn(*synthesized.shape) * 0.01
        synthesized = synthesized + noise
        norms = np.linalg.norm(synthesized, axis=1, keepdims=True) + 1e-8
        synthesized = synthesized / norms
        
        return synthesized
