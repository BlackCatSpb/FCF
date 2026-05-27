"""
EVA — TrajectoryStore: база всех траекторий декодирования.

Хранит каждую траекторию (координаты + метаданные) при encode→decode.
При генерации — извлекает похожие траектории как контекст (RAG).

Knowledge = trajectories in ℝ²⁴, not weights.
"""

import torch, numpy as np, pickle, os, time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class HierarchicalTrajectory:
    """Multi-level trajectory — agent's proposed structure."""
    symbol_trajectory: np.ndarray       # [L, D] — raw coords
    word_boundaries: List[Tuple[int,int]]  # [(start,end),...]
    word_centroids: np.ndarray          # [num_words, D]
    word_weights: np.ndarray            # [num_words]
    connection_coords: np.ndarray       # [num_words-1, D]
    sentence_centroid: np.ndarray       # [D]
    text: str
    ids: List[int]
    length: int = 0
    
    def __post_init__(self):
        self.length = len(self.ids)


class TrajectoryStore:
    """
    Хранилище траекторий: flat + hierarchical.
    """
    
    def __init__(self, max_trajectories=1000000):
        self.max_trajectories = max_trajectories
        
        self.trajectories = []      # [L, D] flat
        self.ids_list = []          # [L]
        self.texts = []             # str
        self.centroids = []         # [D]
        self.first_coords = []      # [D]
        self.lengths = []           # int
        self.total_stored = 0
        
        # Hierarchical storage
        self.hierarchical = []      # List[HierarchicalTrajectory]
    
    def store_hierarchical(self, htraj: HierarchicalTrajectory):
        """Store multi-level trajectory (agent's proposed method)."""
        if self.total_stored >= self.max_trajectories:
            cutoff = self.max_trajectories // 10
            self.trajectories = self.trajectories[cutoff:]
            self.ids_list = self.ids_list[cutoff:]
            self.texts = self.texts[cutoff:]
            self.centroids = self.centroids[cutoff:]
            self.first_coords = self.first_coords[cutoff:]
            self.lengths = self.lengths[cutoff:]
            self.hierarchical = self.hierarchical[cutoff:]
            self.total_stored = len(self.trajectories)
        
        self.hierarchical.append(htraj)
        # Also store flat for backward compat
        self.trajectories.append(htraj.symbol_trajectory)
        self.ids_list.append(htraj.ids)
        self.texts.append(htraj.text)
        self.centroids.append(htraj.sentence_centroid)
        self.first_coords.append(htraj.symbol_trajectory[0] if len(htraj.symbol_trajectory) > 0 else np.zeros(htraj.sentence_centroid.shape))
        self.lengths.append(htraj.length)
        self.total_stored += 1
    
    def find_similar_hierarchical(self, query_htraj, level_weights=None, top_k=10):
        """Multi-level similarity search (agent's proposed method)."""
        if level_weights is None:
            level_weights = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
        
        if not self.hierarchical:
            return []
        
        scores = []
        for i, h in enumerate(self.hierarchical):
            s = 0.0
            
            # Level 1: Symbol trajectory similarity
            q_sym = query_htraj.symbol_trajectory
            h_sym = h.symbol_trajectory
            min_len = min(len(q_sym), len(h_sym))
            if min_len > 0:
                sym_dist = np.linalg.norm(q_sym[:min_len] - h_sym[:min_len], axis=1).mean()
                s += level_weights.get(1, 0) * (1.0 / (1.0 + sym_dist))
            
            # Level 2: Word centroid similarity
            if len(query_htraj.word_centroids) > 0 and len(h.word_centroids) > 0:
                q_wc = query_htraj.word_centroids.mean(axis=0)
                h_wc = h.word_centroids.mean(axis=0)
                w_sim = np.dot(q_wc, h_wc) / (np.linalg.norm(q_wc) * np.linalg.norm(h_wc) + 1e-8)
                s += level_weights.get(2, 0) * max(0, w_sim)
            
            # Level 3: Connection pattern similarity
            if len(query_htraj.connection_coords) > 0 and len(h.connection_coords) > 0:
                q_cc = query_htraj.connection_coords.mean(axis=0)
                h_cc = h.connection_coords.mean(axis=0)
                c_sim = np.dot(q_cc, h_cc) / (np.linalg.norm(q_cc) * np.linalg.norm(h_cc) + 1e-8)
                s += level_weights.get(3, 0) * max(0, c_sim)
            
            # Level 4: Sentence centroid distance
            sc_dist = np.linalg.norm(query_htraj.sentence_centroid - h.sentence_centroid)
            s += level_weights.get(4, 0) * (1.0 / (1.0 + sc_dist))
            
            scores.append((s, i))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        return [self.hierarchical[i] for _, i in scores[:top_k]]
        self.texts = []             # source texts
        self.centroids = []         # [N, 24]
        self.first_coords = []      # [N, 24] — first point (start index)
        self.lengths = []           # [N]
        
        self.total_stored = 0
        
    def store(self, text, ids, trajectory):
        """
        Сохранить траекторию декодирования.
        
        Args:
            text: исходный текст
            ids: список ID символов
            trajectory: numpy массив [L, 24] координат
        """
        if self.total_stored >= self.max_trajectories:
            # Remove oldest 10%
            cutoff = self.max_trajectories // 10
            self.trajectories = self.trajectories[cutoff:]
            self.ids_list = self.ids_list[cutoff:]
            self.texts = self.texts[cutoff:]
            self.centroids = self.centroids[cutoff:]
            self.first_coords = self.first_coords[cutoff:]
            self.lengths = self.lengths[cutoff:]
            self.total_stored = len(self.trajectories)
        
        self.trajectories.append(trajectory.astype(np.float32))
        self.ids_list.append(ids)
        self.texts.append(text)
        self.centroids.append(trajectory.mean(axis=0).astype(np.float32))
        self.first_coords.append(trajectory[0].astype(np.float32))
        self.lengths.append(len(ids))
        self.total_stored += 1
    
    def find_similar(self, query_traj, top_k=10, max_len_diff=5):
        """
        Найти траектории, похожие на query.
        
        Uses centroid distance + first-point similarity for fast filtering.
        """
        if self.total_stored == 0:
            return []
        
        query_centroid = query_traj.mean(axis=0)
        query_first = query_traj[0]
        query_len = len(query_traj)
        
        scores = []
        for i in range(self.total_stored):
            # Fast filter: length similarity
            len_diff = abs(self.lengths[i] - query_len)
            if len_diff > max_len_diff:
                continue
            
            # Centroid distance (fast)
            c_dist = np.linalg.norm(self.centroids[i] - query_centroid)
            
            # First-point similarity
            f_sim = np.dot(self.first_coords[i], query_first) / (
                np.linalg.norm(self.first_coords[i]) * np.linalg.norm(query_first) + 1e-8
            )
            
            # Combined score
            score = -c_dist + f_sim * 2.0 - len_diff * 0.1
            scores.append((score, i))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, idx in scores[:top_k]:
            results.append({
                'text': self.texts[idx],
                'ids': self.ids_list[idx],
                'trajectory': self.trajectories[idx],
                'distance': np.linalg.norm(self.centroids[idx] - query_centroid),
            })
        
        return results
    
    def find_by_prefix(self, prefix_ids, top_k=10):
        """Найти траектории, начинающиеся с prefix_ids."""
        if self.total_stored == 0:
            return []
        
        prefix_len = len(prefix_ids)
        scores = []
        
        for i in range(self.total_stored):
            if self.lengths[i] <= prefix_len:
                continue
            
            # Check if first N IDs match
            match = 0
            for j in range(min(prefix_len, len(self.ids_list[i]))):
                if self.ids_list[i][j] == prefix_ids[j]:
                    match += 1
                else:
                    break
            
            if match >= prefix_len - 1:  # at most 1 mismatch
                # Score by trajectory similarity
                traj_prefix = self.trajectories[i][:prefix_len]
                # Simple: use centroid distance
                dist = np.linalg.norm(self.centroids[i])
                scores.append((-dist, i))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        return [{
            'text': self.texts[idx],
            'ids': self.ids_list[idx],
            'trajectory': self.trajectories[idx],
        } for _, idx in scores[:top_k]]
    
    def get_context_for_generation(self, current_ids, trajectory, top_k=5):
        """
        Получить контекст для генерации: похожие траектории.
        Используется при RAG (retrieval-augmented generation).
        """
        similar = self.find_similar(trajectory, top_k=top_k)
        prefix_matches = self.find_by_prefix(current_ids, top_k=3)
        
        # Merge and deduplicate
        seen = set()
        context = []
        for item in similar + prefix_matches:
            if item['text'] not in seen:
                seen.add(item['text'])
                context.append(item)
                if len(context) >= top_k:
                    break
        
        return context
    
    def stats(self):
        if self.total_stored == 0:
            return "TrajectoryStore: empty"
        avg_len = np.mean(self.lengths)
        return (f"TrajectoryStore: {self.total_stored:,} trajectories, "
                f"avg_len={avg_len:.1f}")
    
    def save(self, path):
        data = {
            'trajectories': self.trajectories,
            'ids_list': self.ids_list,
            'texts': self.texts,
            'centroids': self.centroids,
            'first_coords': self.first_coords,
            'lengths': self.lengths,
            'total_stored': self.total_stored,
            'hierarchical': self.hierarchical,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.trajectories = data['trajectories']
        self.ids_list = data['ids_list']
        self.texts = data['texts']
        self.centroids = data['centroids']
        self.first_coords = data['first_coords']
        self.lengths = data['lengths']
        self.total_stored = data['total_stored']
        self.hierarchical = data.get('hierarchical', [])
        return self
