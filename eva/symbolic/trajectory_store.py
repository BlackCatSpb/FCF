"""
EVA — TrajectoryStore: база всех траекторий декодирования.

Хранит каждую траекторию (координаты + метаданные) при encode→decode.
При генерации — извлекает похожие траектории как контекст (RAG).

Knowledge = trajectories in ℝ²⁴, not weights.
"""

import torch, numpy as np, pickle, os, time
from collections import defaultdict

class TrajectoryStore:
    """
    Хранилище траекторий с быстрым поиском похожих.
    
    Каждая запись:
    - traj: [L, 24] — координаты траектории
    - ids: [L] — ID символов
    - text: str — исходный текст
    - centroid: [24] — центр масс траектории
    - length: int — длина
    """
    
    def __init__(self, max_trajectories=1000000, index_dim=8):
        self.max_trajectories = max_trajectories
        self.index_dim = index_dim  # use first N dims for indexing
        
        self.trajectories = []      # list of numpy arrays [L, 24]
        self.ids_list = []          # list of lists [L]
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
        return self
