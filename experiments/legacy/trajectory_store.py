"""
EVA — TrajectoryStore: база всех траекторий декодирования.
Хранит каждую траекторию в сжатом формате (TRACK_DTYPE, ~16 байт/шаг).
При генерации — извлекает похожие траектории как контекст (RAG).
Knowledge = trajectories in 16 bytes/step, not 1536.
"""

import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, pickle, os, time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional

from coordinate_packer import CoordinatePacker, TRACK_DTYPE


@dataclass
class HierarchicalTrajectory:
    """Multi-level trajectory — agent's proposed structure."""
    compact_track: np.ndarray         # [L] TRACK_DTYPE — сжатый трек
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

    def decompress(self) -> np.ndarray:
        """Развернуть сжатый трек в [L, 384]."""
        cp = CoordinatePacker()
        return cp.decompress_track(self.compact_track)


class ConsolidationTransformer(nn.Module):
    """
    Learnable Consolidation: Conv1d + element-wise gate — взвешивание шагов траектории.
    ~6.5K params.
    """
    
    def __init__(self, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=8)
        self.gate = nn.Parameter(torch.zeros(d_model))
        self.importance_head = nn.Linear(d_model, 1)
    
    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        L, D = traj.shape
        conv_in = traj.unsqueeze(0).transpose(1, 2)
        conv_out = self.conv(conv_in).transpose(1, 2)
        gate = torch.sigmoid(self.gate)
        combined = gate * conv_out + (1 - gate) * traj.unsqueeze(0)
        weights = self.importance_head(combined).squeeze(-1)
        weights = F.softmax(weights, dim=-1)
        return weights.squeeze(0)


class TrajectoryStore:
    """
    Хранилище траекторий: сжатый компакт-формат (TRACK_DTYPE).
    """
    
    def __init__(self, max_trajectories=1000000):
        self.max_trajectories = max_trajectories
        self.packer = CoordinatePacker()
        
        self.compact_tracks = []    # List[TRACK_DTYPE] — сжатые треки
        self.ids_list = []          # [L]
        self.texts = []             # str
        self.centroids = []         # [D] — для быстрого поиска
        self.first_coords = []      # [D]
        self.lengths = []           # int
        self.total_stored = 0
        
        # Hierarchical storage
        self.hierarchical = []      # List[HierarchicalTrajectory]
        
        # Learnable Consolidation
        self.consolidation_net = None

    def _centroid_from_compact(self, compact: np.ndarray) -> np.ndarray:
        """Вычислить центроид [384] из сжатого трека."""
        return self.packer.decompress_track(compact).mean(axis=0)

    def _first_from_compact(self, compact: np.ndarray) -> np.ndarray:
        """Первый шаг [384] из сжатого трека."""
        return self.packer.decompress_track(compact[:1])[0]
    
    def _lazy_init_consolidation(self, device='cpu'):
        if self.consolidation_net is None:
            self.consolidation_net = ConsolidationTransformer(d_model=128).to(device)
    
    def consolidate(self, htraj: HierarchicalTrajectory, device='cpu') -> HierarchicalTrajectory:
        if len(htraj.compact_track) < 3:
            return htraj
        self._lazy_init_consolidation(device)
        traj = htraj.decompress()
        traj_t = torch.tensor(traj, dtype=torch.float32, device=device)
        with torch.no_grad():
            weights = self.consolidation_net.forward(traj_t).cpu().numpy()
        weighted_centroid = (traj * weights[:, np.newaxis]).sum(axis=0)
        return HierarchicalTrajectory(
            compact_track=htraj.compact_track,
            word_boundaries=htraj.word_boundaries,
            word_centroids=htraj.word_centroids,
            word_weights=htraj.word_weights,
            connection_coords=htraj.connection_coords,
            sentence_centroid=weighted_centroid,
            text=htraj.text,
            ids=htraj.ids,
        )
    
    def store_hierarchical(self, htraj: HierarchicalTrajectory):
        """Store multi-level trajectory (agent's proposed method)."""
        if self.total_stored >= self.max_trajectories:
            cutoff = self.max_trajectories // 10
            self.compact_tracks = self.compact_tracks[cutoff:]
            self.ids_list = self.ids_list[cutoff:]
            self.texts = self.texts[cutoff:]
            self.centroids = self.centroids[cutoff:]
            self.first_coords = self.first_coords[cutoff:]
            self.lengths = self.lengths[cutoff:]
            self.hierarchical = self.hierarchical[cutoff:]
            self.total_stored = len(self.compact_tracks)
        
        self.hierarchical.append(htraj)
        self.compact_tracks.append(htraj.compact_track)
        self.ids_list.append(htraj.ids)
        self.texts.append(htraj.text)
        self.centroids.append(htraj.sentence_centroid)
        first = htraj.decompress()[0] if len(htraj.compact_track) > 0 else np.zeros(384)
        self.first_coords.append(first)
        self.lengths.append(htraj.length)
        self.total_stored += 1
    
    def find_similar_hierarchical(self, query_htraj, level_weights=None, top_k=10):
        if level_weights is None:
            level_weights = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
        if not self.hierarchical:
            return []
        q_traj = query_htraj.decompress()
        scores = []
        for i, h in enumerate(self.hierarchical):
            s = 0.0
            h_traj = h.decompress()
            min_len = min(len(q_traj), len(h_traj))
            if min_len > 0:
                sym_dist = np.linalg.norm(q_traj[:min_len] - h_traj[:min_len], axis=1).mean()
                s += level_weights.get(1, 0) * (1.0 / (1.0 + sym_dist))
            if len(query_htraj.word_centroids) > 0 and len(h.word_centroids) > 0:
                q_wc = query_htraj.word_centroids.mean(axis=0)
                h_wc = h.word_centroids.mean(axis=0)
                w_sim = np.dot(q_wc, h_wc) / (np.linalg.norm(q_wc) * np.linalg.norm(h_wc) + 1e-8)
                s += level_weights.get(2, 0) * max(0, w_sim)
            if len(query_htraj.connection_coords) > 0 and len(h.connection_coords) > 0:
                q_cc = query_htraj.connection_coords.mean(axis=0)
                h_cc = h.connection_coords.mean(axis=0)
                c_sim = np.dot(q_cc, h_cc) / (np.linalg.norm(q_cc) * np.linalg.norm(h_cc) + 1e-8)
                s += level_weights.get(3, 0) * max(0, c_sim)
            sc_dist = np.linalg.norm(query_htraj.sentence_centroid - h.sentence_centroid)
            s += level_weights.get(4, 0) * (1.0 / (1.0 + sc_dist))
            scores.append((s, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [self.hierarchical[i] for _, i in scores[:top_k]]

    def store(self, text, ids, trajectory):
        """Сохранить траекторию (автоматически сжимает [L, 384] → TRACK_DTYPE)."""
        compact = self.packer.compress_track(trajectory)
        self.store_compact(text, ids, compact)

    def store_compact(self, text, ids, compact, text_id=0):
        """Сохранить сжатый трек (7- или 10-польный). Авто-конвертация в TRACK_DTYPE."""
        if self.total_stored >= self.max_trajectories:
            cutoff = self.max_trajectories // 10
            self.compact_tracks = self.compact_tracks[cutoff:]
            self.ids_list = self.ids_list[cutoff:]
            self.texts = self.texts[cutoff:]
            self.centroids = self.centroids[cutoff:]
            self.first_coords = self.first_coords[cutoff:]
            self.lengths = self.lengths[cutoff:]
            self.total_stored = len(self.compact_tracks)
        
        if compact.dtype.names is not None and compact.dtype.names != TRACK_DTYPE.names:
            compact = self.packer.convert_generation_compact(compact, text_id)
        elif compact.dtype != TRACK_DTYPE:
            compact = compact.astype(TRACK_DTYPE)
        
        self.compact_tracks.append(compact)
        self.ids_list.append(ids)
        self.texts.append(text)
        self.centroids.append(self._centroid_from_compact(compact))
        self.first_coords.append(self._first_from_compact(compact))
        self.lengths.append(len(ids))
        self.total_stored += 1
    
    def get_trajectory(self, idx: int) -> np.ndarray:
        """Развернуть сжатый трек в [L, 384]."""
        return self.packer.decompress_track(self.compact_tracks[idx])

    def find_similar(self, query_traj, top_k=10, max_len_diff=5):
        if self.total_stored == 0:
            return []
        query_centroid = query_traj.mean(axis=0)
        query_first = query_traj[0]
        query_len = len(query_traj)
        scores = []
        for i in range(self.total_stored):
            len_diff = abs(self.lengths[i] - query_len)
            if len_diff > max_len_diff:
                continue
            c_dist = np.linalg.norm(self.centroids[i] - query_centroid)
            f_sim = np.dot(self.first_coords[i], query_first) / (
                np.linalg.norm(self.first_coords[i]) * np.linalg.norm(query_first) + 1e-8)
            score = -c_dist + f_sim * 2.0 - len_diff * 0.1
            scores.append((score, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, idx in scores[:top_k]:
            results.append({
                'text': self.texts[idx],
                'ids': self.ids_list[idx],
                'compact': self.compact_tracks[idx],
                'trajectory': self.get_trajectory(idx),
                'distance': np.linalg.norm(self.centroids[idx] - query_centroid),
            })
        return results
    
    def find_by_prefix(self, prefix_ids, top_k=10):
        if self.total_stored == 0:
            return []
        prefix_len = len(prefix_ids)
        scores = []
        for i in range(self.total_stored):
            if self.lengths[i] <= prefix_len:
                continue
            match = 0
            for j in range(min(prefix_len, len(self.ids_list[i]))):
                if self.ids_list[i][j] == prefix_ids[j]:
                    match += 1
                else:
                    break
            if match >= prefix_len - 1:
                dist = np.linalg.norm(self.centroids[i])
                scores.append((-dist, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [{
            'text': self.texts[idx],
            'ids': self.ids_list[idx],
            'compact': self.compact_tracks[idx],
            'trajectory': self.get_trajectory(idx),
        } for _, idx in scores[:top_k]]
    
    def get_context_for_generation(self, current_ids, trajectory, top_k=5):
        similar = self.find_similar(trajectory, top_k=top_k)
        prefix_matches = self.find_by_prefix(current_ids, top_k=3)
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
            'compact_tracks': self.compact_tracks,
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
        self.compact_tracks = data.get('compact_tracks',
            [self.packer.compress_track(t) for t in data.get('trajectories', [])])
        self.ids_list = data['ids_list']
        self.texts = data['texts']
        self.centroids = data['centroids']
        self.first_coords = data['first_coords']
        self.lengths = data['lengths']
        self.total_stored = data['total_stored']
        self.hierarchical = data.get('hierarchical', [])
        return self
