"""
ContradictionScorer — обнаружение противоречий в координатном пространстве ℝ¹²⁸.

5 типов противоречий:
1. STRUCTURAL  — резкий поворот на границе слов
2. ORDINAL    — движение против направления предложения
3. SEMANTIC   — противоположные направления между словами
4. CONTEXTUAL — слово аномально далеко от контекста
5. FREQUENCY  — паттерн не встречался в TrajectoryStore

Выход: per-position combined contradiction score [0,1] + 5-hot type vector.
Только значимые позиции (границы слов) получают score > 0.
"""
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ContradictionLabels:
    probs: np.ndarray          # [L] — combined contradiction per position
    types: np.ndarray          # [L, 5] — one-hot contradiction type
    confidence: np.ndarray     # [L] — max type probability
    n_contradictions: int      # число позиций с probs > 0.5
    TYPE_NAMES = ['structural', 'ordinal', 'semantic', 'contextual', 'frequency']


class ContradictionScorer:
    """
    Обнаружение противоречий. Работает полностью в ℝ¹²⁸.
    Противоречия возникают ТОЛЬКО на границах слов или между словами.
    """

    def __init__(self, trajectory_store=None, coord_dim=128):
        self.store = trajectory_store
        self.coord_dim = coord_dim

        self.thresholds = {
            'structural': 0.3,    # cos шагов < 0.3
            'ordinal': 0.2,       # cos с направлением предложения < 0.2
            'semantic': -0.1,     # cos между word-level шагами < -0.1
            'contextual': 2.5,    # z-score расстояния > 2.5
            'frequency': 0.4,     # cos к ближайшему в сторе < 0.4
        }
        self.scales = {
            'structural': 0.3, 'ordinal': 0.3, 'semantic': 0.3,
            'contextual': 0.3, 'frequency': 0.3,
        }

    def score_trajectory(self, trajectory: np.ndarray,
                          word_boundaries: List[Tuple[int, int]],
                          sentence_centroid: np.ndarray,
                          word_centroids: np.ndarray,
                          token_ids: Optional[List[int]] = None) -> ContradictionLabels:
        """
        Оценить противоречия. Противоречия считаются ТОЛЬКО:
        - На границах слов (переход </W> → <W>)
        - Внутри слова если аномалия

        Args:
            trajectory: [L, D] — полная траектория
            word_boundaries: [(start,end),...] — границы слов (в char-индексах)
            sentence_centroid: [D] — центроид предложения
            word_centroids: [W, D] — центроиды слов
            token_ids: [L] — ID токенов
        """
        L = trajectory.shape[0]
        probs = np.zeros(L)
        types = np.zeros((L, 5), dtype=np.float32)
        confidence = np.zeros(L)

        W = len(word_centroids)
        if W < 2:
            return ContradictionLabels(probs, types, confidence, 0)

        # Word-level векторы
        w_steps = word_centroids[1:] - word_centroids[:-1]  # [W-1, D]
        w_norms = np.linalg.norm(w_steps, axis=-1)
        w_steps_n = w_steps / w_norms.reshape(-1, 1).clip(1e-8)

        # Направление предложения
        sent_dir = sentence_centroid - trajectory[0]
        sent_dir_n = sent_dir / (np.linalg.norm(sent_dir) + 1e-8)

        # Структура: карта слово → char-индексы
        word_chars = [(s, e) for s, e in word_boundaries]

        for w in range(W):
            # Позиции последнего символа этого слова и первого символа следующего
            w_start, w_end = word_chars[w]

            # === TYPE 0: STRUCTURAL — на границе между словами ===
            if w < W - 1 and w + 1 < len(w_steps_n):
                nxt_start, _ = word_chars[w + 1]
                # Косинус между направлением от этого слова к следующему
                # и направлением от предыдущего к этому
                if w >= 1:
                    cos_step = float(w_steps_n[w-1] @ w_steps_n[w])
                    if cos_step < self.thresholds['structural']:
                        raw = self.thresholds['structural'] - cos_step
                        val = self._sigmoid(raw, self.scales['structural'])
                        if val > 0.5:
                            # Ставим на последнем символе текущего слова
                            probs[w_end] = val
                            types[w_end, 0] = 1.0
                            confidence[w_end] = val

            # === TYPE 2: SEMANTIC — противоположные направления слов ===
            if w < W - 1 and w < len(w_steps_n):
                next_start, _ = word_chars[w + 1]
                if w + 1 < len(w_steps_n):
                    cos_between = float(w_steps_n[w] @ w_steps_n[w+1]) \
                        if w + 1 < len(w_steps_n) else 1.0
                else:
                    # Если это последний word-level шаг — нормально
                    cos_between = 1.0

                if cos_between < self.thresholds['semantic']:
                    raw = self.thresholds['semantic'] - cos_between
                    val = self._sigmoid(raw, self.scales['semantic'])
                    if val > 0.5:
                        pos = min(w_end, L - 1)
                        if val > probs[pos]:
                            probs[pos] = val
                            types[pos, :] = 0
                            types[pos, 2] = 1.0
                            confidence[pos] = val

            # === TYPE 1: ORDINAL — против направления предложения ===
            if w < len(w_steps_n):
                cos_sent = float(w_steps_n[w] @ sent_dir_n)
                if cos_sent < self.thresholds['ordinal']:
                    raw = self.thresholds['ordinal'] - cos_sent
                    val = self._sigmoid(raw, self.scales['ordinal'])
                    if val > 0.5:
                        pos = min(w_end, L - 1)
                        if val > probs[pos]:
                            probs[pos] = val
                            types[pos, :] = 0
                            types[pos, 1] = 1.0
                            confidence[pos] = val

            # === TYPE 3: CONTEXTUAL — слово далеко от контекста ===
            wc = word_centroids[w]
            dist = np.linalg.norm(wc - sentence_centroid)
            all_dists = [np.linalg.norm(w2 - sentence_centroid) for w2 in word_centroids]
            if len(all_dists) > 1:
                mean_d = np.mean(all_dists)
                std_d = np.std(all_dists) + 1e-8
                z = (dist - mean_d) / std_d
                if z > self.thresholds['contextual']:
                    raw = z - self.thresholds['contextual']
                    val = self._sigmoid(raw, self.scales['contextual'])
                    if val > 0.5:
                        pos = min(w_end, L - 1)
                        if val > probs[pos]:
                            probs[pos] = val
                            types[pos, :] = 0
                            types[pos, 3] = 1.0
                            confidence[pos] = val

            # === TYPE 4: FREQUENCY — центроид не в сторе ===
            if self.store is not None and self.store.total_stored > 0:
                wc_n = wc / np.linalg.norm(wc)
                best_cos = 0.0
                for h in self.store.hierarchical:
                    if len(h.word_centroids) > 0:
                        for hwc in h.word_centroids:
                            hwc_n = hwc / np.linalg.norm(hwc)
                            cos = float(wc_n @ hwc_n)
                            if cos > best_cos:
                                best_cos = cos
                if best_cos < self.thresholds['frequency']:
                    raw = self.thresholds['frequency'] - best_cos
                    val = self._sigmoid(raw, self.scales['frequency'])
                    if val > 0.5:
                        pos = min(w_end, L - 1)
                        if val > probs[pos]:
                            probs[pos] = val
                            types[pos, :] = 0
                            types[pos, 4] = 1.0
                            confidence[pos] = val

        n_contra = int((probs > 0.5).sum())
        return ContradictionLabels(probs=probs, types=types, confidence=confidence,
                                    n_contradictions=n_contra)

    def _sigmoid(self, x: float, scale: float = 0.3) -> float:
        return 1.0 / (1.0 + np.exp(-x / scale))
