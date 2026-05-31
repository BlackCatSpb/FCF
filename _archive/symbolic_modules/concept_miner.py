"""
ConceptScorer — поиск концептов в координатном пространстве ℝ¹²⁸.

Концепт = слово, чей центроид образует устойчивый семантический узел:
1. Часто встречается в успешных траекториях (StoreFrequency)
2. Играет роль хаба в структуре связей (ConnectionCentrality)  
3. Образует плотный кластер с контекстом (ClusterTightness)

Выход: per-word score [0,1] — training target для ConceptHead (Transformer 2).
"""
import numpy as np
import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ConceptLabels:
    """Per-word concept analysis for one trajectory."""
    scores: np.ndarray           # [num_words] — concept probability per word
    types: List[str]             # per-word: 'frequent', 'hub', 'cluster', 'mixed', 'none'
    word_texts: List[str]        # исходный текст каждого слова
    avg_score: float             # средний concept score по всей траектории
    n_concepts: int              # число слов со score > 0.6


class ConceptScorer:
    """
    Оценка концептуальной важности слов в траектории.
    Работает полностью в ℝ¹²⁸, без трансформера.
    """

    def __init__(self, trajectory_store=None, coord_dim=128):
        self.store = trajectory_store
        self.coord_dim = coord_dim

        # Веса для комбинированной оценки
        self.w_freq = 0.35
        self.w_centrality = 0.35
        self.w_tightness = 0.30

        # Пороги
        self.concept_threshold = 0.6
        self.store_match_cos = 0.85  # мин. косинус для "того же слова" в сторе

    def score_trajectory(self, word_centroids: np.ndarray,
                         connection_coords: np.ndarray,
                         sentence_centroid: np.ndarray,
                         word_texts: Optional[List[str]] = None) -> ConceptLabels:
        """
        Оценить концептуальность каждого слова в траектории.

        Args:
            word_centroids: [num_words, D] — центроиды слов
            connection_coords: [num_words-1, D] — векторы связей между словами
            sentence_centroid: [D] — центроид всего предложения
            word_texts: список строк (опционально)

        Returns:
            ConceptLabels с per-word scores
        """
        N = word_centroids.shape[0]
        scores = np.zeros(N)

        # 1. Store Frequency Score
        freq_scores = self._store_frequency(word_centroids) if self.store else np.ones(N) * 0.3

        # 2. Connection Centrality Score
        centr_scores = self._connection_centrality(word_centroids, connection_coords)

        # 3. Cluster Tightness Score
        tight_scores = self._cluster_tightness(word_centroids, sentence_centroid)

        # Комбинируем
        scores = (self.w_freq * freq_scores +
                  self.w_centrality * centr_scores +
                  self.w_tightness * tight_scores)

        scores = np.clip(scores, 0.0, 1.0)

        # Типология
        types = []
        for i in range(N):
            t = []
            if freq_scores[i] > 0.5: t.append('frequent')
            if centr_scores[i] > 0.5: t.append('hub')
            if tight_scores[i] > 0.5: t.append('cluster')
            if len(t) >= 2: types.append('mixed')
            elif len(t) == 1: types.append(t[0])
            elif scores[i] > 0.3: types.append('weak')
            else: types.append('none')

        if word_texts is None:
            word_texts = [f'w{i}' for i in range(N)]

        return ConceptLabels(
            scores=scores,
            types=types,
            word_texts=word_texts,
            avg_score=float(scores.mean()),
            n_concepts=int((scores > self.concept_threshold).sum()),
        )

    def _store_frequency(self, word_centroids: np.ndarray) -> np.ndarray:
        """
        Частотность: сколько раз центроид этого слова встречается в TrajectoryStore.
        """
        if self.store is None or self.store.total_stored == 0:
            return np.ones(word_centroids.shape[0]) * 0.3

        N = word_centroids.shape[0]
        scores = np.zeros(N)

        # Нормализуем
        wc_norm = word_centroids / np.linalg.norm(word_centroids, axis=-1, keepdims=True).clip(1e-8)

        # Собираем все сохранённые центроиды из стора
        all_centroids = []
        for h in self.store.hierarchical:
            if len(h.word_centroids) > 0:
                for wc in h.word_centroids:
                    all_centroids.append(wc)
        if not all_centroids:
            return np.ones(N) * 0.3

        all_centroids = np.array(all_centroids)
        ac_norm = all_centroids / np.linalg.norm(all_centroids, axis=-1, keepdims=True).clip(1e-8)

        # Для каждого слова: сколько центроидов в сторе имеют cos > threshold
        cos_mat = wc_norm @ ac_norm.T  # [N, store_size]
        for i in range(N):
            matches = (cos_mat[i] > self.store_match_cos).sum()
            scores[i] = min(matches / 10.0, 1.0)  # насыщение на 10+ совпадениях

        return scores

    def _connection_centrality(self, word_centroids: np.ndarray,
                                connection_coords: np.ndarray) -> np.ndarray:
        """
        Центральность слова в структуре связей между словами.
        """
        N = word_centroids.shape[0]
        if N < 2:
            return np.ones(N) * 0.3

        # Для каждого слова: норма суммы входящего и исходящего векторов связи
        conn_norms = np.linalg.norm(connection_coords, axis=-1)  # [N-1]
        scores = np.zeros(N)
        for i in range(N):
            incoming = conn_norms[i-1] if i > 0 else 0.0
            outgoing = conn_norms[i] if i < N-1 else 0.0
            scores[i] = (incoming + outgoing) / 2.0

        # Нормализуем по максимуму
        max_s = scores.max()
        if max_s > 0:
            scores = scores / max_s

        return scores

    def _cluster_tightness(self, word_centroids: np.ndarray,
                            sentence_centroid: np.ndarray) -> np.ndarray:
        """
        Плотность кластера: насколько слово близко к центроиду предложения
        относительно других слов.
        """
        N = word_centroids.shape[0]
        if N == 0:
            return np.array([])

        # Расстояния от каждого слова до центроида предложения
        dists = np.linalg.norm(word_centroids - sentence_centroid, axis=-1)
        mean_dist = dists.mean()
        std_dist = dists.std() + 1e-8

        # Чем ближе к центру, тем выше score
        # Используем z-score: слова ближе 1σ получают высокий score
        z_scores = np.abs(dists - mean_dist) / std_dist
        scores = np.exp(-z_scores)  # exp(-z): 1.0 при z=0, 0.37 при z=1, 0.14 при z=2

        return scores


def concept_label_from_text(ids: List[int], word_centroids: np.ndarray,
                             connection_coords: np.ndarray,
                             sentence_centroid: np.ndarray,
                             cv, scorer: ConceptScorer,
                             word_boundaries: List[Tuple[int, int]]) -> ConceptLabels:
    """
    Создать ConceptLabels из сырого текста + траектории.
    Извлекает текст каждого слова из boundary-токенов.
    """
    # Извлекаем текст слов
    word_texts = []
    for start, end in word_boundaries:
        chars = []
        for pos in range(start, end):
            if pos < len(ids):
                ch = cv.idx_to_char(ids[pos])
                if ch not in ('<W>', '</W>', '<S>', '</S>', '<PAD>', '<UNK>', '<BOS>', '<EOS>'):
                    chars.append(ch)
        word_texts.append(''.join(chars))

    return scorer.score_trajectory(
        word_centroids=word_centroids,
        connection_coords=connection_coords,
        sentence_centroid=sentence_centroid,
        word_texts=word_texts or None,
    )
