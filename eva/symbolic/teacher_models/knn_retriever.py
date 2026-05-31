"""
kNN-LM Retriever — retrieval из TrajectoryStore для bias генерации.

На каждом шаге генерации:
1. Берём текущий h (скрытое состояние EVA, ℝ¹²⁸)
2. Ищем K ближайших траекторных точек в TrajectoryStore
3. Для каждой найденной точки смотрим, какой токен был на этой позиции
4. Строим bias: bias[t] = sum(exp(-dist/τ) for retrieved tokens == t)
5. Добавляем к логитам: logits += bias * w_knn

Это knowledge source для MetaWeighter.
"""
import torch, torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict


class KNNRetriever:
    """
    kNN-LM retrieval из TrajectoryStore.

    На вход: h_current [128] — скрытое состояние EVA
    На выход: bias [V] — поToken добавка к логитам

    Хранит матрицу [N_trajectory_points, 128 + 1] где:
    - первые 128: координаты точки (h)
    - последняя 1: token_id на этой позиции
    """
    def __init__(self, k: int = 8, temperature: float = 0.5, knn_weight: float = 0.3):
        self.k = k
        self.temperature = temperature
        self.knn_weight = knn_weight
        self.key_matrix = None   # [N, 128]
        self.value_ids = None    # [N]
        self._built = False

    def build_from_trajectory_store(self, store, model=None, device='cpu'):
        """
        Строит key-value матрицу из TrajectoryStore.

        Для каждой траектории в store:
        - достаём h (скрытое состояние)
        - если h нет, вычисляем через model.forward
        - сохраняем h → token_id

        Args:
            store: TrajectoryStore с stored_trajectories
            model: EVA модель (нужна если h не сохранены)
            device: torch.device
        """
        keys, values = [], []

        trajectories = getattr(store, 'stored_trajectories', [])
        if not trajectories and hasattr(store, 'trajectories'):
            trajectories = store.trajectories

        for traj in trajectories:
            tokens = traj.get('tokens', traj.get('token_ids', []))
            h_states = traj.get('hidden_states', None)

            if h_states is not None:
                # h_states: [L, 128] уже есть в траектории
                if isinstance(h_states, np.ndarray):
                    h_states = torch.from_numpy(h_states)
                for i, h_i in enumerate(h_states):
                    if i < len(tokens):
                        keys.append(h_i)
                        values.append(tokens[i])
            elif model is not None and tokens:
                # Вычисляем h через модель
                inp = torch.tensor([tokens], dtype=torch.long, device=device)
                with torch.no_grad():
                    h_out = model.forward(inp, return_heads=False)
                    if isinstance(h_out, tuple):
                        h_out = h_out[0]
                h_i = h_out[0]
                for i in range(min(len(tokens), h_i.shape[0])):
                    keys.append(h_i[i].cpu())
                    values.append(tokens[i])

        if not keys:
            print('[KNN] No trajectories in store')
            return

        self.key_matrix = torch.stack(keys).float()  # [N, 128]
        self.value_ids = torch.tensor(values, dtype=torch.long)  # [N]
        self._built = True
        print(f'[KNN] Built index: {len(self.value_ids)} points')

    def build_from_tensors(self, h_matrix: torch.Tensor, token_ids: torch.Tensor):
        """
        Прямая сборка из тензоров.

        Args:
            h_matrix: [N, 128] — скрытые состояния
            token_ids: [N] — соответствующие токены
        """
        self.key_matrix = h_matrix.float()
        self.value_ids = token_ids.long()
        self._built = True
        print(f'[KNN] Built index: {len(self.value_ids)} points')

    def retrieve(self, h_current: torch.Tensor, vocab_size: int = 161) -> torch.Tensor:
        """
        kNN retrieval + bias.

        Args:
            h_current: [128] — текущее скрытое состояние
            vocab_size: размер словаря

        Returns:
            bias: [vocab_size] — bias для логитов
        """
        if not self._built or len(self.value_ids) == 0:
            return torch.zeros(vocab_size, device=h_current.device)

        # Normalize for cosine search
        h_norm = F.normalize(h_current.unsqueeze(0), dim=-1)  # [1, 128]
        keys_norm = F.normalize(self.key_matrix.to(h_current.device), dim=-1)  # [N, 128]

        # Cosine similarity
        sim = h_norm @ keys_norm.T  # [1, N]
        weights, indices = sim.topk(min(self.k, sim.shape[-1]), dim=-1)

        # Weighted vote for each token
        bias = torch.zeros(vocab_size, device=h_current.device)
        exp_w = torch.exp(weights[0] / self.temperature)
        for i, idx in enumerate(indices[0]):
            token = self.value_ids[idx].item()
            if token < vocab_size:
                bias[token] += exp_w[i].item()

        # Normalize
        if bias.sum() > 0:
            bias = bias / bias.sum()

        return bias * self.knn_weight

    def retrieve_batch(self, h_batch: torch.Tensor,
                        vocab_size: int = 161) -> torch.Tensor:
        """
        Batch retrieval.

        Args:
            h_batch: [B, 128]
        Returns:
            bias: [B, vocab_size]
        """
        if not self._built:
            return torch.zeros(h_batch.shape[0], vocab_size, device=h_batch.device)

        B = h_batch.shape[0]
        h_norm = F.normalize(h_batch, dim=-1)
        keys_norm = F.normalize(self.key_matrix.to(h_batch.device), dim=-1)

        sim = h_norm @ keys_norm.T  # [B, N]
        weights, indices = sim.topk(min(self.k, sim.shape[-1]), dim=-1)

        bias = torch.zeros(B, vocab_size, device=h_batch.device)
        exp_w = torch.exp(weights / self.temperature)
        for b in range(B):
            for i in range(indices.shape[1]):
                token = self.value_ids[indices[b, i]].item()
                if token < vocab_size:
                    bias[b, token] += exp_w[b, i].item()

        bias = bias / (bias.sum(dim=-1, keepdim=True) + 1e-8)
        return bias * self.knn_weight
