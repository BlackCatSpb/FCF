"""
AtomicBasis — SVD-разложение весов модели на атомарный базис.

Назначение:
  Каждая матрица весов W раскладывается через SVD на атомы (u_i, σ_i, v_i).
  Первые K атомов сохраняются. Любая модификация ΔW представляется
  как линейная комбинация атомов: ΔW = Σ c_i · σ_i · u_i · v_iᵀ.
  Это позволяет хранить знания не в полных матрицах (миллионы чисел),
  а в компактных коэффициентах c_i (сотни чисел).

Задача:
  - Разложить все матрицы модели через SVD
  - Адаптивно выбрать K атомов (ошибка реконструкции ≤ 10⁻³)
  - Кодировать ΔW → коэффициенты c_i
  - Декодировать c_i → ΔW через атомы
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from loguru import logger


TARGET_MATRICES = ["W_Q", "W_K", "W_V", "W_O", "gate_proj", "up_proj", "down_proj"]


@dataclass
class AtomicLayer:
    """Атомарный базис для одного слоя модели."""
    layer_name: str
    atoms: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    k_values: Dict[str, int] = field(default_factory=dict)
    reconstruction_error: Dict[str, float] = field(default_factory=dict)
    total_atoms: int = 0


class AtomicBasis:
    """
    SVD-разложение весов модели на атомарный базис.

    Использование:
        basis = AtomicBasis()
        basis.decompose(layer)                  # разложить модель
        coeffs = basis.encode(layer, "W_Q")     # ΔW → c_i
        delta = basis.decode(coeffs, "W_Q")     # c_i → ΔW
    """

    def __init__(self, error_threshold: float = 1e-3):
        self.error_threshold = error_threshold
        self.layers: Dict[str, AtomicLayer] = {}
        self._original_weights: Dict[str, Dict[str, torch.Tensor]] = {}

    def decompose(self, layer, layer_name: str = "layer_0") -> AtomicLayer:
        """
        Разложить все матрицы слоя через SVD на атомы.

        Для каждой матрицы W:
        1. W = U Σ Vᵀ — полное SVD-разложение
        2. Выбрать K: минимальное число компонент с ошибкой ≤ threshold
        3. Сохранить атомы (u_i, σ_i, v_i) для i = 1..K
        """
        logger.info(f"[SVD] Разложение слоя {layer_name} (threshold={self.error_threshold})")

        al = AtomicLayer(layer_name=layer_name)

        matrices = {
            "W_Q": layer.transformer.attention.W_Q.weight,
            "W_K": layer.transformer.attention.W_K.weight,
            "W_V": layer.transformer.attention.W_V.weight,
            "W_O": layer.transformer.attention.W_O.weight,
            "gate_proj": layer.transformer.ffn.gate_proj.weight,
            "up_proj": layer.transformer.ffn.up_proj.weight,
            "down_proj": layer.transformer.ffn.down_proj.weight,
        }

        for name, W in matrices.items():
            W_np = W.detach().cpu().numpy().astype(np.float64)

            U, S, Vt = np.linalg.svd(W_np, full_matrices=False)

            K = self._find_optimal_k(W_np, U, S, Vt)

            al.atoms[name] = {
                "U": U[:, :K].astype(np.float32),
                "S": S[:K].astype(np.float32),
                "Vt": Vt[:K, :].astype(np.float32),
            }
            al.k_values[name] = K

            W_reconstructed = (U[:, :K] * S[:K]) @ Vt[:K, :]
            error = np.linalg.norm(W_np - W_reconstructed) / (np.linalg.norm(W_np) + 1e-8)
            al.reconstruction_error[name] = float(error)
            al.total_atoms += K

            logger.info(
                f"  {name}: K={K}, error={error:.6f}, "
                f"compression={W_np.size / (K * (U.shape[0] + 1 + Vt.shape[1])):.1f}x"
            )

        self.layers[layer_name] = al

        for name in matrices:
            self._original_weights[name] = matrices[name].clone()

        logger.info(
            f"[SVD] Итого: {al.total_atoms} атомов в {len(al.atoms)} матрицах"
        )
        return al

    def _find_optimal_k(
        self,
        W: np.ndarray,
        U: np.ndarray,
        S: np.ndarray,
        Vt: np.ndarray,
    ) -> int:
        """
        Адаптивный выбор K: минимальное число компонент,
        при котором ошибка реконструкции ≤ threshold.
        Использует формулу: error² = Σ_{i=K+1}^{d} σ_i² / ∥W∥²
        что позволяет найти K за O(d) без перебора.
        """
        frob_sq = np.sum(S ** 2)
        max_k = min(256, len(S))
        if frob_sq < 1e-10:
            return min(64, max_k)

        target_error_sq = self.error_threshold ** 2

        squared_tail = 0.0
        for k in range(len(S) - 1, -1, -1):
            squared_tail += S[k] ** 2
            error_sq = squared_tail / frob_sq
            if error_sq > target_error_sq:
                candidate = min(k + 2, len(S))
                return min(candidate, max_k)

        return min(128, max_k)

    def encode(self, layer, matrix_name: str, layer_name: str = "layer_0") -> np.ndarray:
        """
        Кодировать отклонение весов ΔW = W_current - W_original в коэффициенты c_i.

        ΔW ≈ Σ c_i · σ_i · u_i · v_iᵀ

        Коэффициенты находятся через проекцию:
          c_i = u_iᵀ · ΔW · v_i / σ_i
        """
        if layer_name not in self.layers:
            raise ValueError(f"Слой {layer_name} не разложен. Вызовите decompose().")

        atoms = self.layers[layer_name].atoms[matrix_name]

        W_current = self._get_weight(layer, matrix_name)
        W_original = self._original_weights[matrix_name]

        delta_W = (W_current - W_original).detach().cpu().numpy().astype(np.float64)

        U = atoms["U"].astype(np.float64)
        S = atoms["S"].astype(np.float64)
        Vt = atoms["Vt"].astype(np.float64)
        V = Vt.T

        K = len(S)
        coeffs = np.zeros(K, dtype=np.float32)

        for i in range(K):
            coeffs[i] = float(
                U[:, i].T @ delta_W @ V[:, i] / (S[i] + 1e-10)
            )

        return coeffs

    def decode(
        self,
        coeffs: np.ndarray,
        matrix_name: str,
        layer_name: str = "layer_0",
    ) -> np.ndarray:
        """
        Декодировать коэффициенты c_i обратно в матрицу ΔW.

        ΔW = Σ c_i · σ_i · u_i · v_iᵀ
        """
        if layer_name not in self.layers:
            raise ValueError(f"Слой {layer_name} не разложен. Вызовите decompose().")

        atoms = self.layers[layer_name].atoms[matrix_name]
        U = atoms["U"]
        S = atoms["S"]
        Vt = atoms["Vt"]
        K = len(S)

        if len(coeffs) != K:
            raise ValueError(f"Ожидается {K} коэффициентов, получено {len(coeffs)}")

        delta_W = np.zeros((U.shape[0], Vt.shape[1]), dtype=np.float32)

        for i in range(K):
            delta_W += coeffs[i] * S[i] * np.outer(U[:, i], Vt[i, :])

        return delta_W

    def apply_coeffs(
        self,
        layer,
        coeffs_dict: Dict[str, np.ndarray],
        layer_name: str = "layer_0",
    ):
        """
        Применить коэффициенты к весам слоя: W_new = W_original + Σ c_i · σ_i · u_i · v_iᵀ.

        Восстанавливает оригинальные веса и добавляет модификации из коэффициентов.
        """
        for name in TARGET_MATRICES:
            if name not in coeffs_dict:
                continue
            delta = self.decode(coeffs_dict[name], name, layer_name)
            self._set_weight(layer, name, self._original_weights[name] +
                            torch.from_numpy(delta).float())

    def incremental_update(self, layer, matrix_name: str, delta_c: np.ndarray,
                           layer_name: str = "layer_0"):
        current_coeffs = self.encode(layer, matrix_name, layer_name)
        new_coeffs = current_coeffs + delta_c
        delta_W = self.decode(new_coeffs, matrix_name, layer_name)
        orig = self._original_weights[matrix_name]
        self._set_weight(layer, matrix_name,
                         orig + torch.from_numpy(delta_W).float())

    def restore_original(self, layer):
        """Восстановить оригинальные веса (до всех модификаций)."""
        for name, original in self._original_weights.items():
            self._set_weight(layer, name, original.clone())

    def get_compression_ratio(self, layer_name: str = "layer_0") -> Dict[str, float]:
        """Вычислить степень сжатия для каждой матрицы."""
        if layer_name not in self.layers:
            return {}
        ratios = {}
        al = self.layers[layer_name]
        for name in TARGET_MATRICES:
            if name in al.atoms:
                W_shape = al.atoms[name]["U"].shape[0] * al.atoms[name]["Vt"].shape[1]
                atoms_size = al.k_values[name] * (al.atoms[name]["U"].shape[0] + 1 + al.atoms[name]["Vt"].shape[1])
                ratios[name] = W_shape / atoms_size
        return ratios

    def save(self, path: str):
        """Сохранить атомарный базис."""
        import pickle, os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "error_threshold": self.error_threshold,
            "layers": {
                name: {
                    "layer_name": al.layer_name,
                    "atoms": al.atoms,
                    "k_values": al.k_values,
                    "reconstruction_error": al.reconstruction_error,
                    "total_atoms": al.total_atoms,
                }
                for name, al in self.layers.items()
            },
            "original_weights": {
                name: w.detach().cpu().numpy()
                for name, w in self._original_weights.items()
            },
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"[SVD] Базис сохранён: {path} ({len(self.layers)} слоёв)")

    @classmethod
    def load(cls, path: str) -> "AtomicBasis":
        """Загрузить атомарный базис."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        basis = cls(error_threshold=data["error_threshold"])
        for name, ld in data["layers"].items():
            al = AtomicLayer(
                layer_name=ld["layer_name"],
                atoms=ld["atoms"],
                k_values=ld["k_values"],
                reconstruction_error=ld["reconstruction_error"],
                total_atoms=ld["total_atoms"],
            )
            basis.layers[name] = al
        basis._original_weights = {
            name: torch.from_numpy(w)
            for name, w in data["original_weights"].items()
        }
        logger.info(f"[SVD] Базис загружен: {path}")
        return basis

    def _get_weight(self, layer, name: str) -> torch.Tensor:
        """Извлечь матрицу весов из слоя по имени."""
        if name in ("W_Q", "W_K", "W_V", "W_O"):
            return getattr(layer.transformer.attention, name).weight
        return getattr(layer.transformer.ffn, name).weight

    def _set_weight(self, layer, name: str, value: torch.Tensor):
        """Установить матрицу весов в слой по имени."""
        target = self._get_weight(layer, name)
        target.data = value.to(target.device).to(target.dtype)

    def summary(self, layer_name: str = "layer_0") -> str:
        """Краткая сводка по атомарному базису."""
        if layer_name not in self.layers:
            return f"AtomicBasis: слой {layer_name} не разложен"
        al = self.layers[layer_name]
        lines = [
            f"AtomicBasis({al.layer_name}):",
            f"  Атомов всего: {al.total_atoms}",
            f"  Порог ошибки: {self.error_threshold}",
        ]
        for name in TARGET_MATRICES:
            if name in al.k_values:
                lines.append(
                    f"  {name}: K={al.k_values[name]}, "
                    f"error={al.reconstruction_error[name]:.6f}"
                )
        return "\n".join(lines)
