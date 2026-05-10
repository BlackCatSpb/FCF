"""
KCA Engine — Knowledge-Conscious Attention (Пункт 7).

Итеративное уточнение латентного кода с гарантированной сходимостью.
Заменяет простую рефлексию формальным итеративным процессом.

Утилитарная функция:
  L_KCA(z) = -λ1 * SRG_conf + λ2 * D_KL(p || p_target) + λ3 * ||c_out - c_target||^2

Протокол сходимости:
1. Адаптивное демпфирование: η_t = η0 * ρ^t (ρ=0.85)
2. Детектор осцилляции: cos(∇_t, ∇_{t-1}) < -0.5 → усреднение
3. Монитор гейта: γ < 0.05 дважды → остановка
4. Жёсткий лимит: не более 5 итераций
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from loguru import logger


@dataclass
class KCACorrection:
    gap_embedding: np.ndarray
    contra_embedding: np.ndarray
    total_correction: np.ndarray
    gate_value: float
    layer_idx: int
    confidence: float


class ConvergenceController:

    def __init__(
        self,
        max_cycles: int = 5,
        rho: float = 0.85,
        osc_threshold: float = -0.5,
        gate_threshold: float = 0.05,
    ):
        self.max_cycles = max_cycles
        self.rho = rho
        self.osc_threshold = osc_threshold
        self.gate_threshold = gate_threshold

        self.history_states: List[np.ndarray] = []
        self.history_deltas: List[np.ndarray] = []
        self.history_gates: List[float] = []

    def check(
        self,
        X_current: np.ndarray,
        X_prev: np.ndarray,
        gamma_mean: float,
        step_idx: int,
    ) -> Tuple[str, np.ndarray]:
        self.history_states.append(X_current.copy())
        self.history_gates.append(gamma_mean)

        current_delta = X_current - X_prev
        self.history_deltas.append(current_delta)

        if len(self.history_gates) >= 2:
            if all(g < self.gate_threshold for g in self.history_gates[-2:]):
                return "SATURATED", X_current

        if len(self.history_deltas) >= 2:
            d_curr = self.history_deltas[-1].flatten()
            d_prev = self.history_deltas[-2].flatten()

            norm_c = np.linalg.norm(d_curr) + 1e-8
            norm_p = np.linalg.norm(d_prev) + 1e-8
            cos_sim = np.dot(d_curr, d_prev) / (norm_c * norm_p)

            if cos_sim < self.osc_threshold:
                logger.warning(
                    f"[KCA] Осцилляция (cos={cos_sim:.2f}). Стабилизация..."
                )
                stable = np.mean(
                    self.history_states[-3:] + [X_current, X_prev],
                    axis=0,
                )
                return "OSCILLATION_DETECTED", stable

        if step_idx >= self.max_cycles - 1:
            return "MAX_CYCLES", X_current

        return "CONTINUE", X_current

    def reset(self):
        self.history_states.clear()
        self.history_deltas.clear()
        self.history_gates.clear()


class KCAEngine:

    def __init__(
        self,
        hidden_dim: int = 2560,
        max_iterations: int = 5,
        rho: float = 0.85,
        eta0: float = 0.01,
        lambda_gap: float = 0.3,
        lambda_contra: float = 0.2,
    ):
        self.hidden_dim = hidden_dim
        self.lambda_gap = lambda_gap
        self.lambda_contra = lambda_contra
        self.rho = rho
        self.eta0 = eta0

        self.convergence = ConvergenceController(
            max_cycles=max_iterations,
            rho=rho,
        )
        self.correction_history: List[Dict] = []

    def refine(
        self,
        z_init: np.ndarray,
        c_query: np.ndarray,
        c_target: Optional[np.ndarray] = None,
        graph_embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        z = z_init.copy()
        z_prev = z.copy()

        for iteration in range(self.convergence.max_cycles):
            eta = self.eta0 * (self.rho ** iteration)

            loss, grad = self._compute_loss_and_grad(
                z, c_query, c_target, graph_embeddings
            )

            z_new = z - eta * grad

            gamma = self._compute_gate(z_new, z, graph_embeddings)

            status, z_out = self.convergence.check(z_new, z, gamma, iteration)

            self.correction_history.append({
                "iteration": iteration,
                "loss": float(loss),
                "grad_norm": float(np.linalg.norm(grad)),
                "gamma": float(gamma),
                "status": status,
            })

            if status != "CONTINUE":
                confidence = self._compute_confidence(z_out, c_query, c_target)
                return z_out, confidence

            z_prev = z
            z = z_new

        confidence = self._compute_confidence(z, c_query, c_target)
        return z, confidence

    def _compute_loss_and_grad(
        self,
        z: np.ndarray,
        c_query: np.ndarray,
        c_target: Optional[np.ndarray],
        graph_embeddings: Optional[np.ndarray],
    ) -> Tuple[float, np.ndarray]:
        z = z.flatten()
        c_q = c_query.flatten()

        sim = np.dot(z, c_q) / (np.linalg.norm(z) * np.linalg.norm(c_q) + 1e-8)
        loss = 1.0 - sim

        norm_z = np.linalg.norm(z) + 1e-8
        norm_c = np.linalg.norm(c_q) + 1e-8
        grad = -(c_q / (norm_z * norm_c) - z * sim / (norm_z ** 2))
        grad = grad / (np.linalg.norm(grad) + 1e-8)

        if c_target is not None:
            c_t = c_target.flatten()
            diff = z - c_t
            loss += 0.5 * np.mean(diff ** 2)
            grad += 0.5 * diff

        if graph_embeddings is not None and len(graph_embeddings) > 0:
            g_emb = graph_embeddings.mean(axis=0).flatten()
            if len(g_emb) == len(z):
                gap_loss = self.lambda_gap * np.mean((z - g_emb) ** 2)
                loss += gap_loss
                grad += self.lambda_gap * (z - g_emb)

        return loss, grad

    def _compute_gate(
        self,
        z_new: np.ndarray,
        z_old: np.ndarray,
        graph_embeddings: Optional[np.ndarray],
    ) -> float:
        diff = np.linalg.norm(z_new - z_old)
        gate = float(np.exp(-diff) if diff < 10 else 0.0)

        if graph_embeddings is not None and len(graph_embeddings) > 0:
            g_emb = graph_embeddings.mean(axis=0).flatten()
            if len(g_emb) == len(z_new.flatten()):
                align = np.dot(z_new.flatten(), g_emb) / (
                    np.linalg.norm(z_new) * np.linalg.norm(g_emb) + 1e-8
                )
                gate = float(gate * max(0.0, align))

        return gate

    def _compute_confidence(
        self,
        z: np.ndarray,
        c_query: np.ndarray,
        c_target: Optional[np.ndarray],
    ) -> float:
        z_f = z.flatten()
        c_q = c_query.flatten()
        sim = np.dot(z_f, c_q) / (
            np.linalg.norm(z_f) * np.linalg.norm(c_q) + 1e-8
        )
        conf = float((sim + 1.0) / 2.0)

        if c_target is not None:
            c_t = c_target.flatten()
            target_sim = np.dot(z_f, c_t) / (
                np.linalg.norm(z_f) * np.linalg.norm(c_t) + 1e-8
            )
            conf = 0.7 * conf + 0.3 * float((target_sim + 1.0) / 2.0)

        return max(0.0, min(1.0, conf))

    def reset(self):
        self.convergence.reset()
        self.correction_history.clear()
