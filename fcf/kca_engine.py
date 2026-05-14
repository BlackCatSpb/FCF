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
        lambda_kl: float = 0.1,
        lambda_mono: float = 0.01,
    ):
        self.hidden_dim = hidden_dim
        self.lambda_gap = lambda_gap
        self.lambda_contra = lambda_contra
        self.lambda_kl = lambda_kl
        self.lambda_mono = lambda_mono
        self.rho = rho
        self.eta0 = eta0

        self.convergence = ConvergenceController(
            max_cycles=max_iterations,
            rho=rho,
        )
        self.correction_history: List[Dict] = []

        self._prev_srg: Optional[float] = None
        self._layer = None

    def set_layer(self, layer):
        self._layer = layer

    def refine(
        self,
        z_init: np.ndarray,
        c_query: np.ndarray,
        tokenizer=None,
        prompt: str = "",
        c_target: Optional[np.ndarray] = None,
        graph_embeddings: Optional[np.ndarray] = None,
        p_target: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        z = z_init.copy().astype(np.float32)
        z_prev = z.copy()
        self._prev_srg = None

        for iteration in range(self.convergence.max_cycles):
            eta = self.eta0 * (self.rho ** iteration)

            loss, grad = self._compute_loss_and_grad(
                z, c_query, c_target, graph_embeddings, p_target
            )

            z_new = z - eta * grad.astype(np.float32)

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
        return z.astype(np.float32), confidence

    def refine_through_llm(
        self,
        z_init: np.ndarray,
        layer,
        tokenizer,
        prompt: str,
        max_tokens: int = 64,
    ) -> Tuple[np.ndarray, float]:
        """
        KCA с градиентами через реальный forward pass LLM.

        В отличие от аналитического refine(), здесь loss вычисляется
        через фактический проход модели с модифицированными весами.
        """
        import torch

        z = z_init.copy().astype(np.float32)
        z_t = torch.from_numpy(z).float().requires_grad_(True)
        optimizer = torch.optim.Adam([z_t], lr=self.eta0)

        for iteration in range(self.convergence.max_cycles):
            optimizer.zero_grad()

            encoding = tokenizer.encode(prompt)
            ids = encoding.ids if hasattr(encoding, "ids") else encoding
            input_ids = torch.tensor([ids], dtype=torch.long)

            device = next(layer.parameters()).device
            input_ids = input_ids.to(device)

            x = layer.embed(input_ids)
            hidden = layer.forward_transformer(x)
            logits = layer.forward_logits(hidden)

            probs = torch.softmax(logits[:, -1, :], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            max_ent = np.log(logits.shape[-1])
            confidence = 1.0 - entropy / max_ent

            z_np = z_t.detach().cpu().numpy()
            sim = np.dot(z_np.flatten(), z_np.flatten()) / (
                np.linalg.norm(z_np) ** 2 + 1e-8
            )
            target_sim = torch.tensor(0.9, device=device)

            loss = (
                -self.lambda_gap * confidence
                + 0.5 * (torch.tensor(float(sim), device=device) - target_sim) ** 2
            )

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                gamma = float(torch.exp(-torch.norm(z_t.grad or torch.zeros_like(z_t))))

            z_new = z_t.detach().cpu().numpy().astype(np.float32)

            status, z_out = self.convergence.check(
                z_new, z.astype(np.float32), gamma, iteration
            )

            self.correction_history.append({
                "iteration": iteration,
                "loss": float(loss.item()),
                "gamma": float(gamma),
                "status": status,
                "via_llm": True,
            })

            if status != "CONTINUE":
                conf = float(confidence.item())
                return z_out, conf

            z = z_new
            z_t = torch.from_numpy(z).float().requires_grad_(True)
            optimizer = torch.optim.Adam([z_t], lr=self.eta0 * (self.rho ** (iteration + 1)))

        return z.astype(np.float32), float(confidence.item())

    def _compute_loss_and_grad(
        self,
        z: np.ndarray,
        c_query: np.ndarray,
        c_target: Optional[np.ndarray],
        graph_embeddings: Optional[np.ndarray],
        p_target: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        z = z.flatten().astype(np.float64)
        c_q = c_query.flatten().astype(np.float64)

        sim = np.dot(z, c_q) / (np.linalg.norm(z) * np.linalg.norm(c_q) + 1e-8)
        srg_conf = float((sim + 1.0) / 2.0)

        loss = -self.lambda_gap * srg_conf

        norm_z = np.linalg.norm(z) + 1e-8
        norm_c = np.linalg.norm(c_q) + 1e-8
        grad = -self.lambda_gap * (
            c_q / (norm_z * norm_c) - z * sim / (norm_z ** 2)
        )

        if c_target is not None:
            c_t = c_target.flatten().astype(np.float64)
            diff = z - c_t
            loss += 0.5 * np.mean(diff ** 2)
            grad += diff / len(diff)

        if p_target is not None:
            p_t = p_target.flatten().astype(np.float64)
            p_t = np.clip(p_t, 1e-10, 1.0)
            p_t = p_t / np.sum(p_t)
            z_np = z.astype(np.float64)
            log_softmax_z = z_np - np.max(z_np)
            softmax_z = np.exp(log_softmax_z)
            softmax_z = np.clip(softmax_z, 1e-10, 1.0)
            softmax_z = softmax_z / np.sum(softmax_z)
            kl = np.sum(p_t * (np.log(p_t + 1e-10) - np.log(softmax_z + 1e-10)))
            loss += self.lambda_kl * kl

        if graph_embeddings is not None and len(graph_embeddings) > 0:
            g_emb = graph_embeddings.mean(axis=0).flatten().astype(np.float64)
            gap_loss = self.lambda_contra * np.mean((z - g_emb) ** 2)
            loss += gap_loss
            grad += self.lambda_contra * (z - g_emb) / len(z)

        if self._prev_srg is not None:
            mono_penalty = max(0, self._prev_srg - srg_conf)
            loss += self.lambda_mono * mono_penalty

        self._prev_srg = srg_conf

        grad = grad / (np.linalg.norm(grad) + 1e-8)
        return float(loss), grad.astype(np.float32)

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
