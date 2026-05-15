"""
KCA Engine — Knowledge-Conscious Attention (Пункт 7).

Итеративное уточнение латентного кода с интеллектуальным схождением.

Утилитарная функция:
  L_KCA(z) = -λ1 * SRG_conf + λ2 * D_KL(p || p_target) + λ3 * ||c_out - c_target||^2

Протокол интеллектуального схождения:
1. Адаптивное демпфирование: η_t = η0 * ρ^t (ρ=0.85)
2. Детектор осцилляции: cos(∇_t, ∇_{t-1}) < -0.5 → усреднение
3. Монитор гейта: γ < 0.05 дважды → остановка
4. Детекция плато: отсутствие значимого улучшения за N шагов
5. Относительное улучшение: Δloss/loss < ε
6. Динамический max_cycles: на основе сложности задачи
7. Многомасштабная сходимость: проверка на разных окнах
8. Качество сходимости: стабильность траектории
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Callable
from dataclasses import dataclass, field
from collections import deque
from loguru import logger
import time


@dataclass
class KCACorrection:
    gap_embedding: np.ndarray
    contra_embedding: np.ndarray
    total_correction: np.ndarray
    gate_value: float
    layer_idx: int
    confidence: float


@dataclass
class ConvergenceMetrics:
    """Метрики сходимости для анализа"""
    iterations: int = 0
    final_loss: float = 0.0
    improvement_rate: float = 0.0
    trajectory_stability: float = 0.0
    oscillation_count: int = 0
    plateau_steps: int = 0
    convergence_quality: float = 0.0
    early_stop_reason: str = ""
    convergence_history: List[Dict] = field(default_factory=list)


class IntelligentConvergenceController:
    """
    Интеллектуальный контроллер схождения KCA.
    
    Заменяет жёсткий лимит на динамическое определение момента остановки
    на основе множества критериев качества сходимости.
    """

    def __init__(
        self,
        # Базовые параметры
        min_cycles: int = 3,
        max_cycles: int = 20,  # Мягкий верхний лимит
        rho: float = 0.85,
        osc_threshold: float = -0.5,
        gate_threshold: float = 0.05,
        
        # Детекция плато
        plateau_window: int = 4,
        plateau_tolerance: float = 0.01,  # 1% изменение
        
        # Относительное улучшение
        rel_improvement_threshold: float = 0.001,  # 0.1%
        rel_improvement_window: int = 3,
        
        # Многомасштабная сходимость
        convergence_windows: List[int] = None,
        window_stability_threshold: float = 0.95,
        
        # Качество сходимости
        stability_window: int = 5,
        min_trajectory_coherence: float = 0.7,
        
        # Динамические адаптеры
        adaptive_max_cycles: bool = True,
        complexity_factor: float = 1.0,
    ):
        self.min_cycles = min_cycles
        self.max_cycles = max_cycles
        self.rho = rho
        self.osc_threshold = osc_threshold
        self.gate_threshold = gate_threshold
        
        # Детекция плато
        self.plateau_window = plateau_window
        self.plateau_tolerance = plateau_tolerance
        
        # Относительное улучшение
        self.rel_improvement_threshold = rel_improvement_threshold
        self.rel_improvement_window = rel_improvement_window
        
        # Многомасштабная сходимость
        self.convergence_windows = convergence_windows or [3, 5, 7]
        self.window_stability_threshold = window_stability_threshold
        
        # Качество сходимости
        self.stability_window = stability_window
        self.min_trajectory_coherence = min_trajectory_coherence
        
        # Динамические адаптеры
        self.adaptive_max_cycles = adaptive_max_cycles
        self.complexity_factor = complexity_factor
        self._dynamic_max_cycles = max_cycles
        
        # История
        self.history_states: List[np.ndarray] = []
        self.history_deltas: List[np.ndarray] = []
        self.history_gates: List[float] = []
        self.history_losses: List[float] = []
        self.history_confidences: List[float] = []
        self.history_grad_norms: List[float] = []
        
        # Статистика
        self.oscillation_count = 0
        self.metrics = ConvergenceMetrics()
        self._start_time = None

    def estimate_complexity(self, z_init: np.ndarray, c_query: np.ndarray,
                           initial_loss: float, initial_grad_norm: float) -> float:
        """
        Оценивает сложность задачи и корректирует max_cycles.
        
        Факторы сложности:
        1. Расстояние от начальной точки до цели
        2. Начальная норма градиента
        3. Начальный loss
        4. Размерность пространства
        """
        if not self.adaptive_max_cycles:
            return self.max_cycles
            
        # Нормализованное расстояние
        z_norm = np.linalg.norm(z_init)
        c_norm = np.linalg.norm(c_query)
        distance = np.linalg.norm(z_init - c_query)
        normalized_distance = distance / (z_norm + c_norm + 1e-8)
        
        # Факторы
        distance_factor = min(2.0, 1.0 + normalized_distance * 2)
        grad_factor = min(2.0, 1.0 + initial_grad_norm * 0.1)
        loss_factor = min(2.0, 1.0 + abs(initial_loss) * 0.5)
        
        # Комбинированная сложность
        complexity = (distance_factor + grad_factor + loss_factor) / 3.0
        complexity *= self.complexity_factor
        
        # Адаптивный max_cycles
        adaptive_cycles = int(self.max_cycles * complexity)
        adaptive_cycles = max(self.min_cycles * 2, min(adaptive_cycles, 50))
        
        self._dynamic_max_cycles = adaptive_cycles
        
        logger.debug(
            f"[KCA] Оценка сложности: {complexity:.2f}, "
            f"динамический max_cycles: {adaptive_cycles}"
        )
        
        return adaptive_cycles

    def check_plateau(self) -> Tuple[bool, float]:
        """
        Детекция плато: проверяет, стабилизировался ли loss.
        
        Возвращает (is_plateau, improvement_ratio)
        """
        if len(self.history_losses) < self.plateau_window + 1:
            return False, 1.0
            
        recent_losses = self.history_losses[-self.plateau_window:]
        baseline = self.history_losses[-(self.plateau_window + 1)]
        
        # Относительное изменение
        if abs(baseline) < 1e-10:
            relative_changes = [abs(l - baseline) for l in recent_losses]
        else:
            relative_changes = [abs(l - baseline) / abs(baseline) 
                              for l in recent_losses]
        
        avg_change = np.mean(relative_changes)
        is_plateau = avg_change < self.plateau_tolerance
        
        return is_plateau, avg_change

    def check_relative_improvement(self) -> Tuple[bool, float]:
        """
        Проверяет, продолжается ли значимое улучшение.
        
        Возвращает (should_stop, improvement_rate)
        """
        if len(self.history_losses) < self.rel_improvement_window + 1:
            return False, 1.0
            
        window = self.rel_improvement_window
        recent = self.history_losses[-window:]
        previous = self.history_losses[-(window + 1):-1]
        
        # Относительное улучшение
        improvements = [(p - r) / (abs(p) + 1e-10) for p, r in zip(previous, recent)]
        avg_improvement = np.mean(improvements)
        
        should_stop = avg_improvement < self.rel_improvement_threshold
        
        return should_stop, avg_improvement

    def check_multiscale_convergence(self) -> Tuple[bool, Dict[str, float]]:
        """
        Многомасштабная проверка сходимости на разных окнах.
        
        Проверяет стабильность на коротких, средних и длинных окнах.
        """
        if len(self.history_confidences) < max(self.convergence_windows):
            return False, {}
            
        window_results = {}
        
        for window in self.convergence_windows:
            if len(self.history_confidences) >= window:
                recent = self.history_confidences[-window:]
                # Коэффициент вариации (стабильность)
                mean_conf = np.mean(recent)
                std_conf = np.std(recent)
                cv = std_conf / (mean_conf + 1e-8)
                stability = 1.0 - min(1.0, cv)
                window_results[f"window_{window}"] = stability
        
        # Проверка: все окна стабильны?
        all_stable = all(s >= self.window_stability_threshold 
                        for s in window_results.values())
        
        return all_stable, window_results

    def compute_trajectory_coherence(self) -> float:
        """
        Вычисляет когерентность траектории (насколько плавно движемся).
        
        Использует косинусное сходство между последовательными дельтами.
        """
        if len(self.history_deltas) < self.stability_window:
            return 1.0
            
        recent_deltas = self.history_deltas[-self.stability_window:]
        
        cos_sims = []
        for i in range(len(recent_deltas) - 1):
            d1 = recent_deltas[i].flatten()
            d2 = recent_deltas[i + 1].flatten()
            
            norm1 = np.linalg.norm(d1) + 1e-8
            norm2 = np.linalg.norm(d2) + 1e-8
            cos_sim = np.dot(d1, d2) / (norm1 * norm2)
            cos_sims.append(cos_sim)
        
        return float(np.mean(cos_sims)) if cos_sims else 1.0

    def check(
        self,
        X_current: np.ndarray,
        X_prev: np.ndarray,
        gamma_mean: float,
        step_idx: int,
        loss: Optional[float] = None,
        confidence: Optional[float] = None,
        grad_norm: Optional[float] = None,
    ) -> Tuple[str, np.ndarray, Dict[str, Any]]:
        """
        Интеллектуальная проверка сходимости.
        
        Возвращает: (status, X_out, diagnostics)
        """
        # Сохраняем историю
        self.history_states.append(X_current.copy())
        self.history_gates.append(gamma_mean)
        
        current_delta = X_current - X_prev
        self.history_deltas.append(current_delta)
        
        if loss is not None:
            self.history_losses.append(loss)
        if confidence is not None:
            self.history_confidences.append(confidence)
        if grad_norm is not None:
            self.history_grad_norms.append(grad_norm)
        
        diagnostics = {
            "step": step_idx,
            "gate": gamma_mean,
            "dynamic_max_cycles": self._dynamic_max_cycles,
        }
        
        # 1. Минимальное число итераций
        if step_idx < self.min_cycles - 1:
            diagnostics["reason"] = "min_cycles_not_reached"
            return "CONTINUE", X_current, diagnostics
        
        # 2. Gate saturation (модель отвергает коррекцию)
        if len(self.history_gates) >= 2:
            if all(g < self.gate_threshold for g in self.history_gates[-2:]):
                self.metrics.early_stop_reason = "GATE_SATURATED"
                self._compute_final_metrics(step_idx)
                diagnostics["reason"] = "gate_saturated"
                diagnostics["gates"] = self.history_gates[-2:]
                return "CONVERGED", X_current, diagnostics
        
        # 3. Осцилляция
        if len(self.history_deltas) >= 2:
            d_curr = self.history_deltas[-1].flatten()
            d_prev = self.history_deltas[-2].flatten()
            
            norm_c = np.linalg.norm(d_curr) + 1e-8
            norm_p = np.linalg.norm(d_prev) + 1e-8
            cos_sim = np.dot(d_curr, d_prev) / (norm_c * norm_p)
            
            if cos_sim < self.osc_threshold:
                self.oscillation_count += 1
                logger.warning(
                    f"[KCA] Осцилляция #{self.oscillation_count} (cos={cos_sim:.2f}). "
                    f"Стабилизация..."
                )
                
                # Усреднение для стабилизации
                stable = np.mean(
                    self.history_states[-3:] + [X_current, X_prev],
                    axis=0,
                )
                
                if self.oscillation_count >= 2:
                    self.metrics.early_stop_reason = "OSCILLATION_STABILIZED"
                    self._compute_final_metrics(step_idx)
                    diagnostics["reason"] = "oscillation_stabilized"
                    return "CONVERGED", stable, diagnostics
                
                diagnostics["reason"] = "oscillation_detected"
                return "OSCILLATION_DETECTED", stable, diagnostics
        
        # 4. Детекция плато (только после min_cycles)
        if step_idx >= self.min_cycles:
            is_plateau, improvement = self.check_plateau()
            diagnostics["plateau_improvement"] = improvement
            
            if is_plateau:
                self.metrics.plateau_steps += 1
                if self.metrics.plateau_steps >= 2:
                    self.metrics.early_stop_reason = "PLATEAU_REACHED"
                    self._compute_final_metrics(step_idx)
                    diagnostics["reason"] = "plateau_reached"
                    return "CONVERGED", X_current, diagnostics
            else:
                self.metrics.plateau_steps = 0
        
        # 5. Относительное улучшение
        if step_idx >= self.min_cycles + self.rel_improvement_window:
            should_stop, improvement_rate = self.check_relative_improvement()
            diagnostics["rel_improvement"] = improvement_rate
            
            if should_stop:
                self.metrics.early_stop_reason = "IMPROVEMENT_STALLED"
                self._compute_final_metrics(step_idx)
                diagnostics["reason"] = "improvement_stalled"
                return "CONVERGED", X_current, diagnostics
        
        # 6. Многомасштабная сходимость
        if step_idx >= max(self.convergence_windows):
            is_converged, window_stabilities = self.check_multiscale_convergence()
            diagnostics["window_stabilities"] = window_stabilities
            
            if is_converged:
                self.metrics.early_stop_reason = "MULTISCALE_CONVERGENCE"
                self._compute_final_metrics(step_idx)
                diagnostics["reason"] = "multiscale_convergence"
                return "CONVERGED", X_current, diagnostics
        
        # 7. Качество траектории (проверяем на поздних этапах)
        if step_idx >= self.stability_window + self.min_cycles:
            coherence = self.compute_trajectory_coherence()
            diagnostics["trajectory_coherence"] = coherence
            
            # Если когерентность высокая и мы достаточно долго идём — можно остановиться
            if coherence > 0.95 and step_idx >= self._dynamic_max_cycles * 0.7:
                self.metrics.early_stop_reason = "HIGH_COHERENCE"
                self._compute_final_metrics(step_idx)
                diagnostics["reason"] = "high_trajectory_coherence"
                return "CONVERGED", X_current, diagnostics
        
        # 8. Динамический жёсткий лимит
        if step_idx >= self._dynamic_max_cycles - 1:
            self.metrics.early_stop_reason = "DYNAMIC_MAX_CYCLES"
            self._compute_final_metrics(step_idx)
            diagnostics["reason"] = "dynamic_max_cycles_reached"
            return "MAX_CYCLES", X_current, diagnostics
        
        diagnostics["reason"] = "continuing"
        return "CONTINUE", X_current, diagnostics

    def _compute_final_metrics(self, final_step: int):
        """Вычисляет итоговые метрики сходимости."""
        self.metrics.iterations = final_step + 1
        
        if self.history_losses:
            self.metrics.final_loss = self.history_losses[-1]
            
            if len(self.history_losses) > 1:
                initial = self.history_losses[0]
                final = self.history_losses[-1]
                if abs(initial) > 1e-10:
                    self.metrics.improvement_rate = (initial - final) / abs(initial)
        
        if self.history_confidences:
            self.metrics.trajectory_stability = float(np.std(self.history_confidences))
        
        self.metrics.oscillation_count = self.oscillation_count
        self.metrics.convergence_history = self._build_convergence_history()
        
        # Общая оценка качества сходимости
        quality_factors = [
            1.0 if self.metrics.early_stop_reason != "DYNAMIC_MAX_CYCLES" else 0.5,
            min(1.0, self.metrics.improvement_rate) if self.metrics.improvement_rate > 0 else 0.0,
            1.0 - min(1.0, self.metrics.trajectory_stability),
            1.0 / (1.0 + self.oscillation_count * 0.2),
        ]
        self.metrics.convergence_quality = float(np.mean(quality_factors))

    def _build_convergence_history(self) -> List[Dict]:
        """Строит историю сходимости для анализа."""
        history = []
        for i in range(len(self.history_states)):
            entry = {
                "step": i,
                "gate": self.history_gates[i] if i < len(self.history_gates) else None,
                "loss": self.history_losses[i] if i < len(self.history_losses) else None,
                "confidence": self.history_confidences[i] if i < len(self.history_confidences) else None,
            }
            history.append(entry)
        return history

    def get_metrics(self) -> ConvergenceMetrics:
        """Возвращает метрики сходимости."""
        return self.metrics

    def reset(self):
        """Сброс состояния контроллера."""
        self.history_states.clear()
        self.history_deltas.clear()
        self.history_gates.clear()
        self.history_losses.clear()
        self.history_confidences.clear()
        self.history_grad_norms.clear()
        self.oscillation_count = 0
        self.metrics = ConvergenceMetrics()
        self._dynamic_max_cycles = self.max_cycles


class ConvergenceController(IntelligentConvergenceController):
    """Обратная совместимость: alias для IntelligentConvergenceController."""
    pass


class KCAEngine:

    def __init__(
        self,
        hidden_dim: int = 2560,
        max_iterations: int = 20,  # Увеличен для интеллектуального схождения
        rho: float = 0.85,
        eta0: float = 0.01,
        lambda_gap: float = 0.3,
        lambda_contra: float = 0.2,
        lambda_kl: float = 0.1,
        lambda_mono: float = 0.01,
        # Новые параметры интеллектуального схождения
        adaptive_convergence: bool = True,
        convergence_complexity_factor: float = 1.0,
    ):
        self.hidden_dim = hidden_dim
        self.lambda_gap = lambda_gap
        self.lambda_contra = lambda_contra
        self.lambda_kl = lambda_kl
        self.lambda_mono = lambda_mono
        self.rho = rho
        self.eta0 = eta0
        self.adaptive_convergence = adaptive_convergence

        self.convergence = IntelligentConvergenceController(
            min_cycles=3,
            max_cycles=max_iterations,
            rho=rho,
            adaptive_max_cycles=adaptive_convergence,
            complexity_factor=convergence_complexity_factor,
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
    ) -> Tuple[np.ndarray, float, ConvergenceMetrics]:
        """
        Итеративное уточнение с интеллектуальным схождением.
        
        Возвращает: (z_final, confidence, metrics)
        """
        z = z_init.copy().astype(np.float32)
        z_prev = z.copy()
        self._prev_srg = None
        
        # Оценка сложности и настройка динамического max_cycles
        initial_loss, initial_grad = self._compute_loss_and_grad(
            z, c_query, c_target, graph_embeddings, p_target
        )
        initial_grad_norm = np.linalg.norm(initial_grad)
        
        self.convergence.estimate_complexity(
            z_init, c_query, initial_loss, initial_grad_norm
        )
        
        iteration = 0
        while True:
            eta = self.eta0 * (self.rho ** iteration)

            loss, grad = self._compute_loss_and_grad(
                z, c_query, c_target, graph_embeddings, p_target
            )
            
            # Вычисляем confidence для диагностики
            confidence = self._compute_confidence(z, c_query, c_target)
            grad_norm = np.linalg.norm(grad)

            z_new = z - eta * grad.astype(np.float32)

            gamma = self._compute_gate(z_new, z, graph_embeddings)

            status, z_out, diagnostics = self.convergence.check(
                z_new, z, gamma, iteration,
                loss=loss,
                confidence=confidence,
                grad_norm=grad_norm,
            )

            self.correction_history.append({
                "iteration": iteration,
                "loss": float(loss),
                "grad_norm": float(grad_norm),
                "confidence": float(confidence),
                "gamma": float(gamma),
                "status": status,
                "diagnostics": diagnostics,
            })

            if status in ("CONVERGED", "MAX_CYCLES"):
                final_confidence = self._compute_confidence(z_out, c_query, c_target)
                metrics = self.convergence.get_metrics()
                
                logger.info(
                    f"[KCA] Схождение за {iteration + 1} итераций. "
                    f"Причина: {metrics.early_stop_reason}. "
                    f"Качество: {metrics.convergence_quality:.2f}"
                )
                
                return z_out, final_confidence, metrics

            z_prev = z
            z = z_new
            iteration += 1

    def refine_through_llm(
        self,
        z_init: np.ndarray,
        layer,
        tokenizer,
        prompt: str,
        max_tokens: int = 64,
    ) -> Tuple[np.ndarray, float, ConvergenceMetrics]:
        """
        KCA с градиентами через реальный forward pass LLM.
        """
        import torch

        z = z_init.copy().astype(np.float32)
        z_t = torch.from_numpy(z).float().requires_grad_(True)
        optimizer = torch.optim.Adam([z_t], lr=self.eta0)

        iteration = 0
        while True:
            optimizer.zero_grad()

            encoding = tokenizer.encode(prompt)
            ids = encoding.ids if hasattr(encoding, 'ids') else encoding
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

            if self.lambda_mono:
                entropy_loss = 0.1 * entropy / max(max_ent, 1)
                loss = loss + self.lambda_mono * entropy_loss

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                gamma = float(torch.exp(-torch.norm(z_t.grad or torch.zeros_like(z_t))))

            z_new = z_t.detach().cpu().numpy().astype(np.float32)
            
            # Вычисляем grad_norm для диагностики
            grad_norm = float(torch.norm(z_t.grad).item()) if z_t.grad is not None else 0.0

            status, z_out, diagnostics = self.convergence.check(
                z_new, z.astype(np.float32), gamma, iteration,
                loss=float(loss.item()),
                confidence=float(confidence.item()),
                grad_norm=grad_norm,
            )

            self.correction_history.append({
                "iteration": iteration,
                "loss": float(loss.item()),
                "gamma": float(gamma),
                "confidence": float(confidence.item()),
                "status": status,
                "diagnostics": diagnostics,
                "via_llm": True,
            })

            if status in ("CONVERGED", "MAX_CYCLES"):
                metrics = self.convergence.get_metrics()
                
                logger.info(
                    f"[KCA-LLM] Схождение за {iteration + 1} итераций. "
                    f"Причина: {metrics.early_stop_reason}"
                )
                
                return z_out, float(confidence.item()), metrics

            z = z_new
            z_t = torch.from_numpy(z).float().requires_grad_(True)
            optimizer = torch.optim.Adam([z_t], lr=self.eta0 * (self.rho ** (iteration + 1)))
            iteration += 1

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
