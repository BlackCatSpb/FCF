"""
Extensions — Minimal Code Principle, Operator Training, Recursive Self-Improvement,
Forgetfulness Gate training, Adaptive KCA Scheduling, Combined Loss.

Собраны в один модуль для компактности.
"""

import random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from loguru import logger


class MinimalCodePrinciple:
    """
    При сохранении кода проверяет, можно ли достичь того же эффекта
    с меньшим числом атомов. Если да — сохраняет более компактную версию.
    """

    def __init__(self):
        self.savings: List[Dict] = []

    def minimize(self, z: np.ndarray, atomic_basis,
                 matrix_name: str = "W_Q") -> Tuple[np.ndarray, int]:
        """Пытается сократить число атомов для кодирования z."""
        if atomic_basis is None:
            return z, len(z)

        try:
            delta = atomic_basis.decode(z, matrix_name)
            tolerance = 0.01 * np.linalg.norm(delta)
            reduced = z.copy()

            for k in range(len(z) - 1, max(len(z) // 4, 1), -1):
                test = z[:k]
                test_padded = np.zeros_like(z)
                test_padded[:k] = test
                delta_test = atomic_basis.decode(test_padded, matrix_name)
                error = np.linalg.norm(delta - delta_test)
                if error < tolerance:
                    reduced = test_padded

            saving = len(z) - np.count_nonzero(reduced)
            if saving > 0:
                logger.info(f"[MinCode] Сокращено на {saving} атомов")
                self.savings.append({"original": len(z), "reduced": np.count_nonzero(reduced), "saving": saving})

            return reduced, int(np.count_nonzero(reduced))
        except Exception:
            return z, len(z)


class OperatorTrainer:
    """Обучает ВСЕ операторы State Algebra на синтетических парах."""

    def __init__(self, state_algebra):
        self.algebra = state_algebra

    def train_all(self, pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                  epochs: int = 50, lr: float = 1e-3):
        """Обучить sum, scale, subtract, cross_attend на тройках."""
        params = list(self.algebra.projector.parameters())
        params += list(self.algebra.cross_attn_block.parameters())
        params += [self.algebra.translator.weight]
        optimizer = torch.optim.Adam(params, lr=lr)

        for epoch in range(epochs):
            total_loss = 0.0
            for z_a, z_b, z_target in pairs:
                optimizer.zero_grad()

                a = torch.from_numpy(z_a).float().unsqueeze(0)
                b = torch.from_numpy(z_b).float().unsqueeze(0)
                target = torch.from_numpy(z_target).float().unsqueeze(0)

                loss_sum = F.mse_loss(self.algebra.projector(a + b), target)

                alpha = torch.rand(1).item() * 1.5 + 0.5
                loss_scale = F.mse_loss(
                    self.algebra.projector(alpha * a), target
                )

                diff = torch.clamp(a - b, -1.0, 1.0)
                loss_sub = F.mse_loss(self.algebra.projector(diff), target)

                combined = torch.cat([
                    a.unsqueeze(1), b.unsqueeze(1)
                ], dim=1)
                attn_out = self.algebra.cross_attn_block(combined)
                loss_cross = F.mse_loss(
                    self.algebra.projector(attn_out), target
                )

                loss = loss_sum + loss_scale + loss_sub + loss_cross
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                logger.debug(f"[OpTrain] epoch={epoch}, loss={total_loss:.4f}")


class RecursiveSelfImprovement:
    """
    Переоценивает старые коды текущей версией SRG.
    Для деградировавших кодов запускает KCA для обновления.
    """

    def __init__(self, srg_threshold: float = 0.5):
        self.srg_threshold = srg_threshold
        self.updated: List[str] = []

    def improve(self, codes: Dict[str, Tuple[np.ndarray, float]],
                layer, tokenizer, kca_engine) -> int:
        improved = 0
        for code_id, (z, old_score) in codes.items():
            if old_score >= self.srg_threshold:
                continue

            z_opt, new_score = kca_engine.refine(z, z)
            if new_score > old_score:
                codes[code_id] = (z_opt, new_score)
                improved += 1
                self.updated.append(code_id)
                logger.info(
                    f"[ReSelf] Код {code_id}: {old_score:.3f} → {new_score:.3f}"
                )

        return improved


class ForgetfulnessGateTrainer:
    """Обучает Forgetfulness Gate на истории удалений."""

    def __init__(self, gate: nn.Module):
        self.gate = gate
        self.history: List[Dict] = []

    def record_deletion(self, context: np.ndarray, usage: int,
                        confidence: float, age: float, was_needed: bool):
        self.history.append({
            "context": context.copy(),
            "usage": usage,
            "confidence": confidence,
            "age": age,
            "was_needed": was_needed,
        })
        if len(self.history) > 1000:
            self.history.pop(0)

    def train(self, epochs: int = 10, lr: float = 1e-4):
        if len(self.history) < 10:
            return

        optimizer = torch.optim.Adam(self.gate.parameters(), lr=lr)
        for epoch in range(epochs):
            total_loss = 0.0
            random.shuffle(self.history)
            for record in self.history[:100]:
                ctx = torch.from_numpy(record["context"]).float().unsqueeze(0)
                usage = torch.tensor([record["usage"]])
                conf = torch.tensor([record["confidence"]])
                age = torch.tensor([record["age"]])
                label = torch.tensor([record["was_needed"]])

                pred = self.gate(ctx, usage, conf, age)
                loss = F.binary_cross_entropy(pred, label.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 3 == 0:
                logger.debug(f"[Gate] epoch={epoch}, loss={total_loss:.4f}")


class AdaptiveKCAScheduler:
    """Динамическая глубина KCA по критичности запроса."""

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth

    def get_depth(self, query_confidence: float,
                  domain_confidence: float, is_critical: bool = False) -> int:
        if is_critical:
            return self.max_depth
        if query_confidence < 0.3:
            return self.max_depth
        if query_confidence < 0.6:
            return min(self.max_depth, 3)
        if domain_confidence > 0.8:
            return 1
        return 2
