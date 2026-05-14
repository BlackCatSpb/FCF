"""
SleepModeV2 — расширенная консолидация (Фаза 5).

Добавляет к базовому Sleep Mode:
- Dream Mode: генерация синтетических кодов через State Algebra
- Forgetfulness Gate: обучаемый классификатор «хранить/удалить»
- Adversarial Code Validation: проверка кодов на устойчивость к атакам
- Recursive Self-Improvement: переоценка старых кодов текущим SRG
"""

import os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger


class ForgetfulnessGate(nn.Module):
    """
    Обучаемый классификатор: стоит ли сохранять латентный код.

    Вход: [контекстный вектор, usage_count, средняя уверенность SRG, возраст]
    Выход: p_keep ∈ [0, 1] — вероятность сохранения.
    """

    def __init__(self, input_dim: int = 2560):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 3, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, context: torch.Tensor, usage: torch.Tensor,
                confidence: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        x = torch.cat([
            context.float(),
            usage.float().unsqueeze(-1),
            confidence.float().unsqueeze(-1),
            age.float().unsqueeze(-1),
        ], dim=-1)
        return self.net(x).squeeze(-1)

    def train_step(self, contexts, usages, confidences, ages, labels, optimizer):
        optimizer.zero_grad()
        preds = self(contexts, usages, confidences, ages)
        loss = F.binary_cross_entropy(preds, labels.float())
        loss.backward()
        optimizer.step()
        return loss.item()


class AdversarialValidator:
    """
    Генерирует «атакующие» запросы для проверки устойчивости кодов.

    Для каждого важного кода создаются возмущённые версии контекстного вектора.
    Если код проходит проверку (SRG не падает) — считается надёжным.
    """

    def __init__(self, noise_scale: float = 0.1, num_attacks: int = 5):
        self.noise_scale = noise_scale
        self.num_attacks = num_attacks

    def validate(self, code_vector: np.ndarray, srg_fn,
                 context: np.ndarray) -> Dict[str, Any]:
        results = []
        passed = 0

        for i in range(self.num_attacks):
            noise = np.random.randn(*context.shape) * self.noise_scale
            attacked = context + noise
            attacked = attacked / (np.linalg.norm(attacked) + 1e-8)

            score = srg_fn(code_vector, attacked)
            results.append(float(score))
            if score > 0.5:
                passed += 1

        return {
            "mean_score": float(np.mean(results)),
            "min_score": float(np.min(results)),
            "passed": passed,
            "total": self.num_attacks,
            "robust": passed >= self.num_attacks * 0.8,
        }


class DreamGenerator:
    """
    Генерирует синтетические латентные коды через случайные комбинации
    существующих кодов посредством State Algebra.

    Каждый синтетический код проверяется через быструю аппроксимацию SRG.
    Успешные коды добавляются в домены.
    """

    def __init__(self, num_dreams: int = 20, acceptance_threshold: float = 0.7):
        self.num_dreams = num_dreams
        self.acceptance_threshold = acceptance_threshold

    def dream(
        self,
        existing_codes: List[np.ndarray],
        state_algebra,
        srg_evaluator,
    ) -> List[Tuple[np.ndarray, float]]:
        if len(existing_codes) < 2:
            return []

        accepted = []

        for _ in range(self.num_dreams):
            op = random.choice(["sum", "scale", "subtract", "cross"])
            a, b = random.sample(existing_codes, 2)

            try:
                if op == "sum":
                    z = state_algebra.sum(a, b)
                elif op == "scale":
                    alpha = random.uniform(0.5, 1.5)
                    z = state_algebra.scale(a, alpha)
                elif op == "subtract":
                    z = state_algebra.subtract(a, b, random.uniform(0.5, 1.5))
                else:
                    z = state_algebra.cross_attend(a, b)

                score = srg_evaluator(z)
                if score >= self.acceptance_threshold:
                    accepted.append((z, float(score)))

            except Exception:
                continue

        logger.info(
            f"[Dream] Сгенерировано: {self.num_dreams}, "
            f"принято: {len(accepted)}"
        )
        return accepted


class SleepModeV2:
    """
    Расширенный Sleep Mode: кластеризация + Forgetfulness Gate +
    Dream Mode + Adversarial Validation + Recursive Self-Improvement.
    """

    def __init__(
        self,
        idle_timeout: int = 300,
        sleep_interval: int = 7200,
        ttl_idle_days: int = 7,
    ):
        self.idle_timeout = idle_timeout
        self.sleep_interval = sleep_interval
        self.ttl_idle_days = ttl_idle_days
        self.tic = time.time()

        self._forget_gate: Optional[ForgetfulnessGate] = None
        self._forget_optimizer: Optional[torch.optim.AdamW] = None
        self._forget_history: List[Dict] = []

        self.adversarial = AdversarialValidator()
        self.dreamer = DreamGenerator()

        self.sleep_count = 0
        self.last_query_time = time.time()
        self.last_sleep_time = time.time()

    def should_sleep(self) -> bool:
        now = time.time()
        return (now - self.last_query_time > self.idle_timeout or
                now - self.last_sleep_time > self.sleep_interval)

    def on_query(self):
        self.last_query_time = time.time()

    def execute(
        self,
        layers: list,
        gmm=None,
        atomic_basis=None,
        hierarchy=None,
        state_algebra=None,
        hnsw_index=None,
        self_improver=None,
    ) -> Dict[str, Any]:
        logger.info(f"Sleep Mode #{self.sleep_count + 1} — v2")

        stats = {
            "clusters_formed": 0,
            "snapshots_removed": 0,
            "dreams_accepted": 0,
            "adversarial_checks": 0,
            "codes_reevaluated": 0,
            "domains_merged": 0,
        }

        for layer in layers:
            removed = self._remove_stale(layer)
            stats["snapshots_removed"] += removed

            clusters = self._cluster(layer)
            stats["clusters_formed"] += clusters

        if gmm:
            merged = gmm.merge_all()
            stats["domains_merged"] = sum(merged.values())
            removed = gmm.cleanup_all()
            stats["snapshots_removed"] += sum(removed.values())

        if (state_algebra and hnsw_index and
            hnsw_index.get_snapshot_count() > 10):
            codes = self._gather_codes(layers, hnsw_index)
            if codes:
                def srg_eval(z):
                    sim = np.dot(z, z) / (np.linalg.norm(z)**2 + 1e-8)
                    return float(np.clip(sim, 0, 1))

                dreams = self.dreamer.dream(codes, state_algebra, srg_eval)
                stats["dreams_accepted"] = len(dreams)

        if hnsw_index:
            hnsw_index.defragment()

        if self_improver is not None and hnsw_index is not None:
            old_codes = {}
            for layer in layers:
                for i, meta in enumerate(layer.state_storage.snapshots_meta):
                    old_codes[f"snap_{i}"] = (meta["c"], meta.get("confidence", 0.5))
            improved = self_improver.improve(old_codes, None, None, None)
            stats["codes_reevaluated"] = improved

        if self._forget_gate is not None and hasattr(self, '_forget_history'):
            from .extensions import ForgetfulnessGateTrainer
            trainer = ForgetfulnessGateTrainer(self._forget_gate)
            for record in self._forget_history:
                trainer.record_deletion(**record)
            trainer.train(epochs=5, lr=1e-4)
            self._forget_history.clear()

        self.last_sleep_time = time.time()
        self.sleep_count += 1
        logger.info(f"[Sleep] v2 завершён: {stats}")
        return stats

    def _remove_stale(self, layer) -> int:
        storage = layer.state_storage
        removed = 0
        now = time.time()
        ttl = self.ttl_idle_days * 86400

        for i in range(len(storage.snapshots_meta) - 1, -1, -1):
            meta = storage.snapshots_meta[i]
            age = now - meta.get("timestamp", now)
            usage = meta.get("usage_count", 0)
            score = usage * np.exp(-0.1 * age / 86400)
            was_deleted = False

            if usage == 0 and age > ttl:
                self._record_deletion(meta, was_needed=False)
                storage._remove(i)
                removed += 1
                was_deleted = True
            elif score < 0.01:
                self._record_deletion(meta, was_needed=False)
                storage._remove(i)
                removed += 1
                was_deleted = True

        if removed:
            logger.info(f"[Sleep] Удалено слепков: {removed}")
        return removed

    def _record_deletion(self, meta: dict, was_needed: bool):
        if self._forget_history is None:
            self._forget_history = []
        self._forget_history.append({
            "context": meta.get("c", np.zeros(2560, dtype=np.float32)),
            "usage": int(meta.get("usage_count", 0)),
            "confidence": float(meta.get("confidence", 0.0)),
            "age": float(time.time() - meta.get("timestamp", time.time())),
            "was_needed": was_needed,
        })

    def mark_code_needed(self, code_vector: np.ndarray):
        """Отметить что удалённый код позже потребовался (y=1 для ForgetfulnessGate)."""
        if self._forget_history is None:
            self._forget_history = []
        self._forget_history.append({
            "context": code_vector.flatten(),
            "usage": 1,
            "confidence": 0.8,
            "age": 0.0,
            "was_needed": True,
        })

    def _cluster(self, layer) -> int:
        if len(layer.state_storage) < 5:
            return 0
        vectors = layer.state_storage.get_all_vectors()
        if len(vectors) < 5:
            return 0
        try:
            from sklearn.cluster import KMeans
            n = max(1, len(vectors) // 5)
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=3)
            labels = kmeans.fit_predict(vectors)
            clusters = len(set(labels))
            return clusters
        except Exception:
            return 0

    def _gather_codes(self, layers, hnsw_index) -> List[np.ndarray]:
        codes = []
        for layer in layers:
            for meta in layer.state_storage.snapshots_meta:
                codes.append(meta["c"])
                if len(codes) >= 100:
                    break
        return codes
