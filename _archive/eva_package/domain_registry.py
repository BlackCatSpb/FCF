"""
DomainRegistry — реестр доменных правил.

Хранит LoRA-адаптеры, сгруппированные по доменам.
Обеспечивает:
- Быстрый поиск домена по косинусному сходству с центроидом
- Применение адаптера к слою на время обработки запроса
- Сохранение/загрузку всех доменных правил
- Обновление центроидов при добавлении новых слепков
"""

import os
import time
import pickle
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class DomainRule:
    domain_id: str
    adapter_path: str
    context_centroid: np.ndarray
    usage_count: int = 1
    confidence_history: List[float] = field(default_factory=list)
    domain_data_path: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def record_confidence(self, confidence: float):
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > 100:
            self.confidence_history.pop(0)
        self.usage_count += 1

    def average_confidence(self, window: int = 20) -> float:
        if not self.confidence_history:
            return 0.0
        recent = self.confidence_history[-window:]
        return sum(recent) / len(recent)

    def update_centroid(
        self, new_vector: np.ndarray, weight: float = 0.1
    ):
        new_vector = new_vector.flatten()
        self.context_centroid = (
            (1.0 - weight) * self.context_centroid + weight * new_vector
        )
        norm = np.linalg.norm(self.context_centroid) + 1e-8
        self.context_centroid = self.context_centroid / norm


class DomainRegistry:

    def __init__(
        self,
        storage_dir: str = None,
        threshold: float = 0.7,
    ):
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(__file__), "..", "domain_rules"
        )
        os.makedirs(self.storage_dir, exist_ok=True)

        self.threshold = threshold
        self.rules: Dict[str, DomainRule] = {}
        self._centroids_cache: Optional[np.ndarray] = None
        self._domain_ids: List[str] = []

    def add(
        self,
        domain_id: str,
        context_centroid: np.ndarray,
        adapter_path: str = "",
        domain_data_path: Optional[str] = None,
    ) -> DomainRule:
        centroid = context_centroid.flatten()
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        if domain_id in self.rules:
            self.rules[domain_id].record_confidence(1.0)
            self.rules[domain_id].update_centroid(centroid)
            return self.rules[domain_id]

        if not adapter_path:
            adapter_path = os.path.join(
                self.storage_dir, f"{domain_id}.lora"
            )

        rule = DomainRule(
            domain_id=domain_id,
            adapter_path=adapter_path,
            context_centroid=centroid,
            domain_data_path=domain_data_path,
        )
        self.rules[domain_id] = rule
        self._rebuild_cache()
        logger.info(f"[Domain] Добавлен домен: {domain_id}")
        return rule

    def find_best(
        self, c_query: np.ndarray, threshold: float = None
    ) -> Optional[str]:
        if not self.rules:
            return None

        threshold = threshold or self.threshold
        c_norm = c_query.flatten()
        c_norm = c_norm / (np.linalg.norm(c_norm) + 1e-8)

        best_id = None
        best_sim = -1.0

        for domain_id, rule in self.rules.items():
            sim = np.dot(c_norm, rule.context_centroid)
            if sim > best_sim:
                best_sim = sim
                best_id = domain_id

        if best_sim >= threshold and best_id is not None:
            self.rules[best_id].usage_count += 1
            return best_id

        return None

    def get_centroids_matrix(self) -> np.ndarray:
        if self._centroids_cache is None:
            self._rebuild_cache()
        return self._centroids_cache if self._centroids_cache is not None else np.array([])

    def _rebuild_cache(self):
        self._domain_ids = list(self.rules.keys())
        if not self._domain_ids:
            self._centroids_cache = None
            return

        d = len(self.rules[self._domain_ids[0]].context_centroid)
        self._centroids_cache = np.zeros((len(self._domain_ids), d), dtype=np.float32)
        for i, domain_id in enumerate(self._domain_ids):
            self._centroids_cache[i] = self.rules[domain_id].context_centroid

    def get_rule(self, domain_id: str) -> Optional[DomainRule]:
        return self.rules.get(domain_id)

    def remove(self, domain_id: str):
        if domain_id in self.rules:
            del self.rules[domain_id]
            self._rebuild_cache()

            adapter_path = os.path.join(self.storage_dir, f"{domain_id}.lora")
            if os.path.exists(adapter_path):
                os.remove(adapter_path)

            logger.info(f"[Domain] Удалён домен: {domain_id}")

    def get_usage_stats(self) -> Dict[str, int]:
        return {
            domain_id: rule.usage_count
            for domain_id, rule in self.rules.items()
        }

    def get_stable_domains(
        self, min_usage: int = 10, min_confidence: float = 0.7
    ) -> List[str]:
        stable = []
        for domain_id, rule in self.rules.items():
            if (
                rule.usage_count >= min_usage
                and rule.average_confidence() >= min_confidence
            ):
                stable.append(domain_id)
        return stable

    def save(self, path: str = None):
        path = path or os.path.join(self.storage_dir, "registry.pkl")
        data = {
            "rules": {
                did: {
                    "domain_id": r.domain_id,
                    "adapter_path": r.adapter_path,
                    "context_centroid": r.context_centroid,
                    "usage_count": r.usage_count,
                    "confidence_history": r.confidence_history,
                    "domain_data_path": r.domain_data_path,
                    "timestamp": r.timestamp,
                }
                for did, r in self.rules.items()
            }
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"[Domain] Реестр сохранён: {path} ({len(self.rules)} доменов)")

    @classmethod
    def load(cls, path: str = None, storage_dir: str = None) -> "DomainRegistry":
        registry = cls(storage_dir=storage_dir)

        if path and os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)

            for did, rdata in data.get("rules", {}).items():
                rule = DomainRule(
                    domain_id=rdata["domain_id"],
                    adapter_path=rdata["adapter_path"],
                    context_centroid=rdata["context_centroid"],
                    usage_count=rdata.get("usage_count", 1),
                    confidence_history=rdata.get("confidence_history", []),
                    domain_data_path=rdata.get("domain_data_path"),
                    timestamp=rdata.get("timestamp", time.time()),
                )
                registry.rules[did] = rule

            registry._rebuild_cache()
            logger.info(f"[Domain] Реестр загружен: {len(registry.rules)} доменов")

        return registry

    def __len__(self) -> int:
        return len(self.rules)

    def __contains__(self, domain_id: str) -> bool:
        return domain_id in self.rules

    def summary(self) -> str:
        lines = [f"DomainRegistry({len(self.rules)} domains):"]
        for did, rule in self.rules.items():
            lines.append(
                f"  {did}: usage={rule.usage_count}, "
                f"conf={rule.average_confidence():.3f}"
            )
        return "\n".join(lines)
