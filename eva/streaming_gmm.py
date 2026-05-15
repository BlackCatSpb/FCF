"""
StreamingGMM — динамические домены через потоковую Gaussian Mixture Model.

Назначение:
  Заменяет статический DomainRegistry на динамическую вероятностную модель.
  Домены рождаются, сливаются и удаляются автоматически на основе
  поступающих контекстных векторов, без предопределённого числа кластеров.

Задача:
  - Рождение домена: если новый вектор далёк от всех центроидов → новый домен
  - Слияние доменов: если два домена статистически неразличимы (KL < порог) → слить
  - Удаление домена: если все коды истекли или usage_count = 0 → удалить
  - Обновление центроида: экспоненциальное скользящее среднее по новым векторам
  - Регуляризация Тихонова для устойчивости ковариационных оценок

Формулы:
  P(c | domain_d) = N(c; μ_d, Σ_d + εI)
  KL(domain_a || domain_b) = 0.5 * [tr(Σ_b⁻¹ Σ_a) + (μ_b-μ_a)ᵀΣ_b⁻¹(μ_b-μ_a) - k + ln|Σ_b|/|Σ_a|]
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
from loguru import logger


@dataclass
class StreamingDomain:
    """Один домен в Streaming GMM."""
    domain_id: str
    level: str
    centroid: np.ndarray
    covariance: np.ndarray
    count: int = 1
    usage_count: int = 0
    confidence_history: List[float] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)

    def update(self, vector: np.ndarray, alpha: float = 0.1):
        v = vector.flatten()
        diff = v - self.centroid
        self.centroid = self.centroid + alpha * diff
        self.covariance = (1 - alpha) * self.covariance + alpha * np.outer(diff, diff)
        self.covariance += 1e-6 * np.eye(len(self.centroid))
        self.count += 1
        self.last_used_at = time.time()

    def likelihood(self, vector: np.ndarray) -> float:
        v = vector.flatten()
        diff = v - self.centroid
        d = len(v)
        cov_reg = self.covariance + 1e-6 * np.eye(d)
        try:
            inv_cov = np.linalg.inv(cov_reg)
            _, logdet = np.linalg.slogdet(cov_reg)
            mahalanobis = diff.T @ inv_cov @ diff
            return float(-0.5 * (d * np.log(2 * np.pi) + logdet + mahalanobis))
        except np.linalg.LinAlgError:
            return -np.inf

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

    def is_expired(self, ttl_seconds: float = 86400 * 7) -> bool:
        return (time.time() - self.last_used_at) > ttl_seconds and self.usage_count == 0


class StreamingGMM:
    """
    Потоковая Gaussian Mixture Model для динамического управления доменами.

    Параметры:
      dim: размерность контекстных векторов
      birth_threshold: минимальная вероятность для отнесения к существующему домену
      merge_threshold: максимальная симметричная KL-дивергенция для слияния
      alpha: скорость обновления центроида (EMA)
      min_domain_size: минимальное число векторов в домене для сохранения
    """

    def __init__(
        self,
        dim: int = 2560,
        level: str = "word",
        birth_threshold: float = -50.0,
        merge_threshold: float = 0.5,
        alpha: float = 0.1,
        min_domain_size: int = 3,
    ):
        self.dim = dim
        self.level = level
        self.birth_threshold = birth_threshold
        self.merge_threshold = merge_threshold
        self.alpha = alpha
        self.min_domain_size = min_domain_size

        self.domains: Dict[str, StreamingDomain] = {}
        self._next_id: int = 0

    def classify(self, vector: np.ndarray) -> Optional[str]:
        """Найти ближайший домен или вернуть None (новый домен)."""
        if not self.domains:
            return None

        best_id = None
        best_ll = -np.inf

        for domain_id, domain in self.domains.items():
            ll = domain.likelihood(vector)
            if ll > best_ll:
                best_ll = ll
                best_id = domain_id

        if best_ll < self.birth_threshold:
            return None

        return best_id

    def add_or_update(self, vector: np.ndarray, inherit_from: str = None) -> str:
        """Добавить вектор: обновить существующий домен или создать новый с наследованием."""
        domain_id = self.classify(vector)

        if domain_id is not None:
            self.domains[domain_id].update(vector, self.alpha)
            return domain_id

        return self._create_domain(vector, inherit_from=inherit_from)

    def _create_domain(self, vector: np.ndarray, inherit_from: str = None) -> str:
        domain_id = f"{self.level}_{self._next_id}"
        self._next_id += 1

        v = vector.flatten()

        if inherit_from and inherit_from in self.domains:
            parent = self.domains[inherit_from]
            centroid = 0.7 * v + 0.3 * parent.centroid
            covariance = parent.covariance.copy()
            count = parent.count // 2
        else:
            centroid = v.copy()
            covariance = np.eye(self.dim) * 0.1
            count = 1

        domain = StreamingDomain(
            domain_id=domain_id,
            level=self.level,
            centroid=centroid,
            covariance=covariance,
            count=count,
        )
        self.domains[domain_id] = domain

        if inherit_from:
            logger.info(
                f"[GMM] Новый домен: {domain_id} "
                f"(унаследован от {inherit_from}, level={self.level})"
            )
        else:
            logger.info(f"[GMM] Новый домен: {domain_id} (level={self.level})")

        return domain_id

    def merge_similar(self) -> int:
        """Слить пары доменов с KL-дивергенцией ниже порога."""
        if len(self.domains) < 2:
            return 0

        merged = 0
        ids = list(self.domains.keys())

        for i in range(len(ids)):
            if ids[i] not in self.domains:
                continue
            for j in range(i + 1, len(ids)):
                if ids[j] not in self.domains:
                    continue

                kl = self._symmetric_kl(
                    self.domains[ids[i]], self.domains[ids[j]]
                )
                if kl < self.merge_threshold:
                    self._merge(ids[i], ids[j])
                    merged += 1

        if merged > 0:
            logger.info(f"[GMM] Слито доменов: {merged} (level={self.level})")

        return merged

    def _merge(self, id_a: str, id_b: str):
        dom_a = self.domains[id_a]
        dom_b = self.domains[id_b]
        total = dom_a.count + dom_b.count

        dom_a.centroid = (
            dom_a.count * dom_a.centroid + dom_b.count * dom_b.centroid
        ) / total
        dom_a.covariance = (
            dom_a.count * dom_a.covariance + dom_b.count * dom_b.covariance
        ) / total + 1e-6 * np.eye(self.dim)
        dom_a.count = total
        dom_a.usage_count += dom_b.usage_count

        del self.domains[id_b]

    def remove_expired(self, ttl_seconds: float = 86400 * 7) -> int:
        """Удалить домены, в которых все коды истекли."""
        removed = 0
        for domain_id in list(self.domains.keys()):
            if self.domains[domain_id].is_expired(ttl_seconds):
                del self.domains[domain_id]
                removed += 1

        if removed > 0:
            logger.info(f"[GMM] Удалено доменов: {removed} (level={self.level})")

        return removed

    def remove_small(self) -> int:
        """Удалить домены с числом векторов ниже порога."""
        removed = 0
        for domain_id in list(self.domains.keys()):
            if self.domains[domain_id].count < self.min_domain_size:
                del self.domains[domain_id]
                removed += 1

        return removed

    def _symmetric_kl(
        self, dom_a: StreamingDomain, dom_b: StreamingDomain
    ) -> float:
        kl_ab = self._kl_divergence(dom_a, dom_b)
        kl_ba = self._kl_divergence(dom_b, dom_a)
        return 0.5 * (kl_ab + kl_ba)

    def _kl_divergence(
        self, dom_a: StreamingDomain, dom_b: StreamingDomain
    ) -> float:
        d = self.dim
        mu_a, cov_a = dom_a.centroid, dom_a.covariance + 1e-6 * np.eye(d)
        mu_b, cov_b = dom_b.centroid, dom_b.covariance + 1e-6 * np.eye(d)

        try:
            L_b = np.linalg.cholesky(cov_b)
            diff = mu_b - mu_a
            z = np.linalg.solve(L_b, diff)
            mahalanobis = float(z.T @ z)

            z_cov_a = np.linalg.solve(L_b, cov_a)
            trace_term = float(np.trace(np.linalg.solve(L_b.T, z_cov_a)))

            _, logdet_a = np.linalg.slogdet(cov_a)
            _, logdet_b = np.linalg.slogdet(cov_b)

            kl = 0.5 * (trace_term + mahalanobis - d + logdet_b - logdet_a)
            return float(max(kl, 0.0))
        except np.linalg.LinAlgError:
            return float("inf")

    def get_centroids_matrix(self) -> np.ndarray:
        if not self.domains:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack(
            [d.centroid for d in self.domains.values()]
        ).astype(np.float32)

    def get_domain_ids(self) -> List[str]:
        return list(self.domains.keys())

    def __len__(self) -> int:
        return len(self.domains)

    def __contains__(self, domain_id: str) -> bool:
        return domain_id in self.domains

    def summary(self) -> str:
        lines = [f"StreamingGMM(level={self.level}, domains={len(self.domains)}):"]
        for did, dom in self.domains.items():
            lines.append(
                f"  {did}: count={dom.count}, usage={dom.usage_count}, "
                f"conf={dom.average_confidence():.3f}"
            )
        return "\n".join(lines)


class MultiLevelGMM:
    """
    Streaming GMM для всех уровней фрактальной иерархии.

    Каждый уровень (sym, word, sent, text) имеет свой независимый StreamingGMM.
    """

    LEVELS = ["sym", "word", "sent", "text"]

    def __init__(self, dim: int = 2560):
        self.gmms: Dict[str, StreamingGMM] = {
            level: StreamingGMM(dim=dim, level=level) for level in self.LEVELS
        }

    def classify(self, vector: np.ndarray, level: str = "word") -> Optional[str]:
        return self.gmms[level].classify(vector)

    def add_or_update(self, vector: np.ndarray, level: str = "word") -> str:
        return self.gmms[level].add_or_update(vector)

    def merge_all(self) -> Dict[str, int]:
        return {level: gmm.merge_similar() for level, gmm in self.gmms.items()}

    def cleanup_all(self) -> Dict[str, int]:
        return {level: gmm.remove_expired() for level, gmm in self.gmms.items()}

    def summary(self) -> str:
        lines = ["MultiLevelGMM:"]
        for level, gmm in self.gmms.items():
            lines.append(f"  {level}: {len(gmm)} domains")
        return "\n".join(lines)
