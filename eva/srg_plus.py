"""
SRGPlus — расширенная самооценка (Фаза 6).

Добавляет к базовому SRG:
- Anomaly Detection: выявление кодов с аномальными SRG-оценками
- Meta-SRG: отслеживание тренда уверенности за N запросов
- Code Uncertainty: дисперсия SRG для оценки надёжности кода
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class SRGStats:
    """Статистика SRG для одного кода или домена."""
    mean: float = 0.0
    variance: float = 0.0
    count: int = 0
    scores: deque = field(default_factory=lambda: deque(maxlen=100))

    def update(self, score: float):
        self.scores.append(score)
        self.count = len(self.scores)
        if self.count > 0:
            arr = np.array(list(self.scores))
            self.mean = float(np.mean(arr))
            self.variance = float(np.var(arr)) if self.count > 1 else 0.0


class SRGAnomalyDetector:
    """
    Выявляет латентные коды, дающие статистически выбросные SRG-оценки.

    Примеры аномалий:
    - SRG > 0.9 для заведомо неверного ответа
    - SRG < 0.3 для очевидно правильного ответа
    """

    def __init__(self, window: int = 50, z_threshold: float = 2.5):
        self.window = window
        self.z_threshold = z_threshold
        self.global_scores: deque = deque(maxlen=window * 10)
        self.code_stats: Dict[str, SRGStats] = {}
        self.anomalies: List[Dict] = []

    def record(self, code_id: str, score: float):
        self.global_scores.append(score)
        if code_id not in self.code_stats:
            self.code_stats[code_id] = SRGStats()
        self.code_stats[code_id].update(score)

    def check(self, code_id: str, score: float) -> bool:
        """True = аномалия обнаружена."""
        if len(self.global_scores) < self.window:
            return False

        global_arr = np.array(list(self.global_scores))
        global_mean = float(np.mean(global_arr))
        global_std = float(np.std(global_arr)) + 1e-8

        z_score = abs(score - global_mean) / global_std

        if z_score > self.z_threshold:
            self.anomalies.append({
                "code_id": code_id,
                "score": score,
                "z_score": z_score,
                "global_mean": global_mean,
            })
            logger.warning(
                f"[SRG] Аномалия: code={code_id}, score={score:.3f}, "
                f"z={z_score:.1f} (mean={global_mean:.3f})"
            )
            return True

        return False

    def get_suspicious_codes(self) -> List[str]:
        return list(set(a["code_id"] for a in self.anomalies[-50:]))


class MetaSRG:
    """
    Оценивает не отдельный ответ, а тренд уверенности за последние N запросов.

    Если средняя уверенность падает → диагностика или запрос внешней помощи.
    """

    def __init__(self, window: int = 50, decline_threshold: float = 0.05):
        self.window = window
        self.decline_threshold = decline_threshold
        self.scores: deque = deque(maxlen=window)
        self.trend_history: List[Dict] = []

    def record(self, score: float):
        self.scores.append(score)

    def get_trend(self) -> Dict[str, float]:
        if len(self.scores) < self.window:
            return {"mean": 0.0, "trend": 0.0, "declining": False}

        arr = list(self.scores)
        half = self.window // 2
        first = arr[:half]
        second = arr[half:]

        mean_first = sum(first) / len(first)
        mean_second = sum(second) / len(second)
        trend = mean_second - mean_first
        declining = trend < -self.decline_threshold

        self.trend_history.append({
            "mean_first": mean_first, "mean_second": mean_second,
            "trend": trend, "declining": declining,
        })

        if declining:
            logger.warning(
                f"[MetaSRG] Падение уверенности: {mean_first:.3f} → "
                f"{mean_second:.3f} (Δ={trend:.3f})"
            )

        return {
            "mean": (mean_first + mean_second) / 2,
            "trend": trend,
            "declining": declining,
        }

    def should_diagnose(self) -> bool:
        if len(self.trend_history) < 3:
            return False
        return all(t["declining"] for t in self.trend_history[-3:])


class CodeUncertainty:
    """
    Хранит дисперсию SRG для каждого кода.
    Позволяет KCA точнее планировать глубину коррекции.
    """

    def __init__(self):
        self.stats: Dict[str, SRGStats] = {}

    def record(self, code_id: str, score: float):
        if code_id not in self.stats:
            self.stats[code_id] = SRGStats()
        self.stats[code_id].update(score)

    def get_uncertainty(self, code_id: str) -> float:
        if code_id not in self.stats:
            return 1.0
        return self.stats[code_id].variance

    def get_mean(self, code_id: str) -> float:
        if code_id not in self.stats:
            return 0.5
        return self.stats[code_id].mean

    def get_kca_depth(self, code_id: str, default: int = 3) -> int:
        var = self.get_uncertainty(code_id)
        if var > 0.1:
            return min(default + 2, 5)
        elif var > 0.05:
            return default
        else:
            return max(default - 1, 1)


class SRGPlus:
    """
    Расширенный SRG: базовая оценка + аномалии + тренд + неопределённость.
    """

    def __init__(self):
        self.anomaly_detector = SRGAnomalyDetector()
        self.meta_srg = MetaSRG()
        self.uncertainty = CodeUncertainty()

    def evaluate(self, code_id: str, score: float) -> Dict[str, Any]:
        self.anomaly_detector.record(code_id, score)
        self.meta_srg.record(score)
        self.uncertainty.record(code_id, score)

        is_anomaly = self.anomaly_detector.check(code_id, score)
        trend = self.meta_srg.get_trend()
        kca_depth = self.uncertainty.get_kca_depth(code_id)

        return {
            "score": score,
            "is_anomaly": is_anomaly,
            "trend": trend["trend"],
            "declining": trend["declining"],
            "kca_depth": kca_depth,
            "uncertainty": self.uncertainty.get_uncertainty(code_id),
        }

    def get_suspicious_codes(self) -> List[str]:
        return self.anomaly_detector.get_suspicious_codes()

    def should_diagnose(self) -> bool:
        return self.meta_srg.should_diagnose()
