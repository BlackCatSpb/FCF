"""
MetaMemory — мета-память слоя.

Отслеживает:
- usage_count: сколько раз слой использовался
- confidence_history: история оценок SRG (последние N)
- created_at: время создания
- last_clarification: последний уточняющий вопрос

Используется GrowthController для принятия решений о расширении.
"""

import time
from typing import Optional, List
from collections import deque


class MetaMemory:

    def __init__(self, history_size: int = 100):
        self.usage_count: int = 0
        self.confidence_history: List[float] = []
        self._confidence_deque: deque = deque(maxlen=history_size)
        self.history_size: int = history_size
        self.created_at: float = time.time()
        self.last_clarification: Optional[str] = None

    def record(self, confidence: float):
        self.confidence_history.append(confidence)
        self._confidence_deque.append(confidence)

        if len(self.confidence_history) > self.history_size:
            self.confidence_history.pop(0)

        self.usage_count += 1

    def average_confidence(self, window: int = 20) -> float:
        if not self._confidence_deque:
            return 0.0

        recent = list(self._confidence_deque)[-window:]
        if not recent:
            return 0.0

        return sum(recent) / len(recent)

    def recent_confidence_trend(self, window: int = 10) -> float:
        if len(self._confidence_deque) < window:
            return 0.0

        recent = list(self._confidence_deque)[-window:]
        half = window // 2
        first_half = recent[:half]
        second_half = recent[half:]

        avg_first = sum(first_half) / len(first_half) if first_half else 0.0
        avg_second = sum(second_half) / len(second_half) if second_half else 0.0

        return avg_second - avg_first

    def consecutive_low_confidence(self, threshold: float = 0.6) -> int:
        count = 0
        for c in reversed(list(self._confidence_deque)):
            if c < threshold:
                count += 1
            else:
                break
        return count

    def reset(self):
        self.usage_count = 0
        self.confidence_history.clear()
        self._confidence_deque.clear()
        self.last_clarification = None

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at
