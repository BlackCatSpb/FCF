"""
GrowthController — принимает решение о расширении архитектуры.

Сигналы:
- EXPAND_WIDTH: низкая уверенность + высокие градиенты → создать LoRA-адаптер
- EXPAND_DEPTH: систематически низкая уверенность + patience исчерпан → новый слой
- NO_GROWTH: архитектура справляется
"""

import numpy as np
from typing import Optional
from .meta_memory import MetaMemory


class GrowthController:

    def __init__(
        self,
        width_threshold: float = 0.5,
        depth_threshold: float = 0.3,
        gradient_threshold: float = 1.0,
        patience: int = 20,
    ):
        self.width_threshold = width_threshold
        self.depth_threshold = depth_threshold
        self.gradient_threshold = gradient_threshold
        self.patience = patience

        self._width_signal_count: int = 0
        self._depth_signal_count: int = 0
        self._last_layer_created_at: float = 0.0

    def evaluate(
        self,
        meta: MetaMemory,
        gradient_norm: Optional[float] = None,
        recursion_exhausted: bool = False,
    ) -> str:
        avg = meta.average_confidence()

        if avg < self.depth_threshold and len(meta.confidence_history) >= self.patience:
            if recursion_exhausted:
                self._depth_signal_count += 1
                if self._depth_signal_count >= 3:
                    self._depth_signal_count = 0
                    return "EXPAND_DEPTH"
            else:
                return "TRY_RECURSION"

        if avg < self.width_threshold:
            if gradient_norm is not None and gradient_norm > self.gradient_threshold:
                self._width_signal_count += 1
                if self._width_signal_count >= 2:
                    self._width_signal_count = 0
                    return "EXPAND_WIDTH"

        self._width_signal_count = max(0, self._width_signal_count - 1)
        self._depth_signal_count = max(0, self._depth_signal_count - 1)
        return "NO_GROWTH"

    def can_create_layer(self, min_interval_seconds: float = 300.0) -> bool:
        import time
        elapsed = time.time() - self._last_layer_created_at
        return elapsed >= min_interval_seconds

    def mark_layer_created(self):
        import time
        self._last_layer_created_at = time.time()

    def reset_counters(self):
        self._width_signal_count = 0
        self._depth_signal_count = 0
