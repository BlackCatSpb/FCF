"""
IntrinsicCuriosity — предотвращение атрофии редко используемых доменов.

С малой вероятностью генерирует синтетический запрос к «забытому»
домену и проверяет, насколько хорошо соответствующий код справляется.
Если уверенность низка — запускает KCA для обновления кода.
"""

import random
import time
import numpy as np
from typing import Dict, List, Any, Optional
from loguru import logger


class IntrinsicCuriosity:
    """
    Периодически проверяет редко используемые домены и коды.
    Предотвращает «забывание» — деградацию неактивных знаний.
    """

    def __init__(
        self,
        probe_probability: float = 0.05,
        min_idle_seconds: float = 3600,
        confidence_threshold: float = 0.6,
    ):
        self.probe_probability = probe_probability
        self.min_idle_seconds = min_idle_seconds
        self.confidence_threshold = confidence_threshold
        self.probe_history: List[Dict] = []

    def should_probe(self, domain) -> bool:
        idle = time.time() - domain.last_used_at
        if idle < self.min_idle_seconds:
            return False
        return random.random() < self.probe_probability

    def probe(
        self,
        domain,
        layer,
        tokenizer,
        srg_plus=None,
    ) -> Optional[Dict[str, Any]]:
        if not self.should_probe(domain):
            return None

        logger.info(f"[Curiosity] Проверка домена {domain.domain_id} (idle={time.time() - domain.last_used_at:.0f}s)")

        query = f"Расскажи о {domain.domain_id.replace('_', ' ')}"

        try:
            result = layer.process_query(query, tokenizer, max_new_tokens=30)
        except Exception as e:
            logger.warning(f"[Curiosity] Ошибка запроса: {e}")
            return None

        confidence = result.get("confidence", 0.0)

        record = {
            "domain_id": domain.domain_id,
            "confidence": confidence,
            "timestamp": time.time(),
            "needs_update": confidence < self.confidence_threshold,
        }
        self.probe_history.append(record)

        if confidence < self.confidence_threshold:
            logger.warning(
                f"[Curiosity] Домен {domain.domain_id} деградировал "
                f"(conf={confidence:.3f}). Требуется KCA-обновление."
            )
        else:
            logger.info(
                f"[Curiosity] Домен {domain.domain_id} в порядке "
                f"(conf={confidence:.3f})"
            )

        domain.last_used_at = time.time()
        return record

    def get_degraded_domains(self) -> List[str]:
        return list(set(
            r["domain_id"] for r in self.probe_history[-50:]
            if r.get("needs_update", False)
        ))
