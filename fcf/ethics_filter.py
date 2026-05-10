"""
EthicsFilter — неизменяемый этический фильтр на 5 аксиомах.

Аксиомы:
1. Не навреди — без призывов к насилию, дискриминации, незаконным действиям.
2. Будь честен — если неуверен, обязан явно указать.
3. Уважай приватность — не сохранять личные данные.
4. Стремись к истине — при противоречии пытаться разрешить.
5. Будь полезен — ответ направлен на помощь пользователю.

Реализация: regex-паттерны со штрафами.
Этический скор < ethics_threshold → безусловное отклонение ответа.
"""

import re
from typing import Dict, Tuple
import numpy as np


class EthicsFilter:
    HARM_PATTERNS = [
        r"убей",
        r"причини\s+вред",
        r"уничтожь",
        r"насилие",
        r"террор(изм|ист|ировать)?",
        r"дискриминац",
        r"пыт(ать|ка|ки)",
        r"избей",
        r"задуши",
        r"отрави",
        r"взорви",
        r"подожги",
        r"наркот(ик|ики)",
        r"сутен(ёр|ер)",
        r"раб(ство|оторгов)",
        r"педофил",
        r"геноцид",
    ]

    PRIVACY_PATTERNS = [
        r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}",
        r"\b\d{10,12}\b",
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        r"(паспорт|passport).*\d{6,10}",
        r"(снилс|СНИЛС).*\d{3}[\s-]?\d{3}[\s-]?\d{3}",
        r"(инн|ИНН).*\d{10,12}",
    ]

    DISHONESTY_PATTERNS = [
        r"(я\s+уверен|я\s+знаю|точно|100%|наверняка).*(?=.*\b(?"
        r"не\s+знаю|не\s+уверен|сомнев|предполож|возможно|вероятно)\b)",
        r"это\s+факт.*(?:кажется|возможно|наверное)",
        r"(?:кажется|возможно|наверное).*(?=.*\bэто\s+факт\b)",
    ]

    USELESS_PATTERNS = [
        r"^\s*(да|нет|ок|ага|угу|не\s+знаю|хз)\s*[.!]*\s*$",
        r"^я\s+не\s+могу\s+(?:ответить|помочь).*$",
        r"^это\s+сложный\s+вопрос.*$",
    ]

    PENALTY_PER_MATCH = 0.2

    def __init__(self, ethics_threshold: float = 0.3):
        self.ethics_threshold = ethics_threshold

    def evaluate(self, text: str) -> Tuple[float, Dict[str, float]]:
        if not text or not isinstance(text, str):
            return 0.0, {ax: 0.0 for ax in ["harm", "honesty", "privacy", "truth", "usefulness"]}

        text_lower = text.lower()
        scores = {
            "harm": self._check_patterns(text_lower, self.HARM_PATTERNS),
            "privacy": self._check_patterns(text, self.PRIVACY_PATTERNS),
            "honesty": self._check_honesty(text_lower),
            "usefulness": self._check_usefulness(text_lower),
            "truth": self._check_truth(text_lower),
        }

        total_score = np.mean(list(scores.values()))
        return max(total_score, 0.0), scores

    def _check_patterns(self, text: str, patterns: list) -> float:
        score = 1.0
        for pattern in patterns:
            if re.search(pattern, text):
                score -= self.PENALTY_PER_MATCH
            if score <= 0.0:
                break
        return max(score, 0.0)

    def _check_honesty(self, text: str) -> float:
        score = 1.0

        confidence_markers = [
            r"\bя\s+уверен\b",
            r"\bя\s+знаю\b",
            r"\bточно\b",
            r"\b100%\b",
            r"\bнаверняка\b",
        ]
        uncertainty_markers = [
            r"\bне\s+знаю\b",
            r"\bне\s+уверен\b",
            r"\bсомнев",
            r"\bпредполож",
            r"\bвозможно\b",
            r"\bвероятно\b",
            r"\bкажется\b",
            r"\bнаверное\b",
        ]

        has_confidence = any(re.search(p, text) for p in confidence_markers)
        has_uncertainty = any(re.search(p, text) for p in uncertainty_markers)

        if has_confidence and has_uncertainty:
            score -= self.PENALTY_PER_MATCH * 1.5

        if has_uncertainty:
            disclaimer = any(
                re.search(p, text) for p in [
                    r"\bмо(?:гу|жет)\s+(?:быть|ошиб)",
                    r"\bуточн",
                    r"\bпровер",
                    r"\bпо\s+моим\s+данным\b",
                    r"\bнасколько\s+(?:я|мне)\s+(?:известно|знаю)\b",
                ]
            )
            if not disclaimer:
                score -= self.PENALTY_PER_MATCH * 0.5

        return max(score, 0.0)

    def _check_usefulness(self, text: str) -> float:
        score = 1.0

        trivial = [
            r"^\s*(?:да|нет|ок|ага|угу)\s*[.!]*\s*$",
            r"^\s*я\s+не\s+(?:могу|знаю|буду)\s+(?:ответить|помочь|делать)",
            r"^\s*это\s+(?:сложный|трудный|хороший)\s+вопрос\s*[.!]*\s*$",
            r"^\s*пожалуйста\s*[.!]*\s*$",
            r"^\s*спасибо\s*[.!]*\s*$",
        ]

        for pattern in trivial:
            if re.match(pattern, text, re.IGNORECASE):
                score -= self.PENALTY_PER_MATCH

        words = len(text.split())
        if words < 3:
            score -= self.PENALTY_PER_MATCH
        elif words < 5:
            score -= self.PENALTY_PER_MATCH * 0.5

        return max(score, 0.0)

    def _check_truth(self, text: str) -> float:
        score = 1.0

        contradiction_markers = [
            r"\bс\s+одной\s+стороны\b.*\bс\s+другой\s+стороны\b",
            r"\b(?:но|однако)\b.*\b(?:поэтому|следовательно|таким\s+образом)\b",
        ]

        for pattern in contradiction_markers:
            if re.search(pattern, text, re.DOTALL):
                score -= self.PENALTY_PER_MATCH * 0.5

        return max(score, 0.0)

    def is_acceptable(self, text: str) -> bool:
        total_score, _ = self.evaluate(text)
        return total_score >= self.ethics_threshold
