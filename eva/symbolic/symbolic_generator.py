"""
SymbolicGenerator — генерация текста, управляемая символьными метаданными.

Не "предсказать следующий токен по вероятности", а:
"выбрать допустимый следующий символ, который:
 1. НЕ запрещён (contradiction filter)
 2. СОГЛАСОВАН с контекстом (conditional binding)
 3. ОБРАЗУЕТ валидный паттерн (grammar)
 4. БЛИЗОК в многообразии (topological field)
 5. МОЖЕТ быть новым концептом (concept miner)"

Логиты трансформера — сырой материал. Символьные метаданные — фильтры и усилители.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from loguru import logger


class SymbolicGenerator:
    """
    Генератор, управляемый символьными метаданными.

    Каждый шаг генерации:
    raw_logits → filter(contradictions) → boost(grammar) → boost(concepts) → sample
    """

    def __init__(
        self,
        layer,
        char_vocab,
        potential_field,
        contradiction_filter,
        grammar,
        concept_miner,
        topological_field,
        conditional_binding=None,
        ngram_context=None,
        multi_level_predictor=None,
    ):
        self.layer = layer
        self.char_vocab = char_vocab
        self.potential_field = potential_field
        self.contradiction_filter = contradiction_filter
        self.grammar = grammar
        self.concept_miner = concept_miner
        self.topological_field = topological_field
        self.conditional_binding = conditional_binding
        self.ngram_context = ngram_context
        self.multi_level_predictor = multi_level_predictor

        # Веса для разных сигналов
        self.grammar_boost: float = 1.5
        self.concept_boost: float = 1.2
        self.topology_boost: float = 1.1
        self.context_boost: float = 1.3

    @torch.no_grad()
    def generate(
        self,
        prompt_symbols: List[int],
        max_new_symbols: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> List[int]:
        device = next(self.layer.parameters()).device
        self.layer.eval()

        generated = list(prompt_symbols)
        all_symbols = list(prompt_symbols)
        adaptive_temp = temperature

        for _ in range(max_new_symbols):
            # 1. БАЗА: NGramContext (если есть) или single-symbol continuation
            if self.ngram_context is not None and len(all_symbols) >= 2:
                base_logits = self.ngram_context.get_continuation(all_symbols)
            elif all_symbols:
                last = all_symbols[-1]
                base_logits = self.potential_field.get_continuation_potential(last).cpu().numpy().copy()
            else:
                base_logits = np.ones(self.potential_field.vocab_size)

            # 2. Противоречивый фильтр
            forbidden_mask = self.contradiction_filter.get_forbidden_mask(all_symbols)
            base_logits[forbidden_mask] = float('-inf')

            # 3. Grammar boost: digrams (level 0) + N-grams (level 1)
            if len(all_symbols) >= 1:
                last = all_symbols[-1]
                for sym_idx in range(len(base_logits)):
                    if np.isneginf(base_logits[sym_idx]):
                        continue
                    pair = (last, sym_idx)
                    # Level 0: digrams
                    for _, pat in self.grammar.patterns.get(0, {}).items():
                        if len(pat.symbol_indices) >= 2 and tuple(pat.symbol_indices[:2]) == pair:
                            base_logits[sym_idx] *= 1.5
                            break
                # Level 1: N-grams (if we have 2+ context symbols)
                if len(all_symbols) >= 2:
                    prev_pair = (all_symbols[-2], all_symbols[-1])
                    for _, pat in self.grammar.patterns.get(1, {}).items():
                        if len(pat.symbol_indices) >= 3 and tuple(pat.symbol_indices[:2]) == prev_pair:
                            next_sym = pat.symbol_indices[2] if len(pat.symbol_indices) > 2 else None
                            if next_sym is not None and next_sym < len(base_logits):
                                base_logits[next_sym] *= 1.3
                            break

            # 4. Adaptive temperature: lower when confident, higher when uncertain
            max_pot = base_logits.max()
            if max_pot > 0 and not np.isinf(max_pot):
                # Нормализуем "уверенность" из разброса top значений
                top_vals = np.sort(base_logits[np.isfinite(base_logits)])[-5:]
                if len(top_vals) > 1:
                    confidence = top_vals[-1] / (top_vals.mean() + 1e-8)
                    adaptive_temp = temperature / min(max(confidence, 0.5), 2.0)

            # 5. Sample
            next_symbol = self._sample_from_logits(base_logits, adaptive_temp, top_k, top_p)
            if next_symbol == self.char_vocab.EOS_IDX and len(generated) > 4:
                break

            generated.append(next_symbol)
            all_symbols.append(next_symbol)
            if len(all_symbols) > 512:
                all_symbols = all_symbols[-512:]

        return generated

    def _sample_from_logits(
        self,
        logits: np.ndarray,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> int:
        """Семплирование с temperature, top-k, top-p."""
        logits_safe = np.where(np.isneginf(logits), -1e10, logits)
        logits_safe = logits_safe / max(temperature, 0.01)

        if top_k > 0:
            top_k = min(top_k, len(logits_safe))
            top_indices = np.argpartition(logits_safe, -top_k)[-top_k:]
            mask = np.ones_like(logits_safe, dtype=bool)
            mask[top_indices] = False
            logits_safe[mask] = -1e10

        probs = F.softmax(torch.tensor(logits_safe), dim=-1).numpy()

        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            cumulative = np.cumsum(probs[sorted_indices])
            cutoff = np.searchsorted(cumulative, top_p)
            if cutoff < len(probs):
                probs[sorted_indices[cutoff:]] = 0
                probs = probs / probs.sum()

        return int(np.random.choice(len(probs), p=probs))

    def summary(self) -> str:
        return f"SymbolicGenerator(grammar_boost={self.grammar_boost}, concept_boost={self.concept_boost})"
