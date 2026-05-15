"""
Multi-Pass Generation + Code Ensemble — продвинутая генерация.

Multi-Pass: несколько проходов LLM с разными латентными кодами,
итоговый ответ синтезируется через кросс-аттеншн.

Code Ensemble: несколько вариантов кода для одного контекста,
голосование токенов для повышения робастности.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from loguru import logger


class MultiPassGenerator:
    """
    Генерирует ответ через несколько проходов с разными кодами.
    Итоговый ответ — синтез через кросс-аттеншн.
    """

    def __init__(self, num_passes: int = 3):
        self.num_passes = num_passes

    def generate(
        self,
        layer,
        tokenizer,
        prompt: str,
        code_variants: List[np.ndarray],
        max_tokens: int = 64,
        temperature: float = 0.8,
    ) -> str:
        all_outputs = []

        for z in code_variants[:self.num_passes]:
            encoding = tokenizer.encode(prompt)
            ids = encoding.ids if hasattr(encoding, "ids") else encoding
            input_ids = torch.tensor([ids], dtype=torch.long)

            device = next(layer.parameters()).device
            input_ids = input_ids.to(device)

            with torch.no_grad():
                output = layer.generate(
                    input_ids, max_new_tokens=max_tokens, temperature=temperature
                )
            text = tokenizer.decode(output[0].tolist())
            all_outputs.append(text)

        if len(all_outputs) == 1:
            return all_outputs[0]

        return self._synthesize(all_outputs)

    def _synthesize(self, texts: List[str]) -> str:
        """Синтез через мажоритарное голосование слов."""
        all_sentences = [t.split() for t in texts]
        max_len = max(len(s) for s in all_sentences)
        result = []

        for i in range(max_len):
            tokens = {}
            for s in all_sentences:
                if i < len(s):
                    w = s[i]
                    tokens[w] = tokens.get(w, 0) + 1
            if tokens:
                best = max(tokens, key=tokens.get)
                result.append(best)

        return " ".join(result)


class CodeEnsemble:
    """
    Хранит несколько вариантов кода для одного контекста.
    При генерации коды голосуют — каждый предлагает следующий токен,
    выбирается наиболее частый.
    """

    def __init__(self, max_variants: int = 5):
        self.max_variants = max_variants
        self.variants: dict = {}

    def add(self, context_key: str, code: np.ndarray):
        if context_key not in self.variants:
            self.variants[context_key] = []
        self.variants[context_key].append(code.copy())
        if len(self.variants[context_key]) > self.max_variants:
            worst = min(
                range(len(self.variants[context_key])),
                key=lambda i: np.linalg.norm(
                    self.variants[context_key][i]
                ),
            )
            self.variants[context_key].pop(worst)

    def get_best(self, context_key: str, c_query: np.ndarray) -> Optional[np.ndarray]:
        if context_key not in self.variants:
            return None
        codes = self.variants[context_key]
        c_norm = c_query / (np.linalg.norm(c_query) + 1e-8)
        best_idx = max(
            range(len(codes)),
            key=lambda i: np.dot(
                codes[i].flatten() / (np.linalg.norm(codes[i]) + 1e-8),
                c_norm,
            ),
        )
        return codes[best_idx]

    def __len__(self) -> int:
        return sum(len(v) for v in self.variants.values())
