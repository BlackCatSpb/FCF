"""
SelfDescriptiveCodes — авто-описание латентных кодов.

Генерирует текстовое описание каждого кода через LLM с промптом
«Опиши, какое знание закодировано в этом латентном коде».

Описания не используются в вычислениях, но дают интерпретируемость:
- Аудит: что именно хранит система
- Отладка: почему SRG низкий для этого кода
- Документирование: автоматический каталог знаний
"""

import time
import torch
import numpy as np
from typing import Dict, List, Optional
from loguru import logger


class SelfDescriptiveCodes:
    """
    Генерирует текстовые описания латентных кодов.
    """

    def __init__(self, max_description_tokens: int = 50):
        self.max_description_tokens = max_description_tokens
        self.descriptions: Dict[str, str] = {}

    def describe(self, code_id: str, layer, tokenizer,
                 domain_name: str = "") -> str:
        prompt = (
            "<|im_start|>system\n"
            "Ты анализируешь латентный код. Опиши одним предложением, "
            "какое знание или концепт закодирован в этом коде.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Домен: {domain_name or code_id}\n"
            f"Опиши знание в этом латентном коде.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        try:
            encoding = tokenizer.encode(prompt)
            ids = encoding.ids if hasattr(encoding, "ids") else encoding
            input_ids = torch.tensor([ids], dtype=torch.long)

            device = next(layer.parameters()).device
            input_ids = input_ids.to(device)

            with torch.no_grad():
                output = layer.generate(
                    input_ids,
                    max_new_tokens=self.max_description_tokens,
                    temperature=0.7,
                )

            full = tokenizer.decode(output[0].tolist())
            marker = "<|im_start|>assistant\n"
            if marker in full:
                description = full[full.rfind(marker) + len(marker):]
            else:
                description = full[len(prompt):]

            description = description.strip().replace("<|im_end|>", "")
            description = description[:200]

            self.descriptions[code_id] = description
            return description

        except Exception as e:
            logger.warning(f"[SelfDesc] Ошибка для {code_id}: {e}")
            return f"[Ошибка генерации: {e}]"

    def get(self, code_id: str) -> str:
        return self.descriptions.get(code_id, "")

    def catalog(self) -> List[Dict[str, str]]:
        return [
            {"code_id": cid, "description": desc}
            for cid, desc in self.descriptions.items()
        ]
