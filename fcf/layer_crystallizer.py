"""
LayerCrystallizer — кристаллизация новых слоёв (Пункт 5).

Создаёт новый PrimordialLayer как копию последнего существующего слоя
и специализирует его на запросах, которые проваливались на предшественнике.

Процесс:
1. Копировать веса последнего слоя
2. Инициализировать пустые хранилища (StateStorage, DomainRegistry)
3. Дообучить новый слой на failed_queries (100 шагов)
4. Добавить слой в цепочку (sequential pipeline)

Защита:
- Мин. интервал между слоями (300 сек)
- Макс. слоёв (100)
- Проверка can_create_layer() перед созданием
"""

import os
import time
import copy
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Any
from loguru import logger


class LayerCrystallizer:

    def __init__(
        self,
        max_layers: int = 100,
        min_interval_seconds: float = 300.0,
        finetune_steps: int = 100,
        device: str = "cpu",
    ):
        self.max_layers = max_layers
        self.min_interval_seconds = min_interval_seconds
        self.finetune_steps = finetune_steps
        self.device = device

        self.layers: list = []
        self._last_created_at: float = 0.0

    def set_layers(self, layers: list):
        self.layers = layers

    def can_create_layer(self) -> bool:
        if len(self.layers) >= self.max_layers:
            logger.warning(f"[Crystallize] Достигнут лимит слоёв ({self.max_layers})")
            return False

        elapsed = time.time() - self._last_created_at
        if elapsed < self.min_interval_seconds:
            logger.warning(
                f"[Crystallize] Слишком рано: {elapsed:.0f}с < "
                f"{self.min_interval_seconds}с"
            )
            return False

        return True

    def crystallize(
        self,
        tokenizer,
        failed_queries: List[Dict[str, Any]],
        checkpoint_dir: str = None,
    ) -> Optional[Any]:
        if not self.can_create_layer():
            return None

        if not self.layers:
            logger.error("[Crystallize] Нет слоёв для копирования")
            return None

        last_layer = self.layers[-1]
        PrimordialLayer = type(last_layer)

        new_layer = PrimordialLayer(last_layer.config)
        new_layer.load_state_dict(
            copy.deepcopy(last_layer.state_dict())
        )

        new_layer.state_storage = type(last_layer.state_storage)(
            dim=last_layer.config.d_model,
            max_snapshots=last_layer.config.max_snapshots,
        )
        new_layer.meta.reset()
        new_layer.layer_idx = len(self.layers)

        logger.info(
            f"[Crystallize] Новый слой #{new_layer.layer_idx} создан "
            f"(скопирован с #{last_layer.layer_idx})"
        )

        if failed_queries:
            self._specialize(new_layer, tokenizer, failed_queries)
        else:
            logger.info("[Crystallize] Нет failed_queries для специализации")

        self.layers.append(new_layer)
        self._last_created_at = time.time()

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(
                checkpoint_dir, f"layer_{new_layer.layer_idx:03d}"
            )

            from .utils import save_primordial_layer
            save_primordial_layer(new_layer, path)

        return new_layer

    def _specialize(
        self,
        layer,
        tokenizer,
        failed_queries: List[Dict[str, Any]],
    ):
        logger.info(
            f"[Crystallize] Специализация на {len(failed_queries)} "
            f"проваленных запросах ({self.finetune_steps} шагов)"
        )

        layer.train()
        layer.to(self.device)

        optimizer = torch.optim.AdamW(
            layer.parameters(), lr=1e-5
        )

        blocks = []
        for fq in failed_queries[:50]:
            text = f"Запрос: {fq.get('query', '')}\nОтвет: {fq.get('response', '')}"
            try:
                encoding = tokenizer.encode(text)
                ids = encoding.ids if hasattr(encoding, "ids") else encoding
                ids = ids[:128]
                while len(ids) < 128:
                    ids.append(3)

                input_ids = torch.tensor([ids[:-1]], dtype=torch.long)
                labels = torch.tensor([ids[1:]], dtype=torch.long)
                blocks.append((input_ids, labels))
            except Exception:
                continue

            if len(blocks) >= 20:
                break

        if not blocks:
            logger.warning("[Crystallize] Не удалось токенизировать failed_queries")
            return

        for step in range(self.finetune_steps):
            total_loss = 0.0
            optimizer.zero_grad()

            for input_ids, labels in blocks[:2]:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                x = layer.embed(input_ids)
                hidden = layer.forward_transformer(x)
                logits = layer.forward_logits(hidden)

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=3,
                )
                total_loss += loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=1.0)
            optimizer.step()

            if step % 20 == 0:
                logger.debug(
                    f"[Crystallize] step={step}, loss={total_loss.item():.4f}"
                )

        layer.eval()
        logger.info(
            f"[Crystallize] Специализация завершена: loss={total_loss.item():.4f}"
        )

    def forward_all(
        self, x: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer.forward_transformer(x, attention_mask)
        return x

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def summary(self) -> str:
        lines = [f"LayerCrystallizer({len(self.layers)} layers):"]
        for i, layer in enumerate(self.layers):
            lines.append(
                f"  Layer #{i}: snapshots={len(layer.state_storage)}, "
                f"conf={layer.meta.average_confidence():.3f}, "
                f"usage={layer.meta.usage_count}"
            )
        return "\n".join(lines)
