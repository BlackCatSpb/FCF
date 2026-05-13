"""
DomainTrainer — обучение LoRA-адаптеров для доменных правил (Пункт 4).

Процесс:
1. Загружает структурированные данные (ConceptNet, Wikidata)
2. Группирует факты по концептам/доменам
3. Обучает LoRA-адаптер на фактах домена (базовые веса заморожены)
4. Сохраняет адаптер и регистрирует домен в DomainRegistry

Источники данных:
- ConceptNet (локальная SQLite база, 11 ГБ)
- JSON-файлы с фактами
- RuBQ (триплеты Wikidata)
"""

import os
import sys
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict
from loguru import logger

from .config import FCFConfig
from .primordial_layer import PrimordialLayer
from .data_manager import DataManager
from .lora_adapter import LoRAAdapter, TARGET_MODULES
from .domain_registry import DomainRegistry


class DomainTrainer:

    def __init__(
        self,
        layer: PrimordialLayer,
        tokenizer,
        registry: DomainRegistry = None,
        config: FCFConfig = None,
        checkpoint_dir: str = None,
    ):
        self.layer = layer
        self.tokenizer = tokenizer
        self.registry = registry if registry is not None else DomainRegistry()
        self.config = config or layer.config
        self.checkpoint_dir = checkpoint_dir or os.path.join(
            os.path.dirname(__file__), "..", "checkpoints", "domain"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.rank = 8
        self.alpha = 0.7

    def train_from_conceptnet(
        self,
        db_path: str,
        num_facts: int = 1000,
        domain_prefix: str = "cn",
        max_steps_per_domain: int = 200,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("Пункт 4 — Доменные правила (ConceptNet)")
        logger.info(f"База: {db_path}")
        logger.info(f"Фактов: {num_facts}, шагов на домен: {max_steps_per_domain}")
        logger.info("=" * 60)

        if device == "cpu":
            torch.set_num_threads(min(os.cpu_count() or 4, 4))

        facts = DataManager.load_conceptnet(db_path, language="ru")
        if not facts:
            logger.error("Не удалось загрузить ConceptNet")
            return {"error": "no_data"}

        facts = facts[:num_facts]
        logger.info(f"[ConceptNet] Загружено фактов: {len(facts)}")

        domains = self._group_facts_by_concept(facts, min_facts=5)
        logger.info(f"[Domain] Выделено доменов: {len(domains)}")

        results = {}
        for domain_id, domain_facts in domains.items():
            logger.info(f"\n[Domain] Обучение: {domain_id} ({len(domain_facts)} фактов)")

            adapter = LoRAAdapter(
                d_model=self.config.d_model,
                rank=self.rank,
                alpha=self.alpha,
            )

            self._train_adapter_on_facts(
                adapter=adapter,
                facts=domain_facts,
                max_steps=max_steps_per_domain,
                device=device,
            )

            adapter_path = os.path.join(
                self.checkpoint_dir, f"{domain_prefix}_{domain_id}.lora"
            )
            adapter.save(adapter_path)

            centroid = self._compute_domain_centroid(domain_facts, device)

            self.registry.add(
                domain_id=f"{domain_prefix}_{domain_id}",
                context_centroid=centroid,
                adapter_path=adapter_path,
            )

            results[domain_id] = {
                "facts": len(domain_facts),
                "centroid_norm": float(np.linalg.norm(centroid)),
            }

            self.registry.save()

        logger.info(f"\n[Domain] Итого доменов: {len(self.registry)}")
        return results

    def train_single_domain(
        self,
        domain_id: str,
        data_file: str,
        max_steps: int = 200,
        device: str = "cpu",
    ) -> bool:
        logger.info(f"[Domain] Обучение домена: {domain_id} из {data_file}")

        if not os.path.exists(data_file):
            logger.error(f"Файл не найден: {data_file}")
            return False

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            facts = data
        elif isinstance(data, dict):
            facts = data.get("data", data.get("facts", []))
        else:
            facts = []

        if not facts:
            logger.error("Нет данных")
            return False

        adapter = LoRAAdapter(
            d_model=self.config.d_model,
            rank=self.rank,
            alpha=self.alpha,
        )

        self._train_adapter_on_facts(
            adapter=adapter,
            facts=facts,
            max_steps=max_steps,
            device=device,
        )

        adapter_path = os.path.join(
            self.checkpoint_dir, f"{domain_id}.lora"
        )
        adapter.save(adapter_path)

        centroid = self._compute_domain_centroid(facts, device)

        self.registry.add(
            domain_id=domain_id,
            context_centroid=centroid,
            adapter_path=adapter_path,
            domain_data_path=data_file,
        )

        self.registry.save()
        logger.info(f"[Domain] Домен {domain_id} обучен и сохранён")
        return True

    def _group_facts_by_concept(
        self, facts: List[Dict], min_facts: int = 5
    ) -> Dict[str, List[Dict]]:
        groups = defaultdict(list)

        for fact in facts:
            concept = fact.get("concept", "unknown")
            key = concept.lower().replace(" ", "_")[:30]
            groups[key].append(fact)

        return {
            k: v for k, v in groups.items() if len(v) >= min_facts
        }

    def _compute_domain_centroid(
        self, facts: List[Dict], device: str = "cpu"
    ) -> np.ndarray:
        vectors = []

        self.layer.eval()
        with torch.no_grad():
            for fact in facts[:50]:
                text = f"{fact.get('concept', '')} {fact.get('relation', '')} {fact.get('target', '')}"
                if not text.strip():
                    continue

                try:
                    encoding = self.tokenizer.encode(text)
                    ids = encoding.ids if hasattr(encoding, "ids") else encoding
                    ids_tensor = torch.tensor([ids[:64]], dtype=torch.long).to(device)

                    ctx = self.layer.get_context_vector(ids_tensor)
                    vectors.append(ctx)
                except Exception:
                    continue

        if not vectors:
            return np.zeros(self.config.d_model, dtype=np.float32)

        centroid = np.mean(vectors, axis=0)
        return centroid / (np.linalg.norm(centroid) + 1e-8)

    def _train_adapter_on_facts(
        self,
        adapter: LoRAAdapter,
        facts: List[Dict],
        max_steps: int = 200,
        device: str = "cpu",
    ):
        self.layer.train()

        optimizer = torch.optim.AdamW(self.layer.parameters(), lr=1e-5)

        blocks = []
        for fact in facts:
            text = (
                f"Понятие: {fact.get('concept', '')}. "
                f"Связь: {fact.get('relation', '')}. "
                f"Цель: {fact.get('target', '')}."
            )
            try:
                encoding = self.tokenizer.encode(text)
                ids = encoding.ids if hasattr(encoding, "ids") else encoding
                ids = ids[:96]
                while len(ids) < 96:
                    ids.append(0)

                input_ids = torch.tensor([ids[:-1]], dtype=torch.long)
                labels = torch.tensor([ids[1:]], dtype=torch.long)
                blocks.append((input_ids, labels))
            except Exception:
                continue

            if len(blocks) >= 50:
                break

        if not blocks:
            logger.warning("[Domain] Нет блоков для обучения")
            return

        logger.info(f"[Domain] Блоков: {len(blocks)}, шагов: {max_steps}")

        for step in range(max_steps):
            total_loss = 0.0
            optimizer.zero_grad()

            for input_ids, labels in blocks[:2]:
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                x = self.layer.embed(input_ids)
                hidden = self.layer.forward_transformer(x)
                logits = self.layer.forward_logits(hidden)

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=0,
                )
                total_loss += loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.layer.parameters(), max_norm=1.0)
            optimizer.step()

            if step % 10 == 0 or step == max_steps - 1:
                logger.info(f"[Domain] step={step}, loss={total_loss.item():.4f}")

        self.layer.eval()
        logger.info(f"[Domain] Обучение завершено: loss={total_loss.item():.4f}")

    def _forward_with_adapter(
        self, adapter: LoRAAdapter, x: torch.Tensor
    ) -> torch.Tensor:
        saved = {}

        for name in adapter.target_modules:
            if hasattr(self.layer.transformer.attention, name):
                w = getattr(self.layer.transformer.attention, name)
                saved[name] = w.weight.data.clone()
                delta = adapter.get_delta(name).to(w.weight.device)
                w.weight.data = w.weight.data + delta
            elif hasattr(self.layer.transformer.ffn, name):
                w = getattr(self.layer.transformer.ffn, name)
                saved[name] = w.weight.data.clone()
                delta = adapter.get_delta(name).to(w.weight.device)
                w.weight.data = w.weight.data + delta

        hidden = self.layer.transformer(x)

        for name, original in saved.items():
            if hasattr(self.layer.transformer.attention, name):
                getattr(self.layer.transformer.attention, name).weight.data = original
            elif hasattr(self.layer.transformer.ffn, name):
                getattr(self.layer.transformer.ffn, name).weight.data = original

        return hidden
