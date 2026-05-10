"""
AutoTrainer — автономное фоновое дообучение.

Аналог LearningOrchestrator + OnlineTrainer из EVA-Ai.

Цикл в фоновом потоке:
- Мониторит уверенность доменов и слоёв
- При деградации — дообучает
- При обнаружении нового домена — создаёт адаптер
- Пауза во время генерации ответов (ResourceManager)
"""

import os
import time
import threading
import json
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from loguru import logger


class ResourceManager:

    def __init__(self):
        self._state = "idle"
        self._lock = threading.Lock()
        self._generation_count = 0

    def set_generating(self):
        with self._lock:
            self._state = "busy"
            self._generation_count += 1

    def set_idle(self):
        with self._lock:
            self._state = "idle"

    def can_train(self) -> bool:
        with self._lock:
            return self._state == "idle"

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def generation_count(self) -> int:
        with self._lock:
            return self._generation_count


class AutoTrainer:

    def __init__(
        self,
        layer,
        tokenizer,
        domain_registry=None,
        tuner=None,
        checkpoint_dir: str = None,
    ):
        self.layer = layer
        self.tokenizer = tokenizer
        self.domain_registry = domain_registry
        self.tuner = tuner

        self.resource = ResourceManager()
        self.checkpoint_dir = checkpoint_dir or os.path.join(
            os.path.dirname(__file__), "..", "checkpoints", "auto"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.training_history: List[Dict[str, Any]] = []
        self.failed_queries: List[Dict[str, Any]] = []

        self.check_interval: float = 60.0

        self.domain_degradation_threshold: float = 0.6
        self.domain_degradation_window: int = 20
        self.layer_degradation_threshold: float = 0.5
        self.layer_degradation_window: int = 50
        self.min_failed_queries: int = 10
        self.finetune_steps: int = 100

    def start(self, check_interval: float = None):
        if self._running:
            return

        if check_interval:
            self.check_interval = check_interval

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(
            f"[AutoTrainer] Фоновое обучение запущено "
            f"(интервал={self.check_interval}с)"
        )

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("[AutoTrainer] Фоновое обучение остановлено")

    def _loop(self):
        while self._running:
            try:
                if not self.resource.can_train():
                    time.sleep(5)
                    continue

                if self.tuner and not self.tuner.can_train():
                    time.sleep(5)
                    continue

                had_work = False

                had_work = self._check_domain_degradation() or had_work
                had_work = self._check_layer_degradation() or had_work
                had_work = self._process_failed_queries() or had_work

                if not had_work:
                    time.sleep(self.check_interval)
                else:
                    time.sleep(10)

            except Exception as e:
                logger.warning(f"[AutoTrainer] Ошибка цикла: {e}")
                time.sleep(self.check_interval)

    def _check_domain_degradation(self) -> bool:
        if self.domain_registry is None or len(self.domain_registry) == 0:
            return False

        had_work = False
        for domain_id in list(self.domain_registry.rules.keys()):
            rule = self.domain_registry.get_rule(domain_id)
            if rule is None:
                continue

            recent = rule.confidence_history[-self.domain_degradation_window:]
            if len(recent) < self.domain_degradation_window:
                continue

            avg_conf = sum(recent) / len(recent)
            if avg_conf < self.domain_degradation_threshold:
                logger.info(
                    f"[AutoTrainer] Деградация домена {domain_id}: "
                    f"avg_conf={avg_conf:.3f}"
                )
                self._retrain_domain(domain_id, rule)
                had_work = True

        return had_work

    def _check_layer_degradation(self) -> bool:
        if len(self.layer.meta.confidence_history) < 10:
            return False

        avg_conf = self.layer.meta.average_confidence(
            window=self.layer_degradation_window
        )
        if avg_conf < self.layer_degradation_threshold and avg_conf > 0.0:
            logger.info(
                f"[AutoTrainer] Деградация слоя: "
                f"avg_conf={avg_conf:.3f}"
            )
            self._retrain_layer()
            return True
        return False

    def _process_failed_queries(self) -> bool:
        if len(self.failed_queries) < self.min_failed_queries:
            return False

        logger.info(
            f"[AutoTrainer] Обработка failed_queries: "
            f"{len(self.failed_queries)} шт."
        )

        self._finetune_on_queries(self.failed_queries, self.finetune_steps)
        self.failed_queries.clear()
        return True

    def _retrain_domain(self, domain_id: str, rule):
        if not os.path.exists(rule.adapter_path):
            return

        try:
            from .lora_adapter import LoRAAdapter
            adapter = LoRAAdapter.load(rule.adapter_path)
        except Exception as e:
            logger.warning(f"[AutoTrainer] Ошибка загрузки адаптера: {e}")
            return

        domain_data = rule.domain_data_path
        facts = []
        if domain_data and os.path.exists(domain_data):
            try:
                with open(domain_data, "r", encoding="utf-8") as f:
                    data = json.load(f)
                facts = data if isinstance(data, list) else data.get("facts", [])
            except Exception:
                pass

        if not facts:
            logger.warning(f"[AutoTrainer] Нет данных для домена {domain_id}")
            return

        self.layer.eval()

        optimizer = torch.optim.AdamW(
            list(adapter.get_trainable_parameters()),
            lr=1e-5,
        )

        blocks = []
        for fact in facts[:100]:
            text = (
                f"{fact.get('concept', '')} "
                f"{fact.get('relation', '')} "
                f"{fact.get('target', '')}"
            )
            try:
                encoding = self.tokenizer.encode(text)
                ids = encoding.ids if hasattr(encoding, "ids") else encoding
                ids = ids[:64]
                while len(ids) < 64:
                    ids.append(0)
                blocks.append((
                    torch.tensor([ids[:-1]], dtype=torch.long),
                    torch.tensor([ids[1:]], dtype=torch.long),
                ))
            except Exception:
                continue

            if len(blocks) >= 20:
                break

        if not blocks:
            return

        logger.info(f"[AutoTrainer] Дообучение домена {domain_id}: {len(blocks)} блоков")

        for step in range(self.finetune_steps):
            total_loss = 0.0
            optimizer.zero_grad()

            for input_ids, labels in blocks[:2]:
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
            optimizer.step()

            if step % 20 == 0:
                logger.debug(
                    f"[AutoTrainer] domain={domain_id} step={step} "
                    f"loss={total_loss.item():.4f}"
                )

        adapter.save(rule.adapter_path)
        self.training_history.append({
            "type": "domain_retrain",
            "domain": domain_id,
            "steps": self.finetune_steps,
            "loss": total_loss.item(),
        })
        logger.info(f"[AutoTrainer] Домен {domain_id} дообучен")

    def _retrain_layer(self):
        if not self.failed_queries:
            logger.warning("[AutoTrainer] Нет failed_queries для дообучения слоя")
            return

        self._finetune_on_queries(self.failed_queries, self.finetune_steps)
        self.failed_queries = self.failed_queries[-20:]

    def _finetune_on_queries(self, queries: list, steps: int):
        self.layer.train()

        optimizer = torch.optim.AdamW(
            self.layer.parameters(), lr=1e-5
        )

        blocks = []
        for fq in queries[:50]:
            text = f"Запрос: {fq.get('query', '')}\nОтвет: {fq.get('response', '')}"
            try:
                encoding = self.tokenizer.encode(text)
                ids = encoding.ids if hasattr(encoding, "ids") else encoding
                ids = ids[:64]
                while len(ids) < 64:
                    ids.append(0)
                blocks.append((
                    torch.tensor([ids[:-1]], dtype=torch.long),
                    torch.tensor([ids[1:]], dtype=torch.long),
                ))
            except Exception:
                continue

            if len(blocks) >= 20:
                break

        if not blocks:
            self.layer.eval()
            return

        logger.info(f"[AutoTrainer] Дообучение слоя: {len(blocks)} блоков, {steps} шагов")

        for step in range(steps):
            total_loss = 0.0
            optimizer.zero_grad()

            for input_ids, labels in blocks[:2]:
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

            if step % 20 == 0:
                logger.debug(
                    f"[AutoTrainer] step={step} loss={total_loss.item():.4f}"
                )

        self.layer.eval()
        self.training_history.append({
            "type": "layer_retrain",
            "steps": steps,
            "queries": len(blocks),
            "loss": total_loss.item(),
        })
        logger.info(f"[AutoTrainer] Слой дообучен")

    def add_failed_query(self, query: str, response: str, confidence: float):
        self.failed_queries.append({
            "query": query,
            "response": response,
            "confidence": confidence,
        })
        if len(self.failed_queries) > 200:
            self.failed_queries.pop(0)

    def get_history(self) -> List[Dict]:
        return list(self.training_history)

    def summary(self) -> str:
        lines = [
            f"AutoTrainer:",
            f"  State: {self.resource.state}",
            f"  Generations: {self.resource.generation_count}",
            f"  Failed queries: {len(self.failed_queries)}",
            f"  Training events: {len(self.training_history)}",
        ]
        return "\n".join(lines)

    @property
    def is_running(self) -> bool:
        return self._running
