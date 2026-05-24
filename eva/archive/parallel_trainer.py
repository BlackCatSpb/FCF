"""
ParallelSymbolicTrainer — параллельное обучение символьной модели.

Раскидывает датасет по N worker'ам. Каждый worker:
1. Независимо обрабатывает свой чанк текста
2. Обновляет локальную матрицу аффинности
3. Периодически синхронизирует с глобальной

Синхронизация: только матрица 156×156 (96 KB), не веса модели.
Веса трансформера одинаковы у всех worker'ов и не синхронизируются.

Масштабируется на любое количество ядер CPU.
"""

import os, sys, time, json, math
import threading
import multiprocessing as mp
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from loguru import logger


@dataclass
class WorkerStats:
    worker_id: int
    steps: int = 0
    valid: int = 0
    invalid: int = 0
    elapsed: float = 0.0
    lines_per_sec: float = 0.0


class ParallelSymbolicTrainer:
    """
    Параллельный тренер: N worker'ов → одна матрица аффинности.

    Каждый worker — поток (thread), работающий со своим чанком данных.
    Глобальная матрица аффинности синхронизируется каждые sync_interval шагов.
    """

    def __init__(
        self,
        layer_class,        # класс для создания слоя (PrimordialLayer)
        config,             # конфиг
        char_vocab,         # символьный словарь
        text_file: str,
        num_workers: int = None,
        sync_interval: int = 500,
        chunk_lines: int = 5000,
    ):
        self.layer_class = layer_class
        self.config = config
        self.char_vocab = char_vocab
        self.text_file = text_file
        self.num_workers = num_workers or max(1, os.cpu_count() - 2)
        self.sync_interval = sync_interval
        self.chunk_lines = chunk_lines

        # Глобальная матрица аффинности — shared memory
        self.vocab_size = char_vocab.vocab_size
        self.global_affinity = np.full(
            (self.vocab_size, self.vocab_size), 0.5, dtype=np.float32
        )
        self.global_usage_count = np.zeros(
            (self.vocab_size, self.vocab_size), dtype=np.float32
        )

        # Статистика
        self.workers_stats: List[WorkerStats] = []
        self.total_steps = 0
        self._lock = threading.Lock()
        self._running = False

    def _split_file(self) -> List[List[str]]:
        """Разбить текстовый файл на чанки для worker'ов."""
        chunks = [[] for _ in range(self.num_workers)]

        try:
            with open(self.text_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line or len(line) < 5:
                        continue
                    cyr = sum(1 for c in line if 0x0400 <= ord(c) <= 0x04FF)
                    if cyr < len(line) * 0.5:
                        continue
                    worker_idx = i % self.num_workers
                    chunks[worker_idx].append(line)
                    if i >= self.chunk_lines * self.num_workers:
                        break
        except Exception as e:
            logger.error(f"Ошибка чтения файла: {e}")

        return chunks

    def _worker_loop(
        self,
        worker_id: int,
        lines: List[str],
        max_steps_per_worker: int,
    ):
        """
        Цикл worker'а: обрабатывает свои строки, обновляет локальную аффинность,
        периодически синхронизирует с глобальной.
        """
        import torch
        from eva.symbolic import PotentialField, SemanticClosureChecker, ContinuationPropagator

        # Создаём локальный слой и potential field
        layer = self.layer_class(self.config)
        device = 'cpu'
        if torch.cuda.is_available():
            try:
                layer = layer.cuda()
                device = 'cuda'
            except Exception:
                pass

        local_pf = PotentialField(
            vocab_size=self.char_vocab.vocab_size,
            embed_dim=self.config.d_model,
        )

        checker = SemanticClosureChecker()
        propagator = ContinuationPropagator(local_pf, checker, self.config.d_model)

        stats = WorkerStats(worker_id=worker_id)
        start_time = time.time()
        local_step = 0

        for line in lines:
            if local_step >= max_steps_per_worker:
                break

            try:
                symbol_ids = self.char_vocab.encode(line)[:256]

                inp = torch.tensor([symbol_ids], dtype=torch.long).to(device)

                with torch.no_grad():
                    layer.eval()
                    x = layer.embed(inp)
                    hidden = layer.forward_transformer(x)
                    attn = layer.transformer.attention.last_attention

                # Строим сборку
                n = len(symbol_ids)
                attention_matrix = None
                if attn is not None:
                    attention_matrix = attn[0].mean(dim=0).cpu().numpy()[:n, :n]
                else:
                    attention_matrix = np.eye(n)

                # Оцениваем coherence
                sym_pots, seq_pot = propagator.propagate_sequence(symbol_ids, attention_matrix)
                passed, score, _ = checker.full_check(sym_pots, attention_matrix, seq_pot)

                if score > 0.5:
                    stats.valid += 1
                    if attn is not None:
                        local_pf.strengthen_batch(inp, attn, confidence=score)
                    checker.add_valid_potential(seq_pot)
                else:
                    stats.invalid += 1
                    local_pf.weaken_all(factor=0.999)

                stats.steps += 1
                local_step += 1

                # Синхронизация с глобальной
                if local_step % self.sync_interval == 0:
                    self._sync_local_to_global(local_pf, stats.steps)

            except Exception as e:
                logger.debug(f"Worker {worker_id} error: {e}")

        # Финальная синхронизация
        self._sync_local_to_global(local_pf, stats.steps)

        stats.elapsed = time.time() - start_time
        stats.lines_per_sec = stats.steps / max(stats.elapsed, 0.01)

        with self._lock:
            self.workers_stats.append(stats)
            self.total_steps += stats.steps

    def _sync_local_to_global(self, local_pf: 'PotentialField', local_steps: int):
        """Синхронизация: локальная аффинность → глобальная (взвешенное среднее)."""
        local_aff = local_pf.affinity.cpu().numpy()
        local_count = local_pf.co_occurrence_count.cpu().numpy()

        with self._lock:
            alpha = 0.3  # Вес новых данных
            self.global_affinity = (
                (1.0 - alpha) * self.global_affinity +
                alpha * local_aff
            )
            self.global_usage_count += local_count

            # Применяем обратно ко всем worker'ам через потенциальное поле
            self._broadcast_affinity(local_pf)

    def _broadcast_affinity(self, local_pf: 'PotentialField'):
        """Копируем глобальную аффинность в локальное поле."""
        import torch
        local_pf.affinity.copy_(torch.tensor(self.global_affinity))
        local_pf.co_occurrence_count.copy_(torch.tensor(self.global_usage_count))

    def train(
        self,
        max_steps: int = 10000,
    ) -> Dict:
        """
        Запустить параллельное обучение.

        max_steps — всего шагов (распределяется по worker'ам).
        """
        logger.info(f"[Parallel] Запуск: {self.num_workers} worker'ов, {max_steps} шагов")

        chunks = self._split_file()
        if not chunks or not any(chunks):
            logger.error("Нет данных")
            return {"error": "no_data"}

        steps_per_worker = max_steps // self.num_workers

        threads = []
        self._running = True
        start_time = time.time()

        for wid in range(self.num_workers):
            if not chunks[wid]:
                continue
            t = threading.Thread(
                target=self._worker_loop,
                args=(wid, chunks[wid], steps_per_worker),
                daemon=True,
            )
            t.start()
            threads.append(t)

        # Ждём завершения
        for t in threads:
            t.join()

        self._running = False
        elapsed = time.time() - start_time

        total_valid = sum(s.valid for s in self.workers_stats)
        total_invalid = sum(s.invalid for s in self.workers_stats)
        avg_lines_per_sec = (
            sum(s.lines_per_sec for s in self.workers_stats) / max(len(self.workers_stats), 1)
        )

        result = {
            "workers": self.num_workers,
            "total_steps": self.total_steps,
            "valid": total_valid,
            "invalid": total_invalid,
            "elapsed": elapsed,
            "lines_per_sec_per_worker": avg_lines_per_sec,
            "total_lines_per_sec": avg_lines_per_sec * self.num_workers,
            "worker_stats": [
                {
                    "id": s.worker_id,
                    "steps": s.steps,
                    "valid": s.valid,
                    "lines_per_sec": s.lines_per_sec,
                }
                for s in self.workers_stats
            ],
        }

        logger.info(f"[Parallel] Завершено: {result}")
        return result

    def summary(self) -> str:
        mean_aff = float(self.global_affinity.mean())
        active = int((self.global_usage_count > 10).sum())
        return (
            f"ParallelTrainer(workers={self.num_workers}, "
            f"steps={self.total_steps}, "
            f"mean_aff={mean_aff:.4f}, "
            f"active_connections={active})"
        )
