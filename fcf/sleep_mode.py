"""
SleepMode — консолидация памяти (Пункт 6).

В периоды простоя система анализирует накопленный опыт:
1. Кластеризация слепков (HDBSCAN) — группировка похожих состояний
2. Удаление устаревшего — temporal decay, низкий usage_count
3. Дистилляция LoRA → базовые веса — стабильные адаптеры вплавляются
4. Слияние избыточных слоёв — слои с близкими выходами объединяются
5. Дефрагментация FAISS — перестроение индекса после удалений

Условия активации:
- IDLE_TIMEOUT: 5 минут без запросов
- SLEEP_INTERVAL: 2 часа планово
- Ручной триггер
"""

import os
import time
import math
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

try:
    import hdbscan

    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


class SleepMode:

    def __init__(
        self,
        idle_timeout: int = 300,
        sleep_interval: int = 7200,
        ttl_idle_days: int = 7,
        distill_usage_threshold: int = 100,
        min_cluster_size: int = 5,
    ):
        self.idle_timeout = idle_timeout
        self.sleep_interval = sleep_interval
        self.ttl_idle_days = ttl_idle_days
        self.distill_usage_threshold = distill_usage_threshold
        self.min_cluster_size = min_cluster_size

        self.last_query_time = time.time()
        self.last_sleep_time = time.time()
        self.sleep_count = 0

    def should_sleep(self) -> bool:
        now = time.time()
        idle_trigger = (now - self.last_query_time) > self.idle_timeout
        interval_trigger = (now - self.last_sleep_time) > self.sleep_interval
        return idle_trigger or interval_trigger

    def on_query(self):
        self.last_query_time = time.time()

    def execute(
        self,
        layers: list,
        domain_registry=None,
        crystallizer=None,
    ) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info(f"Sleep Mode #{self.sleep_count + 1} — Консолидация")
        logger.info("=" * 60)

        stats = {
            "clusters_formed": 0,
            "snapshots_removed": 0,
            "domains_distilled": 0,
            "layers_merged": 0,
            "faiss_defragmented": False,
        }

        for layer in layers:
            removed = self._remove_stale_snapshots(layer)
            stats["snapshots_removed"] += removed

        for layer in layers:
            clusters = self._cluster_snapshots(layer)
            stats["clusters_formed"] += clusters

        if domain_registry and len(domain_registry) > 0:
            distilled = self._distill_domains(domain_registry, layers)
            stats["domains_distilled"] += distilled

        if crystallizer and crystallizer.num_layers > 1:
            merged = self._merge_redundant_layers(crystallizer)
            stats["layers_merged"] += merged

        for layer in layers:
            self._defragment_faiss(layer)
        stats["faiss_defragmented"] = True

        self.last_sleep_time = time.time()
        self.sleep_count += 1

        logger.info(f"[Sleep] Консолидация завершена: {stats}")
        return stats

    def _remove_stale_snapshots(self, layer) -> int:
        removed = 0
        now = time.time()
        ttl_seconds = self.ttl_idle_days * 86400

        to_remove = []
        for i, meta in enumerate(layer.state_storage.snapshots_meta):
            age = now - meta.get("timestamp", now)
            usage = meta.get("usage_count", 0)

            score = usage * math.exp(-0.1 * age / 86400)

            if usage == 0 and age > ttl_seconds:
                to_remove.append(i)
            elif score < 0.01:
                to_remove.append(i)

        for idx in sorted(to_remove, reverse=True):
            layer.state_storage._remove(idx)
            removed += 1

        if removed > 0:
            logger.info(f"[Sleep] Удалено устаревших слепков: {removed}")

        return removed

    def _cluster_snapshots(self, layer) -> int:
        if len(layer.state_storage) < self.min_cluster_size:
            return 0

        vectors = layer.state_storage.get_all_vectors()
        if len(vectors) < self.min_cluster_size:
            return 0

        try:
            if HAS_HDBSCAN:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    metric="euclidean",
                )
                labels = clusterer.fit_predict(vectors)
            else:
                from sklearn.cluster import KMeans

                n_clusters = max(1, len(vectors) // self.min_cluster_size)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(vectors)

            unique_labels = set(labels)
            clusters = len([l for l in unique_labels if l >= 0])

            logger.info(
                f"[Sleep] Кластеризация: {clusters} кластеров "
                f"из {len(vectors)} векторов"
            )
            return clusters

        except Exception as e:
            logger.warning(f"[Sleep] Ошибка кластеризации: {e}")
            return 0

    def _distill_domains(self, registry, layers) -> int:
        distilled = 0
        stable_domains = registry.get_stable_domains(
            min_usage=self.distill_usage_threshold,
            min_confidence=0.8,
        )

        for domain_id in stable_domains:
            rule = registry.get_rule(domain_id)
            if not rule or not os.path.exists(rule.adapter_path):
                continue

            logger.info(f"[Sleep] Дистилляция домена: {domain_id}")

            try:
                from .lora_adapter import LoRAAdapter
                adapter = LoRAAdapter.load(rule.adapter_path)

                for layer in layers:
                    for name in adapter.target_modules:
                        delta = adapter.get_delta(name)
                        for target in [layer.transformer.attention, layer.transformer.ffn]:
                            if hasattr(target, name):
                                w = getattr(target, name)
                                w.weight.data = w.weight.data + delta.to(
                                    w.weight.device
                                )

                registry.remove(domain_id)
                distilled += 1
                logger.info(f"[Sleep] Домен {domain_id} вплавлен в базовые веса")

            except Exception as e:
                logger.warning(f"[Sleep] Ошибка дистилляции {domain_id}: {e}")

        return distilled

    def _merge_redundant_layers(self, crystallizer) -> int:
        merged = 0
        layers = crystallizer.layers

        if len(layers) < 2:
            return 0

        test_input = torch.randn(1, 4, layers[0].config.d_model)

        i = 0
        while i < len(layers) - 1:
            layer_a = layers[i]
            layer_b = layers[i + 1]

            try:
                with torch.no_grad():
                    out_a = layer_a.forward_transformer(test_input)
                    out_b = layer_b.forward_transformer(test_input)

                diff = torch.mean((out_a - out_b) ** 2).item()

                if diff < 0.01:
                    logger.info(
                        f"[Sleep] Слияние слоёв #{i} и #{i + 1} "
                        f"(diff={diff:.6f})"
                    )

                    for pa, pb in zip(
                        layer_a.parameters(), layer_b.parameters()
                    ):
                        pa.data = (pa.data + pb.data) / 2.0

                    del layers[i + 1]
                    merged += 1
                else:
                    i += 1

            except Exception as e:
                logger.warning(f"[Sleep] Ошибка слияния: {e}")
                i += 1

        if merged > 0:
            logger.info(f"[Sleep] Слито слоёв: {merged}")

        return merged

    def _defragment_faiss(self, layer):
        storage = layer.state_storage
        if hasattr(storage, "_rebuild_index"):
            try:
                storage._rebuild_index()
            except Exception:
                pass
