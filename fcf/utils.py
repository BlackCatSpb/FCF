"""
Сериализация и десериализация компонентов системы.

save() / load() для:
- Весов модели (state_dict)
- FAISS-индекса
- Метаданных хранилища (snapshots_meta)
- DomainRegistry
- Конфигурации
"""

import os
import pickle
import torch
import numpy as np
from typing import Optional
from loguru import logger

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def save_primordial_layer(layer, path: str):
    os.makedirs(path, exist_ok=True)

    torch.save(layer.state_dict(), os.path.join(path, "weights.pt"))
    logger.info(f"[Save] Веса сохранены: {path}/weights.pt")

    snapshots_path = os.path.join(path, "snapshots.pkl")
    with open(snapshots_path, "wb") as f:
        pickle.dump(layer.state_storage.snapshots_meta, f)
    logger.info(f"[Save] Слепки сохранены: {len(layer.state_storage.snapshots_meta)} шт.")

    if HAS_FAISS and layer.state_storage.index is not None:
        index_path = os.path.join(path, "index.faiss")
        faiss.write_index(layer.state_storage.index, index_path)
        logger.info(f"[Save] FAISS-индекс сохранён: {index_path}")

    meta_path = os.path.join(path, "meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump({
            "usage_count": layer.meta.usage_count,
            "confidence_history": layer.meta.confidence_history,
            "created_at": layer.meta.created_at,
            "last_clarification": layer.meta.last_clarification,
        }, f)
    logger.info(f"[Save] Мета-память сохранена: {meta_path}")

    config_path = os.path.join(path, "config.json")
    layer.config.to_json(config_path)
    logger.info(f"[Save] Конфигурация сохранена: {config_path}")

    logger.info(f"[Save] Слой полностью сохранён в {path}")


def load_primordial_layer(path: str, layer_class):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Директория не найдена: {path}")

    config_path = os.path.join(path, "config.json")
    if os.path.exists(config_path):
        from .config import FCFConfig
        config = FCFConfig.from_json(config_path)
    else:
        from .config import FCFConfig
        config = FCFConfig()

    layer = layer_class(config)

    weights_path = os.path.join(path, "weights.pt")
    if os.path.exists(weights_path):
        layer.load_state_dict(torch.load(weights_path, map_location="cpu"))
        logger.info(f"[Load] Веса загружены: {weights_path}")

    snapshots_path = os.path.join(path, "snapshots.pkl")
    if os.path.exists(snapshots_path):
        with open(snapshots_path, "rb") as f:
            layer.state_storage.snapshots_meta = pickle.load(f)
        layer.state_storage.rebuild_from_meta()
        logger.info(f"[Load] Слепки загружены: {len(layer.state_storage.snapshots_meta)} шт.")

    index_path = os.path.join(path, "index.faiss")
    if HAS_FAISS and os.path.exists(index_path):
        layer.state_storage.index = faiss.read_index(index_path)
        logger.info(f"[Load] FAISS-индекс загружен: {index_path}")

    meta_path = os.path.join(path, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta_data = pickle.load(f)
        layer.meta.usage_count = meta_data.get("usage_count", 0)
        layer.meta.confidence_history = meta_data.get("confidence_history", [])
        layer.meta.created_at = meta_data.get("created_at", layer.meta.created_at)
        layer.meta.last_clarification = meta_data.get("last_clarification")
        logger.info(f"[Load] Мета-память загружена")

    logger.info(f"[Load] Слой полностью загружен из {path}")
    return layer
