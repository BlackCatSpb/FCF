"""
FCFSystem — единая интеграция всех компонентов FCF v2.

Назначение:
  Связывает AtomicBasis, FractalHierarchy, StreamingGMM, HNSWIndex,
  KCAEngine, SRGPlus, SleepModeV2, CodeProvenance, IntrinsicCuriosity
  в единый runtime, реализующий полный когнитивный цикл.

Задача:
  - Прогрессивный bootstrapping: пошаговый запуск системы
  - Трёхфазный цикл: Восприятие → Порождение (KCA) → Сохранение
  - Автоматическая маршрутизация запросов через HNSW
  - Динамическое управление доменами через GMM
  - Фоновая консолидация через Sleep Mode
  - Отслеживание происхождения кодов
"""

import os, time, threading
import torch
import numpy as np
from typing import Optional, Dict, Any, List
from loguru import logger


class FCFSystem:
    """Единая система FCF v2."""

    def __init__(self, config=None):
        from .config import load_config
        from .environment_tuner import EnvironmentAutoTuner

        self.config = config or load_config()

        tuner = EnvironmentAutoTuner()
        tuner.discover()
        self.device = tuner.get_training_device()
        self._inference_device = tuner.get_inference_device()
        tuner.apply()

        self.layer = None
        self.tokenizer = None
        self.atomic_basis = None
        self.hierarchy = None
        self.gmm = None
        self.hnsw = None
        self.kca = None
        self.kca_scheduler = None
        self.srg_plus = None
        self.sleep_mode = None
        self.provenance = None
        self.curiosity = None
        self.state_algebra = None
        self.cross_domain = None
        self.minimal_code = None
        self.federated = None
        self.collaborative_srg = None
        self.ensemble = None
        self.multi_pass = None
        self.context_compressor = None
        self.self_improver = None
        self.code_mutation_module = None
        self.self_descriptive = None

        self._background_thread = None
        self._running = False
        self._query_count = 0
        self._dialog_history = []

        logger.info(f"[FCF] Device: train={self.device}, inference={self._inference_device}")

    def bootstrap(self, checkpoint_path: str = None):
        """
        Прогрессивный bootstrapping системы.

        1. Загрузить или создать PrimordialLayer
        2. Загрузить токенизатор
        3. Инициализировать AtomicBasis (SVD при наличии чекпоинта)
        4. Инициализировать FractalHierarchy
        5. Инициализировать MultiLevelGMM
        6. Инициализировать HNSWIndex
        7. Инициализировать KCAEngine, SRGPlus, SleepModeV2
        """
        from .primordial_layer import PrimordialLayer
        from .tokenizer_utils import load_tokenizer
        from .atomic_basis import AtomicBasis
        from .fractal_hierarchy import FractalHierarchy
        from .streaming_gmm import MultiLevelGMM
        from .hnsw_index import HNSWIndex
        from .kca_engine import KCAEngine
        from .srg_plus import SRGPlus
        from .sleep_mode_v2 import SleepModeV2
        from .code_provenance import CodeProvenance
        from .intrinsic_curiosity import IntrinsicCuriosity
        from .state_algebra import StateAlgebra

        from .federated import FederatedFabric, CollaborativeSRG
        from .cross_domain import CrossDomainModule
        from .temporal_context import TemporalContextCompressor
        from .multi_pass import MultiPassGenerator, CodeEnsemble
        from .code_mutation import CodeMutation, CodeDistillation
        from .self_descriptive import SelfDescriptiveCodes
        from .extensions import (
            MinimalCodePrinciple, RecursiveSelfImprovement,
            ForgetfulnessGateTrainer, AdaptiveKCAScheduler,
        )

        logger.info("=" * 60)
        logger.info("FCF v2 — Bootstrap")
        logger.info("=" * 60)

        if checkpoint_path and os.path.exists(checkpoint_path):
            from .utils import load_primordial_layer
            self.layer = load_primordial_layer(checkpoint_path, PrimordialLayer)
            logger.info(f"[1/9] Слой загружен из {checkpoint_path}")
        else:
            self.layer = PrimordialLayer(self.config)
            logger.info(f"[1/9] Слой создан: {sum(p.numel() for p in self.layer.parameters()):,} params")

        self.tokenizer = load_tokenizer("tokenizer.json")
        logger.info(f"[2/9] Токенизатор: {self.tokenizer.get_vocab_size()} слов")

        if checkpoint_path and os.path.exists(checkpoint_path):
            basis_path = os.path.join(checkpoint_path, "atomic_basis.pkl")
            if os.path.exists(basis_path):
                self.atomic_basis = AtomicBasis.load(basis_path)
            else:
                self.atomic_basis = AtomicBasis()
                self.atomic_basis.decompose(self.layer)
                self.atomic_basis.save(
                    os.path.join(checkpoint_path, "atomic_basis.pkl")
                )
        else:
            self.atomic_basis = AtomicBasis()
        logger.info(f"[3/9] AtomicBasis готов")

        self.hierarchy = FractalHierarchy(self.config.d_model)
        logger.info(f"[4/9] {self.hierarchy.summary()}")

        self.gmm = MultiLevelGMM(dim=self.config.d_model)
        logger.info(f"[5/9] MultiLevelGMM готов")

        self.hnsw = HNSWIndex(dim=self.config.d_model, pq_M=8)
        logger.info(f"[6/9] HNSWIndex готов")

        self.kca = KCAEngine(
            hidden_dim=self.config.d_model,
            max_iterations=self.config.kca.max_iterations,
            rho=self.config.kca.rho,
            eta0=self.config.kca.eta0,
            lambda_gap=self.config.kca.lambda_gap,
            lambda_contra=self.config.kca.lambda_contra,
            lambda_kl=self.config.kca.lambda_kl,
            lambda_mono=self.config.kca.lambda_mono,
        )
        self.kca_scheduler = AdaptiveKCAScheduler()
        self.srg_plus = SRGPlus()
        self.sleep_mode = SleepModeV2()
        self.provenance = CodeProvenance()
        self.curiosity = IntrinsicCuriosity()
        self.state_algebra = StateAlgebra(dim=self.config.d_model)
        self.cross_domain = CrossDomainModule(dim=self.config.d_model)
        self.minimal_code = MinimalCodePrinciple()
        self.federated = FederatedFabric()
        self.collaborative_srg = CollaborativeSRG()
        self.ensemble = CodeEnsemble()
        self.multi_pass = MultiPassGenerator()
        self.context_compressor = TemporalContextCompressor(self.config.d_model)
        self.self_improver = RecursiveSelfImprovement()
        self.code_mutation_module = CodeMutation()
        self.code_distillation = CodeDistillation()
        self.self_descriptive = SelfDescriptiveCodes()

        logger.info(f"[7/9] KCA + SRG+ + SleepV2 + Grammar готовы")

        from .unified_grammar import UnifiedStateGrammar

        self.state_grammar = UnifiedStateGrammar(self.config.d_model)
        logger.info(f"[Grammar] UnifiedStateGrammar: 41 механизм")
        logger.info(f"[8/9] Federated + Ensemble + MultiPass + ContextCompressor готовы")
        logger.info(f"[9/9] MinimalCode + SelfImprovement + Curiosity готовы")

        self._progressive_bootstrap(checkpoint_path)

        logger.info("Bootstrap завершён.")

    def _progressive_bootstrap(self, checkpoint_path: str = None):
        """
        Progressive Bootstrapping шаги 3-4 (§9.1):
        3. Файнтюнинг на 5-10 базовых доменах (пропускаем, если нет данных)
        4. Проекция fine-tuned матриц на атомарный базис, извлечение c_i
        """
        if self.atomic_basis is None or self.layer is None:
            return

        if not (hasattr(self.atomic_basis, 'layers') and self.atomic_basis.layers):
            logger.info("[Boot] SVD разложение выполняется...")
            self.atomic_basis.decompose(self.layer)
            if checkpoint_path:
                self.atomic_basis.save(
                    os.path.join(checkpoint_path, "atomic_basis.pkl")
                )

        for matrix_name in ["W_Q", "W_K", "W_V", "W_O"]:
            try:
                coeffs = self.atomic_basis.encode(
                    self.layer, matrix_name
                )
                logger.debug(
                    f"[Boot] ΔW_{matrix_name}: "
                    f"{len(coeffs)} коэффициентов, "
                    f"||c||={np.linalg.norm(coeffs):.2f}"
                )
            except Exception:
                pass

    def query(self, text: str, max_tokens: int = 128) -> Dict[str, Any]:
        """
        Полный когнитивный цикл обработки запроса.

        Фаза 1: Восприятие — токенизация, контекстный вектор, поиск в HNSW.
        Фаза 2: Порождение — генерация ответа, KCA при низкой уверенности.
        Фаза 3: Сохранение — обновление GMM, HNSW, Provenance.
        """
        self.sleep_mode.on_query()

        c_query = self.layer.get_context_vector(
            self.layer._encode(self.tokenizer, text)
        )

        c_norm = c_query / (np.linalg.norm(c_query) + 1e-8)

        domain_id = None
        scenario = "cold_start"
        z_stored = None

        domain_id = self.hnsw.search_domain(c_norm)

        if domain_id:
            results = self.hnsw.search_snapshot(domain_id, c_norm, top_k=1)
            if results:
                idx, similarity = results[0]
                if similarity > 0.95:
                    scenario = "exact_match"
                    if self.layer.state_storage and idx < len(
                        self.layer.state_storage.snapshots_meta
                    ):
                        z_stored = self.layer.state_storage.snapshots_meta[idx].get("c")
                elif similarity > 0.7:
                    scenario = "partial_match"
                    if self.layer.state_storage and idx < len(
                        self.layer.state_storage.snapshots_meta
                    ):
                        stored = self.layer.state_storage.snapshots_meta[idx].get("c")
                        if stored is not None:
                            noise = np.random.randn(*stored.shape) * 0.05
                            z_stored = stored + noise
                            z_stored = z_stored / (np.linalg.norm(z_stored) + 1e-8)

        if domain_id is None:
            domain_id = self.gmm.classify(c_norm, level="word")
        if domain_id is None:
            domain_id = self.gmm.add_or_update(c_norm, level="word")

        domain = self.gmm.gmms["word"].domains.get(domain_id)

        result = self.layer.process_query(
            text, self.tokenizer, max_new_tokens=max_tokens
        )

        confidence = result["confidence"]
        self.srg_plus.evaluate(domain_id, confidence)

        if domain:
            domain.record_confidence(confidence)
            domain.update(c_norm)

        kca_applied = False
        kca_confidence = 0.0
        if confidence < 0.5:
            z = z_stored if z_stored is not None else np.random.randn(
                self.config.d_model
            ).astype(np.float32)

            if self.atomic_basis is not None:
                try:
                    coeffs = self.atomic_basis.encode(self.layer, "W_Q")
                    for name in ["W_Q", "W_K", "W_V", "W_O"]:
                        try:
                            c = self.atomic_basis.encode(self.layer, name)
                            d = self.atomic_basis.decode(c, name)
                            if hasattr(self.layer.transformer.attention, name):
                                w = getattr(self.layer.transformer.attention, name)
                                orig = self.atomic_basis._original_weights[name]
                                w.weight.data = (orig + torch.from_numpy(d).float()).to(w.weight.device)
                        except Exception:
                            pass
                except Exception:
                    pass

            kca_depth = self.kca_scheduler.get_depth(
                confidence,
                domain.average_confidence() if domain else 0.5,
            )
            self.kca.convergence.max_cycles = kca_depth

            z_opt, kca_conf = self.kca.refine_through_llm(
                z, self.layer, self.tokenizer, text
            )
            kca_applied = True
            kca_confidence = float(kca_conf)

            if kca_conf > confidence:
                result = self.layer.process_query(
                    text, self.tokenizer, max_new_tokens=max_tokens
                )
                confidence = result["confidence"]

            result["kca_applied"] = True
            result["kca_confidence"] = kca_confidence
            result["kca_depth"] = kca_depth

            growth = self.layer.growth.evaluate(self.layer.meta, 0.0)
            if growth == "EXPAND_WIDTH":
                logger.info("[Growth] Сигнал: расширение в ширину")
            elif growth == "EXPAND_DEPTH":
                logger.info("[Growth] Сигнал: рост в глубину")

            if self.atomic_basis is not None:
                try:
                    self.atomic_basis.restore_original(self.layer) if hasattr(self.atomic_basis, 'restore_original') else None
                except Exception:
                    pass

        self._validate_and_save(c_norm, confidence, domain_id, scenario)

        self.provenance.record(
            code_id=f"q_{self._query_count}",
            domain_id=domain_id,
            level="word",
            created_via="query",
            kca_iterations=len(self.kca.correction_history) if kca_applied else 0,
            final_srg=kca_confidence if kca_applied else confidence,
        )

        if self.curiosity and domain:
            self.curiosity.probe(domain, self.layer, self.tokenizer, self.srg_plus)

        self._query_count += 1

        return result

    def _validate_and_save(self, c_vec, confidence, domain_id, scenario):
        if self.layer is None or self.layer.state_storage is None:
            return
        if confidence < 0.8:
            return
        if scenario == "exact_match":
            return

        snapshot_idx = self.hnsw.search_snapshot(domain_id, c_vec, top_k=1)
        if snapshot_idx and snapshot_idx[0][1] > 0.95:
            return

        if hasattr(self, 'code_distillation'):
            existing = [m["c"] for m in self.layer.state_storage.snapshots_meta[-50:]]
            if existing:
                distilled = self.code_distillation.try_distill(c_vec, existing)
                if distilled:
                    self.provenance.record(
                        code_id=f"d_{self._query_count}",
                        domain_id=domain_id, level="word",
                        created_via="distillation",
                        parent_codes=[f"q_{self._query_count}"],
                        final_srg=float(confidence),
                    )
                    return

        self.hnsw.add_snapshot(domain_id, c_vec)

        self.layer._eval_context_vector = c_vec
        self.layer.save_snapshot_if_confident(domain=domain_id)

        if hasattr(self, 'code_mutation_module'):
            code_mutator = self.code_mutation_module
            if code_mutator.should_mutate():
                z = c_vec.copy()
                mutated, score, improved = code_mutator.mutate(z, lambda z, c: confidence)
                if improved:
                    self.hnsw.add_snapshot(domain_id, mutated)

    def start_background(self, interval: float = 300.0):
        """Запустить фоновый цикл: Sleep Mode + Intrinsic Curiosity."""
        if self._running:
            return

        self._running = True

        def _loop():
            while self._running:
                try:
                    if self.sleep_mode.should_sleep():
                        self.sleep_mode.execute(
                            layers=[self.layer],
                            gmm=self.gmm,
                            hnsw_index=self.hnsw,
                            state_algebra=self.state_algebra,
                            self_improver=self.self_improver,
                        )
                    time.sleep(interval)
                except Exception as e:
                    logger.warning(f"[FCF] Ошибка фона: {e}")
                    time.sleep(interval)

        self._background_thread = threading.Thread(target=_loop, daemon=True)
        self._background_thread.start()
        logger.info(f"[FCF] Фоновый цикл запущен (интервал={interval}с)")

    def stop_background(self):
        self._running = False
        if self._background_thread:
            self._background_thread.join(timeout=5.0)
        logger.info("[FCF] Фоновый цикл остановлен")

    def stats(self) -> Dict[str, Any]:
        return {
            "queries": self._query_count,
            "layer_snapshots": len(self.layer.state_storage) if self.layer else 0,
            "hnsw_domains": self.hnsw.size()["domains"] if self.hnsw else 0,
            "hnsw_snapshots": self.hnsw.size()["snapshots"] if self.hnsw else 0,
            "gmm_domains": {lvl: len(gmm) for lvl, gmm in self.gmm.gmms.items()} if self.gmm else {},
            "srp_declining": self.srg_plus.should_diagnose() if self.srg_plus else False,
            "provenance_codes": len(self.provenance.records) if self.provenance else 0,
        }

    def summary(self) -> str:
        s = self.stats()
        return (
            f"FCFSystem(queries={s['queries']}, "
            f"snapshots={s['layer_snapshots']}, "
            f"hnsw={s['hnsw_domains']}d/{s['hnsw_snapshots']}s, "
            f"gmm={s['gmm_domains']}, "
            f"declining={s['srp_declining']})"
        )
