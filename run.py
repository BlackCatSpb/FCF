"""
Точка входа FCF — Единая Вычислительная Архитектура (ЕВА).

Режимы:
- --init: Создать PrimordialLayer и протестировать прямой проход
- --train-tokenizer: Обучить BPE-токенизатор на Wikipedia
- --train-language: Запустить самообучение языку (Пункт 2)
- --train-instruction: Запустить инструктивное дообучение (Пункт 3)
- --interactive: Интерактивный диалог с системой
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fcf.config import load_config, FCFConfig
from fcf.primordial_layer import PrimordialLayer
from fcf.utils import save_primordial_layer, load_primordial_layer
from fcf.tokenizer_utils import (
    load_tokenizer,
    train_tokenizer_on_wikipedia,
    create_fallback_tokenizer,
)
from fcf.language_trainer import LanguageTrainer
from fcf.instruction_trainer import InstructionTrainer
from fcf.domain_trainer import DomainTrainer
from fcf.domain_registry import DomainRegistry
from fcf.recursive_processor import RecursiveProcessor
from fcf.layer_crystallizer import LayerCrystallizer
from fcf.sleep_mode import SleepMode
from fcf.kca_engine import KCAEngine
from fcf.state_algebra import StateAlgebra
from fcf.hnsw_index import HNSWIndex
from fcf.environment_tuner import EnvironmentAutoTuner
from fcf.auto_trainer import AutoTrainer


def cmd_init(config_path: str = None):
    logger.info("=" * 60)
    logger.info("FCF — Пункт 1. Первооснова")
    logger.info("=" * 60)

    config = load_config(config_path)
    logger.info(f"[Init] Конфигурация: d_model={config.d_model}, num_heads={config.num_heads}")

    layer = PrimordialLayer(config)

    total_params = sum(p.numel() for p in layer.parameters())
    logger.info(f"[Init] PrimordialLayer создан: {total_params:,} параметров")
    logger.info(f"[Init] StateStorage: FAISS IndexFlatIP ({config.d_model}d)")
    logger.info(f"[Init] SRG: w_sim={config.srg.w_sim}, w_ent={config.srg.w_ent}, w_eth={config.srg.w_eth}")
    logger.info(f"[Init] EthicsFilter: 5 аксиом, threshold={config.srg.ethics_threshold}")
    logger.info(f"[Init] GrowthController: width_thr={config.growth.width_threshold}, depth_thr={config.growth.depth_threshold}")
    logger.info(f"[Init] CuriosityLoop: threshold={config.curiosity.threshold}")

    test_input = torch.randint(0, min(config.vocab_size, 1000), (1, 16))
    with torch.no_grad():
        x = layer.embed(test_input)
        hidden = layer.forward_transformer(x)
        logits = layer.forward_logits(hidden)
    logger.info(f"[Init] Тестовый прямой проход: OK (input={test_input.shape}, embedding={x.shape}, hidden={hidden.shape}, logits={logits.shape})")

    test_text = "Привет! Как дела?"
    ethics_score, axiom_scores = layer.srg.ethics_filter.evaluate(test_text)
    logger.info(f"[Init] Тест EthicsFilter: score={ethics_score:.2f}, axioms={axiom_scores}")

    return layer


def cmd_interactive(config_path: str = None, checkpoint_path: str = None):
    logger.info("=" * 60)
    logger.info("FCF — Интерактивный режим")
    logger.info("=" * 60)

    if checkpoint_path and os.path.exists(checkpoint_path):
        layer = load_primordial_layer(checkpoint_path, PrimordialLayer)
        logger.info(f"[Load] Загружен из {checkpoint_path}")
    else:
        layer = cmd_init(config_path)

    tokenizer = _load_or_create_tokenizer()

    print()
    print("EVA (FCF) — интерактивный режим")
    print("Введите 'exit' для выхода, 'save' для сохранения, 'stats' для статистики")
    print()

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЗавершение...")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            break

        if user_input.lower() == "save":
            save_path = os.path.join(os.path.dirname(__file__), "checkpoints", "manual")
            save_primordial_layer(layer, save_path)
            continue

        if user_input.lower() == "stats":
            print(f"Слой: {layer.summary()}")
            print(f"Слепков: {len(layer.state_storage)}")
            print(f"Уверенность (avg): {layer.meta.average_confidence():.3f}")
            print(f"Счётчик любопытства: {layer.curiosity.counter}/{layer.curiosity.threshold}")
            continue

        result = layer.process_query(
            query=user_input,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.8,
        )

        print(f"\nEVA: {result['response']}\n")
        print(f"    [confidence={result['confidence']:.3f}, "
              f"ethics={result['ethics_score']:.3f}, "
              f"growth={result['growth_signal']}]")

        if result.get("clarification_question"):
            print(f"    [Уточняющий вопрос: {result['clarification_question']}]")

    logger.info("Завершение работы.")


def cmd_train_tokenizer(config_path: str = None):
    logger.info("=" * 60)
    logger.info("FCF — Обучение BPE-токенизатора")
    logger.info("=" * 60)

    output_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")

    tokenizer = train_tokenizer_on_wikipedia(
        output_path=output_path,
        vocab_size=50257,
        num_texts=100000,
    )

    if tokenizer:
        logger.info(f"[Tokenizer] Готов: vocab_size={tokenizer.get_vocab_size()}")
        _test_tokenizer(tokenizer)
    else:
        logger.error("[Tokenizer] Не удалось обучить токенизатор")


def cmd_train_language(
    config_path: str = None,
    checkpoint_path: str = None,
    text_file: str = None,
    max_steps: int = None,
    device: str = "cpu",
    use_wikipedia: bool = False,
):
    logger.info("=" * 60)
    logger.info("FCF — Пункт 2. Самообучение языку")
    logger.info("=" * 60)

    if checkpoint_path and os.path.exists(checkpoint_path):
        layer = load_primordial_layer(checkpoint_path, PrimordialLayer)
        logger.info(f"[Load] Загружен из {checkpoint_path}")
    else:
        layer = cmd_init(config_path)

    tokenizer = _load_or_create_tokenizer()
    _test_tokenizer(tokenizer)

    trainer = LanguageTrainer(
        layer=layer,
        tokenizer=tokenizer,
        checkpoint_dir=os.path.join(os.path.dirname(__file__), "checkpoints", "language"),
    )

    stats = trainer.train(
        text_file=text_file,
        max_steps=max_steps,
        device=device,
        use_wikipedia=use_wikipedia,
    )

    logger.info(f"[Train] Статистика: {stats}")
    return stats


def cmd_train_domain(
    config_path: str = None,
    checkpoint_path: str = None,
    conceptnet_db: str = None,
    data_file: str = None,
    domain_id: str = None,
    max_steps: int = 200,
    device: str = "cpu",
):
    logger.info("=" * 60)
    logger.info("FCF — Пункт 4. Доменные правила")
    logger.info("=" * 60)

    if checkpoint_path and os.path.exists(checkpoint_path):
        layer = load_primordial_layer(checkpoint_path, PrimordialLayer)
        logger.info(f"[Load] Загружен из {checkpoint_path}")
    else:
        layer = cmd_init(config_path)

    tokenizer = _load_or_create_tokenizer()
    _test_tokenizer(tokenizer)

    registry = DomainRegistry()
    if checkpoint_path and os.path.exists(checkpoint_path):
        reg_path = os.path.join(checkpoint_path, "domain_registry.pkl")
        if os.path.exists(reg_path):
            registry = DomainRegistry.load(reg_path)

    trainer = DomainTrainer(
        layer=layer,
        tokenizer=tokenizer,
        registry=registry,
        checkpoint_dir=os.path.join(os.path.dirname(__file__), "checkpoints", "domain"),
    )

    if conceptnet_db and os.path.exists(conceptnet_db):
        results = trainer.train_from_conceptnet(
            db_path=conceptnet_db,
            max_steps_per_domain=max_steps,
            device=device,
        )
        logger.info(f"[Domain] Результаты ConceptNet: {len(results)} доменов")
    elif data_file and domain_id:
        ok = trainer.train_single_domain(
            domain_id=domain_id,
            data_file=data_file,
            max_steps=max_steps,
            device=device,
        )
        logger.info(f"[Domain] Домен {domain_id}: {'OK' if ok else 'FAIL'}")
    else:
        logger.error("Укажите --conceptnet-db или --data-file + --domain-id")

    logger.info(f"[Domain] Реестр: {registry.summary()}")
    return registry


def cmd_train_depth(
    config_path: str = None,
    checkpoint_path: str = None,
    text_file: str = None,
    max_steps: int = 50,
    device: str = "cpu",
):
    logger.info("=" * 60)
    logger.info("FCF — Пункт 5. Рост в глубину")
    logger.info("=" * 60)

    if checkpoint_path and os.path.exists(checkpoint_path):
        layer = load_primordial_layer(checkpoint_path, PrimordialLayer)
        logger.info(f"[Load] Загружен из {checkpoint_path}")
    else:
        layer = cmd_init(config_path)

    tokenizer = _load_or_create_tokenizer()

    crystallizer = LayerCrystallizer(device=device)
    crystallizer.set_layers([layer])

    recursive = RecursiveProcessor()

    test_queries = [
        "Объясни сложную взаимосвязь между квантовой физикой и сознанием",
        "Расскажи про устройство вселенной",
        "Что такое жизнь с точки зрения философии?",
    ]

    for query in test_queries:
        logger.info(f"[Depth] Тест: {query[:60]}...")
        result = layer.process_query(query=query, tokenizer=tokenizer)

        if result["confidence"] < 0.5:
            encoding = tokenizer.encode(query)
            ids = encoding.ids if hasattr(encoding, "ids") else encoding
            input_ids = torch.tensor([ids], dtype=torch.long)

            rec_result = recursive.process(
                layer=layer,
                input_ids=input_ids,
                tokenizer=tokenizer,
            )

            if rec_result["recursion_exhausted"]:
                recursive.add_failed_query(query, result["response"], result["confidence"])

    if recursive.should_crystallize():
        logger.info("[Depth] Кристаллизация нового слоя...")
        new_layer = crystallizer.crystallize(
            tokenizer=tokenizer,
            failed_queries=recursive.get_failed_queries(),
            checkpoint_dir=os.path.join(os.path.dirname(__file__), "checkpoints", "depth"),
        )
        if new_layer:
            logger.info(f"[Depth] Новый слой создан: всего слоёв={crystallizer.num_layers}")
    else:
        logger.info("[Depth] Кристаллизация не требуется")

    return crystallizer


def cmd_sleep(
    config_path: str = None,
    checkpoint_path: str = None,
):
    logger.info("=" * 60)
    logger.info("FCF — Пункт 6. Sleep Mode (Консолидация)")
    logger.info("=" * 60)

    if checkpoint_path and os.path.exists(checkpoint_path):
        layer = load_primordial_layer(checkpoint_path, PrimordialLayer)
        logger.info(f"[Load] Загружен из {checkpoint_path}")
    else:
        layer = cmd_init(config_path)

    registry = DomainRegistry()
    reg_path = os.path.join(os.path.dirname(__file__), "domain_rules", "registry.pkl")
    if os.path.exists(reg_path):
        registry = DomainRegistry.load(reg_path)

    sleep = SleepMode()

    stats = sleep.execute(
        layers=[layer],
        domain_registry=registry,
    )

    logger.info(f"[Sleep] Статистика: {stats}")
    return stats


def cmd_auto_tune(config_path: str = None):
    logger.info("=" * 60)
    logger.info("FCF — Автонастройка среды исполнения")
    logger.info("=" * 60)

    tuner = EnvironmentAutoTuner()
    profile = tuner.discover()
    tuner.apply()

    print()
    print(tuner.summary())
    print()

    config = tuner.get_training_config()
    logger.info(f"[AutoTune] Конфиг обучения: {config}")

    return tuner


def cmd_auto_train(
    config_path: str = None,
    checkpoint_path: str = None,
):
    logger.info("=" * 60)
    logger.info("FCF — Фоновое автодообучение")
    logger.info("=" * 60)

    if checkpoint_path and os.path.exists(checkpoint_path):
        layer = load_primordial_layer(checkpoint_path, PrimordialLayer)
    else:
        layer = cmd_init(config_path)

    tokenizer = _load_or_create_tokenizer()

    tuner = EnvironmentAutoTuner()
    tuner.discover()
    tuner.apply()

    registry = DomainRegistry()
    reg_path = os.path.join(os.path.dirname(__file__), "domain_rules", "registry.pkl")
    if os.path.exists(reg_path):
        registry = DomainRegistry.load(reg_path)

    trainer = AutoTrainer(
        layer=layer,
        tokenizer=tokenizer,
        domain_registry=registry,
        tuner=tuner,
    )

    trainer.start(check_interval=30.0)

    print()
    print("Автодообучение запущено в фоне.")
    print("Триггеры: деградация доменов, деградация слоёв, failed_queries.")
    print(f"Интервал проверки: 30с")
    print(f"Доменов в реестре: {len(registry)}")
    print()

    try:
        while True:
            time.sleep(30)
            stats = tuner.get_runtime_stats()
            print(
                f"\r  CPU: {stats.cpu_percent:.0f}% | "
                f"RAM free: {stats.ram_free_gb:.1f}GB | "
                f"Training events: {len(trainer.get_history())} | "
                f"Failed queries: {len(trainer.failed_queries)}",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        print()
        trainer.stop()
        logger.info("Автодообучение остановлено.")

    return trainer


def cmd_full_test(
    config_path: str = None,
    checkpoint_path: str = None,
):
    logger.info("=" * 60)
    logger.info("FCF — Пункт 7. Полноценный FCF (тест всех компонентов)")
    logger.info("=" * 60)

    if checkpoint_path and os.path.exists(checkpoint_path):
        layer = load_primordial_layer(checkpoint_path, PrimordialLayer)
    else:
        layer = cmd_init(config_path)

    tokenizer = _load_or_create_tokenizer()

    logger.info("--- KCA Engine ---")
    kca = KCAEngine(hidden_dim=layer.config.d_model)
    z = np.random.randn(layer.config.d_model).astype(np.float32)
    c_q = np.random.randn(layer.config.d_model).astype(np.float32)
    z_opt, conf = kca.refine(z, c_q)
    logger.info(f"KCA: z.shape={z_opt.shape}, confidence={conf:.3f}")

    logger.info("--- State Algebra ---")
    algebra = StateAlgebra(dim=layer.config.d_model)
    za = np.random.randn(layer.config.d_model).astype(np.float32)
    zb = np.random.randn(layer.config.d_model).astype(np.float32)
    z_sum = algebra.sum(za, zb)
    logger.info(f"StateAlgebra: sum.shape={z_sum.shape}")

    logger.info("--- HNSW Index ---")
    hnsw = HNSWIndex(dim=layer.config.d_model)
    hnsw.add_domain("test", c_q)
    found = hnsw.search_domain(c_q)
    logger.info(f"HNSW: domain={found}")

    logger.info("--- Sleep Mode ---")
    sleep = SleepMode()
    sleep.on_query()

    logger.info("--- Layer Crystallizer ---")
    crystallizer = LayerCrystallizer(device="cpu")
    crystallizer.set_layers([layer])
    logger.info(f"Crystallizer: {crystallizer.summary()}")

    logger.info("=== Все компоненты FCF работоспособны ===")
    return True


def cmd_train_instruction(
    config_path: str = None,
    checkpoint_path: str = None,
    instructions_file: str = None,
    max_steps: int = None,
    device: str = "cpu",
):
    logger.info("=" * 60)
    logger.info("FCF — Пункт 3. Инструктивное дообучение")
    logger.info("=" * 60)

    if checkpoint_path and os.path.exists(checkpoint_path):
        layer = load_primordial_layer(checkpoint_path, PrimordialLayer)
        logger.info(f"[Load] Загружен из {checkpoint_path}")
    else:
        layer = cmd_init(config_path)

    tokenizer = _load_or_create_tokenizer()
    _test_tokenizer(tokenizer)

    trainer = InstructionTrainer(
        layer=layer,
        tokenizer=tokenizer,
        checkpoint_dir=os.path.join(os.path.dirname(__file__), "checkpoints", "instruction"),
    )

    stats = trainer.train(
        instructions_file=instructions_file,
        max_steps=max_steps,
        device=device,
    )

    logger.info(f"[Train] Статистика: {stats}")
    return stats


def cmd_lazy_learn(config_path: str = None, checkpoint_path: str = None):
    logger.info("=" * 60)
    logger.info("FCF — Ленивое обучение + Интерактивный режим")
    logger.info("=" * 60)

    if checkpoint_path and os.path.exists(checkpoint_path):
        layer = load_primordial_layer(checkpoint_path, PrimordialLayer)
        logger.info(f"[Load] Загружен из {checkpoint_path}")
    else:
        layer = cmd_init(config_path)

    tokenizer = _load_or_create_tokenizer()

    tuner = EnvironmentAutoTuner()
    tuner.discover()
    tuner.apply()

    registry = DomainRegistry()
    reg_path = os.path.join(os.path.dirname(__file__), "domain_rules", "registry.pkl")
    if os.path.exists(reg_path):
        registry = DomainRegistry.load(reg_path)

    trainer = AutoTrainer(
        layer=layer,
        tokenizer=tokenizer,
        domain_registry=registry,
        tuner=tuner,
    )
    trainer.start(check_interval=60.0)

    import threading
    training_active = [True]
    training_thread = None

    from fcf.language_trainer import LanguageTrainer
    lt = LanguageTrainer(layer=layer, tokenizer=tokenizer,
                        checkpoint_dir=os.path.join(os.path.dirname(__file__), "checkpoints", "lazy"))

    def _background_training():
        while training_active[0]:
            try:
                lt.train(max_steps=500, device="cpu", use_wikipedia=True)
                save_path = os.path.join(os.path.dirname(__file__), "checkpoints", "lazy")
                save_primordial_layer(layer, save_path)
            except Exception as e:
                logger.warning(f"[Lazy] Ошибка: {e}")
                time.sleep(10)

    training_thread = threading.Thread(target=_background_training, daemon=True)
    training_thread.start()
    logger.info("[Lazy] Фоновое Wikipedia-обучение запущено")

    print()
    print("=" * 60)
    print("  FCF — Ленивое обучение активно")
    print("=" * 60)
    print(f"  Слой: {layer.summary()}")
    print(f"  Доменов: {len(registry)}")
    print(f"  Автодообучение: проверка каждые 60с")
    print(f"  Wikipedia-обучение: ФОНОВОЕ (100 шагов/цикл)")
    print()
    print("  Команды:")
    print("    stats  — статистика")
    print("    train  — обучение на Wikipedia (5000 шагов)")
    print("    save   — сохранить чекпоинт")
    print("    exit   — выход")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЗавершение...")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            break

        if user_input.lower() == "save":
            save_path = os.path.join(os.path.dirname(__file__), "checkpoints", "lazy")
            save_primordial_layer(layer, save_path)
            print(f"Сохранено в {save_path}")
            continue

        if user_input.lower() == "stats":
            print(f"  Слой: {layer.summary()}")
            print(f"  Слепков: {len(layer.state_storage)}")
            print(f"  SRG avg: {layer.meta.average_confidence():.3f}")
            print(f"  Доменов: {len(registry)}")
            print(f"  Training events: {len(trainer.get_history())}")
            print(f"  Failed queries: {len(trainer.failed_queries)}")
            print(f"  CPU: {tuner.get_runtime_stats().cpu_percent:.0f}%")
            continue

        if user_input.lower() == "train":
            print("Запуск обучения на Wikipedia (5000 шагов)...")
            from fcf.language_trainer import LanguageTrainer
            lt = LanguageTrainer(layer=layer, tokenizer=tokenizer)
            lt.train(max_steps=5000, device="cpu", use_wikipedia=True)
            save_path = os.path.join(os.path.dirname(__file__), "checkpoints", "lazy")
            save_primordial_layer(layer, save_path)
            print(f"Обучение завершено. Сохранено в {save_path}")
            continue

        trainer.resource.set_generating()
        result = layer.process_query(
            query=user_input,
            tokenizer=tokenizer,
            max_new_tokens=128,
            temperature=0.7,
        )
        trainer.resource.set_idle()

        confidence = result["confidence"]
        if confidence < 0.6:
            trainer.add_failed_query(user_input, result["response"], confidence)

        print(f"\nEVA: {result['response']}\n")
        print(f"    [conf={confidence:.3f}, ethics={result['ethics_score']:.2f}]")

        if result.get("clarification_question"):
            print(f"    [?: {result['clarification_question']}]")

    training_active[0] = False
    if training_thread:
        training_thread.join(timeout=5.0)
    trainer.stop()
    logger.info("Завершение.")
def cmd_fcf_system(config_path: str = None, checkpoint_path: str = None):
    logger.info("=" * 60)
    logger.info("FCFSystem — Полный когнитивный цикл")
    logger.info("=" * 60)

    from fcf.fcf_system import FCFSystem
    fcf = FCFSystem()
    fcf.bootstrap(checkpoint_path)
    fcf.start_background(interval=300.0)

    print()
    print("=" * 60)
    print(f"  FCFSystem active — {fcf.summary()}")
    print("  Фоновый цикл: Sleep Mode каждые 300с")
    print("  Команды: stats, exit")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down...")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "stats":
            s = fcf.stats()
            for k, v in s.items():
                print(f"  {k}: {v}")
            continue

        result = fcf.query(user_input)
        print(f"\nEVA: {result.get('response', '?')}\n")
        print(f"    [conf={result.get('confidence', 0):.3f}, "
              f"domain={result.get('domain_id', '?')}, "
              f"scenario={result.get('scenario', '?')}]")
        if result.get('code_description'):
            print(f"    [desc: {result['code_description'][:100]}]")

    fcf.stop_background()
    logger.info("FCFSystem stopped.")
    tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    if os.path.exists(tokenizer_path):
        try:
            return load_tokenizer(tokenizer_path)
        except Exception as e:
            logger.warning(f"[Token] Ошибка загрузки: {e}")
    logger.warning("[Token] tokenizer.json не найден. Используется fallback.")
    return create_fallback_tokenizer()


def _test_tokenizer(tokenizer):
    test_text = "Привет! Как дела? Это тест токенизатора."
    try:
        encoding = tokenizer.encode(test_text)
        ids = encoding.ids if hasattr(encoding, "ids") else encoding
        decoded = tokenizer.decode(ids)
        logger.info(f"[Token] Тест: '{test_text}' -> {len(ids)} токенов -> '{decoded}'")
    except Exception as e:
        logger.warning(f"[Token] Тест не пройден: {e}")


def main():
    parser = argparse.ArgumentParser(description="FCF — Единая Вычислительная Архитектура")
    parser.add_argument("--init", action="store_true", help="Инициализировать PrimordialLayer")
    parser.add_argument("--interactive", action="store_true", help="Интерактивный режим")
    parser.add_argument("--train-tokenizer", action="store_true", help="Обучить BPE-токенизатор")
    parser.add_argument("--train-language", action="store_true", help="Самообучение языку (Пункт 2)")
    parser.add_argument("--train-instruction", action="store_true", help="Инструктивное дообучение (Пункт 3)")
    parser.add_argument("--train-domain", action="store_true", help="Обучение доменных правил (Пункт 4)")
    parser.add_argument("--train-depth", action="store_true", help="Рост в глубину (Пункт 5)")
    parser.add_argument("--sleep", action="store_true", help="Запустить консолидацию (Пункт 6)")
    parser.add_argument("--full-test", action="store_true", help="Полный тест всех компонентов (Пункт 7)")
    parser.add_argument("--auto-tune", action="store_true", help="Автонастройка среды исполнения")
    parser.add_argument("--auto-train", action="store_true", help="Запустить фоновое автодообучение")
    parser.add_argument("--lazy-learn", action="store_true", help="Интерактивный режим + фоновое дообучение")
    parser.add_argument("--fcf", action="store_true", help="FCFSystem — полный когнитивный цикл")
    parser.add_argument("--config", type=str, default=None, help="Путь к config.json")
    parser.add_argument("--checkpoint", type=str, default=None, help="Путь к чекпоинту для загрузки")
    parser.add_argument("--max-steps", type=int, default=None, help="Максимальное число шагов обучения")
    parser.add_argument("--device", type=str, default="cpu", help="Устройство (cpu/cuda)")
    parser.add_argument("--text-file", type=str, default=None, help="Путь к текстовому файлу для обучения")
    parser.add_argument("--wikipedia", action="store_true", help="Использовать Wikipedia для обучения")
    parser.add_argument("--instructions-file", type=str, default=None, help="Путь к JSON с инструкциями")
    parser.add_argument("--conceptnet-db", type=str, default=None, help="Путь к ConceptNet SQLite базе")
    parser.add_argument("--data-file", type=str, default=None, help="Путь к JSON с фактами для домена")
    parser.add_argument("--domain-id", type=str, default=None, help="Идентификатор домена")

    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

    if args.interactive:
        cmd_interactive(config_path=args.config, checkpoint_path=args.checkpoint)
    elif args.train_tokenizer:
        cmd_train_tokenizer(config_path=args.config)
    elif args.train_language:
        cmd_train_language(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            text_file=args.text_file,
            max_steps=args.max_steps,
            device=args.device,
            use_wikipedia=args.wikipedia,
        )
    elif args.train_instruction:
        cmd_train_instruction(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            instructions_file=args.instructions_file,
            max_steps=args.max_steps,
            device=args.device,
        )
    elif args.train_domain:
        cmd_train_domain(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            conceptnet_db=args.conceptnet_db,
            data_file=args.data_file,
            domain_id=args.domain_id,
            max_steps=args.max_steps,
            device=args.device,
        )
    elif args.train_depth:
        cmd_train_depth(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            text_file=args.text_file,
            max_steps=args.max_steps or 50,
            device=args.device,
        )
    elif args.sleep:
        cmd_sleep(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
        )
    elif args.full_test:
        cmd_full_test(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
        )
    elif args.auto_tune:
        cmd_auto_tune(config_path=args.config)
    elif args.auto_train:
        cmd_auto_train(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
        )
    elif args.lazy_learn:
        cmd_lazy_learn(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
        )
    elif args.fcf:
        cmd_fcf_system(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
        )
    elif args.init:
        layer = cmd_init(config_path=args.config)
        save_path = os.path.join(os.path.dirname(__file__), "checkpoints", "init")
        save_primordial_layer(layer, save_path)
        logger.info(f"[Init] Сохранено в {save_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
