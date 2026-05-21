"""
����� ����� EVA � ������ �������������� �����������
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

from eva.config import load_config, FCFConfig
from eva.primordial_layer import PrimordialLayer
from eva.utils import save_primordial_layer, load_primordial_layer
from eva.tokenizer_utils import (
    load_tokenizer,
    train_tokenizer_on_wikipedia,
    create_fallback_tokenizer,
)
from eva.legacy.language_trainer import LanguageTrainer
from eva.legacy.instruction_trainer import InstructionTrainer
from eva.legacy.domain_trainer import DomainTrainer
from eva.legacy.domain_registry import DomainRegistry
from eva.legacy.sleep_mode import SleepMode
from eva.legacy.auto_trainer import AutoTrainer


def cmd_init(config_path: str = None):
    logger.info("=" * 60)
    logger.info("EVA � ����� 1. �����������")
    logger.info("=" * 60)

    config = load_config(config_path)
    logger.info(f"[Init] ������������: d_model={config.d_model}, num_heads={config.num_heads}")

    layer = PrimordialLayer(config)

    total_params = sum(p.numel() for p in layer.parameters())
    logger.info(f"[Init] PrimordialLayer ������: {total_params:,} ����������")
    logger.info(f"[Init] StateStorage: FAISS IndexFlatIP ({config.d_model}d)")
    logger.info(f"[Init] SRG: w_sim={config.srg.w_sim}, w_ent={config.srg.w_ent}, w_eth={config.srg.w_eth}")
    logger.info(f"[Init] EthicsFilter: 5 ������, threshold={config.srg.ethics_threshold}")
    logger.info(f"[Init] GrowthController: width_thr={config.growth.width_threshold}, depth_thr={config.growth.depth_threshold}")
    logger.info(f"[Init] CuriosityLoop: threshold={config.curiosity.threshold}")

    test_input = torch.randint(0, min(config.vocab_size, 1000), (1, 16))
    with torch.no_grad():
        x = layer.embed(test_input)
        hidden = layer.forward_transformer(x)
        logits = layer.forward_logits(hidden)
    logger.info(f"[Init] �������� ������ ������: OK (input={test_input.shape}, embedding={x.shape}, hidden={hidden.shape}, logits={logits.shape})")

    test_text = "������! ��� ����?"
    ethics_score, axiom_scores = layer.srg.ethics_filter.evaluate(test_text)
    logger.info(f"[Init] ���� EthicsFilter: score={ethics_score:.2f}, axioms={axiom_scores}")

    return layer


def cmd_interactive(config_path: str = None, checkpoint_path: str = None):
    logger.info("=" * 60)
    logger.info("EVA � ������������� �����")
    logger.info("=" * 60)

    if checkpoint_path and os.path.exists(checkpoint_path):
        layer = load_primordial_layer(checkpoint_path, PrimordialLayer)
        logger.info(f"[Load] �������� �� {checkpoint_path}")
    else:
        layer = cmd_init(config_path)

    tokenizer = _load_or_create_tokenizer()

    print()
    print("EVA (FCF) � ������������� �����")
    print("������� 'exit' ��� ������, 'save' ��� ����������, 'stats' ��� ����������")
    print()

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n����������...")
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
            print(f"����: {layer.summary()}")
            print(f"�������: {len(layer.state_storage)}")
            print(f"����������� (avg): {layer.meta.average_confidence():.3f}")
            print(f"������� �����������: {layer.curiosity.counter}/{layer.curiosity.threshold}")
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
            print(f"    [���������� ������: {result['clarification_question']}]")

    logger.info("���������� ������.")


def cmd_train_tokenizer(config_path: str = None):
    logger.info("=" * 60)
    logger.info("EVA � �������� BPE-������������")
    logger.info("=" * 60)

    output_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")

    tokenizer = train_tokenizer_on_wikipedia(
        output_path=output_path,
        vocab_size=50257,
        num_texts=100000,
    )

    if tokenizer:
        logger.info(f"[Tokenizer] �����: vocab_size={tokenizer.get_vocab_size()}")
        _test_tokenizer(tokenizer)
    else:
        logger.error("[Tokenizer] �� ������� ������� �����������")


def cmd_train_language(
    config_path: str = None,
    checkpoint_path: str = None,
    text_file: str = None,
    max_steps: int = None,
    device: str = "cpu",
    use_wikipedia: bool = False,
):
    logger.info("=" * 60)
    logger.info("EVA � ����� 2. ������������ �����")
    logger.info("=" * 60)

    if checkpoint_path and os.path.exists(checkpoint_path):
        layer = load_primordial_layer(checkpoint_path, PrimordialLayer)
        logger.info(f"[Load] �������� �� {checkpoint_path}")
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

    logger.info(f"[Train] ����������: {stats}")
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
    logger.info("EVA � ����� 4. �������� �������")
    logger.info("=" * 60)

    if checkpoint_path and os.path.exists(checkpoint_path):
        layer = load_primordial_layer(checkpoint_path, PrimordialLayer)
        logger.info(f"[Load] �������� �� {checkpoint_path}")
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
        logger.info(f"[Domain] ���������� ConceptNet: {len(results)} �������")
    elif data_file and domain_id:
        ok = trainer.train_single_domain(
            domain_id=domain_id,
            data_file=data_file,
            max_steps=max_steps,
            device=device,
        )
        logger.info(f"[Domain] ����� {domain_id}: {'OK' if ok else 'FAIL'}")
    else:
        logger.error("������� --conceptnet-db ��� --data-file + --domain-id")

    logger.info(f"[Domain] ������: {registry.summary()}")
    return registry


def cmd_train_depth(
    config_path: str = None,
    checkpoint_path: str = None,
    text_file: str = None,
    max_steps: int = 50,
    device: str = "cpu",
):
    logger.info("=" * 60)
    logger.info("EVA � ����� 5. ���� � �������")
    logger.info("=" * 60)

    if checkpoint_path and os.path.exists(checkpoint_path):
        layer = load_primordial_layer(checkpoint_path, PrimordialLayer)
        logger.info(f"[Load] �������� �� {checkpoint_path}")
    else:
        layer = cmd_init(config_path)

    tokenizer = _load_or_create_tokenizer()

    crystallizer = LayerCrystallizer(device=device)
    crystallizer.set_layers([layer])

    recursive = RecursiveProcessor()

    test_queries = [
        "������� ������� ����������� ����� ��������� ������� � ���������",
        "�������� ��� ���������� ���������",
        "��� ����� ����� � ����� ������ ���������?",
    ]

    for query in test_queries:
        logger.info(f"[Depth] ����: {query[:60]}...")
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
        logger.info("[Depth] �������������� ������ ����...")
        new_layer = crystallizer.crystallize(
            tokenizer=tokenizer,
            failed_queries=recursive.get_failed_queries(),
            checkpoint_dir=os.path.join(os.path.dirname(__file__), "checkpoints", "depth"),
        )
        if new_layer:
            logger.info(f"[Depth] ����� ���� ������: ����� ����={crystallizer.num_layers}")
    else:
        logger.info("[Depth] �������������� �� ���������")

    return crystallizer


def cmd_sleep(
    config_path: str = None,
    checkpoint_path: str = None,
):
    logger.info("=" * 60)
    logger.info("EVA � ����� 6. Sleep Mode (������������)")
    logger.info("=" * 60)

    if checkpoint_path and os.path.exists(checkpoint_path):
        layer = load_primordial_layer(checkpoint_path, PrimordialLayer)
        logger.info(f"[Load] �������� �� {checkpoint_path}")
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

    logger.info(f"[Sleep] ����������: {stats}")
    return stats


def cmd_auto_tune(config_path: str = None):
    logger.info("=" * 60)
    logger.info("EVA � ������������� ����� ����������")
    logger.info("=" * 60)

    tuner = EnvironmentAutoTuner()
    profile = tuner.discover()
    tuner.apply()

    print()
    print(tuner.summary())
    print()

    config = tuner.get_training_config()
    logger.info(f"[AutoTune] ������ ��������: {config}")

    return tuner


def cmd_auto_train(
    config_path: str = None,
    checkpoint_path: str = None,
):
    logger.info("=" * 60)
    logger.info("EVA � ������� ��������������")
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
    print("�������������� �������� � ����.")
    print("��������: ���������� �������, ���������� ����, failed_queries.")
    print(f"�������� ��������: 30�")
    print(f"������� � �������: {len(registry)}")
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
        logger.info("�������������� �����������.")

    return trainer


def cmd_full_test(
    config_path: str = None,
    checkpoint_path: str = None,
):
    logger.info("=" * 60)
    logger.info("EVA � ����� 7. ����������� FCF (���� ������������ �����)")
    logger.info("=" * 60)

    from eva.legacy.fcf_system import FCFSystem

    system = FCFSystem()
    system.bootstrap(checkpoint_path)

    test_queries = [
        "��� ����� �������?",
        "�������, ��� �������� ���������.",
        "�������� � ������� �����.",
    ]

    results = []
    for i, query in enumerate(test_queries):
        logger.info(f"\n--- ����������� ���� {i+1}/{len(test_queries)} ---")
        logger.info(f"������: {query}")

        result = system.query(query, max_tokens=64)

        confidence = result.get("confidence", 0.0)
        response_preview = result.get("response", "")[:100]
        kca_applied = result.get("kca_applied", False)
        domain = result.get("domain", "unknown")

        logger.info(f"  �����: {response_preview}...")
        logger.info(f"  �����������: {confidence:.3f}")
        logger.info(f"  KCA ��������: {kca_applied}")
        logger.info(f"  �����: {domain}")

        results.append(result)

    logger.info(f"\n--- �������� ���������� ---")
    stats = system.stats()
    logger.info(f"  ��������: {stats['queries']}")
    logger.info(f"  �������: {stats['layer_snapshots']}")
    logger.info(f"  HNSW �������: {stats['hnsw_domains']}")
    logger.info(f"  HNSW �������: {stats['hnsw_snapshots']}")
    logger.info(f"  GMM �������: {stats['gmm_domains']}")

    avg_conf = np.mean([r.get("confidence", 0.0) for r in results])
    logger.info(f"  ������� �����������: {avg_conf:.3f}")

    consistency = system.validate_consistency()
    logger.info(f"  ���������������: {consistency}")

    logger.info("=== ����������� ���� FCF ������������� ===")
    return True


def cmd_train_instruction(
    config_path: str = None,
    checkpoint_path: str = None,
    instructions_file: str = None,
    max_steps: int = None,
    device: str = "cpu",
):
    logger.info("=" * 60)
    logger.info("EVA � ����� 3. ������������� ����������")
    logger.info("=" * 60)

    if checkpoint_path and os.path.exists(checkpoint_path):
        layer = load_primordial_layer(checkpoint_path, PrimordialLayer)
        logger.info(f"[Load] �������� �� {checkpoint_path}")
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

    logger.info(f"[Train] ����������: {stats}")
    return stats


def cmd_lazy_learn(config_path: str = None, checkpoint_path: str = None):
    logger.info("=" * 60)
    logger.info("EVA � ������� �������� + ������������� �����")
    logger.info("=" * 60)

    if checkpoint_path and os.path.exists(checkpoint_path):
        layer = load_primordial_layer(checkpoint_path, PrimordialLayer)
        logger.info(f"[Load] �������� �� {checkpoint_path}")
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

    from eva.legacy.language_trainer import LanguageTrainer
    from eva.unified_grammar import UnifiedStateGrammar

    grammar = UnifiedStateGrammar(layer.config.d_model)

    lt = LanguageTrainer(layer=layer, tokenizer=tokenizer,
                         checkpoint_dir=os.path.join(os.path.dirname(__file__), "checkpoints", "lazy"),
                         state_grammar=grammar, benchmark_interval=500)

    # ����������: lower LR ����� �� ������� ����
    if checkpoint_path and os.path.exists(checkpoint_path):
        lt.config.training.learning_rate = 1e-5
        for pg in lt.optimizer.param_groups:
            pg['lr'] = 1e-5
        logger.info("[Lazy] LR ������ �� 1e-5 (����� ����������)")

    from eva.legacy.auto_curriculum import AutoCurriculum
    auto_curriculum = AutoCurriculum(layer=layer, tokenizer=tokenizer,
                                     checkpoint_dir=os.path.join(os.path.dirname(__file__), "checkpoints", "auto_curriculum"))
    lt.auto_curriculum = auto_curriculum
    auto_curriculum.start()
    logger.info("[Lazy] AutoCurriculum �������: ����-����� + ����������")

    real_data_dir = os.path.join(os.path.dirname(__file__), "real_data")

    train_file = None

    # ����� ������� ������� (3.4 GB Wikipedia)
    wiki_large = os.path.join(real_data_dir, "wiki_ru_large.txt")
    if os.path.exists(wiki_large) and os.path.getsize(wiki_large) > 100000000:
        train_file = wiki_large
        logger.info(f"[Lazy] �������: wiki_ru_large.txt ({os.path.getsize(wiki_large) / 1024 / 1024:.1f} MB)")

    # Fallback: ������ combined_ru.txt
    if not train_file:
        combined = os.path.join(real_data_dir, "combined_ru.txt")
        if os.path.exists(combined) and os.path.getsize(combined) > 100000:
            train_file = combined
            logger.info(f"[Lazy] �������: combined_ru.txt ({os.path.getsize(combined) / 1024 / 1024:.1f} MB)")

    if not train_file:
        logger.error("[Lazy] ��� ������!")
        return

    def _background_training():
        # ���� ��������� checkpoint ��� resume
        checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "lazy")
        resume_from = None
        if os.path.exists(checkpoint_dir):
            checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("step_")]
            if checkpoints:
                checkpoints.sort()
                resume_from = os.path.join(checkpoint_dir, checkpoints[-1])
                logger.info(f"[Lazy] Resume ��: {resume_from}")
        
        logger.info("[Lazy] �������� ������ (����-��������� ����� �������� ��������)")
        lt.train(
            max_steps=20000, 
            device="cpu", 
            text_file=train_file,
            resume_from=resume_from,
        )
        save_path = os.path.join(os.path.dirname(__file__), "checkpoints", "lazy")
        save_primordial_layer(layer, save_path)
        logger.info(f"[Lazy] �������� ���������. ��������: {save_path}")
        training_active[0] = False

    training_thread = threading.Thread(target=_background_training, daemon=True)
    training_thread.start()
    logger.info("[Lazy] ������� Wikipedia-�������� ��������")

    print()
    print("=" * 60)
    print("  EVA � ������� �������� �������")
    print("=" * 60)
    print(f"  ����: {layer.summary()}")
    print(f"  �������: {len(registry)}")
    print(f"  ��������������: �������� ������ 60�")
    print(f"  Wikipedia-��������: ������� (100 �����/����)")
    print()
    print("  �������:")
    print("    stats    � ���������� ��������")
    print("    train    � Wikipedia (5000 �����)")
    print("    grammar  � ������������ ����������")
    print("    discover � ������ rule discovery")
    print("    bench    � �������� �������")
    print("    auto     � ������ AutoCurriculum")
    print("    save     � ��������� ��������")
    print("    exit     � �����")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n����������...")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            break

        if user_input.lower() == "save":
            save_path = os.path.join(os.path.dirname(__file__), "checkpoints", "lazy")
            save_primordial_layer(layer, save_path)
            print(f"��������� � {save_path}")
            continue

        if user_input.lower() == "grammar":
            print(grammar.visualize())
            continue

        if user_input.lower() == "discover":
            meta = layer.state_storage.snapshots_meta
            if len(meta) < 10:
                print("����� ������� 10 ������� ��� discovery")
                continue
            pairs = []
            for i in range(0, len(meta) - 2, 2):
                pairs.append((meta[i]["c"], meta[i+1]["c"],
                             (meta[i]["c"] + meta[i+1]["c"]) * 0.5))
            result = grammar.discover(pairs, epochs=20)
            print(f"Discovery: loss={result.get('discovery_loss', 0):.4f}")
            v = grammar.validate_rules(pairs)
            print(f"Validation: improvement={v.get('improvement', 0):.2%}")
            continue

        if user_input.lower() == "bench":
            history = lt.benchmark_history
            if not history:
                print("������� ���������� �����")
                continue
            print(f"Benchmark history ({len(history)} �������):")
            for b in history[-10:]:
                print(f"  step={b['step']:6d} conf={b['avg_confidence']:.3f} "
                       f"snap={b['snapshots']} loss={b['loss_recent']:.4f}")
            continue

        if user_input.lower() == "auto":
            if auto_curriculum is None:
                print("AutoCurriculum �� �������")
                continue
            s = auto_curriculum.summary()
            print(f"  AutoCurriculum:")
            print(f"    �������: {s['searches']}")
            print(f"    ������ ���������: {s['facts_added']}")
            print(f"    ����������: {s['trainings']}")
            print(f"    ����� ������: {s['buffer_size']}")
            print(f"    ������� ��������: {s['gap_counter']}")
            if s['weak_topics']:
                print(f"    ������ ����: {', '.join(s['weak_topics'][:5])}")
            continue

        if user_input.lower() == "stats":
            print(f"  ����: {layer.summary()}")
            print(f"  �������: {len(layer.state_storage)}")
            print(f"  SRG avg: {layer.meta.average_confidence():.3f}")
            print(f"  �������: {len(registry)}")
            print(f"  Training events: {len(trainer.get_history())}")
            print(f"  Failed queries: {len(trainer.failed_queries)}")
            print(f"  CPU: {tuner.get_runtime_stats().cpu_percent:.0f}%")
            continue

        if user_input.lower() == "train":
            print("������ �������� �� Wikipedia (5000 �����)...")
            from eva.legacy.language_trainer import LanguageTrainer
            lt_train = LanguageTrainer(layer=layer, tokenizer=tokenizer)
            lt_train.train(max_steps=5000, device="cpu", use_wikipedia=True)
            save_path = os.path.join(os.path.dirname(__file__), "checkpoints", "lazy")
            save_primordial_layer(layer, save_path)
            print(f"�������� ���������. ��������� � {save_path}")
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
    logger.info("����������.")
def cmd_fcf_system(config_path: str = None, checkpoint_path: str = None):
    logger.info("=" * 60)
    logger.info("EVASystem � ������ ����������� ����")
    logger.info("=" * 60)

    from eva.legacy.fcf_system import FCFSystem as EVASystem
    fcf = EVASystem()
    fcf.bootstrap(checkpoint_path)
    fcf.start_background(interval=300.0)

    print()
    print("=" * 60)
    print(f"  EVASystem active � {fcf.summary()}")
    print("  ������� ����: Sleep Mode ������ 300�")
    print("  �������: stats, exit")
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
    logger.info("EVASystem stopped.")

def cmd_symbolic_train(config_path: str = None, checkpoint_path: str = None,
                       text_file: str = None, max_steps: int = None):
    """Symbolic-��������: ���������� �������� ������ Causal LM."""
    from eva.symbolic import CharacterVocab, PotentialTrainer
    from eva.primordial_layer import PrimordialLayer
    from eva.config import FCFConfig

    config = load_config(config_path)
    config.d_model = 256
    config.vocab_size = 156
    config.num_heads = 8
    config.max_seq_len = 512

    layer = PrimordialLayer(config)
    
    import torch
    if torch.cuda.is_available():
        layer = layer.cuda()
        logger.info(f"[Symbolic] GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("[Symbolic] CPU �����")

    char_vocab = CharacterVocab()
    logger.info(f"[Symbolic] CharacterVocab: {len(char_vocab)} ��������")

    trainer = PotentialTrainer(
        layer=layer,
        char_vocab=char_vocab,
        embed_dim=256,
        checkpoint_dir=os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic"),
    )

    text_file = text_file or os.path.join(os.path.dirname(__file__), "real_data", "wiki_ru_large.txt")
    if not os.path.exists(text_file):
        logger.error(f"[Symbolic] ���� �� ������: {text_file}")
        return

    max_steps = max_steps or 50000
    stats = trainer.train_on_file(text_file, max_steps=max_steps)
    print(f"\nSymbolic-�������� ���������: {stats}")

def _load_or_create_tokenizer():
    tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    if os.path.exists(tokenizer_path):
        try:
            return load_tokenizer(tokenizer_path)
        except Exception as e:
            logger.warning(f"[Token] ������ ��������: {e}")
    logger.warning("[Token] tokenizer.json �� ������. ������������ fallback.")
    return create_fallback_tokenizer()


def _test_tokenizer(tokenizer):
    test_text = "������! ��� ����? ��� ���� ������������."
    try:
        encoding = tokenizer.encode(test_text)
        ids = encoding.ids if hasattr(encoding, "ids") else encoding
        decoded = tokenizer.decode(ids)
        logger.info(f"[Token] ����: '{test_text}' -> {len(ids)} ������� -> '{decoded}'")
    except Exception as e:
        logger.warning(f"[Token] ���� �� �������: {e}")


def main():
    parser = argparse.ArgumentParser(description="EVA � ������ �������������� �����������")
    parser.add_argument("--init", action="store_true", help="���������������� PrimordialLayer")
    parser.add_argument("--interactive", action="store_true", help="������������� �����")
    parser.add_argument("--train-tokenizer", action="store_true", help="������� BPE-�����������")
    parser.add_argument("--train-language", action="store_true", help="������������ ����� (����� 2)")
    parser.add_argument("--train-instruction", action="store_true", help="������������� ���������� (����� 3)")
    parser.add_argument("--train-domain", action="store_true", help="�������� �������� ������ (����� 4)")
    parser.add_argument("--train-depth", action="store_true", help="���� � ������� (����� 5)")
    parser.add_argument("--sleep", action="store_true", help="��������� ������������ (����� 6)")
    parser.add_argument("--full-test", action="store_true", help="������ ���� ���� ����������� (����� 7)")
    parser.add_argument("--auto-tune", action="store_true", help="������������� ����� ����������")
    parser.add_argument("--auto-train", action="store_true", help="��������� ������� ��������������")
    parser.add_argument("--lazy-learn", action="store_true", help="������������� ����� + ������� ����������")
    parser.add_argument("--fcf", action="store_true", help="EVASystem � ������ ����������� ����")
    parser.add_argument("--config", type=str, default=None, help="���� � config.json")
    parser.add_argument("--checkpoint", type=str, default=None, help="���� � ��������� ��� ��������")
    parser.add_argument("--max-steps", type=int, default=None, help="������������ ����� ����� ��������")
    parser.add_argument("--device", type=str, default="cpu", help="���������� (cpu/cuda)")
    parser.add_argument("--text-file", type=str, default=None, help="���� � ���������� ����� ��� ��������")
    parser.add_argument("--wikipedia", action="store_true", help="������������ Wikipedia ��� ��������")
    parser.add_argument("--instructions-file", type=str, default=None, help="���� � JSON � ������������")
    parser.add_argument("--conceptnet-db", type=str, default=None, help="���� � ConceptNet SQLite ����")
    parser.add_argument("--data-file", type=str, default=None, help="���� � JSON � ������� ��� ������")
    parser.add_argument("--domain-id", type=str, default=None, help="������������� ������")
    parser.add_argument("--symbolic", action="store_true", help="Symbolic-��������: ���������� �������� ������ LM")

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
    elif args.symbolic:
        cmd_symbolic_train(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            text_file=args.text_file,
            max_steps=args.max_steps,
        )
    elif args.init:
        layer = cmd_init(config_path=args.config)
        save_path = os.path.join(os.path.dirname(__file__), "checkpoints", "init")
        save_primordial_layer(layer, save_path)
        logger.info(f"[Init] ��������� � {save_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
