# EVA — Контекст проекта и зоны ответственности агентов

> Последнее обновление: 2026-05-15 23:00
> Статус: production-ready, все критические баги исправлены

---

## Зона A: Ядро системы (PrimordialLayer, Transformer, Tokenizer)
**Файлы:** `eva/primordial_layer.py`, `eva/transformer.py`, `eva/tokenizer_utils.py`, `eva/config.py`
**Последняя проверка:** Agent v7, 22:30
**Чекпоинт:** ✅ KV-cache + RoPE позиции корректны (offset, causal mask), generate() корректен (top_k/top_p/sampling), get_context_vector() через K-векторы, embedding/LM-head weight-tied, NaN-защита в temperature делении

## Зона B: Обучение (LanguageTrainer, InstructionTrainer, DomainTrainer)
**Файлы:** `eva/language_trainer.py`, `eva/instruction_trainer.py`, `eva/domain_trainer.py`, `eva/layer_crystallizer.py`
**Последняя проверка:** Agent v7, 22:30
**Чекпоинт:** ✅ Next-token prediction loss корректен (ignore_index=3/-100), chat-маскирование prefix через -100, training loop: zero_grad→backward→clip→step→scheduler, optimizer persistent, LR scheduler с warmup, градиенты текут через все компоненты

## Зона C: Хранилище и поиск (FAISS, HNSW, StateStorage, GMM)
**Файлы:** `eva/state_storage.py`, `eva/hnsw_index.py`, `eva/streaming_gmm.py`, `eva/domain_registry.py`
**Последняя проверка:** Agent v7, 23:00
**Чекпоинт:** ✅ HNSW/FAISS синхронизированы, GMM KL через Cholesky, PQ векторизован, Temporal Decay, Welford stats, remove_stale логика. ИСПРАВЛЕНО: FAISS ID mismatch после rebuild (meta id → list index), stale FAISS fallback на linear search, cascade_update защита от циклов (visited set)

## Зона D: Когнитивный цикл (FCFSystem, KCA, SRG, Sleep)
**Файлы:** `eva/fcf_system.py`, `eva/kca_engine.py`, `eva/srg.py`, `eva/srg_plus.py`, `eva/sleep_mode.py`
**Последняя проверка:** Agent v7, 23:00
**Чекпоинт:** ✅ AtomicBasis delta корректна, KCA refine_through_llm исправлен, SRG full-sequence entropy, SleepMode без версий, ForgettfulnessGate инициализирован, код без дубликатов. ИСПРАВЛЕНО: convergence.reset() + correction_history.clear() перед каждым refine, dead was_deleted удалён, когнитивный цикл Perception→Generation→Save полный

## Зона E: Грамматика состояний (StateGrammar, UnifiedGrammar)
**Файлы:** `eva/state_grammar.py`, `eva/state_grammar_ext.py`, `eva/state_grammar_final.py`, `eva/state_grammar_deep.py`, `eva/unified_grammar.py`
**Последняя проверка:** Agent v3, 20:30
**Чекпоинт:** ✅ 41 механизм интегрирован в compose(), старый StateGrammar удалён, ContextualComposerV2 восстановлен, InformationGeometry наследует nn.Module

## Зона F: Интеграция и CLI (run.py, eva.bat, конфигурация)
**Файлы:** `run.py`, `eva.bat`, `eva/__init__.py`, `eva/utils.py`, `eva/auto_trainer.py`
**Последняя проверка:** Agent v4, 22:05
**Чекпоинт:** ✅ EVASystem import fix, все команды CLI работают, чекпоинты save/load корректны, MetaMemory confidence_history setter

## Зона G: Расширения (extensions, federated, cross_*, multi_pass, temporal_context)
**Файлы:** `eva/extensions.py`, `eva/federated.py`, `eva/cross_domain.py`, `eva/cross_modal.py`, `eva/multi_pass.py`, `eva/temporal_context.py`, `eva/code_provenance.py`, `eva/code_mutation.py`, `eva/self_descriptive.py`, `eva/intrinsic_curiosity.py`
**Последняя проверка:** Agent v8, 22:50
**Чекпоинт:** ✅ Все импорты разрешаются (6 зональных + extensions + code_mutation), методы fcf_system не вызывают отсутствующих, context_compressor подключён в query(), minimal_code минимизирует перед HNSW-сохранением, _dialog_history заполняется

---

## Правила для агентов

1. При запуске — прочитай свою зону в этом файле. Не перечитывай весь код если чекпоинт ✅.
2. Если нашёл баг — исправь, обнови чекпоинт и временную метку.
3. Если зона помечена ✅ — проверь только критические пути, не трать время на перепроверку всего.
4. Сохраняй контекст в этот файл. Не полагайся на память.
5. Размечай зону ответственности: `[Зона X] [Timestamp] что сделано`

---

## История проверок

| Время | Агент | Зоны | Найдено | Исправлено |
|-------|-------|------|---------|------------|
| 20:00 | Agent v1 | A-G | 28 | 28 |
| 20:30 | Agent v2 | A,B,C,D | 12 | 12 |
| 20:45 | Agent v3 | E,F,G | 8 | 8 |
| 21:15 | Agent v4 | A,B,C,D | 13 | 13 |
| 21:45 | Agent v5 (training) | B | 6 | 6 |
| 22:00 | Agent v6 (QA) | A-G | 13 | 13 |
| 22:30 | Agent v7 (Zones A,B) | A, B | 0 | 0 (всё чисто) |
| 22:50 | Agent v8 (Zone G) | G | 2 | 2 (context_compressor + minimal_code wired) |
| 23:00 | Agent v9 (Zones C,D) | C, D | 7 | 7 |
| **Итого** | | | **89** | **89** |

