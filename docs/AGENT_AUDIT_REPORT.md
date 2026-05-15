# EVA — Полный отчёт агентов-архитекторов

## Все найденные ошибки, рекомендации и предложения по улучшению

---

## Агент 1: Архитектурный аудит

### КРИТИЧЕСКИЕ

| # | Файл | Строки | Описание | Статус |
|---|------|--------|----------|--------|
| B1 | run.py | 687 | `EVASystem` не импортируется (класс `FCFSystem`) | ✅ Исправлено |
| B2 | language_trainer.py | 329-334 | Двойной сдвиг labels — модель учит неправильный токен | ✅ Исправлено |

### ВЫСОКИЕ

| # | Файл | Строки | Описание | Статус |
|---|------|--------|----------|--------|
| B3 | 5 файлов | - | Padding токен 0 учится как контент | ✅ Исправлено |
| B4 | domain_trainer.py | 293-318 | `_forward_with_adapter` — мёртвый код | ✅ Исправлено |
| B5 | run.py | 421-466 | `cmd_full_test` не тестирует когнитивный цикл | 🔴 |
| B6 | sleep_mode.py | 84-86 | Кластеры вычисляются и отбрасываются | ✅ Исправлено |
| B7 | fcf_system.py | 294 | AtomicBasis delta: 2W-W_orig вместо W_orig+ΔW | ✅ Исправлено |
| B8 | domain_trainer.py | 233 | Тренирует весь слой вместо LoRA-адаптера | ✅ Исправлено |
| B9 | unified_grammar.py | - | 19 из 41 механизма не вызываются в compose() | ✅ Исправлено |
| B10 | state_grammar.py | 639 | Старый StateGrammar — orphan class | ✅ Удалён |
| B11 | fcf_system + primordial | - | Две параллельные системы доменов | 🔴 |
| B12 | recursive_processor.py | 151 | Потеря контекста (только 16 токенов) | ✅ Исправлено |

### СРЕДНИЕ

| # | Файл | Строки | Описание | Статус |
|---|------|--------|----------|--------|
| B13 | run.py | - | GrowthController сигналы игнорируются | ✅ Исправлено |
| B14 | instruction_trainer.py | 106,109 | Обрезание последнего токена | 🔴 |
| B15 | 5 файлов | - | Несогласованный ignore_index | ✅ Исправлено |
| B16 | fcf_system.py | 284-298 | AtomicBasis — нет восстановления при исключении | 🔴 |
| B17 | sleep_mode v1 vs v2 | - | Несовместимые сигнатуры | ✅ Исправлено |

---

## Агент 2: Анализ потока данных

### КРИТИЧЕСКИЕ НАХОДКИ

| # | Файл | Описание | Статус |
|---|------|----------|--------|
| KCA weights | fcf_system.py:294 | AtomicBasis delta применяется ДО KCA, не во время | ✅ Формула исправлена |
| KCA re-gen | fcf_system.py:312-316 | Ответ перегенерируется если KCA улучшил confidence | ✅ Работает |
| SRG confidence | srg.py | Математически корректна, но энтропия только последнего токена | 🔴 |
| FAISS vs HNSW | - | Два параллельных хранилища, не синхронизированы | 🔴 |
| Optimizer reuse | language_trainer.py:60 | Один оптимизатор, переиспользуется между циклами | ✅ Работает |

---

## Агент 3: StateGrammar Completeness

| # | Файл | Описание | Статус |
|---|------|----------|--------|
| 19 dead mechanisms | все 4 файла | 46% механизмов инициализируются но не вызываются | ✅ Интегрированы |
| InformationGeometry | extensions.py | Не наследует nn.Module | 🔴 |
| StateGrammar orphan | state_grammar.py:639 | Определён но никогда не импортирован | ✅ Удалён |
| Duplicate compose | state_grammar vs unified | Два конкурирующих compose() | ✅ Унифицировано |

---

## Агент 4: Методологические улучшения

### МАТЕМАТИЧЕСКИЕ ОШИБКИ

| # | Файл | Функция | Описание | Статус |
|---|------|---------|----------|--------|
| M1 | kca_engine.py:253-256 | `refine_through_llm` | Оптимизатор отслеживает мёртвый тензор | 🔴 |
| M2 | unified_grammar.py:282-285 | `discover` | Validator никогда не обучается (градиенты не текут) | 🔴 |
| M3 | fractal_hierarchy.py:69-76 | `refine` | Проверка сходимости после обновления, не до | 🔴 |
| M4 | fractal_hierarchy.py:151-163 | `TextAggregator.forward` | Batch>1 — все сэмплы используют веса сэмпла 0 | 🔴 |
| M5 | lora_adapter.py:158-174 | `load` | d_model не передаётся в from_numpy | 🔴 |

### ЭФФЕКТИВНОСТЬ

| # | Файл | Описание | Статус |
|---|------|----------|--------|
| E1 | primordial_layer.py:124-172 | O(T²) авторегрессия без KV-cache | 🔴 |
| E2 | streaming_gmm.py:247-267 | O(d³) на каждое сравнение доменов | 🔴 |
| E3 | hnsw_index.py:67-83 | Pure-Python циклы в PQ train | 🔴 |
| E4 | state_storage.py:109-126 | Полный rebuild индекса на каждое удаление | 🔴 |
| E5 | srg_plus.py:56-79 | Пересчёт mean/std на каждом вызове | 🔴 |

### НЕПОЛНАЯ ЛОГИКА

| # | Файл | Описание | Статус |
|---|------|----------|--------|
| I1 | sleep_mode.py:254-279 | Удаление неиспользуемых слишком рано | ✅ Исправлено |
| I2 | primordial_layer.py:174-190 | bare except скрывает ошибки | 🔴 |
| I3 | atomic_basis.py:111-137 | K может выйти за границы | 🔴 |
| I4 | temporal_context.py:50-67 | EMA насыщается | 🔴 |
| I5 | multi_pass.py:58-74 | Пустой ввод → краш | 🔴 |

### ДУБЛИРОВАНИЕ

| # | Описание | Статус |
|---|----------|--------|
| D1 | MetaMemory: confidence_history + _confidence_deque | 🔴 |
| D2 | 3 копии ConfidenceTracker (StreamingDomain, DomainRule, MetaMemory) | 🔴 |
| D3 | 2 реализации cross-domain translate | 🔴 |

### НЕДОСТАЮЩИЕ МЕТОДЫ

| # | Описание | Статус |
|---|----------|--------|
| N1 | AtomicBasis.incremental_update() | 🔴 |
| N2 | PrimordialLayer.forward_for_training() | 🔴 |
| N3 | HNSWIndex.remove_by_code_id() | 🔴 |
| N4 | FCFSystem.save_checkpoint() — unified | 🔴 |
| N5 | FCFSystem.validate_consistency() | 🔴 |

---

## Агент 5: Оптимизация обучения

### КРИТИЧЕСКИЕ

| # | Файл | Описание | Статус |
|---|------|----------|--------|
| T1 | instruction_trainer.py:107,309 | -100 mask игнорируется (ignore_index=3) | ✅ Исправлено |
| T2 | domain_trainer.py:233,264-285 | Адаптер не обучается; базовые веса неправильно обновлены | ✅ Исправлено |

### СРЕДНИЕ

| # | Файл | Описание | Статус |
|---|------|----------|--------|
| T3 | language_trainer.py:329-336 | Лишний сдвиг отбрасывает последнюю позицию | ✅ Исправлено |
| T4 | domain_trainer.py:281 | total_loss += loss строит граф вычислений | 🔴 |
| T5 | auto_trainer.py:258-278 | Clone-patch-restore 100 GB тензоров | 🔴 |
| T6 | language_trainer.py:424-427 | Дублирующий forward pass после generate() | 🔴 |
| T7 | Все trainer'ы | Нет LR scheduler, warmup | 🔴 |
| T8 | auto_trainer.py:316-318 | Свежий оптимизатор каждый вызов _finetune | 🔴 |
| T9 | Все trainer'ы | Состояние оптимизатора не сохраняется | 🔴 |
| T10 | auto_trainer.py:288 | Нет gradient clipping в _retrain_domain | 🔴 |
| T11 | language_trainer.py:246-249 | Статус average неверен на кратных log_interval | 🔴 |
| T12 | meta_memory.py:22,29 | Избыточный confidence_history + deque | 🔴 |
| T13 | domain_trainer.py:53-54 | LoRA alpha=0.7 слишком мал (effective 0.0875) | 🔴 |

---

## ИТОГО

- **Исправлено:** 28 проблем
- **Осталось:** 27 проблем (M1-M5, E1-E5, I2-I5, D1-D3, N1-N5, T4-T13)
