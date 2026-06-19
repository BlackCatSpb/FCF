# FCF Architect Report V6 — Коллегия AI-агентов (После рефакторинга V5)

**Дата**: 2026-06-17
**Проект**: Fractal Cognitive Field (FCF) — нейро-символическая языковая модель
**Версия отчёта**: V6 (после масштабного рефакторинга V5: AM-6/7/9/10/11/12, G-9/10, TN-1/2/3/4/10 и др.)
**Состав коллегии**: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent

---

## Сводка

| Агент | Баги | P0 | P1 | P2 | P3 | Улучшения |
|-------|:----:|:--:|:--:|:--:|:--:|:---------:|
| Architect-AI | 3 + 2 санитарных | — | 5 | 4 | — | 11 предложений |
| Neuro-Symbolic Specialist | 5 найдено | — | 3 | 3 | — | 6 предложений |
| GPU-Opt Agent | 4 найдено | — | 4 | 5 | — | — |
| Training-Dynamics Agent | 2 критических | 2 | 2 | 2 | 1 | — |
| Quality-Safety Agent | 3 пробела | — | 3 | 2 | 1 | — |
| **Итого** | **19** | **2** | **17** | **16** | **2** | **17** |

---

# 1. Architect-AI: Архитектурный аудит V6

## Критические баги (crash при штатной работе)

### AM-14 (P1): `_use_torch` не определён в CrystalGenerator.__init__

- **Файл**: `stdp_trainer.py:66`, `crystal_generator.py:1121`
- **Суть**: `gen._use_torch` читается в `STDPTrainer._train` и `crystal_generator.train_batch`/`train_from_text`, но **никогда не присваивается** в конструкторе. Если `use_torch=None` (дефолт), код упадёт с `AttributeError`.
- **Приоритет**: **P1** (crash)
- **Фикс**: `self._use_torch = _HAS_TORCH and torch.cuda.is_available()` в `CrystalGenerator.__init__`

### AM-21 (P1): `cfg.checkpoint_keep` — несуществующий атрибут

- **Файл**: `train_full.py:392`
- **Суть**: `TrainingPipeline.__init__` обращается к `cfg.checkpoint_keep` — такого атрибута в FCFConfig нет (есть `cleanup_keep: int = 5`). `AttributeError` при инициализации TrainingPipeline.
- **Приоритет**: **P1** (crash при активации TrainingPipeline)
- **Фикс**: `cfg.checkpoint_keep` → `cfg.cleanup_keep`

### AM-22 (P1): `gen.lattice.total_freq` — несуществующий атрибут

- **Файл**: `stdp_trainer.py:81`
- **Суть**: `total_freq = gen.lattice.total_freq` — в `SyntaxLattice` нет атрибута `total_freq`, есть метод `total_freq()` или `self._get_total_freq()`.
- **Приоритет**: **P1** (crash при GPU-тренировке)
- **Фикс**: `gen.lattice.total_freq()` или `gen._get_total_freq()`

## Архитектурные улучшения

### AM-13 (P1): Удаление dead code из crystal_generator.py

- **Суть**: В `crystal_generator.py:745-1116` висят 7 методов-дубликатов с полными реализациями, которые никогда не вызываются (делегация в STDPTrainer):
  - `_cpu_stdp_apply`, `_gpu_stdp_apply`, `_negative_sampling_cpu`, `_negative_sampling_gpu`, `_contrastive_objective`, `_contrastive_objective_cpu`, `_contrastive_objective_gpu`
  - **Особенно опасно**: `_gpu_stdp_apply` — вторая, **отличающаяся** реализация GPU STDP (строка 745). Любые правки STDPTrainer не отразятся на этой копии.
- **Сложность**: 2
- **Результат**: crystal_generator сократится с ~1245 до ~1050 строк

### AM-15 (P1): Активация TrainingPipeline (завершение AM-9)

- **Суть**: `TrainingPipeline` (train_full.py:366) — 145 строк написанного кода, который **никем не используется**. Основной цикл (стр. 685-884) полностью дублирует логику. В результате:
  - AM-7 (Async Checkpoint Manager) не используется
  - TN-4 (Early Stopping) не работает
  - 150 строк dead code
- **Сложность**: 5

### AM-19 (P1): GPU Contrastive Objective — полная векторизация (SN-5)

- **Суть**: `_contrastive_objective_gpu` (stdp_trainer.py:629) содержит CPU-цикл для hard-negative mining (per-gen_cid topk через Python set). Узкое место: ~5ms/call.
- **Сложность**: 5
- **Ожидаемый прирост**: 10-15x ускорение contrastive

### AM-16 (P2): Fused Scatter-Add for Negative Sampling (G-12)

- **Суть**: В `_negative_sampling_gpu` два scatter_add (acc_shifts + acc_elr). Объединить в fused тензор (D+1), как уже сделано в STDP.
- **Сложность**: 3

### AM-17 (P2): Экспорт Gradient Noise и Momentum через public API

- **Суть**: `noise_scale` и `momentum_mu` есть в `_gpu_stdp_apply`, но не принимаются `train_from_text`/`train_batch` и не проброшены через crystal_generator → train_full.py.
- **Сложность**: 2

### AM-18 (P2): Устранение дублирования concept_error EMA

- **Суть**: Concept error EMA обновляется в двух местах (crystal_generator.`_gpu_stdp_apply`:787-795 + STDPTrainer.`_gpu_stdp_apply`:373-386). После удаления dead code останется только STDPTrainer.
- **Сложность**: 2

### AM-20 (P2): Async CPU→GPU Pipeline (G-11)

- **Суть**: `_AsyncPipeline` (crystal_generator.py:1185) — пустой стаб, вызывает `train_from_text` синхронно. Нет ни `torch.cuda.Stream`, ни реальной асинхронности.
- **Сложность**: 6
- **Ожидаемый прирост**: 2x throughput CPU+GPU

### AM-23 (P2): Сборка мусора CUDA кэша

- **Суть**: `torch.cuda.empty_cache()` вызывается только при OOM. На 2GB GPU с FP16 `_vecs_t` (~107MB) + FP32 `_ema_vecs_t` (~215MB) + временные тензоры — стоит логировать alloc_mb в чекпоинтах.
- **Сложность**: 1

---

# 2. Neuro-Symbolic Specialist: Аудит V6

## Баги, найденные при рефакторинге

### SN-B1 (P1): EMA update потерян при рефакторинге

- **Файл**: `stdp_trainer.py:322-463` (весь метод `_gpu_stdp_apply`)
- **Суть**: Старая CG версия `_gpu_stdp_apply` (crystal_generator.py:837-845) обновляла `_ema_vecs_t`. Новая версия в STDPTrainer **не содержит EMA-обновления**. EMA — мёртвый груз: создаётся, но никогда не обновляется → никогда не используется.
- **Приоритет**: **P1**

### SN-B2 (P1): `_negative_sampling_cpu` — поведенчески изменена

- **Файл**: `stdp_trainer.py:497-526`
- **Суть**: Старая CG:1087-1116 делает per-pair negative sampling с field_gate и concept_error. Новая STDPTrainer — per-gen_cid с `avg_elr * 0.3` и **без** field_gate/concept_error. CPU-тренировка даёт другие результаты, чем до рефакторинга.
- **Приоритет**: **P1**

### SN-B3 (P2): All-Cands dead code в `_negative_sampling_gpu`

- **Файл**: `stdp_trainer.py:544-549`
- **Суть**: `all_cands` вычисляется через цикл + `.stack()` + `.T`, но никогда не используется. `noise` (строка 544) используется напрямую.
- **Приоритет**: P2

### SN-B4 (P2): Двойной sync `_vecs_t` в STDPTrainer + hook

- **Файл**: `STDPTrainer._gpu_stdp_apply:452-453`
- **Суть**: STDPTrainer вручную обновляет `_vecs_t[gen_cid]` после `cs._apply_vector_update`, который через хук `_on_vector_update` (CG:152-155) делает то же самое. Лишняя копия на GPU.
- **Приоритет**: P2

### SN-B5 (P2): FP16 нормализация после записи теряет точность

- **Файл**: `crystal_generator.py:453`
- **Суть**: `_vecs_t` в FP16, математика через `.float()`. Нормализация после записи в FP16 теряет ~3 десятичных знака. Для 384D единичного вектора приемлемо, но накопление ошибок через много итераций может быть проблемой.
- **Приоритет**: P2

## Новые методы

### SN-9 (P1): Subspace-Kinetic STDP (Dual-Timescale)

- **Суть**: Применить разные learning rates к z_c (identity, ×0.01), z_a (attention, ×1.0), z_m (meta, самообучаемый). Сейчас 50% латентного кода имеет одинаковую plasticity rate.
- **Сложность**: 6
- **Обоснование**: z_c (256/512) должен сохранять identity (embedding lookup), z_a адаптироваться (attention), z_m meta-learn свою пластичность.

### SN-10 (P2): Cross-Modal Field Alignment

- **Суть**: Пары с одинаковым октантным префиксом — принудительно повышать сходство их z_a-компонент. Свяжет нейро-символическую структуру с вниманием.
- **Сложность**: 5

### SN-11 (P2): AIM (Attention-In-Meta) Gate

- **Суть**: Использовать норму z_meta как адаптивный вентиль learning rate: `lr *= sigmoid(norm(z_m[cid]))` — каждый концепт получает meta-learned темп.
- **Сложность**: 4

### SN-12 (P1): EMA Sync for Evaluation

- **Суть**: Перед evaluate/generate, если `_ema_steps > 100`, скопировать `_ema_vecs_t` как frozen eval representation. Без этого EMA — мёртвый код.
- **Сложность**: 2

### SN-13 (P1): GPU Contrastive Objective — полная векторизация

- **Суть**: Матрица сходств g_vecs @ all_vecs.T, маска cooc_set через бинарную матрицу, hard negative mining через `torch.where` + scatter. Полностью на GPU.
- **Сложность**: 7

### SN-14 (P2): Adaptive Destab per Linear Schedule

- **Суть**: Destab с per-concept модуляцией через concept_error и линейным расписанием от `destab_scale_start` до `destab_scale_end`. Параметры конфига уже есть, но не подключены к STDPTrainer.
- **Сложность**: 3

---

# 3. GPU-Opt Agent: Аудит V6

## Статус V5 G-*

| G | Название | Статус V6 |
|:-:|----------|-----------|
| G-9 | FP16 storage | ✅ Корректно (dtype=torch.float16, чтение через .float()) |
| G-10 | Pre-allocated buffers | ⚠️ **Частично** — `_build_torch_tensors` ✅, но **5 мест без .copy_()** |
| G-11 | Async CPU→GPU Pipeline | ❌ Не реализован |
| G-12 | Fused scatter | ✅ Реализован в STDPTrainer |
| G-13 | Chunked Full-V Matmul | ⚠️ Lateral inhibition теперь n×n (лучше), старый V×V код мёртв |
| G-14 | Sparse Approx Inhibition | ❌ Не реализован |
| G-15 | CUDA Events profiling | ⚠️ Реализован, но events пересоздаются каждый вызов |
| G-16 | GPU Concept Error EMA | ✅ Реализован (in-place _ce_t) |
| G-17 | Kernel Fusion | ❌ Не реализован |
| G-18 | CUDAGraphs evaluate | ❌ Не реализован |
| G-19 | WARP Shuffle | ❌ Не реализован |
| G-20 | Torch.compile | ❌ Не реализован |

## Баги

### G-10.1 (P1): `_on_vector_update` — тройная аллокация

- **Файл**: `crystal_generator.py:155`
- **Суть**: `self._vecs_t[cid] = torch.from_numpy(v_new.astype(np.float32)).to(self._vecs_t.device).to(self._vecs_t.dtype)` — 3 аллокации (astype, .to(device), .to(dtype)). Фикс:
  ```python
  v_t = torch.from_numpy(v_new.astype(np.float32)).to(self._vecs_t.device, non_blocking=True)
  self._vecs_t[cid].copy_(v_t)
  ```
- **Приоритет**: **P1**

### G-10.2 (P1): 4 места в stdp_trainer.py без .copy_()

- **Файлы**: `stdp_trainer.py:453, 491, 578, 692`
- **Суть**: Все vector row-апдейты через `gen._vecs_t[gen_cid] = tensor.to(...)` вместо `.copy_()`. Создают новые GPU тензоры на каждой итерации.
- **Приоритет**: **P1**

### N-1 (P1): `_use_torch` не определён

- Дублирует AM-14. `AttributeError` при `use_torch=None` (дефолт).
- **Приоритет**: **P1**

### N-4 (P1): Gradient formula inconsistency в `_negative_sampling_gpu`

- **Файл**: `stdp_trainer.py:566`
- **Суть**: `grad = (v_neg - sim * v_self * v_self).mean(dim=0)` — лишнее умножение на `v_self`. Правильная формула: `v_neg - sim * v_self` (как в CPU версии строка 518).
- **Приоритет**: **P1** — потенциально неверный градиент

### N-2 (P2): FP16→FP32 конверсия при каждом rebuild EMA

- **Файл**: `crystal_generator.py:225-228`
- **Суть**: `_vecs_t.float()` создаёт полный FP32 тензор V×D (146K×384×4 = ~225MB) при каждом rebuild.
- **Фикс**: conditional — только если `_torch_dirty`.

### N-3 (P2): `all_cands` dead code

- Дублирует SN-B3.

### N-5 (P2): Разная семантика old vs new `_negative_sampling_gpu`

- Old CG:998 — push-away на neg_cid (все пары)
- New STDP:528 — push-away на unique_gen
- Разные алгоритмы, но вызывается STDP версия.

### N-6 (P2): EMA CPU roundtrip в evaluate

- **Файл**: `stdp_trainer.py:842-845`
- **Суть**: `torch.from_numpy(np.array(gen_vecs, dtype=np.float32)).to(device)` — CPU roundtrip. gen_vecs уже на GPU.

---

# 4. Training-Dynamics Agent: Аудит V6

## Статус V5 TN-*

| TN | Название | Статус V6 |
|:--:|----------|-----------|
| TN-1 | Self-Paced Learning | ⚠️ **Реализован, но НЕ РАБОТАЕТ (P0)** |
| TN-2 | EMA Vectors | ⚠️ **Создаётся, но не обновляется (P1)** |
| TN-3 | Cosine Annealing | ✅ Корректно |
| TN-4 | Early Stopping | ⚠️ **Только в мёртвом TrainingPipeline (P1)** |
| TN-5 | Batch Size Warmup | ✅ Корректно |
| TN-6 | Gradient Noise | ⚠️ **Не подключен (noise_scale=0 всегда, P2)** |
| TN-7 | TensorBoard | ❌ Не реализован |
| TN-8 | Adaptive Destab | ✅ Per-CID через concept_error |
| TN-9 | Switched Eval | ❌ Параметры есть, логика отсутствует |
| TN-10 | Decay Protection | ✅ `rare_concept_protect=True` |

## Критические баги

### T-B1 (P0): Self-Paced Learning не влияет на обучение

- **Файл**: `train_full.py:350-361, 504`
- **Суть**: `epoch_train = _rescore_lines(epoch_train[remaining:], gen)` переписывает локальную переменную, но цикл `for idx, line in enumerate(epoch_train[start_line:], ...)` уже зафиксировал итератор при входе в цикл. Rescore никогда не применяется.
- **Фикс**: Прерывать текущую эпоху, начинать новую с переранжированным списком.

### T-B2 (P0): TrainingPipeline — 150 строк мёртвого кода

- **Файл**: `train_full.py:366-511`
- **Суть**: Класс определён, но не инстанциируется. Основной цикл (стр. 685-884) содержит прямую копию логики. В результате:
  - TN-4 (Early Stopping) не работает
  - AM-7 (Async Checkpoint Manager) не используется
  - Дублирование ~100 строк кода
- **Фикс**: Активировать TrainingPipeline (AM-15) или удалить как dead code.

### T-B3 (P1): EMA не обновляется в STDPTrainer

- Дублирует SN-B1. `_ema_vecs_t` — мёртвый код: никогда не обновляется, никогда не используется.

### T-B4 (P1): Destab-decay-lines дублирование в конфиге

- **Файл**: `fcf_config.py:405`
- **Суть**: `destab_decay_lines` есть как поле FCFConfig И как `ParamDef` в списке params. Поле FCFConfig не используется — код берёт `opt.p['destab_decay_lines'].current`. Вводит в заблуждение.

---

# 5. Quality-Safety Agent: Аудит V6

## Статус QN-* из V5

| QN | Название | Статус V6 |
|:--:|----------|-----------|
| QN-6 | OOM Stress (monkeypatch) | ✅ Реализован |
| QN-7 | _branch fuzz | ✅ Реализован |
| QN-8 | generate property tests | ⚠️ 2 skipped (требуют SentencePiece) |
| QN-9 | Save/Load roundtrip | ✅ Реализован |
| QN-10 | octree_fields correctness | ✅ Реализован |
| QN-11 | HormonalSystem (6 tests) | ✅ Реализован |
| QN-12 | GPU/CPU parity tightened | ✅ Реализован |
| QN-14 | _apply_vector_update fuzz | ✅ Реализован |
| QN-15 | FractalEncoding boundary | ✅ Реализован |
| QN-13 | Memory Stress | ❌ Не реализован |

## Пробелы тестового покрытия

### Q-B1 (P1): STDPTrainer — 798 строк без dedicated тестов

- **Файл**: `stdp_trainer.py`
- **Суть**: Все ключевые методы (`_gpu_stdp_apply`, `_negative_sampling_gpu`, `_build_pairs`, `train_from_text`, `train_batch`, `evaluate`) не имеют прямых тестов. Косвенное покрытие через CrystalGenerator делегацию — минимально.
- **Риск**: Любая регрессия в STDP, negative sampling, contrastive, evaluate незаметна до продакшн-рана.
- **Предложение**: Минимум 4 теста: (1) `train_from_text` smoke, (2) `_gpu_stdp_apply` smoke, (3) `_negative_sampling_gpu` smoke, (4) `evaluate` smoke.

### Q-B2 (P1): CheckpointManager — 119 строк, 0% coverage

- **Файл**: `checkpoint_manager.py`
- **Суть**: Async save, cleanup, recovery от сбоев не тестированы. Поломка чекпоинтов останется незамеченной до потери данных.
- **Предложение**: 3 теста: (1) save+wait целостность, (2) cleanup по лимиту, (3) recovery после сбоя записи.

### Q-B3 (P2): RNGRegistry — 47 строк, 0% coverage

- **Файл**: `rng_registry.py`
- **Суть**: Детерминизм сидирования, reset_all, изоляция имён не проверены.
- **Предложение**: 2 теста: (1) get('a') даёт одинаковые последовательности при одинаковом seed, (2) reset_all сбрасывает состояние.

### Q-B4 (P2): AdaptiveErrorTracker — минимальное покрытие

- **Файл**: `adaptive_error_tracker.py`
- **Суть**: Только косвенное покрытие через concept_error FIFO test. Нет прямых unit-тестов: update, get, EMA decay, FIFO eviction, copy.
- **Предложение**: 3 теста: (1) EMA decay корректность, (2) FIFO eviction по max_size, (3) copy.

## Анализ покрытия по модулям

| Модуль | Строк | Покрытие | Статус |
|--------|------|----------|--------|
| `crystal_generator.py` | ~1245 | ~30% | ⚠️ |
| `stdp_trainer.py` | 798 | ~1% | ❌ **P1** |
| `concept_space.py` | ~851 | ~30% | ⚠️ |
| `checkpoint_manager.py` | 119 | 0% | ❌ **P1** |
| `adaptive_error_tracker.py` | 71 | ~5% | ❌ P2 |
| `rng_registry.py` | 47 | 0% | ❌ P2 |
| `parameter_optimizer.py` | ~353 | ~90% | ✅ |
| `fcf_config.py` | 519 | ~80% | ✅ |
| `train_full.py` | 921 | N/A | (CLI) |

---

# 6. Матрица приоритетов V6

## Критические (P0)

| ID | Проблема | Агент | Сложность |
|:--:|----------|:-----:|:---------:|
| **T-B1** | Self-Paced Learning не работает (захват итератора) | TD | 1 |
| **T-B2** | TrainingPipeline — 150 строк мёртвого кода | TD | 3 |

## Высокие (P1)

| ID | Проблема | Агент | Сложность |
|:--:|----------|:-----:|:---------:|
| AM-14 | `_use_torch` не определён → AttributeError | Arch | 1 |
| AM-21 | `checkpoint_keep` → `cleanup_keep` typo | Arch | 1 |
| AM-22 | `total_freq` → AttributeError | Arch | 1 |
| AM-13 | Dead code: 7 дублирующих методов (~400 строк) | Arch | 2 |
| AM-15 | Активировать TrainingPipeline | Arch | 5 |
| AM-19 | GPU Contrastive Objective — векторизация | Arch | 5 |
| SN-B1 | EMA update потерян при рефакторинге | NS | 2 |
| SN-B2 | `_negative_sampling_cpu` — изменено поведение | NS | 3 |
| SN-9 | Subspace-Kinetic STDP (Dual-Timescale) | NS | 6 |
| SN-12 | EMA Sync for Evaluation | NS | 2 |
| SN-13 | GPU Contrastive — полная векторизация | NS | 7 |
| G-10.1 | `_on_vector_update` — тройная аллокация | GPU | 1 |
| G-10.2 | 4 места в stdp_trainer без `.copy_()` | GPU | 2 |
| N-1 | `_use_torch` (дубль AM-14) | GPU | 1 |
| N-4 | Gradient formula bug в neg sampling | GPU | 3 |
| T-B3 | EMA не обновляется (дубль SN-B1) | TD | 2 |
| T-B4 | Destab-decay-lines дублирование config | TD | 1 |
| Q-B1 | STDPTrainer — 798 строк без тестов | QA | 4 |
| Q-B2 | CheckpointManager — 0% coverage | QA | 3 |

## Средние (P2, 16 задач)

AM-16 (Fused scatter, сложность 3), AM-17 (Noise/Momentum API, 2), AM-18 (Dedup concept_error, 2), AM-20 (Async Pipeline, 6), AM-23 (CUDA GC, 1),
SN-10 (Field Alignment, 5), SN-11 (AIM Gate, 4), SN-14 (Linear Destab, 3), SN-B3 (all_cands dead code, 1), SN-B4 (Double sync, 1), SN-B5 (FP16 precision, 2),
N-2 (EMA rebuild alloc, 2), N-3 (all_cands — дубль), N-5 (semantics old/new, 2), N-6 (EMA CPU roundtrip, 2),
Q-B3 (RNGRegistry tests, 2), Q-B4 (AdaptiveErrorTracker tests, 2)

## Низкие (P3 — 2 задачи)

TN-7 (TensorBoard), QN-13 (Memory Stress)

---

# 7. Общая картина V6

### Что реализовано с V5:
- ✅ FP16 storage (G-9)
- ✅ Pre-allocated buffers (G-10, частично)
- ✅ Fused scatter (G-12)
- ✅ GPU Concept Error EMA (G-16)
- ✅ EMA vectors — структура (TN-2)
- ✅ Cosine Annealing (TN-3)
- ✅ Batch Size Warmup (TN-5)
- ✅ Adaptive Destab (TN-8)
- ✅ Decay Protection (TN-10)
- ✅ Self-Paced Learning — структура (TN-1, но не работает)
- ✅ Early Stopping — структура (TN-4, но мёртвый код)
- ✅ Generator-Trainer Separation (AM-6)
- ✅ Async Checkpoint Manager (AM-7)
- ✅ TrainingPipeline — структура (AM-9, но мёртвый код)
- ✅ AdaptiveErrorTracker (AM-10)
- ✅ Lazy _fb_t (AM-11)
- ✅ RNGRegistry (AM-12)
- ✅ Config Schema Validation (AM-8)
- ✅ 56 тестов (54 passed, 2 skipped)

### Ключевые проблемы V6:
1. **3 P1 crash-бага**: `_use_torch`, `checkpoint_keep`, `total_freq` (AM-14, AM-21, AM-22)
2. **2 P0 логических**: Self-Paced не работает, TrainingPipeline мёртв (T-B1, T-B2)
3. **EMA — полностью мёртвый код**: создаётся, но не обновляется, не используется (SN-B1, T-B3)
4. **400 строк dead code** в crystal_generator.py (AM-13)
5. **5 мест без `.copy_()`** — лишние GPU аллокации (G-10.1, G-10.2)
6. **Gradient formula bug** в `_negative_sampling_gpu` (N-4)
7. **2 модуля с 0% тестов**: CheckpointManager, RNGRegistry (Q-B2, Q-B3)
8. **STDPTrainer**: 798 строк, ~1% тестов (Q-B1)

---

*Отчёт составлен коллегией AI-агентов: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent*
