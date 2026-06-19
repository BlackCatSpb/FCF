# FCF Architect Report V7 — Коллегия AI-агентов (Улучшения и новые методы)

**Дата**: 2026-06-17
**Проект**: Fractal Cognitive Field (FCF) — нейро-символическая языковая модель
**Версия отчёта**: V7 (после V6, фокус на улучшениях и новых методах в парадигме существующей архитектуры)
**Состав коллегии**: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent

---

## Сводка

| Агент | Новых методов | P0 | P1 | P2 | P3 |
|-------|:-------------:|:--:|:--:|:--:|:--:|
| Architect-AI | 14 (AM-24..37) | — | 5 | 6 | 3 |
| Neuro-Symbolic Specialist | 7 (SN-15..21) + 5 улучшений | — | 4 | 2 | 1 |
| GPU-Opt Agent | 10 (G-21..30) + 6 багов подтверждено | — | 2 | 7 | 1 |
| Training-Dynamics Agent | 5 (TN-11..15) + 4 бага подтверждено | 3 | 2 | 2 | 1 |
| Quality-Safety Agent | 8 (QN-16..23) + анализ покрытия | — | 2 | 5 | 1 |
| **Итого** | **44 новых** + **19 багов/проблем** | **3** | **15** | **22** | **7** |

---

## 1. Architect-AI: Архитектурные улучшения

### Статус AM-13..23 из V6

| ID | Предложение | Статус |
|:--:|-------------|--------|
| AM-13 | Dead STDP-методы в crystal_generator.py (9 forwarding, ~400 строк) | ❌ Не исправлен |
| AM-14 | `_use_torch` не определён → AttributeError | ❌ **Баг (P1)** |
| AM-15 | TrainingPipeline мёртвый (не вызывается) | ⚠️ Частично |
| AM-16 | Fused scatter | ✅ Реализован |
| AM-17 | Noise API через public API | ✅ Реализован |
| AM-18 | GPU dedup concept_error | ✅ Реализован |
| AM-19 | GPU Contrastive Objective — CPU цикл | ❌ Не исправлен |
| AM-20 | Async pipeline — пустой стаб | ❌ Stub |
| AM-21 | `checkpoint_keep` → `cleanup_keep` typo | ✅ Исправлен |
| AM-22 | `total_freq` — несуществующий атрибут | ❌ **Баг (P1)** |
| AM-23 | CUDA GC (только при OOM) | ⚠️ Неполно |

### AM-24 (P1): Удалить STDP forwarding-методы

- **Суть**: `crystal_generator.py:758–808` — 9 методов-прокладок, которые делегируют в STDPTrainer. Удалить; все вызовы через `gen._trainer.method()` напрямую. Экономия ~50 строк.
- **Сложность**: 1

### AM-25 (P2): Удалить CPU-путь STDP

- **Суть**: Дублирование CPU/GPU в 4 парах методов (~400 строк). При `use_torch=True` по умолчанию CPU-путь не нужен. Оставить только GPU.
- **Сложность**: 3

### AM-26 (P1): Полная векторизация GPU Contrastive Objective

- **Суть**: `stdp_trainer.py:663` — заменить Python-цикл на batched `scatter_add_` по аналогии с fused STDP. `push_total` уже аллоцирован, но не используется.
- **Сложность**: 4

### AM-27 (P1): Активировать или удалить TrainingPipeline

- **Суть**: Основной цикл train_full.py:706–826 дублирует `TrainingPipeline.run_epoch()`. Либо вызвать `run_epoch()`, либо удалить класс.
- **Сложность**: 3

### AM-28 (P2): Удалить `_AsyncPipeline`

- **Суть**: Мёртвый класс в crystal_generator.py:819–843. Никто не вызывает.
- **Сложность**: 1

### AM-29 (P2): Консолидировать RNG

- **Суть**: 7+ независимых RNG: `gen.main_rng`, `rng_registry`, `cs.rng`, `cs._inhibit_rng`, `cs.fractal._fluct_rng`, `branch_rngs`, `np.random.RandomState(...)`. RNGRegistry существует, но почти не используется.
- **Сложность**: 3

### AM-30 (P2): EMA update CPU-GPU sync

- **Суть**: `stdp_trainer.py:452-455` — EMA обновляется поэлементно в CPU-цикле, вызывая GPU sync на каждой итерации. Переписать на batched `gen._ema_vecs_t[unique_gen] = ...`.
- **Сложность**: 2

### AM-31 (P2): ConceptError bidirectional sync

- **Суть**: `stdp_trainer.py:378-385` — `_ce_t` обновляется на GPU, затем копируется на CPU через цикл. Нужен batch copy: `gen.concept_error.batch_update(unique_gen, avg_err_cpu)`.
- **Сложность**: 2

### AM-32 (P2): `_graph_cache` без эвикции

- **Суть**: `crystal_generator.py:71` — неограниченный рост. Добавить LRU-эвикцию (maxsize=1000).
- **Сложность**: 1

### AM-33 (P2): HormonalSystem.reset() между generate()

- **Суть**: Состояние гормонов переносится между вызовами генерации. Нужен `self.hormones.reset()` в начале `generate()`.
- **Сложность**: 1

### AM-34 (P3): Homeostatic cache magic number

- **Суть**: `concept_space.py:661` — `_hboost_cache_step % 1000` — вынести в конфиг.
- **Сложность**: 1

### AM-35 (P3): PathConfig → FCFConfig дублирование

- **Суть**: FCFConfig (lines 241–282) дублирует все `@property` из PathConfig. Использовать `cfg.paths.*` напрямую.
- **Сложность**: 2

### AM-36 (P3): Version counter для GPU-тензоров

- **Суть**: Флаги `_torch_dirty` / `_fb_dirty` — хрупкая система. Заменить на глобальный монотонный счётчик версий в ConceptSpace.
- **Сложность**: 3

### AM-37 (P3): Fractal subspace structure loss

- **Суть**: `_apply_vector_update` (concept_space.py:511) перезаписывает код через `v @ basis.T`, разрушая структуру `z_c | z_a | z_m`. Нужен subspace-preserving update.
- **Сложность**: 5

---

## 2. Neuro-Symbolic Specialist: Улучшения концептуального пространства

### Статус SN-9..14 из V6

| ID | Предложение | Статус |
|:--:|-------------|--------|
| SN-9 | Subspace-Kinetic STDP (Dual-Timescale) | ❌ Не реализован |
| SN-12 | EMA Sync for Evaluation | ⚠️ `_sync_ema`/`_restore_vectors` есть, но не вызываются |
| SN-13 | GPU Contrastive Objective | ⚠️ Существует, но не векторен — Python-цикл |
| SN-14 | Adaptive Destab per Linear Schedule | — |

### SN-B1 (P1): EMA update потерян при рефакторинге

- **Файл**: `stdp_trainer.py:452-455`
- **Суть**: EMA обновляется в STDPTrainer, но к этому моменту `_apply_vector_update` уже вызвал `_on_vector_update`, скопировав новый вектор в `_vecs_t`. Формула: `ema = decay * old_ema + (1-decay) * new_vec` — то есть EMA отслеживает почти то же, что и `_vecs_t`. **EMA должна обновляться ДО `_apply_vector_update`**, используя старый v.
- **Приоритет**: **P1**

### SN-B2 (P1): `_negative_sampling_cpu` — поведенчески изменена

- **Файл**: `stdp_trainer.py:498-531`
- **Суть**: Новая CPU версия не использует field_gate и concept_error. Отличается от старой CG:1087-1116.
- **Приоритет**: **P1**

### SN-15 (P1): Subspace-Kinetic STDP

- **Суть**: Текущая архитектура делает `_apply_vector_update` → перезапись кода через `v @ basis.T`. Subspace-LR теряется. Решение — проекция градиента на z_c, z_a, z_m через basis с разными learning rates.
  ```python
  code_grad = grad @ self.cs.fractal.basis.T
  mask_c = torch.zeros(latent_dim); mask_c[:l_c] = 1
  mask_a = torch.zeros(latent_dim); mask_a[l_c:l_c+l_a] = 1
  mask_m = torch.zeros(latent_dim); mask_m[l_c+l_a:] = 1
  lr_c, lr_a, lr_m = 0.01, 0.05, 0.001
  code_new = code + code_grad * (lr_c * mask_c + lr_a * mask_a + lr_m * mask_m)
  ```
- **Сложность**: 6

### SN-16 (P1): Field-Aware Contrastive Decoupling

- **Суть**: Contrastive objective должен работать по-разному для концептов из одного поля и из разных полей:
  - overlap > 0: push только если cos < 0.3 (мягкий)
  - overlap = 0: push если cos > 0.0 (жёсткий — разносим кластеры)
- **Сложность**: 4

### SN-17 (P2): Kinetic Energy Buffer per Subspace

- **Суть**: Трек кинетической энергии `E_c = ||Δz_c||²`, `E_a`, `E_m` отдельно. Если энергия падает ниже порога — увеличиваем lr (anti-freeze).
- **Сложность**: 3

### SN-18 (P1): EMA-Synced Evaluation Hook (завершение SN-12)

- **Суть**: `_sync_ema` имеет обратную семантику — копирует `_vecs_t` в `_ema_vecs_t` вместо `_ema_vecs_t → _vecs_t`. Перед evaluate: `self._vecs_t.copy_(self._ema_vecs_t.to(self._vecs_t.dtype))`.
- **Сложность**: 2

### SN-19 (P1): Fully Vectorized GPU Contrastive (замена SN-13)

- **Суть**: Batched self-similarity, маска (self + cooc + connected), hard negative mining векторизовано, field-weighted push, scatter-add как в G-12.
- **Сложность**: 7

### SN-20 (P2): Adaptive EMA Decay per Concept

- **Суть**: Для концептов с high concept_error → быстрый EMA (small decay: 0.949), для low error → медленный (0.999). `ema_decay = self._ema_decay * (1.0 - ce * 0.1)`.
- **Сложность**: 2

### SN-21 (P3): Riemannian STDP with Exponential Map

- **Суть**: Заменить евклидов шаг + нормализация на Riemannian SGD: проекция на касательную, экспоненциальное отображение. Нет потери norm accuracy, нет дрейфа.
  ```python
  Δ = grad - (v · grad) * v
  v_new = cos(||Δ||·lr) * v + sin(||Δ||·lr) * Δ/||Δ||
  ```
- **Сложность**: 8

### Дополнительные улучшения

- **Field gate в CPU negative sampling**: `_negative_sampling_cpu` не использует `field_gate` — добавить field overlap filter (P1).
- **Градиентный noise по subspace**: noise только в `z_a`, не трогать `z_c` (P2).

---

## 3. GPU-Opt Agent: GPU-оптимизация

### Статус V6 багов

| ID | Баг | Статус |
|:--:|-----|--------|
| G-10.1 | `_on_vector_update` тройная аллокация | ❌ Актуально |
| G-10.2 | 4 места в stdp_trainer без `.copy_()` | ❌ Актуально |
| N-1 | `_use_torch` не определён (CRITICAL) | ❌ **Актуально** — код падает на GPU |
| N-4 | Gradient formula bug в `_negative_sampling_gpu` | ❌ **Актуально** — 2 расхождения с CPU |
| N-5 | Разная семантика old/new `_negative_sampling_gpu` | ❌ Актуально |
| N-6 | EMA CPU roundtrip в evaluate + `_sync_ema` инвертирована | ❌ Актуально |
| G-15 | CUDA Events пересоздаются каждый вызов | ❌ Актуально |
| N-2 | FP16→FP32 EMA rebuild alloc | ❌ Актуально |

### N-4 детали (P1): Gradient bug в `_negative_sampling_gpu`

**3a. Пропущено concept_error reweighting** (stdp_trainer.py:556):
```python
# CPU (line 511-513):
ce = gen.concept_error.get(gen_cid, 0.0)
neg_lr *= (1.0 + ce * 2.0)
# GPU — concept_error не учтён
```

**3b. Неверная константа**: `neg_lr_ratio * 0.2` в GPU vs `neg_lr_ratio * 0.3` в CPU.

### G-21 (P2): Persistent CUDA events в `_build_torch_tensors`

- **Суть**: CUDA Events создаются в каждом `_gpu_stdp_apply` (stdp_trainer.py:331-334). Инициализировать один раз.
- **Сложность**: 1

### G-22 (P2): Векторизованный EMA update

- **Суть**: `_ema_vecs_t[gen_t] = ...` вместо per-concept loop.
- **Сложность**: 2

### G-23 (P2): Pre-allocated `ctx_t`/`tgt_t`/`meta_t`

- **Суть**: Переиспользовать тензоры между вызовами `_gpu_stdp_apply` вместо пересоздания.
- **Сложность**: 3

### G-24 (P2): Fused negative sampling — без CPU-лупа по unique_gen

- **Суть**: Весь `_negative_sampling_gpu` на GPU.
- **Сложность**: 5

### G-25 (P2): `lerp_` для EMA вместо `copy_` с временным

- **Суть**: `_ema_vecs_t[gen_cid].lerp_(gen._vecs_t[gen_cid].float(), 1 - decay)`.
- **Сложность**: 1

### G-26 (P3): Векторизованная lateral inhibition

- **Суть**: Весь цикл per-gen_cid → batched matrix ops.
- **Сложность**: 4

### G-27 (P3): Выделенный CUDA stream для async H2D

- **Суть**: Overlap CPU↔GPU через отдельный stream.
- **Сложность**: 5

### G-28 (P3): Переиспользование `elr_sum` из `_gpu_stdp_apply`

- **Суть**: Не пересоздавать `meta_t` — сохранять и переиспользовать.
- **Сложность**: 2

### G-29 (P3): Fused `norm + normalize` после градиентного шага

- **Суть**: Объединить нормализацию после apply в одну операцию.
- **Сложность**: 2

### G-30 (P3): `torch.compile` на `_gpu_stdp_apply`

- **Суть**: PyTorch ≥2.0 torch.compile для fusion.
- **Сложность**: 4

**Рекомендуемый порядок**: N-1 (CRITICAL) → G-10.1 → G-21 → G-24 → G-26 → G-30.

---

## 4. Training-Dynamics Agent: Улучшения цикла обучения

### Статус V6 багов

| ID | Баг | Статус |
|:--:|-----|--------|
| T-B1 (P0) | Self-Paced Learning — idx сбрасывается | ❌ **Подтверждён** — `idx = -1` после rescore, curriculum откатывается |
| T-B2 (P0) | TrainingPipeline — 150 строк мёртвого кода | ❌ **Подтверждён** — `run_epoch()` не вызывается |
| T-B3 (P1) | EMA не обновляется корректно | ❌ **Подтверждён** — `_sync_ema` инвертирована, EMA после apply |
| T-B4 (P1) | Destab-decay-lines дублирование | ✅ **Исправлен** — уже чисто |

### T-B1 детали (P0)

После `epoch_train = _rescore_lines(epoch_train[remaining:], gen)` на строке 463 идёт `idx = -1; start_line = -1`. На строке 466 `idx += 1` → `idx = 0`. Но `_curriculum_max_len(idx)`, `_curriculum_p(idx)` используют `idx` как глобальный счётчик — теперь он указывает на начало ресортированного суффикса. Куррикулум откатывается назад, batch size падает.

### T-B3 детали (P1)

`stdp_trainer.py:452-455`: `_apply_vector_update` (строка 450) уже вызвал `_on_vector_update`, скопировав `v_new` в `_vecs_t[gen_cid]`. То есть `_vecs_t[gen_cid]` уже содержит **новый** вектор. Формула `ema = decay * old_ema + (1-decay) * new_vec` — EMA отслеживает то же, что и `_vecs_t`. Нужно обновлять EMA ДО `_apply_vector_update`.

### TN-11 (P1): Gradient Accumulation (N-step STDP)

- **Суть**: Аккумулировать градиенты через `momentum_mu` > 0 по умолчанию (сейчас 0.0). Уже есть зачаток: `_mom_buf`, `momentum_mu`. Снижает шум на малых батчах.
- **Сложность**: 2

### TN-12 (P1): Switched Evaluation (реализация TN-9)

- **Суть**: Быстрый eval (64 строки, только vec_perplexity) каждый `eval_every_fast=1000`. Полный eval (300 строк, все метрики) каждый `eval_every_full=5000`. Параметры конфига уже есть (`eval_fast_lines=64`, `eval_full_lines=300`) — осталось добавить счётчик `_eval_cycle` и ветвление. Экономия ~4× на eval-вычислениях (~20% времени обучения).
- **Сложность**: 2

### TN-13 (P2): Progressive Batch Size with Plateaus

- **Суть**: При детекции плато увеличивать batch_size на +4 до 64. Даёт второй рывок после насыщения куррикулума.
- **Сложность**: 2

### TN-14 (P1): Field-Aware Contrastive Regularization

- **Суть**: В `_contrastive_objective_gpu` добавить штраф за высокое сходство между концептами с непересекающимися полями. Использовать `_fb_t` для проверки.
- **Сложность**: 3

### TN-15 (P2): Decay Warmup with Protect Threshold Ramp

- **Суть**: Линейно рамповать `rare_threshold` от 1 до 5 в первые 20K линий. В начале почти все концепты редкие — защита не мешает. Позже защита усиливается.
- **Сложность**: 2

### Резюме приоритетов обучения

1. **P0**: T-B1 (починить idx в `_rescore_lines`) + T-B2 (активировать/удалить TrainingPipeline)
2. **P0**: T-B3 (EMA до apply, а не после)
3. **P0**: Передать `noise_scale` в `train_batch` (TN-6)
4. **P1**: TN-12 (Switched Eval) — тривиально, ~20% экономии
5. **P1**: TN-11 (Gradient Accumulation) — поднять `momentum_mu > 0`
6. **P1**: TN-14 (Field-Aware Contrastive)
7. **P2**: TN-13 (Progressive BSize), TN-15 (Decay Warmup)

---

## 5. Quality-Safety Agent: Тестирование и качество

### Анализ покрытия V7

| Модуль | Строк | Покрытие | Статус |
|--------|-------|----------|--------|
| `stdp_trainer.py` | 797 | ~1% (4 smoke) | ❌ **P1** |
| `checkpoint_manager.py` | 119 | 0% | ❌ **P1** |
| `rng_registry.py` | 47 | 0% | ❌ **P2** |
| `adaptive_error_tracker.py` | 71 | ~5% (1 косвенный) | ❌ P2 |
| `crystal_generator.py` | 879 | ~30% (20 косвенных) | ⚠️ |
| `parameter_optimizer.py` | ~353 | ~90% | ✅ |
| `fcf_config.py` | 519 | ~80% | ✅ |

### QN-16 (P1): STDPTrainer Integration Tests

- **Суть**: 12 тестов для STDPTrainer — `test_build_pairs_basic`, `test_cpu_stdp_vector_update`, `test_cpu_stdp_lateral_inhibition`, `test_cpu_stdp_gradient_clipping`, `test_cpu_stdp_destab`, `test_negative_sampling_cpu_divergence`, `test_contrastive_objective_cpu_runs`, `test_centroid_pull_batch`, `test_train_from_text_short_input`, `test_contrastive_objective_gpu_runs`, `test_gpu_stdp_momentum`.
- **Оценка**: ~200 строк, покрытие ~40% STDPTrainer.

### QN-17 (P1): CheckpointManager Error Resilience

- **Суть**: 7 тестов — save roundtrip, cleanup, shutdown, save with opt, save with extras, failure cleanup, remove_tag.
- **Оценка**: ~100 строк, покрытие ~80%.

### QN-18 (P2): RNGRegistry Isolation Property

- **Суть**: 8 тестов — get caches, determinism, isolation, reset_all, reset_single, names, custom factory, master_seed change.
- **Оценка**: ~80 строк, покрытие ~90%.

### QN-19 (P2): AdaptiveErrorTracker FIFO Boundary

- **Суть**: 8 тестов — update EMA, convergence, FIFO eviction, dict interface, iteration, move_to_end, bool, repr.
- **Оценка**: ~80 строк, покрытие ~90%.

### QN-20 (P2): Lateral Inhibition Divergence

- **Суть**: 2 вектора с cos→1 → после inhibition cos ↓ и нормы = 1.
- **Сложность**: 3

### QN-21 (P2): Contrastive Hard Negative Selection

- **Суть**: hard negatives: cos ∈ (0.05, 0.5), no cooc, conn_strength ≤ 0.1.
- **Сложность**: 3

### QN-22 (P3): Centroid Pull Normalization

- **Суть**: n=5 → все сохраняют норму 1, cos к centroid ↑.
- **Сложность**: 2

### QN-23 (P3): Memory Stress (реализация QN-13)

- **Суть**: AdaptiveErrorTracker 100k items: FIFO O(1), update latency < 1ms.
- **Сложность**: 4

### Рекомендуемый порядок реализации тестов

```
Спринт 1: QN-16 (STDPTrainer) + QN-17 (CheckpointManager) → снимает P1
Спринт 2: QN-18 (RNGRegistry) + QN-19 (AdaptiveErrorTracker) → снимает P2
Спринт 3: QN-20, QN-21, QN-22 → поднимает покрытие до ~50%
Спринт 4: QN-23 → закрывает QN-13
```

После Спринтов 1-2: STDPTrainer ~40%, CheckpointManager ~80%, RNGRegistry ~90%, AdaptiveErrorTracker ~90%, общее ~50%.

---

## 6. Итоговая матрица приоритетов V7

### P0 (3 — критические баги)

| ID | Проблема | Агент | Сложность |
|:--:|----------|:-----:|:---------:|
| T-B1 | Self-Paced Learning — idx сбрасывается, curriculum откатывается | TD | 2 |
| T-B2 | TrainingPipeline — 150 строк мёртвого кода | TD | 3 |
| T-B3 | EMA обновляется после apply вместо до (инвертирована семантика) | TD/NS | 2 |

### P1 (15 — баги и критичные улучшения)

| ID | Проблема | Агент | Сложность |
|:--:|----------|:-----:|:---------:|
| AM-14 | `_use_torch` не определён → AttributeError | Arch | 1 |
| AM-22 | `gen.lattice.total_freq` → AttributeError | Arch | 1 |
| AM-24 | Удалить STDP forwarding-методы (~50 строк) | Arch | 1 |
| AM-26 | GPU Contrastive Objective — векторизация | Arch | 4 |
| AM-27 | Активировать/удалить TrainingPipeline | Arch | 3 |
| SN-B1 | EMA update после apply (инвертирована) — дубль T-B3 | NS | 2 |
| SN-B2 | `_negative_sampling_cpu` — изменено поведение | NS | 3 |
| SN-15 | Subspace-Kinetic STDP | NS | 6 |
| SN-16 | Field-Aware Contrastive Decoupling | NS | 4 |
| SN-18 | EMA-Synced Evaluation Hook | NS | 2 |
| SN-19 | Fully Vectorized GPU Contrastive | NS | 7 |
| N-1 | `_use_torch` — дубль AM-14 | GPU | 1 |
| N-4 | Gradient formula bug (concept_error пропущен, константа 0.2 vs 0.3) | GPU | 3 |
| TN-11 | Gradient Accumulation (momentum_mu > 0) | TD | 2 |
| TN-12 | Switched Evaluation | TD | 2 |
| TN-14 | Field-Aware Contrastive Regularization | TD | 3 |
| QN-16 | STDPTrainer тесты (12 шт) | QA | 4 |
| QN-17 | CheckpointManager тесты (7 шт) | QA | 3 |
| G-10.1 | `_on_vector_update` — тройная аллокация | GPU | 1 |

### P2 (22)

AM-25 (CPU path removal), AM-28 (AsyncPipeline stбыль), AM-29 (RNG consolidate), AM-30 (EMA sync), AM-31 (CE bidirectional), AM-32 (graph_cache eviction), AM-33 (hormones reset),
SN-17 (Kinetic Energy Buffer), SN-20 (Adaptive EMA Decay),
G-10.2 (4 places .copy_()), G-21..G-25 (CUDA events, vec EMA, pre-alloc, fused NS, lerp_),
N-2, N-5, N-6 (FP16 rebuild, semantics mismatch, EMA roundtrip),
TN-13 (Progressive BSize), TN-15 (Decay Warmup),
QN-18 (RNGRegistry tests), QN-19 (AET tests), QN-20, QN-21,
G-15 (CUDA Events init)

### P3 (7)

AM-34 (magic number), AM-35 (PathConfig dup), AM-36 (version counter), AM-37 (subspace loss),
SN-21 (Riemannian STDP),
G-26..G-30 (vec inhibition, CUDA stream, reuse, fuse norm, compile),
QN-22 (centroid pull), QN-23 (memory stress)

---

## 7. Рекомендуемый план работ

### Фаза 1 (P0 + P1 crashfixes) — 5 задач

1. AM-14: `self._use_torch = _HAS_TORCH and torch.cuda.is_available()` — 1 строка
2. AM-22: `total_freq = gen._get_total_freq()` — 1 строка
3. T-B1: Исправить `idx` после `_rescore_lines` — 3 строки
4. T-B3: Перенести EMA-обновление до `_apply_vector_update` — 2 строки
5. T-B2: Активировать TrainingPipeline вместо мёртвого цикла — рефакторинг

### Фаза 2 (P1 улучшения) — 10 задач

G-10.1, AM-24 (dead code), N-4 (gradient bug), SN-B2 (CPU neg sampling), TN-11 (momentum), TN-12 (switched eval), QN-16 (STDPTrainer tests), QN-17 (CheckpointManager tests)

### Фаза 3 (P2) — 22 задачи

### Фаза 4 (P3) — 7 задач

---

*Отчёт составлен коллегией AI-агентов: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent*
