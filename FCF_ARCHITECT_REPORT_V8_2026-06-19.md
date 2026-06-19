# FCF Architect Report V8 — V7→V8 Аудит изменений и новые проблемы

**Дата**: 2026-06-19
**Проект**: Fractal Cognitive Field (FCF)
**Версия отчёта**: V8 (post-V7 аудит незакоммиченных изменений)
**Состав коллегии**: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent

---

## Сводка изменений (working tree vs V7 HEAD)

| Файл | V7 (строк) | V8 (строк) | Δ |
|------|:----------:|:----------:|:-:|
| `crystal_generator.py` | ~1245 | 817 | **−428** |
| `stdp_trainer.py` | ~798 | 860 | **+62** |
| `train_full.py` | ~921 | 867 | **−54** |
| `tests/test_stdp.py` | ~522 | 809 | **+287** |
| `checkpoint_manager.py` | ~119 | 114 | −5 |

---

## 1. Статус AM-13..AM-37 из V7

### Критические баги (P0)

| ID | Баг | Статус | Детали |
|:--:|-----|:------:|--------|
| **T-B1** | Self-Paced Learning — idx сбрасывается | ⚠️ **Частично** | Исправлен внутри `TrainingPipeline.run_epoch()` (строки 521-526), но основной while-цикл (строка 710) вызывает `pipeline._checkpoint()` на строке 820 без захвата возвращаемого значения `(idx, start_line, epoch_train)`. Исправление не работает в активном path. **Регрессия: `idx` не сбрасывается, но `_rescore_lines` получает неверный срез.** |
| **T-B2** | TrainingPipeline — мёртвый код | ✅ **Исправлен** | Pipeline создаётся на строке 684, вызывается на строке 820. |
| **T-B3** | EMA после apply → до apply | ✅ **Исправлен** | `stdp_trainer.py:467-470` — EMA обновляется ДО `_apply_vector_update`. |

### Архитектурные улучшения (AM-14..AM-37)

| ID | Предложение | Приоритет | Статус | Детали |
|:--:|-------------|:---------:|:------:|--------|
| AM-13 | Dead STDP forwarding methods | P1 | ✅ **Исправлен** | 9 методов-прокладок удалены. Остались 3 публичных wrapper (`train_from_text`, `train_batch`, `evaluate`). |
| AM-14 | `_use_torch` не определён | P1 | ✅ **Исправлен** | `crystal_generator.py:101` — `self._use_torch = _HAS_TORCH and torch.cuda.is_available()` |
| AM-15 | TrainingPipeline мёртвый | P1 | ✅ **Исправлен** | Создаётся и используется. |
| AM-16 | Fused scatter | P1 | ✅ **Исправлен** (V6) | |
| AM-17 | Noise API | P1 | ✅ **Исправлен** (V6) | |
| AM-18 | GPU dedup concept_error | P1 | ✅ **Исправлен** (V6) | |
| AM-19 | GPU Contrastive Objective — CPU цикл | P1 | ❌ **Не исправлен** | `_contrastive_objective_gpu` (строка 656) содержит `for i in range(ng)` с вложенными циклами, `.item()`, mix numpy/torch. |
| AM-20 | Async pipeline — пустой стаб | P2 | ✅ **Исправлен** | `_AsyncPipeline` удалён. |
| AM-21 | `checkpoint_keep` → `cleanup_keep` typo | P2 | ✅ **Исправлен** | |
| AM-22 | `total_freq` — несуществующий атрибут | P1 | ✅ **Исправлен** | `_get_total_freq()` определён в crystal_generator.py:130-133. |
| AM-23 | CUDA GC only at OOM | P2 | ✅ **Исправлен** | OOM fallback в `_ensure_torch`. |
| AM-24 | Удалить STDP forwarding (~50 строк) | P1 | ✅ **Исправлен** | 7 forwarding-методов удалены. |
| AM-25 | Удалить CPU-путь STDP | P2 | ❌ **Не исправлен** | `_cpu_stdp_apply`, `_lateral_inhibition_cpu`, `_negative_sampling_cpu` всё ещё существуют (~300 строк). |
| AM-26 | Полная векторизация GPU Contrastive | P1 | ❌ **Не исправлен** | См. AM-19. Python-циклы по-прежнему доминируют. `push_total` аллоцирован (строка 690) но не используется. |
| AM-27 | Активировать/удалить TrainingPipeline | P1 | ✅ **Исправлен** | |
| AM-28 | Удалить `_AsyncPipeline` | P2 | ✅ **Исправлен** | |
| AM-29 | Консолидировать RNG | P2 | ❌ **Не исправлен** | 7+ RNG: `gen.main_rng`, `rng_registry`, `cs.rng`, `cs._inhibit_rng`, `cs.fractal._fluct_rng`, `branch_rngs`, `np.random.RandomState(...)`. RNGRegistry создан (crystal_generator.py:84) но используется ТОЛЬКО как контейнер — ни один метод не вызывает `rng_registry.get('name')`. |
| AM-30 | EMA update CPU-GPU sync | P2 | ⚠️ **Частично** | EMA перенесена ДО apply (P1 fix). Но всё ещё per-concept loop вместо `gen._ema_vecs_t[unique_gen] = ...`. |
| AM-31 | ConceptError bidirectional sync | P2 | ❌ **Не исправлен** | `stdp_trainer.py:400-401` — per-concept loop `for gi, gen_cid in enumerate(unique_gen): gen.concept_error.update(...)`. Нет batch copy. |
| AM-32 | `_graph_cache` без эвикции | P2 | ❌ **Не исправлен** | `crystal_generator.py:71` — `self._graph_cache = {}`. Неограниченный рост (ключи = все уникальные контекстные tuple). На 146K токенов с context_window=3 → ~10⁹ комбинаций. |
| AM-33 | HormonalSystem.reset() между generate() | P2 | ❌ **Не исправлен** | Состояние гормонов переносится между вызовами. Нет `self.hormones.reset()`. |
| AM-34 | Homeostatic cache magic number | P3 | ❌ **Не исправлен** | `concept_space.py:661` — `% 1000` хардкод. |
| AM-35 | PathConfig → FCFConfig дублирование | P3 | ❌ **Не исправлен** | FCFConfig:240-282 дублирует все `@property` из PathConfig. 11 пар методов с идентичным телом. |
| AM-36 | Version counter для GPU-тензоров | P3 | ❌ **Не исправлен** | `_torch_dirty` / `_fb_dirty` остаются. |
| AM-37 | Subspace structure loss | P3 | ❌ **Не исправлен + Критическая регрессия** | См. раздел 4 (AM-37 Regression). |

---

## 2. Статус SN-9..SN-21 из V7

| ID | Предложение | Приоритет | Статус | Детали |
|:--:|-------------|:---------:|:------:|--------|
| SN-B1 | EMA update потерян при рефакторинге | P1 | ✅ **Исправлен** | EMA до apply. |
| SN-B2 | `_negative_sampling_cpu` поведенчески изменена | P1 | ✅ **Исправлен** | Field-gate guard + concept_error reweighting добавлены. |
| SN-9 | Subspace-Kinetic STDP (Dual-Timescale) | P1 | ❌ **Не реализован** | |
| SN-12 | EMA Sync for Evaluation | P2 | ✅ **Исправлен** | `_sync_ema`/`_restore_vectors` существуют. |
| SN-13 | GPU Contrastive Objective | P1 | ❌ **Не векторен** | Python-цикл, см. AM-19/AM-26. |
| SN-14 | Adaptive Destab per Linear Schedule | — | ⚠️ **Неполно** | Существует destab ramp через `destab_pct`, но без per-concept адаптации. |
| SN-15 | Subspace-Kinetic STDP | P1 | ⚠️ **Внедрён, но сломан** | `_subspace_update()` (stdp_trainer.py:143-155) реализован, но `_apply_vector_update` (concept_space.py:544-549) перезаписывает код через `v_new @ basis.T`, уничтожая subspace-LR. **НОЛЬ-ЭФФЕКТ.** |
| SN-16 | Field-Aware Contrastive Decoupling | P1 | ✅ **Исправлен** | CPU + GPU. |
| SN-17 | Kinetic Energy Buffer per Subspace | P2 | ❌ **Не исправлен** | |
| SN-18 | EMA-Synced Evaluation Hook | P1 | ✅ **Исправлен** | `_sync_ema`: правильная семантика (`_ema_vecs_t → _vecs_t`). |
| SN-19 | Fully Vectorized GPU Contrastive | P1 | ❌ **Не исправлен** | См. AM-26. |
| SN-20 | Adaptive EMA Decay per Concept | P2 | ❌ **Не исправлен** | |
| SN-21 | Riemannian STDP with Exponential Map | P3 | ❌ **Не исправлен** | |

---

## 3. Статус GPU-Opt (G-21..G-30) и багов (N-1..N-6)

| ID | Баг/Улучшение | Приоритет | Статус | Детали |
|:--:|---------------|:---------:|:------:|--------|
| N-1/G-10.1 | `_use_torch` не определён | P1 | ✅ **Исправлен** | |
| N-2 | FP16→FP32 EMA rebuild alloc | P2 | ❌ **Актуально** | `_ema_vecs_t = self._vecs_t.float().clone()` на каждой перестройке (crystal_generator.py:227). |
| N-4 | Gradient bug — concept_error + const | P1 | ✅ **Исправлен** | `neg_lr_i *= (1.0 + ce * 2.0)` в GPU. Константа 0.3 совпадает с CPU. |
| N-5 | Разная семантика old/new GPU negative sampling | P2 | ⚠️ **Частично** | GPU negative sampling переписан (векторизованная загрузка noise), но всё ещё per-concept loop. |
| N-6 | EMA CPU roundtrip в evaluate | P2 | ✅ **Исправлен** | `_sync_ema` правильная семантика. |
| G-15 | CUDA Events пересоздаются каждый вызов | P2 | ❌ **Актуально** | `stdp_trainer.py:348-350`: новые Event на каждый вызов. |
| G-21 | Persistent CUDA events | P2 | ❌ **Не исправлен** | |
| G-22 | Векторизованный EMA update | P2 | ❌ **Не исправлен** | Per-concept loop. |
| G-23 | Pre-allocated `ctx_t`/`tgt_t`/`meta_t` | P2 | ❌ **Не исправлен** | `torch.tensor(gpu_ctx_l, ...)` каждый вызов. |
| G-24 | Fused negative sampling (без CPU-лупа) | P2 | ❌ **Не исправлен** | Per-concept loop + `.item()` + `.cpu().numpy()`. |
| G-25 | `lerp_` для EMA | P2 | ❌ **Не исправлен** | Формула вручную. |
| G-26 | Векторизованная lateral inhibition | P3 | ❌ **Не исправлен** | `_lateral_inhibition_gpu` — per-concept loop `for gi in range(n)`. |
| G-27 | Выделенный CUDA stream для H2D | P3 | ❌ **Не исправлен** | Все операции на default stream. |
| G-28 | Переиспользование `elr_sum` | P3 | ❌ **Не исправлен** | `meta_t` пересоздаётся каждый вызов. |
| G-29 | Fused norm + normalize | P3 | ❌ **Не исправлен** | |
| G-30 | `torch.compile` на `_gpu_stdp_apply` | P3 | ❌ **Не исправлен** | |

---

## 4. Статус TN-11..TN-15 и багов (T-B1..T-B4)

| ID | Улучшение | Приоритет | Статус | Детали |
|:--:|-----------|:---------:|:------:|--------|
| T-B4 | Destab-decay-lines дублирование | P2 | ✅ **Исправлен** (V7) | |
| TN-11 | Gradient Accumulation (momentum_mu) | P1 | ✅ **Исправлен** | `momentum_mu=0.9` передаётся, `_mom_buf` реализован. **НО**: CPU-путь не использует momentum, только GPU. |
| TN-12 | Switched Evaluation | P1 | ✅ **Исправлен** | `_eval_count` + ветвление fast/full eval. |
| TN-13 | Progressive Batch Size (plateaus) | P2 | ❌ **Не исправлен** | Только линейная рампа `batch_size_start→batch_size_end` через curriculum. Нет plateau-детекции. |
| TN-14 | Field-Aware Contrastive Regularization | P1 | ✅ **Исправлен** | `stdp_trainer.py:716-741` — cross-field penalty с `reg_lam=0.05`. |
| TN-15 | Decay Warmup with Protect Threshold Ramp | P2 | ❌ **Не исправлен** | `rare_threshold=3` хардкод. |


## 5. Статус Quality-Safety (QN-16..QN-23)

| ID | Улучшение | Приоритет | Статус | Детали |
|:--:|-----------|:---------:|:------:|--------|
| QN-16 | STDPTrainer Integration Tests (12) | P1 | ⚠️ **Частично (9/12)** | 9 тестов в `TestSTDPIntegration`. Отсутствуют: `test_contrastive_objective_gpu_runs` (только CPU test), `test_cpu_stdp_destab` есть, но нет GPU destab test. |
| QN-17 | CheckpointManager Error Resilience (7) | P1 | ✅ **Исполнено** | 7 тестов в `TestCheckpointManagerResilience`. |
| QN-18 | RNGRegistry Isolation Property | P2 | ❌ **Не реализованы** | 0 тестов. |
| QN-19 | AdaptiveErrorTracker FIFO Boundary | P2 | ❌ **Не реализованы** | Только 1 косвенный тест (`test_concept_error_fifo`). |
| QN-20 | Lateral Inhibition Divergence | P2 | ❌ **Не реализованы** | |
| QN-21 | Contrastive Hard Negative Selection | P2 | ❌ **Не реализованы** | |
| QN-22 | Centroid Pull Normalization | P3 | ❌ **Не реализованы** | |
| QN-23 | Memory Stress | P3 | ❌ **Не реализованы** | |

---

## 6. НОВЫЕ архитектурные проблемы (V8 Regressions + Fresh Issues)

### AM-37-REG (P0): Subspace-Kinetic STDP — полный NO-OP

**Файлы**: `stdp_trainer.py:143-155` + `concept_space.py:511-554`

`_subspace_update()` применяет subspace-specific LRs (lr_c, lr_a, lr_m) через проецирование градиента в code space и маскирование. **Однако** `_apply_vector_update` (concept_space.py:544-549) перезаписывает код:
```python
new_code = v_new @ self.fractal.basis.T   # ← полная перезапись
```
Это вычисляет код **из финального вектора**, игнорируя subspace-LR. Маски в `_subspace_update` — мёртвый код. Эффект: все три subspace LR равнозначны, `subspace_lr` параметр — плацебо.

**Воспроизведение**: Любой вызов `train_from_text` с `subspace_lr=(0.01, 99.0, 0.001)` даст тот же результат, что и `subspace_lr=(0.01, 0.01, 0.01)`.

### REG-1 (P0): Два параллельных цикла обучения в train_full.py

**Файл**: `train_full.py:397-457` (TrainingPipeline.run_epoch) + `train_full.py:710-830` (main while-loop)

`TrainingPipeline` создаётся (строка 684) и содержит два цикла:
1. `run_epoch()` — **НИГДЕ не вызывается** (мёртвый код, ~60 строк)
2. Main while-loop (строка 710) — активный path, вызывает `pipeline._checkpoint()` без захвата возврата

**Проблема**: `run_epoch` содержит исправление T-B1 (захват возврата `_checkpoint`), но main-loop игнорирует возврат:
```python
pipeline._checkpoint(epoch, idx, elapsed, epoch_lines, destab_scale, t_start)
# возвращаемый (idx, start_line, epoch_train) — потерян!
```

### REG-2 (P0): Два параллельных checkpoint с разными именованиями

**Файлы**: `checkpoint_manager.py:67-68` + `train_full.py:99-121`

- **CheckpointManager** сохраняет как `cs_{tag}.json`, `lat_{tag}.json`
- **cleanup_old_checkpoints** ищет `concept_space_*k.json`, `syntax_lattice_*k.json`
- **resume logic** (train_full.py:160-161) ищет `concept_space_{resume_tag}.json`

**Конфликт**: CheckpointManager не совместим с resume. После первого checkpoint от `ckpt_mgr.save()`:
- Resume не найдёт `concept_space_e1_l500.json` (файл называется `cs_e1_l500.json`)
- cleanup_old_checkpoints не очистит файлы CheckpointManager

### REG-3 (P1): `_checkpoint` игнорирует возврат в main-loop

**Файл**: `train_full.py:819-820`

Как указано в REG-1. T-B1 частично исправлен в `run_epoch`, но main-loop теряет возврат. Это означает:
1. `idx` никогда не сбрасывается
2. `_rescore_lines` получает `epoch_train[remaining:]` где `remaining = idx - start_line + 1` с `start_line=0` → весь массив
3. Self-paced learning работает некорректно

### REG-4 (P1): `noise_scale` не передаётся в `train_batch`

**Файл**: `train_full.py:752-760`

Параметр `noise_scale` не передаётся:
```python
n_pairs = gen.train_batch(batch_buffer, ..., use_torch=..., destab_scale=batch_destab)
# noise_scale отсутствует — всегда 0.0
```

Шум градиента (TN-6) не применяется при обучении, хотя имплементация готова (`stdp_trainer.py:385-386`).

### REG-5 (P1): GPU Contrastive Objective — смесь numpy/torch, не векторизован

**Файл**: `stdp_trainer.py:656-754`

- `for i in range(ng)` + вложенные `for j` циклы
- `neg_cid = int(best_idx[i, j].item())` — CPU roundtrip
- `cooc_set = {ctx for ctx, _ in gen_updates[gen_idxs[i]]}` — dict access на каждый концепт
- `cs._apply_vector_update(gen_idxs[i], v2.cpu().numpy())` — CPU sync после каждого шага
- `push_total` (строка 690) аллоцирован, но **НЕ ИСПОЛЬЗУЕТСЯ**

### REG-6 (P1): GPU negative sampling — per-concept loop с CPU sync

**Файл**: `stdp_trainer.py:549-595`

```python
for gi, gen_cid in enumerate(unique_gen):
    neg_lr_i = neg_lr
    ce = gen.concept_error.get(gen_cid, 0.0)       # dict lookup (CPU)
    ...
    v_np = cs.concept_vectors.get(gen_cid)          # dict lookup (CPU)
    ...
    cs._apply_vector_update(gen_cid, v_new)          # CPU write
```

Каждая итерация: 2 dict lookup + 1 `_apply_vector_update` (с numpy ops + hook call).

### REG-7 (P1): `_graph_cache` unbounded

**Файл**: `crystal_generator.py:71` + `crystal_generator.py:556-558`

Ключ: `tuple(sorted(set(sources)))` где sources = контекстные токены (до 3). С vocab_size=146K число уникальных комбинаций = C(146000, 1) + C(146000, 2) + C(146000, 3) ≈ 5×10¹⁴. На практике растёт бесконечно, cache не чистится (очистка только в `train_from_text` при `gen.lattice.update` → `gen._graph_cache.clear()`). В генерации cache **никогда не очищается**.

### REG-8 (P2): CUDA Events — аллокация на каждый вызов

**Файл**: `stdp_trainer.py:348-350`

```python
gen._prof_start = torch.cuda.Event(enable_timing=True)
gen._prof_end = torch.cuda.Event(enable_timing=True)
```
В `_gpu_stdp_apply`, который вызывается каждый batch. Events пересоздаются. Нужно инициализировать 1 раз в `__init__` или первом вызове.

### REG-9 (P2): `_fb_dirty` / `_torch_dirty` — хрупкая система

**Файл**: `crystal_generator.py:168, 240-243, 267` + `concept_space.py:111`

Флаги `_fb_dirty` и `_torch_dirty` синхронизируются через множество точек:
- `crystal_generator.py:168` — проверка обоих
- `crystal_generator.py:233` — сброс `_fb_dirty = False`
- `crystal_generator.py:243` — `_torch_dirty = True`
- `crystal_generator.py:267` — `_fb_dirty = False` (в `_ensure_fb_tensor`)
- `train_full.py:139` — `gen._torch_dirty = True` (в `_train`)

Любой пропуск синхронизации → stale tensors или бесконечный rebuild. Нужен version counter.

### REG-10 (P2): `GenerationResult` — dataclass с мёртвыми полями

**Файл**: `crystal_generator.py:47-57`

Поля `chunks`, `time`, `word_count`, `max_words` — нигде не используются. `score` вычисляется, но метрика качества генерации отсутствует.

### REG-11 (P2): `_destab_field_fallback` — мёртвый код (почти)

**Файл**: `crystal_generator.py:135-151`

Вызывается только из CPU-пути (`_cpu_stdp_apply:276`). В GPU-пути (`_gpu_stdp_apply:440`) аналогичный код — inline. При `use_torch=True` по умолчанию CPU-путь не используется → fallback не вызывается.

### REG-12 (P2): RNGRegistry не используется

**Файл**: `crystal_generator.py:84`

Создаётся как `self.rng_registry = RNGRegistry(master_seed=42)`, но ни один метод не вызывает `self.rng_registry.get('name')`. Все RNG — прямые `random.Random(42)`, `np.random.RandomState(42)`.

---

## 7. НОВЫЕ улучшения (AM-38+)

### AM-38 (P0): Fix subspace-LR — сохранять subspace code в `_apply_vector_update`

**Файл**: `concept_space.py:544-549`

**Проблема**: `new_code = v_new @ self.fractal.basis.T` перезаписывает subspace-specific код.

**Решение**: Сохранять старый code, применять subspace-LR маски к градиенту в code space, обновлять code напрямую:
```python
if hasattr(self._last_code, cid) and self._last_code[cid] is not None:
    old_code = self._last_code[cid]
    code_grad = (v_new @ self.fractal.basis.T) - old_code
    # Применить subspace LR к code_grad
    code_grad *= subspace_lr_masks  # из конфига
    new_code = old_code + code_grad
    # Нормализовать
    ...
```

ИЛИ изменить архитектуру на `v = code @ basis` + обновлять code (не v) как каноническое представление.

### AM-39 (P1): Единый цикл обучения — удалить `TrainingPipeline.run_epoch()` или переключиться на него

**Файл**: `train_full.py`

**Вариант A**: Удалить `run_epoch()`, починить main-loop для захвата возврата `_checkpoint`.
**Вариант B**: Переключить main-loop на `pipeline.run_epoch()`, удалить 120 строк дублирующего кода.

Рекомендуется **Вариант B** — меньше кода, T-B1 уже исправлен внутри `run_epoch`.

### AM-40 (P1): Единый checkpoint naming — CheckpointManager + resume совместимость

**Файлы**: `checkpoint_manager.py`, `train_full.py`

CheckpointManager должен использовать naming, совместимый с resume:
- `concept_space_{tag}.json` + `syntax_lattice_{tag}.json` (как ожидает `train_full.py:160-161`)
- ИЛИ обновить resume logic на формат CheckpointManager

### AM-41 (P1): Передать `noise_scale` в `train_batch`

**Файл**: `train_full.py:752`

Добавить `noise_scale=pipeline.opt.p['noise_scale'].current` в вызов `gen.train_batch(...)`.

### AM-42 (P1): Векторизовать GPU Contrastive Objective

**Файл**: `stdp_trainer.py:656-754`

- Заменить `for i in range(ng)` на batched tensor ops
- Убрать `.item()`, CPU roundtrips, per-concept `cs._apply_vector_update`
- Использовать `push_total` (уже аллоцирован, строка 690)
- Batched `scatter_add_` для push градиентов

Сложность: 7 (как SN-19).

### AM-43 (P1): Векторизовать GPU negative sampling

**Файл**: `stdp_trainer.py:574-595`

- Заменить per-concept loop на batched tensor ops
- `concept_error` уже есть в `gen._ce_t` (тензор) — использовать его вместо `gen.concept_error.get()`
- Batched `scatter_add_` для градиентов
- Единый `cs._apply_vector_update` после цикла

### AM-44 (P2): LRU eviction для `_graph_cache`

**Файл**: `crystal_generator.py:71`

```python
from functools import lru_cache
# Или OrderedDict с maxlen
self._graph_cache = OrderedDict()
self._graph_cache_max = 5000  # или из конфига
```

### AM-45 (P2): Init CUDA Events 1 раз

**Файл**: `stdp_trainer.py:348-350`

Перенести в `__init__` или ленивую инициализацию (проверять `hasattr(gen, '_prof_start')`).

### AM-46 (P2): Consolidate RNG — migrate all users to RNGRegistry

**Файлы**: Все, где есть `random.Random(...)` / `np.random.RandomState(...)`

```python
self.rng_registry = RNGRegistry(master_seed=42)
# Вместо self.main_rng = random.Random(42):
self.main_rng = self.rng_registry.get('main')
# Вместо cs.rng = np.random.RandomState(42):
self.rng = self.rng_registry.get('cs')
```

### AM-47 (P3): Magic number → FCFConfig

**Файл**: `concept_space.py:661`

```python
_hboost_cache_refresh = self.config.get('hboost_cache_refresh', 1000)
if self._hboost_cache_step % _hboost_cache_refresh == 0:
```

### AM-48 (P3): Global version counter вместо dirty flags

**Файлы**: `crystal_generator.py`, `concept_space.py`

```python
class ConceptSpace:
    _version = 0
    def _bump_version(self):
        self._version += 1

class CrystalGenerator:
    _cached_version = -1
    def _ensure_torch(self):
        if self._cached_version == cs._version and self._vecs_t is not None:
            return
        self._cached_version = cs._version
        self._build_torch_tensors(...)
```

### AM-49 (P3): `GenerationResult` — удалить мёртвые поля

**Файл**: `crystal_generator.py:47-57`

Удалить `chunks`, `time`, `max_words` (есть в kwargs).

### AM-50 (P3): Fuse concept_error update — batched copy

**Файл**: `stdp_trainer.py:398-401`

```python
# VM: уже есть _ce_t на GPU
# CPU sync одной операцией:
ce_vals = avg_err.cpu().numpy()
for gi, gen_cid in enumerate(unique_gen):
    gen.concept_error._data[gen_cid] = float(ce_vals[gi])  # прямой set, без move_to_end
```

Или реализовать `batch_update()` на AdaptiveErrorTracker.

---

## 8. Итоговая матрица приоритетов V8

### P0 (4 критические проблемы)

| ID | Проблема | Зона | Сложность |
|:--:|----------|:----:|:---------:|
| **AM-37-REG** | Subspace-Kinetic STDP — NO-OP (`_subspace_update` + `_apply_vector_update` конфликт) | Arch/NS | 5 |
| **REG-1** | Два парал. цикла обучения — main-loop игнорирует возврат `_checkpoint` | TD | 3 |
| **REG-2** | Два checkpoint naming — resume сломан | TD | 2 |
| **REG-3** | T-B1 не работает в main-loop — idx не сбрасывается | TD | 1 |

### P1 (12 проблем)

| ID | Проблема | Зона | Сложность |
|:--:|----------|:----:|:---------:|
| AM-19/AM-26 | GPU Contrastive — Python цикл, не векторизован | Arch/GPU | 7 |
| REG-4 | `noise_scale` не передаётся в `train_batch` (TN-6 мёртв) | TD | 1 |
| REG-5 | GPU Contrastive — numpy/torch mix, push_total не используется | NS/GPU | 7 |
| REG-6 | GPU neg sampling — per-concept loop, CPU sync | GPU | 5 |
| SN-19 | Full vectorization of GPU Contrastive | NS/GPU | 7 |
| AM-24/AM-25 | CPU path всё ещё существует (~300 строк) | Arch | 3 |
| AM-32/REG-7 | `_graph_cache` unbounded | Arch | 1 |
| AM-39 | Единый цикл обучения | TD | 3 |
| AM-40 | Checkpoint naming совместимость | TD | 2 |
| AM-41 | noise_scale передача | TD | 1 |
| AM-42 | GPU Contrastive векторизация | Arch/NS | 7 |
| AM-43 | GPU neg sampling векторизация | Arch/GPU | 5 |

### P2 (15 проблем)

AM-29 (RNG), AM-30 (EMA vec), AM-31 (CE batch), AM-33 (hormones reset), AM-34 (magic number), AM-35 (PathConfig dup), AM-36 (version counter), AM-44 (graph cache LRU),
AM-45 (CUDA Events init), AM-46 (RNG consolidate),
G-22..G-25 (vec EMA, pre-alloc, fused NS, lerp_),
TN-13 (progressive BSize), TN-15 (decay warmup),
QN-18, QN-19, QN-20, QN-21,
REG-8 (CUDA Events), REG-9 (dirty flags), REG-10 (GenerationResult), REG-11 (dead fallback), REG-12 (RNGRegistry dead)

### P3 (6 проблем)

AM-37 (subspace loss — уже P0 в части REG), AM-47..AM-50,
SN-21 (Riemannian),
G-26..G-30 (vec inhibition, CUDA stream, reuse, fuse norm, compile),
QN-22, QN-23

---

## 9. Рекомендуемый план работ

### Фаза 0 (CRITICAL — 4 задачи)

1. **AM-37-REG**: Починить subspace-LR — сохранять code, не перезаписывать в `_apply_vector_update`
2. **REG-1/REG-3**: Переключить main-loop на `pipeline.run_epoch()` или починить захват возврата `_checkpoint`
3. **REG-2**: Унифицировать checkpoint naming — CheckpointManager под resume
4. **REG-4**: Передать `noise_scale` в `train_batch`

### Фаза 1 (P1 — 6 задач)

5. **AM-42/AM-43**: Векторизация GPU Contrastive + GPU negative sampling
6. **AM-25**: Удалить CPU-путь (оставить только `use_torch=True` path)
7. **AM-39**: Единый цикл обучения (Вариант B — `run_epoch()`)
8. **AM-40**: Checkpoint naming fix
9. **AM-32**: `_graph_cache` LRU
10. **G-21**: CUDA Events 1 раз

### Фаза 2 (P2 — 8 задач)

11. AM-44 (LRU), AM-45 (CUDA init), AM-29/AM-46 (RNG), AM-30 (EMA vec), AM-31 (CE batch)
12. G-22..G-25
13. TN-13, TN-15
14. QN-18, QN-19, QN-20, QN-21

### Фаза 3 (P3 — 6 задач)

15. AM-34..AM-37, AM-47..AM-50
16. G-26..G-30, QN-22, QN-23

---

## 10. Изменения в коде — что СДЕЛАНО (summary)

### crystal_generator.py
- ✅ AM-24: 7 forwarding-методов удалены, `_trainer` создаётся в `__init__`
- ✅ AM-14: `_use_torch` определён
- ✅ AM-22: `_get_total_freq()`
- ✅ AM-20: `_AsyncPipeline` удалён

### stdp_trainer.py
- ✅ SN-15: `_subspace_update()` (но сломан — см. AM-37-REG)
- ✅ T-B3/SN-B1: EMA перед `_apply_vector_update`
- ✅ N-4: concept_error reweighting + const 0.3
- ✅ SN-16: Field-Aware Contrastive Decoupling (CPU + GPU)
- ✅ TN-14: Field-Aware Contrastive Regularization
- ✅ SN-18: `_sync_ema` правильная семантика
- ✅ momentum_mu (TN-11)

### train_full.py
- ✅ AM-15/T-B2: TrainingPipeline активирован
- ✅ TN-12: Switched Evaluation
- ✅ `_checkpoint()` возвращает tuple
- ✅ while-цикл вместо for-enumerate

### tests/test_stdp.py
- ✅ QN-16: 9 тестов STDPTrainer Integration
- ✅ QN-17: 7 тестов CheckpointManager
- ✅ ВСЕ вызовы обновлены: `gen._method()` → `gen._trainer._method()`

### checkpoint_manager.py
- ✅ `.json` base (вместо `.npz`)
- ✅ cleanup по 3 расширениям

---

## V8 Commit Fixes

| ID | Статус | Фикс |
|:--:|:------:|------|
| **AM-37-REG / SN-15** | ✅ Исправлен | Добавлен `cs._apply_subspace_update()` (concept_space.py:556-582), который обновляет code напрямую и реконструирует vector из code. `_subspace_update()` заменён на прямой вызов `_apply_subspace_update` в CPU и GPU путях. |
| **REG-1 / T-B2** | ✅ Исправлен | Удалён мёртвый `run_epoch()`. Main loop (строка 710) — единственный активный цикл. |
| **REG-2** | ✅ Исправлен | CheckpointManager переименован на `concept_space_{tag}.json` / `syntax_lattice_{tag}.json`, opt файл — `concept_space_{tag}.opt.json`. Tag — `{k}k` формат (совместим с resume). |
| **REG-3 / T-B1** | ✅ Исправлен | Main loop захватывает return `_checkpoint`: `idx, start_line, epoch_train = result`. |
| **REG-4 (noise_scale)** | ✅ Исправлен | `noise_scale=opt.p['noise_scale'].current` и `momentum_mu=0.9` передаются в `gen.train_batch()`. |
| **REG-5 (TN-14 batch-stale)** | ✅ Исправлен | TN-14 использует локальную `v_local = g_vecs[i].clone()`, единственный `_apply_vector_update` в конце. |
| **opt.save_state()** | ✅ Исправлен | Через `state = opt.save_state()` + `json.dump()` вместо передачи path. |

*Отчёт составлен коллегией AI-агентов: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent*
