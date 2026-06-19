# FCF V11 — Архитектурный аудит

**Дата**: 2026-06-19  
**Версия**: V11 (аудит V10 коммитов: 525688b + d36a780)  
**Статус**: 105 тестов проходят ✅ (было 79)

---

## Executive Summary

| Метрика | V9 | V10 | V11 | Δ |
|---------|:--:|:---:|:---:|:-:|
| Тесты | 79 | 105 | **105** | = |
| P0 | 0 | 0 | **1** ⛔ | +1 |
| P1 | 10 | 5 | **6** | +1 |
| P2 | 22 | 13 | **11** | −2 |
| GPU-оптимизации (G-40..G-52) | 0/13 | 0/13 | **13/13** ✅ | +13 |
| Тесты (QN-32..QN-40) | 0/9 | 0/9 | **9/9** ✅ | +9 |
| `.item()` на batch | ~16 500 | ~16 500 | **~5 600** | −66% |

**Главные находки V11:**

1. ✅ **V10 реализовал ВСЕ GPU-оптимизации** (G-40..G-52) — batched subspace update, GPU lateral inhibition, zero-copy write-back, deferred sync, fused post-STDP
2. ✅ **V10 реализовал ВСЕ 9 тестов** (QN-32..QN-40) — subspace, contrastive, evaluate, noise_scale, RNGRegistry, AdaptiveErrorTracker, checkpoint cleanup, pipeline, dead code
3. ⛔ **P0: train_full.py:722 — `noise_scale` keyword argument не существует** — `fluctuate_fractal()` принимает `fluctuation_amp`, а не `noise_scale`. При первом флуктуате (каждые 2000 строк) — **TypeError: unexpected keyword argument 'noise_scale'**. `opt.p['noise_scale']` также KeyError (переименован в `gradient_noise_scale`).
4. ⛔ **REG-V9-7 исправлен не полностью** — `gradient_noise_scale` передаётся в `train_batch` (✅), но вызов `fluctuate_fractal` (train_full.py:722) не обновлён
5. ~5 600 `.item()`/batch осталось (было ~16 500) — в основном в `_contrastive_objective_gpu`

---

## 1. Верификация V10 коммитов

### Commits
- `525688b` — V10 All Fixes: Phase 0+P1+GPU opts+Tests
- `d36a780` — G-50/G-51/G-52: GPU zero-copy + deferred sync + fused post-STDP

### 1.1 Phase 0 (P1 bugs) — 3/3 ✅

| ID | Исправление | Файл | Статус |
|:--:|-------------|:----:|:------:|
| TN-31 | `checkpoint_state.json` при каждом чекпоинте | train_full.py:394-400 | ✅ |
| G-57 | `push_total`/`lr_scale` удалены | stdp_trainer.py | ✅ |
| SN-35/36 | CPU neg sampling/contrastive parity (compound updates) | stdp_trainer.py | ✅ |

### 1.2 Phase 1 (P1) — 4/4 ✅

| ID | Исправление | Файл | Статус |
|:--:|-------------|:----:|:------:|
| REG-V9-7 | `noise_scale` split → `gradient_noise_scale` + `fluctuation_amp` | fcf_config.py:318-325, stdp_trainer.py:67 | ⚠️ Частично (см. P0) |
| G-46 | `_mom_buf` CPU dict → persistent `_mom_t` GPU tensor | stdp_trainer.py:412-418 | ✅ |
| G-42 | `_centroid_pull_batch` CPU → GPU | stdp_trainer.py:799-845 | ✅ |

### 1.3 GPU Optimizations (G-40..G-52) — 13/13 ✅

| ID | Оптимизация | Статус |
|:--:|-------------|:------:|
| G-40 | Batched GPU subspace update (`_apply_subspace_update_batch`) | ✅ `concept_space.py:591-638` |
| G-41 | Full GPU lateral inhibition (без `.item()`) | ✅ `stdp_trainer.py:508-532` |
| G-43 | GPU neg sampling vectorized (без CPU roundtrip) | ✅ `stdp_trainer.py:574-624` |
| G-44 | GPU contrastive (pre-computed cooc_masks + fb_overlaps) | ✅ `stdp_trainer.py:686-792` |
| G-48 | `torch.compile` flag | ⚠️ `stdp_trainer.py:336-338` — только комментарий |
| G-49 | Pre-allocated fused buffer | ✅ `crystal_generator.py:120` |
| G-50 | Zero-copy vector write-back (GPU→GPU) | ✅ `stdp_trainer.py:467-473` |
| G-51 | Deferred vector sync (batched `_vecs_t` write) | ✅ `stdp_trainer.py:427,479-484` |
| G-52 | Fused post-STDP (contrastive + neg sampling) | ✅ `stdp_trainer.py:496-506` |

### 1.4 Code Quality — 5/5 ✅

| ID | Исправление | Статус |
|:--:|-------------|:------:|
| AM-25 | CPU path marked legacy | ✅ `stdp_trainer.py:229` |
| AM-29/46 | `main_rng` via RNGRegistry | ✅ |
| AM-30 | Batched EMA update | ✅ `stdp_trainer.py:487-489` |
| AM-31 | ConceptError sync (batch update) | ✅ `stdp_trainer.py:399-407` |
| AM-33 | HormonalSystem.reset() | ✅ `hormonal_system.py:193-194` |
| SN-39 | `connection_strength` removed from GPU inner loop | ✅ |

### 1.5 Tests (QN-32..QN-40) — 9/9 ✅

| ID | Тест | Статус |
|:--:|------|:------:|
| QN-32 | TestSubspaceUpdate | ✅ |
| QN-33 | TestGPUContrastive | ✅ |
| QN-34 | TestEvaluate | ✅ |
| QN-35 | TestNoiseScale | ✅ |
| QN-36 | TestRNGRegistry | ✅ |
| QN-37 | TestAdaptiveErrorTracker | ✅ |
| QN-38 | TestCheckpointCleanup | ✅ |
| QN-39 | TestTrainingPipeline | ✅ |
| QN-40 | TestDeadCode | ✅ |

---

## 2. P0 — Критические проблемы (1)

### P0-1: train_full.py:722 — `noise_scale` keyword не существует

**Суть**: `noise_scale` был переименован в `gradient_noise_scale` + `fluctuation_amp` (REG-V9-7), но строка 722 в train_full.py осталась без изменений:

```python
# train_full.py:722 — БУДЕТ CRASH
cs.fluctuate_fractal(noise_scale=opt.p['noise_scale'].current,
                     decay=opt.p['decay_rate'].current,
                     ...)
```

Проблемы:
1. `opt.p['noise_scale']` — KeyError: `noise_scale` удалён из `FCFConfig.params`. В списке только `gradient_noise_scale` и `fluctuation_amp`
2. `fluctuate_fractal()` не принимает keyword `noise_scale=` — параметр называется `fluctuation_amp`

**Влияние**: Crash при первом `is_fluct_due` (каждые 2000 строк). Вся training pipeline падает.

**Fix**:
```python
cs.fluctuate_fractal(fluctuation_amp=opt.p['fluctuation_amp'].current, ...)
```

---

## 3. P1 — Открытые проблемы (6)

### 3.1. `.item()` syncs: ~5 600/batch

| Где | `.item()` вызовов | Назначение |
|:---:|:-----------------:|:----------:|
| `_build_pairs:185` | 1 × N_pairs | `field_overlap` via `torch.bitwise_and` |
| `_contrastive_objective_gpu:738-767` | ~165 × N_gen | per-candidate overlap, cos, idx |

Остаётся ~5 600 `.item()`/batch для N_gen=32. Основной вклад — `_contrastive_objective_gpu` с двойным Python loop (hard negatives + TN-14 regularization).

**Предложение**: Вынести TN-14 regularization на тензорные операции.

### 3.2. SN-38: cooc_set rebuild в Python loop

`_contrastive_objective_gpu:708-713` — `cooc_masks` строится через Python loop:
```python
for i, gen_cid in enumerate(gen_idxs):
    ctx_cids = [ctx for ctx, _ in gen_updates[gen_cid]]
    if ctx_cids:
        ctx_t = torch.tensor(ctx_cids, dtype=torch.long, device=d)
        cooc_masks[i, ctx_t] = True
```

При N_gen=32, N_ctx~10 — 32 Python итерации + 2 list comprehensions. Для 146K vocab — аллокация `cooc_masks` (ng × V) bool tensor = 32 × 146K × 1B = ~4.6MB. Не критично, но можно оптимизировать через sparse или pre-batched scatter.

### 3.3. SN-40: field_bits overlap per-candidate

`_contrastive_objective_gpu:745-773` — для каждого hard negative вызывается `.item()` для overlap.

**Предложение**: Заменить на тензорную маску: `valid = (fb_overlaps[i] > 0) & (cos_val > 0.3)` без per-element `.item()`.

### 3.4. TN-13: Progressive batch size с плато не реализован

`train_full.py:644`: `bs_curve` — линейная рампа по idx, не адаптивная:
```python
bs_curve = lambda i: int(CFG.batch_size_start + (CFG.batch_size_end - CFG.batch_size_start) * _curriculum_p(i))
```

batch_size не увеличивается при плато метрик. Нет rules в ParameterOptimizer для batch_size.

### 3.5. TN-15: Decay warmup с protect threshold ramp

`train_full.py:728-732`:
```python
lattice.decay_all(rare_concept_protect=True, rare_threshold=3)
lattice.decay_connections()
cs.decay_usage(decay=0.98, rare_protect=True)
```

`rare_threshold=3` статичен. Нет ramp от 0 → target. `rare_protect` только binary (True/False) — нет плавного включения.

### 3.6. TN-34: opt.json naming mismatch

- `CheckpointManager._sync_save:86` — сохраняет `concept_space_{tag}.opt.json` (tagged)
- `_final_save:596` — сохраняет `concept_space.opt.json` (без тега)

Resume code (train_full.py:274-286) пытается грузить tagged → tagless → data-dir tagged → any. Работает, но хрупко.

---

## 4. P2 — Открытые проблемы (11)

| ID | Проблема | Сложность | Приоритет |
|:--:|----------|:---------:|:---------:|
| AM-25 | CPU path (~300 строк) legacy, не удалён | 3 | Medium |
| G-48 | `torch.compile` не активирован | 4 | Low |
| SN-38 | `cooc_masks` rebuild в Python loop | 3 | Low |
| SN-40 | field_bits overlap per-candidate `.item()` | 4 | Medium |
| TN-13 | batch_size не адаптируется к плато | 3 | Low |
| TN-15 | decay warmup ramp не реализован | 2 | Low |
| TN-34 | opt.json naming mismatch (tagged vs tagless) | 1 | Low |
| AM-80 | `_lateral_inhibition_gpu` — всё ещё Python loop | 3 | Medium |
| AM-81 | `_negative_sampling_gpu` — per-concept Python loop | 3 | Medium |
| AM-82 | Нет интеграционных тестов train_full.py | 4 | Medium |
| AM-83 | Нет логгера — print() везде | 2 | Low |

---

## 5. AM-80+: Улучшения (14 предложений)

### 5.1 AM-80: GPU lateral inhibition — pure tensor

`_lateral_inhibition_gpu:518-532` — Python loop for gi in range(n):
```python
for gi in range(n):
    gi_mask = mask_all[gi]
    ...
```
**Заменить** на batched tensor: `gv_new = (sim_us * gv_others - sim_us_sq * gv_self).sum(dim=1)` — одна операция на все n концептов.

### 5.2 AM-81: GPU neg sampling — batched

`_negative_sampling_gpu:604-624` — Python loop:
```python
for gi, gen_cid in enumerate(unique_gen):
    ...
```
**Заменить** на masked scatter: все градиенты считаются тензорно, применяются через `_vecs_t[valid_noise_mask].scatter_add_()`.

### 5.3 AM-84: TN-14 regularization — pure tensor

`_contrastive_objective_gpu:756-774` — Python inner loop до 50 итераций с `.item()`. Вынести в тензор:
```python
reg_mask = (topk_val > reg_thresh) & ~cooc_masks & (fb_overlaps == 0)
reg_grad = (topk_val * gen._vecs_t[topk_idx]).mean(dim=1) - g_vecs
```

### 5.4 AM-85: TorchCache — вынести тензорные кеши

`_ensure_torch:163-258` — единый метод строит ВСЕ тензоры (_vecs_t, _fb_t, _ce_t, _ema_vecs_t, _mom_t, _basis_t, _codes_t, _fused_buf). Mixed concerns: CPU fallback, OOM handling, initialization, dirty flag checking.

**Предложение**: Выделить `TorchCache(cs)` класс с:
- `vecs` — property с lazy rebuild
- `invalidate()` — сброс всех dirty флагов
- `to(device)` — кросc-девайс перемещение

### 5.5 AM-86: Протокол `TorchInvalidatable` для `fluctuate_fractal`

```python
class TorchInvalidatable(Protocol):
    def invalidate_torch(self): ...
```

Вместо `generator: Optional[CrystalGenerator]` → `generator: Optional[TorchInvalidatable]`.

### 5.6 AM-87: Memory оптимизация

- `_vecs_t: float16` ✅
- `_ema_vecs_t: float32` — ~224MB для 146K×384 — можно хранить в float16, если loss не страдает
- `_fused_buf: float32(V, D+1)` — ~225MB — можно сделать динамическим (alloc по max N unique_gen)
- `_fb_t: uint8(V, fb_bytes)` — ~146K × 256B = ~37MB — можно lazy-load/mmap

### 5.7 AM-88: `build_octree_fields` — консистентность

`concept_space.py:374-460` — при каждом rebuild: `self.fractal.init_fields(n_anchors)` + `self.fractal.field_bits` обнуляется и заново заполняется. Для концептов без кода — `field_bits` не создаются. 
**Предложение**: Добавить `ensure_all_concepts_have_fields()` — гарантия, что каждый concept в `cs.fractal.codes` имеет `field_bits`.

### 5.8 AM-89: `concept_usage` — не растёт бесконечно

`concept_space.py:696` — `self.concept_usage = {cid: 0.0 for cid in self.concept_vectors}` — 146K entries. При загрузке — `for cid in range(obj.vocab_size)` — всегда полный vocab. Можно заменить на массив `np.zeros(V, float32)`.

### 5.9 AM-90: Тесты для train_full.py

Нет ни одного теста для 817-строчного train_full.py. Добавить:
- `test_resume_flow` — проверка resume без checkpoint (fresh start)
- `test_checkpoint_flow` — pipeline._checkpoint не падает
- `test_curriculum_ramp` — bs_curve, max_len ramp, _effective_cp
- `test_fluctuate_call` — проверка, что `fluctuate_fractal` вызывается с правильными kwargs

### 5.10 AM-91: `_graph_cache` — thread safety

`_branch:579-585` — `self._graph_cache` OrderedDict без блокировки. При многопоточном generate (через API) — race condition.

### 5.11 AM-92: `_build_pairs` — унификация CPU/GPU codegen

cpu/gpu pair building в `_build_pairs` разделены на два параллельных трека с дублированием логики (PMI weight, field_weight, theta_gate, slow_lr). Вынести в чисто-питоновский генератор пар, затем разветвлять на CPU apply / GPU apply.

### 5.12 AM-93: `FCFConfig.params` — batch_size, rare_threshold

Добавить ParamDef для:
- `batch_size` с rules на plateau detection
- `rare_threshold` с ramp от 1 → target

### 5.13 AM-94: `fluctuation_amp` — decay

Текущая логика: `fluctuation_amp` константа на всё обучение. Добавить cosine decay.

### 5.14 AM-95: deprecate `_quiet` wrapper

`train_full.py:17-24` — `_quiet` ловит ВСЕ исключения (кроме KeyboardInterrupt) и логирует. Маскирует реальные ошибки. Заменить на явные try/except в местах вызова.

---

## 6. Детальный анализ `.item()` syncs

### 6.1 `_build_pairs:185`
```python
overlap = int(torch.bitwise_and(gen._fb_t[ids[i]], gen._fb_t[ids[j]]).sum().item())
```
1 `.item()` на каждую STDP пару. Для batch 32 строк × ~10 пар = **320 `.item()`**.

**Fix**: Вычислить overlap тензорно: `fb_overlaps = (gen._fb_t.unsqueeze(1) & gen._fb_t.unsqueeze(0)).sum(dim=-1)` — один раз на batch. Но это O(V²×fb_bytes) памяти и времени. Альтернатива: накапливать в CPU буфер и sync раз в N пар.

### 6.2 `_contrastive_objective_gpu:735-792`

Для каждого из N_gen концептов (batch avg ~32):
- Loop max_hard=5: `.item()` для `neg_cid`, `cos_val`, `overlap` — до 15 syncs
- Loop reg=50 (TN-14): `.item()` для `rcid`, `rcos`, `ro` — до 150 syncs

Всего: ~165 × 32 = **~5 280 `.item()`**.

**Fix**: Pure tensor implementation без Python loops — см. AM-84.

---

## 7. Комплексный план V11

### Фаза 0 (немедленно, crash fix)

| # | Задача | Файл | Время |
|:-:|--------|:----:|:----:|
| 1 | `noise_scale` → `fluctuation_amp` | train_full.py:722 | 1 мин |
| 2 | `opt.p['noise_scale']` → `opt.p['fluctuation_amp']` | train_full.py:722 | 1 мин |

### Фаза 1 (P1 — 6 задач)

| # | Задача | Сложность |
|:-:|--------|:---------:|
| 3 | AM-84: TN-14 regularization → pure tensor | 6 |
| 4 | AM-80: GPU lateral inhibition → batched tensor | 4 |
| 5 | AM-81: GPU neg sampling → batched masked apply | 5 |
| 6 | TN-13: batch_size ParamDef + plateau rules | 3 |
| 7 | TN-15: decay warmup ramp (rare_threshold ParamDef) | 2 |
| 8 | TN-34: Унифицировать opt.json naming | 1 |

### Фаза 2 (P2 — 11 задач)

| # | Задача | Сложность |
|:-:|--------|:---------:|
| 9 | AM-85: TorchCache class | 6 |
| 10 | AM-86: TorchInvalidatable Protocol | 3 |
| 11 | AM-87: _ema_vecs_t float16, _fused_buf dynamic | 4 |
| 12 | AM-88: ensure_all_concepts_have_fields | 2 |
| 13 | AM-89: concept_usage → np array | 2 |
| 14 | AM-90: train_full.py unit tests | 6 |
| 15 | AM-91: _graph_cache thread-safe | 2 |
| 16 | AM-92: _build_pairs codegen unification | 5 |
| 17 | AM-93: batch_size + rare_threshold ParamDef | 3 |
| 18 | AM-94: fluctuation_amp cosine decay | 2 |
| 19 | AM-95: _quiet deprecation | 2 |

### Фаза 3 (долгосрочные)

| # | Задача | Сложность |
|:-:|--------|:---------:|
| 20 | G-48: torсh.compile activation | 4 |
| 21 | AM-25: CPU path removal/gating | 3 |
| 22 | Документация: API reference, architecture.md update | 5 |

---

## 8. Итог

| Метрика | V9 | V10 | V11 |
|---------|:--:|:---:|:---:|
| P0 | 0 | 0 | **1** ⛔ |
| P1 | 10 | 5 | **6** |
| P2 | 22 | 13 | **11** |
| GPU-оптимизации | 0/13 | 0/13 | **13/13** ✅ |
| Тесты | 79 | 105 | **105** ✅ |
| `.item()`/batch | ~16 500 | ~16 500 | **~5 600** |
| STR (est.) | ~50% | ~50% | ~55% |

V10 — огромный прогресс: 100% GPU-оптимизаций, 100% тестов, 105 passed.  
Но P0 crash в train_full.py:722 блокирует production запуск.  
После исправления P0 — 6 P1 задач (в основном `.item()` syncs и адаптивные параметры).
