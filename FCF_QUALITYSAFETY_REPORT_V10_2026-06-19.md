# FCF Quality & Safety Report V10 — 2026-06-19

**Agent:** Quality-Safety Agent  
**Scope:** V9 commits a0fe15b, cccc392, 21ee6ca; `test_stdp.py`, `stdp_trainer.py`, `concept_space.py`, `crystal_generator.py`, `train_full.py`, `fcf_config.py`

---

## 1. V9 Новые тесты? — Обнаружено 0

| Коммит | Фиксы | Новые тесты |
|---|---|---|
| a0fe15b | SN-24, SN-25, SN-22.2, TN-25, TN-26, TN-27, REG-V9-1/2/8 | **0** |
| cccc392 | SN-22.1, AM-32, G-45, G-47 | **0** |
| 21ee6ca | SN-22.3 | **0** |
| **Итого** | **12 фиксов** | **0 тестов** |

## 2. STR (Structural Test Reach) — updated

| Module | Lines | Δ V9 | Covered | STR | Δ vs V9 |
|---|---|---|---|---|---|
| `test_stdp.py` | 809 | — | — | — | — |
| `stdp_trainer.py` | 871 | +11 | ~620 | **71%** | ↔ −1pp |
| `concept_space.py` | 881 | (fixes only) | ~0 new | **~0% new** | ↔ unchanged |
| `crystal_generator.py` | 823 | +5 (AM-32) | ~0 new | **~0% new** | ⬇️ |
| `train_full.py` | ~810 | TN-25/26/27 | ~0 new | **~0% new** | ⬇️ |
| `fcf_config.py` | — | +1 field | — | — | — |

**Overall STR: ~50%** (⬇️ −2pp from V9 due to new untested code)

## 3. V9 QN-32..QN-40 Implementation Status (V10)

Все 9 рекомендованных V9 тестовых сьютов:

| QN | Test Suite | Status | Notes |
|---|---|---|---|
| **QN-32** | `_apply_subspace_update` (5 tests) | **❌ NOT IMPLEMENTED** | V8 gap, still zero |
| **QN-33** | GPU Contrastive (4 tests) | **❌ NOT IMPLEMENTED** | V8 gap, still zero |
| **QN-34** | Evaluate (4 tests) | **❌ NOT IMPLEMENTED** | V8 gap, still zero |
| **QN-35** | TrainingPipeline (4 tests) | **❌ NOT IMPLEMENTED** | V8 gap, still zero |
| **QN-36** | RNGRegistry (5 tests) | **❌ NOT IMPLEMENTED** | V8 gap, still zero |
| **QN-37** | AdaptiveErrorTracker (5 tests) | **❌ NOT IMPLEMENTED** | V8 gap, still zero |
| **QN-38** | Checkpoint NPZ cleanup (2 tests) | **❌ NOT IMPLEMENTED** | V8 gap, still zero |
| **QN-39** | GPU STDP path coverage (4 tests) | **❌ NOT IMPLEMENTED** | V8 gap, still zero |
| **QN-40** | Train Full unit tests (4 tests) | **❌ NOT IMPLEMENTED** | V8 gap, still zero |

**Verdict: 0 / 9 V9-recommended test suites implemented.**
**Cumulative: 0 / 17 (V8+V9) recommended test suites.**

---

## 4. New Untested Code in V9

### 4.1 SN-24: Momentum Blend — stdp_trainer.py:461 (MEDIUM)
```python
grad = momentum_mu * mom_cpu[gi] + (1 - momentum_mu) * grad
```
Было `grad = mom_cpu[gi]` (полная замена), стало blend. `test_gpu_stdp_momentum` (test_stdp.py:679) проверяет только smoke — не детектирует разницу между pure-momentum и blend.

### 4.2 SN-25: Slow STDP — stdp_trainer.py:206-223, 363-368 (MEDIUM)
- `_build_pairs()`: новые slow STDP пары с `_META_SLOW=1.0` (строки 217-224)
- `_gpu_stdp_apply()`: `theta_fast` / `theta_slow` / `slow_mask` логика (строки 364-368)
- Ни один тест не создаёт `gpu_meta_l` с 7 колонками (`_META_SLOW`)

### 4.3 AM-32: graph_cache LRU — crystal_generator.py:72-73, 558-563 (MEDIUM)
```python
self._graph_cache = OrderedDict()        # line 72
self._graph_cache_max = 500              # line 73
...
if len(self._graph_cache) >= self._graph_cache_max:
    self._graph_cache.popitem(last=False) # line 560
...
self._graph_cache.move_to_end(sources_key) # line 563
```
LRU eviction при `len(cache) >= 500` и `move_to_end` on hit. Ни один тест не проверяет:
- Eviction logic (FIFO vs LRU)
- Max size enforcement
- `move_to_end` ordering

### 4.4 G-47: lerp_ for EMA — stdp_trainer.py:464 (MEDIUM)
```python
gen._ema_vecs_t[gen_cid].lerp_(gen._vecs_t[gen_cid].float(), 1.0 - gen._ema_decay)
```
Замена explicit EMA на in-place `lerp_`. Тест `test_gpu_stdp_momentum` не настраивает `_ema_vecs_t`, так что этот код полностью пропускается.

### 4.5 G-45: Persistent CUDA Events — stdp_trainer.py:28-32, 348-349, 478-481 (LOW)
Профилировочные CUDA Event'ы перенесены в `__init__`. Функционально не влияет на результаты, так что риск низкий.

### 4.6 SN-22.3: Per-concept avg_elr — stdp_trainer.py:562-579, 589 (MEDIUM)
- Новый scatter_add per-concept `avg_elr_per_gen` вместо глобальной `avg_elr`
- Изменение `grad.mean(dim=0)` → `grad.sum(dim=0)` (строки 593 vs 571, важно!)
- `test_negative_sampling_gpu_no_crash` не проверяет численную корректность

### 4.7 SN-22.2: field_gate guard — stdp_trainer.py:582-586 (LOW)
```python
if field_gate:
    ce = gen.concept_error.get(gen_cid, 0.0)
    neg_lr_i *= (1.0 + ce * 2.0)
```
Было unconditional; теперь guarded. Parity fix — CPU-эквивалент уже guarded.

### 4.8 TN-25: rescore fix — train_full.py:434-435 (MEDIUM)
```python
epoch_train = _rescore_lines(epoch_train[idx + 1:], gen)
idx = -1; start_line = 0
```
Исправление boundary condition rescore. Без тестов возможен регресс при edge cases (idx на границе буфера).

### 4.9 TN-27: global_step — train_full.py:83, 117-118, 489, 573, 749, 765 (MEDIUM)
- `load_checkpoint_state()` теперь возвращает `global_step`
- `_final_save()` принимает `global_step`
- `global_step` инициализируется из `resume_global_step`
- Resumption без тестов: ошибка кол-ва шагов после рестарта

---

## 5. V9 Safety Regressions

1. **AM-32: `_graph_cache.clear()` совместимость** — `stdp_trainer.py:142` вызывает `.clear()` на `OrderedDict`. Это OK (OrderedDict поддерживает `.clear()`), но при эвакуации из-за `_graph_cache_max=500` консистентность кэша при обучении не гарантируется (generate использует LRU, train чистит весь кэш).

2. **SN-22.3: `sum` vs `mean` change** (`stdp_trainer.py:593`) — Было `grad.mean(dim=0)`, стало `grad.sum(dim=0)`. Это меняет масштаб градиента в `num_valid_hard_negatives` раз. Parity с CPU не гарантирована. **HIGH потенциальный impact.**

3. **SN-25: meta_t shape assumption** — Если `meta_t.shape[1] <= _META_SLOW` (backward compat), `slow_mask` = zeros. OK для старых мета-данных. Но если кто-то передаёт meta_l из другого источника с 7+ колонками — silent fallback на fast-only.

4. **G-47: `lerp_` in-place mutation** — Меняет `_ema_vecs_t` **in-place**. Если другой код держит reference на `_ema_vecs_t[gen_cid]`, он увидит mutated tensor. Ранее explicit assignment создавал новый tensor.

---

## 6. Proposed QN-41+ Tests (V10)

### QN-41: SN-24 Momentum Blend Tests (3 tests)
- `test_momentum_blend_changes_behavior` — `momentum_mu=0.9` с `nesterov=False` → verify blend not identical to pure momentum or pure grad
- `test_momentum_blend_zero_mu` — `momentum_mu=0.0` → equivalent to no momentum (grad only)
- `test_momentum_blend_one_mu` — `momentum_mu=1.0` → equivalent to pure momentum (mom only)

### QN-42: SN-25 Slow STDP Tests (3 tests)
- `test_slow_stdp_pairs_created` — verify `_build_pairs` with GPU path produces entries with `_META_SLOW=1.0` for `slow_lr > 1e-6`
- `test_slow_stdp_theta_computation` — verify `theta_slow` calculation in `_gpu_stdp_apply` with 7-column meta
- `test_slow_stdp_vector_impact` — verify vectors differ vs fast-only when slow STDP active

### QN-43: AM-32 graph_cache LRU Tests (3 tests)
- `test_graph_cache_max_eviction` — insert 501 unique keys → verify oldest evicted
- `test_graph_cache_move_to_end` — access existing key → verify ordering preserved
- `test_graph_cache_clear` — verify `_graph_cache.clear()` works with OrderedDict

### QN-44: G-47 EMA lerp_ Tests (3 tests)
- `test_ema_lerp_initialization` — verify `_ema_vecs_t[gen_cid]` initialized on first lerp_
- `test_ema_lerp_decay_math` — verify numerical correctness vs explicit EMA formula
- `test_ema_lerp_inplace_no_alias` — verify no unintended tensor aliasing

### QN-45: SN-22.3 Per-Concept avg_elr Tests (2 tests)
- `test_negative_sampling_gpu_per_concept_lr` — verify `avg_elr_per_gen` differs per concept
- `test_negative_sampling_gpu_sum_mean_parity` — verify `sum(dim=0)` vs old `mean(dim=0)` gives same direction with proper scaling

### QN-46: TN-25 Rescore Fix Tests (2 tests)
- `test_rescore_boundary` — verify `epoch_train[idx+1:]` with idx at buffer edge
- `test_rescore_noop_on_short_buffer` — verify rescore not triggered when `remaining >= len(epoch_train)`

### QN-47: TN-27 global_step Tests (2 tests)
- `test_global_step_restore` — save with global_step=N, reload, verify global_step=N resumed
- `test_global_step_empty_checkpoint` — no checkpoint → global_step=0

### QN-48: GPU STDP Alpha Mask Edge Cases (2 tests)
- `test_gpu_stdp_meta_shape_6_vs_7` — backward compat: 6-col meta → slow_mask=zeros, no crash
- `test_gpu_stdp_meta_shape_8_plus` — 8+ col meta → only first 7 used, no crash

---

## 7. Updated Coverage Gap Matrix

| Method | File:Line | Covered? | Risk | V10 change |
|---|---|---|---|---|
| `_apply_subspace_update()` | `concept_space.py:556` | **❌ NO** | HIGH | ↔ unchanged |
| `_contrastive_objective_gpu()` | `stdp_trainer.py:665` | **❌ NO** | HIGH | ↔ unchanged |
| `evaluate()` / `_evaluate()` | `stdp_trainer.py:58,802` | **❌ NO** | HIGH | ↔ unchanged |
| **SN-25: slow_mask theta** | `stdp_trainer.py:364-368` | **❌ NO** | MED | **NEW V9** |
| **SN-25: slow STDP pairs** | `stdp_trainer.py:217-224` | **❌ NO** | MED | **NEW V9** |
| **SN-24: momentum blend** | `stdp_trainer.py:461` | **❌ NO** | MED | **NEW V9** |
| **G-47: lerp_ EMA** | `stdp_trainer.py:464` | **❌ NO** | MED | **NEW V9** |
| **SN-22.3: per-concept elr** | `stdp_trainer.py:562-579,593` | **❌ NO** | MED | **NEW V9** |
| **AM-32: graph_cache LRU** | `crystal_generator.py:558-563` | **❌ NO** | MED | **NEW V9** |
| TrainingPipeline | `train_full.py` | **❌ NO** | HIGH | ↔ unchanged |
| **TN-25: rescore fix** | `train_full.py:434-435` | **❌ NO** | MED | **NEW V9** |
| **TN-27: global_step** | `train_full.py:489,573,749,765` | **❌ NO** | MED | **NEW V9** |
| RNGRegistry (all) | `rng_registry.py` | **❌ NO** | HIGH | ↔ unchanged |
| AdaptiveErrorTracker (dict methods) | `adaptive_error_tracker.py` | **❌ NO** | MED | ↔ unchanged |
| GPU STDP noise_scale | `stdp_trainer.py:389-390` | **❌ NO** | MED | ↔ unchanged |
| GPU STDP destab | `stdp_trainer.py:435-452` | **❌ NO** | MED | ↔ unchanged |
| GPU STDP EMA branch | `stdp_trainer.py:463-465` | **❌ NO** | MED | ↔ unchanged |

**Total high-risk uncovered methods: 5** (unchanged)  
**Total medium-risk uncovered (new V9): 7** ⬆️  
**Total uncovered risk items: 22** (V8: 15 → V10: 22) ⬆️

---

## 8. Summary

| Metric | V9 | V10 | Δ |
|---|---|---|---|
| QN-32..QN-40 implemented | 0/9 | **0/9** | ⬜ no progress |
| NEW V9 untested changes | — | **9 areas** | ⬇️⬇️ |
| New tests added in V9 commits | — | **0 tests** | ⬜ |
| Cumulative QN tests implemented | 0/17 | **0/17** | ⬜ |
| STR | ~52% | **~50%** | ⬇️ −2pp |
| Total uncovered risk items | 15 | **22** | ⬆️ +7 |
| HIGH-risk uncovered | 5 | **5** | ⬜ |
| MEDIUM-risk uncovered (new) | 10 | **17** | ⬆️ +7 |

- **0 new tests** despite 12 fixes in 3 V9 commits
- **9 new untested code areas** introduced: SN-24 momentum blend, SN-25 slow STDP, AM-32 LRU, G-47 lerp_, SN-22.3 per-concept elr, TN-25 rescore, TN-27 global_step
- **SN-22.3 `sum` vs `mean` change** changes numerical behavior — **potential parity regression**
- **8 proposed V10 test suites** (QN-41..QN-48, ~20 individual tests)
- **Priority order**: QN-42 (slow STDP) → QN-41 (momentum blend) → QN-45 (per-concept elr) → QN-43 (LRU) → QN-44 (EMA lerp_) → QN-47 (global_step) → QN-46 (rescore) → QN-48 (meta shape edge)

**Safety verdict: DECLINING. STR 50%, 22 uncovered items, 9 new untested V9 changes, `sum`/`mean` parity gap in SN-22.3.**
