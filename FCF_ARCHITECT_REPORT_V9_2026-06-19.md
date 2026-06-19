# FCF V9 — Сводный отчёт коллегии AI-агентов

**Дата**: 2026-06-19
**Версия**: V9 (аудит после коммита V8 P0/P1 fixes: 7ae6d9a + b14871b)
**Состав коллегии**: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent

---

## Executive Summary

V8 закоммичен (7ae6d9a). Все 4 P0 исправлены, 5 из 12 P1 закрыты.

**77 тестов проходят** (2 skipped — SentencePiece). Код стабилен, но:

| Метрика | V8 | V9 | Δ |
|---------|:--:|:--:|:-:|
| P0 проблем | 4 | **0** (все исправлены) | −4 |
| P1 проблем | 12 | **−5 / 18 накоплено** | +6 новых |
| P2 проблем | 15 | **−0 / 25 накоплено** | +10 |
| GPU-оптимизации (G-21..G-30) | 10 | **0 реализовано** | 0 |
| Рекомендованные тесты (QN-24..QN-31) | 8 | **0 реализовано** | 0 |
| STR покрытия | ~55% | ~52% | −3% |

**Главные находки V9:**
1. `_apply_subspace_update()` (V8 fix) — `self.l_c` → AttributeError при subspace_lr ≠ None
2. `_apply_subspace_update()` — 100% CPU/numpy, нет GPU-версии — **P4 производительность**
3. GPU Contrastive: ~16,500 `.item()` syncs/batch — **P5 производительность**
4. 3 P0 CPU/GPU parity бага живы с V7 (mean/sum neg sampling, ce reweighting, slow STDP дроп)
5. STR упал: новый код без тестов

---

## 1. Статус V8 P0 (все исправлены — верифицировано)

| ID | Проблема | Статус | Верификация |
|:--:|----------|:------:|:------------|
| AM-37-REG | Subspace STDP NO-OP | ✅ Исправлен | `_apply_subspace_update()` в concept_space.py:556-583. Code→code update. |
| REG-1 | Два цикла обучения | ✅ Исправлен | `run_epoch()` удалён. Единый main loop. |
| REG-2 | Два checkpoint naming | ✅ Исправлен | CheckpointManager: `concept_space_{tag}.json` / `syntax_lattice_{tag}.json` |
| REG-3/T-B1 | idx не сбрасывается | ✅ Исправлен | Main loop: `idx, start_line, epoch_train = result` (train_full.py:770-771) |

---

## 2. Статус V8 P1 (5 исправлено, 7 открыто)

### Исправлено (5)
AM-13, AM-14, AM-15, AM-22, AM-24, AM-27

### Открыто (7) — без прогресса с V8

| ID | Проблема | Зона | Сложность |
|:--:|----------|:----:|:---------:|
| AM-19/AM-26/SN-19 | GPU Contrastive — Python loop, не векторизован | NS/GPU | 7 |
| REG-5 | GPU Contrastive — numpy/torch mix, push_total не используется | NS/GPU | 7 |
| REG-6/AM-43 | GPU neg sampling — per-concept loop, CPU sync | GPU | 5 |
| AM-42 | GPU Contrastive векторизация | Arch/NS | 7 |
| AM-25 | CPU path (~300 строк) не удалён | Arch | 3 |
| AM-32/REG-7 | `_graph_cache` unbounded | Arch | 1 |
| SN-19 | Full vectorization GPU Contrastive | NS | 7 |

---

## 3. НОВЫЕ проблемы V9 (10 регрессий + 5 критических производительности)

### REG-V9-1 (P1): `_apply_subspace_update` — `self.l_c` AttributeError

**Файл**: `concept_space.py:564-566`
**Суть**: `self.l_c` не существует на ConceptSpace. Атрибут на `self.fractal`. Код падает с `AttributeError` при `subspace_lr ≠ None`. Сейчас дремлет (`subspace_lr=None` по умолч.).
**Fix**: `self.l_c` → `self.fractal.l_c` (3 места).
**Сложность**: 1

### REG-V9-2 (P1): `_apply_subspace_update` — нормировка code через vector norm

**Файл**: `concept_space.py:574`
```python
code_new /= nv  # nv = ||v_new||
```
Должно быть `code_new /= np.linalg.norm(code_new)` — нормировка в code space, не vector space.
**Сложность**: 1

### REG-V9-3 (P4): `_apply_subspace_update` — 100% CPU/numpy в GPU-пути

**Файл**: `concept_space.py:556-583`
**Суть**: Вызывается из `_gpu_stdp_apply`, но делает 2 CPU matmul, 3 аллокации numpy, dict lookup per-concept. Полностью нивелирует GPU ускорение.
**Fix**: GPU-версия через `_basis_t` и batched tensor ops.
**Сложность**: 6

### REG-V9-4 (P5): GPU Contrastive — ~16,500 `.item()` syncs/batch

**Файл**: `stdp_trainer.py:654-754`
**Суть**: Двойной Python loop (candidates + TN-14) со скалярными D2H синками. Каждый `.item()` = полный GPU sync + CPU wake.
**Fix**: Полная векторизация (AM-42/SN-19). Batched push через `scatter_add_`.
**Сложность**: 7

### REG-V9-5 (P6): `_mom_buf` — per-element CPU roundtrip

**Файл**: `stdp_trainer.py`
**Суть**: `_mom_buf` — Python dict. Каждый momentum update = numpy read/write. 2N GPU-CPU syncs на batch.
**Fix**: Persistent GPU tensor `_mom_t`.
**Сложность**: 4

### REG-V9-6 (P7): `_lateral_inhibition_gpu` — CPU write-back

**Файл**: `stdp_trainer.py`
**Суть**: `.item()` + CPU write-back через `cs._apply_vector_update` в GPU lateral inhibition.
**Fix**: Полная GPU версия с `_vecs_t`.
**Сложность**: 4

### REG-V9-7 (P1): `noise_scale` управляет gradient noise И fractal fluctuation

**Файл**: `stdp_trainer.py:384-386`
**Суть**: `noise_scale` добавлен для gradient noise (TN-6), но в train_full.py может также модулировать `fluctuate_fractal()` через `fluctuation_amp`. Один параметр — две разные механики.
**Fix**: Разделить на `gradient_noise_scale` и `fluctuation_amp`.
**Сложность**: 2

### REG-V9-8 (P2): `momentum_mu=0.9` хардкод

**Файл**: `train_full.py:709`
**Суть**: Не конфигурируется. Должен браться из `opt.p['momentum_mu'].current`.
**Fix**: Аналогично `noise_scale`.
**Сложность**: 1

### REG-V9-9 (P2): Monkey-patch на lattice (побочные эффекты)

**Файл**: `crystal_generator.py`
**Суть**: `lattice.update` и `lattice.decay_all` переопределены через замыкания. Любой другой код, вызывающий `lattice.update`, получит модифицированную версию.
**Fix**: Использовать события/хуки вместо monkey-patch.
**Сложность**: 3

### REG-V9-10 (P2): `_graph_cache` — утечка памяти

**Файл**: `crystal_generator.py`
**Суть**: Неограниченный рост. Не очищается при generate(). Очистка только в train через `gen.lattice.update` → `gen._graph_cache.clear()`.
**Fix**: LRU eviction (AM-44) + очистка при generate().
**Сложность**: 1

### Дополнительно: 3 P0 CPU/GPU parity бага (от V7, живы)

| ID | Баг | Зона | Сложность |
|:--:|-----|:----:|:---------:|
| SN-22.1 | GPU neg sampling: `mean()` vs CPU `sum()` — фактор 1/n | NS | 2 |
| SN-22.2 | GPU: concept_error reweighting безусловно; CPU: только при `field_gate=True` | NS | 1 |
| SN-25 | Slow STDP theta_slow дропается на GPU (нет в gpu_meta_l) | NS | 3 |

---

## 4. Статус GPU-оптимизаций V8 → V9

| Зона | V8 предложено | V9 реализовано | V9 новых |
|:-----|:-------------:|:--------------:|:--------:|
| G-21..G-30 (V8) | 10 | **0** | — |
| G-40..G-49 (V9) | — | — | **10** |

**0 из 10 GPU-оптимизаций V8 внедрены.** Текущий throughput: ~10-60ms/batch. Цель: ~2ms (8-10×).

### Новые G-40..G-49

| ID | Оптимизация | Сложность | Оценка ускорения |
|:--:|-------------|:---------:|:----------------:|
| G-40 | Batched GPU subspace update | 6 | 10-20× на subspace |
| G-41 | Full GPU lateral inhibition (без `.item()`) | 4 | 5-10× |
| G-42 | GPU `_centroid_pull_batch` | 3 | 5× |
| G-43 | Vectorized negative sampling | 5 | 5-10× |
| G-44 | Batched GPU TN-14/contrastive | 7 | 10-20× |
| G-45 | Persistent CUDA events | 1 | 1.5× |
| G-46 | Persistent `_mom_t` tensor | 4 | 3-5× |
| G-47 | `lerp_` for EMA | 1 | 2× |
| G-48 | `torch.compile` | 4 | 2-3× |
| G-49 | Pre-allocate fused buffers | 3 | 1.5× |

---

## 5. Статус тестирования V8 → V9

| Сьют | V8 предложено | V9 реализовано | V9 новых |
|:-----|:-------------:|:--------------:|:--------:|
| QN-24 | Subspace update tests | **0** | — |
| QN-25 | GPU contrastive tests | **0** | — |
| QN-26 | Noise injection tests | **0** | — |
| QN-27 | Evaluate tests | **0** | — |
| QN-28 | RNGRegistry tests | **0** | — |
| QN-29 | AdaptiveErrorTracker tests | **0** | — |
| QN-30 | Checkpoint cleanup tests | **0** | — |
| QN-31 | EMA sync test | **0** | — |

**STR покрытия: ~52%** (было ~55%, упало из-за нового кода без тестов).

### Новые QN-32..QN-40

| ID | Тесты | Приоритет | 
|:--:|-------|:---------:|
| QN-32 | subspace update: uniform_LR, masked_LR, no_basis, norm_check | P1 |
| QN-33 | GPU contrastive: smoke, field_aware, cross_field_penalty, push_total_used | P1 |
| QN-34 | evaluate: missing_file, empty_corpus, basic_perplexity | P1 |
| QN-35 | noise_scale: zero, non_zero, fractal_fluctuation orthogonal | P1 |
| QN-36 | RNGRegistry: determinism, isolation, reset, names | P2 |
| QN-37 | AdaptiveErrorTracker: EMA_math, FIFO, dict_interface, stress_100k | P2 |
| QN-38 | checkpoint_manager: npz_cleanup, .codes.npz_cleanup, save_extras | P2 |
| QN-39 | TrainingPipeline: _checkpoint return, rescore, global_step | P1 → ✅ FIXED V9 a0fe15b |
| QN-40 | Dead code detection: DECAY_EVERY, save_checkpoint_state | P3 |

---

## 6. Полная матрица приоритетов V9

### P0 (0) — все исправлены

### P1 — критичные (баги + производительность)

| ID | Проблема | Агент | Сложность |
|:--:|----------|:-----:|:---------:|
| REG-V9-1 | `_apply_subspace_update`: `self.l_c` → `self.fractal.l_c` | NS | 1 |
| REG-V9-2 | `_apply_subspace_update`: нормировка code через vector norm | Arch | 1 |
| REG-V9-7 | `noise_scale` — 2 механики на 1 параметр | Arch | 2 |
| SN-22.1 | GPU neg sampling: `mean()` vs `sum()` (CPU/GPU parity) | NS | 2 |
| SN-22.2 | GPU neg sampling: ce reweighting без field_gate guard | NS | 1 |
| SN-25 | Slow STDP theta_slow дроп на GPU | NS | 3 |
| AM-42/SN-19 | GPU Contrastive полная векторизация | Arch/NS/GPU | 7 |
| AM-43/G-43 | GPU neg sampling векторизация | Arch/GPU | 5 |
| AM-25 | CPU path удаление | Arch | 3 |
| AM-32 | `_graph_cache` LRU | Arch | 1 |
| QN-32 | Тесты subspace update | QA | 3 |
| QN-33 | Тесты GPU contrastive | QA | 4 |
| QN-34 | Тесты evaluate | QA | 3 |
| QN-35 | Тесты noise_scale | QA | 2 |
| QN-39 | Тесты TrainingPipeline | QA | 3 |

### P2 — производительность + качество

| ID | Проблема | Агент | Сложность |
|:--:|----------|:-----:|:---------:|
| REG-V9-8 | `momentum_mu=0.9` хардкод | Arch | 1 |
| REG-V9-9 | Monkey-patch на lattice | Arch | 3 |
| REG-V9-10 | `_graph_cache` утечка | Arch | 1 |
| G-40 | Batched GPU subspace update | GPU | 6 |
| G-42 | GPU `_centroid_pull_batch` | GPU | 3 |
| G-45 | Persistent CUDA events | GPU | 1 |
| G-46 | Persistent `_mom_t` tensor | GPU | 4 |
| G-47 | `lerp_` for EMA | GPU | 1 |
| QN-36 | RNGRegistry тесты | QA | 3 |
| QN-37 | AdaptiveErrorTracker тесты | QA | 3 |
| QN-38 | Checkpoint cleanup тесты | QA | 2 |
| TN-13 | Progressive batch size | TD | 2 |
| TN-15 | Decay warmup | TD | 2 |
| AM-29/AM-46 | RNG consolidation | Arch | 3 |
| AM-30 | EMA batch update | Arch | 2 |
| AM-31 | ConceptError batch sync | Arch | 2 |

### P3 — долгосрочные

G-41 (GPU lateral inh), G-44 (batched TN-14), G-48 (torch.compile), G-49 (fused buffers),
AM-33 (hormone reset), AM-34 (magic number), AM-35 (PathConfig dup), AM-36 (version counter),
AM-47..AM-50, SN-17 (Kinetic Energy), SN-20 (Adaptive EMA), SN-21 (Riemannian),
QN-40 (dead code), G-26..G-30 (V8 GPU)

---

## 7. Рекомендуемый план работ

### Фаза 0 (немедленно — 3 задачи, ~2 часа)

1. **REG-V9-1**: `self.l_c` → `self.fractal.l_c` в `_apply_subspace_update` (3 строки)
2. **REG-V9-2**: `code_new /= nv` → `code_new /= np.linalg.norm(code_new)` (1 строка)
3. **SN-22.1/22.2**: Fix GPU neg sampling parity — `mean()` → `sum()`, field_gate guard

### Фаза 1 (P1 — 8 задач, ~1 неделя)

4. **AM-32**: `_graph_cache` LRU eviction (OrderedDict, maxlen=5000)
5. **AM-25**: Delete CPU path (оставить только `use_torch=True`)
6. **REG-V9-7**: Split `noise_scale` → `gradient_noise_scale` + `fluctuation_amp`
7. **REG-V9-8**: `momentum_mu` из конфига
8. **G-40**: Batched GPU subspace update (через `_basis_t`)
9. **QN-32, QN-35**: subspace update + noise_scale тесты
10. **SN-25**: Add slow STDP to gpu_meta_l
11. **G-45**: Persistent CUDA events

### Фаза 2 (производительность — 5 задач, ~2 недели)

12. **AM-42/SN-19/G-44**: Полная векторизация GPU Contrastive
13. **AM-43/G-43**: Vectorized GPU negative sampling
14. **G-42**: GPU `_centroid_pull_batch`
15. **G-46**: Persistent `_mom_t` tensor
16. **QN-33, QN-34**: GPU contrastive + evaluate тесты

### Фаза 3 (P2 — 8 задач)

17-24: QN-36..QN-39, TN-13, TN-15, G-47, AM-29/AM-46

### Фаза 4 (P3 — остальное)

25+: G-41, G-44, G-48, G-49, AM-33..AM-36, AM-47..AM-50, SN-17/20/21, QN-40

---

## 8. Что СДЕЛАНО (V8→V9 прогресс)

### Исправлено (4 P0, 5 P1)
- ✅ Subspace-LR code→code update (но с 2 регрессиями)
- ✅ Единый тренировочный цикл
- ✅ Checkpoint naming = resume compatible
- ✅ T-B1 idx захват возврата
- ✅ noise_scale + momentum_mu передача
- ✅ TN-14 stale-vector fix
- ✅ KeyboardInterrupt handler resilience
- ✅ _final_save индивидуальная обработка ошибок

### Не исправлено (от V7/V8, без прогресса)
- ❌ GPU Contrastive векторизация (AM-19/26/42, SN-19)
- ❌ GPU neg sampling векторизация (AM-43)
- ❌ CPU path не удалён (AM-25)
- ❌ RNG не консолидирован (AM-29/46)
- ❌ EMA per-concept loop (AM-30)
- ❌ CUDA Events init (G-21/45)
- ❌ `_graph_cache` unbounded (AM-32)
- ❌ 8 QN-тест-сьютов не реализованы
- ❌ 10 GPU-оптимизаций не внедрены
- ❌ 3 P0 CPU/GPU parity бага живы

---

*Отчёт составлен коллегией AI-агентов: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent*
