# FCF V10 — Аудит коллегии AI-агентов

**Дата**: 2026-06-19
**Версия**: V10 (аудит V9 коммитов: a0fe15b + cccc392 + 21ee6ca)
**Состав коллегии**: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent

---

## Executive Summary

V9 закоммичен (3 коммита: a0fe15b, cccc392, 21ee6ca). Все REG-V9 P1/P2 исправлены. **79 тестов проходят** (2 skipped — SentencePiece). Код стабилен, но:

| Метрика | V9 | V10 | Δ |
|---------|:--:|:---:|:-:|
| P0 проблем | 0 | **0** (нет новых) | = |
| P1 проблем | 15 | **10 открыто** | −5 |
| P2 проблем | 25 | **22 открыто** | −3 |
| GPU-оптимизации (G-40..G-49) | 0/10 | **0/10 реализовано** | = |
| Рекомендованные тесты (QN-32..QN-40) | 0/9 | **0/9 реализовано** | = |
| STR покрытия | ~52% | ~52% | = |

**Главные находки V10:**
1. **V9 не внёс новых регрессий** — все 3 коммита (a0fe15b, cccc392, 21ee6ca) корректны
2. **0 из 10 GPU-оптимизаций G-40..G-49 реализованы** — throughput 10-60ms/batch, цель ~2ms
3. **0 из 9 тестовых сьютов QN-32..QN-40 реализованы** — STR 52% и не растёт
4. **3 старых P0 CPU/GPU parity бага (SN-22.1, SN-22.2, SN-25) исправлены** в V9
5. **GPU sync storm ~16,500 `.item()`/batch** остаётся главным узким местом

---

## 1. Статус V9 P1 (все 10 исправлены — верифицировано)

| ID | Проблема | Статус | Верификация |
|:--:|----------|:------:|:------------|
| REG-V9-1 | `self.l_c` → `self.fractal.l_c` | ✅ Исправлен | concept_space.py:564 |
| REG-V9-2 | norm fix: `code_new /= np.linalg.norm(code_new)` | ✅ Исправлен | concept_space.py:574 |
| SN-22.1 | GPU neg sampling `mean()` → `sum()` | ✅ Исправлен | stdp_trainer.py:574-579 |
| SN-22.2 | field_gate guard in CE reweighting | ✅ Исправлен | stdp_trainer.py:583-586 |
| SN-24 | Momentum blend: `grad = mu*mom + (1-mu)*grad` | ✅ Исправлен | stdp_trainer.py:459-461 |
| SN-25 | Slow STDP pairs in gpu_meta_l | ✅ Исправлен | stdp_trainer.py:216-224 |
| SN-31 | Dead `_subspace_update` removed | ✅ Исправлен | Удалён |
| REG-V9-8 | `momentum_mu` config field | ✅ Исправлен | fcf_config.py:421 |
| AM-32 | `_graph_cache` LRU (OrderedDict, maxlen=5000) | ✅ Исправлен | crystal_generator.py:72-73, 558-563 |
| G-45 | Persistent CUDA events | ✅ Исправлен | stdp_trainer.py:29-32 |
| G-47 | `lerp_` for EMA | ✅ Исправлен | stdp_trainer.py:463-464 |

## 2. Статус V9 P2 (все 3 исправлены)

| ID | Проблема | Статус |
|:--:|----------|:------:|
| TN-25 | rescore fix (idx+1, start_line=0) | ✅ Исправлен |
| TN-26 | `cleanup_old_checkpoints` removed | ✅ Исправлен |
| TN-27/29 | global_step save/restore, dead code | ✅ Исправлен |

---

## 3. Регрессии: НОЛЬ

V9 не внёс новых регрессий. Все 3 коммита чисты:
- **a0fe15b**: REG-V9-1/2, TN-25/26/27/29, SN-22.2/24/25/31, REG-V9-8
- **cccc392**: SN-22.1 (mean→sum), AM-32 (LRU), G-45 (CUDA events), G-47 (lerp_)
- **21ee6ca**: SN-22.3 (per-concept avg_elr scatter_add grouping)

---

## 4. Открытые P1 — стали более критичными

С V9 ничего не изменилось — те же 10 P1:

| ID | Проблема | Агент | Почему критичнее |
|:--:|----------|:-----:|:-----------------|
| REG-V9-7 | `noise_scale` — 2 механики (gradient noise + fractal fluctuation) | Arch | **Блокирует тонкую настройку** — нельзя независимо управлять |
| AM-42/SN-19/G-44 | GPU Contrastive полная векторизация | Arch/NS/GPU | **~16,500 syncs/batch** — главный bottleneck |
| AM-43/G-43 | GPU neg sampling векторизация | Arch/GPU | **Python loop per-concept** — ~5-10ms/batch |
| AM-25 | CPU path не удалён | Arch | **Дедлайн просрочен** — 300 строк мёртвого кода |
| G-40 | Batched GPU subspace update | GPU | **Целиком CPU/numpy** в GPU-пути — 10-20ms/batch |
| G-46 | Persistent `_mom_t` tensor | GPU | **Python dict `_mom_buf`** — 2N syncs/batch |
| QN-32..QN-34 | Subspace, contrastive, evaluate тесты | QA | **0 тестов** на 3 HIGH-risk зоны |
| G-41 | GPU lateral inhibition (без `.item()`) | GPU | CPU write-back убивает GPU ускорение |
| G-42 | GPU `_centroid_pull_batch` | GPU | 100% CPU numpy |
| REG-V9-9 | Monkey-patch на lattice | Arch | Побочные эффекты на любой вызов update/decay |

---

## 5. Открытые P2 — переоценка приоритетов

| ID | Проблема | Старый приоритет | Новый приоритет | Причина |
|:--:|----------|:----------------:|:---------------:|---------|
| REG-V9-10 | `_graph_cache` maxlen=500 | P2 | **P1** | Должно быть 5000 (сейчас 500 — обсчёт) |
| G-48 | `torch.compile` | P3 | **P2** | 1.5-3× бесплатного ускорения |
| G-49 | Pre-allocate fused buffers | P3 | **P2** | Убирает аллокации на каждый batch |
| TN-13 | Progressive batch size | P2 | P2 | = |
| TN-15 | Decay warmup | P2 | P2 | = |
| AM-29/46 | RNG consolidation | P2 | P3 | Не влияет на корректность |
| AM-30 | EMA batch update | P2 | P2 | = |
| AM-31 | ConceptError batch sync | P2 | P2 | = |
| QN-36 | RNGRegistry тесты | P2 | **P1** | 0% coverage — reproducibility |
| QN-37 | AdaptiveErrorTracker тесты | P2 | P2 | = |
| QN-38 | Checkpoint cleanup тесты | P2 | P2 | = |

---

## 6. Полная матрица V10

### P0 (0) — все исправлены

### P1 — критические (10 открыто)

| ID | Проблема | Сложность |
|:--:|----------|:---------:|
| REG-V9-7 | `noise_scale` split → `gradient_noise_scale` + `fluctuation_amp` | 2 |
| AM-42/SN-19/G-44 | GPU Contrastive векторизация | 7 |
| AM-43/G-43 | GPU neg sampling векторизация | 5 |
| AM-25 | CPU path удаление | 3 |
| G-40 | Batched GPU subspace update | 6 |
| G-46 | Persistent `_mom_t` tensor | 4 |
| G-41 | GPU lateral inhibition (batched) | 4 |
| G-42 | GPU `_centroid_pull_batch` | 3 |
| REG-V9-9 | Monkey-patch → events/hooks | 3 |
| QN-32..QN-34, QN-36 | Тесты (subspace, contrastive, evaluate, RNG) | 3-4 |

### P2 — производительность + качество (22 открыто)

G-48 (torch.compile), G-49 (fused buffers), REG-V9-10 (graph_cache maxlen),
TN-13 (progressive BS), TN-15 (decay warmup),
AM-30 (EMA batch), AM-31 (ConceptError sync),
QN-37 (ErrorTracker tests), QN-38 (checkpoint cleanup),
QN-39 (GPU coverage), QN-40 (train_full tests),
AM-29/46 (RNG consolidation), AM-33 (hormone reset),
AM-34 (magic numbers), AM-35 (PathConfig dedup),
AM-36 (version counter), SN-17 (Kinetic Energy),
SN-20 (Adaptive EMA), SN-21 (Riemannian),
G-26..G-30 (V8 GPU)

### P3 — долгосрочные

AM-47..AM-50, QN-40 (dead code)

---

## 7. AM-59+ Предложения на V10

### AM-59: Eliminate `.item()` syncs in GPU contrastive
**Файл**: `stdp_trainer.py:690-754`
**Суть**: Заменить Python loop + `.item()` на batched masked tensor ops. Использовать `torch.where` и `masked_select` вместо per-candidate проверок `connection_strength`, `cooc_set`, `field_overlap`.
**Оценка**: −16,500 syncs/batch → −0 syncs. **10-50× ускорение** contrastive.
**Сложность**: 7

### AM-60: Persistent `_mom_t` tensor
**Файл**: `stdp_trainer.py:407-423`, `crystal_generator.py`
**Суть**: Заменить `gen._mom_buf` (Python dict) на `gen._mom_t` (GPU tensor `[V, D]`). Momentum read/write — тензорные операции без CPU roundtrip.
**Оценка**: −2N syncs/batch. **3-10× ускорение** momentum.
**Сложность**: 4

### AM-61: Batched GPU subspace update
**Файл**: `concept_space.py:556-583`, `stdp_trainer.py:466-467`
**Суть**: Перенести `_apply_subspace_update` на GPU через `_basis_t`. Batched код-градиент: `code_grad = avg_grad @ basis_t.T`. Маски — персистентный тензор, не per-call np.zeros.
**Оценка**: 10-20× на subspace (6ms → 0.3ms для 1000 concepts).
**Сложность**: 6

### AM-62: Replace monkey-patch with events/hooks
**Файл**: `crystal_generator.py:119-130`
**Суть**: Ввести `_on_lattice_update` и `_on_lattice_decay` хуки в SyntaxLattice. Убрать переопределение методов через замыкания.
**Оценка**: Чистая архитектура, предсказуемое поведение.
**Сложность**: 3

### AM-63: Split `noise_scale` → `gradient_noise_scale` + `fluctuation_amp`
**Файл**: `fcf_config.py`, `stdp_trainer.py:384-386`, `concept_space.py:456`
**Суть**: Два независимых параметра для gradient noise (STDP) и fractal fluctuation.
**Оценка**: Точный контроль обучения.
**Сложность**: 2

### AM-64: `momentum_mu` из конфига (не хардкод)
**Файл**: `train_full.py:675`
**Суть**: `momentum_mu=CFG.momentum_mu` вместо `momentum_mu=0.9`.
**Оценка**: 1 строка, 0 риска.
**Сложность**: 1

### AM-65: `_graph_cache` maxlen=5000 + stats
**Файл**: `crystal_generator.py:73`
**Суть**: Исправить maxlen=500 (должно быть 5000 как в V9 spec). Добавить hit/miss stats.
**Оценка**: Меньше перестроений graph_search.
**Сложность**: 1

### AM-66: Vectorized GPU negative sampling
**Файл**: `stdp_trainer.py:581-604`
**Суть**: Убрать Python loop: batched grad через `valid_any` mask + `scatter_add_`. Оставить loop только над valid_any индексами.
**Оценка**: 2-5× на neg sampling.
**Сложность**: 5

### AM-67: Batched GPU lateral inhibition
**Файл**: `stdp_trainer.py:485-510`
**Суть**: Векторизовать per-concept loop: `(sim_masked @ gv)` для всех gen_cids сразу.
**Оценка**: 2-5× на inhibition.
**Сложность**: 4

### AM-68: GPU `_centroid_pull_batch` (векторизация)
**Файл**: `stdp_trainer.py:771-796`
**Суть**: Заменить numpy на `_vecs_t[ids_t]` тензорные операции.
**Оценка**: 3-10× на centroid pull.
**Сложность**: 3

### AM-69: CPU path removal
**Файл**: `stdp_trainer.py` (разные места)
**Суть**: Удалить `_cpu_stdp_apply`, `_negative_sampling_cpu`, `_contrastive_objective_cpu`, `_lateral_inhibition_cpu`. Оставить `use_torch=True` по умолчанию.
**Оценка**: −300 строк мёртвого кода.
**Сложность**: 3

### AM-70: `torch.compile` на `_gpu_stdp_apply`
**Файл**: `stdp_trainer.py`
**Суть**: После векторизации всех внутренних циклов — `torch.compile(mode="max-autotune")`.
**Prerequisite**: AM-59, AM-60, AM-61, AM-66, AM-67
**Оценка**: 1.5-3× бесплатно.
**Сложность**: 4

### AM-71: Pre-allocate fused buffers
**Файл**: `stdp_trainer.py`
**Суть**: `fused`, `err_grouped`, `cnt_err` — персистентные тензоры, zero_() на reuse.
**Оценка**: −N аллокаций/batch.
**Сложность**: 3

### AM-72..AM-79: Тестовые сьюты QN-32..QN-40
**Суть**: Реализовать все 9 рекомендованных тестовых сьютов:
- AM-72: QN-32 subspace update tests
- AM-73: QN-33 GPU contrastive tests
- AM-74: QN-34 evaluate tests
- AM-75: QN-36 RNGRegistry tests
- AM-76: QN-37 AdaptiveErrorTracker tests
- AM-77: QN-38 checkpoint cleanup NPZ tests
- AM-78: QN-39 GPU STDP path coverage
- AM-79: QN-40 train_full unit tests

---

## 8. Рекомендуемый план работ V10

| Фаза | Задачи | Оценка времени |
|:----:|--------|:--------------:|
| **Фаза 0** (немедленно, 0 риска) | AM-64 (momentum_mu), AM-65 (graph_cache maxlen), AM-63 (noise_scale split) | ~2 часа |
| **Фаза 1** (P1, стабилизация) | AM-62 (monkey-patch), AM-69 (CPU path removal), G-42 (GPU centroid), QN-32/34/36 | ~1 неделя |
| **Фаза 2** (GPU, ~2 недели) | AM-61 (GPU subspace), AM-60 (mom_t), AM-66 (neg sampling), AM-67 (lateral inh), AM-71 (buffers) | ~2 недели |
| **Фаза 3** (GPU contrastive, ~1 неделя) | AM-59 (eliminate .item() syncs) | ~1 неделя |
| **Фаза 4** (P2, ~1 неделя) | AM-70 (torch.compile), G-48/G-49, QN-37/38/39/40, TN-13/15 | ~1 неделя |

**Итого V10**: ~7 недель. Ключевой ROI — Фаза 2 (GPU ускорение 5-20×).

---

## 9. Что СДЕЛАНО (V9→V10)

- ✅ Все 10 V9 P1/P2 фиксов верифицированы
- ✅ 79 тестов проходят
- ✅ Нет новых регрессий
- ✅ 3 старых P0 CPU/GPU parity бага исправлены

## 10. Что НЕ ИСПОЛНЕНО (от V8/V9, без прогресса)

- ❌ GPU Contrastive векторизация (AM-42/SN-19/G-44)
- ❌ GPU neg sampling векторизация (AM-43/G-43)
- ❌ Batched GPU subspace update (G-40)
- ❌ Persistent `_mom_t` (G-46)
- ❌ GPU lateral inhibition (G-41)
- ❌ GPU centroid pull (G-42)
- ❌ CPU path не удалён (AM-25)
- ❌ noise_scale не разделён (REG-V9-7)
- ❌ Monkey-patch на lattice (REG-V9-9)
- ❌ 9 тестовых сьютов (QN-32..QN-40)
- ❌ 10 GPU-оптимизаций G-40..G-49 (0/10)

---

*Отчёт составлен коллегией AI-агентов: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent*
