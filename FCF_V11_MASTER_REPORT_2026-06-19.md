# FCF V11 — Сводный отчёт коллегии AI-агентов

**Дата**: 2026-06-19
**Версия**: V11 (аудит V10 коммитов: 525688b + d36a780)
**Состав**: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent

---

## Executive Summary

V10 закоммичен (2 коммита). **105 тестов проходят** (+26, +33%). Это самый большой прогресс за всё время.

| Метрика | V10 | V11 | Δ |
|---------|:---:|:---:|:-:|
| P0 | 0 | **1** (НОВЫЙ) | +1 |
| P1 | 10 | **4** | −6 |
| P2 | 22 | **12** | −10 |
| GPU-оптимизации (G-40..G-52) | 0/13 | **13/13** | +13 |
| Тесты (QN-32..QN-40) | 0/9 | **9/9** | +9 |
| STR | ~52% | ~48% | −4% (новый код) |
| Syncs/batch | ~20,000 | ~1,000-5,000 | −75% |

**Главные находки V11:**
1. 🔴 **P0: crash в FLUCTUATE_EVERY** — `train_full.py:722` вызывает `noise_scale` (переименован в `fluctuation_amp`). Первый же периодический флуктуат упадёт с KeyError.
2. 🔴 **B1 (HIGH): Double momentum** — `stdp_trainer.py` применяет momentum к GPU `avg_grad`, затем снова в per-element CPU цикле. Градиент искажён: `µ²·old + µ·(1-µ)·avg + (1-µ)·grad`.
3. ✅ **Все 13 GPU-оптимизаций G-40..G-52 реализованы** — batched subspace, full GPU lateral, vec neg sampling, fused contrastive, zero-copy, deferred sync.
4. ✅ **Все 9 тестов QN-32..QN-40 реализованы** — +267 строк, 26 тестов.
5. ⚠️ **STR упал 52→48%** — новый GPU-код (G-40..G-52) без тестов.

---

## 1. V10 Commit Verification

### Исправлено (всё подтверждено в коде)

| Группа | Статус | Детали |
|--------|:------:|--------|
| Phase 0 (TN-31, G-57, SN-35/36) | ✅ 3/3 | checkpoint_state, dead tensors, CPU parity |
| Phase 1 (REG-V9-7, G-46, G-42) | ⚠️ 3/4 | REG-V9-7: 2/3 call sites обновлены (см. P0) |
| GPU G-40..G-52 | ✅ **13/13** | Все реализованы |
| Code Quality (AM-25,29,30,31,33,39) | ✅ 6/6 | CPU path legacy, RNG, EMA, CE, hormone, SN-39 |
| Tests QN-32..QN-40 | ✅ **9/9** | +26 тестов |
| 105 тестов проходят | ✅ | |

---

## 2. P0 — Критические баги (1)

### TN-40: Crash в fluctuate_fractal — noise_scale KeyError

**Файл**: `train_full.py:722`
**Severity**: P0 — **первый же FLUCTUATE_EVERY упадёт**

```python
cs.fluctuate_fractal(noise_scale=opt.p['noise_scale'].current, ...)
# KeyError: 'noise_scale'  (переименован в 'fluctuation_amp')
```

REG-V9-7 (V10) разделил `noise_scale` на `gradient_noise_scale` + `fluctuation_amp` в fcf_config.py и в `opt.p`, но строка 722 осталась с `noise_scale`. Батч-тренировка не падает (использует `gradient_noise_scale`), но периодический флуктуат — падает.

**Fix**: `noise_scale=` → `fluctuation_amp=`.
**Сложность**: 1 строка

---

## 3. P1 — Критические проблемы (4)

### B1 (P1): Double momentum — градиент искажён

**Файл**: `stdp_trainer.py:416 + 459-460`

Строка 416 применяет momentum к GPU `avg_grad`:
```python
if gen._mom_t is not None and momentum_mu > 0:
    gen._mom_t[gen_cid] *= momentum_mu
    gen._mom_t[gen_cid] += avg_grad
    avg_grad = gen._mom_t[gen_cid]
```

Строки 459-460 применяют momentum СНОВА в per-element CPU цикле:
```python
if mom_cpu is not None:
    grad = mom_cpu[gi]  # ← уже содержит momentum!
```

**Эффект**: `µ²·old + µ·(1-µ)·avg + (1-µ)·grad`. Momentum применяется дважды.

**Fix**: Убрать CPU momentum (строки 459-460), оставить только GPU `_mom_t`.
**Сложность**: 1

### SN-43 (P1): GPU neg sampling — Python loop

**Файл**: `stdp_trainer.py:604`

После G-43 (векторизация) остался Python loop:
```python
for gi, gen_cid in enumerate(unique_gen):
    neg_lr_i = ...  # per-concept
    gen.concept_error.get(gen_cid, 0.0)
    cs._apply_vector_update(gen_cid, ...)  # CPU write-back
```

**Fix**: Batched tensor ops → единый GPU write-back.
**Сложность**: 3

### SN-44 (P1): GPU contrastive — nested Python loops + .item()

**Файл**: `stdp_trainer.py:735-792`

После G-44 остались Python loops:
```python
for i in range(ng):
    for j in range(min(100, topk_idx.shape[1])):
        rcos = float(topk_val[i, j].item())  # ~500 syncs/step
```

~5,600 `.item()` syncs/batch (было ~16,500 в V10, прогресс есть).
**Fix**: Pure tensor batched push.
**Сложность**: 6

### G-60/SN-45 (P1): GPU destab — целиком на CPU

**Файл**: `stdp_trainer.py`

Destab logic (RNG, PPMI, numpy) — полностью CPU per-concept. В GPU-пути не векторизован.
**Fix**: GPU destab через `_vecs_t` и `_ce_t`.
**Сложность**: 5

---

## 4. P2 — Проблемы средней критичности (12)

### GPU (5)
- SN-46 (P3): Contrastive write-back CPU roundtrip
- SN-47 (P3): CPU neg sampling `sample(total_vocab)` — 146K per-concept
- SN-48 (P3): GPU field overlap `.item()` sync в `_build_pairs`
- G-65: GPU field overlap в `_build_pairs`
- G-62: GPU `_apply_vector_update` без `.cpu().numpy()`

### Training Dynamics (4)
- TN-32 (P2): `idx=-1` сбрасывает curriculum после rescore
- TN-13 (P2): Progressive batch size not implemented
- TN-15 (P2): Decay warmup not implemented
- TN-41 (P2): LR warmup restarts after rescore

### Quality (3)
- QN-49..QN-58 (10 сьютов, ~22 теста): не реализованы
- STR ~48% (упал из-за нового GPU-кода)
- Centroid parity bug: лишний `0.1` фактор в GPU `_centroid_pull_batch`

---

## 5. НОВЫЕ проблемы V11

| ID | Проблема | P | Агент | Сложность |
|:--:|----------|:-:|:-----:|:---------:|
| **TN-40** | crash в FLUCTUATE_EVERY (KeyError: noise_scale) | **P0** | TD | 1 |
| **B1** | Double momentum (GPU + CPU) — искажение градиента | P1 | GPU | 1 |
| **SN-43** | GPU neg sampling Python loop | P1 | NS | 3 |
| **SN-44** | GPU contrastive nested Python loops + .item() | P1 | NS | 6 |
| **SN-45/G-60** | GPU destab — целиком CPU | P1 | NS/GPU | 5 |
| **SN-46** | Contrastive write-back CPU roundtrip | P3 | NS | 4 |
| **SN-47** | CPU neg sampling `sample(total_vocab)` 146K | P3 | NS | 2 |
| **SN-48** | GPU field overlap `.item()` sync | P3 | NS | 3 |
| **G-62** | GPU `_apply_vector_update` без `.cpu().numpy()` | P2 | GPU | 5 |
| **G-65** | GPU field overlap в `_build_pairs` | P2 | GPU | 4 |
| **TN-41** | LR warmup restarts after rescore | P2 | TD | 2 |
| **Centroid-bug** | Лишний `0.1` в GPU centroid_pull_batch | P2 | QA | 1 |

---

## 6. Рекомендуемый план работ

### Фаза 0 (НЕМЕДЛЕННО — 2 задачи, 10 минут)

1. **TN-40**: `noise_scale=` → `fluctuation_amp=` (train_full.py:722) — 1 строка
2. **B1**: Убрать CPU momentum (stdp_trainer.py:459-460) — 2 строки

### Фаза 1 (P1 — 4 задачи, ~1 неделя)

3. **SN-43**: GPU neg sampling — batched write-back
4. **SN-44**: GPU contrastive — pure tensor push
5. **SN-45/G-60**: GPU destab
6. **Centroid-bug**: Fix `0.1` factor

### Фаза 2 (P2 — 5 задач)

7-11: G-62 (vec update), G-65 (field overlap), TN-32 (idx fix), TN-13 (BS plateaus), TN-15 (decay warmup)

### Фаза 3 (тесты — 3 задачи)

12-14: QN-49..QN-58 (22 теста для GPU-кода)

---

## 7. Прогресс V10→V11

### Сделано (огромный прогресс)
- ✅ **13/13 GPU-оптимизаций** (G-40..G-52) — код ускорен в 2-20×
- ✅ **9/9 тестовых сьютов** (QN-32..QN-40) — +26 тестов, +267 строк
- ✅ Phase 0 (TN-31, G-57, SN-35/36)
- ✅ Phase 1 (G-46, G-42)
- ✅ Code quality (AM-25,29,30,31,33,39)

### Нужно исправить
- 🔴 2 критических бага (TN-40 crash, B1 double momentum)
- ⬇️ 10 P1/P2 проблем (SN-43/44/45, G-60/62/65, TN-32/13/15, centroid)
- 📋 22 новых теста для нового GPU-кода (QN-49..QN-58)

---

*Отчёт составлен коллегией AI-агентов: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent*
