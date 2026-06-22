# FCF Training-Dynamics Report V12 — V11 Post-Commit Audit

**Дата**: 2026-06-21
**Агент**: Training-Dynamics Agent
**Версия**: V12 (аудит 024f1aa → 1768f27 → a705223 → HEAD)

---

## 1. Проверка V11.2 + hotfixes (заявлены в V11 report, раздел 6)

| ID | Статус в V11 report | Статус в HEAD | Детали |
|:---|:-------------------:|:-------------:|--------|
| **TN-40** `noise_scale` KeyError | 🔴 P0 NEW | ✅ **FIXED** | `024f1aa` исправил `fluctuation_amp` на обеих строчках (742, 789) |
| **TN-41** LR warmup after rescore | 🟡 P2 NEW | ✅ **FIXED** | `idx + _lr_offset` через `_rescore_line` (строки 680-682) |
| **TN-13** Progressive BS plateaus | ❌ Not fixed | ✅ **IMPLEMENTED** | Плато-адаптивное удвоение при stuck≥3 (строка 797-801), НО: |
| — TN-13 regression | — | ⚠️ **NEEDS FIX** | BATCH_SIZE перезаписывается на каждой итерации (строка 692), удвоение не сохраняется |
| **TN-15** Decay warmup ramp | ❌ Not fixed | ✅ **IMPLEMENTED** | Рампа 0.998→target за `decay_warmup_lines` (строка 737-740) |
| **TN-32** rescore idx=-1 | ⚠️ Mitigated | ⚠️ **Mitigated** | `_rescore_cp` механизм есть, но `idx = -1` (строка 465) **остался** |
| **TN-34** opt.json naming | ⚠️ Mitigated | ⚠️ **Mitigated** | 4-уровневый fallback (строки 286-303), root cause не исправлен |
| **1768f27** 160s/batch slowdown | — | ✅ **FIXED** | `_torch_dirty` удалён из `_train()` (stdp_trainer.py:140-142) |
| **TN-4** Early Stopping | — | ✅ **FIXED** | `patience_counter`, `best_score`, `best_ckpt_name` (строка 446-452) |
| **TN-33** pipeline.global_step sync | ✅ FIXED | ✅ **FIXED** | `pipeline.global_step = global_step` перед `_checkpoint()` (строка 785) |
| **TN-31** checkpoint_state при чекпоинтах | ✅ FIXED | ✅ **FIXED** | Через `ckpt_state` в `ckpt_mgr.save()` + `_sync_save` пишет атомарно |

---

## 2. Проверка V11 TD-проблем (раздел 7 V11 report)

### TN-40 (P0): `noise_scale` → `fluctuation_amp` — ✅ FIXED

- `train_full.py:742`: `cs.fluctuate_fractal(fluctuation_amp=opt.p['fluctuation_amp'].current, ...)` ✅
- `train_full.py:789`: `cs.fluctuate_fractal(fluctuation_amp=opt.p['fluctuation_amp'].current, ...)` ✅
- `fcf_config.py:318-325`: оба параметра `gradient_noise_scale` + `fluctuation_amp` с адаптивными правилами ✅

### TN-41 (P2): LR warmup after rescore — ✅ FIXED

```python
# train_full.py:680-682
_lr_offset = pipeline._rescore_line if pipeline._rescore_line is not None else 0
gen.train_lr = get_lr(idx + _lr_offset)
```

`_rescore_line` сохраняет pre-rescore idx (строка 463). После rescore `get_lr(0 + 5000)` = пропуск warmup. **Корректно**.

### TN-42 (P3): `last_fluct_lines` desync — ❌ NOT FIXED

V11 предложил `last_fluct_lines = min(last_fluct_lines, idx)` при rescore. В коде нет. После rescore `idx=0`, `last_fluct_lines=5000` → `0 - 5000 = -4999` → fluctuate не срабатывает до `idx >= 5000 + FLUCTUATE_EVERY`.

### TN-43 (P2): `momentum_mu` не адаптивен — ❌ NOT FIXED

`train_full.py:720`: `momentum_mu=CFG.momentum_mu` (статическая константа 0.9 из `fcf_config.py:426`). Не использует `opt.p['momentum_mu'].current`.

### TN-44 (P2): `idx=-1` не удалён — ❌ NOT FIXED

`train_full.py:465`: `idx = -1; start_line = 0` всё ещё в коде. V11 предложил:
```python
epoch_train = epoch_train[:idx+1] + _rescore_lines(epoch_train[idx+1:], gen)
# idx не меняется
```
Не реализовано.

### TN-45 (P3): `pipeline.last_fluct_lines` мёртвый код — ❌ NOT FIXED

`TrainingPipeline.__init__` (строка 367) устанавливает `self.last_fluct_lines = 0`. Это поле нигде не читается — main loop использует глобальную `last_fluct_lines`.

### TN-13 (P2): Progressive BS plateaus — ⚠️ IMPLEMENTED WITH BUG

```python
# train_full.py:797-801
if pipeline.opt._full_stuck_counter >= 3:
    BATCH_SIZE = min(BATCH_SIZE * 2, CFG.batch_size_end * 4)
```

**Баг**: `BATCH_SIZE` вычисляется на **каждой итерации** цикла (строка 692):
```python
BATCH_SIZE = int(CFG.batch_size_start + (CFG.batch_size_end - CFG.batch_size_start) * _eff_cp)
```

Плато-удвоение применяется в `_checkpoint()`, но на следующей же строке BATCH_SIZE возвращается к `bs_curve()`. Эффект: удвоение действует ровно на 1 batch (пока буфер не заполнится по новому BATCH_SIZE). Для 32→64 это даёт 1 batch по 64 строки вместо 32, после чего всё сбрасывается.

### TN-15 (P2): Decay warmup ramp — ✅ IMPLEMENTED

```python
# train_full.py:737-740
_decay_pct = min(idx / max(CFG.decay_warmup_lines, 1), 1.0)
_decay_target = opt.p['decay_rate'].current
_decay_warmup = 0.998 + (_decay_target - 0.998) * _decay_pct
```

**Новый баг (TN-49)**: `_decay_pct` использует сырой `idx`, не `idx + _lr_offset`. После rescore рампа стартует с 0.998 заново, хотя LR остаётся на pre-rescore уровне.

---

## 3. Новые проблемы V12

### TN-46 (P2): TN-13 plateau doubling ineffective

**Файл/строка**: `train_full.py:692, 797-801`

**Суть**: Механизм из V11 report TN-13 реализован, но **не работает**:
1. На чекпоинте при stuck≥3: `BATCH_SIZE = min(BATCH_SIZE * 2, CFG.batch_size_end * 4)`
2. Следующая строка цикла (строка 692): `BATCH_SIZE = int(CFG.batch_size_start + ... * _eff_cp)` — **перезаписывает** удвоенное значение
3. Единственный эффект: один batch с удвоенным размером (если буфер успел заполниться по `CHECKPOINT_EVERY`)

**Fix**: Использовать отдельную переменную `plateau_bs`, не перезаписываемую основной кривой:
```python
# вне цикла:
_plateau_bs = 0
# в checkpoint:
if pipeline.opt._full_stuck_counter >= 3:
    _plateau_bs = min(BATCH_SIZE * 2, CFG.batch_size_end * 4)
# в строке 692:
BATCH_SIZE = int(CFG.batch_size_start + (CFG.batch_size_end - CFG.batch_size_start) * _eff_cp) + _plateau_bs
```

### TN-47 (P2): `static_fluctuation` — нет обратной связи от метрик

**Файл/строка**: `train_full.py:742-746`

**Суть**: Периодический `fluctuate_fractal` (каждые `FLUCTUATE_EVERY` строк) выполняется независимо от состояния модели:
- Параметры `fluctuation_amp`, `decay`, `repel_strength` адаптируются через `ParameterOptimizer` раз в чекпоинт
- Но **между чекпоинтами** (5000 строк) значения фиксированы
- Если cos collapsed (mean_cos > 0.05), а `fluctuation_amp` ещё не адаптирован — модель остаётся в collapsed состоянии до следующего чекпоинта

**Fix**: Передавать среднюю косинусную близость как feedback в `fluctuate_fractal`:
```python
cs.fluctuate_fractal(fluctuation_amp=opt.p['fluctuation_amp'].current,
                     decay=_decay_warmup,
                     repel_strength=opt.p['repel_strength'].current,
                     current_cos=last_cos_sim[0],  # NEW
                     generator=gen)
```

### TN-48 (P2): `field_gate` не в `opt.p`

**Файл/строка**: `train_full.py:717`

```python
neg_lr_ratio=CFG.neg_lr_ratio, field_gate=CFG.field_gate,
```

`field_gate=True/False` — бинарный переключатель. Но для gate-based contrastive objective (SN-28) это критичный параметр. Если `field_gate=False`:
1. `_contrastive_objective_cpu/gpu` — не применяет CE-reweighting
2. Контрастивный push ослабевает

**Fix**: Сделать `field_gate` динамическим порогом (float 0.0–1.0) через `opt.p`:
```python
ParamDef('field_gate', 0.0, 1.0, 1.0, 0.05, rules=[
    AdaptRule('acc1_plateau', 'field_gate', 'shift', -0.05),
    AdaptRule('cos_trend > 0.001 and mean_cos > 0.01', 'field_gate', 'shift', 0.05),
])
```

### TN-49 (P2): `decay_warmup` не учитывает rescore

**Файл/строка**: `train_full.py:738`

```python
_decay_pct = min(idx / max(CFG.decay_warmup_lines, 1), 1.0)
```

После rescore `idx=0`, но `_lr_offset` (строка 681) есть, а для decay — нет. Рампа restart с 0.998.

**Fix**:
```python
_decay_pct = min((idx + _lr_offset) / max(CFG.decay_warmup_lines, 1), 1.0)
```

### TN-50 (P3): `_rescore_lines` полный пересчёт

**Файл/строка**: `train_full.py:340-351`

V10 TN-35. Не оптимизирован. Каждый rescore (5K–10K строк) пересчитывает score для ВСЕХ оставшихся строк.

**Fix**: Кэшировать scores с инвалидацией по `concept_error.timestamp`. Rescore только строки, где error изменился >0.01.

---

## 4. Матрица статуса V12

| Область | Статус | Приоритет |
|:--------|:------:|:---------:|
| **TN-40** noise_scale→fluctuation_amp | ✅ FIXED | — |
| **TN-41** LR warmup after rescore | ✅ FIXED | — |
| **TN-13** Progressive BS (implemented) | ⚠️ **BUG: ineffective** | **P1** |
| **TN-15** Decay warmup ramp | ✅ FIXED | — |
| **TN-32** idx=-1 rescore | ⚠️ Mitigated (не исправлен) | P2 |
| **TN-34** opt.json naming | ⚠️ Mitigated (не исправлен) | P2 |
| **1768f27** 160s/batch slowdown | ✅ FIXED | — |
| **TN-31** checkpoint_state | ✅ FIXED | — |
| **TN-33** pipeline.global_step | ✅ FIXED | — |
| **TN-42** last_fluct_lines desync | ❌ Not fixed | P3 |
| **TN-43** momentum_mu не адаптивен | ❌ Not fixed | P2 |
| **TN-44** idx=-1 не удалён | ❌ Not fixed | P2 |
| **TN-45** pipeline.last_fluct_lines dead | ❌ Not fixed | P3 |
| **TN-35** _rescore_lines пересчёт | ❌ Not fixed | P3 |
| **TN-46** TN-13 plateau doubling неэффективен | 🔴 **НОВЫЙ** | **P1** |
| **TN-47** Нет feedback в fluctuate | 🟡 **НОВЫЙ** | P2 |
| **TN-48** field_gate не в opt.p | 🟡 **НОВЫЙ** | P2 |
| **TN-49** decay_warmup не учитывает rescore | 🟡 **НОВЫЙ** | P2 |
| **TN-50** _rescore_lines без кэша | 🟢 **НОВЫЙ** | P3 |

---

## 5. Предложения TN-45+

### TN-45 (P3): Удалить мёртвый код `pipeline.last_fluct_lines`

**Файл**: `train_full.py:367`
**Fix**: Удалить `self.last_fluct_lines` из `TrainingPipeline.__init__`.

### TN-46 (P1): Исправить TN-13 plateau doubling

**Файл**: `train_full.py:692, 797-801`
**Fix**: Вынести plateau-удвоение в отдельную переменную, не сбрасываемую `bs_curve()`.

### TN-47 (P2): Передавать `current_cos` в fluctuate

**Файл**: `train_full.py:742-746`
**Fix**: 
```python
cs.fluctuate_fractal(fluctuation_amp=..., current_cos=last_cos_sim[0], generator=gen)
```

### TN-48 (P2): `field_gate` через `opt.p`

**Файлы**: `fcf_config.py`, `train_full.py:717`
**Fix**: Создать `ParamDef('field_gate', 0.0, 1.0, 1.0, 0.05)` с адаптивными правилами.

### TN-49 (P2): `decay_warmup` с `_lr_offset`

**Файл**: `train_full.py:738`
**Fix**: `_decay_pct = min((idx + _lr_offset) / max(CFG.decay_warmup_lines, 1), 1.0)`

### TN-50 (P3): Кэш для `_rescore_lines`

**Файл**: `train_full.py:340-351`
**Fix**: Добавить `_rescore_cache = {}` с инвалидацией по `concept_error` timestamp.

### TN-51 (P2): Устранить `idx=-1` полностью

**Файл**: `train_full.py:458-465`

Заменить:
```python
epoch_train = _rescore_lines(epoch_train[idx + 1:], gen)
idx = -1; start_line = 0
```
на:
```python
epoch_train = epoch_train[:idx+1] + _rescore_lines(epoch_train[idx+1:], gen)
# idx не меняется — следующая итерация обработает rescored строку
```

Это устраняет TN-32 (curriculum reset), TN-41 (LR warmup — уже исправлен отдельно) и TN-42 (last_fluct_lines desync).

### TN-52 (P2): `momentum_mu` через `opt.p`

**Файлы**: `fcf_config.py` добавить `ParamDef`, `train_full.py:720` заменить на `opt.p['momentum_mu'].current`.

---

## 6. Вывод

**V11.2 (024f1aa) исправил 2/2 критических P0/P1 (TN-40 + SN-28/field_gate).**
**1768f27 исправил P0-регрессию (160s/batch).**

**Из 6 V11 TD-проблем:**
- 5 исправлены (TN-40, TN-41, TN-13 impl, TN-15, TN-31, TN-33)
- 3 не исправлены (TN-42, TN-43, TN-44, TN-45)
- 1 новая P1 (TN-46: TN-13 plateau doubling ineffective)

| V11 report | Status |
|:-----------|:------:|
| TN-40 (P0) noise_scale | ✅ |
| TN-41 (P2) LR warmup | ✅ |
| TN-42 (P3) last_fluct_lines | ❌ |
| TN-43 (P2) momentum_mu | ❌ |
| TN-44 (P2) idx=-1 | ❌ |
| TN-45 (P3) dead code | ❌ |

### Рекомендуемый порядок V12:
1. **TN-46** (P1) — исправить TN-13 plateau doubling (1 строка)
2. **TN-51** (P2) — убрать `idx=-1` (корень TN-32/42/44)
3. **TN-48** (P2) — `field_gate` в `opt.p`
4. **TN-52** (P2) — `momentum_mu` в `opt.p`
5. **TN-49** (P2) — `decay_warmup` + `_lr_offset`
6. **TN-47** (P2) — feedback в fluctuate
7. **TN-50** (P3) — кэш rescore
8. **TN-45** (P3) — мёртвый код
