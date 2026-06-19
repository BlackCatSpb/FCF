# FCF Training-Dynamics Report V11 — V10 Post-Commit Audit

**Дата**: 2026-06-19
**Агент**: Training-Dynamics Agent
**Версия**: V11 (аудит коммита 525688b — V10 TD fixes)

---

## 1. Проверка V10 TD Fixes (заявлены в V10 report, раздел 5)

| ID | Статус в V10 report | Статус в коде (525688b) | Детали |
|:---|:-------------------:|:----------------------:|--------|
| **TN-31** checkpoint_state чекпоинты | 🟡 НОВЫЙ P1 | ✅ **FIXED** | `_checkpoint()` атомарно сохраняет `.tmp` + `os.replace` |
| **TN-33** pipeline.global_step sync | 🟡 НОВЫЙ P2 | ✅ **FIXED** | `pipeline.global_step = global_step` + параметр `_checkpoint(..., global_step)` |
| **REG-V9-7** noise_scale split | 🔴 P1 | ⚠️ **ЧАСТИЧНО** | ParamDef split ✅, но **train_full.py:722 пропущен** (см. TN-40) |
| **REG-V9-8** momentum_mu из `opt.p[]` | ⚠️ P2 | ⚠️ **ЧАСТИЧНО** | Из `CFG.momentum_u` (не хардкод), но не из `opt.p['momentum_mu']` |
| **TN-34** opt.json naming | 🟡 НОВЫЙ P2 | ✅ **MITIGATED** | 4-уровневый fallback — рабочий |
| **TN-32** rescore idx=-1 curriculum | 🟡 НОВЫЙ P2 | ⚠️ **MITIGATED** | `_rescore_cp` сохраняет curriculum, но `idx=-1` остаётся |
| **TN-35** _rescore_lines рекомпьют | 🟢 НОВЫЙ P3 | ❌ **NOT FIXED** | Не реализован |
| **TN-13** Progressive BS plateaus | ❌ P2 | ❌ **NOT FIXED** | Не реализован |
| **TN-15** Decay warmup ramp | ❌ P2 | ❌ **NOT FIXED** | Не реализован |

---

## 2. TN-40 (P0): `noise_scale` KeyError — REGULAR FLUCTUATE COMPLETELY BROKEN

**Файл/строка**: `train_full.py:722`

**Суть**: V10 коммит (525688b) split `noise_scale` ParamDef на `gradient_noise_scale` и `fluctuation_amp` в `fcf_config.py`. Обновлены 2 из 3 call site в `train_full.py`, но **регулярный периодический fluctuate пропущен**:

```python
# train_full.py:722 — НЕ ИСПРАВЛЕНО (KeyError при первом FLUCTUATE_EVERY)
cs.fluctuate_fractal(noise_scale=opt.p['noise_scale'].current, ...)
```

**Что исправлено в V10:**
- Строка 703: `noise_scale=opt.p['noise_scale'].current` → `gradient_noise_scale=opt.p['gradient_noise_scale'].current` ✅
- Строка 770: `noise_scale=opt.p['noise_scale'].current` → `fluctuation_amp=opt.p['fluctuation_amp'].current` ✅

**Двойная поломка строки 722:**
1. `opt.p['noise_scale']` — KeyError: `noise_scale` больше не зарегистрирован в `ParameterOptimizer.p`
2. `noise_scale=` — TypeError: `fluctuate_fractal()` не принимает `noise_scale` kwargs (параметр называется `fluctuation_amp`)

**Последствия:**
- Первый же `idx % FLUCTUATE_EVERY == 0` (по умолчанию 2000) вызывает **unhandled crash**
- Редкий сценарий: если `FLUCTUATE_EVERY > CHECKPOINT_EVERY` и тренировка не доходит до 2000 строк, баг не проявляется
- Force-fluctuate (full stuck) работает ✅ — этот call site исправлен

**Fix:**
```python
# train_full.py:722
cs.fluctuate_fractal(fluctuation_amp=opt.p['fluctuation_amp'].current,
                     decay=opt.p['decay_rate'].current,
                     repel_strength=opt.p['repel_strength'].current,
                     generator=gen)
```

---

## 3. TN-41 (P2): `get_lr(idx)` сбрасывается на warmup после rescore

**Файл/строка**: `train_full.py:666`

**Суть**: `_rescore_cp` механизм (TN-32) корректно сохраняет `batch_size` и `max_len` после `idx=-1`, но `gen.train_lr = get_lr(idx)` (строка 666) использует **сырой idx**, не `_effective_cp`:

```python
# idx=0 после rescore
gen.train_lr = get_lr(0)  # → lr_warmup_lines: (0+1) / lr_warmup_lines → минимальный warmup
```

**Эффект**: После каждого rescore LR падает до уровня warmup (≈ `base_lr / 1000`), хотя `_rescore_cp` удерживает batch_size/max_len на уровне pre-rescore.

**Цепочка:**
1. Checkpoint на idx=5000: `_rescore_cp = _curriculum_p(5001) = 0.171`
2. `idx = -1` → `idx = 0` → `get_lr(0)` = `opt.p['full_lr'].current * 1/1000`
3. `_effective_cp(0)` = 0.171 (сохранён), batch_size=12, max_len=~170K
4. LR растёт только естественно через cosine annealing с restart

**Fix:**
```python
gen.train_lr = get_lr(idx) * _eff_cp + gen.train_lr * (1 - _eff_cp)
# Или: передавать idx через _eff_cp с пересчётом
```

---

## 4. TN-42 (P3): `last_fluct_lines` desync после rescore

**Файл/строка**: `train_full.py:726`

**Суть**: После `idx=-1` (rescore) → `idx=0`, `last_fluct_lines` остаётся на pre-rescore значении (например, 5000). Следующий `idx - last_fluct_lines` отрицателен, `if idx > 0` не срабатывает, но fluctuate сдвигается:

```
pre-rescore: last_fluct_lines = 5000
post-rescore: idx = 1 → idx - 5000 = -4999 < FLUCTUATE_EVERY → skip
             idx = 7000 → idx - 5000 = 2000 >= FLUCTUATE_EVERY → trigger
```

**Влияние**: fluctuate на больших интервалах не равномерен — после rescore следующий fluctuate задерживается. Некритично, но нарушает предположения `fluctuation_amp` адаптации.

**Fix:**
```python
last_fluct_lines = min(last_fluct_lines, idx)  # после rescore сбросить
```

---

## 5. TN-43 (P2): `momentum_mu` не адаптивен

**Файл/строка**: `train_full.py:704`, `fcf_config.py:425`

**Суть**: `momentum_mu` передаётся как `CFG.momentum_mu = 0.9` (статическая константа). V9 report требовал `opt.p['momentum_mu'].current`. V10 исправил только `CFG` вместо хардкода.

**Зачем адаптивный momentum_mu:**
- На ранних этапах: низкий momentum (0.5) — быстрая адаптация
- На поздних этапах: высокий momentum (0.95) — стабильность
- `ParameterOptimizer` уже имеет механизм rule-based adaptation

**Fix:**
```python
# fcf_config.py добавить:
ParamDef('momentum_mu',     0.5,    0.99,   0.9,   0.05, rules=[
    AdaptRule('cos_flat >= 5', 'momentum_mu', 'shift', 0.02),
    AdaptRule('cos_trend > 0.001 and mean_cos > 0.01', 'momentum_mu', 'shift', 0.01),
]),

# train_full.py:704
momentum_mu=opt.p['momentum_mu'].current,
```

---

## 6. Матрица статуса V11

| Область | Статус | Приоритет |
|:--------|:------:|:---------:|
| **TN-31** checkpoint_state при чекпоинтах | ✅ FIXED | — |
| **TN-33** pipeline.global_step sync | ✅ FIXED | — |
| **REG-V9-7** noise_scale split (fcf_config) | ✅ DONE | — |
| **REG-V9-7** noise_scale split (train_full:703) | ✅ FIXED | — |
| **REG-V9-7** noise_scale split (train_full:770) | ✅ FIXED | — |
| **TN-34** opt.json naming | ✅ MITIGATED | — |
| **REG-V9-8** momentum_mu из CFG | ✅ DONE | — |
| **TN-32** rescore idx=-1 (_rescore_cp) | ⚠️ MITIGATED | P2 |
| **TN-40** `noise_scale` KeyError (train_full:722) | 🔴 **НОВЫЙ** | **P0** |
| **TN-41** get_lr(idx) warmup after rescore | 🟡 **НОВЫЙ** | P2 |
| **TN-42** last_fluct_lines desync | 🟢 **НОВЫЙ** | P3 |
| **TN-43** momentum_mu не адаптивен | 🟢 **НОВЫЙ** | P2 |
| **TN-13** Progressive BS plateaus | ❌ Не реализован | P2 |
| **TN-15** Decay warmup ramp | ❌ Не реализован | P2 |
| **TN-35** _rescore_lines рекомпьют | ❌ Не реализован | P3 |

---

## 7. Предложения TN-40+

### TN-40 (P0): Исправить `noise_scale` → `fluctuation_amp` в train_full.py:722
```python
cs.fluctuate_fractal(fluctuation_amp=opt.p['fluctuation_amp'].current, ...)
```

### TN-41 (P2): `get_lr` не должен сбрасываться на warmup после rescore
Либо:
- Сохранять `train_lr` в `_rescore_cp` и использовать `pipeline._rescore_lr`
- Либо не сбрасывать `idx` до 0, а использовать `idx = start_line - 1`

### TN-42 (P3): Синхронизировать `last_fluct_lines` после rescore
```python
last_fluct_lines = min(last_fluct_lines, idx)
```

### TN-43 (P2): `momentum_mu` через `opt.p` (адаптивный)
Добавить `ParamDef('momentum_mu', ...)` в fcf_config.py и использовать `opt.p['momentum_mu'].current`.

### TN-44 (P2): Удалить `idx=-1` — заменить на `idx = idx` (no-op)
Rescore должен сортировать _remaining_ lines, не перезапуская цикл:
```python
epoch_train = epoch_train[:idx+1] + _rescore_lines(epoch_train[idx+1:], gen)
# idx не меняется
```
Текущий `_rescore_cp` workaround — костыль. Причина (`idx=-1`) не исправлена.

### TN-45 (P3): `pipeline.last_fluct_lines` — мёртвый код
`TrainingPipeline.__init__` устанавливает `self.last_fluct_lines = 0`, но это поле нигде не читается. Удалить.

---

## 8. Вывод

**V10 коммит (525688b) исправил 3 из 6 заявленных TD-проблем:**

| Заявлено | Исправлено |
|:---------|:----------:|
| TN-31 | ✅ |
| TN-33 | ✅ |
| REG-V9-7 noise_scale split | ⚠️ **2/3 call sites** |
| REG-V9-8 momentum_mu | ⚠️ Частично |
| TN-34 | ✅ Mitigated |
| TN-32 idx=-1 | ⚠️ Mitigated |

**Критический новый баг TN-40 (P0)**: `train_full.py:722` ссылается на `opt.p['noise_scale']`, которого больше нет. Регулярный периодический fluctuate упадёт с KeyError при первом же `FLUCTUATE_EVERY`.

**Вторичная проблема TN-41 (P2)**: После rescore (TN-32) `get_lr(0)` даёт warmup LR, сводя на нет `_rescore_cp` сохранение curriculum.

### Рекомендуемый порядок следующей итерации:
1. **TN-40** (P0) — исправить `noise_scale` → `fluctuation_amp` в train_full.py:722
2. **TN-41** (P2) — не сбрасывать LR на warmup после rescore
3. **TN-44** (P2) — заменить `idx=-1` на `epoch_train = epoch_train[:idx+1] + _rescore_lines(...)`
4. **TN-43** (P2) — momentum_mu в opt.p
5. **TN-45** (P3) — удалить мёртвый `self.last_fluct_lines`
6. **TN-42** (P3) — last_fluct_lines sync после rescore
7. **TN-13** (P2) — Progressive BS plateaus
8. **TN-15** (P2) — Decay warmup ramp
