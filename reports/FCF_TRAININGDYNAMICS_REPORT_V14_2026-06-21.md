# FCF Training-Dynamics Report V14 — V13 Post-Commit Audit

**Дата**: 2026-06-21
**Агент**: Training-Dynamics Agent
**Версия**: V14 (аудит V13 фиксов: TN-47, TN-32/44, TN-48, syntax_lattice)

---

## 1. Проверка V13 фиксов

| ID | Статус в V12 | Статус в HEAD | Детали |
|:---|:------------:|:-------------:|--------|
| **TN-47** current_cos→fluctuate_fractal | 🟡 NEW | ✅ **FIXED** | `concept_space.py:462-479`: `current_cos` модулирует amp (high cos>0.25 → reduce, low cos<0.05 → reduce). Вызовы `train_full.py:749,799` передают `last_cos_sim[0]`. |
| **TN-32/44** idx=-1→idx=0 при rescore | ❌ Не исправлен | ✅ **FIXED** | `train_full.py:465`: `idx = 0; start_line = 0` (было `idx = -1`). `_rescore_cp` сохраняет curriculum (стр. 462). |
| **TN-48** field_gate bool→float threshold | 🟡 NEW | ✅ **FIXED** | `fcf_config.py:362`: `ParamDef('field_gate_threshold', 0.0, 1.0, 1.0, 0.05)`. `train_full.py:719`: `field_gate=opt.p['field_gate_threshold'].current`. Все gate-проверки используют `> 0.5`. |
| **syntax_lattice** инкрементальный prefix_total | — | ✅ **FIXED** | `syntax_lattice.py:222`: `self._prefix_total[prefix] = self._prefix_total.get(prefix, 0) + 1`. Без полного перестроения на каждом `update()`. |
| **TN-46** plateau doubling multiplier | 🔴 НОВЫЙ | ✅ **FIXED** | `_batch_mult` (стр. 662) не перезаписывается `bs_curve()`. Удвоение при stuck≥3 (стр. 802-804) сохраняется между итерациями. |

---

## 2. Проблемы, перенесённые из V12 (не исправлены)

### TN-42 (P3) — `last_fluct_lines` desync после rescore — ❌ NOT FIXED

**Файл/строка**: `train_full.py:465-466, 529, 698, 744-750`

**Суть**: После rescore `pipeline._checkpoint()` устанавливает `self.last_fluct_lines = 0` (стр. 466). Но это **мёртвое поле** (`pipeline.last_fluct_lines`). Главный цикл использует глобальную `last_fluct_lines` (стр. 529), которая **не сбрасывается**. После rescore `idx = 0`, `last_fluct_lines = 6000` → `0 - 6000 < FLUCTUATE_EVERY` → fluctuate не срабатывает, пока `idx` не достигнет `6000 + FLUCTUATE_EVERY`.

**Влияние**: При 4000+ оставшихся строках после rescore fluctuate может никогда не вызваться до конца эпохи.

**Fix**: Обновлять глобальную `last_fluct_lines` в `_checkpoint()` или после rescore:
```python
# train_full.py:465-466
epoch_train = _rescore_lines(epoch_train[idx + 1:], gen)
idx = 0; start_line = 0
last_fluct_lines = min(last_fluct_lines, idx)  # или просто idx
```

### TN-43 (P2) — `momentum_mu` не адаптивен — ❌ NOT FIXED

**Файл/строка**: `train_full.py:720`, `fcf_config.py:430`

`momentum_mu=CFG.momentum_mu` (статическая константа 0.9 из `fcf_config.py:430`). Не используется `opt.p['momentum_mu']`.

### TN-45 (P3) — `pipeline.last_fluct_lines` мёртвый код — ❌ NOT FIXED

**Файл/строка**: `train_full.py:367, 466`

`TrainingPipeline.__init__` (стр. 367) и `_checkpoint()` (стр. 466) устанавливают `self.last_fluct_lines`. Это поле **никогда не читается** — главный цикл использует одноимённую глобальную переменную (стр. 529).

### TN-34 (P2) — opt.json naming mismatch — ⚠️ MITIGATED, НЕ ИСПРАВЛЕН

**Файл/строка**: `train_full.py:286-303`, `checkpoint_manager.py:75`, `train_full.py:609-613`

4-уровневый fallback в restore работает, но root cause не исправлен:
- `CheckpointManager._sync_save` → `concept_space_{tag}.opt.json`
- `_final_save` → `concept_space.opt.json` (без тега)
- При загрузке после `_final_save` + перезапуска без `--resume` может не найти `.opt.json` с нужным именем

### TN-49 (P2) — `decay_warmup` не учитывает rescore — ❌ NOT FIXED

**Файл/строка**: `train_full.py:740`

```python
_decay_pct = min(idx / max(CFG.decay_warmup_lines, 1), 1.0)
```

После rescore `idx = 0`, рампа restart с 0.998. `_lr_offset` существует (стр. 681) для LR warmup, но не используется для decay:
```python
# Должно быть:
_decay_pct = min((idx + _lr_offset) / max(CFG.decay_warmup_lines, 1), 1.0)
```

### TN-50 (P3) — `_rescore_lines` без кэша — ❌ NOT FIXED

**Файл/строка**: `train_full.py:340-351`

Каждый rescore пересчитывает score для всех оставшихся строк.

---

## 3. Новые проблемы V14

### TN-53 (P3) — `field_gate: bool` мёртвый конфиг

**Файл/строка**: `fcf_config.py:429`

```python
field_gate: bool = True
```

Это поле больше нигде не используется — `train_full.py:719` использует `opt.p['field_gate_threshold'].current`. Поле можно удалить.

### TN-54 (P3) — `_final_save` не использует `CheckpointManager`

**Файл/строка**: `train_full.py:599-627`

`_final_save` (вызывается при завершении/прерывании) дублирует логику `CheckpointManager`:
- Не использует thread pool → блокирует GIL на время записи
- Сохраняет opt.json с другим именем (`concept_space.opt.json` вместо `concept_space_{tag}.opt.json`)
- Не удаляет `.tmp` файлы при ошибках

**Fix**: Вызвать `pipeline.ckpt_mgr.save('final', ...)` и `pipeline.ckpt_mgr.wait()`.

### TN-55 (P2) — FULL_STUCK force-fluctuate может использовать stale `last_cos_sim`

**Файл/строка**: `train_full.py:793-800`

`last_cos_sim` обновляется только каждые `COS_REFRESH=5s` (стр. 769-771). При force-fluctuate (стр. 793-799) может использоваться значение возрастом до 5 секунд. Для FULL_STUCK (застревание на 5+ чекпоинтах = 25K+ строк) это приемлемо, но может быть неточным.

### TN-56 (P3) — `_rescore_cp` не сбрасывается при новом epoch

**Файл/строка**: `train_full.py:650-651`

При старте новой эпохи `pipeline._rescore_line = None`, но `pipeline._rescore_cp` не сбрасывается. Это может вызвать `_effective_cp` возврат старого cp из предыдущей эпохи.

---

## 4. Матрица статуса V14

| ID | Статус | Приоритет | Тип |
|:---|:------:|:---------:|:---:|
| **TN-46** plateau _batch_mult | ✅ FIXED | — | V13 fix |
| **TN-47** current_cos feedback | ✅ FIXED | — | V13 fix |
| **TN-32/44** idx=-1→idx=0 | ✅ FIXED | — | V13 fix |
| **TN-48** field_gate float threshold | ✅ FIXED | — | V13 fix |
| **syntax_lattice** prefix_total incremental | ✅ FIXED | — | V13 fix |
| **TN-42** last_fluct_lines desync | ❌ Not fixed | P3 | V11 carryover |
| **TN-43** momentum_mu static | ❌ Not fixed | P2 | V11 carryover |
| **TN-45** pipeline.last_fluct_lines dead | ❌ Not fixed | P3 | V11 carryover |
| **TN-34** opt.json naming | ⚠️ Mitigated | P2 | V11 carryover |
| **TN-49** decay_warmup rescore | ❌ Not fixed | P2 | V12 |
| **TN-50** _rescore_lines cache | ❌ Not fixed | P3 | V12 |
| **TN-53** field_gate bool dead config | 🟡 НОВЫЙ | P3 | V14 |
| **TN-54** _final_save not async | 🟡 НОВЫЙ | P3 | V14 |
| **TN-55** FULL_STUCK stale cos | 🟡 НОВЫЙ | P3 | V14 |
| **TN-56** _rescore_cp epoch reset | 🟡 НОВЫЙ | P3 | V14 |

---

## 5. Предложения TN-55+

### TN-54 (P3): Использовать `CheckpointManager` в `_final_save`
**Файл**: `train_full.py:599-627`
**Fix**: Заменить прямой save на `pipeline.ckpt_mgr.save('final', ...)` + `pipeline.ckpt_mgr.wait()`.

### TN-55 (P3): Force-fluctuate с актуальным cos
**Файл**: `train_full.py:793-799`
**Fix**: Перед force-fluctuate выполнить `last_cos_sim = mean_cosine_sim(cs)` для свежего значения.

### TN-56 (P3): Сброс `_rescore_cp` при новой эпохе
**Файл**: `train_full.py:647-653`
**Fix**: `pipeline._rescore_cp = None` при старте эпохи.

### TN-42 (P3): `last_fluct_lines` sync после rescore
**Файл**: `train_full.py:465-466`
**Fix**: Установить `last_fluct_lines = idx` после `idx = 0` (синхронизировать с новым idx).

### TN-49 (P2): `decay_warmup` с `_lr_offset`
**Файл**: `train_full.py:740`
**Fix**: `_decay_pct = min((idx + _lr_offset) / max(CFG.decay_warmup_lines, 1), 1.0)`

### TN-43 (P2): `momentum_mu` через `opt.p`
**Файлы**: `fcf_config.py`, `train_full.py:720`
**Fix**: Добавить `ParamDef('momentum_mu', ...)` и использовать `opt.p['momentum_mu'].current`.

### TN-53 (P3): Удалить мёртвый `field_gate: bool`
**Файл**: `fcf_config.py:429`

---

## 6. Вывод

**V13 исправил 5/5 заявленных проблем. 139 тестов проходят, 0 падают.**

**Остаются неисправленными**: 2×P2 (TN-43 momentum_mu, TN-49 decay_warmup), 1×P2 mitigated (TN-34 opt.json), 5×P3.

**Рекомендуемый порядок V14:**
1. **TN-42** (P3) — 1 строка, `last_fluct_lines = idx` после rescore
2. **TN-49** (P2) — 1 строка, `(idx + _lr_offset)`
3. **TN-53** (P3) — удалить мёртвый `field_gate: bool`
4. **TN-43** (P2) — `momentum_mu` в `opt.p`
5. **TN-34** (P2) — унифицировать naming opt.json
6. **TN-50** (P3) — кэш rescore
7. **TN-54/55/56** (P3) — гигиена кода
