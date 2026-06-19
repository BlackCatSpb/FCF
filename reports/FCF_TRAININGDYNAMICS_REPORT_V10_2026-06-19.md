# FCF Training-Dynamics Report V10 — V9 Post-Commit Audit

**Дата**: 2026-06-19
**Агент**: Training-Dynamics Agent
**Версия**: V10 (аудит коммита a0fe15b — V9 TD fixes)

---

## 1. Проверка V9 TD Fixes (заявлены в V9 report, раздел 10)

| ID | Заявлен | Статус в коде | Детали |
|:---|:-------:|:-------------:|--------|
| **TN-25** rescore fix | ✅ FIXED | ⚠️ **Частично** | `epoch_train[idx+1:]` ✅, `start_line=0` ✅, но `idx = -1` **остался** |
| **TN-26** cleanup_old_checkpoints удалён | ✅ FIXED | ✅ | Функция отсутствует. Только CheckpointManager |
| **TN-27** global_step при resume | ✅ FIXED | ✅ | Сохраняется в checkpoint_state.json, восстанавливается |
| **TN-29** мёртвый код удалён | ✅ FIXED | ✅ | DECAY_EVERY, save_checkpoint_state, run_epoch — удалены |

### TN-25: глубокая проверка (train_full.py:430-434)

```python
# Текущий код:
if epoch_train is not None and start_line is not None:
    remaining = idx - start_line + 1
    if remaining > 0 and remaining < len(epoch_train):
        epoch_train = _rescore_lines(epoch_train[idx + 1:], gen)
        idx = -1; start_line = 0
```

**Что исправлено (V9):**
- `epoch_train[idx + 1:]` вместо `epoch_train[remaining:]` — корректно: срез после idx
- `start_line = 0` вместо `start_line = -1` — корректно

**Что НЕ исправлено:**
- `idx = -1` **остался** (V9 report в разделе 10 утверждал "idx = idx - remaining", но в код попал только `-1`)

**Цепочка последствий `idx = -1`:**
1. `idx += 1` (строка 747) → `idx = 0`
2. `_curriculum_p(0) = 0` → `batch_size = bs_curve(0) = 8`, `max_len = 16`, `get_lr(0)` = warmup
3. Если `last_fluct_lines > 0` → `idx - last_fluct_lines` отрицательно → force-fluctuate баг
4. Все ранее обработанные строки переобучаются заново (waste)

**Корректный фикс:**
```python
epoch_train = epoch_train[:idx+1] + _rescore_lines(epoch_train[idx+1:], gen)
# idx не меняется — следующий while продолжит с idx+1 (новая rescored строка)
```

---

## 2. Проверка V9 REG-проблем (из Architect V9 report)

| ID | Описание | Приоритет | Статус |
|:---|----------|:---------:|:------:|
| **REG-V9-7** | `noise_scale` — 2 механики на 1 параметр | P1 | **НЕ ИСПРАВЛЕН** |
| **REG-V9-8** | `momentum_mu=0.9` хардкод | P2 | **ЧАСТИЧНО** |

### REG-V9-7: noise_scale dual mechanics (train_full.py:674, 693, 741)

Один параметр `opt.p['noise_scale'].current` управляет двумя разными механиками:

1. **Gradient noise injection** (stdp_trainer.py:389-390) — torch.randn шум на аккумулятор градиента
2. **Fractal fluctuation** (train_full.py:693) — `cs.fluctuate_fractal(noise_scale=...)` — perturbation code-векторов

Разделение не произошло. `noise_scale` по умолчанию = 0.001 — для gradient noise OK, для fractal fluctuation слишком мало. При адаптации через `opt.p['noise_scale']` обе механики меняются синхронно, что некорректно.

### REG-V9-8: momentum_mu (train_full.py:675)

```python
momentum_mu=CFG.momentum_mu  # fcf_config.py:421: 0.9
```

Теперь читается из конфига (было хардкод 0.9) — **шаг вперёд**.
Но V9 report требовал `opt.p['momentum_mu'].current` — т.е. адаптивный параметр через ParameterOptimizer. Сейчас это статическая константа.

---

## 3. Новые проблемы V10

### TN-31 (P1): checkpoint_state.json не обновляется при чекпоинтах

**Файл**: `train_full.py`
**Суть**: `TrainingPipeline._checkpoint()` (строка 381) сохраняет cs/lattice/opt через `ckpt_mgr.save()`, но **НЕ пишет** `checkpoint_state.json`. Этот файл обновляется только в `_final_save()` — при штатном завершении или Ctrl+C.

**Сценарий сбоя:**
1. Чекпоинт сохранён на 5000 строк (`concept_space_5k.json`)
2. `checkpoint_state.json` указывает на 3000 (последний `_final_save`)
3. Сбой → resume грузит чекпоинт 3000, теряя 2000 строк прогресса

**Fix**: В `_checkpoint()` добавить запись `checkpoint_state.json`:
```python
with open(CFG.ckpt_state_path + '.tmp', 'w') as f:
    json.dump({'line': idx, 'epoch': epoch, 'global_step': self.global_step, 'timestamp': time.time()}, f)
os.replace(CFG.ckpt_state_path + '.tmp', CFG.ckpt_state_path)
```

### TN-32 (P2): Rescore idx=-1 сбрасывает curriculum

**Файл**: `train_full.py:434`
**Суть**: `idx = -1` → после `idx += 1` → `idx = 0`. Это сбрасывает:
- `_curriculum_p(0) = 0` → batch_size=8, max_len=16
- `get_lr(0)` → warmup режим
- Все обработанные строки до сброса переобучаются повторно

**Вес**: На больших корпусах rescore теряет смысл — вместо сортировки оставшихся строк по error, он переобучает всё с нуля.

**Fix**: `idx = idx` (не менять) или `idx = start_line - 1` (перемотка к началу rescored суффикса).

### TN-33 (P2): TrainingPipeline.global_step не синхронизирован

**Файл**: `train_full.py`
**Суть**: Две независимые переменные `global_step`:
- `pipeline.global_step` (строка 347) — инициализирован 0, никогда не обновляется
- `global_step` (строка 488) — работает в main loop

`_checkpoint()` использует `self.global_step`, который всегда 0. При сохранении чекпоинта через `ckpt_mgr.save()` значение global_step теряется.

**Fix**: Удалить `pipeline.global_step`, использовать единую переменную из main loop. Или синхронизировать: `pipeline.global_step = global_step` перед `_checkpoint()`.

### TN-34 (P2): opt.json naming mismatch между CheckpointManager и _final_save

**Файл**: `train_full.py:568`, `checkpoint_manager.py:86`
**Суть**:
- `CheckpointManager._sync_save()` → `concept_space_{tag}.opt.json` (тег: "5k")
- `_final_save()` → `concept_space.opt.json` (без тега)

Resume ищет tagged, потом tagless — работает. Но:
- `_final_save` перезаписывает `concept_space.opt.json` при каждом сохранении
- CheckpointManager.cleanup() не удаляет tagless opt.json
- После 10 чекпоинтов в директории: 10× tagged opt.json + 1 tagless

**Fix**: `_final_save` должен писать `concept_space_final.opt.json` или CheckpointManager должен чистить и tagless.

### TN-35 (P2): _rescore_lines — полный пересчёт на каждой итерации

**Файл**: `train_full.py:319-330`
**Суть**: `_rescore_lines()` вычисляет score для ВСЕХ оставшихся строк на каждом чекпоинте. На 146K корпусе:
- Каждый rescore: до 145K encode + error lookup
- `_encode_input` вызывает SentencePiece → CPU-bound
- На медленных CPU может занимать секунды

**Бенчмарк**: ~50K tokens/sec через SentencePiece → 145K строк × ~50 BPE = ~7M токенов → ~140ms. Некритично, но на более частых чекпоинтах растёт.

**Fix**: Кэшировать scores между rescore вызовами. Или rescore каждые N-й чекпоинт.

---

## 4. Дополнительные наблюдения

### 4.1. Двойная очистка чекпоинтов — НЕ ОБНАРУЖЕНА

`cleanup_old_checkpoints()` удалён ✅. Единственная система очистки — `CheckpointManager.cleanup()` (train_full.py:382). Дублирования нет.

### 4.2. global_step при resume — КОРРЕКТНО

- `load_checkpoint_state()` (строка 77-87) возвращает `global_step`
- `global_step = resume_global_step` (строка 488) — корректно
- `global_step += 1` в цикле (строка 625) — корректно
- `_final_save` записывает `global_step` (строка 573)

Проблема только в том, что `_checkpoint` не обновляет `checkpoint_state.json` (TN-31).

### 4.3. TN-13 (Progressive batch size with plateaus) — НЕ РЕАЛИЗОВАН

Только `bs_curve = lambda i: int(CFG.batch_size_start + ... * _curriculum_p(i))` (строка 616). Линейная рампа. Нет плато-детектора.

### 4.4. TN-15 (Decay warmup with protect threshold ramp) — НЕ РЕАЛИЗОВАН

`lattice.decay_all(rare_concept_protect=True, rare_threshold=3)` (строка 700). `rare_threshold=3` — константа. Нет warmup.

### 4.5. DECAY_EVERY — всё ещё мёртвый код? 

Проверка: `DECAY_EVERY` в train_full.py отсутствует ✅ (удалён в V9). Используется `CFG.decay_every_pairs`.

### 4.6. start_line обновление при чекпоинте

После `_checkpoint()` главный цикл захватывает возврат:
```python
result = pipeline._checkpoint(..., epoch_train, start_line)
if result is not None:
    idx, start_line, epoch_train = result
```
`start_line` обновляется корректно через возврат.

---

## 5. Матрица статуса V10

| Область | Статус | Приоритет |
|:--------|:------:|:---------:|
| **TN-25** rescore idx+1 fix | ⚠️ `idx = -1` остался | **P1** |
| **TN-26** cleanup_old_checkpoints | ✅ Удалён | — |
| **TN-27** global_step resume | ✅ Работает | — |
| **TN-29** мёртвый код | ✅ Удалён | — |
| **REG-V9-7** noise_scale dual | ❌ Не разделён | **P1** |
| **REG-V9-8** momentum_mu из `opt.p[]` | ⚠️ Из CFG, не из `opt.p` | P2 |
| **TN-13** Progressive BS plateaus | ❌ Не реализован | P2 |
| **TN-15** Decay warmup ramp | ❌ Не реализован | P2 |
| **TN-31** checkpoint_state не обновляется | 🔴 **НОВЫЙ** | **P1** |
| **TN-32** rescore idx=-1 curriculum сброс | 🟡 **НОВЫЙ** | P2 |
| **TN-33** pipeline.global_step мёртв | 🟡 **НОВЫЙ** | P2 |
| **TN-34** opt.json naming mismatch | 🟡 **НОВЫЙ** | P2 |
| **TN-35** _rescore_lines полный пересчёт | 🟢 **НОВЫЙ** | P3 |
| **Двойная очистка чекпоинтов** | ✅ Не обнаружена | — |

---

## 6. Предложения TN-30+

### TN-30 (P3): EMA Batched Update (GPU)
Из V9. Заменить for-loop на тензорную операцию.

### TN-31 (P1): Сохранять checkpoint_state при каждом чекпоинте
В `_checkpoint()` добавить запись `checkpoint_state.json` атомарно через `.tmp` + `os.replace`. Использовать `idx`, `epoch`, `global_step`.

### TN-32 (P2): Исправить idx=-1 в rescore
Заменить:
```python
idx = -1; start_line = 0
```
на:
```python
# idx не меняем — rescored строки начнут обрабатываться на следующей итерации
```
Или, если нужен ресет к началу rescored суффикса:
```python
idx = start_line - 1  # перемотка к началу
```

### TN-33 (P2): Синхронизировать pipeline.global_step
Удалить `self.global_step` из `TrainingPipeline.__init__` (строка 347). Передавать `global_step` из main loop в `_checkpoint()`. Или синхронизировать перед вызовом.

### TN-34 (P2): Унифицировать opt.json naming
`_final_save()` должен писать tagged opt.json, соответствующий последнему чекпоинту. Или CheckpointManager.cleanup() должен удалять tagless `concept_space.opt.json` (но он может быть нужен для fallback resume).

### TN-35 (P3): Оптимизировать _rescore_lines
Добавить кэш scores (dict) с инвалидацией при изменении concept_error. Пересчитывать только для строк, где error изменился > threshold.

### TN-36 (P2): Разделить noise_scale (REG-V9-7)
Добавить второй параметр `fluctuation_amp` в FCFConfig.params. Использовать `opt.p['fluctuation_amp'].current` в `fluctuate_fractal()`. `noise_scale` оставить только для gradient noise.

### TN-37 (P2): momentum_mu в opt.p (REG-V9-8)
Добавить `ParamDef('momentum_mu', 0.5, 0.99, 0.9, 0.05)` в fcf_config.py. Передавать `opt.p['momentum_mu'].current` в train_batch.

### TN-38 (P3): Eval independent of checkpoint
```python
# TN-28 из V9
if idx >= self._eval_next_idx:
    self._eval_next_idx += self.cfg.eval_every_fast
```
Отвязать eval от `idx % CHECKPOINT_EVERY == 0`.

### TN-39 (P2): Rescore не сохраняет обработанные строки
```python
epoch_train = epoch_train[:idx+1] + _rescore_lines(epoch_train[idx+1:], gen)
# idx не меняется — сохранение прогресса
```

---

## 7. Вывод

**V9 коммит (a0fe15b) исправил 3 из 4 заявленных TD-проблем**, но **TN-25 не полностью** — `idx = -1` остаётся в коде, вызывая сброс curriculum при rescore.

**Новый P1 баг**: `checkpoint_state.json` не обновляется при чекпоинтах (TN-31) — при сбое теряется прогресс между `_final_save` вызовами.

**REGV9-7** (noise_scale dual) и **REGV9-8** (momentum_mu) также не исправлены.

### Рекомендуемый порядок следующей итерации:
1. **TN-31** (P1) — сохранять checkpoint_state при чекпоинтах
2. **TN-36** (P1) — разделить noise_scale/fluctuation_amp
3. **TN-32** (P2) — исправить idx=-1 в rescore
4. **TN-33** (P2) — synch pipeline.global_step
5. **TN-37** (P2) — momentum_mu в opt.p
6. **TN-39** (P2) — rescore сохраняет processed lines
7. **TN-38** (P3) — независимый eval от checkpoint
