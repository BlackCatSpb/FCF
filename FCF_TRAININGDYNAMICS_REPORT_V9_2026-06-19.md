# FCF Training-Dynamics Report V9 — V8 Post-Commit Audit

**Дата**: 2026-06-19  
**Агент**: Training-Dynamics Agent  
**Версия**: V9 (аудит коммита 7ae6d9a — V8 P0/P1 fixes)

---

## 1. Проверка V8 Commit Fixes (заявлены в V8 report, раздел 9)

| Проблема | Статус в V8 report | Статус в коде | Детали |
|:---------|:------------------:|:-------------:|--------|
| **T-B1 rescope return** | ✅ Исправлен | ✅ | Main loop захватывает return `_checkpoint` (train_full.py:769-771) |
| **T-B2 run_epoch удалён** | ✅ Исправлен | ✅ | `run_epoch()` отсутствует в файле |
| **TN-14 batch-stale** | ✅ Исправлен | ✅ | `v_local = g_vecs[i].clone()`, единственный `_apply_vector_update` (stdp_trainer.py:714-754) |
| **noise_scale** | ✅ Исправлен | ✅ | `noise_scale=opt.p['noise_scale'].current` (train_full.py:708) |
| **momentum_mu** | ✅ Исправлен | ✅ | `momentum_mu=0.9` (train_full.py:709) |
| **TN-16 (rescore idx fix)** | ✅ **Исправлен** | ❌ **НЕ ИСПРАВЛЕН** | **СМ. РАЗДЕЛ 2** |
| **TN-17 (dual loop)** | ✅ Исправлен | ✅ | run_epoch удалён |
| **TN-18 (noise/momentum)** | ✅ Исправлен | ✅ | Параметры передаются |
| **TN-19 (TN-14 fix)** | ✅ Исправлен | ✅ | Локальная копия `v_local` |

---

## 2. ❌ TN-16 (T-B1): idx = -1 bug — НЕ ИСПРАВЛЕН

V8 report (раздел 9) утверждает, что TN-16 реализован. **Фактический код опровергает это.**

### Текущий код (train_full.py:461-465):
```python
if epoch_train is not None and start_line is not None:
    remaining = idx - start_line + 1
    if remaining > 0 and remaining < len(epoch_train):
        epoch_train = _rescore_lines(epoch_train[remaining:], gen)
        idx = -1; start_line = -1   # ← СТАРЫЙ БАГ!
```

### Ожидаемое поведение (согласно TN-16):
```python
idx = idx - remaining   # перемотка к началу rescored суффикса
start_line = idx        # или start_line = idx - remaining
```

### Последствия `idx = -1`:
1. После `idx += 1` (строка 781) → `idx = 0`
2. Сброс curriculum: `_curriculum_p(0) = 0` → `batch_size = bs_curve(0) = batch_size_start`, `max_len = CURICULUM_MIN_LEN = 16`
3. `last_fluct_lines` ещё не сброшен → `(idx=0) - last_fluct_lines` отрицательное → **force-fluctuate на каждой итерации**
4. `get_lr(0)` = warmup с начала
5. **Rescore фактически делает больше вреда, чем пользы**

### V8 report содержал корректный код в разделе 7 (TN-16), но в код он не попал.

---

## 3. Единый тренировочный цикл — регрессии

### 3.1. run_epoch удалён — OK
`run_epoch()` отсутствует. Main loop (строки 634-781) — единственный.

### 3.2. _checkpoint return захвачен — OK
```python
result = pipeline._checkpoint(epoch, idx, elapsed, epoch_lines, destab_scale, t_start, epoch_train, start_line)
if result is not None:
    idx, start_line, epoch_train = result
```

### 3.3. DECAY_EVERY — мёртвый код
```python
DECAY_EVERY = CFG.decay_every_fast if FAST else CFG.decay_every_slow  # строка 509
```
`DECAY_EVERY` нигде не используется. Main loop проверяет `total_pairs_since_last_decay >= CFG.decay_every_pairs` (строка 733). Переменная-призрак.

### 3.4. save_checkpoint_state — мёртвый код
Функция определена (строка 77), но не вызывается. Checkpoint state пишется только в `_final_save` (строка 606-609).

---

## 4. Checkpoint naming — resume работает?

### 4.1. Единый naming — OK
CheckpointManager создаёт: `concept_space_{tag}.json`, `syntax_lattice_{tag}.json`, `concept_space_{tag}.opt.json`
Train_full resume конструирует: `concept_space_{resume_tag}.json` — ✅ совпадает.

### 4.2. Двойная очистка чекпоинтов — ❌
Две независимые системы очистки:

| Система | Локация | Критерий |
|---------|---------|----------|
| `CheckpointManager.cleanup()` | train_full.py:413 | По порядку сохранения (`_saved_tags`) |
| `cleanup_old_checkpoints()` | train_full.py:618 | По mtime (glob `concept_space_*k.json`) |

**Проблемы**:
- Разные критерии "keep recent" → могут удалять разные файлы
- `cleanup_old_checkpoints` не удаляет `.lattice.npz`, `.meta.json` (но CheckpointManager.cleanup удаляет все)
- `cleanup_old_checkpoints` удаляет только по mtime, не синхронизирована с CheckpointManager

**Рекомендация**: Удалить `cleanup_old_checkpoints()` и оставить только `CheckpointManager.cleanup()`.

### 4.3. Mismatch: opt.json suffix
CheckpointManager сохраняет: `concept_space_{tag}.opt.json`
Resume ищет: `concept_space_{tag}.opt.json` — ✅ ок

### 4.4. Первый чекпоинт на 0k
`ckpt_k = idx // 1000`. При idx=500 → ckpt_k=0 → `concept_space_0k.json`. Ok, но неинформативно.

---

## 5. T-B1 фикс — idx обработка (детальный анализ)

### 5.1. Код в _checkpoint (строка 460-465):
```python
# T-B1: Self-paced learning rescore (only in run_epoch path)
if epoch_train is not None and start_line is not None:
    remaining = idx - start_line + 1
    if remaining > 0 and remaining < len(epoch_train):
        epoch_train = _rescore_lines(epoch_train[remaining:], gen)
        idx = -1; start_line = -1
```
Комментарий `"(only in run_epoch path)"` — теперь дезинформация. `run_epoch` удалён, этот код **выполняется** в main loop.

### 5.2. Коррекция (TN-16):
```python
# T-B1: Self-paced learning rescore
if epoch_train is not None and start_line is not None:
    remaining = idx - start_line + 1
    if remaining > 0 and remaining < len(epoch_train):
        epoch_train = _rescore_lines(epoch_train[remaining:], gen)
        idx = idx - remaining      # перемотка к началу rescored суффикса
        start_line = idx
```

### 5.3. start_line не обновляется в main loop
`start_line` устанавливается один раз (0 или resume_line) и никогда не переустанавливается.
```python
start_line = idx  # нужно перед _checkpoint
```

---

## 6. noise_scale и momentum_mu — полный path

### 6.1. Передача из train_full.py в CrystalGenerator — ✅
```python
n_pairs = gen.train_batch(
    ...
    noise_scale=opt.p['noise_scale'].current,
    momentum_mu=0.9)  # строки 708-709
```

### 6.2. CrystalGenerator → STDPTrainer — ✅
```python
def train_batch(self, texts, ..., momentum_mu=0.9, noise_scale=0.0):
    return self._trainer.train_batch(..., momentum_mu=momentum_mu, noise_scale=noise_scale)
```

### 6.3. STDPTrainer → _gpu_stdp_apply — ✅
```python
unique_gen = self._gpu_stdp_apply(..., noise_scale=noise_scale, momentum_mu=momentum_mu)
```

### 6.4. Использование в _gpu_stdp_apply:
- `noise_scale`: строка 385 — ✅
- `momentum_mu`: строки 404-418 — ✅

**Вердикт**: полный path корректен.

---

## 7. TN-14: Field-Aware Contrastive — batch-stale fix

### 7.1. Текущая реализация (stdp_trainer.py:714-754):
```python
v_local = g_vecs[i].clone()           # копия (строка 714)
# TN-14 cross-field repel:
v_local = v_local + rep_grad * ...    # модификация локальной копии (строка 736)
# Main contrastive push:
grad = (cos_v[:, None] * v_neg).mean(dim=0) - v_local   # использует v_local (строка 740)
v_new = v_local + push                # (строка 745)
cs._apply_vector_update(gen_idxs[i], v_new.cpu().numpy())  # единственный apply (строка 749)
```

### 7.2. Результат:
- ✅ Batch-stale bug **исправлен** — нет перезаписи TN-14 main push
- ✅ Единственный `_apply_vector_update` за итерацию

### 7.3. Остаточная проблема (P2):
Цикл `for j in range(min(50, topk_idx.shape[1]))` на каждой итерации может модифицировать `v_local`. Все cross-field repel накапливаются корректно, но GPU sync на каждую проверку `torch.bitwise_and(fb_gn, fb_rn)` может быть дорогим.

---

## 8. Дополнительные проблемы

### 8.1. global_step не сохраняется при resume (P1)
`global_step` (строка 521) не восстанавливается из `ParameterOptimizer.state`. При resume:
- `destab_pct` (строка 673) рассчитывается с `global_step = 0` → destab_scale занижен
- `_curriculum_p(idx)` использует `TOTAL_LINES_GLOBAL` — корректно, т.к. зависит от idx

### 8.2. TN-13: Progressive Batch Size — не реализован (P2)
Только `bs_curve` (линейная). Плато-детектор отсутствует.

### 8.3. TN-15: Decay Warmup — не реализован (P2)
`rare_threshold=3` — константа (строка 734).

### 8.4. EMA CPU for-loop GPU sync (stdp_trainer.py:460) — не исправлен (P2)
```python
for ...:
    gen._ema_vecs_t[gen_cid] = ...  # GPU sync на каждую итерацию
```

### 8.5. _graph_cache leak (P2)
`crystal_generator.py:71`: `_graph_cache` — dict без эвикции.

### 8.6. HormonalSystem.reset() отсутствует (P2)
Состояние ACh/DA переносится между generate() вызовами.

### 8.7. Eval зависит от checkpoint interval (P2)
`eval` вызывается только при `idx % CHECKPOINT_EVERY == 0`. Если изменить `checkpoint_every`, eval станет реже/чаще.

---

## 9. Матрица статуса V9

| Область | Статус | Приоритет |
|---------|:------:|:---------:|
| **T-B1 (idx=-1 bug)** | ✅ **Исправлен** (V9 a0fe15b) | **P0** |
| **T-B2 (run_epoch)** | ✅ Исправлен | — |
| **T-B3 (EMA before apply)** | ✅ Исправлен | — |
| **TN-6 (noise_scale)** | ✅ Исправлен | — |
| **TN-11 (momentum_mu)** | ✅ Исправлен | — |
| **TN-12 (Switched Eval)** | ✅ Исправлен | — |
| **TN-13 (Progressive BS)** | ❌ Не реализован | P2 |
| **TN-14 (Contrastive)** | ⚠️ batch-stale fix ✅, но for-loop GPU sync | P2 |
| **TN-15 (Decay Warmup)** | ❌ Не реализован | P2 |
| **Двойная очистка чекпоинтов** | ✅ **Исправлен** (cleanup_old_checkpoints удалён) | **P1** |
| **global_step при resume** | ✅ **Исправлен** | **P1** |
| **DECAY_EVERY мёртвый код** | ✅ **Удалён** | P2 |
| **save_checkpoint_state мёртвый код** | ✅ **Удалён** | P2 |
| **EMA GPU sync (AM-30)** | ❌ Не исправлен | P2 |
| **_graph_cache leak (AM-32)** | ❌ Не исправлен | P2 |
| **Hormone reset (AM-33)** | ❌ Не исправлен | P2 |
| **Eval зависит от checkpoint** | ❌ Нет независимости | P2 |

---

## 10. Предложения TN-25+

### TN-25 (P0): Исправить `idx = -1` на `idx = idx - remaining` ✅ FIXED in V9 a0fe15b
В `_checkpoint` (train_full.py:465):
```python
# Applied fix: epoch_train[idx+1:] instead of epoch_train[remaining:], start_line=0 instead of -1
```

### TN-26 (P1): Удалить `cleanup_old_checkpoints()` ✅ FIXED
Оставить только `CheckpointManager.cleanup()`. `_final_save` больше не вызывает `cleanup_old_checkpoints`.

### TN-27 (P1): Сохранять и восстанавливать `global_step` ✅ FIXED
Добавить `global_step` в `checkpoint_state.json`:

### TN-28 (P2): Eval independent of checkpoint
```python
if idx >= self._eval_next_idx:
    self._eval_next_idx += self.cfg.eval_every_fast
```

### TN-29 (P2): Удалить мёртвый код ✅ FIXED
- `DECAY_EVERY` (train_full.py:509) — удалён
- `save_checkpoint_state` (train_full.py:77-80) — удалён
- Комментарий `"(only in run_epoch path)"` — исправлен

### TN-30 (P2): EMA Batched Update (GPU)
Заменить for-loop на тензорную операцию.

---

## 11. Вывод

V8 коммит (7ae6d9a) исправил **4 из 5 заявленных P0/P1 проблем**, но **TN-16 (idx=-1) не был применён к коду** — V8 report содержит некорректную информацию в разделе 9.

**V9 коммит (a0fe15b) исправил все P0/P1 блокеры**:
1. ✅ T-B1/`idx = -1` — фикс: start_line=0, epoch_train[idx+1:] вместо epoch_train[remaining:]
2. ✅ Двойная система очистки чекпоинтов — `cleanup_old_checkpoints()` удалён, остался только `CheckpointManager.cleanup()`
3. ✅ `global_step` при resume — сохраняется/восстанавливается через checkpoint_state.json
4. ✅ DECAY_EVERY и save_checkpoint_state — мёртвый код удалён

**Рекомендуемый порядок следующей итерации**: TN-28 → TN-30 → AM-32 → SN-22.1
