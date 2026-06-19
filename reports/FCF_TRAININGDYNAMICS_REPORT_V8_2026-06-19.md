# FCF Training-Dynamics Report V8 — Аудит цикла обучения

**Дата**: 2026-06-19  
**Агент**: Training-Dynamics Agent  
**Версия**: V8 (аудит после V7→V8 изменений)

---

## 1. Статус T-B* и TN-* из V7

| ID | Приоритет | Описание | Статус V8 | Детали |
|:--:|:---------:|----------|:---------:|--------|
| **T-B1** | P0 | Self-Paced Learning — idx сброс после rescore | ❌ **Не исправлен** | `_rescore_lines` в `_checkpoint` есть, но return-значения игнорируются в main loop (строка 820). Rescore мёртвый код. |
| **T-B2** | P0 | TrainingPipeline не активирован | ⚠️ **Частично** | `pipeline` создан (строка 684), но `run_epoch()` **никогда не вызывается**. Main loop (строки 710-830) дублирует `run_epoch`. Два параллельных цикла. |
| **T-B3** | P1 | EMA после apply (инвертирована) | ✅ **Исправлен** | В `_gpu_stdp_apply` (строка 466-470): EMA обновляется ДО `_apply_vector_update`, использует старый `_vecs_t`. |
| **T-B4** | P1 | destab_decay_lines дублирование | ✅ Исправлен | Чисто. |
| **TN-6** | P0 | noise_scale не передавался | ⚠️ **Неполно** | Параметр добавлен в STDPTrainer (default=0.0), но `train_full.py` (строка 752) не передаёт `noise_scale` в `gen.train_batch()`. Фактически `noise_scale=0.0` всегда. |
| **TN-11** | P1 | Gradient Accumulation (momentum_mu) | ✅ **Исправлен** | `momentum_mu=0.9` в default. `_mom_buf` используется. |
| **TN-12** | P1 | Switched Evaluation | ✅ **Исправлен** | `_eval_count` корректно переключает fast/full. Формула `(_eval_count * eval_every_fast) % eval_every_full == 0` — верна. |
| **TN-13** | P2 | Progressive Batch Size | ❌ **Не реализован** | Только базовая кривая `bs_curve`. Плато-детектор отсутствует. |
| **TN-14** | P1 | Field-Aware Contrastive Regularization | ✅ **Реализован** | В `_contrastive_objective_gpu` (строки 715-741). Но есть **баги** (см. раздел 3). |
| **TN-15** | P2 | Decay Warmup Protect Threshold | ❌ **Не реализован** | `rare_threshold=3` константа. |

---

## 2. Проблемы TrainingPipeline (после активации — регрессии)

### 2.1. run_epoch() не вызывается — двойной цикл (P0)
`train_full.py` имеет **два цикла обучения**:
- `TrainingPipeline.run_epoch()` (строки 397-457) — никогда не вызывается
- Main loop (строки 710-830) — непосредственно в `try` блоке

**Проблема**: `run_epoch` не содержит:
- `lattice.decay_all` / `lattice.decay_connections` / `cs.decay_usage`
- `cs.fluctuate_fractal`
- `live_status` вывод
- `batch_log` CSV
- `full_stuck` forced fluctuate
- Status JSON (`_train_status.json`)

Поэтому `run_epoch` не может заменить main loop без доработки.

### 2.2. `_checkpoint` return-значения игнорируются (P0)
Строка 820:
```python
pipeline._checkpoint(epoch, idx, elapsed, epoch_lines, destab_scale, t_start)
```
Return `(idx, start_line, epoch_train)` **не захвачен**. В результате:
- `_rescore_lines` внутри `_checkpoint` выполняется, но `epoch_train` в main loop не меняется
- `idx = -1` внутри `_checkpoint` не влияет на main loop

### 2.3. `epoch_train` и `start_line` не передаются в `_checkpoint`
Строка 820 вызывает `_checkpoint` без `epoch_train`/`start_line`. Внутри `_checkpoint` (строка 522) проверка `epoch_train is not None` → False → rescope не выполняется.

---

## 3. T-B1: Self-Paced Learning — анализ idx после rescore

### 3.1. Rescore не работает в main loop (P0)
Даже если исправить передачу параметров, rescore имеет **логическую ошибку V7**:

```python
# train_full.py:522-526 (в _checkpoint)
remaining = idx - start_line + 1
if remaining > 0 and remaining < len(epoch_train):
    epoch_train = _rescore_lines(epoch_train[remaining:], gen)
    idx = -1; start_line = -1
```

После `idx = -1`, `idx += 1` на строке 456 → `idx = 0`. Это сбрасывает `_curriculum_p(idx)` и `_curriculum_max_len(idx)` в начальное состояние: batch_size падает до `bs_curve(0)`, `max_len` до `CURICULUM_MIN_LEN=16`.

**Корректное решение**: `idx -= remaining` (перемотка на начало ресортированного суффикса), не сброс в 0.

### 3.2. `start_line` в main loop никогда не меняется
В main loop `start_line` установлен один раз (строка 674 = 0 или resume_line) и никогда не обновляется. Для корректного rescore нужно:

```python
start_line = idx  # перед вызовом _checkpoint
```

---

## 4. TN-12: Switched Evaluation — анализ

### 4.1. Логика корректна
```python
self._eval_count += 1
is_full = (self._eval_count * self.cfg.eval_every_fast) % self.cfg.eval_every_full == 0
```
`eval_every_fast=1000`, `eval_every_full=5000`:
- idx=1000: cnt=1, 1×1000%5000=1000 → fast ✓
- idx=2000: cnt=2, 2×1000%5000=2000 → fast ✓
- idx=3000: cnt=3, 3×1000%5000=3000 → fast ✓
- idx=4000: cnt=4, 4×1000%5000=4000 → fast ✓
- idx=5000: cnt=5, 5×1000%5000=0 → full ✓

### 4.2. Проблема: `eval_full_lines`/`eval_fast_lines` из конфига не синхронизированы
`fcf_config.py:376-377`:
```python
eval_fast_lines: int = 64
eval_full_lines: int = 300
```
`gen.evaluate` на строке 494 вызывается с `max_lines=`. Всё корректно.

### 4.3. Уязвимость: eval вызывается из `_checkpoint`, который вызывается из main loop на строке 820
Но `eval_every_fast=1000` и `checkpoint_every=500`. Значит, eval происходит на каждом 2-м чекпоинте. Это нормально, но если изменить `checkpoint_every`, eval станет реже/чаще. Нет независимости.

---

## 5. TN-14: Field-Aware Contrastive Regularization — баги

### 5.1. Batch-Stale Vector (P1)
В `_contrastive_objective_gpu` (строки 673, 715-754):
```python
g_vecs = gen._vecs_t[gen_idxs].float()  # строка 673 — КОПИЯ
# ...
# TN-14 блок: cs._apply_vector_update(gen_idxs[i], v2)  # строка 741
# ...
# Main push: v_new = g_vecs[i] + push  # строка 750 — использует СТАРЫЙ g_vecs[i]
```

Если TN-14 сработал для gen_cid i, то последующий main push на строке 750 использует **старый** `g_vecs[i]` (до TN-14 update). Обновление TN-14 перезаписывается.

### 5.2. Множественные `_apply_vector_update` на одну итерацию (P1)
TN-14 вызывает `_apply_vector_update` внутри цикла по `min(100, topk_idx.shape[1])` кандидатам. Для одного gen_cid может быть до 100 вызовов `_apply_vector_update` за батч. Это:
- GPU sync на каждый вызов
- Несколько обновлений одного концепта за батч (конфликтующие градиенты)

---

## 6. Дополнительные проблемы V8

### 6.1. `noise_scale` не передаётся в train_batch (P1)
Строка 752:
```python
n_pairs = gen.train_batch(batch_buffer, pmi_strength=..., ...)
```
Параметр `noise_scale` отсутствует. `opt.p['noise_scale'].current` существует в `ParameterOptimizer`, но не передаётся.

### 6.2. `momentum_mu` не передаётся в train_batch (P1)
Там же: `momentum_mu` отсутствует. Default 0.9 в STDPTrainer (из V8 фикса), но явная передача отсутствует.

### 6.3. `_graph_cache` неограниченный рост (P2 — AM-32)
`crystal_generator.py:71`: `_graph_cache` — обычный dict без эвикции. Растёт бесконечно, хранит до `n_pairs × 10` записей.

### 6.4. `HormonalSystem.reset()` отсутствует (P2 — AM-33)
Состояние гормонов (ACh, DA) переносится между вызовами `generate()`. Строка 89: `self.hormones = HormonalSystem()` — один раз в `__init__`.

### 6.5. EMA CPU-for-loop GPU sync (P2 — AM-30)
`stdp_trainer.py:466-470`: per-concept for-loop с `gen._ema_vecs_t[gen_cid] = ...`. Каждая итерация — GPU sync.

---

## 7. Предложения TN-16+

### TN-16 (P0): Rescore с сохранением позиции
Исправить T-B1:
```python
# Вместо idx = -1; start_line = -1
idx = idx - remaining  # перемотка к началу rescored суффикса
start_line = idx - remaining  # или просто сохранить
```
А также передавать `epoch_train` и `start_line` в `_checkpoint` из main loop.

### TN-17 (P0): Устранить двойной цикл
Либо:
- (a) Дополнить `run_epoch` недостающими фичами (decay, fluctuate, live_status, batch_log, status JSON) и вызвать его вместо main loop
- (b) Удалить `run_epoch`, удалить `_rescore_lines` из `_checkpoint`, перенести rescore в main loop

### TN-18 (P1): Передача noise_scale и momentum_mu в train_batch
```python
gen.train_batch(..., noise_scale=opt.p['noise_scale'].current,
                momentum_mu=0.9)
```

### TN-19 (P1): Fix TN-14 batch-stale vector
Перечитать `g_vecs[i]` из `_vecs_t` после TN-14 update, или сохранять v в локальной переменной:
```python
v_local = g_vecs[i].clone()
# TN-14: обновлять v_local вместо cs._apply_vector_update
# Main push: использовать v_local
cs._apply_vector_update(gen_idxs[i], v_new.cpu().numpy())  # единственный apply
```

### TN-20 (P1): Rescore с сохранением curriculum
После rescore `idx` должен быть скорректирован так, чтобы curriculum продолжался с того же процента прогресса:
```python
p_before = _curriculum_p(idx_before)
# после rescore: подобрать idx так, чтобы _curriculum_p(idx) ≈ p_before
```

### TN-21 (P2): EMA batched update (GPU)
```python
gen_t = torch.tensor(unique_gen, device=device)
gen._ema_vecs_t[gen_t] = ema_decay * gen._ema_vecs_t[gen_t] + (1 - ema_decay) * gen._vecs_t[gen_t].float()
```

### TN-22 (P2): Hormone reset per generate
```python
def generate(self, ...):
    self.hormones.reset()
    ...
```

### TN-23 (P2): LR warmup respect after rescore
После rescore не сбрасывать `get_lr(idx)` — использовать глобальный `idx`, не локальный.

### TN-24 (P2): Eval independent of checkpoint interval
Сделать eval по собственному счётчику `_eval_next_idx`, а не `idx % checkpoint_every == 0`:

```python
if idx >= self._eval_next_idx:
    self._eval_next_idx += self.cfg.eval_every_fast
    # ... eval
```

---

## 8. Матрица статуса V8

| Область | Статус |
|---------|--------|
| T-B1 (Self-Paced idx) | ❌ Не исправлен — rescore dead code |
| T-B2 (TrainingPipeline) | ⚠️ Частично — создан, но не используется |
| T-B3 (EMA before apply) | ✅ Исправлен |
| TN-6 (noise_scale) | ⚠️ API есть, но не вызывается |
| TN-11 (momentum_mu) | ✅ Исправлен (0.9) |
| TN-12 (Switched Eval) | ✅ Исправлен |
| TN-14 (Field-Aware Contrastive) | ⚠️ Реализован, но batch-stale bug |
| TN-13/15 | ❌ Не реализованы |
| Двойной цикл (regression) | ❌ P0 — два параллельных цикла |
| _checkpoint return ignored | ❌ P0 — rescope не доходит |
| noise_scale/momentum в train_batch | ❌ P1 — не передаются |
| _graph_cache leak | ❌ P2 — AM-32 |
| Hormone reset | ❌ P2 — AM-33 |

---

## 9. Вывод

V8 внёс критические исправления (T-B3, TN-11, TN-12), но T-B1 (rescore) остался сломан, а T-B2 (TrainingPipeline) создал **регрессию двойного цикла** — `run_epoch()` жив, но мёртв, а main loop не получает rescore.

**Три P0 блокера**:
1. Rescore не работает — return `_checkpoint` игнорируется
2. Два параллельных цикла обучения — `run_epoch` не вызывается
3. TN-14 batch-stale vector — перезапись обновлений

## V8 Commit Fixes

| Проблема | Статус | Фикс |
|:---------|:------:|------|
| **T-B1 rescore** | ✅ Исправлен | Main loop захватывает return `_checkpoint`. |
| **T-B2 run_epoch** | ✅ Исправлен | `run_epoch()` удалён. Main loop — единственный цикл. |
| **TN-14 batch-stale** | ✅ Исправлен | `v_local = g_vecs[i].clone()`, единственный `_apply_vector_update`. |
| **noise_scale** | ✅ Исправлен | Передаётся в `gen.train_batch()`. |
| **momentum_mu** | ✅ Исправлен | Явно передаётся `momentum_mu=0.9` в `train_batch()`. |
| **TN-16 (rescore)** | ✅ Исправлен | TN-16 реализован: `_checkpoint` return захвачен, `epoch_train` и `start_line` передаются. |
| **TN-17 (dual loop)** | ✅ Исправлен | Вариант (b): `run_epoch` удалён. |
| **TN-18 (noise/momentum)** | ✅ Исправлен | Параметры передаются. |
| **TN-19 (TN-14 fix)** | ✅ Исправлен | Локальная копия `v_local`. |

**Рекомендованный порядок фикса**: TN-16 → TN-17 → TN-18 → TN-19 → TN-21..24
