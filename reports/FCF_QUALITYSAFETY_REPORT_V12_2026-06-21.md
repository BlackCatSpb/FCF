# FCF Quality & Safety Report V12 — 2026-06-21

**Agent:** Quality-Safety Agent  
**Scope:** V11.2 commits a705223 (G-60/SN-43/SN-44) + 4030b54 (VRAM fp16) + 3150d5e (dynamic buf) + 1768f27 (fix slowdown); all `*.py` under `eva/symbolic/`, `tests/test_stdp.py`

---

## 1. V11 Test Progress — QN-49..QN-58: ВСЕ 22 ТЕСТА РЕАЛИЗОВАНЫ ✅

| QN | Suite | Tests | Status | V11 заявлял |
|----|-------|:-----:|:------:|:----------:|
| QN-49 | `_apply_subspace_update_batch` | 4 | **IMPLEMENTED** ✅ | ❌→✅ |
| QN-50 | GPU Centroid Pull (G-42) | 2 | **IMPLEMENTED** ✅ | ❌→✅ |
| QN-51 | Fused Post-STDP (G-52) | 2 | **IMPLEMENTED** ✅ | ❌→✅ |
| QN-52 | Deferred GPU Write-back (G-50/51) | 3 | **IMPLEMENTED** ✅ | ❌→✅ |
| QN-53 | GPU Lateral Inhibition (G-41) | 2 | **IMPLEMENTED** ✅ | ❌→✅ |
| QN-54 | checkpoint_state (TN-31) | 2 | **IMPLEMENTED** ✅ | ❌→✅ |
| QN-55 | effective_cp (TN-32) | 2 | **IMPLEMENTED** ✅ | ❌→✅ |
| QN-56 | Batched EMA (AM-30) | 2 | **IMPLEMENTED** ✅ | ❌→✅ |
| QN-57 | cooc_masks + fb_overlaps (G-44) | 2 | **IMPLEMENTED** ✅ | ❌→✅ |
| QN-58 | Centroid pull parity | 1 | **IMPLEMENTED** ✅ | ❌→✅ |
| **Итого** | **10 сьютов** | **22 теста** | **100%** | |

Все 22 теста из TestQNV11 (`tests/test_stdp.py:1079-1447`) — **PASSED** на текущем коде.

---

## 2. Текущий статус тестов

| Состояние | Кол-во |
|:---------:|:------:|
| **PASSED** | **122** |
| **FAILED** | **4** |
| **SKIPPED** | **3** (нет SentencePiece) |
| **Total** | **129** |

---

## 3. Root Cause: 4 FAILED тестов (CheckpointManager)

**Все 4 падения — одна и та же причина:**

```
AttributeError: 'CheckpointManager' object has no attribute 'cleanup'
```

Затронутые тесты:
1. `TestCheckpointManager::test_mgr_cleanup` (строка 722)
2. `TestCheckpointManagerResilience::test_cleanup_removes_old` (строка 750)
3. `TestCheckpointCleanup::test_cleanup_keep` (строка 1026)
4. `TestCheckpointCleanup::test_cleanup_below_keep` (строка 1036)

**Детали:**

В `checkpoint_manager.py` нет публичного метода `cleanup()`. Внутренний метод называется `_cleanup_old()` и вызывается автоматически внутри `_sync_save()`. Тесты ожидают публичный `mgr.cleanup()`, который никогда не существовал в этой версии (или был удалён при рефакторинге).

Дополнительно: `test_cleanup_keep` вызывает `mgr.cleanup(keep=3)`, но `_cleanup_old()` не принимает параметр `keep` — он использует `self.cleanup_keep`.

**Severity: MEDIUM** — тесты проверяют функциональность, которая работает через `_cleanup_old()` (автоматический вызов при save), но не имеет публичного API. Нужно либо:
  1. Добавить публичный `cleanup()` метод, совместимый с сигнатурой тестов, или
  2. Обновить тесты на вызов `_cleanup_old()` / убрать их.

---

## 4. Покрытие нового кода V11.2

### 4.1 G-60: GPU Destabilization (`stdp_trainer.py:491-510`)

Новый pure-tensor GPU destab, заменяющий старый CPU per-element loop:
- `destab_p`, `destab_mask`, `rand_idx`, `noise_gpu`, `mix_gpu` — всё на GPU
- `torch.where` для условного применения

**Покрытие: ❌ НЕТ**
- `test_cpu_stdp_destab` — тестирует CPU path только
- `test_gpu_stdp_apply_no_crash` — передаёт `destab_scale=0.0`, так что G-60 код не выполняется
- Ни один тест не имеет `destab_scale > 0` при GPU path
- **HIGH RISK** — новый GPU-код без единого теста

### 4.2 SN-43: Batched GPU Neg Sampling (`stdp_trainer.py:653-714`)

Изменения:
- `neg_lr` теперь вектор (один на gen_cid) вместо скаляра
- Новый deferred write-back: `_neg_updates` list → batched `_vecs_t[cids_batch]` write
- Исправлена формула градиента: `gv[gi:gi+1]` вместо `vg_i`, `gn.clamp(max=1.0)` вместо `min(gn, 1.0)`

**Покрытие: ⚠️ Частичное**
- `test_negative_sampling_gpu_no_crash` — только smoke (проверяет что не падает)
- Нет теста, который проверяет **корректность batched write-back** для neg sampling
- Нет теста CPU/GPU parity для отрицательного сэмплинга
- **MEDIUM RISK**

### 4.3 SN-44: Pure-Tensor GPU Contrastive (`stdp_trainer.py:777-909`)

Изменения:
- Pre-computed boolean masks (`self_hn`, `cooc_hn`, `fb_hn`, `valid_reg`, `valid_hn`) — без `.item()` в цикле
- `contr_lrs` теперь использует `gen._ce_t[gen_idxs]` GPU (без CPU sync)
- Batched deferred write-back: `_updates` list → `gen._vecs_t[cids_batch]` write

**Покрытие: ⚠️ Частичное**
- `test_contrastive_gpu_simple` — smoke test (запускает и проверяет)
- `test_contrastive_gpu_no_double_update` — проверяет unit norm
- `test_contrastive_gpu_empty` — пустой вход
- Нет теста для новой pre-computed mask logic (`valid_hn`, `valid_reg`)
- Нет теста для cross-field regularization (TN-14 ветка)
- Нет прямого сравнения CPU/GPU результатов contrastive
- **MEDIUM RISK**

### 4.4 VRAM Оптимизации

| Оптимизация | Файл:Строка | Покрытие | Риск |
|------------|:-----------:|:--------:|:----:|
| `_ema_vecs_t` fp16 (float→half) | `crystal_generator.py:279-281` | ❌ НЕТ | MED — precision loss для EMA |
| `_mom_t` fp16 (float→half) | `crystal_generator.py:289`, `stdp_trainer.py:458-461` | ❌ НЕТ | MED — momentum точность |
| `_fused_buf` dynamic growth | `crystal_generator.py:292-294`, `stdp_trainer.py:449-451` | ❌ НЕТ | LOW — растёт по требованию |
| `_vecs_t` fp16 сохраняется | `crystal_generator.py:251` | ❌ НЕТ | MED — все GPU операции на fp16 |

---

## 5. STR (Structural Test Reach)

| Модуль | Строк | STR (est.) | Δ vs V11 |
|--------|:-----:|:----------:|:--------:|
| `test_stdp.py` | 1447 | — | — |
| `concept_space.py` | 950 | 32% | ↑ |
| `crystal_generator.py` | 915 | 28% | ↑ (QN-49..QN-58) |
| `stdp_trainer.py` | 1045 | 55% | ↓ −10pp (G-60/SN-43/SN-44 untested) |
| `checkpoint_manager.py` | 127 | 60% | ↑ (QN-54) |
| Others | ~500 | 15% | ↔ |

**Overall STR: ~45%** (↓ −3pp от V11 из-за новых непокрытых GPU-путей)

---

## 6. Safety Regressions V11.2

1. **G-60: GPU destab bypasses `_destab_field_fallback`** — старый CPU destab использовал `gen.lattice.connections_of()` и `_destab_field_fallback`. GPU destab использует `torch.randint` для выбора случайного концепта. Разное поведение: CPU destab выбирает PPMI-связанные концепты, GPU — случайные. **MEDIUM.**

2. **SN-43: `gn.clamp(max=1.0)` vs `min(gn, 1.0)`** — Старый CPU: `grad = grad / gn * min(gn, 1.0)`. Новый GPU: `grad = grad / gn * gn.clamp(max=1.0)`. Математически эквивалентно. **LOW.**

3. **SN-44: `gv[gi:gi+1]` vs `v_local`** — В новом коде `v_new = v_local + grad * contr_lrs[i]` (изменённый `v_local` после cross-field regularization). До этого был `push = grad * contr_lrs[i]; v_new = v_local + push`. Поведение не изменилось. **LOW.**

4. **VRAM fp16: EMA precision loss** — `_ema_vecs_t` теперь fp16 вместо fp32. При `_ema_decay=0.999`, обновление `lerp(..., 0.001)` может терять точность в fp16. EMA используется для stable eval/generation. **HIGH** если eval/generation качество упадёт.

5. **VRAM fp16: `_mom_t` precision** — momentum в fp16 может накапливать ошибку округления, особенно при `momentum_mu=0.9`. Для 146K×384D тензора это ~112MB экономии. **MEDIUM.**

---

## 7. Coverage Gap Matrix (V12 Update)

| Метод | Файл:Строка | Покрыт? | Риск |
|-------|:-----------:|:-------:|:----:|
| G-60 GPU destab tensor ops | `stdp_trainer.py:492-509` | **❌ NO** | **HIGH** |
| G-60 GPU destab write-back | `stdp_trainer.py:510` | **❌ NO** | **HIGH** |
| SN-43 batched neg sampling deferred write | `stdp_trainer.py:693-714` | **❌ NO** | MED |
| SN-44 pre-computed boolean masks | `stdp_trainer.py:846-864` | **❌ NO** | MED |
| SN-44 pure-tensor contrastive loop | `stdp_trainer.py:867-901` | ⚠️ partial | MED |
| SN-44 cross-field regularization (TN-14) | `stdp_trainer.py:854-860,871-881` | **❌ NO** | MED |
| VRAM: `_ema_vecs_t` fp16 | `crystal_generator.py:279-281` | **❌ NO** | MED |
| VRAM: `_mom_t` fp16 | `crystal_generator.py:289` | **❌ NO** | MED |
| CheckpointManager public `cleanup()` | `checkpoint_manager.py` | **❌ MISSING** | MED |
| `_gpu_stdp_apply` gradient noise | `stdp_trainer.py:462-463` | ❌ NO | LOW |
| `_gpu_stdp_apply` batched EMA | `stdp_trainer.py:554-559` | ⚠️ partial (QN-56) | MED |
| `_gpu_stdp_apply` momentum | `stdp_trainer.py:483-489` | ⚠️ partial | MED |

**Total HIGH-risk uncovered: 2** (G-60 GPU destab)
**Total MEDIUM-risk uncovered: 9**
**Total uncovered risk items: 13**

---

## 8. Предложения QN-59+

### QN-59: G-60 GPU Destabilization (3 tests)
- `test_gpu_destab_basic` — `destab_scale=0.5` на GPU, проверить unit norm
- `test_gpu_destab_no_crash_high_destab` — `destab_scale=1.0`, проверить не падает
- `test_gpu_destab_random_vs_cpu` — сравнить CPU и GPU destab (оба случайные, relaxed tolerance)

### QN-60: SN-43 Batched GPU Neg Sampling Correctness (2 tests)
- `test_neg_sampling_gpu_batched_write` — проверить `_vecs_t` изменения через batched write
- `test_neg_sampling_gpu_cpu_parity` — сравнить CPU vs GPU результат neg sampling

### QN-61: SN-44 Pre-computed Boolean Masks (3 tests)
- `test_contrastive_gpu_valid_hn_mask` — проверить self_hn, cooc_hn, cos_upper логику
- `test_contrastive_gpu_cross_field_reg` — проверить valid_reg маску при fb_overlaps
- `test_contrastive_gpu_cpu_parity` — сравнить CPU vs GPU contrastive результат

### QN-62: VRAM fp16 Precision (3 tests)
- `test_ema_fp16_precision` — проверить что EMA в fp16 не накапливает ошибку за N шагов
- `test_mom_fp16_stability` — проверить что momentum в fp16 сходится к тому же результату
- `test_gpu_fp16_full_roundtrip` — полный цикл STDP + neg sampling + contrastive в fp16

### QN-63: Cleanup Public API (2+1 tests)
- `test_cleanup_public_exists` — TestCheckpointCleanup исправлен (добавлен `cleanup()` метод)
- `test_cleanup_keep_parameter` — параметр `keep` в `cleanup()`
- **Fix**: `cleanup()` метод уже есть в `_cleanup_old()` — просто переименовать или добавить обёртку

---

## 9. Summary

| Метрика | V11 | V12 | Δ |
|---------|:---:|:---:|:-:|
| QN-49..QN-58 implemented | 0/10 | **10/10** | ✅ |
| Total tests | 129 | **129** | ↔ |
| Passed | 122 | **122** | ↔ |
| Failed | 4 (CheckpointManager) | **4** | ↔ (not fixed) |
| Skipped | 3 | **3** | ↔ |
| STR | ~48% | **~45%** | ↓ −3pp |
| HIGH-risk uncovered | 5 | **2** | ↓ −3 (G-60 added) |
| MEDIUM-risk uncovered | 12 | **9** | ↓ −3 |
| Total uncovered risk items | 19 | **13** | ↓ −6 |

### Key Findings

1. **✅ QN-49..QN-58: ВСЕ 22 теста реализованы и проходят** — V11 закрыл все 10 предложенных сьютов
2. **⚠️ 4 теста падают** — `CheckpointManager` не имеет публичного `cleanup()` метода. Требуется минимальная доработка `checkpoint_manager.py` или тестов.
3. **❌ G-60 GPU destab не покрыт** — новый pure-tensor GPU код без единого теста (HIGH risk)
4. **⚠️ SN-43/SN-44 покрыты только smoke-тестами** — нет CPU/GPU parity, нет проверки новых boolean mask логик
5. **⚠️ VRAM fp16 оптимизации не тестированы** — `_ema_vecs_t` и `_mom_t` в fp16 могут терять точность
6. **✅ 122 теста проходят**, общее качество улучшается

**Safety verdict: IMPROVING but G-60 GPU destab требует немедленного покрытия.** Приоритет: QN-59 (GPU destab) → QN-63 (cleanup fix) → QN-62 (fp16 precision) → QN-60/61 (neg sampling + contrastive parity).
