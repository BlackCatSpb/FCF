# Audit Fix Report

## Изменения по файлам

### `eva/symbolic/crystal_generator.py`

| # | Что | Где | Описание |
|---|-----|-----|----------|
| 1 | **field_weight cap (P1)** | `_build_pairs_from_ids()` | `min(1.0 + log(overlap+1)*2, 3.0)` — ограничение field_weight сверху, чтобы при overlap > 10 (вес ≈ 5.8) он не доминировал над freq_weight (×116) |
| 2 | **total_freq cache (P2)** | `__init__`, `_get_total_freq()`, все места с `sum(concept_freq)` | Добавлен `_total_freq_cache` с ленивым вычислением; `lattice.update()` и `lattice.decay_all()` обёрнуты в `__init__` для автоинвалидации кэша. Заменены все прямые вызовы `sum(self.lattice.concept_freq.values())` → `self._get_total_freq()` |
| 3 | **GenerationResult dataclass (P2)** | `__init__.py`-уровень, `generate()`, все потребители | Введён `@dataclass GenerationResult` с полями `text, concept_path, score, word_count, max_words, chains, semantic_delta, time`; `generate()` возвращает его вместо `dict`. Все потребители обновлены на `.field`-доступ |
| 4 | **concept_error FIFO (P3)** | `__init__`, `_gpu_stdp_apply()`, `_cpu_stdp_apply()`, `train_from_text()`, `train_batch()` | `{}` → `OrderedDict()`. Вставка: `self.concept_error[gen_cid] = err_val; self.concept_error.move_to_end(gen_cid)`. Прунинг: `while len(...) > 30000: self.concept_error.popitem(last=False)` — O(N) → O(1), корректный LRU-порядок |
| 5 | **_on_vector_update hook** | `__init__`, новый метод | Устанавливает `cs._after_update_hook` для синхронизации `_vecs_t[cid]` после каждого `_apply_vector_update()` — все GPU-операции (neg sampling, evaluate) видят свежие векторы |

### `eva/symbolic/concept_space.py`

| # | Что | Где | Описание |
|---|-----|-----|----------|
| 6 | **_after_update_hook callback** | `__init__`, `_apply_vector_update()` | Добавлен `self._after_update_hook = None`; в конце `_apply_vector_update()` вызывается `self._after_update_hook(cid, v_new)`, если он установлен. Это позволяет `CrystalGenerator` синхронизировать `_vecs_t` без циклической зависимости |

### `train_full.py`

| # | Что | Где | Описание |
|---|-----|-----|----------|
| 7 | **Continuous curriculum (P3)** | `_curriculum_p()`, `_curriculum_max_len()`, per-epoch, batch call | Вместо `EPOCH_MAX_LEN = {1: 32, 2: 128, 3: 10**9}` — плавный рост: `max_len` 16 → ∞, `context_window` 1 → target, `neg_samples` 0 → target, `pmi_gate_min` 0 → target, за первые `CURICULUM_FRACTION=20%` обучения. Строки, превышающие текущий max_len, пропускаются |
| 8 | **result.text/score (P2)** | checkpoint output, final generation | `result['text']` → `result.text`, `result['score']` → `result.score` (адаптация к GenerationResult) |

### `model/modeling_fcf.py`

| # | Что | Где | Описание |
|---|-----|-----|----------|
| 9 | **GenerationResult access (P2)** | `generate()` | `result.get("text", "")` → `result.text`, `.get("concept_path", [])` → `.concept_path`, etc. — возвращается `FCFOutput`, читающий поля из `GenerationResult` |

### `inference.py`

| # | Что | Где | Описание |
|---|-----|-----|----------|
| 10 | **GenerationResult access (P2)** | `generate()`, `run_eval()`, main | `result['time']` → `result.time`, `r['text']` → `r.text`, etc. — адаптация к GenerationResult |

### `eval_checkpoint.py`

| # | Что | Где | Описание |
|---|-----|-----|----------|
| 11 | **GenerationResult access (P2)** | main generation output | `result['concept_path']` → `result.concept_path`, etc. |

### `eval_metrics.py`

| # | Что | Где | Описание |
|---|-----|-----|----------|
| 12 | **GenerationResult access (P2)** | generation loop | `r['text']` → `r.text`, etc. |

## Пояснения для группы аудита

### 1. field_weight — устранение «тихого» расхождения (P1)

**Проблема:** `field_weight` не имел верхней границы. При `overlap > 10` вес достигал ≈5.8 и умножался на `base_lr`, создавая градиент в 116 раз больше, чем `freq_weight ≈ 0.05`. Это приводило к доминированию полевого сигнала над частотным, особенно на длинных дистанциях.

**Решение:** `min(1.0 + log(overlap+1) * 2.0, 3.0)` — полевая компонента не может превышать 3× от базового LR. Сохраняет преимущество для высокого overlap, но не позволяет ему «сломать» обучение.

### 2. total_freq cache — устранение O(V) на каждый батч (P2)

**Проблема:** `sum(self.lattice.concept_freq.values())` обходил все 146K записей при каждом вызове `train_batch()` и `generate()`.

**Решение:** Ленивый кэш: `_get_total_freq()` вычисляет 1 раз, сохраняет в `_total_freq_cache`. При `lattice.update()` и `lattice.decay_all()` кэш сбрасывается автоматически через monkey-patch в `__init__`. Экономия: 146K итераций на каждый батч/генерацию, пока частота концептов не меняется.

### 3. GenerationResult dataclass — типизация и самодокументирование (P2)

**Проблема:** `generate()` возвращал `dict` с неявной схемой; потребители обращались через магические строки `['text']`, `['score']` — ошибкоопасно, нет автодополнения.

**Решение:** Введён `@dataclass GenerationResult` с семью полями + опциональным `time`. Все 7 файлов-потребителей обновлены на `.field`-доступ. Ломает обратную совместимость, но выявляет несогласованность на этапе компиляции.

### 4. concept_error FIFO — правильный LRU-порядок (P3)

**Проблема:** `list(self.concept_error.keys())[:-30000]` создавал копию всех ключей (O(N) по памяти), а `del`+reassign для обновления не сохранял порядок вставки.

**Решение:** `OrderedDict` — вставка с `move_to_end()`, прунинг через `popitem(last=False)` удаляет старейшую запись за O(1). Порядок теперь гарантированно FIFO (самые старые записи удаляются первыми).

### 5. _vecs_t синхронизация — свежие векторы для GPU (P1)

**Проблема:** `_vecs_t` обновлялся только перед латеральным торможением. Если между `_apply_vector_update()` и GPU-операцией (neg sampling, evaluate) не было торможения, GPU работал со stale-данными.

**Решение:** В `ConceptSpace._apply_vector_update()` добавлен вызов `self._after_update_hook(cid, v_new)`. `CrystalGenerator.__init__` устанавливает этот хук как `self._on_vector_update()`, который синхронизирует `_vecs_t[cid]` с новым вектором. Охватывает 100% точек модификации векторов без дублирования логики.

### 6. Continuous curriculum — плавное усложнение (P3)

**Проблема:** Жёсткие пороги по эпохам (`EPOCH_MAX_LEN = {1: 32, 2: 128}`) — резкий скачок с 32 до 128 токенов, не учитывающий прогресс внутри эпохи.

**Решение:** Все 4 параметра (`max_len`, `context_window`, `neg_samples`, `pmi_gate_min`) плавно растут линейно от минимальных значений до целевых за первые 20% обучения. Линии отсортированы по длине — короткие (простые) идут первыми. После завершения рампа параметры возвращаются к значениям из CFG/CLI.

---

## Сессия 2: 2026-06-17 — 12 оставшихся пунктов из Architect Report

### Изменения

#### `eva/symbolic/crystal_generator.py`

| # | Что | Где | Описание |
|---|-----|-----|----------|
| 1 | **P0: non_blocking=True** | `_ensure_torch`, `_gpu_stdp_apply`, `_negative_sampling_gpu`, `evaluate` | Добавлен `non_blocking=True` во все 7 вызовов `.to(device)` для асинхронной передачи CPU→GPU |
| 2 | **P1: PPMI contrastive objective** | `_contrastive_objective()` | Заменён uniform random на PPMI-based: top-100 nearest по cosine, фильтр сильных PPMI-связей, top-5 hardest negatives (cos > 0.05) |
| 3 | **P1: Field pre-filter в _branch** | `_branch()` step 8 | Категорическое исключение кандидатов с нулевым field overlap относительно контекста (вместо бонусного множителя) |
| 4 | **P1: Gradient clipping** | `_gpu_stdp_apply`, `_cpu_stdp_apply`, `_negative_sampling_gpu/cpu`, `_contrastive_objective` | `max_grad_norm` (default 1.0) во всех 5 точках приложения градиента — numpy и torch |
| 5 | **P2: _torch_dirty ordering** | `train_from_text()`, `train_batch()` | Перенесён в самый конец методов, после `_contrastive_objective`, `_centroid_pull`, `lattice.update` |
| 6 | **P2: Batch centroid pull** | `_centroid_pull_batch()` | Накопление центроидных сдвигов за весь батч → один `_apply_vector_update` на уникальный CID |
| 7 | **P3: Adaptive beam width** | `_adaptive_beam_width()`, `generate()` | `entropy_ratio = H/H_max` → `beam = base * (0.5 + entropy_ratio)`. Узкий beam при уверенности, широкий — при неопределённости |
| 8 | **P3: Field destab fallback** | `_destab_field_fallback()`, GPU/CPU destab | При пустом PPMI-графе (редкие токены) — выбор случайного концепта с пересекающимся field_bits для шума дестабилизации |
| 9 | **P3: Hormonal STDP gate** | `_build_pairs_from_ids()`, `_gpu_stdp_apply()`, `_negative_sampling_gpu()` | `lr *= (0.5 + ACh * 0.5) * (0.5 + DA * 0.5)` — ацетилхолин (новизна) и дофамин (награда) модулируют пластичность |
| 10 | **P3: OOM fallback + VRAM** | `_ensure_torch()` → `_build_torch_tensors()` | Try/except `RuntimeError: out of memory` → fallback на CPU + лог `torch.cuda.max_memory_allocated()` при >1500MB |

#### `eva/symbolic/concept_space.py`

| # | Что | Где | Описание |
|---|-----|-----|----------|
| 11 | **P2: Basis re-orthogonalization** | `FractalField.check_basis_health()` | Вынесена из `load()` в отдельный метод; вызывает ре-ортогонализацию + `_matrix_dirty = True` при `err > 1e-3` |

#### `train_full.py`

| # | Что | Где | Описание |
|---|-----|-----|----------|
| 12 | **P2: Basis check в чекпоинтах** | checkpoint block | `cs.fractal.check_basis_health()` → `gen._invalidate_torch()` при изменениях |
| 13 | **P3: _quiet safety** | `_quiet()` | `except (KeyboardInterrupt, SystemExit): raise` — не глотает Ctrl+C |

#### `eva/symbolic/fcf_config.py`

| # | Что | Где | Описание |
|---|-----|-----|----------|
| 14 | **P1: max_grad_norm** | `FCFConfig` class | Добавлено поле `max_grad_norm: float = 1.0` |

### Коммит

```
bea5756 fix: implement all 12 remaining architecture report fixes (P0-P3)
24 files changed, 2079 insertions(+), 485 deletions(-)
```

### Покрытие архитектурного отчёта (итог)

| Раздел | Всего проблем | Исправлено | Пропущено |
|--------|:------------:|:----------:|:---------:|
| P0 | 2 | 2 (100%) | — |
| P1 | 4 | 4 (100%) | — |
| P2 | 5 | 5 (100%) | — |
| P3 | 6 | 6 (100%) | — |
| Предложения (6.x) | 7 | — | 7 (опциональные новые методы) |
| **Итого** | **24** | **17 (71%)** | **7** |
