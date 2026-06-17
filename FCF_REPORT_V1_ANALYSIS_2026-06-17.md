# Анализ отчёта V1 и сверка с реализацией V2

## Методология

Сверил каждый пункт отчёта V1 (Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent) с фактическими изменениями в коде после 4 коммитов. Для каждого пункта — статус, gap-анализ и вывод.

---

## 1. Проверка Architect-AI

### A-1: Циркулярная зависимость ConceptSpace ↔ CrystalGenerator

| Пункт | Статус |
|-------|--------|
| V1: Ввести protocol `TorchCacheOwner` | ❌ Не реализовано |
| V1: Или `TorchCacheManager` | ❌ Не реализовано |
| V2: Внедрён `_after_update_hook` | ✅ Реализовано |

**Вывод**: Задача решена частично — hook устранил жёсткую связь для уведомлений об обновлении векторов, но `fluctuate_fractal()` всё ещё принимает `generator=None` и вызывает `generator._invalidate_torch()` напрямую.

**Решение**: внедрить protocol:
```python
from typing import Protocol

class TorchInvalidatable(Protocol):
    def _invalidate_torch(self) -> None: ...

def fluctuate_fractal(self, ..., generator: Optional[TorchInvalidatable] = None):
    ...
```
Это формализует контракт без циклической зависимости.

### A-2: FCFConfig — God Object

| Пункт | Статус |
|-------|--------|
| V1: Выделить PathConfig | ❌ Не реализовано |
| V1: Выделить MetricPairBuilder | ❌ Не реализовано |

**Вывод**: `FCFConfig` не изменился. 440 строк, 5+ ответственностей.

**Решение**: сейчас не критично (работает), но при добавлении новых builder-ов (например, `build_relation_pairs`) без рефакторинга разрастётся до 600+ строк. Установить порог: при превышении 500 строк — форсировать рефакторинг.

### A-3: Нет абстракции для генерации

| Пункт | Статус |
|-------|--------|
| V1: Создать `GenerationResult` dataclass | ✅ Реализовано |

**Gap**: `inference.py` (строки 184, 240, 266) и `eval_checkpoint.py` (строка 48) всё ещё используют `result['text']` вместо `result.text`.

**Решение**: патч inference.py и eval_checkpoint.py для использования атрибутов dataclass.

### A-4: Нет мониторинга ресурсов

| Пункт | Статус |
|-------|--------|
| V1: `torch.cuda.max_memory_allocated()` | ❌ Не реализовано |
| V1: OOM fallback на CPU | ❌ Не реализовано |

**Вывод**: GPU MX550 с 2GB VRAM подвержен OOM при больших batch. Нет ни одного измерения.

**Решение**: добавить в `_ensure_torch()`:
```python
if hasattr(torch.cuda, 'max_memory_allocated'):
    allocated_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  GPU memory: {allocated_mb:.0f}MB / {torch.cuda.get_device_properties(0).total_memory/1024**2:.0f}MB")
```

---

## 2. Проверка Neuro-Symbolic Specialist

### S-1: field_weight доминирование

| Пункт | Статус |
|-------|--------|
| V1: Ограничить сверху `min(..., 3.0)` | ✅ Реализовано |

**Код**: `crystal_generator.py:1069` — `min(1.0 + math.log(overlap + 1) * 2.0, 3.0)`

**Вывод**: решено. Дополнительно: сделать cap конфигурируемым `field_weight_max: float = 3.0` в FCFConfig.

### S-2: Contrastive objective — неэффективные кандидаты

| Пункт | Статус |
|-------|--------|
| V1: PPMI-ранжированные негативы | ❌ Не реализовано |

**Вывод**: contrastive objective всё ещё использует uniform random sampling 80 кандидатов из 146K.

**Решение**: перевести на PPMI-based отбор (см. метод 6.6 из V1). Сейчас с hook `_after_update_hook` можно легко добавить GPU-версию.

### S-3: Нет адаптивного neg_sampling

| Пункт | Статус |
|-------|--------|
| V1: adaptive_neg_sampling по concept_error | ❌ Не реализовано |

**Вывод**: `neg_samples` — константа для всех пар. Нет hard negative mining.

**Решение**: добавить weighting на основе `concept_error[cid]`:
```python
error_weight = 1.0 + self.concept_error.get(cid, 0.0) * 2.0
effective_neg = max(1, int(neg_samples * error_weight))
```

### S-4: field_overlap bottleneck

| Пункт | Статус |
|-------|--------|
| V1: LRU cache unpackbits | ❌ Не реализовано |
| V1: GPU field_overlap matmul | ❌ Не реализовано |

**Вывод**: `np.unpackbits(np.bitwise_and(ba, bb)).sum()` вызывается для каждой пары.

**Решение**: поскольку `_fb_t` уже есть на GPU, можно вычислять overlap через torch:
```python
def field_overlap_gpu(cid_a, cid_b):
    fb_a = self._fb_t[cid_a]
    fb_b = self._fb_t[cid_b]
    return int(torch.bitwise_and(fb_a, fb_b).sum().item())
```
Но для CPU-пути — LRU cache на 10000 unpackbits результатов.

### S-5: Дестабилизация без fallback

| Пункт | Статус |
|-------|--------|
| V1: field_bits fallback | ❌ Не реализовано |

**Вывод**: редкие токены без PPMI-соединений никогда не дестабилизируются.

**Решение**: fallback на `field_bits`:
```python
if not ppmi_candidates:
    # Выбрать случайный концепт с пересекающимся field_bits
    fb_gen = cs.fractal.get_field_bits(gen_cid)
    if fb_gen is not None:
        all_cids_with_field = [cid for cid, fb in cs.fractal.field_bits.items()
                               if cid != gen_cid and np.bitwise_and(fb, fb_gen).sum() > 0]
        if all_cids_with_field:
            ppmi_cid = random.choice(all_cids_with_field)
```

---

## 3. Проверка GPU-Opt Agent

### 3.1: _vecs_t stale — ключевая проблема

| Пункт | Статус |
|-------|--------|
| V1: Hook после _apply_vector_update | ✅ Реализовано |
| V1: Частичное обновление вместо полного | ✅ Реализовано |

**Но**: реализация использует `.to(device)`, что аллоцирует новый тензор при каждом обновлении (~1600 раз/мин).

**Решение**: заменить на `.copy_()`:
```python
def _on_vector_update(self, cid, v_new):
    if self._vecs_t is not None:
        v_t = torch.from_numpy(v_new.astype(np.float32))
        self._vecs_t[cid].copy_(v_t)
```
Это in-place операция без аллокации.

### 3.2: _fb_t stale

| Пункт | Статус |
|-------|--------|
| V1: _fb_dirty flag | ❌ Не реализовано |

**Вывод**: field_bits не меняются после загрузки, но флаг нужен на будущее.

**Решение**: `_fb_dirty` не срочно. Отложить до момента, когда field_bits станут динамическими.

### 3.3: CPU/GPU асимметрия

| Пункт | Статус |
|-------|--------|
| V1: Уже исправлено | ✅ Подтверждено |

### 3.4: non_blocking не используется

| Пункт | Статус |
|-------|--------|
| V1: pin_memory + non_blocking | ❌ Не реализовано |

**Вывод**: при каждом `_ensure_torch()` ~225 MB копируется на GPU синхронно (блокирующий transfer).

**Решение**: использовать для `_vecs_t`:
```python
vecs = np.zeros((V, D), dtype=np.float32)
# ... fill vecs ...
self._vecs_t = torch.from_numpy(vecs).pin_memory().to(device, non_blocking=True)
```

### 3.5: Нет gradient scaling

| Пункт | Статус |
|-------|--------|
| V1: max_grad_norm | ❌ Не реализовано |

**Вывод**: STDP-обновления без клиппинга могут приводить к взрывным сдвигам при больших `pmi_w * field_weight`.

**Решение**: добавить в `_gpu_stdp_apply`:
```python
grad_norm = torch.norm(grad)
if grad_norm > max_grad_norm:
    grad = grad / grad_norm * max_grad_norm
```
Параметр `max_grad_norm: float = 1.0` в FCFConfig.

---

## 4. Проверка Training-Dynamics Agent

### 4.1: Curriculum Learning

| Пункт | Статус |
|-------|--------|
| V1: Непрерывный curriculum | ✅ Реализовано |

**Нюанс**: `CURICULUM_MIN_LEN = 16` — строки короче 16 BPE токенов теряются. Многие биграммы короче.

**Решение**: уменьшить до 4-6, или вычислить автоматически: `CURICULUM_MIN_LEN = int(np.percentile(train_lens, 5))`.

### 4.2: ParameterOptimizer — нет saturation rule

| Пункт | Статус |
|-------|--------|
| V1: `full_stuck` правило | ❌ Не реализовано |

**Вывод**: optimizer может попасть в цикл: PPL plateaus → увеличить LR → дестабилизация → PPL растёт → уменьшить LR → PPL plateaus → ...

**Решение**: добавить детектор full stuck:
```python
if (self.m['mean_cos'].plateau(patience=5) and 
    self.m['vec_ppl'].plateau(patience=3) and 
    self.m['acc1'].plateau(patience=3)):
    # Force fluctuate + increase destab
    ctx['_force_fluct'] = True
```

### 4.3: _torch_dirty до contrastive

| Пункт | Статус |
|-------|--------|
| V1: Перенести в конец | ❌ Не реализовано |

**Код**: `crystal_generator.py:1146-1150`
```python
if use_torch:
    self._torch_dirty = True  # строка 1146
self._contrastive_objective(gen_updates)  # строка 1150
```

**Риск**: низкий (contrastive CPU-only), но если кто-то добавит GPU-контрастив, сломается.

**Решение**: поменять порядок:
```python
self._contrastive_objective(gen_updates)
if use_torch:
    self._torch_dirty = True
```

### 4.4: Centroid pull — per-text

| Пункт | Статус |
|-------|--------|
| V1: Batch centroid pull | ❌ Не реализовано |

**Вывод**: при batch=32 → 32 отдельных вызова `_centroid_pull`, каждый пересчитывает центроид.

**Решение**: накопить все CIDs за batch, вызвать один batch-centroid-pull.

### 4.5: decay_every не по парам

| Пункт | Статус |
|-------|--------|
| V1: decay_every_pairs | ❌ Не реализовано |

**Вывод**: decay по строкам не учитывает разную длину строк в curriculum.

**Решение**: добавить счётчик пар:
```python
total_pairs += len(ids) * context_window * 2
if total_pairs - last_decay_pairs > DECAY_EVERY_PAIRS:
    lattice.decay_all()
```

---

## 5. Проверка Quality-Safety Agent

### 5.1: Type hints

| Пункт | Статус |
|-------|--------|
| V1: GenerationResult | ✅ Реализовано |
| V1: _branch() → List | ❌ Не реализовано |
| V1: save() → None | ❌ Не реализовано |

**Решение**: добавить аннотации в `_branch()`, `_apply_vector_update()`, `save()`.

### 5.2: _quiet silencing

| Пункт | Статус |
|-------|--------|
| V1: Заменить на try/except + sys.exit | ✅ Частично реализовано |

**Код**: `train_full.py` — `_quiet` для `ConceptSpace.load()`, `lattice.load()` заменены на try/except с `sys.exit(1)`. `_quiet` остался для `save_3d_vis()`, `gen.evaluate()`, `Periodic save` — что корректно (эти ошибки не фатальны).

**Статус**: корректно. ✅

### 5.3: total_freq per-line

| Пункт | Статус |
|-------|--------|
| V1: Кэш + invalidation | ✅ Реализовано |

**Код**: `_get_total_freq()` + `_total_freq_cache`, сброс через hook на `lattice.update` и `lattice.decay_all`.

**Вывод**: решено элегантно. ✅

### 5.4: concept_error FIFO

| Пункт | Статус |
|-------|--------|
| V1: OrderedDict | ✅ Реализовано |

**Изменение**: порог 50000 → 30000.

**Риск**: при batch=32 и 150K строках ~4.8M пар, 30000 — 0.6%. Для редких CIDs (10 появлений) ошибка забывается через ~50 batch-ей.

**Решение**: вернуть порог 50000 или сделать конфигурируемым.

### 5.5: float→int32 concept_freq

| Пункт | Статус |
|-------|--------|
| V1: np.float32 | ❌ Не реализовано |

**Код**: `syntax_lattice.py:443` — `np.array([v for _, v in cf_items], dtype=np.int32)`

**Ошибка**: float-частоты (с десятичной частью после экспоненциального сглаживания) сохраняются как int32.

**Решение**: `dtype=np.float32`. Исправить немедленно (P0).

### 5.6: word_to_cid

| Пункт | Статус |
|-------|--------|
| V1: Удалён | ✅ Подтверждено |

### 5.7: Нет тестов

| Пункт | Статус |
|-------|--------|
| V1: tests/test_stdp.py | ❌ Не реализовано |

**Решение**: начать с теста GPU/CPU паритета — критично после внедрения hook `_on_vector_update`.

---

## 6. Проверка новых методов (из V1, раздел 6)

| Метод | Статус | Приоритет |
|-------|--------|-----------|
| 6.1: Adaptive beam width | ❌ Не реализовано | P3 |
| 6.2: Dynamic context windowing | ❌ Не реализовано | P3 |
| 6.3: Basis re-orthogonalization | ❌ Не реализовано | P2 |
| 6.4: Hormonal STDP gate | ❌ Не реализовано | P2 |
| 6.5: Multi-source RRF + field filter | ❌ Не реализовано | P2 |
| 6.6: PPMI-based hard negative mining V2 | ❌ Не реализовано | P1 |
| 6.7: Field-aware fluctuation | ❌ Не реализовано | P3 |

---

## 7. Итоговая статистика

| Категория | Всего | Решено | Не решено |
|-----------|-------|--------|-----------|
| Architect-AI (A-*) | 4 | 1.5 | 2.5 |
| Neuro-Symbolic (S-*) | 5 | 1 | 4 |
| GPU-Opt (G-*) | 5 | 2 | 3 |
| Training-Dynamics (T-*) | 5 | 1 | 4 |
| Quality-Safety (Q-*) | 7 | 4 | 3 |
| Новые методы | 7 | 0 | 7 |
| **Итого** | **33** | **9.5** | **23.5** |

**Процент выполнения**: ~29%

---

## 8. Ключевые выводы

1. **Самое ценное исправление**: `_vecs_t` stale устранён через `_after_update_hook`. Это была P1-проблема, влияющая на точность всех GPU-операций.

2. **Самая опасная нерешённая проблема**: `float→int32` в concept_freq (P0). Float-частоты с десятичной частью после EMA-затухания округляются до int при сохранении. После загрузки все частоты теряют дробную часть. Влияние — на PMI-gate, который использует concept_freq для вероятности P(next).

3. **Самая ресурсоёмкая проблема**: `_on_vector_update` с `.to()` вместо `.copy_()` — ~1600 аллокаций/мин. При обучении на 7 часов ~672K аллокаций GPU-тензоров.

4. **Наибольший потенциал улучшения обучения**: PPMI-based hard negative mining (S-2) — текущий random sampling из 146K находит hard negative с вероятностью ~0.003%.

5. **Архитектурный риск**: FCFConfig God Object продолжает расти. Установить порог рефакторинга при 500 строках.

6. **Пробел в покрытии**: 0 тестов. Критические алгоритмы (STDP, lateral inhibition, contrastive) не верифицируются. При внедрении hook-ов (как `_on_vector_update`) нужен тест GPU/CPU parity.

---

## 9. Обновлённый план с учётом сверки

| Приор. | Задача | Компонент | Время | Зависит от |
|--------|--------|-----------|-------|------------|
| **P0** | Исправить `np.int32` → `np.float32` в concept_freq | syntax_lattice.py:443 | 2 мин | — |
| **P0** | `_on_vector_update`: `.to()` → `.copy_()` | crystal_generator.py:121 | 5 мин | — |
| **P0** | `inference.py`: `result['text']` → `result.text` | inference.py | 10 мин | — |
| **P1** | Protocol `TorchInvalidatable` для fluctuate_fractal | concept_space.py | 20 мин | — |
| **P1** | PPMI-based hard negatives в contrastive objective | crystal_generator.py | 2 ч | — |
| **P1** | `field_weight_max` в FCFConfig | fcf_config.py + crystal_generator.py | 5 мин | — |
| **P1** | `max_concept_error_size` конфигурируемый | fcf_config.py + crystal_generator.py | 5 мин | — |
| **P2** | Continuous curriculum: `CURICULUM_MIN_LEN` → 4 | train_full.py | 1 мин | — |
| **P2** | Batch centroid pull | crystal_generator.py | 1 ч | — |
| **P2** | `non_blocking=True` для GPU тензоров | crystal_generator.py | 20 мин | — |
| **P2** | Gradient max_norm | crystal_generator.py + fcf_config.py | 20 мин | — |
| **P2** | Basis re-orthogonalization на чекпоинтах | concept_space.py | 30 мин | — |
| **P3** | Hormonal STDP gate | crystal_generator.py | 3 ч | PPMI contrastive |
| **P3** | tests/test_stdp.py | tests/ | 4 ч | — |
| **P3** | Destab field_bits fallback | crystal_generator.py | 1 ч | — |
| **P3** | `_torch_dirty` перенести после contrastive | crystal_generator.py | 2 мин | — |

---

*Анализ выполнен: коллегия AI-агентов (Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent)*
*Дата: 2026-06-17*
