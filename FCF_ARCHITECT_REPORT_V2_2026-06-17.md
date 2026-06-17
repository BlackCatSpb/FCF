# Отчёт архитектурного анализа FCF — РАУНД 2 (2026-06-17)

## Состав агентов-анализаторов

| Агент | Специализация |
|-------|---------------|
| **Architect-AI** | Целостность архитектуры, интерфейсы, модульность |
| **Neuro-Symbolic Specialist** | STDP, фрактальные поля, латеральное торможение, гомеостаз |
| **GPU-Opt Agent** | Батчинг, torch тензоры, CUDA ядра, синхронизация |
| **Training-Dynamics Agent** | Curriculum, параметрическая адаптация, сходимость |
| **Quality-Safety Agent** | Type safety, edge cases, исключения, консистентность |

---

## Сводка изменений с момента предыдущего аудита

За 4 коммита закрыто **144 issue** (0 active). Ключевые изменения:

### Исправлено (P0-P3)
| ID | Описание | Статус |
|----|----------|--------|
| P2-NEW-1 | `_vecs_t` stale — внедрён hook `_after_update_hook` + `_on_vector_update` | ✅ |
| P2-NEW-2 | CPU lateral inhibition для ВСЕХ gen_cid (не только best) | ✅ |
| P3-NEW-1 | `import zlib` вынесен на уровень модуля | ✅ |
| P3-NEW-2 | Мёртвые `rng`, `pct` свойства удалены | ✅ |
| P3-NEW-3 | Guard `fb_bytes == 0` | ✅ |
| P3-NEW-4 | `eval_checkpoint.py` — прямой `cs.concept_vectors.values()` | ✅ |
| P3-NEW-5 | `ARCHITECTURE.md` — `semantic_gate.py` удалён | ✅ |
| P3-NEW-6 | `word_to_cid` удалён | ✅ |
| P3-NEW-7 | `try/except OSError` с `[WARN]` | ✅ |

### Архитектурные улучшения (новые)
1. **GenerationResult dataclass** — `generate()` возвращает типизированный объект вместо dict
2. **Hook `_after_update_hook`** — `_apply_vector_update` уведомляет CrystalGenerator об изменениях
3. **`_vecs_t` live sync** — `_on_vector_update(cid, v_new)` обновляет тензор без полной перестройки
4. **`concept_error` → OrderedDict** — FIFO очистка через `popitem(last=False)`
5. **Непрерывный curriculum** — замена дискретного `EPOCH_MAX_LEN` на линейную рампу `_curriculum_p(idx)` + `_curriculum_max_len(idx)`
6. **Ramping context_window/neg_samples/pmi_gate_min** — curriculum влияет не только на max_len, но и на гиперпараметры обучения
7. **`total_freq` cache** — пересчёт только при мутациях lattice, хук на `lattice.update` и `lattice.decay_all`
8. **Обработка исключений** — все `_quiet` для критических операций заменены на `try/except` с `sys.exit(1)`
9. **`batch_log` CSV header guard** — проверка `os.path.getsize == 0`
10. **`result.text` / `result.score`** — train_full.py использует поля dataclass вместо `result['text']`

---

## 1. Остаточные архитектурные проблемы (Architect-AI)

### A-1: Циркулярная зависимость ConceptSpace ↔ CrystalGenerator

**Статус**: частично решена через `_after_update_hook`. Однако `fluctuate_fractal` (concept_space.py:433) всё ещё принимает `generator` напрямую и вызывает `generator._invalidate_torch()`.

**Рекомендация**: заменить прямой параметр `generator` на протокол `Invalidatable`:
```python
class Invalidatable(Protocol):
    def _invalidate_torch(self) -> None: ...
```
Это разорвёт жёсткую связь без изменения логики.

### A-2: FCFConfig — God Object (440 строк)

**Статус**: не изменён с прошлого аудита. Все методы построения пар и сериализации остаются в `FCFConfig`.

**Рекомендация без изменения**: добавить `# TODO: extract PathConfig` для будущего рефакторинга.

### A-3: `cleanup_old_checkpoints` не удаляет `points_*.html`

**Новая проблема**: `_final_save()` удаляет `points_*.html`, но `cleanup_old_checkpoints()` не трогает визуализационные файлы. При каждой очистке остаются JSON-файлы визуализаций от удалённых чекпоинтов.

**Рекомендация**: добавить в `cleanup_old_checkpoints` удаление соответствующих `points_{k}k.json`:
```python
fp_vis = os.path.join(vis_dir, f'points_{k_label}k.json')
if os.path.exists(fp_vis): os.remove(fp_vis)
```

### A-4: `TeeOut` — потенциальная проблема с garbage collection

`TeeOut.__del__` вызывает `self.close()`. В Python нет гарантии, что `__del__` будет вызван (особенно при `sys.exit()`). Файловый дескриптор может не закрыться.

**Рекомендация**: использовать `atexit.register` вместо `__del__`:
```python
import atexit
atexit.register(lambda: sys.stdout.close() if hasattr(sys.stdout, 'close') else None)
```

---

## 2. Нейро-символическое ядро (Neuro-Symbolic Specialist)

### S-1: field_weight capped — но остаётся доминантным

**Исправлено**: `min(1.0 + math.log(overlap + 1) * 2.0, 3.0)` (crystal_generator.py:1069).

**Анализ**: при overlap=0 → 0.1, overlap=1 → 2.39, overlap=10 → 5.77 (было), теперь max 3.0. Доминирование снижено, но field_weight при overlap=1 всё ещё в ~12 раз больше freq_weight.

**Дополнительная рекомендация**: сделать cap конфигурируемым:
```python
field_weight_max: float = 3.0  # в FCFConfig
```

### S-2: Contrastive objective — CPU-only и неэффективен

**Статус**: не изменён. `_contrastive_objective()` использует random-выборку 80 кандидатов.

**Проблема**: с внедрением `_on_vector_update` hook-а контрастив теперь может использовать GPU. Предлагается GPU-версия:
```python
def _contrastive_objective_gpu(self, gen_updates):
    # Использовать _vecs_t для batch matmul
    # Выбрать негативы с помощью top-k по cosine
```

### S-3: Дестабилизация без fallback

**Статус**: не изменён. Если `lattice.connections_of()` пуст, `destab_scale` не применяется.

**Рекомендация**: fallback на field_bits (как в прошлом отчёте).

### S-4: `lateral_inhibition_fractal` — CPU sampling не учитывает batch

**Новое**: при GPU STDP с G < 50 вызывается CPU `_lateral_inhibition_fractal` (concept_space.py:533). Этот метод использует `_inhibit_rng` для sampling. При batch=32 с несколькими gen_cid (<50) вызов происходит для каждого gen_cid **последовательно** — 32 отдельных random sampling-a.

**Оптимизация**: агрегировать все gen_cid в один batch до вызова торможения:
```python
if len(gen_cids_list) < 50:
    # Batch-торможение: один sampling на все цели
    cs._lateral_inhibition_batch(gen_cids_list, str_val_list, ...)
```

---

## 3. GPU-оптимизация (GPU-Opt Agent)

### G-1: `_vecs_t` live sync — реализовано, но есть нюанс

**Исправлено**: `_on_vector_update` в CrystalGenerator обновляет `_vecs_t[cid]` после каждого `_apply_vector_update`.

**Проблема**: `torch.from_numpy(v_new.astype(np.float32)).to(self._vecs_t.device)` создаёт новый тензор при каждом вызове. При batch=32, каждая строка с ~50 токенами, это ~32 * 50 = 1600 обновлений в минуту. Каждое — аллокация + transfer.

**Оптимизация**: заменить на in-place copy:
```python
def _on_vector_update(self, cid, v_new):
    if self._vecs_t is not None:
        self._vecs_t[cid].copy_(torch.from_numpy(v_new.astype(np.float32)))
```
`copy_` не создаёт новый тензор, а копирует данные в существующий срез.

### G-2: `_vecs_t` полнота после `_sync_from_fractal`

**Проблема**: `fluctuate_fractal` → `_sync_from_fractal` → `concept_vectors` обновлены, но `_invalidate_torch` делает полный rebuild (`_torch_dirty = True`). С hook-ом `_on_vector_update` можно перестроить `_vecs_t` инкрементально.

**Рекомендация**: после `_sync_from_fractal` не делать `_torch_dirty = True`, а обновить `_vecs_t` через hook:
```python
def _sync_from_fractal(self):
    for cid in list(self.fractal.codes.keys()):
        v = self.fractal.compute_vector(cid)
        if v is not None:
            self.concept_vectors[cid] = v
            if self._after_update_hook:
                self._after_update_hook(cid, v)
```
Но это требует, чтобы `_after_update_hook` был установлен ДО вызова `fluctuate_fractal` — что и так происходит (устанавливается в `CrystalGenerator.__init__`).

### G-3: `non_blocking=True` всё ещё не используется для _vecs_t

**Статус**: не изменён.

**Рекомендация**: добавить `pin_memory` для `vecs` перед `torch.from_numpy(vecs).to(device, non_blocking=True)`.

---

## 4. Динамика обучения (Training-Dynamics Agent)

### T-1: Непрерывный curriculum — реализован, но есть перекос

**Исправлено**: `_curriculum_p(idx)` + `_curriculum_max_len(idx)` вместо `EPOCH_MAX_LEN`.

**Анализ**: `CURICULUM_FRACTION = 0.20` — после 20% строк max_len = 10^9 (unlimited). Это значит, что 80% эпохи модель учится на строках любой длины. Для epoch=1, ~124K строк, curriculum длится ~25K строк.

**Проблема**: `CURICULUM_MIN_LEN = 16` — строки короче 16 BPE токенов пропускаются в начале. Но многие короткие строки содержат важные биграммы (например, «да нет»). Рекомендуется `CURICULUM_MIN_LEN = 4`.

### T-2: Ramping context_window/neg_samples — новая функциональность

**Статус**: реализовано в этом раунде (crystal_generator.py:535-546). `cw_ramp`, `ns_ramp`, `pg_ramp` масштабируются с `_curriculum_p(idx)`.

**Проблема**: `cw_ramp = max(1, int(round(cw_target * cp)))` — при `cp=0.01` и `cw_target=2`, `cw_ramp = max(1, 0) = 1`. Округляется вниз. При cp=0.3, `cw_ramp = 1` (2*0.3=0.6→1). Первые 30% curriculum context_window = 1.

**Рекомендация**: использовать `math.ceil` для консервативного округления:
```python
cw_ramp = max(1, int(math.ceil(cw_target * cp)))
```
Или задать `CURICULUM_MIN_WINDOW = 1`.

### T-3: ParameterOptimizer — не адаптируется к новому curriculum

**Проблема**: `opt.p['context_window'].current` читается как `cw_target`, но на него накладывается `cp` (curriculum factor). Optimizer может поднять context_window до 6, но эффективное значение будет `max(1, round(6 * cp))` = 5 при cp=0.8. Optimizer не знает о curriculum.

**Рекомендация**: curriculum должен быть прозрачен для optimizer — отдавать в `opt.step()` фактически использованные значения.

### T-4: `destab_scale` — formula fix

**Исправлено**: `destab_from = max(...)`, `destab_to = min(...)`. Теперь корректно: при destab_scale_start=0.6, destab_scale_end=0.02, pct=0 → 0.6, pct=1 → 0.02. ✅

---

## 5. Качество кода и безопасность (Quality-Safety Agent)

### Q-1: GenerationResult — введён, но consumers используют `result.text`

**Исправлено**: `train_full.py` строки 660, 716 — используют `result.text`, `result.score`.

**Но**: `inference.py` всё ещё использует `result['text']` в нескольких местах:
- `inference.py:266`: `r['text']`
- `inference.py:184`: `r['text']`
- `inference.py:240`: `r['text']`

**Рекомендация**: унифицировать все потребители на `result.text`.

### Q-2: `eval_checkpoint.py` не обновлён под GenerationResult

**Проверка**: `eval_checkpoint.py` — использует `result['text']`. Нужно обновить для консистентности.

### Q-3: `inference.py:run_eval()` — словарь вместо GenerationResult

`run_eval()` сохраняет результаты генерации как dict в JSON:
```python
gen_samples.append({'seed': seed, 'text': r['text'].replace('\n',' ').strip(),
                    'words': r['word_count'], 'score': r['score']})
```
При использовании GenerationResult: `r.text`, `r.word_count`, `r.score`.

### Q-4: `concept_error` OrderedDict — FIFO периметр изменён

**Исправлено**: `concept_error` теперь `OrderedDict`, очистка через `popitem(last=False)`.

**Старая строка**: `if len(self.concept_error) > 50000: cids_to_remove = list(...)[:-30000]`
**Новая строка**: `while len(self.concept_error) > 30000: self.concept_error.popitem(last=False)`

Порог изменён с 50000 → 30000. Это более агрессивная очистка. При batch=32, обучение на 150K строках даёт ~4.8M пар. 30000 — ~0.6% от всех пар. Для редких CIDs (10 появлений) ошибка будет забыта после ~50 batch-ей.

**Рекомендация**: увеличить порог до 50000 или сделать конфигурируемым:
```python
max_concept_error_size: int = 50000  # в FCFConfig
```

### Q-5: `eval_metrics.py` — `int(0)` guard в `math.log`

**Проверено**: `eval_metrics.py` — `math.log` с `max(val, 1e-10)` guard уже везде есть. ✅

### Q-6: `filter_corpus.py` — `import re` на уровне модуля

**Проверено**: `filter_corpus.py` — строка 12: `import re` на уровне модуля. ✅

---

## 6. Предложения по новым методам (Round 2)

### 6.1 Метод: Asymmetric STDP с гормональной модуляцией (Hormonal STDP)

Обновлённая версия с учётом наличия `_after_update_hook`:

```python
def _build_pairs_from_ids(self, ...):
    ...
    # Гормональная модуляция STDP
    ach = self.hormones.acetylcholine  # пластичность
    da = self.hormones.dopamine        # награда
    lr *= (0.5 + ach * 0.5) * (0.5 + da * 0.5)
```

**Интеграция**: Hook `_after_update_hook` идеально подходит для логирования эффектов гормонов на обновления векторов.

### 6.2 Метод: Batch lateral inhibition (CPU)

Вместо N вызовов `_lateral_inhibition_fractal` для каждого gen_cid — один batch-вызов:

```python
def _lateral_inhibition_batch(self, gen_cids, str_vals, threshold):
    # Один sampling на ВСЕ gen_cid
    # Матрица sims: (n_gen × sample_size)
    # Векторизованный push-pull
```

### 6.3 Метод: Curriculum-aware optimizer states

Optimizer должен знать, что `context_window` и `neg_samples` дополнительно масштабируются curriculum:
```python
# Вместо прямого cw_target:
cw_actual = max(1, int(round(cw_target * cp)))
opt.ingest(context_window_actual=cw_actual)
```
И добавить правила, которые учитывают `context_window_actual`, а не target.

### 6.4 Метод: Gradient noise scheduling (для destab)

Вместо стохастической дестабилизации через PPMI (20% вероятности), использовать scheduled gradient noise:
```python
noise_std = destab_scale * (1.0 + ach_phasic)  # больше шума при высоком ACh
grad += np.random.randn(*grad.shape) * noise_std
```

---

## 7. Обновлённый план приоритетов

| Приоритет | Задача | Компонент | Сложность | Статус |
|-----------|--------|-----------|-----------|--------|
| **P0** | `_on_vector_update` — заменить `.to()` на `.copy_()` | crystal_generator.py | 10 мин | 🔴 NEW |
| **P0** | Исправить `inference.py` — `result['text']` → `result.text` | inference.py | 15 мин | 🔴 NEW |
| **P0** | Уменьшить `CURICULUM_MIN_LEN` с 16 до 4 | train_full.py | 1 мин | 🔴 NEW |
| **P1** | `cleanup_old_checkpoints` — удалять `points_*k.json` | train_full.py | 10 мин | 🟡 NEW |
| **P1** | Curriculum-aware optimizer | parameter_optimizer.py | 1 час | 🟡 NEW |
| **P1** | configurable `field_weight_max` | fcf_config.py | 5 мин | 🟡 NEW |
| **P1** | Перевести `eval_checkpoint.py` на `result.text` | eval_checkpoint.py | 5 мин | 🟡 NEW |
| **P2** | GPU contrastive objective | crystal_generator.py | 2 часа | 🟡 NEW |
| **P2** | Batch CPU lateral inhibition | concept_space.py | 1 час | 🟡 NEW |
| **P2** | `non_blocking=True` для `_vecs_t` | crystal_generator.py | 30 мин | 🟡 NEW |
| **P2** | `max_concept_error_size` в FCFConfig | crystal_generator.py, fcf_config.py | 10 мин | 🟡 NEW |
| **P2** | Destab fallback на field_bits | crystal_generator.py | 1 час | 🟡 |
| **P2** | Invalidatable protocol | concept_space.py | 30 мин | 🟡 |
| **P3** | Hormonal STDP gate | crystal_generator.py | 3 часа | 🔵 |
| **P3** | Gradient noise scheduling | crystal_generator.py | 2 часа | 🔵 |
| **P3** | `atexit` вместо `__del__` в TeeOut | train_full.py | 10 мин | 🔵 NEW |
| **P3** | Юнит-тесты STDP (GPU/CPU parity) | tests/ | 4 часа | 🔵 |

---

## 8. Заключение

После второго раунда аудита (4 коммита, 144 закрытых issue) кодовая база FCF значительно улучшилась:

**Решённые архитектурные проблемы:**
- ✅ `_vecs_t` stale: внедрён hook `_after_update_hook` + `_on_vector_update`
- ✅ CPU/GPU асимметрия торможения: исправлена
- ✅ `GenerationResult` dataclass вместо dict
- ✅ Непрерывный curriculum
- ✅ Cached `total_freq`
- ✅ `concept_error` → OrderedDict для O(1) FIFO
- ✅ `import zlib` вынесен

**Остающиеся проблемы P0-P1:**
- `_on_vector_update` избыточно аллоцирует тензоры (нужен `.copy_()`)
- `inference.py` не обновлён под `result.text`
- `CURICULUM_MIN_LEN = 16` слишком высок
- Очистка визуализаций при чекпоинтах

**Общее состояние**: 144 issues закрыто, ~15 новых выявлено. Проект движется к стабильности, но требует завершения миграции на `GenerationResult` во всех потребителях и оптимизации GPU-хука.

---

*Сформировано коллегией AI-агентов: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent*
*Дата: 2026-06-17 (раунд 2)*
