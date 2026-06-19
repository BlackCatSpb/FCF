# Отчёт архитектурного анализа FCF — 2026-06-17

## Состав агентов-анализаторов

| Агент | Роль | Специализация |
|-------|------|---------------|
| **Architect-AI** | Главный архитектор | Целостность архитектуры, интерфейсы, модульность, масштабирование |
| **Neuro-Symbolic Specialist** | Нейро-символический инженер | STDP, фрактальные поля, латеральное торможение, гомеостаз |
| **GPU-Opt Agent** | Оптимизатор GPU | Батчинг, torch тензоры, CUDA ядра, scatter_add, синхронизация |
| **Training-Dynamics Agent** | Специалист по динамике обучения | Curriculum, параметрическая адаптация, сходимость, дестабилизация |
| **Quality-Safety Agent** | Инженер качества кода | Type safety, edge cases, исключения, консистентность данных |

---

## 1. Общая архитектура (Оценка Architect-AI)

### 1.1 Сильные стороны

- **Модульная структура**: проект чётко разделён на `eva/symbolic/` (ядро), `model/` (HF-совместимость), `api/` (сервинг).
- **Единый источник конфигурации**: `FCFConfig` покрывает 100% гиперпараметров с валидацией коридоров.
- **Сохранение/загрузка**: гибридный формат (JSON + NPZ) с атомарной заменой через `.tmp`.
- **Разделение чтения и записи**: `inference.py` копирует чекпоинты во временную директорию для избежания блокировок.
- **Двойной путь CPU/GPU**: бесшовное переключение при отсутствии torch.

### 1.2 Архитектурные проблемы

#### A-1: Циркулярная зависимость между ConceptSpace и CrystalGenerator

`CrystalGenerator` импортирует и использует `ConceptSpace`, но методы `fluctuate_fractal()` в `ConceptSpace` принимают `CrystalGenerator` как параметр и вызывают `_invalidate_torch()`. Это создаёт неявную циклическую связь:

```python
# concept_space.py:432
def fluctuate_fractal(self, ..., generator=None):
    ...
    if generator is not None:
        generator._invalidate_torch()
```

**Риск**: при добавлении новых зависимостей между этими классами сложность будет расти квадратично.

**Предложение**: ввести интерфейс `TorchCacheOwner` (протокол) с методом `invalidate_torch()`, либо вынести управление торч-кэшем в отдельный класс `TorchCacheManager`.

#### A-2: FCFConfig — God Object с 440+ строками

`FCFConfig` (fcf_config.py) содержит:
- Параметры путей (свойства)
- Определения `ParamDef` и `AdaptRule`
- Методы построения метрических пар (`build_antonym_pairs`, `build_morph_pairs`, etc.)
- Сериализацию (`to_dict`, `save`, `load`)
- `__post_init__` с конвертацией типов

**Проблема**: нарушение Single Responsibility Principle. Конфиг знает слишком много о предметной области.

**Предложение**:
- Выделить `PathConfig` (свойства путей)
- Выделить `MetricPairBuilder` (статический класс для построения пар)
- Оставить в `FCFConfig` только гиперпараметры

#### A-3: Нет абстракции для генерации

`CrystalGenerator.generate()` возвращает словарь `dict` с текстом, путём, счётом. Это не типизировано. Если ключи словаря изменятся, сломаются все потребители.

**Предложение**: создать dataclass `GenerationResult`:
```python
@dataclass
class GenerationResult:
    text: str
    concept_path: List[int]
    score: float
    word_count: int
    max_words: int
    semantic_delta: float
    chains: List[Tuple[List[int], float]]
    time: Optional[float] = None
```

#### A-4: Нет мониторинга ресурсов

Проект использует GPU MX550 с 2GB VRAM. При `V=146K, D=384`:
- `_vecs_t` = 146000 × 384 × 4B = ~224 MB
- `_fb_t` = 146000 × 256 × 1B = ~37 MB
- Плюс временные тензоры при STDP

Нет ни одного измерения потребления VRAM, OOM-ловушки, graceful degradation на CPU.

**Предложение**: добавить `torch.cuda.max_memory_allocated()` логирование в `_ensure_torch` и fallback на CPU при `CUDA out of memory`.

---

## 2. Нейро-символическое ядро (Neuro-Symbolic Specialist)

### 2.1 Анализ STDP

#### S-1: Несбалансированность theta-gate и field_gate

В `_gpu_stdp_apply()` (crystal_generator.py:616-618):
```python
lr = torch.clamp(fw_t, min=0.05) * dw_t * pmi_w_t * field_w_t
theta = torch.exp(-torch.clamp(dist, max=5.0) / max(self.theta_tau, 1.0))
effective_lr = lr * torch.clamp(theta, min=0.1)
```

`fw_t` (freq_weight) зажат в `[0.05, 1.0]`, `dw_t` (dist_weight) — `exp(-dist/2)` — примерно `[0.37, 1.0]`, `pmi_w_t` — `[min_weight, 2.0]`, `field_w_t` — `[0.1, inf)` (1 + log(overlap+1)*2).

**Проблема**: `field_w_t` может доминировать над всеми остальными компонентами при большом перекрытии field_bits. При overlap > 10, field_weight ≈ 1 + log(11)*2 ≈ 5.8. Это в 116 раз больше freq_weight.

**Последствие**: STDP перекашивает пространство в сторону концептов с большими field_bits (частотные слова), ослабляя обучение редких, но информативных пар.

**Предложение**: ограничить `field_weight` сверху, например `min(1+log(overlap+1)*2, 3.0)` в `_build_pairs_from_ids`.

#### S-2: Contrastive Objective — неэффективный отбор кандидатов

`_contrastive_objective()` (crystal_generator.py:938-977) использует:
```python
candidates = cs.rng.randint(0, cs.vocab_size, size=n_candidates)
```
Для каждого gen_cid — случайная выборка 80 кандидатов из 146K. Вероятность найти hard negative (cos > 0.05) крайне мала при разреженном пространстве.

**Предложение**: вместо uniform random использовать PPMI-ранжированные негативы:
- Взять top-100 концептов по cosine similarity к gen_cid
- Отфильтровать те, что имеют PMI-связь > порога
- Выбрать те, у которых cos > 0.05 (hard negatives)

Это сократит количество «пустых» итераций и даст более эффективный push-pull.

#### S-3: Отсутствие негативного sample-баланса

`neg_samples` — int, одинаков для всех пар. При `neg_samples=2` для пары с context_window=2 (до 5 пар на токен) получается 10 негативных сэмплов на токен. Нет адаптации сложности негативов (hard negative mining).

**Предложение**: реализовать `adaptive_neg_sampling`:
- Использовать кэш `concept_error` для определения пар с высокой ошибкой
- Для таких пар увеличивать `neg_samples` динамически
- Добавить `curriculum_negatives=True` в конфиг

#### S-4: Field_overlap — потенциальный bottleneck

`cs.fractal.field_overlap(cid_a, cid_b)` (concept_space.py:168-174) вызывается в цикле для каждой пары в `_build_pairs_from_ids()`:
```python
overlap = cs.fractal.field_overlap(ids[i], ids[j])
```
Для каждой строки с ~50 токенами и context_window=2 это ~250 вызовов. Каждый вызов — `np.unpackbits(np.bitwise_and(ba, bb)).sum()`.

**Оптимизация**: 
1. Кэшировать unpackbits результаты для часто используемых CIDs (LRU cache на 10000 записей)
2. Использовать предвычисленную матрицу field_overlap на GPU через `torch.bitwise_and` + `sum(dim=1)` в `_fb_t`

#### S-5: Дестабилизация (PPMI noise) не использует field_bits

`destab_scale` в GPU/CPU STDP добавляет шум из PPMI-соединений (crystal_generator.py:664-677). Если `lattice.connections_of()` пуст, дестабилизация не срабатывает совсем. Для редких токенов PPMI-граф пуст — они никогда не дестабилизируются.

**Предложение**: добавить fallback — при пустом PPMI использовать `field_bits` для выбора случайного концепта с пересекающимся полем.

---

## 3. GPU-оптимизация (GPU-Opt Agent)

### 3.1 Проблема: _vecs_t stale после STDP

`_gpu_stdp_apply` обновляет векторы через `cs._apply_vector_update()`, который модифицирует `cs.concept_vectors`, но не обновляет `self._vecs_t`. После GPU-секции:

```python
self._vecs_t[gen_cids_t] = gv_t  # строка 705 — обновляются только gen_cids
```

Остальные 146K векторов в `_vecs_t` остаются старыми до следующего `_ensure_torch()`.

**Влияние**: 
- `evaluate()` использует `_vecs_t` → получает неточные similarity
- `_negative_sampling_gpu()` использует `_vecs_t` → негативы выбираются на основе неточных векторов
- После `fluctuate_fractal` → `_torch_dirty = True` → полный пересчёт всех тензоров (дорого: ~225 MB копирование + нормализация)

**Предложение**: внедрить частичное обновление `_vecs_t` после каждого `_apply_vector_update`:
```python
def _apply_vector_update(self, cid, v_new):
    ...
    if self._torch_device is not None and self._vecs_t is not None:
        self._vecs_t[cid] = torch.from_numpy(v_new).to(self._torch_device)
```
Это избавит от полного пересчёта и даст актуальные векторы для всех GPU-операций.

### 3.2 Проблема: поле _fb_t не обновляется после STDP

`field_bits` в `FractalField` не меняются в процессе обучения — они фиксированы после `build_octree_fields()`. Однако если когда-либо понадобится динамическое обновление field_bits (например, перестройка октодерева), `_fb_t` останется старым.

**Решение**: добавить `_fb_dirty` флаг и перестраивать `_fb_t` при `_ensure_torch()` если `_fb_dirty == True`.

### 3.3 Проблема: CPU/GPU асимметрия в lateral inhibition

**Текущее состояние** (AUDIT.md P2-NEW-2): исправлено для CPU — теперь торможение для всех gen_cid с достаточным elr.

Однако остаётся асимметрия:
- **CPU**: `concept_vectors.data[sampled_cids]` — свежие векторы
- **GPU**: `self._vecs_t` — потенциально устаревшие

**Рекомендация**: после исправления 3.1 (_apply_vector_update синхронизирует _vecs_t) эта асимметрия исчезнет.

### 3.4 Проблема: `non_blocking=True` не используется

В `evaluate()` (crystal_generator.py:1311):
```python
pv_t = torch.from_numpy(prev_vecs).to(device, non_blocking=True)
```
Здесь `non_blocking=True` — хорошо. Но в `_gpu_stdp_apply` все тензоры создаются через `torch.tensor(...)` и `torch.from_numpy(...).to(device)` без `non_blocking`.

**Предложение**: где возможно, использовать `pin_memory=True` для CPU-тензоров и `non_blocking=True` для transfer-to-GPU. Особенно для `_vecs_t`, который пересылается при каждом `_ensure_torch`.

### 3.5 Отсутствие gradient scaling

Все обновления используют прямой `grad * base_lr_val`. Нет ни градиентного клиппинга, ни адаптивного шага.

**Предложение**: добавить `max_grad_norm` параметр в конфиг и применять:
```python
grad_norm = np.linalg.norm(grad)
if grad_norm > max_grad_norm:
    grad = grad / grad_norm * max_grad_norm
```

---

## 4. Динамика обучения (Training-Dynamics Agent)

### 4.1 Curriculum Learning — недостаточная гибкость

Текущая схема (EPOCH_MAX_LEN = {1: 32, 2: 128, 3: 10**9}) слишком груба:
- В эпохе 1 теряются все длинные предложения (~70% корпуса)
- Переход от 32 до 128 токенов резкий — модели приходится учиться работать с длинным контекстом сразу

**Предложение**: использовать непрерывный curriculum:
```python
max_len = min(32 + (epoch - 1) * 64 + (line_idx % epoch_lines) * 0.01, 10**9)
```
Или ступенчатый: эпоха 1.0 → 32, 1.5 → 64, 2.0 → 128, 2.5 → 256, 3.0 → ∞.

### 4.2 ParameterOptimizer — правила не покрывают насыщение

В `parameter_optimizer.py` есть rule-based адаптация. Но нет правила для случая, когда все метрики стабильны (плато по PPL, cos, acc1). Текущие `plateau()` детекторы работают, но их реакция — увеличить LR, что может привести к дестабилизации.

**Предложение**: добавить правило `full_stuck` — если все метрики в плато > 5 чекпоинтов, то:
- Увеличить `destab_scale` временно
- Уменьшить `pmi_strength` для увеличения стохастичности
- Форсировать `fluctuate_fractal`

### 4.3 Ошибка: `_contrastive_objective` вызывается после GPU _torch_dirty

В `train_from_text()` (crystal_generator.py:1110-1114):
```python
if use_torch:
    self._torch_dirty = True  # строка 1111
self._contrastive_objective(gen_updates)  # строка 1114
```

`_contrastive_objective` использует CPU-путь (numpy), который читает из `cs.concept_vectors` (свежие). Но если позже будет GPU-контрастив, `_torch_dirty = True` помешает.

**Проблемы нет сейчас**, но если кто-то добавит GPU-контрастив до этой строки, сломается.

**Рекомендация**: перенести `_torch_dirty` в конец метода, после всех операций.

### 4.4 Sentiment Centroid Pull — неэффективен при большом batch

`_centroid_pull` вызывается для каждой строки отдельно (per-text). При batch=32 это 32 вызова. Каждый пересчитывает центроид предложения и перебирает все токены.

**Предложение**:
- Накопить обновления центроида за весь batch
- Применить one-shot centroid pull для всех уникальных CIDs в batch

### 4.5 decay_every — не синхронизирован с curriculum

`decay_every` настроен на 2000/3000 строк. Но в эпохе 1 (строки длиной до 32 токенов) 2000 строк ≈ 2000 * 16 пар = 32K пар. В эпохе 3 2000 строк ≈ 2000 * 400 = 800K пар. Decay частоты концептов должен зависеть от количества пар, не строк.

**Предложение**: заменить `decay_every` на `decay_every_pairs`:
```python
if total_pairs_since_last_decay > DECAY_EVERY_PAIRS:
    lattice.decay_all()
```

---

## 5. Качество кода и безопасность (Quality-Safety Agent)

### 5.1 Type hints — неполное покрытие

Критические методы без аннотаций:
- `CrystalGenerator.generate()` → `dict` (строка 180)
- `CrystalGenerator._branch()` → `list` (строка 390)
- `ConceptSpace.save()` → `None` (строка 696)
- `ConceptSpace._apply_vector_update()` → `None` (строка 487)

**Предложение**: добавить `-> Dict[str, Any]`, `-> List[Tuple[int, float]]` и т.д., а лучше использовать dataclass-ы для возвращаемых типов (см. A-3).

### 5.2 Исключения: silence в нескольких местах

```python
# train_full.py:17-22
def _quiet(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"[WARN] {func.__name__} failed: {e}")
        return None
```

Этот декоратор используется для:
- `ConceptSpace.load()` — фатальная ошибка (строка 186)
- `lattice.load()` — фатальная ошибка (строка 192)
- `gen.evaluate()` — диагностическая (строка 653)
- `save_3d_vis()` — диагностическая (строка 649)

**Проблема**: для load() фатальная ошибка НЕ должна быть silenced. `_quiet` перехватывает `Exception` и возвращает `None`, а код проверяет `if cs is None: sys.exit(1)`. Если исключение не `Exception`, а `SystemExit` или `KeyboardInterrupt`, оно будет перехвачено и проглочено.

**Исправление**: использовать `except Exception` (не bare), и добавить перевыброс для `KeyboardInterrupt`, `SystemExit`.

### 5.3 `total_freq` пересчитывается для каждой строки

```python
# crystal_generator.py:1077
total_freq = max(sum(self.lattice.concept_freq.values()), 1)
```
Вызывается для каждой строки в `train_from_text()` и `train_batch()`. `sum()` по 146K элементов каждую итерацию — это дорого.

**Оптимизация**: кэшировать `total_freq` и пересчитывать только при `decay_all()` (когда частоты меняются) или каждые N строк.

### 5.4 `concept_error` FIFO-очистка — неэффективна

```python
# crystal_generator.py:1123-1126
if len(self.concept_error) > 50000:
    cids_to_remove = list(self.concept_error.keys())[:-30000]
    for c in cids_to_remove:
        del self.concept_error[c]
```

`list(dict.keys())[:-30000]` создаёт копию 50K ключей каждые ~500 строк. При 146K токенах.

**Оптимизация**: использовать `collections.OrderedDict` и pop first 20K.

### 5.5 Смешение float и int в concept_freq

`SyntaxLattice.build()` использует `self.concept_freq[c] = self.concept_freq.get(c, 0) * self.decay + 1.0` — float.
`SyntaxLattice.update()` использует `self.concept_freq[next_cid] = self.concept_freq.get(next_cid, 0) * self.decay + 1.0` — float.

`syntax_lattice.py:443`: `npz_data['cf_counts'] = np.array([v for _, v in cf_items], dtype=np.int32)`

**Ошибка**: сохранение float-частот как int32 приводит к потере точности (дробная часть отбрасывается).

**Исправление**: `np.array(..., dtype=np.float32)`.

### 5.6 `_SPTokenizer.word_to_cid` — мёртвый код

AUDIT.md P3-NEW-6 уже отметил удаление `word_to_cid`. Проверка показала — в `model/modeling_fcf.py` строка 45-46 метод присутствовал и был удалён. ✅

### 5.7 Отсутствует интеграционное тестирование

Ни один модуль не имеет юнит-тестов (кроме `if __name__ == '__main__'` в нескольких файлах). Нет `tests/` директории. Критические алгоритмы (STDP, lateral inhibition, contrastive objective) не верифицируются независимо.

**Предложение**: начать с `tests/test_stdp.py`:
- Golden-тест: зафиксировать `cs` с 10 концептами, выполнить STDP на известном тексте
- Проверить, что векторы сдвинулись ожидаемым образом
- Тест на GPU/CPU идентичность результатов

---

## 6. Предложения по новым методам

### 6.1 Метод: Адаптивный контекстный beam width (Trust Region Beam Search)

**Проблема**: `beam_width` фиксирован (default 5). При низкой уверенности (высокая энтропия распределения) нужно больше лучей; при высокой — меньше.

**Решение**:
```python
def _adaptive_beam_width(self, probs: np.ndarray, base_width: int) -> int:
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(probs))
    ratio = entropy / max_entropy  # 0..1
    return max(1, int(base_width * (0.5 + ratio)))
```

### 6.2 Метод: Dynamic Context Windowing

**Проблема**: `context_window` адаптируется параметрически, но одинаков для всех позиций в предложении. Для глаголов нужен широкий контекст (субъект-объект), для предлогов — узкий.

**Решение**: использовать `concept_error` для динамического расширения окна:
- Высокая prediction error для gen_cid → расширить окно для этой позиции
- Низкая error → сузить окно (экономия compute)

### 6.3 Метод: Fractal Basis Re-orthogonalization Scheduler

**Проблема**: при `load()` (concept_space.py:262-280) базис ре-ортогонализуется, если `err > 1e-3`. Но в процессе обучения базис никогда не проверяется.

**Решение**:
```python
def check_basis_health(self):
    QtQ = self.fractal.basis.T @ self.fractal.basis
    err = np.max(np.abs(QtQ - np.eye(self.dim)))
    if err > 1e-3:
        self._reorthogonalize_basis()
        self._invalidate_torch()
```
Вызывать на каждом чекпоинте.

### 6.4 Метод: Hormonal STDP Gate

**Проблема**: гормональная система (`HormonalSystem`) влияет только на генерацию (temperature, beam_width). Не влияет на обучение.

**Решение**: интегрировать гормоны в STDP:
```python
# В _build_pairs_from_ids:
ach_gate = self.hormones.acetylcholine  # plasticity gate (0..1)
da_gate = self.hormones.dopamine        # reward modulation
lr *= (0.5 + ach_gate * 0.5) * (0.5 + da_gate * 0.5)
```
Это свяжет intrinsic motivation с пластичностью: высокая ACh (новизна) → больше LR, высокая DA (успех) → больше LR.

### 6.5 Метод: Multi-Source RRF с field-фильтрацией

**Проблема**: `_branch()` комбинирует graph, syntax, vector similarity через RRF. Field bits используются как бонус, но не как фильтр.

**Решение**: использовать field_bits как pre-filter для категорического исключения кандидатов, чьё field overlap с контекстом = 0:
```python
if ctx_field is not None:
    combined = {cid: score for cid, score in combined.items()
                if cs.fractal.field_overlap_with(ctx_field, cid) > 0}
```
Это резко сократит пространство поиска для _branch().

### 6.6 Метод: PPMI-Based Hard Negative Mining (V2)

Заменить текущий `_contrastive_objective` на версию, использующую PPMI логарифмическое ранжирование:
```python
def _contrastive_objective_v2(self, gen_cid, v_gen, lattice):
    # Top-50 по PPMI (сильные коллокации — плохие негативы, исключаем)
    strong_conns = set(c for c, _ in lattice.connections_of(gen_cid, top_k=50, use_ppmi=True))
    # Bottom-50 по PPMI (отрицательная PMI = взаимоисключающие)
    all_cids = set(cs.concept_vectors.keys())
    weak_conns = all_cids - strong_conns - {gen_cid}
    candidates = random.sample(weak_conns, min(200, len(weak_conns)))
    # Сортируем по cosine, берём top-5 hardest
    hard = topk_by_cos(candidates, v_gen, k=5)  # cosine > 0.05
    for neg_cid in hard:
        v_neg = cs.get_vec(neg_cid)
        push = (cos * v_neg - v_gen) * contr_lr
        cs._apply_vector_update(gen_cid, v_gen + push)
```

### 6.7 Метод: Field-Aware Fluctuation

Текущая `fluctuate_fractal` (concept_space.py:432) дрейфует коды одинаково для всех концептов. Предлагается field-aware fluctuation:

```python
def fluctuate_field_aware(self, field_bits_target, strength=0.003):
    for cid in self.codes:
        overlap = self.field_overlap(cid, field_bits_target)
        noise_scale = strength * (1 + overlap / 32)  # больше шума для пересекающихся
        self.codes[cid] += noise * noise_scale
```
Это позволит направленно дестабилизировать только те концепты, которые пересекаются с целевой областью.

---

## 7. План рекомендуемых изменений

| Приоритет | Задача | Компонент | Сложность | Ожидаемый эффект |
|-----------|--------|-----------|-----------|------------------|
| **P0** | Исправить сохранение float concept_freq как int32 | syntax_lattice.py | 5 мин | Устранение потери точности |
| **P0** | Добавить `non_blocking=True` + pin_memory в GPU тензоры | crystal_generator.py | 30 мин | Ускорение GPU ~5-10% |
| **P1** | Синхронизировать `_vecs_t` после `_apply_vector_update` | concept_space.py, crystal_generator.py | 1 час | Точность evaluate(), neg sampling |
| **P1** | Заменить uniform random выбор негативов на PPMI-based | crystal_generator.py | 2 часа | Качество contrastive objective |
| **P1** | Ограничить `field_weight` сверху (max 3.0) | crystal_generator.py | 10 мин | Баланс компонентов LR |
| **P1** | Добавить field_bits pre-filter в _branch() | crystal_generator.py | 30 мин | Качество генерации |
| **P2** | Создать `GenerationResult` dataclass | crystal_generator.py | 30 мин | Type safety |
| **P2** | Интегрировать гормоны в STDP | crystal_generator.py | 2 часа | Intrinsic motivation → обучение |
| **P2** | Непрерывный curriculum вместо дискретного | train_full.py | 1 час | Плавное обучение |
| **P2** | Cached total_freq с invalidation | crystal_generator.py | 15 мин | Ускорение ~3% |
| **P2** | Ре-ортогонализация базиса на чекпоинтах | concept_space.py | 30 мин | Стабильность кодов |
| **P3** | Рефакторинг FCFConfig (выделить PathConfig, MetricPairBuilder) | fcf_config.py | 3 часа | Maintainability |
| **P3** | Юнит-тесты для STDP (GPU/CPU parity) | tests/ | 4 часа | Регрессионная защита |
| **P3** | Monitor + OOM fallback для GPU | crystal_generator.py | 2 часа | Стабильность |
| **P3** | `_quiet` заменить на специфичные обработчики | train_full.py | 30 мин | Safety |

---

## 8. Заключение

**FCF (Fractal Cognitive Field)** — архитектурно целостный и хорошо спроектированный проект с ясным разделением ответственности и гибридной CPU/GPU поддержкой. Ключевые инновации — STDP без градиентного спуска, фрактальное кодирование и гормональная модуляция — реализованы на высоком уровне.

Основные выявленные риски:
1. **Производительность GPU**: stale `_vecs_t` приводит к неточности латерального торможения и evaluate(). Требует минимальных изменений для синхронизации.
2. **Баланс обучения**: `field_weight` без верхней границы доминирует над остальными компонентами LR.
3. **Качество негативов**: как contrastive, так и neg sampling используют неоптимальные стратегии выбора кандидатов.
4. **Покрытие кода**: отсутствие тестов и неполные type hints увеличивают risk при доработках.

Рекомендуемый порядок действий: начать с P0-задач, перейти к P1 (особенно синхронизация `_vecs_t` и PPMI-based негативы), затем реализовать предложенные новые методы (Hormonal STDP gate, Adaptive beam width, Field-aware fluctuation) для повышения качества генерации.

---

*Сформировано коллегией AI-агентов: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent*
*Дата: 2026-06-17*
