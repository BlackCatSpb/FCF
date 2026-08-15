# FCF Архитектурный аудит V22

**Дата:** 23 июня 2026  
**HEAD:** 7585dfb  
**Предыдущий аудит:** V21 (935cd9f)  
**Коммитов после V21:** 34  
**Python-файлов:** 60 (25 модулей в `eva/symbolic/`)  
**Тестов:** 330 (315 passed, 14 skipped, 1 flaky order-dependent failure)

---

## Содержание

1. [Аудит GPU Chunking: _codes_master_t](#1-аудит-gpu-chunking-codes_master_t)
2. [ZeckendorfQuantizer: интеграция в pipeline](#2-zeckendorfquantizer-интеграция-в-pipeline)
3. [TemporalZeckendorf vs старые временные decay](#3-temporalzeckendorf-vs-старые-временные-decay)
4. [Гейтированные фичи: MorphSTDP, VSAAttention, HDTransformerLayer](#4-гейтированные-фичи-morphstdp-vsaattention-hdtransformerlayer)
5. [VRAM архитектура после GPU Chunking](#5-vram-архитектура-после-gpu-chunking)
6. [FormulaCoefficients: централизация конфигурации](#6-formulacoefficients-централизация-конфигурации)
7. [Потенциальные архитектурные проблемы](#7-потенциальные-архитектурные-проблемы)
8. [Предлагаемые архитектурные методы V23](#8-предлагаемые-архитектурные-методы-v23)
9. [Заключение](#9-заключение)

---

## 1. Аудит GPU Chunking: _codes_master_t

### 1.1 Декларированное изменение

Коммит `3713691` (HEAD~33) декларирует:

> "GPU chunking: remove full-V _codes_master_t (saves ~1.2GB), replace with per-batch compact codes."

### 1.2 Фактическое состояние кода

**crystal_generator.py:123** — поле объявлено как `None` с комментарием DEPRECATED:
```python
self._codes_master_t = None  # DEPRECATED: no longer full-V; use compact per-batch codes
```

**crystal_generator.py:359** — `_invalidate_torch()` обнуляет:
```python
self._codes_master_t = None
```

**crystal_generator.py:392-393** — `_sync_after_fluctuate()` явно подтверждает:
```python
# Keep _codes_master_t as None — no longer stored as full-V tensor
self._codes_master_t = None
```

**crystal_generator.py:300-302** — `_build_torch_tensors()` содержит комментарий:
```python
# ── Latent codes: no full-V GPU tensor (saves ~1.2 GB for 146K×2048×fp32).
# Subspace updates build compact per-batch codes in _gpu_stdp_apply.
```

**stdp_trainer.py:1157** — компактные коды строятся из CPU dict:
```python
# Build compact codes tensor from CPU dict (avoids full-V _codes_master_t)
codes_arr = np.zeros((len(_subspace_cids), latent_dim), dtype=np.float32)
```

### 1.3 Проблема: остаточные референсы

Несмотря на то, что `_codes_master_t` всегда `None` в продакшене, в коде остались три категории референсов:

#### Категория A: Fallback-код в concept_space.py (ОПАСНО)

**concept_space.py:2286, 2306, 2346-2349** — `_apply_subspace_update_batch()`:

```python
def _apply_subspace_update_batch(self, cids, grads, base_lr_val, subspace_lr, gen, codes_t=None):
    """Batched GPU subspace update for multiple CIDs.
    
    If codes_t is provided, uses it as compact codes tensor (n_cids, latent_dim).
    Otherwise reads from gen._codes_master_t (full-V fallback).
    После update, writes codes back to fractal.codes CPU dict.
    """
    ...
    if codes_t is not None:
        codes = codes_t                                      # ← compact path (используется)
    else:
        cids_t = torch.tensor(cids, dtype=torch.long, device=device)
        codes = gen._codes_master_t[cids_t]                  # ← fallback к None → TypeError!
    ...
    # Also sync back to _codes_master_t if it exists (full-V fallback)
    if codes_t is None and hasattr(gen, '_codes_master_t') and gen._codes_master_t is not None:
        cids_t = torch.tensor(cids, dtype=torch.long, device=device)
        gen._codes_master_t[cids_t] = new_codes.to(torch.float32)
```

**Проблема:** Если `codes_t=None` (вызов без аргумента), то `gen._codes_master_t[cids_t]` разыменует `None[cids_t]` → `TypeError: 'NoneType' object is not subscriptable`. На практике этот баг не проявляется, потому что:
- STDPTrainer._gpu_stdp_apply() всегда передаёт `codes_t=` (строка 1167)
- Прямой вызов из тестов QN-49 передаёт `gen` без `codes_t`, но тесты пропускаются при `_codes_master_t is None`

**Оценка риска:** Средний. При добавлении нового кода, вызывающего `_apply_subspace_update_batch` без `codes_t=`, произойдёт краш.

#### Категория B: Тесты, проверяющие _codes_master_t (устаревшие, но безвредные)

**test_stdp.py:1148-1206** — QN-49 тесты (4 теста). Все имеют guard:
```python
if not hasattr(gen, '_codes_master_t') or gen._codes_master_t is None:
    pytest.skip("No _codes_master_t")
```

**test_stdp.py:1674-1686** — `test_codes_fp32_roundtrip` — тест на fp32 roundtrip:
```python
if gen._codes_master_t is None or gen._codes_master_t.dtype != torch.float32:
    pytest.skip("No fp32 _codes_master_t")
```

**test_stdp.py:1746-1766** — `test_sync_after_fluctuate_*` — два теста с guard:
```python
if gen._codes_master_t is None:
    pytest.skip("No _codes_master_t")
```

**Оценка:** 7 тестов пропускаются — они проверяют удалённую функциональность. При включении GPU-тестов с реальной CUDA они будут пропущены. Рекомендуется удалить эти тесты, так как `_codes_master_t` больше не существует по архитектуре.

### 1.4 GpuChunkManager — секторный paging

**crystal_generator.py:1126-1285** — `GpuChunkManager` реализован полностью:

- Инициализация: строка 344-347 в `_ensure_torch()`
- Sector map: `_build_sector_map()` (строка 1168) — строит CID→sector из `fractal._sector_index`
- Загрузка батча: `load_batch()` (строка 1200) — возвращает compact `(all_cids, vecs_np, mapping)`
- LRU-кэш: `_chunks` с `_lru` списком (строка 1157-1160)
- Обратная синхронизация: `mark_dirty()` / `sync_all()` / `_sync_chunk()`

**Проблема `_load_chunk`:** строка 1231 содержит неработающий код:
```python
locs = [cid for cid, loc in self._cid_loc.items() if loc[0] == key] \
    if hasattr(self._cid_loc.get(next(iter([k for k in self._cid_loc \
    if self._cid_loc[k][0] == key])), None), '__iter__') else []
```
Эта строка никогда не выполняется (за ней сразу следует блок с `si[self.depth][key]`), но содержит `StopIteration` при пустом `_cid_loc`. Необходимо удалить.

**Проблема интеграции:** `GpuChunkManager` создаётся, но `load_batch()` не вызывается нигде в pipeline. Метод `_sync_dirty_cpu()` (строка 420) использует `_chunk_mgr.mark_dirty()` и прямой `_vecs_t[cids_t]` — то есть chunk manager используется только для dirty-трекинга, а не для фактического paging векторов. Полный paging не реализован.

### 1.5 Вывод по GPU Chunking

| Аспект | Статус |
|--------|--------|
| `_codes_master_t` удалён | ✅ Да, везде `None` |
| Остаточные референсы | ⚠️ concept_space.py:2306 (TypeError при вызове без codes_t) |
| GpuChunkManager реализован | ✅ Полный класс (1285-1126=159 строк) |
| GpuChunkManager интегрирован | ⚠️ Только dirty-трекинг, paging не задействован |
| VRAM экономия | ✅ ~1140 MB (с 1798 до 657 MB) |

---

## 2. ZeckendorfQuantizer: интеграция в pipeline

### 2.1 Код

**fibonacci_utils.py:72-133** — `ZeckendorfQuantizer`:

```python
class ZeckendorfQuantizer:
    """Quantize float weights to HD vectors via Zeckendorf bundle.
    
    encode(w) → zeckendorf(round(|w| * scale)) → bundle indices → HD sum
    """
    def __init__(self, dim: int = 768, max_fib_value: int = 100000,
                 scale: float = 10000, seed: int = 42):
        ...
        rng = np.random.RandomState(seed)
        vecs = rng.randn(n_vecs, dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        self._vecs = vecs / np.clip(norms, 1e-10, None)
    
    def encode(self, w: float) -> np.ndarray:
        idx = int(round(abs(w) * self.scale))
        ...
        fibs = FibonacciUtils.zeckendorf(idx)
        indices = [self._fib_to_idx[f] for f in fibs if f in self._fib_to_idx]
        vec = np.array(list(self._vecs[i] for i in indices)).sum(axis=0)
        return vec / n if n > 1e-10 else ...
```

### 2.2 Статус интеграции

**Где используется:** Только в тестах.

**test_stdp.py:468-487** — 3 теста:
- `test_quantizer_encode_decode` — smoke test
- `test_quantizer_batch` — batch-кодирование
- `test_quantizer_lossy_compression` — проверка сжатия

**Где НЕ используется:**
- `crystal_generator.py` — не вызывает `ZeckendorfQuantizer`
- `stdp_trainer.py` — не вызывает `ZeckendorfQuantizer`
- `concept_space.py` — не вызывает `ZeckendorfQuantizer`

**Существующая альтернатива:** `_bind_weighted_zeckendorf` в `concept_space.py:99-121` — делает Zeckendorf-взвешивание напрямую через `_hybrid_bind(vec, sub)`, без `ZeckendorfQuantizer`. 

`VSAAttention._scale_bundle()` в `vsa_attention.py:64-79` — делает своё Zeckendorf-взвешивание через `_weight_vector(p, max_val)` + `_hybrid_bind(value, weight_hv)`, тоже без `ZeckendorfQuantizer`.

**Вывод:** `ZeckendorfQuantizer` — изолированный утилитарный класс без точек врезки в pipeline. Это потенциально полезный компонент для будущих методов квантования (например, квантования весов гармонизации или LR), но на данный момент он не соединён с основной логикой.

### 2.3 Проблемы реализации

1. **Потеря знака:** `encode()` использует `abs(w) * self.scale` — знак веса теряется. Для отрицательных весов (например, отталкивание) это проблематично.
2. **Масштаб фиксирован:** `scale=10000` — при весах >0.1, `idx` может превысить `max_fib_value`, что приведёт к пустому `indices` и нулевому вектору.
3. **Float16 не поддерживается:** Векторы хранятся в float32, что увеличивает память при batch-кодировании.

---

## 3. TemporalZeckendorf vs старые временные decay

### 3.1 Код

**fibonacci_utils.py:135-203** — `TemporalZeckendorf`:

```python
class TemporalZeckendorf:
    def __init__(self, max_steps: int = 1000000):
        self._cache = {}
        self._max_depth = len(FibonacciUtils.zeckendorf(max_steps)) + 1
    
    def theta(self, distance: int, fast_window: int = 5, slow_window: int = 10) -> tuple[float, float]:
        """Zeckendorf-based temporal decay for STDP, replaces exp(-d/tau).
        Returns (fast_theta, slow_theta) — both in (0, 1], decreasing with distance.
        No free tau parameter: the Fibonacci hierarchy IS the decay schedule.
        """
        if distance <= 0:
            return (1.0, 1.0)
        idx = self._largest_fib_idx(distance)
        base = (self._max_depth - idx) / max(self._max_depth, 1)
        if distance <= fast_window:
            fast = base
        else:
            fast = 0.0
        if distance <= slow_window:
            slow = base
        else:
            slow = 0.0
        return (max(fast, 0.0), max(slow, 0.0))
```

### 3.2 Интеграция

**stdp_trainer.py:693-713** — подключение через флаг `FCFConfig().use_temporal_zeckendorf`:

```python
if FCFConfig().use_temporal_zeckendorf:
    from eva.symbolic.fibonacci_utils import TemporalZeckendorf as _TZ
    _tz = _TZ()
    fast_th, slow_th = _tz.theta(abs(j-i))
    theta_gate = max(fast_th, _fc.theta_fast_min)
    gen_updates[ids[j]].append((ids[i], lr * theta_gate))
    n_pairs += 1
    theta_slow = max(slow_th, _fc.theta_slow_min) * _fc.theta_slow_scale
    slow_lr = lr * theta_slow if fast_th > _fc.theta_fast_min else 0.0
    ...
else:
    theta_gate = math.exp(-min(abs(j-i), 5) / max(gen.theta_tau, 1.0))
    ...
    theta_slow = math.exp(-min(abs(j-i), 10) / max(gen.theta_tau * _fc.theta_tau_slow_mult, 1.0))
```

### 3.3 Анализ

**Проблема 1: Создание объекта на каждый вызов.** Каждый раз при обработке пары создаётся новый `TemporalZeckendorf`, что приводит к пересчёту `_max_depth` при каждом `__init__`. Переменная `_cache` (Lru-кэш Zeckendorf-разложений) также создаётся заново, лишаясь преимущества кэширования. Необходимо вынести создание `_TZ` на уровень класса/модуля.

**Проблема 2: Семантическое различие.** Старая формула:
```python
theta = exp(-distance / tau)
```
даёт гладкое экспоненциальное затухание. TemporalZeckendorf даёт ступенчатую функцию, где `fast_theta` принимает всего ~6 различных значений (по числу Fib-шагов до 5). Для `distance=3` Fib-индекс `_largest_fib_idx(3) = 4` (F₄=3), `_max_depth ≈ 31` (для 1M), `base = (31-4)/31 ≈ 0.87`. Для `distance=5` Fib-индекс=5, `base = (31-5)/31 ≈ 0.84`. Разница мала — кривая затухания более пологая, чем экспонента с `tau=12`.

**Проблема 3: Константные окна.** `fast_window=5` и `slow_window=10` — фиксированные константы. Для `distance=6` (быстрая память за пределами окна) `fast=0.0`. Это приводит к бинарному поведению (0 или base), без плавного перехода. Zeckendorf-иерархия могла бы дать 6-7 уровней затухания, но из-за окон они редуцируются до 2-3.

**Проблема 4: `assert` в `test_stdp.py`.** Тесты `test_temporal_lcp`, `test_theta_valid_range` (test_stdp.py:493-510) проверяют, что `0 <= theta <= 1` и что LCP работает. Но нет теста, сравнивающего TemporalZeckendorf с `exp(-d/tau)` на репрезентативном распределении distance.

### 3.4 Вывод

TemporalZeckendorf заменяет старые экспоненциальные decay (защищён флагом `use_temporal_zeckendorf: bool = True`). Замена семантически корректна, но:
- Создание объекта на каждый вызов — O(n_pairs) лишних аллокаций
- Кривая затухания существенно отличается от экспоненты
- Нет теста на эквивалентность/преемственность поведения

---

## 4. Гейтированные фичи: MorphSTDP, VSAAttention, HDTransformerLayer

### 4.1 MorphSTDP

**Файл:** `semantic_piece.py:22-131`

**Интеграция:** `stdp_trainer.py:335-341` — ленивая инициализация в `_harmonize_batch()`:
```python
_c = FCFConfig()
if _c.use_morph_stdp and not hasattr(self, '_morph_stdp_batches'):
    from eva.symbolic.semantic_piece import MorphSTDP
    self._morph_stdp = MorphSTDP(dim=cs.dim, cohesion_threshold=_c.morph_stdp_cohesion)
    self._morph_stdp_batches = 0
```

**Проблема:** `MorphSTDP` требует `char_vecs` (поле `self.char_vecs: Dict[int, np.ndarray] = {}`), которое должно быть заполнено из `CharEnvelope`. Однако в `_harmonize_batch` нет вызова `self._morph_stdp.char_vecs.update(char_envelope.vecs)`. В результате `self._morph_stdp.char_vecs` остаётся пустым, `bind_char()` возвращает нулевой вектор (строка 57-58: `if v1 is None or v2 is None: return np.zeros(...)`), и `observe()` не производит никакого обучения.

**Дублирование:** `CharEnvelope.stdp_update()` (semantic_piece.py:183-199) уже реализует STDP между символами. MorphSTDP — это та же идея, но на уровне биграмм символов. Два механизма могут конфликтовать.

### 4.2 VSAAttention

**Файл:** `vsa_attention.py:1-139`

**Интеграция:** `crystal_generator.py:882-894` — в `_branch()`:
```python
if FCFConfig().use_vsa_attention:
    if not hasattr(self, '_vsa_attn'):
        from eva.symbolic.vsa_attention import VSAAttention
        self._vsa_attn = VSAAttention(dim=self.cs.dim, n_heads=1, use_fib_pos=False)
    ctx_vecs = [self.cs.concept_vector(c) for c in seq[-5:] if self.cs.concept_vector(c) is not None]
    if len(ctx_vecs) >= 1 and v_prev is not None:
        attn_out = self._vsa_attn.forward(v_prev, ctx_vecs, ctx_vecs)
        attn_sims = {cid: float(np.dot(self.cs.concept_vector(cid) or np.zeros(self.cs.dim), attn_out))
                     for cid in all_cids}
```

**Проблема (она же единственный падающий тест):** строка 889 — `self.cs.concept_vector(cid) or np.zeros(...)`. Если `concept_vector()` возвращает numpy-массив, то `array or scalar` вызывает `ValueError: The truth value of an array with more than one element is ambiguous`. Тест `test_branch_fuzz_zero_temp` падает при определённом порядке тестов, когда `concept_vector(cid)` может вернуть массив с несколькими элементами, который не может быть приведён к bool.

**Фикс:** заменить `v or np.zeros(...)` на `v if v is not None else np.zeros(...)`.

**Дублирование с HDTransformerLayer:** VSAAttention — это однослойное cross-attention (query, keys → weighted sum). HDTransformerLayer — многослойное self-attention с LSH. Если включены обе фичи, генерация будет использовать VSAAttention для реранкинга, а STDP-train —HDTransformerLayer для рефайнмента. Это разные точки врезки, дублирования нет.

### 4.3 HDTransformerLayer

**Файл:** `hdtransformer_layer.py:1-200`

**Интеграция:** `stdp_trainer.py:268-284` — после негативной выборки:
```python
if FCFConfig().use_hd_transformer:
    if not hasattr(self, '_hd_transformer'):
        from eva.symbolic.hdtransformer_layer import HDTransformerLayer
        self._hd_transformer = HDTransformerLayer(dim=cs.dim, num_heads=2, top_k=5)
    for ids in all_ids:
        seq = [cs.concept_vector(c) for c in ids if cs.concept_vector(c) is not None]
        if len(seq) >= 2:
            out = self._hd_transformer.forward(seq)
            for j, cid in enumerate(ids):
                if j < len(out) and cs.concept_vector(cid) is not None:
                    pull = out[j] - cs.concept_vector(cid)
                    ...
                    cs._apply_vector_update(cid, new_v / nn)
```

**Проблема:** `HDTransformerLayer._lsh_attention()` (строка 60-119) не использует LSH-индекс. Поле `self._lsh = None` (строка 38) не инициализируется. `_lsh_attention` делает полный перебор `kv_pairs` с вычислением косинуса для каждого — O(N²) на последовательность. Без LSH это не `LSH-attention`, а полный pairwise attention.

**Проблема `_random_masks`:** `hdtransformer_layer.py:37`:
```python
self.masks = _random_masks(dim, n_heads=num_heads)
```
`_random_masks` импортируется из `eva.symbolic.experimental.vsa_utils` (строка 16). Если этот модуль не загружен (из-за `try: except ImportError: pass` в `experimental/__init__.py`), то `HDTransformerLayer.__init__` упадёт с `NameError`. Для `num_heads > 0` маски — это список. Для `num_heads=1` маски не используются (строка 147: `if self.num_heads > 1:`). То есть при `num_heads=2` (как в интеграции) маски обязательны, но могут отсутствовать при проблемах с импортом.

### 4.4 Взаимодействие гейтированных фич

Все три фичи управляются булевыми флагами в `FCFConfig` (строки 812-819):

```python
use_morph_stdp: bool = True
use_vsa_attention: bool = True
use_hd_transformer: bool = True
use_temporal_zeckendorf: bool = True
use_morph_manifold: bool = True
```

При всех `True` порядок выполнения в `_train_batch`:
1. STDP pairs → GPU STDP → негативная выборка
2. **HDTransformerLayer** рефайнмент (self-attention на последовательности)
3. Центроид + lattice update + HDC n-gram
4. Гармонизация (MorphSTDP + morph_manifold внутри)
5. При генерации: **VSAAttention** реранкинг в `_branch()`

**Проблема порядка:** HDTransformerLayer применяется к векторам концептов ДО того, как они синхронизированы с CPU (`_sync_dirty_cpu` вызывается позже, при гармонизации). Если GPU-тензоры не синхронизированы, HDTransformerLayer может использовать устаревшие векторы. На практике это не крашится, но создаёт race condition между GPU и CPU представлениями.

### 4.5 Вывод по гейтированным фичам

| Фича | Интеграция | Проблемы |
|------|-----------|----------|
| MorphSTDP | `_harmonize_batch` | ⚠️ `char_vecs` пуст — нет обучения |
| VSAAttention | `_branch` | 🔴 Баг с array `or` scalar (1 failed) |
| HDTransformerLayer | `_train` после негатива | ⚠️ LSH не подключён, O(N²) |
| TemporalZeckendorf | `_build_pairs` | ⚠️ Объект создаётся в цикле |
| MorphManifold | `_harmonize_batch` | ✅ Работает |

---

## 5. VRAM архитектура после GPU Chunking

### 5.1 Полная картина GPU-тензоров

| Тензор | Тип | Размерность | Формула | VRAM |
|--------|-----|-------------|---------|------|
| `_vecs_t` | fp16 | V×D | 146K×768×2 | 213.9 MB |
| `_ema_vecs_t` | fp16 | V×D | 146K×768×2 | 213.9 MB |
| `_mom_t` | fp16 | V×D | 146K×768×2 | 213.9 MB |
| `_ce_t` | fp32 | V | 146K×4 | 0.6 MB |
| `_fb_t` | uint8 | V×64 | 146K×64B | 8.9 MB |
| `_cf_t` | fp32 | V | 146K×4 | 0.6 MB |
| `_pt2_t` | fp32 | V | 146K×4 | 0.6 MB |
| `_skip2_t` | fp32 | V | 146K×4 | 0.6 MB |
| `_basis_t` | fp32 | L×D | 2048×768×4 | 6.0 MB |
| `_fused_buf` | fp32 | 4096×(D+1) | 4096×769×4 | 12.0 MB |
| **Итого** | | | | **~670 MB** |
| *До удаления `_codes_master_t`* | | | | *~1810 MB* |

`_codes_master_t` (fp32, 146K×2048) занимал **1140 MB** — его удаление снизило VRAM с **~1810 MB** до **~670 MB**.

### 5.2 Лимиты

Для GPU с **2 GB VRAM**:
- Доступно: ~1800 MB (после ОС и CUDA runtime)
- Занято: ~670 MB (базовые тензоры)
- Буфер для батча: ~200 MB (пары, коды, градиенты)
- Резерв: **~930 MB** — достаточно для батчей среднего размера

Для GPU с **1 GB VRAM** (например, ноутбучный Intel Arc / старые мобильные):
- Может не хватить — `_vecs_t + _ema_vecs_t + _mom_t` уже 642 MB
- На помощь приходит CPU fallback или `_torch_fallback = True`

### 5.3 Неиспользуемый потенциал GpuChunkManager

Текущая VRAM (670 MB) может быть дополнительно снижена до ~300 MB при включении полного paging через `GpuChunkManager`:
- Только активные сектора (~10% от 1024 = 100 секторов)
- `_vecs_t` вместо 214 MB → ~21 MB на GPU
- `_mom_t` тоже можно paging'овать
- `_ema_vecs_t` можно хранить на CPU

Однако полный paging пока не реализован (см. раздел 1.4).

### 5.4 Проблема `_fused_buf` с плавающим размером

`_build_torch_tensors:330-332`:
```python
if self._fused_buf is None or self._fused_buf.shape[1] != D + 1 or self._fused_buf.device != dev:
    init_rows = min(V, 4096)
    self._fused_buf = torch.zeros(init_rows, D + 1, device=dev, dtype=torch.float32)
```

`init_rows = min(V, 4096)` = 4096. Но `fused_buf` может динамически расти в `_scatter_add_fused` (stdp_trainer.py, функция `_ensure_fused_buf_size`), потенциально до V, что добавит ещё (146K×769×4) = **430 MB** в пике.

### 5.5 VRAM под нагрузкой (batch)

При batch=512 (типичный microbatch):
- `codes_t`: 512×2048×4 = 4 MB (временный)
- `grads_t`: 512×768×4 = 1.5 MB
- `meta_t`: 512×9×4 = 18 KB
- `new_codes`: 512×2048×4 = 4 MB

Всего временных: ~10 MB — незначительно.

---

## 6. FormulaCoefficients: централизация конфигурации

### 6.1 Сбор всех коэффициентов

**fcf_config.py:306-448** — `FormulaCoefficients` датакласс:

```
RRF weights:           7 полей (rrf_graph, rrf_syntax, ...)
Theta-decay:           8 полей (theta_tau_default, theta_tau_slow_mult, ...)
PMI mapping:           7 полей (pmi_slope, pmi_intercept, ...)
Anti-repetition:       1 поле (antirep_decay)
Edge weight:           3 поля (edge_weight_min, edge_ppmi_cap, ...)
Target boost:          2 поля
Novelty:               1 поле
Hybrid bind alpha:     5 полей
Homeostatic:           2 поля
Intent centroid:       1 поле
Confidence:            1 поле
STDP freq/field:       6 полей
Hormonal modulation:   2 поля
Negative sampling:     2 поля
Contrastive:           1 поле
Code mixing:           2 поля
Concept usage:         1 поле
Cluster potential:     3 поля
RRF boost:             1 поле
Intonation/hormonal:  35 полей (DA, ACh, NE, 5-HT, NA)
Итого:                ~92 поля
```

### 6.2 Точки чтения

- **concept_space.py:63** — `_fc = FCFConfig().formula` (hybrid bind alpha)
- **stdp_trainer.py:604** — `_fc = FCFConfig().formula` (STDP pairs)
- **stdp_trainer.py:975** — `_fc = FCFConfig().formula` (CPU contrastive)
- **crystal_generator.py:44** — импорт `FormulaCoefficients`
- **crystal_generator.py:660, 762, 853** — `_f = FCFConfig().formula` (RRF, branch, etc.)
- **hormonal_system.py:35** — `_fc = formula if formula is not None else FCFConfig().formula`

### 6.3 Проблемы централизации

**Проблема 1: Импорт внутри функций.** Почти все модули вызывают `FCFConfig().formula` при каждом вызове (не при инициализации). Например, `_build_pairs` (stdp_trainer.py:604) вызывается для каждого предложения — каждый раз создаётся новый `FCFConfig()` + доступ к `formula`. Хотя `FCFConfig` — синглтон (через `@dataclass` без явного механизма), повторяющееся конструирование — лишние аллокации.

**Проблема 2: Не все модули читают из config.** `qwen_knowledge.py` использует свои константы на уровне модуля:
```python
COS_NEUTRAL = 0.15
COS_REPEL_THRESHOLD = 0.20
LR_REPEL_MAX = 0.15
LR_BOOST_MAX = 0.30
```
Эти константы должны быть в `FormulaCoefficients`.

**Проблема 3: `syntax_lattice.py`** содержит:
```python
NGRAM_DECAY = 0.999
PPMI_THRESHOLD = 0.5
MIN_COUNT = 2
```
Перенесены в конфиг (commit `6494981`), но часть параметров всё ещё может быть хардкожена.

**Проблема 4: `morph_vocab.py`** содержит хардкоженные константы для морф-анализа (пороги, размеры окон).

### 6.4 Вывод

FormulaCoefficients — успешная централизация (~92 параметра), но неполная. В `qwen_knowledge.py`, `syntax_lattice.py`, `morph_vocab.py` остались хардкоженные константы. Архитектурная дисциплина требует провести `grep -rn "= [0-9]\+\\.[0-9]" *.py` для выявления всех неуправляемых чисел.

---

## 7. Потенциальные архитектурные проблемы

### 7.1 Flaky test: test_branch_fuzz_zero_temp

**Корень:** `crystal_generator.py:889`:
```python
self.cs.concept_vector(cid) or np.zeros(self.cs.dim)
```

`concept_vector()` возвращает `np.ndarray` или `None`. Оператор `or` с numpy-массивом вызывает `ValueError`, если массив содержит >1 элемента. Тест проходит изолированно, но падает при определённом порядке тестов, когда глобальное состояние (`_torch_fallback`, `_vecs_t`) влияет на `concept_vector()`.

**Фикс:**
```python
v = self.cs.concept_vector(cid)
attn_sims = {cid: float(np.dot(v if v is not None else np.zeros(self.cs.dim), attn_out))
             for cid in all_cids}
```

### 7.2 MorphSTDP: мёртвая инициализация

`char_vecs` остаётся пустым (раздел 4.1). Лёгкий фикс:
```python
if hasattr(self, '_morph_stdp') and self._morph_stdp is not None:
    if hasattr(ef, 'char_env') and ef.char_env is not None:
        self._morph_stdp.char_vecs.update(ef.char_env.vecs)
```

### 7.3 TemporalZeckendorf: ресорс-лик

`_TZ()` создаётся в `_build_pairs` для каждой пары — O(N) лишних объектов. Фикс: превратить `TemporalZeckendorf` в модульный синглтон или вынести в поле `STDPTrainer.__init__`.

### 7.4 `__init__.py` experimental: тихий ImportError

**eva/symbolic/experimental/__init__.py** — `_random_masks`, `_fractal_convolution` и другие могут не импортироваться. HDTransformerLayer полагается на них без fallback. Рекомендуется:
- Либо гарантировать импорт (убрать `try/except`)
- Либо добавить fallback-реализации

### 7.5 Фрактальное поле: _matrix_dirty не синхронизирован

`_matrix_dirty = True` устанавливается в `_apply_vector_update` (concept_space.py:2278), но `_build_torch_tensors` (crystal_generator.py:338) сбрасывает `_torch_dirty = False` без синхронизации полевых бит. Если поле изменилось (`W_proj` updated), а `_fb_t` не перестроен, то `_ensure_fb_tensor` (строка 462) не обновится до следующего `_torch_dirty`.

### 7.6 VSAGrid факторизация

`ARCHITECTURE.md` утверждает `VSAGrid.factorize(768) = (8, 8, 6, 2)`, но `8×8×6×2 = 768` — это правильно. Однако `experimental/vsa_grid.py` может давать другой результат при разных размерностях (например, `dim=2048` для латентного пространства). Факторизация не проверена для `latent_dim=2048`.

---

## 8. Предлагаемые архитектурные методы V23

### 8.1 SectorPreloader — асинхронная prefetch секторов GPU

Текущий GpuChunkManager синхронно подгружает сектора. Для скрытия latency PCIe:

```python
class SectorPreloader:
    """Async prefetch of next-batch sectors during STDP computation.
    
    Uses CUDA streams to overlap PCIe transfer with kernel execution.
    """
    def __init__(self, chunk_mgr: GpuChunkManager, device: torch.device,
                 prefetch_depth: int = 2):
        self._mgr = chunk_mgr
        self._device = device
        self._prefetch_depth = prefetch_depth
        self._streams = [torch.cuda.Stream() for _ in range(prefetch_depth)]
        self._next_stream = 0
    
    def predict_sectors(self, current_cids: List[int],
                        history: List[List[int]]) -> Set[Any]:
        """Predict next-batch sectors from n-gram transition history.
        
        Args:
            current_cids: CIDs in current batch
            history: list of previous-batch CID sequences
            
        Returns:
            set of predicted sector keys for next batch
        """
        predicted = set()
        for seq in history[-3:]:  # last 3 batches
            for cid in seq[-2:]:  # last 2 tokens
                successors = self._mgr.gen.lattice.predict_next(cid, top_k=5)
                for scid in successors:
                    sk = self._mgr.get_sector_key(scid)
                    if sk is not None:
                        predicted.add(sk)
        # Always keep current sectors
        for cid in current_cids:
            sk = self._mgr.get_sector_key(cid)
            if sk is not None:
                predicted.add(sk)
        return predicted
    
    @torch.no_grad()
    def prefetch(self, sector_keys: Set[Any]) -> None:
        """Async H2D copy of predicted sectors.
        
        Uses round-robin CUDA streams; waits if all streams busy.
        """
        stream = self._streams[self._next_stream % self._prefetch_depth]
        self._next_stream += 1
        with torch.cuda.stream(stream):
            for key in sector_keys:
                if key not in self._mgr._chunks:
                    # Pre-allocate GPU buffer, start async copy
                    cids = self._mgr._sector_cids(key)
                    if not cids:
                        continue
                    vecs_np = np.stack([
                        self._mgr.cs.concept_vectors.get(c, np.zeros(self._mgr.dim))
                        for c in cids
                    ])
                    vecs_t = torch.empty(len(cids), self._mgr.dim,
                                         device=self._device, dtype=torch.float16)
                    vecs_t.copy_(torch.from_numpy(vecs_np), non_blocking=True)
                    self._mgr._chunks[key] = {'vecs': vecs_t, 'cids': cids}
                    self._mgr._lru.append(key)
    
    def synchronize(self) -> None:
        """Sync all prefetch streams before batch computation."""
        for s in self._streams:
            s.synchronize()
```

**Аргументация:** STDP на GPU занимает ~1-5ms на батч. PCIe transfer 1MB (один сектор) — ~50μs. Prefetch позволяет скрыть до 80% transfer latency.

### 8.2 MorphReservoir — разреженная морфемная память с забыванием

MorphSTDP не накапливает морфемы — `discover_morphemes()` находит биграммы с высоким `char_bigram_cohesion`, но не имеет механизма забывания для редких морфем. Предлагается:

```python
class MorphReservoir:
    """Reservoir sampling + recency-weighted morpheme memory.
    
    Maintains a fixed-size cache of (morph_vector, frequency, last_seen).
    On overflow, evicts least-recently-used morph by (freq / (now - last_seen)).
    Integrates with MorphSTDP.discover_morphemes() for periodic refresh.
    """
    
    def __init__(self, capacity: int = 10946,  # F21
                 dim: int = 2048,
                 recency_decay: float = 0.995):
        self.capacity = capacity
        self.dim = dim
        self.recency_decay = recency_decay
        
        self.vectors: Dict[int, np.ndarray] = {}        # morph_id → HD vector
        self.frequencies: Dict[int, int] = {}            # morph_id → count
        self.last_seen: Dict[int, int] = {}              # morph_id → step
        self._step = 0
        
        self._candidates: List[Tuple[int, int, float]] = []  # (char1, char2, cohesion)
        self._candidate_max = 5000
    
    def observe_morph(self, morph_id: int, vector: np.ndarray) -> None:
        """Add or update a morph vector with recency tracking."""
        self._step += 1
        if morph_id in self.vectors:
            # Update: EMA merge with existing
            old = self.vectors[morph_id]
            merged = old * 0.9 + vector * 0.1
            mn = np.linalg.norm(merged)
            self.vectors[morph_id] = merged / mn if mn > 1e-10 else merged
            self.frequencies[morph_id] += 1
        else:
            # Insert with eviction if at capacity
            if len(self.vectors) >= self.capacity:
                self._evict_one()
            self.vectors[morph_id] = vector.astype(np.float16)
            self.frequencies[morph_id] = 1
        
        self.last_seen[morph_id] = self._step
    
    def _evict_one(self) -> None:
        """Score = freq / (now - last_seen + 1); evict lowest score."""
        now = self._step
        best_id = None
        best_score = float('inf')
        for mid, freq in self.frequencies.items():
            age = now - self.last_seen.get(mid, 0)
            score = age / max(freq, 1)
            if score < best_score:
                best_score = score
                best_id = mid
        if best_id is not None:
            del self.vectors[best_id]
            del self.frequencies[best_id]
            del self.last_seen[best_id]
    
    def register_candidate(self, c1: int, c2: int, cohesion: float) -> None:
        """Register a (char1, char2) pair as potential morpheme."""
        if len(self._candidates) >= self._candidate_max:
            # Remove lowest cohesion
            self._candidates.sort(key=lambda x: x[2])
            self._candidates = self._candidates[-self._candidate_max//2:]
        self._candidates.append((c1, c2, cohesion))
    
    def harvest(self, threshold: float = 0.6, min_count: int = 3) -> int:
        """Convert high-cohesion candidates to morphemes.
        
        Returns:
            number of new morphemes harvested
        """
        new_count = 0
        used_chars = set()
        for c1, c2, cohesion in sorted(self._candidates,
                                        key=lambda x: -x[2]):
            if cohesion < threshold:
                break
            if c1 in used_chars or c2 in used_chars:
                continue
            morph_id = abs(hash((c1, c2))) % (2**31 - 1)
            if morph_id not in self.vectors:
                # Build morph vector from char vectors
                v1 = self._char_vecs.get(c1)
                v2 = self._char_vecs.get(c2)
                if v1 is not None and v2 is not None:
                    from eva.symbolic.concept_space import _hybrid_bind
                    bound = _hybrid_bind(v1, np.roll(v2, 1))
                    bn = np.linalg.norm(bound)
                    if bn > 1e-10:
                        self.vectors[morph_id] = (bound / bn).astype(np.float16)
                        self.frequencies[morph_id] = min_count
                        self.last_seen[morph_id] = self._step
                        new_count += 1
        self._candidates.clear()
        return new_count
```

**Аргументация:** Без забывания морфемная память растёт неограниченно. С reservoir и recency-weighted эвикцией MorphSTDP становится автономным — редкие морфемы вытесняются, частотные остаются. Ёмкость 10946 (F21) — Fibonacci-константа.

### 8.3 VSAInterpolation — интерполяция концептов через риманову геодезию

Текущий STDP двигает концепты вдоль хорд (разность векторов). Для единичной сферы правильный путь — геодезическая (дуга большого круга). STDP small-step приближает geodesic, но для больших шагов (high LR, destab) хорда отклоняется от geodesic на O(θ³).

```python
def vsa_interpolate(v_a: np.ndarray, v_b: np.ndarray,
                    t: float) -> np.ndarray:
    """Spherical linear interpolation (slerp) between two unit vectors.
    
    Unlike chord lerp, slerp follows the geodesic on the unit hypersphere.
    Used for smooth concept transition in STDP and Harmonizer.
    
    Args:
        v_a: start vector (unit norm, shape [D])
        v_b: end vector (unit norm, shape [D])
        t: interpolation factor in [0, 1]
    
    Returns:
        interpolated unit vector on geodesic from v_a to v_b
    """
    cos_theta = np.clip(np.dot(v_a, v_b), -1.0, 1.0)
    theta = math.acos(cos_theta)
    
    if theta < 1e-10:
        return v_a.copy()
    
    sin_theta = math.sin(theta)
    return (v_a * math.sin((1 - t) * theta) / sin_theta +
            v_b * math.sin(t * theta) / sin_theta)


def vsa_interpolate_batch(vecs: np.ndarray, target: np.ndarray,
                           t: float, eps: float = 1e-10) -> np.ndarray:
    """Batch slerp: vectorized spherical interpolation.
    
    All target vectors are the same; start vectors vary per row.
    
    Args:
        vecs: (N, D) start matrix (unit row norms)
        target: (D,) target vector (unit norm)
        t: interpolation factor in [0, 1]
    
    Returns:
        (N, D) interpolated matrix (unit row norms)
    """
    dots = np.clip(vecs @ target, -1.0, 1.0)
    thetas = np.arccos(dots)
    sin_thetas = np.sin(thetas)
    mask = sin_thetas > eps
    
    result = np.zeros_like(vecs)
    t_inv = 1.0 - t
    
    for i in range(len(vecs)):
        if mask[i]:
            result[i] = (vecs[i] * math.sin(t_inv * thetas[i]) / sin_thetas[i] +
                         target * math.sin(t * thetas[i]) / sin_thetas[i])
        else:
            result[i] = vecs[i].copy()
    
    return result


def stdp_with_slerp(concept_vec: np.ndarray, pull_vec: np.ndarray,
                    lr: float) -> np.ndarray:
    """STDP update via geodesic interpolation.
    
    Replaces: new_v = normalize(v + lr * (pull - v))
    With:     new_v = slerp(v, pull, lr * decay)
    
    Benefits:
    - Always stays on sphere (no norm drift)
    - Large lr doesn't overshoot
    - Follows true Riemannian gradient
    
    Args:
        concept_vec: current concept vector (unit norm)
        pull_vec: target concept vector (unit norm)
        lr: learning rate in (0, 1]
    
    Returns:
        new concept vector (unit norm)
    """
    effective_lr = min(lr, 1.0)
    return vsa_interpolate(concept_vec, pull_vec, effective_lr)
```

**Аргументация:** Текущий STDP: `v += lr * (v_pull - v)` → `v = normalize(v)`. Это chord lerp + projection. Для маленьких LR (0.01) разница с geodesic незначительна (O(θ²)). Для destab (LR=0.3) и гармонизации (LR=0.05-0.1) отклонение может достигать 0.5° на шаг, что за 1000 шагов даёт накопленную ошибку ~7°. Slerp устраняет этот дрейф принципиально.

**Интеграция в pipeline:**
- `STDPTrainer._cpu_stdp_apply`: заменить `v + lr * pull` на `slerp(v, pull, lr)`
- `ConceptSpace._apply_subspace_update`: опциональный параметр `use_slerp=True`
- Совместимость: при `lr < 0.01` разница пренебрежима, можно оставить старый путь по умолчанию

---

## 9. Заключение

### 9.1 Сводка архитектурных изменений V22

| Компонент | Статус | Эффект |
|-----------|--------|--------|
| GPU chunking: `_codes_master_t` удалён | ✅ | -1140 MB VRAM |
| GpuChunkManager | ⚠️ Частично | Dirty-трекинг есть, paging нет |
| ZeckendorfQuantizer | ⚠️ Изолирован | Есть, не встроен в pipeline |
| TemporalZeckendorf | ✅ | Заменяет exp decay (с оговорками) |
| MorphSTDP | ⚠️ | Создан, но не обучается (пустые vecs) |
| VSAAttention | 🔴 | Баг с array or scalar |
| HDTransformerLayer | ⚠️ | LSH не подключён, O(N²) |
| FormulaCoefficients | ✅ | 92 параметра, но не все модули |
| N-gram pruning | ✅ | ppmi + min_count |
| Semantic lazy harmony | ✅ | cos > 0.95 skip |
| Fibonacci константы | ✅ | γ=1/φ, буферы F19-F22 |

### 9.2 Критические баги (P0)

1. **`crystal_generator.py:889`** — `ValueError: The truth value of an array with more than one element is ambiguous`. Flaky-тест `test_branch_fuzz_zero_temp`. Фикс: `v if v is not None else np.zeros(...)`.
2. **`concept_space.py:2306`** — падение при вызове `_apply_subspace_update_batch` без `codes_t=`. Fallback на `_codes_master_t[cids_t]` разыменовывает `None`.

### 9.3 Архитектурные риски

1. **Рост морфемной памяти:** MorphSTDP без эвикции → OOM при длительном обучении.
2. **HDTransformerLayer O(N²):** без LSH-индекса масштабирование на длинные последовательности (50+ токенов) квадратично.
3. **VRAM spikes:** `_fused_buf` может вырасти до 430 MB при batch=146K.

### 9.4 Рекомендации для V23

1. Исправить P0-баги (VSAAttention, concept_space fallback).
2. Удалить 7 тестов QN-49, проверяющих `_codes_master_t` (dead code).
3. Вынести `TemporalZeckendorf` в синглтон на уровне модуля.
4. Соединить `MorphSTDP.char_vecs` с `CharEnvelope.vecs`.
5. Реализовать `HDTransformerLayer._lsh` (LshIndex из `lsh_index.py`).
6. Провести `grep` хардкоженных чисел в `qwen_knowledge.py`, `syntax_lattice.py`, `morph_vocab.py`.
7. Рассмотреть внедрение методов из раздела 8.
8. Добавить тест на порядок выполнения: GPU→CPU sync before HDTransformerLayer.
9. Задокументировать VRAM лимиты в ARCHITECTURE.md (раздел 5.2).

---

*Аудит выполнен 23 июня 2026 по запросу Architect-AI.*  
*HEAD: 7585dfb, 34 коммита после V21 (935cd9f).*
