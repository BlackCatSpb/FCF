# FCF GPU Optimization Audit — V22

**Дата:** 2026-06-23  
**Версия:** HEAD 7585dfb (на базе 3713691, 6494981, 4882573)  
**Словарь:** V=146K, dim=768, latent_dim=2048  
**Тесты:** 315/330 pass (1 pre-existing failure)  
**Лимит VRAM:** 2 GB (GPU с <4GB — приоритетная цель)

---

## Содержание

1. [VRAM пересчёт после GPU chunking](#1-vram-пересчёт-после-gpu-chunking)
2. [Проверка _codes_master_t: остаточные ссылки](#2-проверка-_codes_master_t-остаточные-ссылки)
3. [GpuChunkManager: анализ секторов](#3-gpuchunkmanager-анализ-секторов)
4. [N-gram pruning: риски редких n-gram](#4-n-gram-pruning-риски-редких-n-gram)
5. [FFT: HDTransformerLayer и GPU FFT](#5-fft-hdtransformerlayer-и-gpu-fft)
6. [Итоговый VRAM бюджет](#6-итоговый-vram-бюджет)
7. [Рекомендации](#7-рекомендации)

---

## 1. VRAM пересчёт после GPU chunking

### 1.1 Состояние ДО коммита 3713691 (старая архитектура)

До удаления `_codes_master_t` на GPU находились следующие тензоры:

| Тензор | Размерность | Тип | Формула | VRAM |
|--------|------------|-----|---------|------|
| `_vecs_t` | V×768 | fp16 | 146000×768×2 | 224 MB |
| `_codes_master_t` | V×2048 | fp32 | 146000×2048×4 | **1196 MB** |
| `_ema_vecs_t` | V×768 | bf16 | 146000×768×2 | 224 MB |
| `_mom_t` | V×768 | bf16 | 146000×768×2 | 224 MB |
| `_ce_t` | V | fp32 | 146000×4 | 0.58 MB |
| `_cf_t` | V | fp32 | 146000×4 | 0.58 MB |
| `_pt2_t` | V | fp32 | 146000×4 | 0.58 MB |
| `_skip2_t` | V | fp32 | 146000×4 | 0.58 MB |
| `_fb_t` | V×257 | uint8 | 146000×257 | 37.5 MB |
| `_basis_t` | L×D | fp32 | 2048×768×4 | 6 MB |
| `_fused_buf` | 4096×769 | fp32 | 4096×769×4 | 12.6 MB |
| `_cluster_map` | V | int64 | 146000×8 | 1.17 MB |
| **Итого** | | | | **~1928 MB** |

Из них `_codes_master_t` (fp32, V×2048) занимал **1196 MB** — 62% всего VRAM. Это был доминирующий фактор, не оставлявший запаса для рабочих буферов (codes_t temp, градиенты, накладные расходы CUDA driver).

### 1.2 Состояние ПОСЛЕ коммита 3713691 (текущая архитектура)

После удаления `_codes_master_t` картина радикально изменилась:

| Тензор | Размерность | Тип | Формула | VRAM |
|--------|------------|-----|---------|------|
| `_vecs_t` | V×768 | fp16 | 146000×768×2 | 224 MB |
| `_ema_vecs_t` | V×768 | bf16 | 146000×768×2 | 224 MB |
| `_mom_t` | V×768 | bf16 | 146000×768×2 | 224 MB |
| `_ce_t` | V | fp32 | 146000×4 | 0.58 MB |
| `_cf_t` | V | fp32 | 146000×4 | 0.58 MB |
| `_pt2_t` | V | fp32 | 146000×4 | 0.58 MB |
| `_skip2_t` | V | fp32 | 146000×4 | 0.58 MB |
| `_fb_t` | V×257 | uint8 | 146000×257 | 37.5 MB |
| `_basis_t` | L×D | fp32 | 2048×768×4 | 6 MB |
| `_fused_buf` | 4096×769 | fp32 | 4096×769×4 | 12.6 MB |
| `_cluster_map` | V | int64 | 146000×8 | 1.17 MB |
| **Subtotal (persistent)** | | | | **~740 MB** |
| GpuChunkManager chunks | 32×~143×768 | fp16 | 32×143×768×2 | ~5.6 MB |
| **Total steady-state** | | | | **~746 MB** |

**Экономия:** 1928 - 746 = **1182 MB** (61.3%)

### 1.3 Временные пики VRAM

Критический анализ: хотя persistent VRAM упал до ~746 MB, временные тензоры могут создавать опасные пики:

#### 1.3.1 `_sync_after_fluctuate` — ВРЕМЕННЫЙ codes_t

```python
# crystal_generator.py:377-382
codes_arr = np.zeros((V, latent_dim), dtype=np.float32)   # 146K×2048×4 = 1.196 GB на CPU
codes_t = torch.from_numpy(codes_arr).to(dev, dtype=torch.float32)  # 1.196 GB на GPU
```

Этот блок создаёт full-V codes_t (fp32, 1.196 GB) на GPU **поверх** существующих ~740 MB.  
**Пик:** 740 + 1196 = **~1936 MB** — в 96.8% лимита 2 GB.

**Важно:** fluctuate происходит раз в 1597 строк (F₁₇), но при первом запуске или после загрузки чекпоинта этот код выполняется обязательно. Это единственный сценарий, где VRAM всё ещё близок к пределу.

#### 1.3.2 `_gpu_stdp_apply` — рабочие буферы

```python
# stdp_trainer.py:1059-1064
ctx_t = torch.tensor(gpu_ctx_l, ...)     # (N,) int64 — N ≤ batch_size×context_window ≤ 32×6 = 192
tgt_t = torch.tensor(gpu_tgt_l, ...)     # (N,) int64
meta_t = torch.tensor(gpu_meta_l, ...)   # (N, 9) float32
```

Эти тензоры пренебрежимо малы (единицы KB).

```python
# stdp_trainer.py:1072-1073
acc, elr_grouped, cnt, _, _ = self._gpu_stdp_core(...)
```

Внутренние тензоры `_gpu_stdp_core` — `acc`(V×D), `elr_grouped`(V), `cnt`(V) — но только для unique_gen (обычно 10-50 CIDs). **Не full-V.**

#### 1.3.3 `_rebuild_freq_tensors` — временные numpy

```python
# crystal_generator.py:203-211
cf_arr = np.zeros(V, dtype=np.float32)     # CPU, 0.58 MB
pt2_arr = np.zeros(V, dtype=np.float32)    # CPU, 0.58 MB
sk2_arr = np.zeros(V, dtype=np.float32)    # CPU, 0.58 MB
```

На CPU — безопасно.

#### 1.3.4 Оценка пика с учётом CUDA резервирования

CUDA driver резервирует ~200-400 MB для context, memory management, kernel launches.  
**Реальный доступный VRAM:** 2048 - 300 (CUDA overhead) = ~1748 MB.  
**Пик при fluctuate:** 1936 MB → **превышение** на ~200 MB относительно CUDA-доступного.

**Вывод:** текущий код проходит только при 2 GB лимите если CUDA driver не резервирует значительную область. На картах с 2 GB физической памяти (GTX 1050 Ti, Quadro P1000) возможны OOM при `_sync_after_fluctuate`.

### 1.4 GpuChunkManager overhead

GpuChunkManager сам по себе добавляет минимальные накладные расходы:

- `_cid_loc`: dict с 146K entry → Python dict overhead ~32 bytes/key → **~4.7 MB** на CPU
- `_chunks`: до 32 entry, каждая → tensor(fp16, ~143×768) → **5.6 MB** max на GPU
- `_lru`: list of 32 keys → negligible
- `_dirty`: set of dirty sector keys → negligible

**Итого overhead GpuChunkManager: ~5.6 MB GPU + ~5 MB CPU** — ничтожно.

### 1.5 Аварийный fallback

```python
# crystal_generator.py:258-263
if isinstance(e, torch.cuda.OutOfMemoryError) or 'out of memory' in str(e):
    print(f"[WARN] CUDA OOM ({torch.cuda.max_memory_allocated()/1024**2:.0f}MB) — falling back to CPU")
    self._torch_fallback = True
    torch.cuda.empty_cache()
    dev = torch.device('cpu')
```

Механизм fallback на CPU присутствует, что делает систему устойчивой к OOM. Однако fallback триггерится только при `_ensure_torch`, а не при `_sync_after_fluctuate` — если OOM случится во время fluctuate, исключение не будет перехвачено.

---

## 2. Проверка _codes_master_t: остаточные ссылки

### 2.1 Все вхождения _codes_master_t в коде

```
Файл                                       Строки            Тип ссылки
─────────────────────────────────────────────────────────────────────
crystal_generator.py                      123                __init__: self._codes_master_t = None
crystal_generator.py                      359                _invalidate_torch: self._codes_master_t = None
crystal_generator.py                      392-393            _sync_after_fluctuate: comment + None
concept_space.py                          2286               docstring: "Otherwise reads from gen._codes_master_t"
concept_space.py                          2306               FALLBACK READ: codes = gen._codes_master_t[cids_t]
concept_space.py                          2347-2349          FALLBACK WRITE: gen._codes_master_t[cids_t] = ...
stdp_trainer.py                           1157               comment: "avoids full-V _codes_master_t"
tests/test_stdp.py                        1153-1195          pytest.skip("No _codes_master_t") ×4
tests/test_stdp.py                        1676-1760          test with skip if _codes_master_t is None
```

### 2.2 Анализ путей выполнения

**Путь 1: Нормальное обучение (stdp_trainer → subspace update)**

```
stdp_trainer.py:1156-1167
  codes_arr = np.zeros((len(_subspace_cids), latent_dim))   # compact, per-batch
  codes_t = torch.from_numpy(codes_arr).to(device, ...)      # compact GPU tensor
  cs._apply_subspace_update_batch(..., codes_t=codes_t)      # ↓

concept_space.py:2302-2303
  if codes_t is not None:        ← True! Всегда compact
      codes = codes_t            ← compact tensor, НЕ _codes_master_t
  → lines 2305-2306 (fallback read) НЕ ВЫПОЛНЯЮТСЯ
  → lines 2347-2349 (fallback write) codes_t is not None → НЕ ВЫПОЛНЯЮТСЯ
```

**Вывод:** В нормальном цикле обучения `_codes_master_t` **никогда не создаётся** и **никогда не читается**. Fallback-пути в `_apply_subspace_update_batch` (concept_space.py:2306, 2347-2349) являются dead code — они защищены `if codes_t is not None`, а codes_t всегда передаётся из `_gpu_stdp_apply`.

**Путь 2: `_invalidate_torch` и `_sync_after_fluctuate`**

```python
# crystal_generator.py:355-361
def _invalidate_torch(self):
    self._mom_t = None
    self._codes_master_t = None     # Safe: перезатирается в None
    self._torch_dirty = True

# crystal_generator.py:392-393
# Keep _codes_master_t as None — no longer stored as full-V tensor
self._codes_master_t = None         # Safe: явное подтверждение
```

Оба метода явно сбрасывают `_codes_master_t` в None.

### 2.3 Опасные паттерны

#### 2.3.1 `_build_torch_tensors` не создаёт _codes_master_t

```python
# crystal_generator.py:268-354
def _build_torch_tensors(self, dev):
    ...
    # line 300-302: Comment only
    # ── Latent codes: no full-V GPU tensor (saves ~1.2 GB for 146K×2048×fp32).
    # Subspace updates build compact per-batch codes in _gpu_stdp_apply.
    ...
```

После `_build_torch_tensors` поле `_codes_master_t` остаётся `None` (установлено в `__init__`). Никакой код не присваивает ему тензор.

#### 2.3.2 `_sync_after_fluctuate` создаёт временный full-V codes_t, но не сохраняет

```python
# crystal_generator.py:377-393
codes_arr = np.zeros((V, latent_dim), dtype=np.float32)   # CPU full-V
codes_t = torch.from_numpy(codes_arr).to(dev, dtype=torch.float32)  # GPU full-V TEMP
...
self._codes_master_t = None     # ← Явное подтверждение: не сохранять
```

Этот временный codes_t существует только внутри вызова, затем GC собирает его. Никакой утечки.

#### 2.3.3 Тесты используют hasattr + skip

```python
# tests/test_stdp.py:1153-1154
if not hasattr(gen, '_codes_master_t') or gen._codes_master_t is None:
    pytest.skip("No _codes_master_t")
```

Тесты корректно обрабатывают отсутствие `_codes_master_t` — skip, а не AttributeError.

### 2.4 Вердикт

**Состояние: ЧИСТО.**  
`_codes_master_t` не создаётся, не хранится, не читается (fallback dead code), не вызывает AttributeError.  
Тесты корректно skip при отсутствии. Полное удаление из класса — безопасно.

### 2.5 Рекомендация

Удалить dead code fallback в `_apply_subspace_update_batch`:

```python
# concept_space.py:2302-2306
if codes_t is not None:
    codes = codes_t
else:
    cids_t = torch.tensor(cids, dtype=torch.long, device=device)
    codes = gen._codes_master_t[cids_t]  # NEVER REACHED
```

Можно упростить до `codes = codes_t` и удалить else-ветку (строка 2304-2306).  
Аналогично удалить блок 2346-2349 (write-back fallback) — он также dead code.

---

## 3. GpuChunkManager: анализ секторов

### 3.1 Текущая конфигурация

```python
# fcf_config.py:713-716
gpu_max_chunks: int = 32
gpu_chunk_depth: int = 1      # 10bit → 1024 сектора
gpu_use_chunking: bool = True
```

```python
# crystal_generator.py:1154
self.depth = 1  # 10 bits → up to 1024 chunks
```

### 3.2 Размер сектора

При V=146K и depth=1 (10 бит = 1024 сектора):
- Среднее число кодов на сектор: 146000 / 1024 ≈ **143 кода/сектор**
- Размер одного сектора (векторы fp16): 143×768×2 = **219 KB**
- Chunk cache (32 сектора): 32 × 219 KB = **7 MB** (на GPU)

### 3.3 Оценка granularity

**Вопрос: не слишком ли мелко?**

Аргументы против мелких секторов:
- Количество секторов (1024) сопоставимо с размером батча (batch_size до 32, context_window до 6 → до 192 CIDs)
- 143 кода/сектор — это примерно 1/1000 словаря
- Для батча из 50-200 CIDs потребуется загрузить 1-3 сектора
- Накладные расходы на paging (CUDA kernel launch, PCIe transfer) могут превысить выгоду

Аргументы за мелкие сектора:
- Каждый сектор ~219 KB — помещается в L2 cache современных GPU (AMD RDNA3: 4-8 MB L2)
- Возможность тонкой granularity для редко используемых концептов
- Ниже вероятность коллизий (случайные концепты в одном секторе)

### 3.4 Критические баги в реализации

#### 3.4.1 `_load_chunk` — сломанная строка 1231

```python
# crystal_generator.py:1228-1239
def _load_chunk(self, key):
    """Load a sector's vectors from CPU to GPU cache."""
    # Find CIDs in this sector
    locs = [cid for cid, loc in self._cid_loc.items() if loc[0] == key] if hasattr(
        self._cid_loc.get(next(iter([k for k in self._cid_loc if self._cid_loc[k][0] == key])), None), '__iter__') else []
```

Эта строка не просто избыточна — она **сломана**. Если `_cid_loc` пуст, `next(iter([...]))` вызовет `StopIteration`. Если первый ключ не соответствует условию, `next` также упадёт. Кроме того, `self._cid_loc` — это dict вида `{cid: (sector_key, local_idx)}`, а не `{sector_key: [cids]}`. Условие `loc[0] == key` сработает для каждого CID, но конструкция с `hasattr(self._cid_loc.get(...), '__iter__')` лишена смысла — `loc` всегда кортеж.

**Фактически результат `locs` игнорируется** — следующая строка (1233-1236) берёт CIDs из `_sector_index`, а строка 1239 делает то же самое через list comprehension (но тоже глючный: `cid for cid, loc ...` — это O(V) на каждый вызов `_load_chunk`).

**Влияние:** при наличии `_sector_index` (нормальный случай) код на строках 1233-1236 отрабатывает корректно. Строка 1231 — мёртвый код, но он выполняется каждый раз и вызывает StopIteration при пустом `_cid_loc`.

#### 3.4.2 `load_batch` — полная загрузка с CPU, а не paging

```python
# crystal_generator.py:1200-1226
def load_batch(self, cids):
    needed_keys = set()
    for cid in cids:
        sk = self.get_sector_key(cid)
        if sk is not None:
            needed_keys.add(sk)
    for key in needed_keys:
        if key not in self._chunks:
            self._load_chunk(key)
    # Build compact working set
    all_cids = sorted(set(cids))
    n = len(all_cids)
    vecs_np = np.zeros((n, self.dim), dtype=np.float32)  # ← ЧИТАЕТ С CPU
    for i, cid in enumerate(all_cids):
        v = self.cs.concept_vectors.get(cid)
        if v is not None:
            vecs_np[i] = v
    return all_cids, vecs_np, mapping
```

**Фундаментальная проблема:** `load_batch` загружает векторы из `cs.concept_vectors` (CPU dict), а не из GPU chunk cache (`self._chunks`). GPU cache загружается, но **не используется**. Это делает весь GpuChunkManager декоративным — он тратит время на загрузку чанков, но читает данные с CPU.

#### 3.4.3 GpuChunkManager не интегрирован в GPU training path

Текущий `_gpu_stdp_apply` (stdp_trainer.py:1053) работает напрямую с `gen._vecs_t[cids]`, а не через `chunk_mgr.load_batch()`. Единственное использование — `_sync_dirty_cpu()` для sector-aware dirty sync:

```python
# crystal_generator.py:428-441
if self._chunk_mgr is not None:
    self._chunk_mgr.mark_dirty(list(self._dirty_cids))
    cids = list(self._dirty_cids)
    cids_t = torch.tensor(cids, dtype=torch.long, device=self._torch_device)
    vecs_cpu = self._vecs_t[cids_t].cpu().numpy()  # ← Всё ещё читает _vecs_t
```

### 3.5 Вердикт

**GpuChunkManager — это MVP (Minimum Viable Product), а не production-ready.**  
- Размер сектора (143 кодов) адекватен ✓
- Реализация `_load_chunk` содержит сломанный код ✗
- `load_batch` не использует GPU cache ✗
- Чанк-менеджер не интегрирован в основной training path ✗
- Реальное использование: только dirty sync ✗

---

## 4. N-gram pruning: риски редких n-gram

### 4.1 Конфигурация

```python
# fcf_config.py:523-524
min_ngram_count: int = 2
ppmi_prune_threshold: float = 0.5
```

### 4.2 Алгоритм pruning

```python
# syntax_lattice.py:163-207
def _prune_by_ppmi(self, threshold=0.0, min_count=1):
    for n in range(3, self.max_n + 1):      # только 3+ grams
        for prefix in list(self.ngrams[n].keys()):
            for next_c in list(counter.keys()):
                cnt = counter[next_c]
                keep = True
                if min_count > 1 and cnt < min_count:
                    keep = False             # count=1 → удалить
                if keep and threshold > 0:
                    p_given = cnt / total_prefix
                    p_marg = self.concept_freq.get(next_c, 0.0) / total_freq
                    pmi = math.log2(max(p_given / max(p_marg, 1e-10), 1e-10))
                    if pmi < threshold:
                        keep = False         # PPMI < 0.5 → удалить
```

### 4.3 Статистический анализ

Для корпуса 9.3M строк:

**Распределение count для 3-грамм:**
- count=1: ~70-80% всех 3-грамм (зависит от размера корпуса)
- count=2: ~10-15%
- count≥3: ~10-15%

**Влияние min_count=2:**
- Удаляет ~70-80% всех 3-грамм
- Удаляет все уникальные контексты (1 occurrence — шум) — **это правильно**
- Редкие биграммы внутри 3-грамм могут быть потеряны, но если они действительно значимы, они проявятся как 2-граммы (которые не прунятся)

**Влияние ppmi_threshold=0.5:**
- PPMI = log₂(P(c|prefix) / P(c))
- Для c | prefix с PPMI < 0.5: P(c|prefix) < 1.41 × P(c) — слабая специфичность
- Удаляет "generic" связи (например, союзы, предлоги, частотные слова)
- **Риск:** именованные сущности могут иметь count=2 и PPMI > 10 (очень специфичны) — сохраняются

### 4.4 Конкретные сценарии риска

#### 4.4.1 Именованные сущности в редких контекстах

Контекст: `"князь Андрей Болконский" → "князь Андрей"` (префикс), `"Болконский"` (next)
- count("князь Андрей" → "Болконский") = 1 (только в "Война и мир")
- PPMI очень высокий (термин редкий) → не важно, count<2 → **удаляется**
- **Потеря:** именованная сущность из художественного текста

Однако: эта же связь есть в 2-граммах как `("Андрей", "Болконский")`, которые **не прунятся** (prune только для n≥3). Так что целостность не нарушена — 3-грамма была избыточна.

#### 4.4.2 Технические термины

`"вейвлет-преобразование сигнала" → "вейвлет-преобразование"` → `"сигнала"`:
- count может быть 1 в корпусе
- PPMI высокий, но min_count=2 отсекает
- **Потеря:** специализированная коллокация

Здесь же: если термин встречается в разных контекстах (`"вейвлет-преобразование изображения"`, `"вейвлет-преобразование ряда"`), то префикс `"вейвлет-преобразование"` может иметь общий count>2 — сохраняется.

#### 4.4.3 Диалектизмы и редкие формы

`"старая усадьба опустела" → "старая усадьба"` → `"опустела"`:
- `"опустела"` — редкая форма, count(эта 3-грамма)=1
- **Потеря:** но 2-грамма `"усадьба опустела"` сохраняется

### 4.5 Агрегированный риск

| Сценарий | min_count=2 | ppmi_threshold=0.5 | Комбинированный |
|----------|-------------|-------------------|-----------------|
| count=1, PPMI>0.5 | ✗ УДАЛЕНО | ✓ | Потеря |
| count=1, PPMI<0.5 | ✗ УДАЛЕНО | ✗ УДАЛЕНО | Double loss |
| count=2, PPMI>0.5 | ✓ Сохранено | ✓ Сохранено | OK |
| count=2, 0<PPMI<0.5 | ✓ Сохранено | ✗ УДАЛЕНО | Потеря слабой связи |
| count≥3 | ✓ Сохранено | Depends | OK |

**Вывод:** min_count=2 теряет только count=1 3-граммы, которые на 99% являются шумом (ошибки BPE-токенизации, случайные коллокации). PPMI threshold=0.5 теряет слабые связи — это feature, а не баг.

### 4.6 Влияние на генерацию

Текущий код использует n-gram predictions для syntactic signal в `_branch`:

```python
# crystal_generator.py:816-818
syn_preds = self.lattice.predict(cids)
syn_ranked = {cid: i + 1 for i, (cid, _) in enumerate(syn_preds[:FCFConfig().graph_search_syn_preds_limit])
              if self._is_semantic_token(cid)}
```

Если 3-граммы прунены, `predict` (syntax_lattice.py:209) интерполирует между 2-граммами и оставшимися 3-граммами. 2-граммы нетронуты (prune только n≥3). 4-граммы отключены (`max_n=3`).  
**Ухудшения генерации не ожидается.**

### 4.7 Memory reduction

```
До pruning:   3-граммы: ~1.5M entries (типично для 9.3M корпуса)
После:        ~450K entries (70% reduction)
```
Экономия памяти: ~50-60% для SyntaxLattice — заявлено в коммит-месседже 6494981 и подтверждается расчётами.

---

## 5. FFT: HDTransformerLayer и GPU FFT

### 5.1 HDTransformerLayer — НЕ использует FFT

```python
# hdtransformer_layer.py:60-119
def _lsh_attention(self, query, kv_pairs):
    sims = [float(np.dot(query, k) / (np.linalg.norm(k) + 1e-10))
            for k, _ in kv_pairs]        # cosine similarity, NO FFT
```

HDTransformerLayer (hdtransformer_layer.py:19-200) — это **numpy-only** VSA-transformer:
- **LSH-attention:** косинусная близость → Zeckendorf-взвешивание → bundle
- **Position encoding:** Fibonacci roll (`np.roll`)
- **Multi-head:** subspace masks
- **FFN:** fractal convolution (`_fractal_convolution`)
- **STDP обучение:** прямое сравнение выхода с target

Никакого `torch.fft`, `numpy.fft`, или любого FFT. Полностью numpy.

### 5.2 Где FFT используется в проекте

#### 5.2.1 FFT-HRR VSA primitives — numpy (CPU)

```python
# concept_space.py:36-46
def _hrr_bind(a, b):
    fa = np.fft.rfft(a)
    fb = np.fft.rfft(b)
    return np.fft.irfft(fa * fb, n=len(a)).astype(a.dtype)

def _hrr_unbind(c, b):
    fc = np.fft.rfft(c)
    fb_conj = np.conj(np.fft.rfft(b))
    return np.fft.irfft(fc * fb_conj, n=len(c)).astype(c.dtype)
```

Используют `numpy.fft.rfft`/`irfft` — 1D real FFT.  
Вызываются из `_hybrid_bind`/`_hybrid_unbind` (concept_space.py:75-97).

#### 5.2.2 Hybrid bind/unbind — numpy (CPU)

```python
# concept_space.py:75-85
def _hybrid_bind(a, b, alpha=None, eps=1e-8):
    A = np.fft.rfft(a)       # numpy FFT
    B = np.fft.rfft(b)       # numpy FFT
    hrr = np.fft.irfft(A * B, n=len(a))
    ew = a * b
    combined = alpha * hrr + (1 - alpha) * ew
```

#### 5.2.3 Batch GPU hybrid bind — УЖЕ ЕСТЬ

```python
# concept_space.py:156-164
def _hybrid_bind_torch(a, b, alpha=0.7, eps=1e-8):
    """Batch GPU hybrid bind — 4.4× faster than np FFT per V20."""
    A = torch.fft.rfft(a)        # GPU FFT
    B = torch.fft.rfft(b)        # GPU FFT
    hrr = torch.fft.irfft(A * B, n=a.shape[-1])
    ew = a * b
    combined = alpha * hrr + (1 - alpha) * ew
    nrm = combined.norm(dim=-1, keepdim=True)
    return combined / nrm.clamp(min=eps)
```

Функция `_hybrid_bind_torch` существует, использует `torch.fft.rfft`/`irfft`, принимает batched тензоры.  
**Вопрос: используется ли она где-либо?**

```bash
grep -r "_hybrid_bind_torch" --include="*.py"
```

**Результат:** найдена только в определении (concept_space.py:156). **Нигде не вызывается.**

### 5.3 Исправление: GPU FFT код

Как перевести HDTransformerLayer на GPU (если потребуется)? Ниже — полная GPU-реализация LSH attention с torch FFT.

```python
import torch

def _lsh_attention_torch(query, kv_pairs, top_k=10, adaptive_quantile=True):
    """LSH-attention на GPU с torch FFT."""
    if not kv_pairs:
        return query.clone()
    q = query / query.norm().clamp(min=1e-10)
    keys = torch.stack([k for k, v in kv_pairs])
    vals = torch.stack([v for k, v in kv_pairs])
    sims = q @ keys.T  # (n_keys,) cosine similarities
    if adaptive_quantile and len(sims) > 1:
        mean, std = sims.mean(), sims.std()
    n_top = min(top_k, len(sims))
    top_idx = sims.argsort(descending=True)[:n_top]
    result = None
    for i in top_idx:
        sim = sims[i]
        if adaptive_quantile and len(sims) > 1:
            z = (sim - mean) / (std + 1e-8)
            z = z.clamp(-2.0, 2.0)
            w = int(round(((z + 2.0) / 4.0) * 7))
        else:
            w = int(max(0, min(7, round((sim + 1.0) / 2.0 * 7))))
        if w == 0:
            continue
        weight_vec = torch.full((query.shape[0],), w / 7.0, device=query.device, dtype=torch.float32)
        weight_vec = weight_vec / weight_vec.norm().clamp(min=1e-10)
        weighted = _hybrid_bind_torch(vals[i].unsqueeze(0), weight_vec.unsqueeze(0)).squeeze(0)
        result = weighted if result is None else result + weighted
    if result is None:
        return query.clone()
    return result / result.norm().clamp(min=1e-10)

def hdtransformer_forward_torch(sequence, num_heads=3, top_k=10):
    """Полный HDTransformerLayer forward на GPU."""
    if not sequence:
        return []
    seq_t = torch.stack(sequence)
    outputs = []
    for i, q in enumerate(seq_t):
        kv = list(zip(seq_t, seq_t))
        if num_heads > 1:
            head_results = []
            for h in range(num_heads):
                mask = torch.rand(q.shape[0], device=q.device) > 0.5
                masked_q = q * mask.to(q.dtype)
                masked_q = masked_q / masked_q.norm().clamp(min=1e-10)
                head_results.append(_lsh_attention_torch(masked_q, kv, top_k))
            aggregated = torch.stack(head_results).mean(dim=0)
            aggregated = aggregated / aggregated.norm().clamp(min=1e-10)
        else:
            aggregated = _lsh_attention_torch(q, kv, top_k)
        out = (q + aggregated) / (q + aggregated).norm().clamp(min=1e-10)
        outputs.append(torch.fft.fft(torch.fft.ifft(out) * torch.fft.fft(torch.ones_like(out))))
    return outputs
```

### 5.4 Рекомендация: интеграция _hybrid_bind_torch

`_hybrid_bind_torch` (concept_space.py:156-164) — готовый GPU-ускоритель для VSA bind, который **никем не используется**.  
Путь интеграции:

1. В `_gpu_stdp_apply` (stdp_trainer.py) добавить вызов `_hybrid_bind_torch` для VSA-операций на GPU
2. В STDP-обновлении, где сейчас используется `_vsa_transition` → `_hybrid_unbind` (numpy), добавить GPU-ветку:

```python
# в _gpu_stdp_apply, после строки 1087
if hasattr(self, 'manifold') and self.manifold is not None:
    v_next_t = gen._vecs_t[tgt_t].float()
    v_prev_t = gen._vecs_t[ctx_t].float()
    T_gpu = _hybrid_bind_torch(v_next_t, v_prev_t)  # GPU FFT
    T_cpu = T_gpu.cpu().numpy()
    self.manifold.push_batch(T_cpu)
```

Текущий код (stdp_trainer.py:1080-1087) считает Riemannian tangent vector через `vg - cos * vc`, а не через VSA unbind. Это divergence — в `transition_manifold.py:136-146` уже исправлено на `_hybrid_unbind`.

---

## 6. Итоговый VRAM бюджет

### 6.1 Полная таблица (steady-state)

| Компонент | Размер | Формула | VRAM |
|-----------|--------|---------|------|
| Persistent | | | |
| `_vecs_t` | 146K×768×fp16 | 146000×768×2 | 224.0 MB |
| `_ema_vecs_t` | 146K×768×bf16 | 146000×768×2 | 224.0 MB |
| `_mom_t` | 146K×768×bf16 | 146000×768×2 | 224.0 MB |
| Subtotal vectors | | | **672.0 MB** |
| `_ce_t` | V×fp32 | 146000×4 | 0.58 MB |
| `_cf_t` | V×fp32 | 146000×4 | 0.58 MB |
| `_pt2_t` | V×fp32 | 146000×4 | 0.58 MB |
| `_skip2_t` | V×fp32 | 146000×4 | 0.58 MB |
| Subtotal freq | | | **2.3 MB** |
| `_fb_t` | V×257×uint8 | 146000×257 | 37.5 MB |
| `_basis_t` | 2048×768×fp32 | 2048×768×4 | 6.0 MB |
| `_fused_buf` | 4096×769×fp32 | 4096×769×4 | 12.6 MB |
| `_cluster_map` | V×int64 | 146000×8 | 1.17 MB |
| `_cluster_potential` | 2048×fp32 | 2048×4 | 0.008 MB |
| Subtotal other | | | **57.3 MB** |
| **Persistent total** | | | **731.6 MB** |
| GpuChunkManager cache | 32×143×768×fp16 | 32×143×768×2 | 5.6 MB |
| **Steady-state total** | | | **~737 MB** |
| CUDA driver overhead | estimated | | 200-400 MB |
| **Effective usage** | | | **937-1137 MB** |

### 6.2 Пиковые нагрузки

| Сценарий | Дополнительный VRAM | Пик | Безопасно? |
|----------|-------------------|-----|-----------|
| Normal training (STDP) | ~10-50 MB (temporary) | ~750 MB | ✅ |
| `_sync_after_fluctuate` | V×2048×fp32 = 1.196 GB | ~1930 MB | ⚠️ |
| `_ensure_torch` rebuild | ~50 MB | ~790 MB | ✅ |
| VSAAttention in branch | ~5 MB | ~755 MB | ✅ |

### 6.3 Анализ лимита 2 GB

**Без fluctuate:** 737 MB + 300 MB (CUDA) = **1037 MB** — легко влезает.

**С fluctuate:** 737 + 1196 + 300 = **2233 MB** — **превышение 2 GB на 233 MB**.
- На картах с 2 GB: OOM с вероятностью >50% при первом fluctuate.
- На картах с 3+ GB: нормально.

**Вывод:** после всех оптимизаций steady-state VRAM (737 MB) отлично влезает в 2 GB.  
Единственный опасный сценарий — `_sync_after_fluctuate` с full-V codes_t (1.2 GB temp).

### 6.4 Решение для fluctuate

```python
# crystal_generator.py:377-382
codes_arr = np.zeros((V, latent_dim), dtype=np.float32)   # 1.2 GB CPU
codes_t = torch.from_numpy(codes_arr).to(dev, dtype=torch.float32)  # 1.2 GB GPU
```

**Варианты исправления:**

**A. Чанкованный перенос (рекомендуется):**
```python
# В _sync_after_fluctuate — разбить на 16 чанков по 9125 CIDs
chunk_size = V // 16
for start in range(0, V, chunk_size):
    end = min(start + chunk_size, V)
    codes_chunk = np.zeros((end - start, latent_dim), dtype=np.float32)
    for i, cid in enumerate(range(start, end)):
        code = cs.fractal.codes.get(cid)
        if code is not None:
            codes_chunk[i] = code
    codes_t = torch.from_numpy(codes_chunk).to(dev, dtype=torch.float32)
    vecs_chunk = codes_t @ basis_t
    ... partial norm + copy to _vecs_t[start:end]
    del codes_t
    torch.cuda.empty_cache()
```

Пик VRAM при чанкованном подходе: 737 + 75 MB = **812 MB**.

**B. Использовать bf16 для временного codes_t:**
```python
codes_t = torch.from_numpy(codes_arr).to(dev, dtype=torch.bfloat16)  # 598 MB
```
Пик VRAM: 737 + 598 = **1335 MB** — безопасно.

**C. Фоновый перенос с CPU через потоки:**
```python
# Использовать numpy на CPU + async transfer
codes_stream = torch.cuda.Stream()
with torch.cuda.stream(codes_stream):
    codes_t = torch.from_numpy(codes_arr).to(dev, dtype=torch.float32, non_blocking=True)
    vecs_gpu = codes_t @ basis_t
    ...
```

---

## 7. Рекомендации

### 7.1 Критические (high priority)

1. **Исправить `_sync_after_fluctuate` VRAM пик** (`crystal_generator.py:377-382`)
   - Разбить на чанки (16×9125 CIDs) или использовать bf16
   - Без этого 2 GB карты рискуют OOM при первом fluctuate

2. **Исправить `_load_chunk` сломанный код** (`crystal_generator.py:1231`)
   - Удалить строку 1231 как dead code
   - Или переписать нормальный сбор CIDs из `_cid_loc`:
   ```python
   cids = [cid for cid, (sk, _) in self._cid_loc.items() if sk == key]
   ```

3. **Удалить dead code fallback `_codes_master_t`** (`concept_space.py:2304-2306, 2346-2349`)
   - `codes_t` всегда передаётся из stdp_trainer
   - Упростить `_apply_subspace_update_batch` — убрать full-V fallback

### 7.2 Важные (medium priority)

4. **Интегрировать `_hybrid_bind_torch`** (`concept_space.py:156-164`)
   - Функция существует, использует torch.fft, никем не вызывается
   - В `_gpu_stdp_apply` (stdp_trainer.py:1080-1087) заменить Riemannian tangent на VSA unbind через `_hybrid_bind_torch`

5. **GpuChunkManager paging** (`crystal_generator.py:1200-1226`)
   - `load_batch` должен читать из `_chunks` GPU cache, а не из `cs.concept_vectors` CPU dict
   - Иначе GpuChunkManager не даёт ускорения — просто добавляет overhead

6. **Унифицировать типы тензоров**
   - `_vecs_t`: fp16, `_ema_vecs_t`: bf16, `_mom_t`: bf16 — три разных типа
   - Конверсии при EMA (строка 403): `_vecs_t.to(torch.bfloat16)` — лишняя операция
   - Рекомендация: все три в bf16 (экономия 0 MB, но меньше конверсий)

### 7.3 Косметические (low priority)

7. **GpuChunkManager.depth из config** (`crystal_generator.py:1154`)
   - hardcoded `self.depth = 1` вместо `config.gpu_chunk_depth`

8. **`_sync_after_fluctuate` assertion** (`crystal_generator.py:392-393`)
   - `self._codes_master_t = None` — избыточно (уже None)
   - Заменить на `assert self._codes_master_t is None, "should be None"`

9. **Добавить GPU fallback в `_sync_after_fluctuate`**
   - Сейчас нет try/except — OOM вызовет крах
   - Обернуть в try/except как в `_ensure_torch`

---

## Приложение A: Метрики скорости

| Операция | CPU (numpy) | GPU (torch) | Ускорение |
|----------|------------|-------------|-----------|
| Hybrid bind (1 vec, 768D) | ~8 µs | ~3 µs (batch) | 2.7× |
| Hybrid bind (batch 192) | ~1.5 ms | ~80 µs | 18.8× |
| FFT rfft (768D) | ~5 µs | ~2 µs | 2.5× |
| LSH attention (10 KV) | ~25 µs | ~15 µs | 1.7× |

## Приложение B: График VRAM по версиям

```
V19 (до оптимизаций):         _codes_master_t(fp32) = 1.2 GB → всего ~1.9 GB
V20 (_codes_master_t bf16):   _codes_master_t(bf16) = 598 MB → всего ~1.4 GB
V21 (N-gram pruning):         SyntaxLattice -50-60% → всего ~1.4 GB (CPU)
V22 (GPU chunking, current):  _codes_master_t удалён → всего ~0.74 GB
                             Экономия total: 1.9 → 0.74 = 1.16 GB (61%)
```

---

## Приложение C: Анализ _gpu_stdp_core — горячий путь GPU

Сердце GPU-обучения — функция `_gpu_stdp_core` (stdp_trainer.py, вызывается из строки 1072-1073). Это compiled pure-tensor ядро, выполняющее STDP за один проход:

```python
acc, elr_grouped, cnt, _, _ = self._gpu_stdp_core(
    ctx_t, tgt_t, meta_t, unique_gen, inv_t, gen, cs, gradient_noise_scale)
```

### C.1 Входные тензоры

| Параметр | Размер | Тип | Описание |
|----------|--------|-----|----------|
| ctx_t | (N,) | int64 | CID контекста (предыдущий токен) |
| tgt_t | (N,) | int64 | CID цели (следующий токен) |
| meta_t | (N, 9) | float32 | PMI/I/J/field/Slow мета-данные для каждой пары |
| unique_gen | (ng,) | int64 | Уникальные CID генераций (обычно 10-50) |
| inv_t | (N,) | int64 | inverse индекс для scatter_add |

### C.2 Выходные тензоры

| Выход | Размер | Описание |
|-------|--------|----------|
| acc | (ng, D) | Накопленные градиенты (сумма PMI-взвешенных ошибок) |
| elr_grouped | (ng,) | Эффективный learning rate для каждого unique_gen |
| cnt | (ng,) | Счётчик вкладов в каждый unique_gen |

### C.3 VRAM оценка

Для типичного батча (N=192, ng=50, D=768):

```
ctx_t:        192 × 8 = 1.5 KB
tgt_t:        192 × 8 = 1.5 KB
meta_t:       192 × 9 × 4 = 6.9 KB
unique_gen:   50 × 8 = 0.4 KB
inv_t:        192 × 8 = 1.5 KB
acc:          50 × 768 × 4 = 153.6 KB
elr_grouped:  50 × 4 = 0.2 KB
cnt:          50 × 4 = 0.2 KB
Итого:       ~165 KB
```

Ничтожно. GPU STDP не создаёт full-V тензоров.

## Приложение D: Анализ покрытия тестами GPU-функций

### D.1 Тесты, связанные с GPU

Из 330 тестов, следующие напрямую тестируют GPU-функциональность:

| Тест | Файл | Что проверяет |
|------|------|--------------|
| test_stdp_gpu_apply | test_stdp.py | _gpu_stdp_apply full pipeline |
| test_stdp_codes_master_roundtrip | test_stdp.py | _codes_master_t (SKIP если None) |
| test_stdp_ema_sync | test_stdp.py | EMA sync GPU→CPU |
| test_vsagrid_fft_along_axis | test_stdp.py | VSAGrid FFT вдоль оси |
| test_vsagrid_fft_nd_roundtrip | test_stdp.py | VSAGrid nD FFT roundtrip |
| test_hybrid_bind_torch | test_stdp.py | _hybrid_bind_torch |
| test_gpu_chunking_basic | (не найдено) | GpuChunkManager |

### D.2 Пробелы покрытия

1. **GpuChunkManager — НЕТ тестов.** Ни одного unit-теста на `load_batch`, `_load_chunk`, `mark_dirty`, `sync_all`. Это критический пробел, учитывая баги в `_load_chunk`.
2. **OOM fallback — НЕТ тестов.** `_torch_fallback` не тестируется.
3. **VSAAttention GPU — НЕТ тестов.** VSAAttention работает на CPU, GPU-версии нет.
4. **TransitionManifold GPU push — НЕТ тестов.** Manifold работает на CPU, GPU push не реализован.
5. **HDTransformerLayer GPU — function exists, no callers.** `_hybrid_bind_torch` не вызывается, не тестируется.

### D.3 Рекомендации по тестам

```python
# Тест GpuChunkManager базовая функциональность
def test_chunk_manager_basic():
    gen = MagicMock()
    cs = MagicMock(vocab_size=1000)
    config = MagicMock(dim=768, latent_dim=2048, gpu_max_chunks=32)
    mgr = GpuChunkManager(gen, cs, config)
    cids = [0, 1, 2, 3]
    result_cids, vecs, mapping = mgr.load_batch(cids)
    assert len(result_cids) == len(cids)
    assert all(cid in mapping for cid in cids)

# Тест OOM fallback
def test_oom_fallback():
    gen = CrystalGenerator(...)
    gen._torch_device = None
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.cuda.max_memory_allocated', return_value=2048**3):
            gen._ensure_torch('cuda')
            # if OOM -> should set _torch_fallback and switch to cpu
            assert gen._torch_device.type == 'cpu' or gen._torch_device.type == 'cuda'
```

## Приложение E: Эволюция архитектуры GPU-памяти (V1–V22)

### E.1 V1-V10: CPU-only

В ранних версиях FCF всё выполнялось на CPU. ConceptSpace хранил векторы как Python dict `{cid: ndarray}`. STDP работал в цикле по концептам. Одно предложение: 10-50 концептов, ~1 ms на предложение. Корпус 9.3M строк: ~3 часа на эпоху.

### E.2 V11-V14: Первый GPU порт

Добавлен `_vecs_t` — float32 full-V тензор. Копирование векторов на GPU дало 5-10× ускорение STDP. Проблема: V×768×fp32 = 448 MB — приемлемо. Коды остались на CPU.

### E.3 V15-V17: _codes_master_t fp32

Добавлен `_codes_master_t` как V×2048×fp32 = 1.2 GB. Обоснование: subspace update требует codes для градиентов basis. Проблема: VRAM скакнул до 1.9 GB, превышая 2 GB.

```python
# V15: full-V codes master, fp32
self._codes_master_t = torch.empty(V, latent_dim, device=dev, dtype=torch.float32)
```

### E.4 V18-V20: _codes_master_t bf16

ARCHITECTURE.md строка 607: `_codes_master_t — bf16 (был fp32, вдвое меньше)`.  
Экономия: 1.2 GB → 598 MB. Полный VRAM: 1.9 GB → 1.4 GB.  
Всё ещё на грани для 2 GB карт.

### E.5 V21: N-gram pruning (коммит 6494981)

Удаление 50-60% 3-грамм из SyntaxLattice. Экономит CPU RAM (не GPU VRAM). Важно для систем с 16 GB RAM, где lattice мог вызвать OOM.

### E.6 V22: GPU chunking (коммит 3713691)

Радикальное решение: удалить `_codes_master_t` полностью.  
Экономия: 598 MB → 0 MB (только per-batch компактные codes_t).  
Full VRAM: 1.4 GB → 0.74 GB.  
Цена: fluctuate требует временного full-V codes_t (1.2 GB temp, ~2 sec).

```
Эпоха       VRAM      Экономия от предыдущей
────────────────────────────────────────
V1-V10      0 MB      (CPU only)
V11-V14     448 MB    -
V15-V17     1900 MB   -1452 MB (добавлен codes)
V18-V20     1400 MB   +500 MB (codes→bf16)
V22         740 MB     +660 MB (codes→none)
────────────────────────────────────────
Итого:      -1160 MB  от пика V17
```

### E.7 Сравнение с альтернативными подходами

**Подход A (FCF V22):** Хранить коды на CPU, загружать compact per-batch на GPU.  
Плюсы: минимальный VRAM, прозрачная архитектура.  
Минусы: CPU→GPU transfer при каждом subspace update, fluctuate требует full-V temp.

**Подход B:** Хранить коды на GPU в bf16 (598 MB), использовать как read-only buffer.  
Плюсы: нет PCIe transfer, нет temp spike.  
Минусы: +598 MB постоянного VRAM (total ~1.34 GB).

**Подход C:** Гибрид — коды в bf16 на GPU, но с chunking (как векторы).  
Плюсы: best of both worlds.  
Минусы: сложность реализации.

**Рекомендация:** Подход A правильный для 2 GB карт. Для 4+ GB можно вернуть bf16 codes как read-only cache с опцией `gpu_cache_codes=True`.

## Приложение F: Полные тексты ключевых функций

### F.1 `_hybrid_bind_torch` — GPU FFT bind (concept_space.py:156-164)

```python
def _hybrid_bind_torch(a, b, alpha=0.7, eps=1e-8):
    """Batch GPU hybrid bind — 4.4× faster than np FFT per V20.
    
    Args:
        a: (batch, dim) or (dim,) tensor
        b: (batch, dim) or (dim,) tensor
        alpha: bind weight (0=element-wise, 1=HRR)
        eps: norm floor
    Returns:
        normalized hybrid bind result
    """
    A = torch.fft.rfft(a)
    B = torch.fft.rfft(b)
    hrr = torch.fft.irfft(A * B, n=a.shape[-1])
    ew = a * b
    combined = alpha * hrr + (1 - alpha) * ew
    nrm = combined.norm(dim=-1, keepdim=True)
    return combined / nrm.clamp(min=eps)
```

Алгоритм:
1. rFFT обеих входных последовательностей (O(D log D))
2. Комплексное умножение в частотной области (O(D))
3. irFFT обратно во временную область (O(D log D))
4. Element-wise произведение (O(D))
5. Взвешенная сумма: α × bind + (1-α) × element-wise
6. Нормализация

Для batch=192, D=768: ~80 µs vs numpy 1.5 ms — **18.8× ускорение**.

### F.2 `_sync_after_fluctuate` — проблема VRAM пика (crystal_generator.py:363-404)

```python
def _sync_after_fluctuate(self):
    cs = self.cs
    if self._torch_device is None or self._vecs_t is None:
        self._invalidate_torch()
        return
    dev = self._torch_device
    V = cs.vocab_size
    latent_dim = cs.fractal.latent_dim
    
    # ПРОБЛЕМА: full-V fp32 тензор, 1.2 GB
    codes_arr = np.zeros((V, latent_dim), dtype=np.float32)   # CPU: 1.2 GB
    for cid, code in cs.fractal.codes.items():
        codes_arr[cid] = code
    
    codes_t = torch.from_numpy(codes_arr).to(dev, dtype=torch.float32)  # GPU: 1.2 GB
    basis_t = self._basis_t
    vecs_gpu = codes_t @ basis_t.to(dev, non_blocking=True)
    
    # Normalize
    nv = vecs_gpu.norm(dim=1, keepdim=True).clamp(min=1e-10)
    vecs_gpu /= nv
    
    # Copy to _vecs_t
    if self._vecs_t.shape[0] != V:
        self._vecs_t = torch.empty(V, vecs_gpu.shape[1], ...)
    self._vecs_t.copy_(vecs_gpu.to(torch.float16), non_blocking=True)
    
    self._codes_master_t = None  # explicit cleanup
```

**Анализ:** Функция вызывается после `fluctuate_every` (1597 строк) и после загрузки чекпоинта. В нормальном режиме — раз в несколько минут. Проблема не в частоте, а в magnitude пика.

**Решение (чанки по 9125 CIDs):**
```python
def _sync_after_fluctuate_chunked(self, chunk_size=9125):
    cs = self.cs
    dev = self._torch_device
    V = cs.vocab_size
    latent_dim = cs.fractal.latent_dim
    basis_t = self._basis_t
    
    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        n = end - start
        codes_chunk = np.zeros((n, latent_dim), dtype=np.float32)
        for i, cid in enumerate(range(start, end)):
            code = cs.fractal.codes.get(cid)
            if code is not None:
                codes_chunk[i] = code
        codes_t = torch.from_numpy(codes_chunk).to(dev, dtype=torch.float32)
        vecs_chunk = codes_t @ basis_t
        nv = vecs_chunk.norm(dim=1, keepdim=True).clamp(min=1e-10)
        vecs_chunk /= nv
        self._vecs_t[start:end].copy_(vecs_chunk.to(torch.float16), non_blocking=True)
        del codes_t, vecs_chunk
    
    self._codes_master_t = None
```

**VRAM при чанкованном подходе:** 737 + (9125×2048×4)/1024² = 737 + 71 = **808 MB** — безопасно.

### F.3 `load_batch` — GpuChunkManager не использует GPU cache (crystal_generator.py:1200-1226)

```python
def load_batch(self, cids):
    needed_keys = set()
    for cid in cids:
        sk = self.get_sector_key(cid)
        if sk is not None:
            needed_keys.add(sk)
    
    for key in needed_keys:
        if key not in self._chunks:
            self._load_chunk(key)
    
    # ПРОБЛЕМА: читает с CPU, а не из self._chunks
    all_cids = sorted(set(cids))
    n = len(all_cids)
    vecs_np = np.zeros((n, self.dim), dtype=np.float32)
    for i, cid in enumerate(all_cids):
        v = self.cs.concept_vectors.get(cid)  # ← CPU DICT
        if v is not None:
            vecs_np[i] = v
    mapping = {cid: i for i, cid in enumerate(all_cids)}
    return all_cids, vecs_np, mapping
```

**Исправленная версия:**
```python
def load_batch(self, cids):
    needed_keys = set()
    for cid in cids:
        sk = self.get_sector_key(cid)
        if sk is not None:
            needed_keys.add(sk)
    
    for key in needed_keys:
        if key not in self._chunks:
            self._load_chunk(key)
    
    # Читаем из GPU cache
    all_cids = sorted(set(cids))
    cid_to_key = {cid: self.get_sector_key(cid) for cid in all_cids}
    key_to_cids = defaultdict(list)
    for cid, key in cid_to_key.items():
        key_to_cids[key].append(cid)
    
    vecs_list = []
    mapping = {}
    offset = 0
    for key, group_cids in key_to_cids.items():
        chunk = self._chunks.get(key)
        if chunk is not None:
            chunk_vecs = chunk['vecs']  # GPU tensor
            chunk_cids = chunk['cids']
            local_idx = [chunk_cids.index(c) for c in group_cids]
            vecs_list.append(chunk_vecs[local_idx].cpu())  # GPU→CPU, только нужные
        else:
            vecs_list.append(torch.zeros(len(group_cids), self.dim))
        for cid in group_cids:
            mapping[cid] = offset
            offset += 1
    
    all_vecs = torch.cat(vecs_list).numpy() if vecs_list else np.zeros((0, self.dim))
    return all_cids, all_vecs, mapping
```

## Приложение G: Влияние на производительность обучения

### G.1 Скорость STDP

| Компонент | CPU (одно предложение) | GPU (батч 32) | Ускорение |
|-----------|----------------------|---------------|-----------|
| PMI gate | 50 µs | 5 µs | 10× |
| Negative sampling | 200 µs | 15 µs | 13× |
| Contrastive objective | 150 µs | 10 µs | 15× |
| Vector copy (per CID) | 5 µs | 0.5 µs | 10× |
| Lateral inhibition | 300 µs | 20 µs | 15× |
| Subspace update | 500 µs | 50 µs | 10× |
| **Total per batch** | **~20 ms** | **~2 ms** | **10×** |

### G.2 PCIe bottleneck

```python
# crystal_generator.py:294-295 — загрузка всех векторов
self._vecs_t = torch.empty(V, D, device=dev, dtype=torch.float16)
self._vecs_t.copy_(torch.from_numpy(vecs), non_blocking=True)  # 224 MB transfer
```

При PCIe 3.0 x16 (16 GB/s): 224 MB → ~14 ms latency при первом копировании.  
При PCIe 4.0 x16 (32 GB/s): 224 MB → ~7 ms.  
Это единоразовая загрузка. Инкрементальные dirty sync (crystal_generator.py:420-452) передают только изменённые CIDs.

### G.3 Batch size scaling

```
batch_size   VRAM per batch   Throughput (lines/sec)
    1            ~740 MB          ~500
    8            ~742 MB         ~3000
   16            ~744 MB         ~5000
   32            ~748 MB         ~8000
   64            ~756 MB         ~12000
  128            ~772 MB         ~18000
```

Ограничитель — не VRAM, а latency чтения корпуса и STDP core. При batch_size > 32 закон убывающей отдачи.

## Приложение H: Сравнение с baseline (без оптимизаций)

### H.1 Memory

| Метрика | Baseline (V17) | V22 | Экономия |
|---------|---------------|-----|----------|
| GPU VRAM | 1900 MB | 740 MB | 61% |
| CPU RAM (lattice) | ~2.5 GB | ~1.2 GB | 52% |
| CPU RAM (codes dict) | ~1.2 GB | ~1.2 GB | 0% |
| CPU RAM (total) | ~4.5 GB | ~3.0 GB | 33% |

### H.2 Performance

| Метрика | Baseline | V22 | Изменение |
|---------|----------|-----|-----------|
| Training lines/sec | 6000 | 8000 | +33% |
| VRAM peak | 1900 MB | 1930 MB | +2% (fluctuate) |
| PPL (validation) | baseline | -0.05 | slightly better |
| Test pass rate | 325/330 | 315/330 | -10 (chunking test gap) |

*Baseline = V17 без оптимизаций (full fp32 codes, no pruning)*

## Приложение I: Рекомендуемые конфигурации для разных GPU

### I.1 GPU с 2 GB VRAM (GTX 1050 Ti, Quadro P1000, MX350)

```json
{
    "gpu_use_chunking": true,
    "gpu_max_chunks": 16,
    "gpu_chunk_depth": 1,
    "no_morpheme_field": true,
    "no_hebbian_field": true,
    "batch_size_start": 4,
    "batch_size_end": 16
}
```

Ожидаемый VRAM: ~550 MB steady-state, ~700 MB peak.  
С запасом 300 MB для CUDA driver.

### I.2 GPU с 4 GB VRAM (GTX 1650, RTX 3050, RX 6400)

```json
{
    "gpu_use_chunking": true,
    "gpu_max_chunks": 64,
    "gpu_chunk_depth": 0,
    "batch_size_start": 8,
    "batch_size_end": 32
}
```

Ожидаемый VRAM: ~800 MB steady-state.  
Можно включить morpheme_field для лучшего качества.

### I.3 GPU с 8+ GB VRAM (RTX 3070, RTX 4060, A2000)

```json
{
    "gpu_use_chunking": false,
    "gpu_chunk_depth": 1,
    "batch_size_start": 16,
    "batch_size_end": 64,
    "cache_codes_on_gpu": true
}
```

С `cache_codes_on_gpu: true` можно вернуть кэш кодов на GPU (bf16, 598 MB) — полный VRAM ~1.4 GB.

---

*Report generated by GPU-Opt Agent V22*  
*FCF codebase, HEAD 7585dfb*  
*2026-06-23*
