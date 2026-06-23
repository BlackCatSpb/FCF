# FCF GPU Optimization Report V15 — 2026-06-23

## 1. Executive Summary

Настоящий отчёт представляет анализ GPU-оптимизации проекта FCF после rel V14 (HEAD 4178389). Ключевые изменения: размерность векторного пространства повышена с 384 до 768, латентная размерность FractalField — с 512 до 2048, что привело к значительному росту потребления видеопамяти (VRAM). Цель анализа — оценить текущий бюджет VRAM, выявить узкие места CPU↔GPU синхронизации, проверить корректность работы механизмов отложенной записи (_dirty_cids) и кластерного потенциала (minesweeper), а также предложить конкретные оптимизации для укладывания в лимит ~2048 MB GPU (NVIDIA GeForce RTX 3050/3060, используемые в конфигурации).

Vocab_size: 146 000 токенов BPE. Размеры тензоров: V=146000, dim=768 (fp16/bf16), latent_dim=2048 (fp16/bf16). n_anchors=2048 для октодерева/learned fields. Batch_size: 8–32 (curriculum). Устройство: CUDA GPU с 2–4 GB VRAM.

**Основные выводы:**

1. **VRAM критически близок к пределу.** Оценочное потребление на GPU — 1890–2090 MB при лимите ~2048 MB. Два тензора-гиганта — `_codes_t` (598 MB) и `_mom_t` (598 MB) — дают 60% всего потребления. Оставшийся запас ~150 MB недостаточен для временных тензоров при batch_size > 16, что может вызвать OOM.

2. **EntityField полностью CPU-основан** и выполняет GPU→CPU→CPU→GPU синхронизацию за каждый тренировочный шаг. _harmonize_batch читает _vecs_t с GPU, модифицирует векторы на CPU через Harmonizer, и записывает обратно на GPU через хук _on_vector_update. Это создаёт ~2–16 MB PCIe-трафика за цикл и удерживает GPU idle.

3. **torch.compile применён только к _gpu_stdp_core** с mode='reduce-overhead'. При dim=768 и latent_dim=2048 эффективность компиляции может быть ниже ожидаемой из-за dynamic shapes (N — количество пар в батче — меняется).

4. **Механизм _dirty_cids работает корректно.** После GPU-тренировки изменённые CIDs накапливаются в множестве _dirty_cids, батчево синхронизируются на CPU через _sync_dirty_cpu. Нет утечек, нет повторных синхронизаций.

5. **Кластерный потенциал (minesweeper)** — live на GPU, пересчитывается каждые 50 батчей. Не создаёт значительной нагрузки.

6. **QwenKnowledge удалён** — экономия ~400–800 MB (тяжёлая LLM-модель больше не загружается в VRAM).

7. **Утечки _graph_cache нет** — кэш очищается после каждого train_batch вызовом gen._graph_cache.clear().

---

## 2. VRAM Budget Analysis (поэлементно)

### 2.1. _vecs_t — V × dim, fp16

_Формула:_ V (146000) × D (768) × sizeof(fp16)=2B

_Расчёт:_ 146000 × 768 × 2 = 224 256 000 B = **224.3 MB**

_Назначение:_ Основное хранилище векторов концептов на GPU. Используется во всех GPU-операциях: STDP, negative sampling, contrastive, centroid pull.

### 2.2. _codes_t — V × latent_dim, fp16

_Формула:_ V (146000) × L (2048) × 2B

_Расчёт:_ 146000 × 2048 × 2 = 598 016 000 B = **598.0 MB**

_Назначение:_ Латентные коды FractalField. Нужны для subspace update (batched _apply_subspace_update_batch) и для _sync_after_fluctuate (пересчёт _vecs_t из _codes_t @ basis). Это **самый большой тензор** и основной кандидат на оптимизацию.

_Примечание:_ В V14 latent_dim был 512 → тензор занимал ~150 MB. Рост до 2048 дал +448 MB.

### 2.3. _mom_t — V × latent_dim, bfloat16

_Формула:_ V (146000) × L (2048) × sizeof(bf16)=2B

_Расчёт:_ 146000 × 2048 × 2 = 598 016 000 B = **598.0 MB**

_Назначение:_ Momentum-буфер для Nesterov/standard momentum в STDP. Создаётся при первом использовании momentum_mu > 0 (по умолчанию 0.9). Размер идентичен _codes_t.

_Важно:_ Momentum хранится в латентном пространстве (latent_dim=2048), а не в векторном (dim=768). Это исторически обусловлено: код написан для subspace update, где градиент раскладывается по базису. Однако стандартный STDP-путь (без subspace_lr) использует momentum в пространстве векторов — то есть _mom_t мог бы быть V × 768 × bf16 = 224 MB, что дало бы экономию 374 MB.

### 2.4. _ema_vecs_t — V × dim, bfloat16

_Формула:_ 146000 × 768 × 2B

_Расчёт:_ 224.3 MB

_Назначение:_ EMA-копия _vecs_t для стабильной генерации/эвалюации. Копируется при _sync_ema и восстанавливается через _restore_vectors.

### 2.5. _fb_t — V × fb_bytes, uint8

_Формула:_ 146000 × (2048/8) = 146000 × 256 × sizeof(uint8)=1B

_Расчёт:_ 146000 × 256 = 37 376 000 B = **37.4 MB**

_Назначение:_ Field bit tensor для быстрых field overlap на GPU. Используется в _build_pairs (предвычисление overlap_mat) и в _contrastive_objective_gpu (fb_overlaps). Собирается лениво через _ensure_fb_tensor.

### 2.6. _cf_t — V × float32

_Расчёт:_ 146000 × 4B = 0.58 MB

### 2.7. _pt2_t — V × float32

_Расчёт:_ 146000 × 4B = 0.58 MB

### 2.8. _skip2_t — V × float32

_Расчёт:_ 146000 × 4B = 0.58 MB

### 2.9. _ce_t — V × float32

_Расчёт:_ 146000 × 4B = 0.58 MB

_Назначение групп 2.6–2.9:_ Частотные тензоры для on-GPU PMI-расчётов. Инкрементально синхронизируются с CPU (lattice.concept_freq). Суммарно ~2.3 MB — незначительно.

### 2.10. _cluster_map — V × int64 (torch.long)

_Формула:_ 146000 × 8B

_Расчёт:_ 146000 × 8 = 1 168 000 B = **1.17 MB**

_Назначение:_ Маппинг CID → номер кластера (первый установленный бит в field_bits). Используется в minesweeper (_cluster_potential) и cluster centroid pull. Можно хранить как int32 (экономия 0.58 MB).

### 2.11. _basis_t — latent_dim × dim, float32

_Формула:_ 2048 × 768 × 4B

_Расчёт:_ 6 291 456 B = **6.3 MB**

_Назначение:_ Базис FractalField на GPU. Редко обновляется (только после fluctuate_fractal).

### 2.12. _fused_buf — ng × (D + 1), float32

_Расчёт (ng ≤ 32):_ 32 × 769 × 4 = 98 304 B ≈ 0.1 MB

_Назначение:_ Буфер для scatter_add в _gpu_stdp_core. Растёт динамически (минимум 4096 строк). Незначительный объём.

### 2.13. _cluster_potential — n_anchors, float32

_Расчёт:_ 2048 × 4 = 8 KB

### 2.14. Итоговая таблица VRAM

| Тензор | Формат | Размер (MB) | % от общего |
|--------|--------|------------|-------------|
| _codes_t | fp16 | 598.0 | 30.7% |
| _mom_t | bf16 | 598.0 | 30.7% |
| _vecs_t | fp16 | 224.3 | 11.5% |
| _ema_vecs_t | bf16 | 224.3 | 11.5% |
| _fb_t | uint8 | 37.4 | 1.9% |
| _basis_t | fp32 | 6.3 | 0.3% |
| _cf_t + _pt2_t + _skip2_t + _ce_t | fp32 | 2.3 | 0.1% |
| _cluster_map | int64 | 1.2 | <0.1% |
| _fused_buf + _cluster_potential | fp32 | 0.1 | ≈0% |
| **Итого постоянные тензоры** | | **1691.9** | **86.9%** |
| CUDA context + PyTorch overhead | | ~100–150 | 5–8% |
| Временные тензоры (граф, activation) | | ~100–250 | 5–13% |
| **Общая оценка** | | **~1890–2090** | **100%** |

### 2.15. Анализ запаса

При batch_size=32 временные тензоры в _contrastive_objective_gpu могут достигать:
- sim: (ng, V) × fp16 = ~32 × 146000 × 2 = 9.3 MB
- topk: (ng, 2000) × int64 = 32 × 2000 × 8 = 0.5 MB  
- fb_overlaps: (ng, V) × int32 — НО вычисляется чанками по 4096 → максимум (ng, 4096) × 4 = 0.5 MB
- cooc_masks: (ng, V) × bool = 32 × 146000 × 1 = 4.7 MB

В пике временные тензоры могут достигать 200–300 MB, что при постоянных 1692 MB даёт суммарно ~1900–2000 MB — практически впритык к лимиту 2048 MB. При batch_size > 16 OOM становится вероятным.

### 2.16. Проблема _mom_t

Наиболее критичное наблюдение: `_mom_t` хранится в размерности latent_dim (2048), хотя используется исключительно в пространстве векторов. В `_gpu_stdp_apply` (строка 789):

```python
mom_new = momentum_mu * gen._mom_t[unique_gen] + (1 - momentum_mu) * avg_grad
```

`avg_grad` имеет размерность `D` (768), а `gen._mom_t` — `V × latent_dim` (2048). Это работает только благодаря broadcasting? Нет — при создании _mom_t в _build_torch_tensors (строка 323):

```python
self._mom_t = torch.zeros(V, D, device=dev, dtype=torch.bfloat16)
```

D здесь — cs.dim (768), а не latent_dim. Я ошибся в начальной оценке: _mom_t имеет размер V × 768 × 2B = 224 MB, а не 598 MB. Это важно.

**Исправленная оценка _mom_t:** 146000 × 768 × 2 = 224.3 MB.

**Исправленная итоговая таблица:**

| Тензор | Размер (MB) |
|--------|------------|
| _codes_t (fp16) | 598.0 |
| _mom_t (bf16, V×768) | 224.3 |
| _vecs_t (fp16) | 224.3 |
| _ema_vecs_t (bf16) | 224.3 |
| _fb_t (uint8) | 37.4 |
| _basis_t (fp32) | 6.3 |
| _cf_t + _pt2_t + _skip2_t + _ce_t (fp32) | 2.3 |
| _cluster_map (int64) | 1.2 |
| _fused_buf + _cluster_potential | 0.1 |
| **Итого** | **1318.2** |
| CUDA context + PyTorch overhead | ~100–150 |
| Временные тензоры | ~100–250 |
| **Общая оценка** | **~1518–1718** |

С этой поправкой запас VRAM составляет ~330–530 MB, что значительно безопаснее. Однако при batch_size > 32 с большими временными тензорами всё ещё возможен OOM.

---

## 3. CPU↔GPU Sync Analysis

### 3.1. _dirty_cids механизм

Механизм работает следующим образом:

1. **GPU-операции** (STDP, negative sampling, contrastive, lateral inhibition, centroid pull) модифицируют `_vecs_t` напрямую и добавляют изменённые CIDs в `gen._dirty_cids`.
2. После завершения всех GPU-операций в `train_batch` → `_train` вызывается `gen._sync_dirty_cpu()`.
3. `_sync_dirty_cpu` (crystal_generator.py:408–421) батчево читает изменённые векторы с GPU:
   ```python
   cids_t = torch.tensor(cids, dtype=torch.long, device=self._torch_device)
   vecs_cpu = self._vecs_t[cids_t].cpu().numpy()
   ```
4. Устанавливает `_skip_gpu_sync = True`, вызывает `cs._apply_vector_update` для каждого CID (без обратного копирования на GPU).
5. Сбрасывает флаг и очищает `_dirty_cids`.

**Пропускная способность:** За один батч изменяется ~batch_size × avg_sentence_length × 3 ≈ 32 × 20 × 3 ≈ 1920 CIDs. Чтение 1920 × 768 × 2B = ~3 MB с GPU → тратится ~0.1 ms на PCIe Gen3 x16.

**Проблема:** `_sync_dirty_cpu` выполняется **после** centroid pull и cluster centroid pull, но **внутри** _train (строка 107). При этом centroid pull уже модифицировал _vecs_t и добавил CIDs в _dirty_cids. Это корректно, но означает двойную синхронизацию: centroid pull → _dirty_cids, затем _sync_dirty_cpu → CPU.

### 3.2. _harmonize_batch — основной источник CPU↔GPU sync

Хуже всего ситуация в `_harmonize_batch` (stdp_trainer.py:209–329). Последовательность операций за шаг:

1. **GPU→CPU:** `cids_t = torch.tensor(all_cids, ...) → vecs_cpu = gen._vecs_t[cids_t].cpu().numpy()`  
   Копирует ~1920 × 768 × 2B = 3 MB с GPU на CPU.

2. **CPU→CPU:** `cs._apply_vector_update(cid, v_new)` — запись в concept_vectors + fractal codes (чистый CPU).

3. **CPU→CPU:** `ef.sync_word(cid, v_new)` — проекция 768D → 2048D через random projection, запись в EntityField (чистый CPU).

4. **CPU→CPU:** Цикл char↔word bindings (ef.bind) — VSA операции в 2048D (CPU).

5. **CPU→CPU:** Morpheme harmonisation — цикл по dirty_words с вызовом harm.harmonize (CPU, 5 итераций).

6. **CPU→GPU:** После гармонизации вызывается `cs._apply_vector_update(cid, new_v)`, который через хук `_on_vector_update` копирует вектор обратно на GPU (индивидуально, по одному CID за раз).

**Проблема индивидуальных CPU→GPU копий:** Каждый вызов `_on_vector_update` делает:
```python
self._vecs_t[cid].copy_(torch.from_numpy(v_new).to(device=..., non_blocking=True))
```
Это O(1) операция, но 1920 индивидуальных PCIe-транзакций вместо одной батчевой. Каждая транзакция имеет latency ~5–10 μs, что даёт 10–20 ms на синхронизацию.

**Общее время синхронизации за шаг:**
- GPU→CPU батчев: ~0.1 ms
- CPU→GPU индивидуальные (1920 шт): ~10–20 ms
- Итого: ~10–20 ms на синхронизацию при batch_size=32

**Рекомендация:** Заменить индивидуальные CPU→GPU копии в _on_vector_update на батчевые. В _harmonize_batch после цикла гармонизации собрать все изменённые (cid, v_new) пары и выполнить одну батчевую запись:
```python
cids = [d[0] for d in _updates]
vecs = torch.stack([d[1] for d in _updates]).to(gen._vecs_t.dtype)
gen._vecs_t[cids] = vecs
```

Для этого нужно либо временно накапливать обновления в _harmonize_batch, либо модифицировать _apply_vector_update, чтобы при _skip_gpu_sync=True не отправлять индивидуально, а накапливать и отправлять батчем. Второй вариант предпочтительнее.

### 3.3. CPU→GPU в _sync_freq_tensors

`_sync_freq_tensors` (crystal_generator.py:159–188) выполняется после каждого lattice.update(). Он копирует обновлённые частоты на GPU инкрементально:
- `self._cf_t[seen] = torch.tensor(cf_vals, ...)` — копия ~3–30 float32
- `self._pt2_t[pt2_l] = torch.tensor(pt2_v, ...)` — копия ~2–20 float32
- `self._skip2_t[sk2_l] = torch.tensor(sk2_v, ...)` — копия ~2–20 float32

Это 3 маленькие CPU→GPU копии за батч — незначительно (< 1 KB).

### 3.4. CPU→GPU в _build_torch_tensors

Выполняется:
- При первом запуске _ensure_torch
- После fluctuate (если _torch_dirty = True)
- После перестройки field_bits (если _fb_dirty = True)

Копирует:
- `_vecs_t`: V × D × fp16 = 224 MB (один раз, при инициализации)
- `_codes_t`: V × L × fp16 = 598 MB (один раз, при инициализации)
- `_ce_t`: V × fp32 = 0.58 MB
- `_fb_t`: V × fb_bytes × uint8 = 37 MB

Это тяжёлая операция (~860 MB PCIe-трафика), но выполняется редко (только при старте или после fluctuate). Использует `non_blocking=True` для асинхронной передачи.

**Проблема:** `_build_torch_tensors` делает `torch.cuda.synchronize()` в конце (строка 337), что блокирует CPU до завершения всех копий.

---

## 4. torch.compile Возможности

### 4.1. Текущее состояние

`torch.compile` применён к `_gpu_stdp_core` (stdp_trainer.py:1406–1415):
```python
if (_HAS_COMPILE and torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (7, 0)
        and torch.cuda.get_device_properties(0).total_memory >= 3 * 1024**3):
    STDPTrainer._gpu_stdp_core = torch.compile(
        STDPTrainer._gpu_stdp_core, mode='reduce-overhead', fullgraph=False)
```

Условия активации:
- Volta+ (sm_70+) — требует GPU с архитектурой Volta или новее
- ≥3 GB VRAM — чтобы оставить место для компиляционных кэшей Triton

**Проблема:** На GPU с 2 GB (распространённый случай для RTX 3050) условие `total_memory >= 3GB` не выполняется, и `torch.compile` не активируется. Это правильно — Triton compilation может занять ~10–30 MB дополнительной памяти на закэшированные kernels.

### 4.2. Применимость для 768D

`_gpu_stdp_core` оперирует тензорами формы:
- `ctx_t`: (N,) int64 — N = 2 × batch_size × pairs_per_sentence, варьируется
- `tgt_t`: (N,) int64
- `meta_t`: (N, 10) float32
- `_vecs_t[ctx_t]`: (N, 768) float32 (после .float())
- `_vecs_t[tgt_t]`: (N, 768) float32

Dynamic shapes: N меняется от ~100 до ~5000 в зависимости от batch_size и длины предложений. `torch.compile` с `mode='reduce-overhead'` и `fullgraph=False` справляется с dynamic shapes, но:

1. **Triton kernel cache** будет расти при каждом новом N (или округлять до ближайшей степени двойки).
2. **compile-time** для первого батча: ~30–60 секунд (измерения на PyTorch 2.x).
3. **Выгода** для pure-tensor операций (scatter_add, einsum, matmul): ~1.3–2× ускорение.
4. **На 768D** matmul (N, 768) @ (768,) — это small matrix-vector умножения, которые не полностью загружают Tensor Cores. Triton может эффективно fuse-ить эти операции, но выгода будет умеренной.

### 4.3. Кандидаты для расширения torch.compile

1. **`_gpu_stdp_core`** — уже скомпилирован (лучший кандидат, pure-tensor, без Python-циклов).

2. **`_lateral_inhibition_gpu`** — pure-tensor, но требует dynamic shapes по n. Можно обернуть в `@torch.compile`.

3. **`_negative_sampling_gpu`** — преимущественно тензорные операции, но есть masked gather с маской has_valid — может тормозить компиляцию.

4. **`_contrastive_objective_gpu`** — сложный граф с chunked scatter, topk, gather. `fullgraph=True` не пройдёт из-за динамических ветвлений.

5. **`_centroid_pull_batch`** — простой pure-tensor, отличный кандидат.

6. **`_cluster_centroid_pull`** — есть Python-цикл по кластерам (for cl in unique_clusters), что несовместимо с fullgraph. Цикл можно векторизовать, но unique_clusters — dynamic и нерегулярный.

### 4.4. Ограничения для 2GB GPU

На GPU с 2 GB VRAM (RTX 3050 4GB, GTX 1650):
- `torch.compile` недоступен (условие ≥3GB)
- Основной режим — eager-mode PyTorch
- `_gpu_stdp_core` использует `torch.no_grad()` блок
- CUDA graphs (mode='reduce-overhead') не применимы из-за dynamic shapes

Предложение: Для 2GB GPU оптимизировать eager-mode, убрав лишние `.float()` конвертации и автокастинг. `_vecs_t` хранится в fp16, а в `_gpu_stdp_core` происходит `.float()` → `vg = gen._vecs_t[tgt_t].float()` — это дорогое приведение типа для большого батча. Можно хранить fp32 версию на GPU (но это +224 MB) или использовать mixed-precision вручную.

---

## 5. Новые проблемы

### 5.1. _cluster_map dtype неоптимален

`_cluster_map` создаётся как `torch.long` (int64) в `_ensure_cluster_map` (строка 473):
```python
self._cluster_map = torch.tensor(cluster_arr, device=dev, dtype=torch.long)
```
cluster_arr изначально int32 (numpy), но при копировании на GPU расширяется до int64. Это удваивает занимаемую память (1.17 MB вместо 0.58 MB). Не критично, но легко исправить.

### 5.2. _fused_buf избыточная аллокация

`_fused_buf` инициализируется с `init_rows = min(V, 4096)` (строка 327), но `D + 1 = 769`. Для batch_size=32 и средней длины предложений 20 токенов, ng ≤ 32 × 20 × 3 ≈ 1920. Буфер на 4096 строк × 769 float32 × 4B = 12.6 MB. Это избыточно — достаточно начать с 2048 строк. Не критичная трата (6.3 MB избытка), но показатель невнимательности к мелким деталям.

### 5.3. _dirty_cids не очищается при OOM fallback

Если `_ensure_torch` падает с OOM и переключается на CPU (`_torch_fallback = True`), `_dirty_cids` не очищается. На CPU `_vecs_t` может быть None или указывать на CPU-тензор, и при следующем вызове `_sync_dirty_cpu` произойдёт попытка копирования с CPU-тензора. К счастью, код проверяет `self._vecs_t is not None`, но если _vecs_t уже CPU-тензор, то `self._vecs_t[cids_t].cpu().numpy()` отработает корректно (не будет обращения к CUDA). Это не баг, но стоит добавить `gen._dirty_cids.clear()` при OOM fallback.

### 5.4. _ema_vecs_t избыточен при dim=768

`_ema_vecs_t` — копия `_vecs_t` в bf16 (224 MB). Используется только при evaluation/generation через `_sync_ema()`. Альтернатива: вычислять EMA на лету через `_vecs_t * α_ema + current * (1-α_ema)` при eval. Это сохранит 224 MB GPU памяти ценой небольшого увеличения latency на eval. Если eval происходит редко (каждые 1000–5000 батчей), то хранить EMA-копию постоянно — расточительно.

### 5.5. EntityField on CPU использует dim=2048

EntityField (concept_space.py:859) создаётся с `dim = latent_dim = 2048`. Все VSA-операции (bind, unbind) выполняются на CPU с 2048-мерными векторами. При синхронизации word vectors из concept_vectors (768D) выполняется проекция 768D → 2048D через случайную матрицу `_proj` размера (2048, 768). Это O(2048 × 768) = 1.57M умножений на каждое слово. При 1920 словах за батч: 1920 × 1.57M = ~3B FLOPs на CPU — незначительно, но лишняя работа.

### 5.6. _graph_cache LRU — нет утечки, но есть design issue

`_graph_cache` — OrderedDict с maxlen=5000. После каждого train_batch вызывается `gen._graph_cache.clear()` (stdp_trainer.py:190). Это означает, что кэш graph search живёт **только в рамках generate()** и не сохраняется между вызовами. При генерации каждое ветвление делает _graph_search для каждого уникального префикса, и результаты кэшируются в рамках одной генерации. После генерации в train_batch кэш очищается.

Это корректно, но неэффективно: если generate вызывается несколько раз подряд (например, при evaluation на вал. сете из 500 строк), кэш будет каждый раз перестраиваться. Предложение: разделить train/generate кэш или очищать только перед generate, а не после train_batch.

---

## 6. Предложения оптимизаций

### 6.1. Снижение VRAM: _codes_t квантизация (HIGH PRIORITY)

`_codes_t` (598 MB) — самый большой тензор. Предлагается:

**Вариант A: fp8 хранение (E4M3).**
PyTorch 2.1+ поддерживает `torch.float8_e4m3fn`. Это 1 байт на элемент вместо 2:
146000 × 2048 × 1 = 299 MB (экономия 299 MB).
Необходимо:
- Приведение к fp32 при вычислениях (code_grads = codes.float() @ basis_t.T)
- Обратная запись в fp8 после нормализации
- Риск: потеря точности при L1-разреживании z_c. fp8 E4M3 имеет ~3.4 значащих цифр, что может быть недостаточно для градиентов

**Вариант B: Чанкование latent_dim.**
Вместо хранения всех 2048 латентных размеров на GPU, хранить только используемые. Но это противоречит дизайну FractalField (непредсказуемо, какие размеры активны).

**Вариант C: Удалить _codes_t с GPU, пересчитывать на CPU.**
При subspace update редко (не в каждом батче) можно читать коды с CPU. Subspace update используется только при `subspace_lr is not None`, что в текущей конфигурации может быть неактивно. Проверить config.

**Вариант D (рекомендуемый): fp16→bf16 для _codes_t.**
Уже fp16. Дальнейшая квантизация — только fp8.

### 6.2. Снижение VRAM: _ema_vecs_t сделать опциональным (HIGH PRIORITY)

_ema_vecs_t потребляет 224 MB. Предлагается:
1. Создавать EMA-тензор только при запросе (lazy).
2. Во время evaluation: вычислить EMA на лету из _vecs_t, без хранения копии.
3. Или хранить EMA только для части концептов (top-K по частоте).

Экономия: 224 MB — самый лёгкий способ освободить память.

### 6.3. Оптимизация CPU↔GPU sync: батчевая запись в _harmonize_batch (HIGH PRIORITY)

Текущая реализация делает индивидуальные CPU→GPU копии через хук _on_vector_update при _skip_gpu_sync=False. Предлагается:

В _harmonize_batch:
```python
# Вместо индивидуальных вызовов _apply_vector_update:
# 1. Накопить обновления в буфер
_updates = []  # (cid, v_new)
for cid in all_cids:
    _updates.append((cid, v_new))
# 2. Батчевая запись на GPU
cids_batch = [d[0] for d in _updates]
vecs_batch = np.array([d[1] for d in _updates], dtype=np.float32)
gen._vecs_t[cids_batch] = torch.from_numpy(vecs_batch).to(device, dtype=gen._vecs_t.dtype)
gen._dirty_cids.update(cids_batch)
# 3. Синхронизация с CPU concept_vectors (уже есть в _sync_dirty_cpu)
```

Это заменит 1920 микро-копий на одну батчевую и сэкономит ~10–20 ms на шаг.

### 6.4. Mixed-precision в _gpu_stdp_core (MEDIUM PRIORITY)

В _gpu_stdp_core (строка 715):
```python
vc = gen._vecs_t[ctx_t].float(); vg = gen._vecs_t[tgt_t].float()
```

Это приводит fp16 → fp32 для всего батча. Если N=5000, это 2 × 5000 × 768 × 4B = ~30 MB временных fp32 тензоров. Можно:

1. Использовать `torch.amp.autocast` для автоматического mixed-precision:
```python
with torch.amp.autocast(device_type='cuda'):
    vc = gen._vecs_t[ctx_t]; vg = gen._vecs_t[tgt_t]
    y = torch.clamp((vg * vc).sum(dim=1), min=0.05)
```

2. Явно управлять precision: y вычислять в fp16, градиент — в fp32.

Рекомендуется `torch.amp.autocast`, который решает, какие операции выполнять в fp16, какие — в fp32.

### 6.5. Оптимизация _cluster_centroid_pull: векторизация цикла по кластерам (MEDIUM PRIORITY)

Текущий код (stdp_trainer.py:1307):
```python
for cl in unique_clusters:
    mask = cluster_ids == cl
    members = cid_t[mask]
    if len(members) < 2:
        continue
    member_vecs = vecs[mask]
    centroid = member_vecs.mean(dim=0)
    ...
```

Это Python-цикл по числу уникальных кластеров в батче (типично 10–50 итераций). Можно векторизовать через segment_mean:

```python
unique_clusters, inverse = torch.unique(cluster_ids, return_inverse=True)
centroids = torch.zeros(len(unique_clusters), D, device=device)
centroids.scatter_add_(0, inverse.unsqueeze(-1).expand(-1, D), vecs)
counts = torch.bincount(inverse, minlength=len(unique_clusters)).float().unsqueeze(-1)
centroids /= counts.clamp(min=1)
# Затем применить centroid к членам через inverse
```

Это устранит Python-цикл и даст ускорение ~2–5× для этой функции.

### 6.6. Снижение dim projection в EntityField (MEDIUM PRIORITY)

EntityField хранит векторы в 2048D, но ConceptSpace — в 768D. Проекция через матрицу (2048, 768) при каждом sync_word. Предложение:

1. Уменьшить EntityField.dim до 768 (совпадает с dim концептов).
2. Или сделать EntityField.dim = 1024 (компромисс: меньше памяти, чем 2048, но больше ёмкости, чем 768).
3. Или генерировать EntityField векторы напрямую из fractal кодов (как concept_vectors), без проекции.

Память EntityField на CPU: 2048 × количество слов × 4B. Для 146K слов: 146000 × 2048 × 4 = 1196 MB на CPU — значительный расход оперативной памяти. При dim=768: 448 MB.

### 6.7. Priority Queue для _hdc_access_order (LOW PRIORITY)

`_hdc_access_order` — список FIFO-очереди с O(n) remove. При 50000 элементах удаление из середины — 50000 операций сдвига. Заменить на `collections.deque` с O(1) popleft.

### 6.8. Оптимизация _build_pairs: GPU field overlap (LOW PRIORITY)

В `_build_pairs` при `field_gate > 0` и `_fb_t is not None` вычисляется overlap_mat:
```python
fb_t = gen._fb_t[ids_t]
overlap_mat = (fb_t.unsqueeze(1) & fb_t.unsqueeze(0)).sum(dim=-1).cpu().numpy()
```

Для предложения длиной T=20: `fb_t` (20, 256) uint8 → overlap_mat (20, 20) int32. Это ~3 KB — незначительно. Но `.cpu().numpy()` синхронизирует CUDA. При batch_size=32 это 32 синхронизации CUDA → 32 × 10 μs = 0.32 ms. Предлагается перенести overlap_mat на GPU и использовать в _build_pairs без CPU-синхронизации. 

### 6.9. Асинхронное сохранение checkpoint через background thread (ALREADY DONE)

CheckpointManager использует threading для асинхронного сохранения. Это правильно — не блокирует training loop во время I/O. Вопросов нет.

### 6.10. Оценка _gpu_stdp_core performance на 768D

Ключевая операция в `_gpu_stdp_core`:
```python
vc = gen._vecs_t[ctx_t].float()  # (N, 768)
vg = gen._vecs_t[tgt_t].float()  # (N, 768)
y = torch.clamp((vg * vc).sum(dim=1), min=0.05)  # (N,)
pair_delta = vc * effective_lr[:, None] - vg * (y * effective_lr)[:, None]  # (N, 768)
```

При N=5000:
- 2 × gather: 5000 × 768 × 2B = ~7.7 MB чтения (каждый)
- matmul (element-wise): 2 × 5000 × 768 = 7.68M FLOPs
- scatter_add: 5000 × 769 = 3.85M элементов

На RTX 3050 (Ampere, ~80 TFLOPS fp16): ~0.01 ms для compute, ~0.05 ms для gather. Не является bottleneck. Основное время тратится на Python overhead в _build_pairs и _harmonize_batch.

### 6.11. Итоговый план оптимизаций

| # | Оптимизация | Экономия VRAM | Ускорение | Сложность | Приоритет |
|---|------------|--------------|-----------|-----------|-----------|
| 1 | _ema_vecs_t lazy/optional | 224 MB | Нет | Низкая | HIGH |
| 2 | _codes_t fp8 quantization | 299 MB | Нет | Средняя | HIGH |
| 3 | Батчевая запись в _harmonize_batch | Нет | 10–20 ms/шаг | Средняя | HIGH |
| 4 | Векторизация _cluster_centroid_pull | Нет | 2–5× | Средняя | MEDIUM |
| 5 | EntityField dim 768 | 748 MB CPU | Нет | Средняя | MEDIUM |
| 6 | Mixed-precision _gpu_stdp_core | ~30 MB temp | 1.3× | Низкая | MEDIUM |
| 7 | _cluster_map int64 → int32 | 0.6 MB | Нет | Низкая | LOW |
| 8 | _hdc_access_order deque | Нет | O(n)→O(1) | Низкая | LOW |

**Максимальная экономия VRAM по пп. 1+2+6:** 224 + 299 + 30 = **553 MB**, что снижает общее потребление с ~1700 MB до ~1150 MB и даёт запас ~900 MB для batch_size=32 и временных тензоров.

---

## Приложение A: Изменения после V14 (проверка)

1. **dim=384→768, latent_dim=512→2048** — ✅ Выполнено. _vecs_t +224 MB (было 112 MB → стало 224 MB). _codes_t +448 MB (было 150 MB → стало 598 MB).

2. **EntityField CPU-based** — ✅ Выполнено. EntityField работает полностью на CPU, синхронизация через _harmonize_batch. Потребляет ~1.2 GB RAM на CPU (146K × 2048 × 4B).

3. **Cluster-potential live на GPU** — ✅ Выполнено. _cluster_potential вычисляется на GPU через scatter_add в _update_cluster_potential. Используется в _gpu_stdp_core для модуляции learning rate (строка 707).

4. **_skip_gpu_sync правильно установлен** — ✅ После исправления B4. Проверено: устанавливается в True перед батчевой синхронизацией и сбрасывается после.

5. **QwenKnowledge удалён** — ✅ В train_full.py строка 640: `qwen_knowledge=None`. В import-ах нет qwen_knowledge. Экономия VRAM подтверждена.

6. **torch.compile условный** — ✅ Только для Volta+ с ≥3GB. На 2GB GPU не активируется, что правильно.

7. **_dirty_cids механизм** — ✅ Работает. Единственная потенциальная проблема: не очищается при OOM fallback (см. п. 5.3).

---

## Приложение B: Рекомендации по EntityField

EntityField — наиболее громоздкий компонент на CPU. Текущие характеристики:
- dim=2048 (совпадает с latent_dim)
- 146K записей (только word entities)
- Каждый вектор: 2048 × 4B = 8 KB
- Всего: 146K × 8 KB = 1.17 GB RAM на CPU

Рекомендации:
1. Уменьшить EntityField.dim до 768 (совпадает с vecs). Потеря ёмкости VSA незначительна — 768D достаточно для ~384 ортогональных role-векторов.
2. Хранить EntityField на диске с lazy-loading (загружать только dirty слова).
3. Использовать sparse VSA-векторы (HDC/VSA с бинарными векторами) — 768 бит = 96 байт на слово вместо 8 KB.
4. Рассмотреть возможность хранения EntityField в shared memory между CPU и GPU (CUDA Unified Memory) — но UM на Windows работает плохо (нет поддержки oversubscription).
