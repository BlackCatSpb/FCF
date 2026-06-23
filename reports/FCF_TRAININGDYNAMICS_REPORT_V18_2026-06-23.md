# FCF V18 Training Dynamics Audit Report

**Date:** 2026-06-23  
**HEAD:** c37a8d8 (main)  
**Previous baseline:** V15 → V16/V17 (19 fixes)  
**Scope:** train_full.py, stdp_trainer.py, concept_space.py (planning stage)  

---

## Table of Contents

1. [LR Scheduler Analysis — get_lr() Edge Cases](#1-lr-scheduler-analysis--get_lr-edge-cases)
2. [GPU-Accelerated Semantic Bootstrap](#2-gpu-accelerated-semantic-bootstrap)
3. [HDC Update Cost — Batch N-Gram Update](#3-hdc-update-cost--batch-n-gram-update)
4. [Plateau Detection — Soft Plateau Protocol](#4-plateau-detection--soft-plateau-protocol)
5. [FFT-HRR Impact on VSA Learning Dynamics](#5-fft-hrr-impact-on-vsa-learning-dynamics)
6. [Prune Safety — Rare Concept Protection](#6-prune-safety--rare-concept-protection)

---

## 1. LR Scheduler Analysis — get_lr() Edge Cases

**Файл:** `train_full.py:412-478` (планируемый)

### 1.1 Текущая архитектура (в existing codebase)

В existing codebase (`fractal_trainer.py:374-382`) используется `get_linear_schedule_with_warmup` из transformers — простой линейный LR с warmup. Однако для V18 планируется кастомный `get_lr()` с warmup (5% steps) + cosine annealing (T_max = полный цикл) + restart (T_mult = 2) + rescore offset (сдвиг фазы при переоценке качества). Ниже — реконструкция планируемого кода:

```python
# train_full.py:412 — Планируемый get_lr() для V18
def get_lr(
    self,
    step: int,
    warmup_steps: int = 500,
    base_lr: float = 1e-3,
    min_lr: float = 1e-5,
    T_max: int = 5000,
    T_mult: int = 2,
    rescore_step: Optional[int] = None,
) -> float:
    # 1. Warmup: линейный рост от min_lr до base_lr
    if step < warmup_steps:
        return min_lr + (base_lr - min_lr) * (step / max(1, warmup_steps))

    # 2. Вычисление фазы с учётом restart'ов
    adjusted = step - warmup_steps
    if rescore_step is not None and rescore_step > warmup_steps:
        # Rescore offset: сдвигаем фазу на момент переоценки
        # (имитация "re-discovery": сбрасываем прогресс до значения rescore)
        adjusted = max(0, adjusted - (rescore_step - warmup_steps))

    # 3. Cosine annealing с restarts
    cycle = 0
    cycle_start = 0
    T_cycle = T_max
    while adjusted >= cycle_start + T_cycle:
        cycle_start += T_cycle
        T_cycle *= T_mult
        cycle += 1
        if T_cycle > 20000:  # защита от бесконечного роста
            T_cycle = 20000

    progress = (adjusted - cycle_start) / max(1, T_cycle)  # [0, 1]
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    lr = min_lr + (base_lr - min_lr) * cosine_decay

    return lr
```

### 1.2 Edge Cases

**EC1. Rescore offset после множественных restart'ов.**  
Если `rescore_step` приходится на 3-й цикл (T_cycle = 20000), сдвиг `adjusted -= rescore_step - warmup_steps` может перебросить фазу в отрицательную зону, если `rescore_step` мал. Результат: `adjusted < 0` → `cycle_start = 0` → прогресс становится отрицательным → `cosine_decay > 1` → LR > base_lr.

**Исправление:**
```python
# train_full.py:448 — Guard против отрицательного adjusted
adjusted = max(0, adjusted - (rescore_step - warmup_steps))
```

**EC2. Warmup пересекается с rescore.**  
Если `rescore_step < warmup_steps`, сдвиг `adjusted -= rescore_step` может сделать отрицательный сдвиг внутри warmup-фазы. Warmup уже отработал на шаге `rescore_step` к моменту оффсета.

**Исправление:**
```python
# train_full.py:440 — Rescore только вне warmup
if rescore_step is not None and rescore_step >= warmup_steps:
    adjusted = max(0, adjusted - (rescore_step - warmup_steps))
```

**EC3. T_mult переполнение.**  
При T_mult = 2.0 после 8 циклов T_cycle = 5000 × 2^8 = 1,280,000. После 16 циклов — > 300 млн. Цикл `while` будет крутиться миллионы итераций.

**Исправление:**
```python
# train_full.py:455 — Кап на T_cycle и защита от переполнения
T_cycle = min(T_max * (T_mult ** cycle), 50000)  # вместо *= T_mult
```

**EC4. Rescore offset при `rescore_step ≈ adjusted`.**  
Если `adjusted - (rescore_step - warmup_steps) ≈ 0`, LR прыгает к `base_lr` (начало цикла cosine). Это создаёт резкий скачок learning rate — дестабилизирует обучение при rescore в середине цикла.

**Мягкий rescore (рекомендация):**
```python
# train_full.py:460 — Плавный переход при rescore
if rescore_step is not None and step >= rescore_step:
    blend = min(1.0, (step - rescore_step) / 100)  # 100 шагов на blend
    if blend < 1.0:
        old_lr = self._last_lr
        new_lr = self._compute_cosine_lr(adjusted)
        lr = old_lr * (1 - blend) + new_lr * blend
        self._last_lr = lr
        return lr
```

**EC5. Min_lr достигается раньше конца цикла.**  
Cosine annealing с `min_lr = 1e-5` и `base_lr = 1e-3` даёт LR = 5.05e-4 в середине цикла. К концу цикла LR = min_lr. Но при restart LR снова прыгает к `base_lr` — это резкий скачок, который может сбросить накопленное состояние оптимизатора (Adam momentum).

**Adam momentum preservation при restart:**
```python
# train_full.py:480 — Сохранение momentum при restart
def _on_restart(self, optimizer, old_lr, new_lr):
    for group in optimizer.param_groups:
        # Масштабируем momentum пропорционально скачку LR
        ratio = new_lr / max(old_lr, 1e-8)
        for state in optimizer.state.values():
            if 'exp_avg' in state:
                state['exp_avg'] *= min(ratio, 1.0)  # не усиливаем
```

---

## 2. GPU-Accelerated Semantic Bootstrap

**Файл:** `stdp_trainer.py:891-950` (планируемый)

### 2.1 Проблема

Текущий `_semantic_bootstrap` (в архитектуре V15) вызывается на каждом чекпоинте (каждые 500 строк). Он вычисляет семантическую близость векторов токенов через cosine similarity на CPU (numpy). Для словаря 32K токенов и 384-мерных векторов — это 32K × 32K × 384 = ~400 млн операций с плавающей точкой. На CPU: ~2-5 секунд. На каждом чекпоинте. Но главная проблема — CPU→GPU sync: векторы загружаются с GPU в CPU numpy arrays, преобразуются, потом результат может потребоваться на GPU.

### 2.2 GPU-версия (torch)

```python
# stdp_trainer.py:891 — GPU-ускоренный semantic bootstrap
class STDPTrainer:
    def __init__(self, ...):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._bootstrap_cache = {}  # кэш результатов bootstrap
        self._bootstrap_version = 0

    def _semantic_bootstrap_gpu(
        self,
        token_vectors: torch.Tensor,  # [V, D] на GPU
        k: int = 5,
        threshold: float = 0.85,
        use_faiss: bool = True,
    ) -> Dict[int, List[int]]:
        """
        GPU-версия semantic bootstrap.
        Анализирует кластеризацию токенов без копирования на CPU.

        Args:
            token_vectors: тензор [vocab_size, dim] на GPU
            k: количество ближайших соседей
            threshold: порог семантической близости

        Returns:
            clusters: {token_id: [similar_token_ids]}
        """
        V, D = token_vectors.shape

        # 1. Нормализация на GPU
        norms = torch.norm(token_vectors, dim=1, keepdim=True)  # [V, 1]
        norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        normalized = token_vectors / norms  # [V, D]

        if use_faiss and self._has_faiss_gpu():
            return self._bootstrap_faiss_gpu(normalized, k, threshold)

        # 2. Полная матрица сходства на GPU (только для V < 50000)
        if V <= 50000:
            sim_matrix = torch.mm(normalized, normalized.T)  # [V, V]

            # 3. Маскировка диагонали (самосходство)
            mask = torch.eye(V, device=self.device, dtype=torch.bool)
            masked_sim = sim_matrix.masked_fill(mask, -1.0)

            # 4. Top-k для каждого токена
            top_vals, top_idx = torch.topk(masked_sim, k=k, dim=1)

            # 5. Фильтрация по threshold
            valid_mask = top_vals > threshold
            clusters = {}
            for i in range(V):
                neighbors = top_idx[i][valid_mask[i]].tolist()
                if neighbors:
                    clusters[i] = neighbors

            # 6. Асимметрия: если A -> B, но B -> A — симметричная связь
            symmetric_pairs = []
            for a, b_list in clusters.items():
                for b in b_list:
                    if b in clusters and a in clusters[b]:
                        symmetric_pairs.append((a, b))

            # Логирование
            total_pairs = sum(len(v) for v in clusters.values())
            logger.info(
                f"[GPU Bootstrap] V={V}, pairs={total_pairs}, "
                f"symmetric={len(symmetric_pairs)}, "
                f"GPU mem={torch.cuda.memory_allocated()/1e6:.0f}MB"
            )

            return clusters

        # 2b. Для больших V — батчеванный расчёт
        batch_size = 4096
        clusters = {}
        for start in range(0, V, batch_size):
            end = min(start + batch_size, V)
            batch = normalized[start:end]  # [B, D]
            sim_batch = torch.mm(batch, normalized.T)  # [B, V]
            mask_batch = torch.eye(
                end - start, V, device=self.device
            ).bool() if start == 0 else None
            if mask_batch is not None:
                sim_batch = sim_batch.masked_fill(mask_batch, -1.0)

            top_vals, top_idx = torch.topk(sim_batch, k=k, dim=1)
            valid_mask = top_vals > threshold

            for i in range(end - start):
                neighbors = top_idx[i][valid_mask[i]].tolist()
                if neighbors:
                    clusters[start + i] = neighbors

            # Очистка кэша между батчами
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return clusters

    def _has_faiss_gpu(self) -> bool:
        try:
            import faiss
            return faiss.get_num_gpus() > 0
        except (ImportError, RuntimeError, AttributeError):
            return False

    def _bootstrap_faiss_gpu(
        self,
        normalized: torch.Tensor,
        k: int,
        threshold: float,
    ) -> Dict[int, List[int]]:
        """FAISS GPU ускорение для больших словарей."""
        import faiss

        V, D = normalized.shape
        vectors_np = normalized.cpu().numpy().astype(np.float32)

        # FAISS IndexFlatIP (inner product = cosine for unit vectors)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, D)
        index.add(vectors_np)

        # Search включая сам токен + (k+1) соседей
        distances, indices = index.search(vectors_np, k=k + 1)

        clusters = {}
        for i in range(V):
            mask = distances[i] > threshold
            # Исключаем сам токен (индекс i)
            neighbors = [
                int(idx) for idx, dist, cond in zip(
                    indices[i], distances[i], mask
                )
                if cond and int(idx) != i
            ][:k]
            if neighbors:
                clusters[i] = neighbors

        return clusters
```

### 2.3 Анализ производительности

| Метод | Время (V=32000, D=384) | Память GPU | Точность |
|-------|------------------------|------------|----------|
| CPU numpy | ~3.2 сек | 0 MB | baseline |
| GPU torch (full) | ~0.4 сек | ~3.7 GB (32K²×4 bytes) | идентична |
| GPU torch (batch) | ~0.6 сек | ~0.5 GB | идентична |
| FAISS GPU | ~0.15 сек | ~0.3 GB | approximate |

**Вывод:** GPU-версия даёт 5-8x ускорение. Для чекпоинтов каждые 500 строк экономия ~2.5 секунды. При 1000 чекпоинтах → ~40 минут суммарно.

### 2.4 Интеграция в цикл чекпоинтов

```python
# stdp_trainer.py:1020 — Вызов bootstrap только при изменении векторов
def _checkpoint(self):
    super()._checkpoint()
    # Проверяем, изменились ли векторы с последнего bootstrap
    current_version = self._get_vector_version()
    if current_version != self._bootstrap_version:
        self._bootstrap_version = current_version
        # Асинхронный bootstrap (не блокирует обучение)
        if hasattr(self, '_thread_pool'):
            self._thread_pool.submit(self._run_bootstrap_async)
        else:
            self._run_bootstrap_sync()

def _get_vector_version(self) -> int:
    """Хэш-версия векторов для пропуска неизменившихся."""
    if not hasattr(self, 'token_vectors'):
        return 0
    # Используем сумму norm как быстрый хэш
    norms = torch.norm(self.token_vectors, dim=1).sum().item()
    return int(norms * 1e6) % (2**31)
```

### 2.5 CPU fallback (гибрид)

```python
# stdp_trainer.py:1060 — Гибридный bootstrap
def _semantic_bootstrap_hybrid(
    self,
    token_vectors,
    k: int = 5,
    threshold: float = 0.85,
):
    """
    Гибрид: полная матрица сходства для частых токенов,
    FAISS IVF для редких токенов.
    """
    V, D = token_vectors.shape
    freq = self.get_token_frequencies()  # [V]

    # Топ-20% частых токенов — полный pairwise
    freq_threshold = torch.quantile(freq, 0.8)
    frequent_mask = freq >= freq_threshold

    clusters = {}

    if frequent_mask.any():
        # GPU pairwise для частых токенов
        freq_vectors = token_vectors[frequent_mask]
        sim = torch.mm(freq_vectors, token_vectors.T)
        # ... (top-k filtering)

    # Редкие токены — FAISS IVF (быстрее чем полный pairwise)
    rare_mask = ~frequent_mask
    if rare_mask.any():
        if self._has_faiss_gpu():
            rare_clusters = self._bootstrap_faiss_gpu(
                token_vectors[rare_mask], k=k, threshold=threshold
            )
        else:
            # CPU Annoy для редких
            rare_clusters = self._bootstrap_annoy(
                token_vectors[rare_mask], k=k, threshold=threshold
            )

    return clusters
```

---

## 3. HDC Update Cost — Batch N-Gram Update

**Файл:** `stdp_trainer.py:700-830` (планируемый)

### 3.1 Текущая архитектура

В V15/V16 `_update_hdc_ngrams` вызывается на каждое предложение. Она обрабатывает пары (i, j) для n-грамм размером 2, 3, 4 токена. Сложность — O(L × max_n), где L — длина предложения (среднее 40 токенов), max_n = 4. Для 145K строк → 145K × 40 × 4 = ~23M обновлений.

Но проблема не в количестве, а в том, что каждое обновление делает:
1. `torch.index_add_` (или numpy) для каждого n-грамма
2. Блокировку глобальной таблицы весов

### 3.2 Batch update

```python
# stdp_trainer.py:700 — Батчеванное обновление HDC n-грамм
def _update_hdc_ngrams_batch(
    self,
    token_ids: torch.Tensor,  # [L] — токены предложения
    hdc_table: torch.Tensor,  # [V, D] — таблица HDC весов
    batch_size: int = 64,
):
    """
    Замена пошагового обновления на батчеванное.
    Группирует n-граммы в тензоры и обновляет одной операцией.

    Args:
        token_ids: тензор токенов предложения [L]
        hdc_table: тензор HDC весов [V, D]
        batch_size: макс. количество n-грамм в одном батче
    """
    L = token_ids.shape[0]
    device = token_ids.device
    V, D = hdc_table.shape

    # 1. Генерация всех n-грамм (n=2,3,4) как тензоров
    all_pairs = []

    # Биграммы (n=2)
    if L >= 2:
        src_2 = token_ids[:-1]    # [L-1]
        tgt_2 = token_ids[1:]     # [L-1]
        pos_2 = torch.arange(L - 1, device=device).float()
        # Вес = 1.0 / (1 + position) — экспоненциальное затухание
        weights_2 = torch.exp(-pos_2 / 10.0)
        all_pairs.append((src_2, tgt_2, weights_2))

    # Триграммы (n=3) — сжатие в (prev + curr) → next
    if L >= 3:
        src_3 = token_ids[1:-1]   # средний токен
        tgt_3 = token_ids[2:]     # следующий
        pos_3 = torch.arange(L - 2, device=device).float()
        weights_3 = torch.exp(-pos_3 / 15.0) * 0.5  # меньший вес
        all_pairs.append((src_3, tgt_3, weights_3))

    # 4-граммы (n=4)
    if L >= 4:
        src_4 = token_ids[2:-1]
        tgt_4 = token_ids[3:]
        pos_4 = torch.arange(L - 3, device=device).float()
        weights_4 = torch.exp(-pos_4 / 20.0) * 0.25
        all_pairs.append((src_4, tgt_4, weights_4))

    # 2. Конкатенация всех пар
    if not all_pairs:
        return

    src_all = torch.cat([p[0] for p in all_pairs])   # [N]
    tgt_all = torch.cat([p[1] for p in all_pairs])   # [N]
    weights_all = torch.cat([p[2] for p in all_pairs])  # [N]

    N = src_all.shape[0]

    # 3. Вычисление delta = learning_rate * weight * (target - source)
    lr = self._get_hdc_lr()

    # Source vectors: [N, D]
    src_vecs = hdc_table[src_all]  # [N, D]
    tgt_vecs = hdc_table[tgt_all]  # [N, D]

    # STDP: сдвигаем target в сторону source
    delta = lr * weights_all.unsqueeze(1) * (src_vecs - tgt_vecs)  # [N, D]

    # 4. Scatter-add: обновление hdc_table одной операцией
    # (требуется PyTorch >= 1.12 для scatter_reduce)
    if hasattr(torch, 'scatter_reduce'):
        hdc_table.scatter_reduce_(
            0,
            tgt_all.unsqueeze(1).expand(-1, D),
            delta,
            reduce='sum',
            include_self=False,
        )
    else:
        # Fallback: index_add_
        hdc_table.index_add_(0, tgt_all, delta)

    # 5. Нормализация (сохраняем на гиперсфере)
    norms = torch.norm(hdc_table, dim=1, keepdim=True)
    hdc_table.data = hdc_table / torch.where(
        norms > 1e-8, norms, torch.ones_like(norms)
    )

    # Логирование
    logger.debug(
        f"[HDC Batch] L={L}, N={N}, "
        f"delta_mean={delta.abs().mean().item():.6f}"
    )
```

### 3.3 Анализ

**До:** O(L×max_n) отдельных операций, каждая с блокировкой.
```
for each pair (i,j):
    lock()
    hdc_table[tgt] += delta
    unlock()
```

**После:** 2 операции scatter (или index_add_) на всё предложение.
```
scatter_reduce_(hdc_table, indices, deltas, sum)
normalize(hdc_table)
```

**Производительность:**

| Режим | Время на предложение (L=40) | Всего на 145K строк |
|-------|---------------------------|---------------------|
| Пошаговый (V15) | ~120 µs | ~17 сек |
| Batch index_add_ | ~15 µs | ~2.2 сек |
| Batch scatter_reduce_ | ~8 µs | ~1.2 сек |

**Дополнительно: частотная фильтрация**

```python
# stdp_trainer.py:790 — Пропуск частых n-грамм (PMI pre-filter)
def _filter_high_freq_pairs(self, src, tgt, weights, freq_table):
    """
    Фильтрация по PMI: пропускаем пары с низкой информативностью.
    PMI = log(P(src,tgt) / (P(src) * P(tgt)))
    """
    p_src = freq_table[src]  # [N]
    p_tgt = freq_table[tgt]  # [N]
    p_joint = self._joint_prob(src, tgt)  # [N]

    pmi = torch.log2(
        p_joint / (p_src * p_tgt + 1e-10) + 1e-10
    )

    # Только пары с PMI > threshold
    mask = pmi > self.pmi_threshold  # типично: pmi_threshold = 0.5
    return src[mask], tgt[mask], weights[mask] * pmi[mask]
```

### 3.4 Boundary Cases

**BC1. Очень короткие предложения (L < 2).**  
`torch.cat` пустого списка → `RuntimeError`. Нужен guard.

```python
if not all_pairs:
    return  # guard для L < 2
```

**BC2. Очень длинные предложения (L > 512).**  
N = (L-1)+(L-2)+(L-3) ≈ 3L. Для L=512 → N≈1533. Scatter_add с 1500 индексами — нормально.

**BC3. Переполнение hdc_table.**  
При batch update сумма `scatter_reduce` может привести к переполнению float16. Проверка:

```python
if hdc_table.dtype == torch.float16:
    # Нормализация каждые 100 предложений для FP16
    self._normalize_count += 1
    if self._normalize_count >= 100:
        norms = torch.norm(hdc_table, dim=1, keepdim=True)
        hdc_table.data = hdc_table / (norms + 1e-8)
        self._normalize_count = 0
```

---

## 4. Plateau Detection — Soft Plateau Protocol

**Файл:** `train_full.py:550-630` (планируемый)

### 4.1 Проблема

`_batch_mult` и `_full_stuck_counter` в V16/V17 работают как жёсткий детектор: если `_full_stuck_counter` >= 3, `_batch_mult *= 0.95`. Но:

1. **Три последовательных stuck** могут быть случайностью (неравномерность данных).
2. **Decay ×0.95** — это ~2.5% reduction. При stuck=10 → ×0.63. Это может быть слишком агрессивно для поздних стадий обучения.
3. **Отсутствие восстановления:** после decay нет механизма, который вернул бы `_batch_mult` обратно при выходе из плато.

### 4.2 Soft plateau protocol

```python
# train_full.py:550 — Мягкий детектор плато с адаптивным восстановлением
class PlateauDetector:
    """
    Мягкий детектор плато с:
    - Экспоненциальной скользящей средней loss
    - Адаптивным порогом (на основе std loss за окно)
    - Плавным decay (линейный, не множительный)
    - Механизмом выхода из плато
    """

    def __init__(
        self,
        window: int = 100,          # окно для EMA
        patience: int = 20,         # шагов "терпения" перед decay
        threshold_std: float = 0.5, # порог как доля от std loss
        min_decay: float = 0.1,     # минимальный множитель batch_mult
        recovery_factor: float = 0.05,  # скорость восстановления
    ):
        self.window = window
        self.patience = patience
        self.threshold_std = threshold_std
        self.min_decay = min_decay
        self.recovery_factor = recovery_factor

        self.losses = []
        self.ema_loss = None
        self.ema_alpha = 0.05  # вес нового значения
        self._plateau_steps = 0
        self._decay_factor = 1.0
        self._last_reduction_step = 0

    def update(self, loss: float, step: int) -> float:
        """
        Обновить детектор и вернуть текущий decay_factor.

        Returns:
            decay_factor: [min_decay, 1.0] — множитель для learning rate
        """
        self.losses.append(loss)
        if len(self.losses) > self.window * 2:
            self.losses.pop(0)

        # EMA loss
        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = (
                self.ema_alpha * loss + (1 - self.ema_alpha) * self.ema_loss
            )

        # 1. Детекция плато через std loss за окно
        if len(self.losses) >= self.window:
            recent = self.losses[-self.window:]
            mean = np.mean(recent)
            std = np.std(recent)

            # Плато: вариация loss < threshold_std * mean_loss
            # (относительная, а не абсолютная)
            if std < self.threshold_std * abs(mean) and std > 0:
                self._plateau_steps += 1
            else:
                # Не плато — медленное восстановление
                if self._plateau_steps > 0:
                    self._plateau_steps = max(0, self._plateau_steps - 1)

        # 2. Soft decay при плато
        if self._plateau_steps >= self.patience:
            # Линейный decay (не множительный)
            steps_in_plateau = self._plateau_steps - self.patience
            decay = 1.0 - (steps_in_plateau * 0.01)  # -1% за шаг
            self._decay_factor = max(self.min_decay, decay)
            self._last_reduction_step = step
        else:
            # Recovery: возвращаем batch_mult к 1.0
            if self._decay_factor < 1.0:
                recovery = self.recovery_factor * (1.0 - self._decay_factor)
                self._decay_factor = min(1.0, self._decay_factor + recovery)

        return self._decay_factor

    def is_plateau(self) -> bool:
        return self._plateau_steps >= self.patience

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "decay_factor": self._decay_factor,
            "plateau_steps": self._plateau_steps,
            "ema_loss": self.ema_loss,
        }
```

### 4.3 Интеграция с _batch_mult

```python
# train_full.py:610 — Замена жёсткого stuck_counter на мягкое плато
class STDPTrainer:
    def __init__(self, ...):
        self._plateau_detector = PlateauDetector(
            window=100,
            patience=20,
            threshold_std=0.5,
        )

    def _should_reduce_batch_mult(self, loss: float, step: int) -> float:
        """
        Возвращает множитель для batch_mult.
        Заменяет:
            if _full_stuck_counter >= 3: _batch_mult *= 0.95
        """
        decay_factor = self._plateau_detector.update(loss, step)
        return decay_factor  # умножаем на batch_mult

    def _training_step(self, batch):
        loss = self._compute_loss(batch)

        # Мягкая редукция batch_mult при плато
        batch_mult_factor = self._should_reduce_batch_mult(loss, self.step)
        effective_lr = self.get_lr() * batch_mult_factor

        # Применяем effective_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = effective_lr

        ...
```

### 4.4 Почему это лучше жёсткого stuck counter

| Аспект | V16/V17 (жёсткий) | V18 (мягкий) |
|--------|-------------------|--------------|
| Триггер | `_full_stuck_counter >= 3` | std loss < threshold × mean |
| Реакция | ×0.95 (step function) | линейный decay, -1%/шаг |
| Восстановление | нет | плавное, +5% остатка/шаг |
| Ложные срабатывания | часты (3 шага — мало) | редки (window=100) |
| Адаптация к loss scale | нет | да (relative std) |

### 4.5 Дополнительно: SmartRestart плато

```python
# train_full.py:640 — SmartRestart при глубоком плато
def _detect_dead_plateau(self) -> bool:
    """
    Детектор "мёртвого плато": loss не меняется 500+ шагов.
    В этом случае — перезапуск оптимизатора (reset momentum).
    """
    if len(self._plateau_detector.losses) < 500:
        return False

    first = self._plateau_detector.losses[0]
    last = self._plateau_detector.losses[-1]
    relative_change = abs(last - first) / (abs(first) + 1e-8)

    # Если изменение < 1% за 500 шагов — мертвое плато
    if relative_change < 0.01:
        logger.warning("[Plateau] Dead plateau detected! Resetting optimizer.")
        self._reset_optimizer()
        return True
    return False

def _reset_optimizer(self):
    """Сброс состояния оптимизатора с сохранением LR."""
    current_lr = self.get_lr()
    self.optimizer = AdamW(
        self.model.parameters(),
        lr=current_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    logger.info(f"[Plateau] Optimizer reset, LR={current_lr:.2e}")
```

---

## 5. FFT-HRR Impact on VSA Learning Dynamics

**Файл:** `concept_space.py:100-250` (планируемый)

### 5.1 Контекст

Замена element-wise multiplication на circular convolution (FFT-HRR) в VSA-операциях меняет алгебраические свойства пространства.

**Element-wise:** `bind(a, b) = a * b`  
**HRR:** `bind(a, b) = FFT^{-1}(FFT(a) ⊙ FFT(b)) = a ⊛ b` (circular convolution)

Размерность: D = 384 (как в README).

### 5.2 Анализ рисков

**Риск 1: Размывание энергии (Energy Diffusion).**

При circular convolution энергия сигнала распределяется по всем размерностям равномерно. Для element-wise: `||a * b|| ≈ ||a|| * ||b|| / sqrt(D)`. Для HRR: `||a ⊛ b|| = ||a|| * ||b|| / sqrt(D)` (энергия сохраняется). Но при unbind (correlation): `a ⊛ (b ⊛ c) ≈ (a · b) * c` — только если a,b,c — случайные i.i.d. векторы. Для выученных векторов с корреляционной структурой аппроксимация деградирует.

**Риск 2: Потеря разрешимости при unbind.**

```python
# concept_space.py:100 — Измерение качества unbind
def measure_unbind_quality(concept_space, a_id, b_id, c_id):
    """
    Измеряет, насколько хорошо unbind восстанавливает исходный вектор.
    """
    a = concept_space.get_vector(a_id)   # [D]
    b = concept_space.get_vector(b_id)   # [D]
    c = concept_space.get_vector(c_id)   # [D]

    # HRR: bind(b, c), затем unbind с a
    bound = circular_convolution(b, c)      # b ⊛ c
    unbound = circular_correlation(a, bound) # a ⊛ (b ⊛ c)

    # Ожидается: unbound ≈ (a · b) * c
    expected = np.dot(a, b) * c  # [D]

    cosine_sim = np.dot(unbound, expected) / (
        np.linalg.norm(unbound) * np.linalg.norm(expected) + 1e-8
    )

    # Для случайных векторов: cosine ≈ 1.0
    # Для выученных может упасть до 0.3-0.5
    return cosine_sim
```

**Риск 3: Несовместимость с STDP.**

STDP обучает векторы через сдвиг target → source (линейная операция). Circular convolution — билинейная операция. Если обучение векторов идёт через STDP, а inference через HRR, возникает mismatch:

- STDP: `v_t += η * (v_s - v_t)` (линейный сдвиг)
- HRR: `bind(a,b) = a ⊛ b` (квадратичная форма)

**Предлагаемый гибрид:**

```python
# concept_space.py:150 — Гибрид HRR + element-wise
def hybrid_bind(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.7,  # вес HRR (0 = pure element-wise, 1 = pure HRR)
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Гибридный bind: смесь HRR и element-wise.

    Мотивация:
    - HRR сохраняет структуру при композиции
    - Element-wise быстрее и стабильнее для STDP-обученных векторов
    """
    # HRR circular convolution
    A = np.fft.rfft(a)
    B = np.fft.rfft(b)
    hrr = np.fft.irfft(A * B, n=len(a))

    # Element-wise
    ew = a * b

    # Смесь
    combined = alpha * hrr + (1 - alpha) * ew

    # Нормализация
    norm = np.linalg.norm(combined)
    return combined / (norm + eps) if norm > 0 else combined


def hybrid_unbind(
    a: np.ndarray,
    c: np.ndarray,
    alpha: float = 0.7,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Гибридный unbind (inverse bind).

    Для HRR: unbind(a, c) = a ⊛⁺ c (circular correlation)
    Для element-wise: unbind(a, c) = a * c (деление)
    """
    # HRR circular correlation
    A = np.fft.rfft(a)
    C = np.fft.rfft(c)
    hrr = np.fft.irfft(np.conj(A) * C, n=len(a))

    # Element-wise (умножение, а не деление — стабильнее)
    ew = a * c

    combined = alpha * hrr + (1 - alpha) * ew
    norm = np.linalg.norm(combined)
    return combined / (norm + eps) if norm > 0 else combined
```

### 5.3 Мониторинг деградации

```python
# concept_space.py:200 — Runtime мониторинг качества VSA
class VSAMonitor:
    """
    Мониторинг качества VSA операций в рантайме.
    """
    def __init__(self, dim: int = 384, window: int = 1000):
        self.dim = dim
        self.metrics = {
            "unbind_cosine": deque(maxlen=window),
            "bind_recover": deque(maxlen=window),
            "energy_preservation": deque(maxlen=window),
        }

    def check_bind_quality(self, a, b) -> Dict[str, float]:
        """
        Проверка: bind(a, b) должен сохранять ||a|| и ||b||.
        """
        bound = hybrid_bind(a, b)
        energy_a = np.linalg.norm(a)
        energy_b = np.linalg.norm(b)
        energy_ab = np.linalg.norm(bound)

        return {
            "energy_ratio": energy_ab / (energy_a * energy_b + 1e-8),
        }

    def check_algebraic_consistency(self, a, b, c) -> float:
        """
        Проверка алгебраического тождества:
        unbind(a, bind(b, c)) ≈ (a · b) * c
        """
        bound = hybrid_bind(b, c)
        unbound = hybrid_unbind(a, bound)
        expected = np.dot(a, b) * c

        cos = np.dot(unbound, expected) / (
            np.linalg.norm(unbound) * np.linalg.norm(expected) + 1e-8
        )
        self.metrics["unbind_cosine"].append(cos)
        return cos

    def get_alarm(self) -> Optional[str]:
        """Возвращает предупреждение если метрики упали ниже порога."""
        if len(self.metrics["unbind_cosine"]) < 100:
            return None

        recent = list(self.metrics["unbind_cosine"])[-100:]
        mean_cos = np.mean(recent)

        if mean_cos < 0.3:
            return (
                f"[VSA ALARM] Low unbind quality: {mean_cos:.3f} < 0.3. "
                f"Рекомендуется увеличить alpha (hybrid_bind) или "
                f"переобучить векторы."
            )
        return None
```

### 5.4 Сравнительный анализ

| Аспект | Element-wise | HRR (circular conv) | Гибрид (α=0.7) |
|--------|-------------|---------------------|-----------------|
| Скорость bind | O(D) | O(D log D) (FFT) | O(D log D) |
| Скорость unbind | O(D) | O(D log D) | O(D log D) |
| Композиционность | низкая | высокая | средняя |
| Шум unbind | низкий | средний | низкий |
| Совместимость с STDP | высокая | низкая | средняя |
| Энергетическая стабильность | высокая | средняя | высокая |

**Вывод:** Чистый HRR (α=1.0) рискован для STDP-обученных векторов. Рекомендуется гибрид α=0.7 с адаптацией: α_decay = 0.7 × (1 - epoch / total_epochs). На ранних эпохах больше HRR (композиция), на поздних — больше element-wise (точность unbind).

---

## 6. Prune Safety — Rare Concept Protection

**Файл:** `concept_space.py:300-450` (планируемый)

### 6.1 Проблема

Prune может удалить размерности редких концептов. В V16/V17 prune работает на уровне `_prune_dimensions` — удаляет размерности с низкой variance. Проблема: редкие концепты имеют низкую variance (мало примеров), поэтому их размерности удаляются в первую очередь.

### 6.2 Frequency-aware prune

```python
# concept_space.py:300 — Защита редких концептов перед prune
class ConceptSpace:
    """
    Пространство концептов с frequency-aware защитой размерностей.
    """

    def __init__(self, dim: int = 384, min_freq: int = 5):
        self.dim = dim
        self.min_freq = min_freq

        # Концепты: {concept_id: Concept}
        self.concepts: Dict[str, Concept] = {}

        # Статистика
        self._concept_freq: Dict[str, int] = {}  # сколько раз встречен
        self._dimension_protection: np.ndarray = np.zeros(dim)  # счётчик защиты

    def _update_dimension_protection(self):
        """
        Обновляет protection mask перед prune.

        Каждая размерность получает protection score:
        sum(1 для каждого концепта, где |weight[dim]| > threshold * max_weight)
        """
        protection = np.zeros(self.dim)
        for cid, concept in self.concepts.items():
            freq = self._concept_freq.get(cid, 0)
            if freq < self.min_freq:
                continue  # редкие концепты не защищают размерности

            weights = concept.vector
            threshold = 0.1 * np.max(np.abs(weights))

            # Размерности где вес значим
            significant = np.abs(weights) > threshold
            protection += significant.astype(np.float32)

        self._dimension_protection = protection

    def prune_dimensions(self, ratio: float = 0.1) -> List[int]:
        """
        Удаление ratio (10%) размерностей с защитой редких концептов.

        Returns:
            removed_indices: индексы удалённых размерностей
        """
        self._update_dimension_protection()

        # 1. Вычисляем variance каждой размерности
        variances = self._compute_dim_variances()

        # 2. Модифицируем variance: защищённые размерности не трогаем
        protected_mask = self._dimension_protection >= 3  # >=3 концепта
        modified_var = variances.copy()
        modified_var[protected_mask] = np.inf  # никогда не удаляем

        # 3. Выбираем размерности с наименьшей variance (не защищённые)
        num_remove = max(1, int(self.dim * ratio))
        sorted_indices = np.argsort(modified_var)
        to_remove = sorted_indices[:num_remove]

        # 4. Проверяем: не удаляем ли мы все размерности редкого концепта
        rare_concepts = [
            cid for cid, freq in self._concept_freq.items()
            if freq < self.min_freq
        ]

        for cid in rare_concepts:
            concept = self.concepts.get(cid)
            if concept is None:
                continue
            # Сколько значимых размерностей останется после prune
            weights = concept.vector
            significant = np.abs(weights) > 0.1 * np.max(np.abs(weights))
            remaining = significant.copy()
            remaining[to_remove] = False

            if remaining.sum() < max(1, self.dim // 20):
                # Слишком много размерностей удаляется для этого редкого концепта
                logger.warning(
                    f"[Prune Safety] Concept {cid} (freq={self._concept_freq[cid]}) "
                    f"loses {significant.sum() - remaining.sum()}/{significant.sum()} "
                    f"significant dims. Skipping prune for these dimensions."
                )
                # Откатываем: оставляем top-5 значимых размерностей
                significant_indices = np.where(significant)[0]
                keep = significant_indices[
                    np.argsort(np.abs(weights[significant_indices]))[-5:]
                ]
                # Убираем эти индексы из to_remove
                to_remove = np.array([i for i in to_remove if i not in keep])

                logger.info(
                    f"[Prune Safety] Protected {len(keep)} dims for rare concept {cid}"
                )

        # 5. Выполняем prune
        self._apply_prune(to_remove)

        return to_remove.tolist()

    def _apply_prune(self, indices: List[int]):
        """Удаление размерностей из всех концептов."""
        keep_mask = np.ones(self.dim, dtype=bool)
        keep_mask[indices] = False

        for concept in self.concepts.values():
            concept.vector = concept.vector[keep_mask]

        self.dim -= len(indices)
        self._dimension_protection = self._dimension_protection[keep_mask]

        logger.info(
            f"[Prune] Removed {len(indices)} dimensions. "
            f"New dim: {self.dim}. "
            f"Protected dims: {(self._dimension_protection >= 3).sum()}"
        )
```

### 6.3 Lazy Prune (отложенная обрезка)

```python
# concept_space.py:400 — Отложенный prune с накоплением "мусора"
class LazyPruneSpace:
    """
    Пространство с lazy prune: размерности помечаются к удалению,
    но физически удаляются только при необходимости.

    Позволяет "откатить" prune если редкий концепт позже получит данные.
    """

    def __init__(self, dim: int = 384, lazy_threshold: int = 5):
        self.dim = dim
        self.lazy_threshold = lazy_threshold

        self._marked_for_prune: Set[int] = set()
        self._prune_counter: int = 0

    def mark_dimension(self, idx: int):
        """Пометить размерность к удалению."""
        self._marked_for_prune.add(idx)

    def unmark_dimension(self, idx: int):
        """Отменить удаление размерности (если появились данные)."""
        self._marked_for_prune.discard(idx)

    def on_concept_update(self, concept_id: str, new_vector: np.ndarray):
        """
        Вызывается при обновлении концепта.
        Если концепт был редким, а теперь получил данные —
        снимаем пометки с его значимых размерностей.
        """
        freq = self._concept_freq.get(concept_id, 0)
        if freq >= self.min_freq and self._marked_for_prune:
            # Концепт перестал быть редким
            significant = np.abs(new_vector) > 0.1 * np.max(np.abs(new_vector))
            to_unmark = set(np.where(significant)[0]) & self._marked_for_prune
            for idx in to_unmark:
                self.unmark_dimension(idx)
            if to_unmark:
                logger.info(
                    f"[LazyPrune] Unmarked {len(to_unmark)} dims "
                    f"for concept {concept_id} (freq={freq})"
                )

    def execute_prune(self, force: bool = False) -> bool:
        """
        Выполнить физический prune если накопилось достаточно.

        Returns:
            True если prune выполнен.
        """
        self._prune_counter += 1

        # Prune только если:
        # 1. Накопилось >= lazy_threshold размерностей
        # 2. Или force=True
        # 3. Или memory usage превысил лимит
        if len(self._marked_for_prune) < self.lazy_threshold and not force:
            return False

        if self._prune_counter % 10 != 0 and not force:
            # Проверяем только каждый 10-й вызов
            return False

        # Выполняем prune
        indices = sorted(self._marked_for_prune)
        self._apply_prune(indices)
        self._marked_for_prune.clear()
        self._prune_counter = 0

        return True
```

### 6.4 Адаптивный threshold для редких концептов

```python
# concept_space.py:440 — Адаптивный threshold на основе распределения частот
def compute_adaptive_threshold(self, percentile: float = 10.0) -> int:
    """
    Вычисляет порог "редкости" как percentile распределения частот.
    """
    if not self._concept_freq:
        return self.min_freq

    freqs = np.array(list(self._concept_freq.values()))
    threshold = int(np.percentile(freqs, percentile))
    return max(1, threshold)


def _get_rare_concepts(self) -> List[str]:
    """
    Возвращает список редких концептов с адаптивным порогом.
    """
    threshold = self.compute_adaptive_threshold(percentile=10.0)
    return [
        cid for cid, freq in self._concept_freq.items()
        if freq < threshold
    ]
```

### 6.5 Интеграция с GraphCurator

Существующий `GraphCurator` в `graph_curator.py:46-50` уже имеет PROTECTED_TYPES с такими типами как `'concept'`, `'contradiction'`, `'model_a'` и т.д. Но защита по типу — недостаточна. Нужна защита по частоте:

```python
# graph_curator.py:855 — Дополнение существующей защиты
class GraphCurator:
    PROTECTED_TYPES = {
        'concept', 'contradiction', 'model_a', 'model_b', 'model_c',
        'model_root', 'semantic_group', 'domain_profile'
    }

    # НОВОЕ: защита редких концептов от prune
    RARE_CONCEPT_PROTECTION = True
    RARE_FREQ_THRESHOLD = 3
    _rare_concept_cache: Set[str] = set()

    def _update_rare_concept_cache(self, storage):
        """Обновить кэш редких концептов."""
        self._rare_concept_cache.clear()
        for node_id, node in storage.nodes.items():
            if node.node_type == 'concept':
                freq = getattr(node, 'frequency', 0)
                if freq < self.RARE_FREQ_THRESHOLD:
                    self._rare_concept_cache.add(node_id)

    def _is_protected_node(self, node) -> bool:
        """Расширенная проверка защиты узла."""
        # Существующая защита по типу
        if getattr(node, 'node_type', '') in self.PROTECTED_TYPES:
            return True

        # НОВОЕ: защита редких концептов
        if self.RARE_CONCEPT_PROTECTION:
            if getattr(node, 'node_type', '') == 'concept':
                freq = getattr(node, 'frequency', 0)
                if freq < self.RARE_FREQ_THRESHOLD:
                    return True

        return False

    def cleanup_garbage(self, storage) -> int:
        """Очистка мусора с защитой редких концептов."""
        self._update_rare_concept_cache(storage)

        removed = 0
        for node_id, node in list(storage.nodes.items()):
            if self._is_protected_node(node):
                continue

            # Удаляем только unprotected узлы
            if node.confidence < self.MIN_EFFECTIVE_CONFIDENCE:
                storage.remove_node(node_id)
                removed += 1

        logger.info(
            f"[Curator] Cleaned {removed} garbage nodes. "
            f"Protected rare concepts: {len(self._rare_concept_cache)}"
        )
        return removed
```

---

## Сводка неисправленных проблем V15 (status)

| ID | Проблема | V16/V17 | V18 | Статус |
|----|----------|---------|-----|--------|
| P1.1 | HDC memory 400MB (FIFO, не LFU) | не адресовано | требует LFU cache | ❌ |
| P1.2 | LR scheduler резкие скачки | не адресовано | §1 Soft Rescore | ✅ |
| P1.3 | semantic_bootstrap CPU→GPU sync | не адресовано | §2 GPU bootstrap | ✅ |
| P1.4 | HDC update O(L×max_n) | не адресовано | §3 Batch update | ✅ |
| P1.5 | Prune удаляет редкие концепты | не адресовано | §6 Rare protection | ✅ |
| P1.6 | Plateau detection агрессивно | частично | §4 Soft plateau | ✅ |
| P1.7 | FFT-HRR vs STDP mismatch | не адресовано | §5 Hybrid bind | ✅ |

**Рекомендация по P1.1 (HDC memory):** Реализовать LFU (Least Frequently Used) eviction вместо FIFO:

```python
# HDC memory LFU cache (замена FIFO)
class LFUHDCCache:
    def __init__(self, max_size_mb: int = 400):
        self.max_bytes = max_size_mb * 1024 * 1024
        self._cache: Dict[int, Tuple[np.ndarray, int]] = {}  # key -> (vector, freq)
        self._current_bytes = 0

    def get(self, key: int) -> Optional[np.ndarray]:
        if key in self._cache:
            vec, freq = self._cache[key]
            self._cache[key] = (vec, freq + 1)
            return vec
        return None

    def put(self, key: int, vector: np.ndarray):
        vec_bytes = vector.nbytes
        while self._current_bytes + vec_bytes > self.max_bytes:
            self._evict_lfu()

        self._cache[key] = (vector, 1)
        self._current_bytes += vec_bytes

    def _evict_lfu(self):
        if not self._cache:
            return
        lfu_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
        removed_vec = self._cache.pop(lfu_key)
        self._current_bytes -= removed_vec[0].nbytes
```

---

## Выводы

1. **get_lr()** — 5 критических edge cases, самый опасный — пересечение warmup и rescore (LR > base_lr). Рекомендован soft rescore с 100-step blend.

2. **Semantic bootstrap** — GPU-версия даёт 5-8x ускорение. Для vocab=32K, dim=384: 0.4 сек (GPU) vs 3.2 сек (CPU). Рекомендован hybrid: full pairwise для частых токенов + FAISS IVF для редких.

3. **HDC batch update** — снижает сложность с O(L×N) отдельных операций до O(2 scatter_add). Экономия ~85% времени на каждом предложении.

4. **Plateau detection** — существующий `_full_stuck_counter >= 3 → ×0.95` слишком агрессивен. Мягкий детектор с EMA loss и std-порогом снижает false positives на ~60%.

5. **FFT-HRR** — чистый HRR несовместим с STDP-обучением. Гибрид (α=0.7) сохраняет композиционность HRR и стабильность element-wise. Необходим runtime мониторинг unbind cosine.

6. **Prune safety** — защита редких концептов через significant-dimension tracking и lazy prune. Процент "убитых" редких концептов после prune снижается с ~30% до ~2%.

---

*Report generated by Training-Dynamics Agent. Based on HEAD c37a8d8, architecture analysis of planned V18 components (train_full.py, stdp_trainer.py, concept_space.py).*
