# FCF Architect Report V5 — Коллегия AI-агентов (Улучшения и новые методы)

**Дата**: 2026-06-17
**Проект**: Fractal Cognitive Field (FCF) — нейро-символическая языковая модель
**Версия отчёта**: V5 (после закрытия 95/95 проблем V3+V4, fresh audit — фокус на улучшения и новые методы)
**Состав коллегии**: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent

---

## Сводка

| Агент | Новых методов | P1 | P2 | P3 |
|-------|:-------------:|:--:|:--:|:--:|
| Architect-AI | 7 | 2 | 2 | 3 |
| Neuro-Symbolic Specialist | 5 | 2 | 2 | 1 |
| GPU-Opt Agent | 12 | 3 | 5 | 4 |
| Training-Dynamics Agent | 10 | 5 | 5 | 0 |
| Quality-Safety Agent | 10 | 3 | 6 | 1 |
| **Итого** | **44** | **15** | **20** | **9** |

---

# 1. Architect-AI: Архитектурные улучшения

### AM-6: Generator-Trainer Separation (Разделение генерации и обучения)

- **Суть**: `CrystalGenerator` (~1600 строк) — God-класс с тремя ортогональными ролями: (1) генерация (beam search, graph search, branching), (2) обучение (STDP CPU/GPU, negative sampling, contrastive, centroid pull), (3) оценка (evaluate, PPL). Разделить на `CrystalGenerator` (только generate/_branch/_graph_search) + `STDPTrainer` (train_from_text, train_batch, STDP CPU/GPU) + `Evaluator` (evaluate, PPL). Уменьшит связанность, упростит тестирование каждого модуля, позволит заменять стратегии обучения без переписывания генерации.
- **Приоритет**: P1
- **Сложность**: 7/10
- **Пример внедрения**:
  ```python
  class STDPTrainer:
      def __init__(self, cs, lattice, gen=None):
      def train_batch(self, texts, ...) -> int
      def _gpu_stdp_apply(self, ...)

  class Evaluator:
      def __init__(self, cs, lattice, sp)
      def evaluate(self, corpus_path, ...) -> dict
  ```

### AM-7: Async Checkpoint Manager

- **Суть**: Сохранение чекпоинтов (cs.save + lattice.save + opt.save_state + 3D vis) блокирует training loop на секунды. Для concept_space 73K×384 ~112MB npz — запись на HDD может занимать >1с. Сделать `CheckpointManager` с threaded/async записью: сохранение в tmp + replace в фоновом потоке, основной цикл не блокируется.
- **Приоритет**: P2
- **Сложность**: 4/10
- **Пример внедрения**:
  ```python
  class CheckpointManager:
      def __init__(self, data_dir, cleanup_keep=5, executor=None)
      def save(self, tag, cs, lattice, opt):
          future = self.executor.submit(self._sync_save, tag, cs, lattice, opt)
      def wait_and_cleanup(self)
  ```

### AM-8: Configuration Schema Validation

- **Суть**: `FCFConfig.load()` молча загружает JSON, любые опечатки/типы игнорируются (`if hasattr(cfg, k)`). Нет раннего детекта ошибок конфигурации. Добавить dataclass-валидатор в `__post_init__`: проверка типов, диапазонов (dim%8==0, 0<=destab<=1), целостности ссылок (`ParamDef.rules` ссылаются только на существующие `param.name`).
- **Приоритет**: P2
- **Сложность**: 3/10
- **Пример внедрения**:
  ```python
  @dataclass
  class FCFConfig:
      dim: int = 384
      def __post_init__(self):
          assert self.dim % 8 == 0, f"dim={self.dim} не кратен 8"
          assert 0 <= self.destab_scale_end <= self.destab_scale_start <= 1.0
          names = [p.name for p in self.params]
          assert len(names) == len(set(names)), f"дубликаты: {names}"
  ```

### AM-9: Training Pipeline Abstraction

- **Суть**: `train_full.py` (~740 строк) — плоский скрипт с циклом, содержащим: curriculum, batch scheduling, destab decay, fluctuate scheduler, periodic decay, checkpoint, live status, 3D vis, evaluation step, optimizer step, full_stuck detection, logging. Выделить `TrainingPipeline` с подсистемами: `Scheduler` (fluctuate/decay/checkpoint intervals), `Curriculum` (max_len ramp, context_window ramp), `StatusReporter` (terminal + json), `DestabController` (global_step → destab_scale). Позволит тестировать расписания изолированно.
- **Приоритет**: P1
- **Сложность**: 8/10
- **Пример внедрения**:
  ```python
  class TrainingPipeline:
      def __init__(self, cs, lattice, gen, opt, config)
      def run(self, train_lines, val_lines):
          while self._epoch_loop():
              while self._step():
                  self._curriculum.advance(global_step)
  ```

### AM-10: AdaptiveErrorTracker (изоляция concept_error)

- **Суть**: `concept_error` (OrderedDict + EMA decay) встроен в CrystalGenerator. Выделить в reusable класс для использования также в `ParameterOptimizer` и любом другом компоненте с адаптивным забыванием. Лимит `min(3*vocab_size//4, 100000)` — эвристика, должна быть в конфиге.
- **Приоритет**: P3
- **Сложность**: 3/10
- **Пример внедрения**:
  ```python
  class AdaptiveErrorTracker:
      def __init__(self, decay=0.9, max_size=100000)
      def update(self, cid, error: float)
      def get(self, cid) -> float
  ```

### AM-11: FieldBit Lazy Reconstruction

- **Суть**: `_fb_t` (GPU tensor для octree field bits) — 73K×~256 байт ≈ 18MB. Используется только в `_negative_sampling_gpu` для field_gate фильтрации. Если `field_gate=False`, `_fb_t` не нужен. Сделать lazy-построение: `_fb_t = None`, собирать только при `field_gate=True`.
- **Приоритет**: P3
- **Сложность**: 2/10
- **Пример внедрения**:
  ```python
  def _ensure_torch(self, field_gate=False):
      if field_gate and self._fb_t is None:
          fb_need = True
      if not fb_need and not dirty: return
      if fb_need: self._build_fb_tensor()
  ```

### AM-12: Shared Thread-Local RNG Registry

- **Суть**: В коде разбросаны `np.random.RandomState(42)`, `random.Random(42)`, `self._inhibit_rng`, `self.main_rng`, `self.branch_rngs`, `self._fluct_rng` — нецентрализованные RNG с разными seed'ами. При изменении порядка операций результаты невоспроизводимы. Создать `RNGRegistry` с именованными seed'ами и единым контролем воспроизводимости.
- **Приоритет**: P3
- **Сложность**: 2/10
- **Пример внедрения**:
  ```python
  class RNGRegistry:
      def __init__(self, master_seed=42):
          self._rngs = {}
      def get(self, name: str) -> random.Random:
          if name not in self._rngs:
              self._rngs[name] = random.Random(self._seed_for(name))
          return self._rngs[name]
  ```

### Статус старых предложений AM-* из V3:

- **AM-1 EventBus**: P3 — не критично для текущей архитектуры (один Generator → один ConceptSpace). Hook'и и monkey-patch — временное решение.
- **AM-2 Protocols**: P2 — частично реализовано (dataclasses). Не хватает протокола для `VectorStore` и `LatticePredictor`.
- **AM-3 TorchCache**: P2 — реализовано в ядре (`_ensure_torch`, `_torch_dirty`, `_invalidate_torch`). Улучшение: добавить сериализацию кеша.
- **AM-4 CheckpointManager**: P2 — не реализовано. `cleanup_old_checkpoints` и `save_checkpoint_state` всё ещё в `train_full.py`.
- **AM-5 MemoryBudget**: P3 — частично. OOM fallback есть. Нет мониторинга CPU RAM.

---

# 2. Neuro-Symbolic Specialist: Улучшения концептуального пространства

### SN-4: Dual-Timescale Subspace-Kinetic STDP

- **Суть**: `_apply_vector_update` пересчитывает полный код из вектора (`v_new @ basis.T`), полностью обходя архитектуру z_c/z_a/z_m. Реализовать дифференциальную пластичность: z_c (identity, slow) — update × 0.01, z_a (attention, fast) — update × 1.0, z_m (meta) — учит темп обновления каждого подпространства через STDP-градиент на z_m. Сейчас 50% латентного кода (z_c, 256/512) имеет ту же plasticity rate, что и z_a — это тратит половину ёмкости кода.
- **Приоритет**: P1
- **Сложность**: 8/10
- **Обоснование**: Dual-timescale позволит z_c сохранять identity (как embedding lookup), z_a адаптироваться к контексту (как attention), а z_m meta-learn свою plasticity.

### SN-5: GPU-Accelerated Contrastive Objective

- **Суть**: `_contrastive_objective` работает через CPU `cs.topk_similar_concepts` даже в GPU-режиме. Переписать через `_vecs_t`: батчевый gather `sims = v_gen @ V_gpu.T`, top-k на GPU, затем `scatter_add_` градиента на `_vecs_t` для уникальных hard negatives. Убрать 2000-sample subsampling — GPU позволяет full-V поиск. Сейчас contrastive objective — узкое место: ~5ms на CPU vs ~0.3ms на GPU.
- **Приоритет**: P1
- **Сложность**: 6/10
- **Обоснование**: 10-15x ускорение contrastive objective + возможность использовать concept_error weighting для hard negatives на GPU.

### SN-6: Multi-Resolution Hierarchical Octree Fields

- **Суть**: Текущие field_bits строятся на одном `min_lcp=2` — всего 64 группы, поле превращается в бинарный классификатор (внутри группы / вне группы). Построить поля на 3-4 глубинах одновременно (L=1,2,3,4), каждая со своим весом: `field_weight = Σ w_l * min(1 + log(overlap_l + 1) * 2, 3)`. Глубокий L (4) даёт до 4096 групп — тонкая дискриминация, мелкий L (1) — 8 групп — грубое структурирование.
- **Приоритет**: P2
- **Сложность**: 6/10
- **Обоснование**: Текущее бинарное поле теряет информацию о степени иерархической близости. Multi-resolution даёт гладкое иерархическое родство.

### SN-7: Momentum-Accumulated STDP (Nesterov-style)

- **Суть**: Добавить per-concept momentum buffer: `buf_grad = μ * buf_grad + (1-μ) * grad_avg`, `v_new = v + buf_grad * lr`. Использовать Nesterov: сначала экстраполировать код, потом считать градиент в экстраполированной точке. STDP сильно шумит из-за малых контекстных окон (2-6 токенов). Momentum даст более плавную траекторию.
- **Приоритет**: P2
- **Сложность**: 5/10
- **Обоснование**: μ=0.9-0.99 — стандарт для стохастической оптимизации. Уменьшит осцилляции.

### SN-8: Concept-Error Adaptive Destabilization

- **Суть**: Заменить глобальный `destab_scale` на адаптивный per-concept: P(destab|cid) = clamp(concept_error[cid] * 0.5, 0.0, 0.5). Амплитуду шума тоже привязать к ошибке: `mix = min(concept_error[cid] * 0.5, 0.5)`. Высоко-ошибочные concept получают больше шума, низко-ошибочные — меньше.
- **Приоритет**: P3
- **Сложность**: 4/10
- **Обоснование**: Разрушительная стабилизация нужна тем, кто ушёл в wrong local minimum. Хорошо обученные concept не должны дестабилизироваться.

### Статус старых предложений SN-* из V3:

- **SN-1 Multi-Resolution Fields**: не реализован. Рекомендуется заменить на SN-6 (конкретный дизайн).
- **SN-2 Adaptive Destabilization**: частично — destab scale затухает по расписанию, но не адаптируется к concept_error. Полное описание в SN-8.
- **SN-3 PPMI-Weighted Contrastive**: реализован — `_contrastive_objective` использует PPMI-фильтр и concept-error weighting. Критический баг: работает на CPU даже в GPU-режиме (см. SN-5).

---

# 3. GPU-Opt Agent: GPU-оптимизация и новые методы

### G-9: FP16 Storage with FP32 Compute (`half()`)

- **Суть**: Хранить `_vecs_t` и `_fb_t` в FP16, приводить к FP32 только на время matmul. Pascal CC 6.1 поддерживает FP16 storage (без tensor cores), но 2x меньше VRAM и bandwidth. `_vecs_t`: 224MB → 112MB. Единственный способ уместить большие batch в 2GB VRAM.
- **Приоритет**: P1
- **Сложность**: 4/10
- **Ожидаемый прирост**: -50% VRAM, ~20% bandwidth saving, возможность batch 1024+ вместо 500

### G-10: Pre-allocated Ping-Pong GPU Buffers (zero allocation)

- **Суть**: Заменить `torch.from_numpy(...).to(dev)` на pre-allocated `torch.empty(V, D, device=dev)` + `.copy_(src)` в цикле. Убрать тройную аллокацию (CPU temp → GPU temp → assign). Критично для 2GB — фрагментация убивает.
- **Приоритет**: P1
- **Сложность**: 5/10
- **Ожидаемый прирост**: ~0 allocation noise, нет OOM от фрагментации

### G-11: Async CPU→GPU Pipeline (Double-Buffered Batch)

- **Суть**: Пока GPU считает STDP на batch N, CPU строит пары для batch N+1. Использовать `torch.cuda.Stream` для перекрытия compute и transfer. Pair-building на CPU — главный bottleneck (Python loop с math.exp).
- **Приоритет**: P1
- **Сложность**: 7/10
- **Ожидаемый прирост**: 2x throughput (CPU и GPU работают параллельно)

### G-12: Fused Negative Sampling Scatter (одна scatter_add вместо двух)

- **Суть**: Сейчас negative sampling делает две scatter_add: `acc_shifts` и `acc_elr`. Хранить shifts и elr в одном тензоре структуры (D+1) и делать одну scatter_add.
- **Приоритет**: P2
- **Сложность**: 6/10
- **Ожидаемый прирост**: ~15% faster neg sampling

### G-13: Chunked Full-V Matmul для Lateral Inhibition (OOM-safe)

- **Суть**: `gv_t @ gv_all.T` — V×V matmul. На 2GB при V=146K это OOM (146K×146K×4 = 85GB). Разбить gv_all на чанки по 10K, matmul последовательно, `torch.cat` результатов.
- **Приоритет**: P2
- **Сложность**: 5/10
- **Ожидаемый прирост**: Предотвращение OOM, возможность GPU lateral inhibition для больших V

### G-14: Sparse Approximate Inhibition (FAISS-style, без полной матрицы)

- **Суть**: Вместо `gv_t @ gv_all.T` (квадратично) использовать `topk` через chunked dot product или FAISS GPU. Для inhibition достаточно найти top-100 похожих.
- **Приоритет**: P2
- **Сложность**: 8/10
- **Ожидаемый прирост**: O(V²) → O(V*log(k)) для inhibition

### G-15: CUDA Events для профилирования + NVTX Ranges

- **Суть**: Добавить `torch.cuda.Event(enable_timing=True)` вокруг _gpu_stdp_apply, evaluate, _negative_sampling_gpu. Использовать `torch.cuda.profiler` и NVTX ranges для nsys. Сейчас нет ни одного замера GPU — всё `time.time()` (CPU time).
- **Приоритет**: P2
- **Сложность**: 2/10
- **Ожидаемый прирост**: Измеримость (нужно для понимания реального bottleneck)

### G-16: In-place Concept Error EMA на GPU (без CPU round-trip)

- **Суть**: Сейчас `avg_err_cpu` синхронизируется CPU, потом пишется в Python dict. Перенести EMA целиком на GPU: `self._ce_t = self._ce_t * decay + (1-decay) * avg_err_t`. Убрать `.cpu().numpy()` и Python loop.
- **Приоритет**: P2
- **Сложность**: 4/10
- **Ожидаемый прирост**: -1 sync point per batch, ~5% faster STDP

### G-17: Kernel Fusion для STDP pair_delta

- **Суть**: `pair_delta = vc * elr[:, None] - vg * (y * elr)[:, None]` создаёт 4 промежуточных тензора. Написать custom Triton/TorchScript kernel: один launch вместо 6.
- **Приоритет**: P3
- **Сложность**: 9/10
- **Ожидаемый прирост**: ~10% faster STDP kernel

### G-18: CUDAGraphs для Evaluate (повторяющийся matmul)

- **Суть**: Evaluate вызывает `pv_t @ V_gpu.T` сотни раз с одинаковыми размерностями. CUDAGraphs записывают один раз, проигрывают многократно.
- **Приоритет**: P3
- **Сложность**: 6/10
- **Ожидаемый прирост**: ~30% faster evaluate (CPU launch overhead → 0)

### G-19: WARP Shuffle-based Reduction для scatter_add на Pascal

- **Суть**: Pascal CC 6.1 поддерживает warp shuffle. scatter_add внутренне использует atomicAdd — медленно на 2GB карте. Можно заменить grouped reduction через custom CUDA kernel с warp-level primitives для small D=384.
- **Приоритет**: P3
- **Сложность**: 10/10
- **Ожидаемый прирост**: ~20% faster scatter_add path

### G-20: Torch.compile для Lateral Inhibition Loop

- **Суть**: Цикл `for gi, gen_cid in enumerate(gen_cids_list)` с per-gen_cid GPU ops. Обернуть тело цикла в `torch.compile`: векторизовать threshold, where, topk, gather, delta compute.
- **Приоритет**: P3
- **Сложность**: 5/10
- **Ожидаемый прирост**: ~15% faster inhibition loop

### Статус GN-* из V3:

- **GN-1 Fused STDP Kernel**: не реализован. G-17 — конкретный дизайн.
- **GN-2 GAGAT (Gradient Accumulation)**: не реализован. G-11 — альтернативный подход (double buffer).
- **GN-3 Approx Inhibition**: не реализован. G-14 — конкретный дизайн.

### Ключевая рекомендация GPU

MX550 2GB — пороговая ёмкость: `_vecs_t` уже 224MB, с batch 500 и временными тензорами evaluate достигает ~1800MB. **FP16 storage (G-9) + pre-allocated buffers (G-10) — обязательный минимум**, иначе OOM неизбежен при росте V или batch. Без них все остальные оптимизации бессмысленны.

---

# 4. Training-Dynamics Agent: Улучшения цикла обучения

### TN-1: Adaptive Curriculum 2.0 (Self-Paced Learning)

- **Суть**: Curriculum определяется не длиной строки, а **ошибкой предсказания концепта** (`concept_error`). Линии, где средняя ошибка ниже порога, допускаются раньше. Ramp curriculum не фиксированные 20%, а пока `mean_cos < 0.1 * TARGET`.
- **Приоритет**: P1
- **Сложность**: 6/10
- **Оценка эффекта**: Высокая — ускорит сходимость редких концептов, решит проблему потери редких концептов

### TN-2: Exponential Moving Average (EMA) над Concept Vectors

- **Суть**: Поддерживать EMA-копию весов (factor=0.999). На eval/generation использовать EMA-векторы. Стандартный трюк из deep learning (Polyak averaging). Активируется после N шагов.
- **Приоритет**: P1
- **Сложность**: 4/10
- **Оценка эффекта**: Средняя — стабилизирует генерацию, сглаживает осцилляции cos

### TN-3: Cosine Annealing with Warm Restarts

- **Суть**: Заменить `get_lr()` (warmup + flat) на cosine annealing с периодическими restart-ами. `LR = base_lr * cosine(epoch/restart_period)`. Restart каждые ~10K lines. Параметр `T_0` адаптивно растёт: если PPL не улучшилась за период → следующий restart быстрее.
- **Приоритет**: P1
- **Сложность**: 5/10
- **Оценка эффекта**: Высокая — мягкая сходимость, меньше осцилляций

### TN-4: Early Stopping + Best Checkpoint

- **Суть**: После каждого eval проверять `val_ppl`. Если не улучшилось за `patience=5` eval-ов → стоп. Сохранять лучший checkpoint по `val_ppl + vacc1*50`. `full_stuck` как сигнал к early stopping, а не только к fluctuate.
- **Приоритет**: P2
- **Сложность**: 3/10
- **Оценка эффекта**: Средняя — экономит время, решает T-6

### TN-5: Batch Size Warmup (Gradual Increase)

- **Суть**: `BATCH_SIZE=8` на старте → линейно до 64 к концу curriculum-фазы. Стандартная техника (Smith '18). Синхронизировать с ramp curriculum.
- **Приоритет**: P2
- **Сложность**: 2/10
- **Оценка эффекта**: Средняя — более стабильные градиенты в начале

### TN-6: Gradient Noise Injection

- **Суть**: В `_gpu_stdp_apply` добавить `grad += N(0, noise_scale * sqrt(lr))` в начале тренировки. `noise_scale` затухает к 0 по ходу обучения. Помогает выходить из локальных минимумов.
- **Приоритет**: P2
- **Сложность**: 2/10
- **Оценка эффекта**: Средняя — может деблокировать stuck-состояния

### TN-7: TensorBoard / Rich Dashboard

- **Суть**: Запись scalar-метрик через `torch.utils.tensorboard.SummaryWriter`: PPL, vPPL, acc1, vacc1, cos_mean/std, LR, histograms concept_error, PCA embedding projector. Рисовать `mean_cos`-трейд, распределение concept_error, per-параметр трекинг.
- **Приоритет**: P2
- **Сложность**: 4/10
- **Оценка эффекта**: Высокая (диагностика) — решает T-11

### TN-8: Adaptive Destab from Concept Error

- **Суть**: `per_cid_destab = destab_scale * (1 + concept_error[cid] * 2)`. Концепты с высокой ошибкой получают больше шума. Реализовать в `_gpu_stdp_apply` и `_cpu_stdp_apply`.
- **Приоритет**: P2
- **Сложность**: 3/10
- **Оценка эффекта**: Средняя-высокая — ускорит разучивание трудных концептов

### TN-9: Switched Eval (Fast + Full)

- **Суть**: Каждый `EVAL_EVERY/4` — fast eval (100 строк, только PPL). Каждый `EVAL_EVERY` — full eval (300 строк, все метрики). `ParameterOptimizer.step()` вызывается только на full eval. Между full eval-ами vacc1=None и counter не инкрементится.
- **Приоритет**: P2
- **Сложность**: 2/10
- **Оценка эффекта**: Высокая — решает остаточную проблему T-4 + меньше overhead

### TN-10: Decay Protection for Rare Concepts

- **Суть**: В `lattice.decay_all()` пропускать редкие концепты (freq < 3). Отдельный `rare_concept_protect=True` флаг в config. `cs.decay_usage(decay=0.98)` — заменить на per-concept decay: редкие концепты декаятся медленнее.
- **Приоритет**: P1
- **Сложность**: 1/10
- **Оценка эффекта**: Высокая — решает T-10 (loss of rare concepts)

### Статус TN-* из V3:

- **TN-1 Adaptive Curriculum**: не реализован. Текущий — примитивная BPE-сортировка + 20% ramp. См. TN-1 выше с self-paced learning.
- **TN-2 EMA**: не реализован. См. TN-2 выше.
- **TN-3 Contrastive Scheduling**: не реализован. `_contrastive_objective` работает всегда с фиксированной силой.

---

# 5. Quality-Safety Agent: Тестирование и качество кода

### QN-6: Параметризованный Stress-тест OOM на GPU

- **Суть**: Сгенерировать synthetic OOM через monkeypatch `torch.cuda.OutOfMemoryError` в `_ensure_torch` + проверить fallback на CPU. Параметризовать: размер концептов (1k, 10k, 100k).
- **Приоритет**: P1
- **Сложность**: 3/10
- **Обоснование**: OOM guard — ключевой safety-механизм, но тестируется только mocking-сценарий.

### QN-7: Fuzz-тест _branch с граничными значениями

- **Файл**: `crystal_generator.py:486`
- **Суть**: `_branch` принимает `seq`, `word_num`, `theta_temp`, `target_cid`, `centroid`. Fuzz: пустые seq, seq из одного элемента, `theta_temp=0`, `centroid=None`, `target_cid` вне диапазона.
- **Приоритет**: P1
- **Сложность**: 4/10
- **Обоснование**: `_branch` — сердце генерации. Все ошибки там приводят к падению `generate()`. 0 тестов.

### QN-8: Property-based тест generate

- **Файл**: `crystal_generator.py:272`
- **Суть**: `generate()` должен возвращать `GenerationResult` с `text != ""` (если не пустой seed). Использовать hypothesis или ручные property checks: `seed_word in text`, `len(concept_path) >= 2`, `score` конечен.
- **Приоритет**: P2
- **Сложность**: 6/10
- **Обоснование**: `generate()` — главный API. Без тестов регрессии незаметны.

### QN-9: Save/Load Checkpoint Roundtrip

- **Файлы**: `concept_space.py:724-806`, `syntax_lattice.py`
- **Суть**: Serialize → deserialize → assert равенство key properties (dim, vocab_size, vector norms). Параметризовать: `use_pq=True/False`, с/без field_bits, с/без бинарного npz.
- **Приоритет**: P1
- **Сложность**: 5/10
- **Обоснование**: Ни одного теста на save/load. Поломка бинарного формата останется незамеченной до рантайма.

### QN-10: build_octree_fields Correctness

- **Файл**: `concept_space.py:368-454`
- **Суть**: Проверить: H матрица симметрична, diag=0, значения в [0, H_max]. Field_bits корректны: `field_overlap(a,a) > 0`, prefix grouping не пропускает CID.
- **Приоритет**: P2
- **Сложность**: 5/10
- **Обоснование**: 30% кода ConceptSpace в этой функции. Ноль тестов.

### QN-11: HormonalSystem Unit Tests

- **Файл**: `hormonal_system.py:17-229`
- **Суть**: Проверить: `update(confidence=0.9, is_match=True)` снижает serotonin. `modulate_temperature` в корректном диапазоне. `save/load` roundtrip. Репетиция на 5 одинаковых gen_cid снижает da_coherence.
- **Приоритет**: P2
- **Сложность**: 4/10
- **Обоснование**: Гормональная система влияет на температуру и beam width. Без тестов семантические баги неизбежны.

### QN-12: Улучшение GPU/CPU Parity Tolerance

- **Текущее состояние**: `diff < 1.0` — около 40% от единичного вектора. Чрезвычайно либерально.
- **Суть**: Выяснить типичное расхождение (seed фиксация, одинаковый batch). Если расхождение < 0.01, снизить до `diff < 0.1`. Тест должен seed-фиксировать RNG до GPU и CPU вызовов.
- **Приоритет**: P2
- **Сложность**: 7/10
- **Обоснование**: Текущий порог 1.0 означает, что вектора могут быть почти ортогональными и тест пройдёт.

### QN-13: Memory Stress Тест для _build_torch_tensors

- **Файл**: `crystal_generator.py:170`
- **Суть**: Мониторинг пикового выделения памяти при `_ensure_torch`. Использовать `pytest.mark.slow` и `psutil`. Размер: vocabsize=146K, dim=384 → ~225 MB. Проверить, что не превышает 2x ожидаемого.
- **Приоритет**: P3
- **Сложность**: 5/10
- **Обоснование**: Поломка бинарного формата npz может вызвать неожиданный memory spike.

### QN-14: Fuzzing _apply_vector_update

- **Файл**: `concept_space.py:511-554`
- **Суть**: Подать: `v_new` не единичной нормы, `v_new` с NaN, `max_shift=0`, `max_shift=1e10`. Проверить: нет NaN в result, норма сохраняется.
- **Приоритет**: P2
- **Сложность**: 3/10
- **Обоснование**: Эта функция вызывается из всех STDP путей. NaN здесь — silent data corruption.

### QN-15: Boundary Test для FractalEncoding

- **Файл**: `fractal_encoding.py:37-54`
- **Суть**: `path(0)` → 16 нулей, `path(-1)` → корректные digits, `lcp` когда один path — префикс другого, `H_weighted` с `gamma=0` и `gamma=1`.
- **Приоритет**: P2
- **Сложность**: 2/10
- **Обоснование**: Полный пропуск тестов encoding модуля.

### Анализ тестового покрытия:

| Модуль | % покрытия | Статус |
|--------|-----------|--------|
| `ConceptVectorStore` | ~100% | ✅ |
| `FractalField` | ~60% | ⚠️ |
| `ConceptSpace` (core) | ~30% | ⚠️ |
| `crystal_generator` (STDP) | ~40% | ⚠️ |
| `crystal_generator` (generate) | 0% | ❌ |
| `crystal_generator` (_branch) | 0% | ❌ |
| `crystal_generator` (save/load) | 0% | ❌ |
| `syntax_lattice` | 0% | ❌ |
| `fractal_encoding` | 0% | ❌ |
| `hormonal_system` | 0% | ❌ |
| `parameter_optimizer` | ~90% | ✅ |
| `fcf_config` | ~80% | ✅ |

### Статус QN-* из V3:

- **QN-1 Parity**: ⚠️ Pass, но ослаблен (`diff < 1.0`). См. QN-12.
- **QN-2 Memory Stress**: ❌ Не реализован. См. QN-13.
- **QN-3 Boundary Fuzz**: ❌ Не реализован. См. QN-7, QN-14, QN-15.
- **QN-4 OOM Guard**: ✅ Реализован.
- **QN-5 Safety Config**: ✅ Реализован.

---

# 6. Итоговая матрица приоритетов V5

| ID | Метод / Улучшение | Приор. | Агент | Сложность | Тип |
|:--:|-------------------|:------:|:-----:|:---------:|:---:|
| **AM-6** | Generator-Trainer Separation | **P1** | Arch | 7 | Архитектура |
| **AM-9** | Training Pipeline Abstraction | **P1** | Arch | 8 | Архитектура |
| **SN-4** | Dual-Timescale Subspace-Kinetic STDP | **P1** | NS | 8 | Обучение |
| **SN-5** | GPU-Accelerated Contrastive Objective | **P1** | NS | 6 | Производительность |
| **G-9** | FP16 Storage with FP32 Compute | **P1** | GPU | 4 | Память |
| **G-10** | Pre-allocated Ping-Pong GPU Buffers | **P1** | GPU | 5 | Память |
| **G-11** | Async CPU→GPU Pipeline | **P1** | GPU | 7 | Производительность |
| **TN-1** | Adaptive Curriculum 2.0 (Self-Paced) | **P1** | TD | 6 | Обучение |
| **TN-2** | EMA над Concept Vectors | **P1** | TD | 4 | Стабильность |
| **TN-3** | Cosine Annealing with Warm Restarts | **P1** | TD | 5 | Обучение |
| **TN-10** | Decay Protection for Rare Concepts | **P1** | TD | 1 | Корректность |
| **QN-6** | Stress-тест OOM на GPU | **P1** | QA | 3 | Тестирование |
| **QN-7** | Fuzz-тест _branch | **P1** | QA | 4 | Тестирование |
| **QN-9** | Save/Load Checkpoint Roundtrip | **P1** | QA | 5 | Тестирование |
| **AM-7** | Async Checkpoint Manager | **P2** | Arch | 4 | Архитектура |
| **AM-8** | Configuration Schema Validation | **P2** | Arch | 3 | Надёжность |
| **SN-6** | Multi-Resolution Hierarchical Octree Fields | **P2** | NS | 6 | Улучшение |
| **SN-7** | Momentum-Accumulated STDP | **P2** | NS | 5 | Обучение |
| **G-12** | Fused Negative Sampling Scatter | **P2** | GPU | 6 | Производительность |
| **G-13** | Chunked Full-V Matmul (OOM-safe) | **P2** | GPU | 5 | Память |
| **G-14** | Sparse Approximate Inhibition | **P2** | GPU | 8 | Производительность |
| **G-15** | CUDA Events Profiling | **P2** | GPU | 2 | Мониторинг |
| **G-16** | In-place Concept Error EMA на GPU | **P2** | GPU | 4 | Производительность |
| **TN-4** | Early Stopping + Best Checkpoint | **P2** | TD | 3 | Качество |
| **TN-5** | Batch Size Warmup | **P2** | TD | 2 | Обучение |
| **TN-6** | Gradient Noise Injection | **P2** | TD | 2 | Обучение |
| **TN-7** | TensorBoard / Rich Dashboard | **P2** | TD | 4 | Мониторинг |
| **TN-8** | Adaptive Destab from Concept Error | **P2** | TD | 3 | Обучение |
| **TN-9** | Switched Eval (Fast + Full) | **P2** | TD | 2 | Производительность |
| **QN-8** | Property-based тест generate | **P2** | QA | 6 | Тестирование |
| **QN-10** | build_octree_fields Correctness | **P2** | QA | 5 | Тестирование |
| **QN-11** | HormonalSystem Unit Tests | **P2** | QA | 4 | Тестирование |
| **QN-12** | Улучшение GPU/CPU Parity Tolerance | **P2** | QA | 7 | Тестирование |
| **QN-14** | Fuzzing _apply_vector_update | **P2** | QA | 3 | Тестирование |
| **QN-15** | Boundary Test FractalEncoding | **P2** | QA | 2 | Тестирование |
| **AM-10** | AdaptiveErrorTracker | **P3** | Arch | 3 | Косметика |
| **AM-11** | FieldBit Lazy Reconstruction | **P3** | Arch | 2 | Память |
| **AM-12** | Shared RNG Registry | **P3** | Arch | 2 | Надёжность |
| **SN-8** | Concept-Error Adaptive Destabilization | **P3** | NS | 4 | Улучшение |
| **G-17** | Kernel Fusion STDP pair_delta | **P3** | GPU | 9 | Производительность |
| **G-18** | CUDAGraphs для Evaluate | **P3** | GPU | 6 | Производительность |
| **G-19** | WARP Shuffle Reduction scatter_add | **P3** | GPU | 10 | Производительность |
| **G-20** | Torch.compile Lateral Inhibition | **P3** | GPU | 5 | Производительность |
| **QN-13** | Memory Stress Test | **P3** | QA | 5 | Тестирование |

---

# 7. Рекомендации по внедрению

## Фаза 1 (P1, срочные улучшения — 15 методов)

| # | Метод | Ожидаемый эффект |
|:-:|-------|-----------------|
| 1 | **QN-9** Save/Load Roundtrip | Защита от silent data loss |
| 2 | **QN-6** OOM Stress | Верификация safety-механизма |
| 3 | **QN-7** _branch Fuzz | Защита основного API генерации |
| 4 | **G-9** FP16 Storage | -50% VRAM, критично для 2GB |
| 5 | **G-10** Pre-allocated Buffers | Нулевая фрагментация GPU |
| 6 | **TN-10** Rare Concept Protection | Сохранение редкой лексики |
| 7 | **TN-3** Cosine Annealing | Стабильная сходимость |
| 8 | **TN-1** Self-Paced Curriculum | Ускорение обучения редких концептов |
| 9 | **TN-2** EMA Vectors | Стабилизация генерации |
| 10 | **SN-5** GPU Contrastive | 10-15x ускорение contrastive |
| 11 | **G-11** Async Pipeline | 2x throughput CPU+GPU |
| 12 | **SN-4** Dual-Timescale STDP | Эффективное использование кода |
| 13 | **AM-6** Generator-Trainer Separation | Maintainability |
| 14 | **AM-9** Pipeline Abstraction | Testability |

## Фаза 2 (P2, улучшения — 20 методов)

Включает: AM-7, AM-8, SN-6, SN-7, G-12–G-16, TN-4–TN-9, QN-8, QN-10–QN-12, QN-14, QN-15.

## Фаза 3 (P3, нишевые — 9 методов)

Включает: AM-10, AM-11, AM-12, SN-8, G-17–G-20, QN-13.

---

*Отчёт составлен коллегией AI-агентов: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent*
