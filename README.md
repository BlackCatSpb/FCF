# FCF — Fractal Cognitive Field

## English

**FCF** is a neuro-symbolic language model that learns without gradient descent. Meaning emerges as self-organization of a vector field in an octant (3D-fractal) coordinate system, driven entirely by local plasticity rules — STDP, lateral inhibition, contrastive divergence, and centroid attraction.

Not a Transformer. Not a word embedding. Every BPE token IS a concept, and every concept is a point on the unit hypersphere in 384-dimensional space. Coordinates are determined by a fractal code (non-repeating octant digits per level) projected through an orthonormal basis.

---

### How It Learns

FCF replaces backpropagation with five complementary plasticity mechanisms that operate directly on concept vectors:

| Mechanism | What it does |
|-----------|--------------|
| **STDP** | If token A precedes token B, B's vector shifts toward A. Co-occurrence strength determines pull magnitude. All pairs in a micro-batch processed as a single GPU operation. |
| **Negative Sampling** | Random concepts are pushed away from each updated concept. Push strength is weighted by per-concept prediction error — harder concepts receive stronger regularization. Field gates filter invalid negatives. |
| **Contrastive Objective** | Hard negative mining over the top-K most similar concepts. Those that are similar but neither co-occur nor share field proximity are pulled apart. Cross-field pairs (different octree regions) are repelled aggressively; within-field pairs are treated gently to preserve cluster structure. |
| **Lateral Inhibition** | All concepts updated in a batch repel each other along the sphere's geodesic. Prevents representational collapse — if every vector becomes identical, all information is lost. GPU-vectorized. |
| **Centroid Pull** | All tokens in a sentence are weakly pulled toward their mean. Functions as a sentence-level regularizer, ensuring co-occurring concepts share directional signal. GPU-accelerated with deferred write-back. |

### Subspace-Kinetic Representation

Every concept vector is a projection of a **fractal code** through an orthonormal basis. The code space is split into three subspaces:

- **z_c** (content) — stable semantics, slow learning rate
- **z_a** (activity) — contextual behavior, fast learning rate
- **z_m** (morphology) — grammatical form, medium learning rate

Gradients are projected back into code space, masked per subspace, and each subspace is updated at its own rate before reconstructing the vector. This prevents catastrophic forgetting: content remains stable while activity adapts rapidly.

### Fractal Coordinates

Codes are sequences of non-repeating digits 0-7 (octants), each digit representing a binary choice at one level of a 3D fractal subdivision. Two concepts whose codes share octant digits are spatially proximate in the field. This overlap serves as a **field gate** modulating all learning signals:

- **No overlap** (different fields) → aggressive repulsion
- **Partial overlap** (same super-field) → gentle treatment
- **Same node** (same concept) → skip

### Adaptive Plasticity

All learning signals are modulated by three adaptive controllers:

1. **Parameter optimizer** — tracks cosine trends, perplexity, and concept churn to adjust learning rate, PMI gate, negative sampling ratio, inhibition strength, and contrastive weight in real time. Uses momentum-based adaptation with plateau detection.

2. **Per-concept error tracking** — each concept maintains an EMA of its prediction error. High-error concepts receive a lower PMI threshold (more STDP pairs), stronger negative reweighting, and higher contrastive learning rate. Low-error concepts converge and stabilize.

3. **Self-paced curriculum** — after each checkpoint, remaining training lines are rescored by prediction difficulty and re-sorted hardest-first. The model faces increasingly challenging material as it improves.

### Training Dynamics

Training proceeds in epochs with a continuous curriculum:
- **Warmup**: linear ramp of learning rate, batch size, context window, and negative samples over the first portion of data
- **Main phase**: full capacity with periodic evaluation and fluctuation
- **Fluctuation**: Langevin-style drift injects noise into the fractal null space, modulated by PPMI-weighted perturbations. Periodically breaks the model out of local attractors.
- **Gradient momentum**: persistent buffer blends gradients across micro-batches to reduce variance from sparse co-occurrence.

Evaluation uses a switched scheme — fast eval (vec-perplexity only) most steps, full eval (all metrics) every N steps — reducing overhead by ~4x.

---

### GPU Implementation

Designed for a **2GB VRAM consumer GPU** — a deliberate constraint to prove viability on accessible hardware:

- FP16 storage for concept vectors, FP32 for operations
- All STDP pairs in a micro-batch processed as a single kernel
- Pre-computed masks (co-occurrence, field overlap) reused across negative sampling and contrastive passes
- Persistent GPU tensors for momentum, concept error, field bits — no per-step reallocation
- Deferred synchronization: all updates accumulated on GPU, single batched write-back to CPU
- Fused post-STDP: contrastive, negative sampling, and centroid pull share one similarity matrix computation
- Zero-copy vector write-back via GPU-to-GPU hooks with persistent event synchronization
- Optional kernel fusion (PyTorch 2.0+)

---

### Data Pipeline

```
Input text
  →
SentencePiece BPE (146K tokens) → token IDs
  →
Pair building (context window 2-5):
  distance-weighted · frequency-weighted · PMI-gated · field-gated · slow-decayed
  →
GPU micro-batch:
  STDP apply → Negative sampling → Contrastive → Centroid pull → Lateral inhibition → Fluctuation
  →
Per-concept EMA update → Parameter optimizer step → Async checkpoint save
```

---

### Motivation

FCF asks: *can semantic space be built without gradient descent, using only local plasticity rules and a fractal coordinate system?*

If yes, it opens a path to fully interpretable language models:

- **Why this token?** Its vector is nearest in a manifold shaped by traceable STDP updates from specific co-occurrence events.
- **Why are two concepts similar?** They co-occurred in measurable contexts, and lateral inhibition prevented collapse.
- **Why is the model uncertain?** That concept's prediction error is high — tracked per-concept, dynamically modulating its own learning signals.

Every number in FCF corresponds to a physical fact about the data — not a latent activation in an uninterpretable deep network.

| Aspect | Transformer | FCF |
|--------|------------|-----|
| Learning | Backpropagation (global) | STDP + contrastive (local) |
| Representation | Latent activations | Unit sphere vectors |
| Interpretability | Post-hoc (attention maps) | By-construction (geometry) |
| Hardware | 80GB H100+ | 2GB MX550 |
| Training cost | Millions of $ | Electricity bill |
| Maturity | Production | Research prototype |

### Status

- Full training pipeline operational: batch processing, checkpoint/resume, switched evaluation
- All five plasticity mechanisms GPU-accelerated with subspace-kinetic learning
- Field-aware contrastive decoupling with octant overlap masks
- Self-paced curriculum, momentum accumulation, adaptive error tracking, homeostatic regulation
- 105 automated tests
- Corpus: 153K lines, 30M characters, Russian language, SentencePiece 146K BPE

---

### Quick Start

```bash
python train_full.py --epochs 3 --fresh    # fresh training
python train_full.py --epochs 3 --fast     # fast start, elevated LR
python train_full.py --epochs 3 --resume   # resume from checkpoint
```

Requires: Python 3.8+, PyTorch (optional, CPU fallback), SentencePiece, NumPy, scikit-learn.

---

## Русский

**FCF** — нейро-символическая языковая модель, которая обучается без градиентного спуска. Смысл возникает как самоорганизация векторного поля в октантной (3D-фрактальной) системе координат. Движущие силы — исключительно локальные правила пластичности: STDP, латеральное торможение, контрастивная дивергенция и центроидное притяжение.

Это не трансформер. Это не эмбеддинг слов. Каждый токен SentencePiece — концепт, и каждый концепт — точка на единичной гиперсфере в 384-мерном пространстве. Координаты задаются фрактальным кодом (неповторяющиеся восьмеричные цифры на уровень), спроецированным через ортонормированный базис.

---

### Как это учится

Обратное распространение заменено пятью механизмами пластичности:

**STDP** — если токен A предшествует B, вектор B притягивается к A. Сила притяжения — от частоты совместной встречаемости. Все пары микробатча — одной GPU-операцией.

**Негативная выборка** — случайные концепты отталкиваются от обновлённого. Сила взвешивается ошибкой предсказания концепта: чем хуже модель его знает, тем сильнее регуляризация. Полевые ключи отфильтровывают недопустимые негативы.

**Контрастивная цель** — жёсткий майнинг среди top-K ближайших концептов. Похожие, но не встречающиеся вместе и не связанные полем — разносятся. Межполевые пары (разные октанты) — агрессивно; внутриполевые — мягко, сохраняя кластеры.

**Латеральное торможение** — все обновлённые в батче концепты отталкиваются по геодезической сферы. Предотвращает коллапс: если все векторы станут одинаковыми, информация потеряна.

**Центроидное притяжение** — токены предложения слабо притягиваются к общему центру. Регуляризатор уровня предложения.

### Подпространственно-кинетическое представление

Вектор концепта — проекция **фрактального кода** через ортонормированный базис. Пространство кода разделено на три подпространства:

- **z_c** (содержание) — стабильная семантика, низкая скорость
- **z_a** (активность) — контекстуальное поведение, высокая скорость
- **z_m** (морфология) — грамматическая форма, средняя скорость

Градиенты проецируются в код, маскируются по подпространствам, каждое обновляется со своей скоростью. Содержание стабильно, активность быстро адаптируется — катастрофического забывания нет.

### Фрактальные координаты

Коды — неповторяющиеся восьмеричные цифры 0-7, каждая — бинарный выбор на одном уровне 3D-фрактала. Совпадающие цифры = пространственная близость. Совпадение служит **полевым ключом**:

- **Нет совпадения** (разные поля) → агрессивное отталкивание
- **Частичное** (общее надполе) → мягко
- **Тот же узел** → пропуск

### Адаптивная пластичность

Три системы модулируют все обучающие сигналы:

1. **Оптимизатор параметров** — отслеживает косинус, перплексию, обновления концептов; на лету настраивает скорость обучения, PMI-ключи, интенсивность выборки.

2. **Поконцептный трекер ошибок** — EMA ошибки на концепт. Высокая ошибка → ниже PMI-порог (больше пар), сильнее негативная выборка, выше контрастивная скорость. Низкая ошибка → стабилизация.

3. **Самостоятельный учебный план** — после чекпойта строки сортируются от трудных к лёгким. Модель получает всё более сложный материал.

### Процесс

- **Разогрев**: линейный рост скорости, батча, окна, негативных семплов
- **Основная фаза**: полная мощность, периодические оценка и флуктуация
- **Флуктуация**: ланжевеновский дрейф с PPMI-шумом, выбивает из локальных аттракторов
- **Момент градиента**: смешивание через микробатчи, уменьшение дисперсии
- **Оценка**: быстрая (только перплексия) на большинстве шагов, полная — каждый N-й

---

### GPU

Спроектировано для **2GB VRAM** — сознательное ограничение:

- FP16-хранение, FP32-операции
- Все пары STDP — одним ядром
- Предвычисленные маски повторно используются
- Постоянные GPU-тензоры без перевыделения
- Отложенная синхронизация: все обновления на GPU, одна пакетная запись на CPU
- Слияние пост-STDP: контрастив, выборка, центроид — одна матрица сходства
- Запись GPU→GPU без копирования на CPU

---

### Мотивация

FCF — эксперимент: *можно ли построить семантическое пространство без градиентного спуска, на одних локальных правилах пластичности и фрактальной системе координат?*

Если да — полностью интерпретируемые языковые модели:

- **Почему этот токен?** Его вектор ближайший в многообразии, сформированном прослеживаемыми STDP-обновлениями.
- **Почему концепты похожи?** Они встречались вместе в измеримых контекстах, латеральное торможение не дало им разойтись.
- **Почему модель неуверена?** Ошибка этого концепта высока — отслеживается индивидуально.

Каждое число в FCF — физический факт о данных, не латентная активация.

| Аспект | Трансформер | FCF |
|--------|------------|-----|
| Обучение | Обратное распространение | STDP + контрастив |
| Представление | Латентные активации | Векторы на сфере |
| Интерпретируемость | Пост-хок (карты внимания) | По построению (геометрия) |
| Оборудование | 80GB H100+ | 2GB MX550 |
| Стоимость | Миллионы $ | Счёт за электричество |

### Текущее состояние

- Конвейер обучения: батчи, чекпойнты, возобновление, оценка
- Все 5 механизмов пластичности на GPU
- Поле-осознанное контрастивное разделение
- Самостоятельный учебный план, момент, трекер ошибок, гомеостаз
- 105 тестов
- Русский язык, 146K BPE, 153K строк

---

### Быстрый старт

```bash
python train_full.py --epochs 3 --fresh    # с нуля
python train_full.py --epochs 3 --fast     # быстро, повышенный LR
python train_full.py --epochs 3 --resume   # продолжить
```

Зависимости: Python 3.8+, PyTorch (опционально), SentencePiece, NumPy, scikit-learn.

---

*FCF — исследовательский проект. Эксперименты и вопросы приветствуются.*
