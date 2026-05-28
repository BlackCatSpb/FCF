# EVA — Топологический ИИ нового поколения

> **Не LLM. Не статистический попугай.**  
> Символ ≡ координата в ℝ¹²⁸. Текст ≡ траектория. Трансформер ≡ навигатор.  
> Знания — в геометрии пространства, рекурсивном тензоре потенциалов и топологии путей, не в весах.  
> **~5.5M параметров. 160 токенов. Полный цикл train→think→generate.**

---

## Идея за 30 секунд

Обычный ИИ: «привет» → токен #4521 → вектор → угадать следующий токен.

**EVA**: «привет» → 6 точек в ℝ¹²⁸ → траектория → рекурсивная декомпозиция координаты → bias от тензора потенциалов на каждом уровне иерархии → генерация.

Модель не «предсказывает следующее слово». Она **навигирует в координатном пространстве**, где каждый символ — точка, каждое слово — путь, а знание — тензор потенциалов [160×160×160] + рекурсивная декомпозиция на K=6 субвекторов + топология всех возможных путей.

---

## Текущее состояние

**Training** на **full_corpus_encoded.npy** (106.5M токенов, 680K предложений).  
Resume с чекпоинта ~step 6500, RecursiveTF вместо flat TPF.

| Параметр | Значение |
|----------|----------|
| Ядро трансформера | 6 HybridFractalBlock слоёв, 128-dim, 32 heads |
| TensorPotentialField | [160,160,160] — 4M params (base) |
| RecursiveTPF | base + decomp / compose / gate — 4,293,383 params |
| WordValenceField | outer-product — 74K params |
| Всего параметров | ~5.49M |
| Батч | B=8, seq=128 (обрезка до `</S>` → `</W>`) |
| Шагов | 100 000 |
| Оптимизатор | AdamW (lr=5e-3, cosine → 0) |
| VRAM | ~0.7 GB (MX550) |
| Скорость | ~160 шагов/мин |
| Recovery | strict=False — старый TPF.P → base_tpf.P |

---

## Архитектура

```
Текст → CharacterVocab (boundary-токены) → CoordinateEmbedding [ℝ¹²⁸]
  → MultiSubspaceEmbedding → RoPE → HybridFractalBlock ×6
  → RMSNorm → CoordinateDecoder (SubHSM) → Текст

Генерация:
  CE_logits
    + RecursiveTPF.recursive_bias(x=hidden, context=ids) × 0.1
    + WVF.get_valence_bias(word_coord, context) × 0.05
    → top-20 → adaptive repetition penalty (p × 0.5^freq)
    → softmax → multinomial

Thinking Phase (каждые 500 шагов):
  Co-occurrence affinity (500 samples) → RecursiveTPF.init_from_affinity()
  Capture attention → TPF.update() (10 блоков)
  Extract trajectories → TrajectoryStore.consolidate()
  Build topology → Fast Path cache

Composition Loss (каждый шаг):
  hidden [B,L,D] → RecursiveTPF.composition_loss()
    = MSE(decompose → compose, identity) - 0.01 × gate_entropy
```

| Компонент | Параметры |
|-----------|-----------|
| **HybridFractalBlock ×6** | ~1.14M |
| **RecursiveTensorPotentialField** | 4,293,383 |
| — base TensorPotentialField [V×V×V] | 4,096,000 |
| — decomp_proj (128→K·128) | 98,304 |
| — compose_proj (K·128→128) | 98,304 |
| — gate_net (128→6→softmax) | 774 |
| — depth_scale (learnable) | 1 |
| **WordValenceField** | 74,305 |
| **TrajectoryPredictor** | 16,576 |
| **ConsolidationTransformer** | 6,529 |
| **SentenceContextField** | 3 |
| **StaticTopologyLayer** | ~500 |
| **All other** (embeds, norms, decoders) | ~40K |

## Инновации (21 total)

| # | Инновация | Параметров | Суть |
|---|-----------|-----------|------|
| 1 | **CharacterVocab** | 0 | Символьный словарь с boundary-токенами `<W></W><S></S>` |
| 2 | **CoordinateEmbedding** | 0 | Символ → точка в ℝ¹²⁸, не lookup-таблица |
| 3 | **FractalConv2D** | В блоке | Causal свёртка по L × Dim с dilation 1,2,4,8 |
| 4 | **AdaptiveAttention** | В блоке | Динамический аллокатор голов по уровням сложности |
| 5 | **HybridFractalBlock** | ~190K | Conv2D + Attention + Gate Merge + SGF + CoordStream |
| 6 | **StaticTopologyLayer** | ~500 | [160,160,3] + Fast Path cache (10K) |
| 7 | **SubHSM** | 516 | 4-group hierarchical softmax + bias |
| 8 | **CoordBias** | 0 | L2-distance bias в attention scores |
| 9 | **Coordinate Residual Stream** | 1,536 | Сквозной поток координат через все слои |
| 10 | **SGF — Subspace-Gated FFN** | 3,072 | 4 gate-вектора + роутер для разных подпространств |
| 11 | **TrajLoss** | 16,576 | Aux loss на предсказание следующей координаты |
| 12 | **Learnable Consolidation** | 6,529 | Conv1d + gate для взвешенной консолидации |
| 13 | **Dilated KV-Cache** | 0 | Прореженный кэш для экономии VRAM |
| 14 | **TensorPotentialField** | 4,096,000 | [V×V×V] — динамический символьный потенциал |
| 15 | **WordValenceField** | 74,305 | Outer-product MLP → valence matrix |
| 16 | **SentenceContextField** | 3 | RBF-поле из top-K центров внимания |
| 17 | **SemanticRelevanceGate** | 0 | Фильтр по similarity + entropy + ethics |
| 18 | **GradientFlowSolver** | 0 | Langevin + oscillation damping |
| 19 | **KCACycle** | 0 | Adam-коррекция латентного кода |
| 20 | **Adaptive Repetition Penalty** | 0 | p × 0.5^freq по всем токенам |
| 21 | **RecursiveTensorPotentialField** | 197,383 | Координата → K=6 субвекторов → BFS quantize → bias на каждом уровне |

---

## Быстрый старт

```bash
git clone https://github.com/BlackCatSpb/FCF.git
cd FCF
pip install torch numpy scikit-learn

# Кодирование корпуса
python encode_full_corpus.py

# Обучение с Resume (старый TPF.P → base_tpf.P)
python train_full_corpus.py
```

Чекпоинты: `checkpoints/symbolic/full_latest.pt` (каждые 500), `full_best.pt` (по curvature).  
Resume автоматически мигрирует старый TPF.P в RecursiveTPF.base_tpf.P.

---

## Структура проекта

```
FCF/
├── train_full_corpus.py         ← Основное обучение (106M токенов, RecursiveTPF)
├── encode_full_corpus.py        ← Кодирование текста в ID
│
├── eva/symbolic/
│   ├── unified_transformer.py   ← Координатный трансформер + enhanced_generate
│   ├── adaptive_fractal.py      ← AdaptiveAttention + LevelController
│   ├── fractal_conv.py          ← HybridFractalBlock + SGF + CoordStream
│   ├── potential_fields.py      ← RecursiveTPF + TPF + WVF + SCF + SRG + KCA
│   ├── trajectory_store.py      ← Иерархическая память + ConsolidationTransformer
│   ├── static_topology.py       ← Топология [160,160,3] + Fast Path
│   ├── subspace_coords.py       ← MultiSubspace + WordWeightEncoder
│   ├── char_vocab.py            ← 160 токенов + boundary-разметка
│   ├── self_reflection.py       ← Анализ траекторий (кривизна, confidence)
│   └── validation_suite.py      ← Автоматическая валидация
│
├── docs/
│   ├── ARCHITECTURE.md          ← Полное описание архитектуры
│   └── ...
│
├── real_data/
│   ├── full_corpus_ru.txt       ← Исходный текст (172 MB)
│   └── full_corpus_encoded.npy  ← Закодированный корпус (106M токенов)
│
└── checkpoints/symbolic/
    ├── full_latest.pt           ← Последний чекпоинт
    ├── full_best.pt             ← Лучший по curvature
    └── trajectory_store_full.pkl ← Иерархическая память
```

---

## Ключевые отличия от LLM

| | LLM (GPT, LLaMA) | EVA |
|---|---|---|
| **Принцип** | Статистическое угадывание | Навигация + рекурсивные потенциалы |
| **Параметры** | 1B–1.7T | **~5.5M** |
| **Знания** | В весах (неотделимы) | Топология + RecursiveTPF + TrajectoryStore |
| **Bias** | Нет (raw CE) | **RecursiveTPF + WVF + SubHSM** |
| **Bias-структура** | Одноуровневый | **Батчевый BFS quantize на K=6 субвекторах** |
| **Ретенция** | Context window | Внешняя память траекторий |
| **Масштабирование** | Больше параметров | Больше траекторий (модель не растёт) |
| **VRAM** | 8–80 GB | **~0.7 GB** |

---

*«Модель не имитирует язык. Она строит карту символьного пространства, рекурсивно декомпозирует каждую координату на субвекторы, заполняет тензор потенциалов на всех уровнях иерархии и учится перемещаться по этой карте, используя bias от каждого уровня рекурсии.»*
