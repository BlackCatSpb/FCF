# EVA — Топологический ИИ нового поколения

> **Не LLM. Не статистический попугай.**  
> Символ ≡ координата в ℝ¹²⁸. Текст ≡ траектория. Трансформер ≡ навигатор.  
> Знания — в рекурсивном тензоре потенциалов [160×160×160] × K=6 субвекторов, топологии путей и TrajectoryStore.  
> **~5.5M параметров. 160 токенов. 0.7 GB VRAM. Полный цикл train→think→generate.**

---

## Идея за 30 секунд

Обычный ИИ: «привет» → токен #4521 → вектор → угадать следующий токен.

**EVA**: «привет» → 6 точек в ℝ¹²⁸ → траектория → **рекурсивная декомпозиция координаты** на K=6 субвекторов → каждый квантизуется к ближайшему символу → bias от тензора потенциалов на каждом уровне → взвешенная сумма → генерация.

Модель не предсказывает следующее слово. Она **навигирует в координатном пространстве**, раскладывает каждый шаг на иерархию субвекторов и использует bias от каждого уровня рекурсии.

---

## Текущее состояние

**Training** на **full_corpus_encoded.npy** (106.5M токенов, 680K предложений).  
Resume с чекпоинта, RecursiveTensorPotentialField, B=12, seq=192, composition loss с diversity penalty.

| Параметр | Значение |
|----------|----------|
| Ядро трансформера | 6 HybridFractalBlock слоёв, 128-dim, 32 heads |
| RecursiveTPF | base [160×160×160] + decomp/compose/gate/depth_scale — 4,293,383 params |
| WordValenceField | outer-product (left×right) — 74K params |
| Всего параметров | ~5.49M |
| Батч | B=12, seq=192 (обрезка до `</S>` → `</W>`) |
| Шагов | 100 000 |
| Оптимизатор | AdamW (lr=5e-3, cosine → 0) |
| Loss | CE + Group×0.05 + TrajLoss×0.1 + Composition×0.01 |
| Composition loss | MSE(recon, x) + 0.1·diversity − 0.01·H(gates) |
| Depth scale | per-level vector[8] (learnable) |
| VRAM | ~0.7 GB (MX550) |
| Скорость | ~160 шагов/мин |
| Recovery | strict=False + shape-mismatch skip + P → base_tpf.P migration |

---

## Архитектура (5.49M params, 21 инновация)

```
Текст → CharacterVocab (boundary-токены) → CoordinateEmbedding [ℝ¹²⁸]
  → MultiSubspaceEmbedding → RoPE → HybridFractalBlock ×6
  → RMSNorm → CoordinateDecoder (SubHSM) → Текст

Генерация:
  CE_logits / temperature
    + RecursiveTPF.recursive_bias(x=hidden[-1], context=ids) × 0.1
      └─ BFS quantize на K=6, max_depth=8, max_cap=4096
      └─ gate-filter (g_k > 0.05) + per-level depth_scale vector[8]
      └─ weighted sum across all paths and levels
    + WVF.get_valence_bias(word_coord, context) × 0.05
    → top-20 → adaptive repetition penalty (p × 0.5^freq, global)
    → softmax → multinomial

Thinking Phase (каждые 500 шагов):
  Co-occurrence affinity (500 samples × 128 tok) → init TPF
  Capture attention → update TPF (10 blocks)
  Extract trajectories → consolidate → store (5 blocks)
  Build topology → Fast Path cache

Composition Loss (каждый шаг):
  hidden [B,L,128] → RecursiveTPF.composition_loss()
    = MSE(compose(decompose(x)), x) + 0.1·diversity − 0.01·H(gates)
```

### Параметры по компонентам

| Компонент | Параметров | Доля |
|-----------|-----------|------|
| **HybridFractalBlock ×6** | ~1,140,000 | 20.8% |
| **RecursiveTensorPotentialField** | **4,293,383** | **78.3%** |
| └ base TPF [160×160×160] | 4,096,000 | 74.7% |
| └ decomp_proj (128→768) | 98,304 | 1.8% |
| └ compose_proj (768→128) | 98,304 | 1.8% |
| └ gate_net (128→6) | 774 | 0.01% |
| └ depth_scale [8] | 1 | <0.01% |
| **WordValenceField** | 74,305 | 1.4% |
| **TrajectoryPredictor** | 16,576 | 0.3% |
| **ConsolidationTransformer** | 6,529 | 0.1% |
| **StaticTopologyLayer** | ~500 | <0.01% |
| **Other** (embeds, norms, decoders) | ~40K | 0.7% |

### 21 инновация

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
| 20 | **Adaptive Repetition Penalty** | 0 | p × 0.5^freq по ВСЕМ токенам |
| 21 | **RecursiveTensorPotentialField** | 197,383 | Координата → K=6 субвекторов → BFS quantize → bias на каждом уровне |

---

## Быстрый старт

```bash
git clone https://github.com/BlackCatSpb/FCF.git
cd FCF
pip install torch numpy scikit-learn

# Кодирование корпуса (172 MB → 106M токенов)
python encode_full_corpus.py

# Обучение (автоматический resume из latest, strict=False + migration)
python train_full_corpus.py
```

Чекпоинты: `checkpoints/symbolic/full_latest.pt` (каждые 500 шагов), `full_best.pt` (по curvature).

---

## Структура проекта

```
FCF/
├── train_full_corpus.py          ← Обучение (full corpus, RecursiveTPF, composition loss)
├── encode_full_corpus.py         ← Кодирование full_corpus_ru.txt → .npy
├── eva.bat                       ← Ярлык запуска (FCF.lnk → eva.bat)
│
├── eva/symbolic/
│   ├── unified_transformer.py    ← Координатный трансформер + enhanced_generate
│   ├── adaptive_fractal.py       ← AdaptiveAttention + LevelController
│   ├── fractal_conv.py           ← HybridFractalBlock + SGF + CoordStream
│   ├── potential_fields.py       ← RecursiveTPF + TPF + WVF + SCF + SRG + KCA
│   ├── trajectory_store.py       ← Иерархическая память + ConsolidationTransformer
│   ├── static_topology.py        ← Топология [160,160,3] + Fast Path
│   ├── subspace_coords.py        ← MultiSubspace + WordWeightEncoder
│   ├── char_vocab.py             ← 160 токенов + boundary-разметка
│   ├── self_reflection.py        ← Анализ траекторий (кривизна, confidence)
│   └── validation_suite.py       ← Автоматическая валидация
│
├── docs/
│   ├── ARCHITECTURE.md           ← Полная архитектура (этот файл)
│
├── real_data/
│   ├── full_corpus_ru.txt        ← Исходный текст (172 MB)
│   └── full_corpus_encoded.npy   ← Закодированный корпус (106M токенов)
│
└── checkpoints/symbolic/
    ├── full_latest.pt            ← Последний чекпоинт
    ├── full_best.pt              ← Лучший по curvature
    └── trajectory_store_full.pkl ← Память траекторий
```

---

## Ключевые отличия от LLM

| | LLM (GPT, LLaMA) | EVA |
|---|---|---|
| **Принцип** | Статистическое угадывание | Навигация + рекурсивные потенциалы |
| **Параметры** | 1B–1.7T | **~5.5M** |
| **VRAM** | 8-80 GB | **0.7 GB** |
| **Знания** | В весах (неотделимы) | Топология + RecursiveTPF + TrajectoryStore |
| **Bias-структура** | Нет (raw CE) | **Рекурсивная: BFS quantize на K=6, max_depth=8** |
| **Ретенция** | Context window | Внешняя память траекторий (50K+) |
| **Масштабирование** | Больше параметров | Больше траекторий (модель не растёт) |
| **Обучение** | Недели, 8-256 GPU | Часы, 1 GPU (MX550) |

---

*«Модель не имитирует язык. Она строит карту символьного пространства, рекурсивно декомпозирует каждую координату на K=6 субвекторов, квантизует их на всех уровнях BFS-дерева, заполняет тензор потенциалов и генерирует ответ, взвешивая bias от каждого пути рекурсии.»*
