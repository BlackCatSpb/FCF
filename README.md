# EVA — Explainable Vector Architecture

**Rule-constrained generation without frequency memorization.**

Каждый выбор — пересечение бинарных gates на ВСЕХ уровнях иерархии одновременно. Никаких весов, никакого обучения. Только SVD cosine similarity для семантического выбора внутри структурных стен.

## Принцип

```
Данные (текст) → BPE токены → Иерархия (глава/абзац/предложение/слово)
  → AssociationGraph (48 концептов, 12 мета-концептов, SVD 32-dim)
  → Gates (бинарные правила на всех уровнях)
  → INTERSECTION gates = valid_mask → SVD scores внутри стен → выбор
```

- **Никакой частотности**: все структурно возможные transitions равны (score=1.0)
- **Каждый выход объясним**: что повлияло (semantic, s_type, cross_sent, gates)
- **Gates только добавляются**: новое наблюдение → новый gate. Старые gates не изменяются.

## Архитектура

### 1. Иерархический парсер (`text_hierarchy.py`)

Война и Мир → 4 тома → 17 частей → 355 глав → 10,853 предложений.
Каждое предложение: BPE-токены, тип (statement/dialogue/question/exclamation/french),
позиция в иерархии.

### 2. AssociationGraph (`association_graph.py`)

Двухуровневая кластеризация type-2 токенов через SVD + K-means:

| Уровень | Размер | Описание |
|---------|--------|----------|
| Meta | 12 кластеров | Верхний уровень абстракции |
| Concept | 48 кластеров | Семантические группы слов |

Каждый type-2 токен (2442 шт.) имеет SVD-вектор (32-dim) и принадлежит
концепту и мета-концепту. None-frequency: все векторы равноправны.

### 3. Structural Rules (`structural_rules.py`)

Бинарные правила (0/1, без частот):

- **Sentence type transitions**: 5 типов × 5 = 21 переход
- **Cross-sentence concept transitions**: 47 source → reachable concepts, 1216/2304 (52.8% density)

### 4. Gate System (`gate_logic.py`)

Многоуровневая система gates:

```
Meta (12) → Concept (48) → Word (2442) → BPE (4101) → S_type (5)
```

- 4 уровня иерархии + sentence type
- `observe(text)` → парсит текст, добавляет gates на ВСЕХ уровнях
- `valid_mask(context)` → INTERSECTION всех gates = единая бинарная маска над V=4101
- `observe_sentence(tokens, s_type)` → self-play learning
- Предвычисленные expansions: level_id → set[BPE tokens] для быстрой конвертации

**Текущие gates** (из Войны и Мира):
| Level | Gates |
|-------|-------|
| meta | 142 |
| concept | 2,230 |
| word | 192,365 |
| bpe | 208,865 |
| s_type | 21 |

### 5. VectorGenerator (`vector_space.py`)

Основной генератор:

```
generate_step(ctx, prev_token, content_token) → next_token
```

1. **Стены** (valid_mask): правила + стены
2. **Структурные scores**: INTERSECTION gates → valid set (все равны 1.0)
3. **Семантические scores**: SVD cosine similarity к последнему content word (top-5, weight 5.0)
4. **Sentence-type boost**: после диалога/вопроса boost глаголов речи
5. **Continuation**: type-2 при piw=0, type-3 при piw>=1
6. **EOS**: вероятностное завершение (1% × word_num, max 25%, min 4 слова)
7. **Anti-repetition**: текстовая (0 повторов) + концептуальная (≥2 cid → блок)
8. **Self-play**: каждое предложение → observe_sentence → gates учатся

### 6. Anti-Frequency Design

| Механизм | Что заменяет |
|----------|-------------|
| Бинарные gates | Frequency-based transition probabilities |
| Все scores равны (1.0) | Weighted/likelihood scores |
| SVD cosine similarity | Frequency-based word similarity |
| Top-5 content boost | All-token boosting |
| Probabilistic EOS (1% ramp) | Learned EOS prediction |
| Intersection of constraints | Neural network softmax |

## Данные

| Файл | Содержание |
|------|-----------|
| `real_data/full_corpus_ru.txt` | Война и Мир (10,853 предложений) |
| `real_data/v8/` | Heads ensemble (structural matrix) |
| `real_data/gates/` | Предвычисленные gates (5 JSON) |
| `real_data/structural_rules.json` | Бинарные правила переходов |
| `hierarchical_data*/` | Обработанные иерархические данные |

## Ключевые метрики

| Метрика | Значение |
|---------|----------|
| Размерность BPE | 4,101 |
| Type-2 токенов | 2,442 |
| Концептов | 48 |
| Мета-концептов | 12 |
| Предложений в корпусе | 10,853 |
| BPE gates | 208,865 |
| Concept transitions (rules) | 1,216 |
| Семантическая размерность | 32 (SVD) |
| Параметров | 0 (все правила data-driven, без обучения) |

## Запуск

```python
from eva.symbolic.bpe_tokenizer import HierarchicalVocab
from eva.symbolic.association_graph import AssociationGraph
from eva.symbolic.heads import HeadsEnsemble
from eva.symbolic.vector_space import VectorGenerator

hv = HierarchicalVocab()
heads = HeadsEnsemble('real_data/v8/heads_meta.pkl', 'real_data/v8')
ag = AssociationGraph(n_clusters=48, n_metas=12)
ag.build(heads.log_prob_csr, hv.token_type, decode_fn=hv.decode)

vg = VectorGenerator(heads_obj=heads, assoc_graph=ag, hv=hv)
vg.load_gates('real_data/gates')
result = vg.generate(max_tokens=80, seed_word='сказал')
print(result['text'])
```

## Структура проекта

```
FCF/
├── eva/symbolic/           # Core modules
│   ├── bpe_tokenizer.py    # BPE токенизатор (V=4101)
│   ├── text_hierarchy.py   # Иерархический парсер
│   ├── association_graph.py# 2-level concept clustering + SVD
│   ├── structural_rules.py # Бинарные transition rules
│   ├── gate_logic.py       # Multi-level gate system
│   ├── vector_space.py     # VectorGenerator (основной генератор)
│   ├── generation_loop.py  # Утилиты генерации
│   ├── heads.py            # Heads ensemble
│   ├── concept_transformer.py   # Concept predictor (blocked)
│   ├── linguistic_rules.py      # Морфологические правила
│   ├── potential_field.py       # Potential field gen (legacy)
│   └── ...                      # Прочие модули
├── real_data/
│   ├── full_corpus_ru.txt  # Война и Мир
│   ├── gates/              # Предвычисленные gates
│   ├── v8/                 # Heads model data
│   └── structural_rules.json
├── experiments/            # Test/build/debug/analysis scripts
│   ├── tests/
│   ├── build/
│   ├── analysis/
│   ├── train/
│   ├── eval/
│   └── legacy/
├── hierarchical_data*/     # Processed data
└── README.md
```
