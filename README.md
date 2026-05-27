# EVA — Топологический ИИ нового поколения

> **Не LLM. Не статистический попугай.**  
> Символ ≡ координата в ℝ¹²⁸. Текст ≡ траектория. Трансформер ≡ навигатор.  
> Знания — в геометрии пространства, не в весах.  
> **1.14M параметров. 160 токенов. Полный цикл encode→decode.**

---

## Идея за 30 секунд

Обычный ИИ: «привет» → токен #4521 → вектор → угадать следующий токен.

**EVA**: «привет» → 6 точек в ℝ¹²⁸ → траектория → найти путь → исполнить инструкцию.

Модель не «предсказывает следующее слово». Она **навигирует в координатном пространстве**, где каждый символ — точка, каждое слово — путь, а знание — топология всех возможных путей.

---

## Архитектура

```
Текст → Boundary Tokens → HybridFractalBlock ×6 → Координаты → Декодер → Текст
              │                    │
         <W></W><S></S>      Conv2D + Attention
                                 │
                          StaticTopology [160,160,3]
                                 │
                          TrajectoryStore (Fast Path)
```

| Компонент | Назначение | Параметры |
|-----------|-----------|-----------|
| **CharacterVocab** | 160 токенов + boundary-разметка | 0 |
| **CoordinateEmbedding** | Символ → ℝ¹²⁸ (MDS из affinity) | 0 |
| **RoPE** | Rotary Position Embedding | 0 |
| **FractalConv2D** | Многомерная фрактальная свёртка (L×Dim) | В составе блока |
| **AdaptiveAttention** | 32 головы, динамические уровни 1-8 | В составе блока |
| **HybridFractalBlock ×6** | Conv2D + Attention + Gate Merge | ~190K/блок |
| **StaticTopologyLayer** | Матрица [160,160,3]: affinity, barrier, forbidden | ~500 |
| **SwiGLU FFN** | Gated Feed-Forward | В составе блока |
| **RMSNorm** | Pre-norm нормализация | ~256 |
| **WordWeightEncoder** | Обратное внимание — вес слов | ~50K |
| **CoordinateDecoder** | ℝ¹²⁸ → 160 классов | ~10K |
| **TrajectoryStore** | Иерархическая память (4 уровня) | ∞ |

---

## Что работает

| # | Возможность | Результат |
|---|-----------|-----------|
| 1 | Воспроизведение всех символов | 100% |
| 2 | Affinity matrix из текста | 160×160 |
| 3 | MDS → ℝ¹²⁸ координаты | eff_dim >80% |
| 4 | Word/Sentence autoencoding | 100% |
| 5 | STDP пластика (PotentialDynamics) | σ ×16 |
| 6 | V(z) скалярный потенциал | real=-1, random=+1 |
| 7 | K-means концепты | 8 лингвистических групп |
| 8 | ContradictionFilter | 5 типов запретов |
| 9 | GradientFlow + седловые точки | Барьеры 0-1.6 |
| 10 | DialecticalSynthesis | 15/18 валидных синтезов |
| 11 | TopologicalPersistence | >82% устойчивость |
| 12 | FractalSelfConsistency | Power-law decay |
| 13 | ConceptNet enrichment | 597K русских слов |
| 14 | Trajectory Genetics | Мутация, кроссовер, селекция |
| 15 | Полный encode→decode цикл | 100% round-trip |
| 16 | Causal генерация с boundary-tokens | Реальные персонажи Толстого |
| 17 | **HybridFractalBlock** | Сходимость ×20 быстрее |
| 18 | **StaticTopologyLayer** | Fast Path кэш траекторий |
| 19 | Иерархический TrajectoryStore | 4 уровня метаданных |
| 20 | Autonomous Consolidation | 2690+ траекторий |

---

## Ключевые отличия от LLM

| | LLM (GPT, LLaMA) | EVA |
|---|---|---|
| **Принцип** | Статистическое угадывание | Навигация в ℝ¹²⁸ |
| **Символ** | Индекс в таблице | Координата (MDS из языка) |
| **Параметры** | 1B–1.7T | **1 144 990** |
| **Знания** | В миллиардах весов | В топологии (отдельно) |
| **Память** | Внутри модели | TrajectoryStore (внешняя) |
| **Внимание** | Multi-Head Self-Attention | **FractalConv2D + AdaptiveAttention** |
| **Генерация** | softmax(logits) | **Исполнение координатных инструкций** |
| **Обучение** | Статичное, $100M+ | **STDP + LTP/LTD + диалектика** |
| **Интерпретация** | Black box | **Траектория в ℝ¹²⁸ — полный trace** |
| **Масштабирование** | Больше параметров | Больше траекторий (модель не растёт) |
| **2+2=4** | Статистически | **Единственный путь в ℝ¹²⁸** |

---

## Научные инновации

### Символ как координата
Позиция символа в ℝ¹²⁸ **вычисляется** из статистики языка через MDS, а не задаётся случайно. Символы, часто встречающиеся рядом, получают близкие координаты. «п» и «р» рядом. «ъ» и «ь» далеко.

### Иерархические метаданные
Четыре уровня вложенности: символ → слово → связь → предложение. Каждый уровень — своё подпространство в ℝ¹²⁸. FractalConv2D видит все уровни одновременно.

### Статическая топология
Матрица [160, 160, 3] предвычисляет affinity, potential barrier и forbidden для каждой пары символов. Используется как read-only bias в attention + Fast Path кэш для мгновенной генерации.

### Живая пластика
Affinity матрица эволюционирует через STDP (Spike-Timing-Dependent Plasticity): часто используемые связи усиливаются, неиспользуемые ослабляются. LTP/LTD + гомеостаз. Модель улучшается от использования.

### Фрактальная размерность
Conv2D + Attention с dilation 1,2,4,8 по обеим осям (L и Dim). Один блок видит и соседние символы, и целые предложения, и меж-subspace корреляции.

### Диалектический синтез
Новые концепты через гегелевскую диалектику: тезис ⊕ антитезис → синтез. Седловая точка между концептами → градиентный спуск → новая котловина.

---

## Быстрый старт

```bash
git clone https://github.com/BlackCatSpb/FCF.git
cd FCF
pip install torch numpy scikit-learn loguru psutil

# Иерархическое обучение на Войне и Мире
python train_warpeace.py

# Консолидация знаний
python train_consolidate.py
```

---

## Структура проекта

```
EVA/
├── train_warpeace.py         ← Иерархическое обучение
├── train_consolidate.py      ← Автономная консолидация
├── train_unified.py          ← Train+Think цикл
├── train_genetics.py         ← Эволюция траекторий
├── train_yandex.py           ← V100-оптимизированное обучение
│
├── eva/symbolic/             ← Символьное ядро (27 модулей)
│   ├── unified_transformer.py  ← Координатный трансформер
│   ├── fractal_conv.py         ← HybridFractalBlock
│   ├── adaptive_fractal.py     ← AdaptiveAttention
│   ├── static_topology.py      ← Статическая топология
│   ├── potential_field.py      ← Affinity matrix
│   ├── potential_function.py   ← V(z) потенциал
│   ├── topological_field.py    ← MDS проекция
│   ├── trajectory_store.py     ← Иерархическая память
│   ├── subspace_coords.py      ← MultiSubspace + WordWeight
│   ├── contradiction_filter.py ← Иммунная система
│   ├── gradient_flow.py        ← GFRE
│   ├── self_reflection.py      ← Анализ траекторий
│   └── ...
│
├── docs/
│   ├── ARCHITECTURE.md       ← Полное описание архитектуры
│   ├── PARAMS_CALC.md        ← Математический расчёт
│   ├── LLM_IMPROVEMENTS.md   ← Заимствования из LLM
│   └── CONVOLUTION_DESIGN.md ← Свёрточная оптимизация
│
└── checkpoints/symbolic/     ← Чекпоинты и TrajectoryStore
```

---

*«Модель не имитирует язык. Она строит карту символьного пространства и учится по ней перемещаться. Знания — не в весах, а в геометрии пространства.»*
