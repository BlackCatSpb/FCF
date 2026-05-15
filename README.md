# EVA — Единая Вычислительная Архитектура

Самоорганизующаяся когнитивная система. Не предобученная модель — **зерно**, из которого вырастает интеллект.

---

## Состояние

```
✅ Production-ready
✅ 9 прогонов агентов-архитекторов
✅ ~95 багов найдено и исправлено
✅ Полный когнитивный цикл (Perception → Generation → Save)
✅ 41 механизм StateGrammar
✅ Обучение на русской литературе до 1917 г. (Толстой, Достоевский, Пушкин, Чехов...)
```

---

## Архитектура

### Принципы

1. **Tabula Rasa** — старт со случайных весов (Xavier/Glorot), без предобученных моделей
2. **Compute-as-Memory** — информация = матричное состояние, нет разделения памяти и процессора
3. **Этическая незыблемость** — 5 аксиом, вшитых в SRG, не могут быть изменены обучением
4. **Органический рост** — расширение архитектуры только по объективной необходимости (GrowthController)
5. **Самооценка** — SRG (Semantic Relevance Gate) как внутренний критик
6. **Коррекция** — KCA (Knowledge-Conscious Attention) с гарантированной сходимостью ≤5 итераций
7. **Консолидация** — Sleep Mode: кластеризация, очистка, дистилляция, Dream Mode

### Вычислительное ядро

```
PrimordialLayer (233M параметров, растёт до N слоёв)
├── Embedding (50257 × 2560) — weight-tied с LM Head
├── TransformerBlock
│   ├── CausalSelfAttention (32 головы, RoPE, KV-cache)
│   ├── SwiGLU FFN (d_model × 4)
│   └── RMSNorm (Pre-Norm)
├── FAISS StateStorage — до 10K слепков (реальные K,V тензоры)
├── SemanticRelevanceGate — w_sim=0.4, w_ent=0.3, w_eth=0.3
├── EthicsFilter — 5 аксиом, 4 категории паттернов
├── MetaMemory — confidence_history, usage_count
├── GrowthController — EXPAND_WIDTH / EXPAND_DEPTH / TRY_RECURSION
├── CuriosityLoop — авто-генерация уточняющих вопросов
└── DomainRegistry — реестр LoRA-адаптеров по доменам
```

### Трёхфазный когнитивный цикл

```
Фаза 1: Восприятие
  Токенизация → c_query (Key-векторы) → HNSW Level 0 (домен) → 3 сценария
  (exact_match / partial_match / cold_start) → HNSW Level 1 (слепок)

Фаза 2: Генерация и коррекция (KCA)
  AtomicBasis.decode → ΔW → LLM forward → генерация → SRG оценка
  → если confidence < 0.5: KCA (≤5 итераций, демпфирование, осцилляции, гейт)
  → если улучшило → перегенерация ответа

Фаза 3: Сохранение
  Валидация (confidence ≥ 0.8) → Code Distillation → Code Mutation (5%)
  → HNSW + FAISS → GMM update → Code Provenance → StateGrammar composition
```

### StateGrammar — 41 механизм композиции

Грамматика состояний: не хранит комбинации — хранит **правила** как состояния взаимодействуют.

| Блок | Механизмы | Ключевые |
|------|-----------|----------|
| Измерение | 1-11 | Valence, Temporal, Negation, Superposition, Validator, Inheritance, Emergence, Distance, Conservation, SelfRef, Entropy |
| Взаимодействие | 12-21 | Causal, TemporalModality, Epistemic, Quantification, Resonance, Frontier, GradientFlow, Persistence, Category, InfoGeometry |
| Самоулучшение | 22-31 | Meta-Modification, Dialectic, Abduction, Analogy, ZeroShot, Fractal, Teleology, Narrative, Emotion, Counterfactual |
| Глубинная семантика | 32-41 | Culture, Dream, Ethics, Economy, Evolution, GameTheory, Attention, Metaphor, Introspection, Curiosity |

### Хранилище

```
3-уровневый HNSW + Product Quantization (сжатие 4x)
  Level 0: центроиды доменов → маршрутизация
  Level 1: слепки внутри домена → поиск
  Level 2: слепки по диапазонам слоёв → уточнение

Temporal Decay: similarity *= exp(-λ·age/86400)
Streaming GMM: динамические домены (рождение/слияние/удаление)
FAISS: реальные K,V тензоры, Welford running stats, Cholesky KL
```

---

## Запуск

```bash
pip install -r requirements.txt
python run.py --lazy-learn    # или двойной клик EVA.lnk
```

**Команды в консоли:** `grammar`, `discover`, `bench`, `stats`, `train`, `save`, `exit`

**Данные для обучения:** русская литература до 1917 г. (Толстой, Достоевский, Пушкин, Чехов, Гоголь, Тургенев, Лермонтов, Гончаров, Горький, Бунин, Куприн, Короленко) — 21 книга, 3.9 млн слов, 22.7 MB

---

## Структура проекта

```
EVA/
├── eva/                        # 49 модулей Python
│   ├── primordial_layer.py     # Ядро: слой + SRG + Ethics + FAISS
│   ├── transformer.py          # Attention, SwiGLU, RMSNorm, RoPE, KV-cache
│   ├── fcf_system.py           # Единый runtime (bootstrap + query + sleep)
│   ├── language_trainer.py     # Causal LM + grammar discovery + auto-benchmark
│   ├── kca_engine.py           # KCA + ConvergenceController
│   ├── state_grammar.py        # Механизмы 1-11
│   ├── state_grammar_ext.py    # Механизмы 12-21
│   ├── state_grammar_final.py  # Механизмы 22-31
│   ├── state_grammar_deep.py   # Механизмы 32-41
│   ├── unified_grammar.py      # UnifiedStateGrammar (все 41)
│   ├── hnsw_index.py           # HNSW + PQ + TemporalDecay + FractalLinks
│   ├── streaming_gmm.py        # Динамические домены (GMM)
│   ├── atomic_basis.py         # SVD-разложение весов
│   ├── sleep_mode.py           # Sleep Mode (Dream, Forgetfulness, Adversarial)
│   ├── srg.py + srg_plus.py    # SRG + Anomaly Detection + Meta-SRG
│   └── ...
├── docs/                       # Документация + контекст агентов
├── real_data/                  # Обучающие данные (литература)
├── config.json                 # Конфигурация
├── run.py                      # CLI (12 команд)
└── eva.bat                     # Запуск (двойной клик)
```

---

## Числа

```
49 модулей Python     |  ~15 000 строк кода
41 механизм StateGrammar |  4 файла грамматики
233M параметров слоя  |  1 слой (растёт до N)
95 багов исправлено   |  9 прогонов агентов
50 257 токенов        |  BPE-словарь (Wikipedia)
```

---

*EVA — Единая Вычислительная Архитектура*
