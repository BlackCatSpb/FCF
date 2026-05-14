# FCF — Fractal Cognitive Fabric

**Единая Вычислительная Архитектура (ЕВА)** — самоорганизующаяся когнитивная система с 41 механизмом грамматики состояний.

---

## Философия

FCF отвергает парадигму «предобученная модель + файнтюнинг + промпт-инжиниринг». Вместо этого система:

1. **Стартует с чистого листа** — единственный слой трансформера со случайными весами (Xavier/Glorot). Никаких внешних весов.
2. **Хранит не веса, а правила композиции состояний** — знания = способность воспроизвести поведение, а не матричные значения.
3. **Растёт и сжимается по метрикам** — GrowthController + Sleep Mode = автономная архитектурная эволюция.
4. **Оценивает себя** — Semantic Relevance Gate (SRG) как внутренний критик.
5. **Корректирует ошибки** — Knowledge-Conscious Attention (KCA) с гарантированной сходимостью.
6. **Имеет неизменяемое этическое ядро** — 5 аксиом, вшитых в SRG.

---

## Архитектурная парадигма

```
Классические LLM:           FCF:

Веса = память               Веса = алфавит (атомарный базис)
Размер фиксирован           Размер адаптивен (GrowthController)
Знания в матрицах           Знания в правилах композиции
Этика = промпт              Этика = архитектура (EthicsFilter)
Обучение = градиенты        Обучение = градиенты + rule discovery
Память = контекст           Память = FAISS + HNSW + GMM
```

---

## 7 фундаментальных принципов

| # | Принцип | Реализация |
|---|---------|-----------|
| 1 | **Compute-as-Memory** | Информация = матричное состояние. Латентный код z — рецепт порождения поведения |
| 2 | **Знания не в весах** | SVD-базис: веса фиксированы как атомы, ΔW собирается из коэффициентов c_i |
| 3 | **Единство знаний и программы** | Латентный код = и знание, и инструкция. StateGrammar: 41 правило трансформации |
| 4 | **Самооценка через SRG** | Семантика + энтропия + этика → confidence. Anomaly Detection, Meta-SRG |
| 5 | **Коррекция через KCA** | Градиентный спуск по z, протокол сходимости (демпфирование, осцилляции, гейт) |
| 6 | **Рост и консолидация** | Вширь (LoRA + GMM), вглубь (рекурсия + кристаллизация), сжатие (Sleep Mode) |
| 7 | **Этическая незыблемость** | 5 аксиом в EthicsFilter, неизменяемы обучением |

---

## Вычислительное ядро

### PrimordialLayer (233M параметров)

```
PrimordialLayer
├── Embedding (50257 × 2560)       — weight-tied с LM Head
├── TransformerBlock
│   ├── CausalSelfAttention (32 головы, RoPE)
│   ├── SwiGLU FFN (d_model × 4)
│   └── RMSNorm (Pre-Norm)
├── FAISS StateStorage             — до 10K слепков (K, V, confidence)
├── SemanticRelevanceGate          — w_sim=0.4, w_ent=0.3, w_eth=0.3
├── EthicsFilter                   — 5 аксиом, 4 категории паттернов
├── MetaMemory                     — confidence_history, usage_count
├── GrowthController               — EXPAND_WIDTH / EXPAND_DEPTH / TRY_RECURSION
├── CuriosityLoop                  — авто-генерация уточняющих вопросов
└── DomainRegistry                 — реестр LoRA-адаптеров по доменам
```

---

## Трёхфазный когнитивный цикл

### Фаза 1: Восприятие (Perception)

```
Запрос → токенизация → embedding → c_query (Key-векторы)
→ HNSW Level 0: поиск домена
→ 3 сценария:
  ├── exact_match (cos > 0.95)    → Fast Path, без KCA
  ├── partial_match (0.7 ≤ cos)   → инициализация z, KCA
  └── cold_start (cos < 0.7)      → случайный z, полный KCA
→ HNSW Level 1: поиск слепка в домене
```

### Фаза 2: Порождение и коррекция (Genesis & KCA)

```
z_t → AtomicBasis.decode → ΔW → LLM forward → генерация
→ SRG оценка → если низкая:
  KCA-цикл (≤5 итераций):
    z_{t+1} = z_t - η_t·∇_z L_KCA(z_t)
    η_t = η₀·ρ^t (ρ=0.85)
    convergence: 1. damping 2. oscillation 3. gate monitor 4. hard limit
→ Adaptive KCA Scheduling: глубина по критичности запроса
```

### Фаза 3: Сохранение (Execution & Memory)

```
Финальный z_T → генерация ответа
→ Валидация: SRG ≥ 0.8 + проверка similarity > 0.95
→ Code Distillation: выразим через существующие? → ссылка
→ Code Mutation: 5% chance мутанта, лучший заменяет
→ Сохранение в HNSW + обновление GMM центроида
→ Code Provenance: цепочка происхождения
```

---

## StateGrammar — 41 механизм композиции состояний

Грамматика состояний — формальная алгебраическая система `S = (Z, ⊕, ⊗, ¬, →, ∀, K, □, M, V, ∇, Pers, Cat, G)`. Не хранит комбинации — хранит **правила** как состояния взаимодействуют.

### Базовые (1–11)

| # | Механизм | Математика | Файл |
|---|----------|-----------|------|
| 1 | **Valence** | z_effective = α·z, полярность ±, Tversky(A,B) | `state_grammar.py` |
| 2 | **TemporalChain** | P(z_t | history), forget gate, skip-connections | `state_grammar.py` |
| 3 | **NegationAlgebra** | z_not(A) ≠ -z_A, 8 скопов, excluded middle | `state_grammar.py` |
| 4 | **SuperpositionCollapse** | Measure(z_superposed, context), декогеренция | `state_grammar.py` |
| 5 | **CompositionalValidator** | valid(A⊕B|C), explainability, контрпримеры | `state_grammar.py` |
| 6 | **StateInheritance** | is_a DAG, C3-линеаризация, прототипы | `state_grammar.py` |
| 7 | **EmergentGenesis** | count(A⊕B) ≥ threshold → новый концепт | `state_grammar.py` |
| 8 | **TransformDistance** | d(A,B) = min|path|, Дейкстра, взвешенные рёбра | `state_grammar.py` |
| 9 | **ConservationLaws** | инвариантные измерения, фазовые переходы | `state_grammar.py` |
| 10 | **SelfReference** | z = f(z), Рассел, well-foundedness, стратификация | `state_grammar.py` |
| 11 | **InformationEntropy** | ΔI = H(A)+H(B)-H(A⊕B), KL, mutual info, entropy rate | `state_grammar.py` |

### Расширенные (12–21)

| # | Механизм | Файл |
|---|----------|------|
| 12 | **CausalReasoning** — do(A), контрфактуалы, necessary_cause_score | `state_grammar_ext.py` |
| 13 | **TemporalModality** — past/present/future/hypothetical | `state_grammar_ext.py` |
| 14 | **EpistemicStates** — know/believe/uncertain, socratic_question | `state_grammar_ext.py` |
| 15 | **Quantification** — ∀, ∃, ∄, quantifier_scope | `state_grammar_ext.py` |
| 16 | **StateResonance** — гармонический резонанс, resonant clusters | `state_grammar_ext.py` |
| 17 | **FrontierStates** — граничные состояния, creative detection | `state_grammar_ext.py` |
| 18 | **GradientFlow** — ∇V(z), седловые точки, potential landscape | `state_grammar_ext.py` |
| 19 | **TopologicalPersistence** — persistence diagram, bottleneck distance | `state_grammar_ext.py` |
| 20 | **CategoryTheory** — функторы, natural transformations, adjunction | `state_grammar_ext.py` |
| 21 | **InformationGeometry** — метрика Фишера, геодезические, кривизна | `state_grammar_ext.py` |

### Финальные (22–31)

| # | Механизм | Файл |
|---|----------|------|
| 22 | **RecursiveSelfModification** — meta-learning правил | `state_grammar_final.py` |
| 23 | **DialecticalSynthesis** — thesis⊕antithesis→synthesis (Hegel) | `state_grammar_final.py` |
| 24 | **Abduction** — effect+rules→cause (Peirce) | `state_grammar_final.py` |
| 25 | **AnalogicalMapping** — A:B::C:D, structure-mapping theory | `state_grammar_final.py` |
| 26 | **ZeroShotComposition** — невиданные комбинации из правил | `state_grammar_final.py` |
| 27 | **FractalSelfConsistency** — масштабная инвариантность правил | `state_grammar_final.py` |
| 28 | **TeleologicalReasoning** — purpose-driven transformation | `state_grammar_final.py` |
| 29 | **NarrativeCoherence** — tension→climax→resolution | `state_grammar_final.py` |
| 30 | **EmotionalValence** — 8 эмоций, valence-arousal | `state_grammar_final.py` |
| 31 | **CounterfactualImagination** — рекурсивные альтернативные миры | `state_grammar_final.py` |

### Глубинные (32–41)

| # | Механизм | Файл |
|---|----------|------|
| 32 | **CulturalRelativity** — Sapir-Whorf, untranslatability | `state_grammar_deep.py` |
| 33 | **DreamRecombination** — сюрреалистическая рекомбинация | `state_grammar_deep.py` |
| 34 | **EthicalCalculus** — утилитаризм vs деонтология, Pareto frontier | `state_grammar_deep.py` |
| 35 | **StateEconomy** — cost, value, amortization, ROI, garbage collect | `state_grammar_deep.py` |
| 36 | **EvolutionaryPressure** — генетический алгоритм, tournament select | `state_grammar_deep.py` |
| 37 | **GameTheoretic** — Nash equilibrium, доминирование, коалиции | `state_grammar_deep.py` |
| 38 | **AttentionEconomy** — бюджет внимания, salience, allocation | `state_grammar_deep.py` |
| 39 | **MetaphorGeneration** — Lakoff концептуальная метафора | `state_grammar_deep.py` |
| 40 | **RecursiveIntrospection** — K^n(z), глубина самосознания | `state_grammar_deep.py` |
| 41 | **EntropySeeking** — curiosity-driven exploration | `state_grammar_deep.py` |

---

## FCFSystem — единый runtime

```
FCFSystem
├── bootstrap()          — 9 шагов инициализации
│   ├── PrimordialLayer   — загрузка/создание
│   ├── Tokenizer         — 50K слов (Wikipedia BPE)
│   ├── AtomicBasis       — SVD-разложение
│   ├── FractalHierarchy  — 4 уровня (sym→word→sent→text)
│   ├── MultiLevelGMM     — динамические домены
│   ├── HNSWIndex+PQ      — 3 уровня + сжатие 4x
│   ├── KCAEngine         — итеративная коррекция
│   ├── SRGPlus           — аномалии, тренд, uncertainty
│   └── StateGrammar      — 41 механизм композиции
├── query()              — когнитивный цикл
│   ├── Perception        — HNSW поиск, 3 сценария
│   ├── Genesis           — генерация + KCA
│   ├── Validation        — 3 критерия, distillation, mutation
│   └── Grammar           — composition_validity, delta_I, creativity
├── start_background()    — Sleep Mode + Grammar discovery
└── stats()              — метрики системы
```

---

## Структура проекта (50 модулей)

```
FCF/
├── fcf/                        # Исходный код
│   ├── config.py               # FCFConfig (dataclass + JSON)
│   ├── transformer.py          # CausalSelfAttention, SwiGLUFFN, RMSNorm, RoPE
│   ├── state_storage.py         # FAISS IndexFlatIP хранилище
│   ├── srg.py + srg_plus.py    # SRG + Meta-SRG + Anomaly Detection
│   ├── ethics_filter.py        # 5 аксиом, 4 категории паттернов
│   ├── meta_memory.py          # confidence_history, usage_count
│   ├── growth_controller.py    # EXPAND_WIDTH/DEPTH/TRY_RECURSION
│   ├── curiosity_loop.py       # Генерация уточняющих вопросов
│   ├── primordial_layer.py     # PrimordialLayer (сборка всех компонентов)
│   ├── tokenizer_utils.py      # BPE-токенизатор (HuggingFace tokenizers)
│   ├── data_manager.py         # Wikipedia, Saiga, ConceptNet, RuBQ
│   ├── language_trainer.py     # Causal LM + grammar discovery + auto-benchmark
│   ├── instruction_trainer.py  # Инструктивное дообучение
│   ├── lora_adapter.py         # LoRA (rank×in, out×rank)
│   ├── domain_registry.py      # Реестр доменов + поиск по центроидам
│   ├── domain_trainer.py       # Обучение LoRA-адаптеров
│   ├── atomic_basis.py         # SVD-разложение, encode/decode ΔW
│   ├── fractal_hierarchy.py    # 4 уровня: sym→word→sent→text
│   ├── streaming_gmm.py        # Динамические домены (GMM)
│   ├── hnsw_index.py           # HNSW + Product Quantization + Temporal Decay
│   ├── kca_engine.py           # KCA + ConvergenceController
│   ├── state_algebra.py        # Операторы + CrossAttendBlock + Translator
│   ├── recursive_processor.py  # Рекурсивная обработка
│   ├── layer_crystallizer.py   # Кристаллизация новых слоёв
│   ├── sleep_mode.py           # Sleep Mode v1
│   ├── sleep_mode_v2.py        # Sleep Mode v2 (Dream, Forgetfulness, Adversarial)
│   ├── auto_trainer.py         # Фоновое автодообучение
│   ├── environment_tuner.py    # Автонастройка CPU/GPU/потоков
│   ├── code_provenance.py      # Отслеживание происхождения кодов
│   ├── code_mutation.py        # Мутация + дистилляция кодов
│   ├── self_descriptive.py     # Авто-описание латентных кодов
│   ├── intrinsic_curiosity.py  # Проверка забытых доменов
│   ├── multi_pass.py           # Multi-Pass Generation + Code Ensemble
│   ├── temporal_context.py     # Сжатие диалога в латентный код
│   ├── cross_domain.py         # Cross-Domain Translation + Attention
│   ├── cross_modal.py          # Image/Audio/Bridge энкодеры
│   ├── federated.py            # Federated Fabric + Collaborative SRG
│   ├── extensions.py           # 5 механизмов (MinimalCode, OpTrainer, ReSelf...)
│   ├── state_grammar.py        # StateGrammar: механизмы 1-11
│   ├── state_grammar_ext.py    # StateGrammar: механизмы 12-21
│   ├── state_grammar_final.py  # StateGrammar: механизмы 22-31
│   ├── state_grammar_deep.py   # StateGrammar: механизмы 32-41
│   ├── unified_grammar.py      # UnifiedStateGrammar: все 41 в одном API
│   ├── fcf_system.py           # FCFSystem: единый runtime
│   ├── benchmark.py            # End-to-end бенчмарк
│   ├── end_to_end_test.py      # Интеграционный тест
│   ├── checkpoint_comparator.py # Сравнение качества по чекпоинтам
│   ├── utils.py                # save/load с FAISS, весами, метаданными
│   └── __init__.py
├── docs/                       # Документация
├── notebooks/                  # Colab/Kaggle ноутбуки
├── real_data/                  # Загруженные датасеты
├── config.json                 # Конфигурация
├── requirements.txt            # Зависимости
├── tokenizer.json              # BPE-токенизатор (50K слов)
├── fcf.bat                     # Запуск (двойной клик)
└── run.py                      # CLI точка входа
```

---

## Запуск

```bash
# Установка
pip install -r requirements.txt

# Ленивое обучение (Wikipedia streaming + Grammar discovery)
python run.py --lazy-learn --checkpoint checkpoints\language\step_023000

# Полный FCFSystem
python run.py --fcf --checkpoint checkpoints\language\step_023000

# Интерактивный режим
python run.py --interactive

# End-to-end тест
python -m fcf.end_to_end_test

# Сравнение чекпоинтов
python -m fcf.checkpoint_comparator
```

**Команды в lazy-learn консоли:**
- `grammar` — визуализация 41 механизма
- `discover` — запуск rule discovery из слепков
- `bench` — история авто-бенчмарков
- `stats` — статистика слоя
- `train` — Wikipedia 5000 шагов
- `save` — сохранить чекпоинт
- `exit` — выход

---

## Технологический стек

| Категория | Технология |
|-----------|-----------|
| ML-фреймворк | PyTorch 2.5+ |
| Векторная БД | FAISS + HNSW + Product Quantization |
| Токенизатор | HuggingFace `tokenizers` (BPE, 50K слов) |
| Оптимизатор | AdamW |
| Кластеризация | HDBSCAN / KMeans / Streaming GMM |
| Логирование | Loguru |
| Данные | Wikipedia API, HuggingFace datasets, OpenSubtitles |

---

## Состояние разработки

| Фаза | Компонент | Статус |
|------|-----------|--------|
| Ядро | PrimordialLayer, SRG, Ethics, FAISS, Curiosity, Growth | ✅ |
| Обучение | LanguageTrainer, InstructionTrainer, DomainTrainer | ✅ |
| Рост | LoRA, DomainRegistry, RecursiveProcessor, LayerCrystallizer | ✅ |
| Консолидация | SleepMode v1 + v2 (Dream, Forgetfulness, Adversarial) | ✅ |
| KCA | KCAEngine + ConvergenceController + refine_through_llm | ✅ |
| SVD | AtomicBasis (encode/decode ΔW через c_i) | ✅ |
| Иерархия | FractalHierarchy (4 уровня, self-referencing attention) | ✅ |
| Домены | StreamingGMM (рождение/слияние/удаление) | ✅ |
| Хранилище | HNSWIndex + PQ (3 уровня, сжатие 4x, Temporal Decay) | ✅ |
| SRG+ | Anomaly Detection, Meta-SRG, Code Uncertainty | ✅ |
| StateGrammar | 41 механизм композиции состояний | ✅ |
| FCFSystem | Единый runtime, 3-фазный когнитивный цикл | ✅ |
| Интеграция | Grammar-guided training, auto-benchmark, e2e test | ✅ |

---

## Числа проекта

```
50 модулей Python    |  ~15 000 строк кода
41 механизм StateGrammar |  4 файла грамматики
233M параметров слоя     |  1 слой (растёт до N)
86 механизмов спецификации |  83% полное покрытие
```

---

*FCF — Fractal Cognitive Fabric. Единая Вычислительная Архитектура. v2.0*
