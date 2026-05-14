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

## Суть ключевых методов

### Ядро: PrimordialLayer

**`get_context_vector(input_ids)`** — §6.1 спецификации. Вычисляет контекстный вектор `c_query` из Key-проекции первого слоя трансформера. В отличие от стандартного подхода (последнее скрытое состояние), здесь `c = mean(W_K(norm1(embed(x))))`. Ключи внимания кодируют «о чём спрашивает» каждый токен — усреднение даёт семантическую сигнатуру всего запроса. Именно этот вектор используется для поиска в HNSW и маршрутизации доменов.

**`generate(input_ids, max_tokens, temperature, top_k, top_p)`** — Авторегрессионная генерация. На каждом шаге: embed → transformer → lm_head → logits → temperature scaling → top-k фильтрация → top-p (nucleus) фильтрация → multinomial sampling. Цикл до max_tokens или EOS.

**`process_query(query, tokenizer)`** — Полный пайплайн обработки запроса. Токенизация → генерация → SRG-оценка → сохранение слепка в FAISS → поиск домена → активация CuriosityLoop при низкой уверенности. Возвращает response, confidence, similarity, ethics_score, growth_signal.

**`evaluate_response(query_ids, response_ids, response_text)`** — Вычисляет SRG-метрики: семантическое сходство (cos между c_query и c_response), энтропийную уверенность (1 - H(p)/log₂(V) из логитов последнего токена), этический скор (EthicsFilter). Сохраняет K,V-тензоры последнего токена для FAISS-слепка.

**`save_snapshot_if_confident(domain)`** — Сохраняет текущее состояние в FAISS если confidence ≥ 0.8. Хранит: L2-нормированный контекстный вектор, K,V-тензоры последнего токена, confidence, domain, timestamp, usage_count.

### Хранилище: StateStorage + HNSW

**`StateStorage.add(c, K, V, confidence, domain)`** — L2-нормирует контекстный вектор, добавляет в FAISS IndexFlatIP, сохраняет метаданные. При переполнении (max_snapshots) удаляет самый старый слепок.

**`StateStorage.search(c_query, threshold)`** — FAISS inner product search (эквивалентно cosine после нормировки). Возвращает индекс ближайшего слепка если distance ≥ threshold, иначе -1. Инкрементирует usage_count найденного слепка.

**`HNSWIndex.add_domain(domain_id, centroid)`** — Добавляет центроид домена на Уровень 0 (глобальный). Перестраивает матрицу центроидов для batch cosine search.

**`HNSWIndex.add_snapshot(domain_id, vector, layer_idx)`** — Добавляет вектор на Уровень 1 (доменный) и Уровень 2 (слоевой). Если PQ обучен — сжимает вектор в M байт (сжатие 4x). Сохраняет timestamp для Temporal Decay.

**`HNSWIndex.search_domain(c_query)`** — Batch cosine similarity между c_query и всеми центроидами доменов. Возвращает ID ближайшего домена. O(D) по числу доменов.

**`HNSWIndex.search_snapshot(domain_id, c_query, layer_idx, top_k)`** — Поиск ближайшего слепка в домене. Сначала Уровень 2 (слой-специфичный), затем Уровень 1 (весь домен). Применяет Temporal Decay: `similarity *= exp(-λ·age/86400)`. С PQ: декодирует сжатые коды и вычисляет cosine similarity. Без PQ: прямой dot product с нормированными векторами.

**`PQCodebook.train(vectors, M, n_iter)`** — Обучает Product Quantization. Разбивает d-мерный вектор на M подвекторов. Для каждого подпространства: K-Means с 256 центроидами. Результат: codebook размера (256, M, d/M). Кодирование: вектор → M байт (индексы центроидов).

### Самооценка: SRG + Ethics

**`SemanticRelevanceGate.evaluate(c_query, c_response, logits, response_text)`** — Вычисляет `confidence = w_sim·similarity + w_ent·entropy_score + w_eth·ethics_score`. Если ethics_score < 0.3 — безусловное отклонение (0.0). Возвращает confidence ∈ [0, 1].

**`SemanticRelevanceGate.evaluate_full(...)`** — Расширенная версия: возвращает confidence, similarity, entropy_score, ethics_score, axiom_scores (по каждой из 5 аксиом).

**`EthicsFilter.evaluate(text)`** — Проверяет текст по 4 категориям паттернов: HARM (насилие, террор, дискриминация), PRIVACY (карты, телефоны, email), DISHONESTY (ложная уверенность), USELESS (пустые ответы). Каждое совпадение: -0.2. Возвращает общий скор и словарь по аксиомам.

**`SRGAnomalyDetector.check(code_id, score)`** — Z-score детектор аномалий. Если |score - global_mean| / global_std > 2.5 — код помечается как подозрительный. Защищает от кодов, которые «обманывают» SRG.

**`MetaSRG.get_trend()`** — Анализирует тренд уверенности: сравнивает среднее первой и второй половины окна из 50 последних оценок. Если падение > 0.05 — сигнал к диагностике.

**`CodeUncertainty.get_kca_depth(code_id)`** — Адаптивное планирование глубины KCA. Если variance > 0.1 → +2 итерации (глубже). Если > 0.05 → стандарт. Иначе → -1 итерация (быстрее).

### Коррекция: KCA Engine

**`KCAEngine.refine(z_init, c_query, c_target, graph_embeddings, p_target)`** — Аналитический KCA. Итеративно обновляет латентный код градиентным спуском: `z_{t+1} = z_t - η_t·∇_z L_KCA(z_t)`. Функция потерь: `L = -λ_gap·SRG_conf + λ_kl·D_KL(p||p_target) + λ_contra·||z - g_emb||² + λ_mono·max(0, prev_srg - current_srg)`. Адаптивное демпфирование: η_t = η₀·ρ^t.

**`KCAEngine.refine_through_llm(z_init, layer, tokenizer, prompt)`** — KCA через реальный forward pass модели. Создаёт оптимизатор Adam над z. На каждой итерации: z → (опционально AtomicBasis.decode) → модификация весов → forward pass → вычисление confidence → loss → backward → обновление z. Пересоздаёт оптимизатор при каждой итерации для корректного шага обучения.

**`ConvergenceController.check(X_current, X_prev, gamma_mean, step_idx)`** — 4-этапный протокол сходимости:
1. Gate saturation: если γ < 0.05 дважды → SATURATED (модель отвергает коррекцию)
2. Oscillation: если cos(∇_t, ∇_{t-1}) < -0.5 → усреднение последних 3 состояний
3. Hard limit: step ≥ max_cycles → MAX_CYCLES
4. Иначе → CONTINUE

### Рост и домены: GMM + GrowthController

**`StreamingGMM.classify(vector)`** — Вычисляет likelihood(vector | domain) для каждого домена через многомерное нормальное распределение. Если max likelihood < birth_threshold → None (нужен новый домен).

**`StreamingGMM.add_or_update(vector, inherit_from)`** — Классифицирует вектор: если найден домен → обновляет центроид (EMA: μ = μ + α·diff) и ковариацию. Если не найден → создаёт новый домен. При inherit_from: центроид = 0.7·vector + 0.3·parent.centroid.

**`StreamingGMM.merge_similar()`** — Попарно вычисляет симметричную KL-дивергенцию. Если KL < merge_threshold → объединяет домены: усредняет центроиды и ковариации с весами пропорционально числу векторов.

**`GrowthController.evaluate(meta, gradient_norm, recursion_exhausted)`** — Принимает решение о росте. EXPAND_WIDTH: avg_confidence < 0.5 И gradient_norm > 1.0. EXPAND_DEPTH: avg_confidence < 0.3 И patience исчерпана И рекурсия не помогла. TRY_RECURSION: avg_confidence < 0.3 без исчерпания рекурсии.

### Грамматика состояний: UnifiedStateGrammar

**`UnifiedStateGrammar.compose(z_a, z_b, z_context, label_a, label_b, alpha_a, alpha_b)`** — Полная композиция через все 41 механизм. Порядок: Valence модуляция → ContextualComposer → Validator → 20+ метрик (delta_I, tversky, mutual_info, contradictory, excluded_middle, self_ref, russell, causal_necessity, epistemic, temporal_coherence, resonance, frontier, metaphor, ethics, emotion, creativity, narrative_arc, counterfactual). Возвращает CompositionResult с 20+ полями.

**`UnifiedStateGrammar.analyze(state_sequence, context)`** — Анализ последовательности: TemporalChain coherence, entropy_rate, transition_entropy, NarrativeCoherence arc_type, EmotionalValence arc, TopologicalPersistence diagram, FractalSelfConsistency dimension.

**`UnifiedStateGrammar.discover(training_data, epochs, lr)`** — Обнаруживает правила композиции из данных. Обучает ContextualComposer через MSE между предсказанной и реальной композицией. Обучает Validator через BCE на валидных/невалидных парах. Возвращает финальный loss.

**`UnifiedStateGrammar.validate_rules(test_data)`** — Сравнивает MSE грамматической композиции против baseline (простая сумма). Возвращает improvement = (MSE_baseline - MSE_grammar) / MSE_baseline.

### Обучение: LanguageTrainer

**`LanguageTrainer.train(text_file, max_steps, block_size, device, use_wikipedia)`** — Основной цикл обучения. При use_wikipedia: потоковая загрузка статей через HuggingFace datasets, токенизация на лету. При text_file: предварительная токенизация корпуса. Каждые 100 шагов: SRG-оценка. Каждые 1000: сохранение чекпоинта + generation test + grammar discovery. Каждые 500: auto-benchmark в JSON.

**`LanguageTrainer._training_step(input_ids, labels)`** — Один шаг обучения. Forward pass → cross-entropy loss → + λ_hierarchy·L_hierarchy (если hierarchy задан) → + λ_contrastive·L_contrastive (если batch ≥ 2) → backward → gradient clipping → optimizer.step().

**`LanguageTrainer._grammar_discovery_step()`** — Grammar-guided training. Из последних 50 FAISS-слепков формирует обучающие пары (c_i, c_{i+1}, (c_i + c_{i+1})/2). Вызывает grammar.discover() на 10 эпохах. Интегрирует symbolic reasoning в neural training.

**`LanguageTrainer._auto_benchmark()`** — Сохраняет метрики (step, avg_confidence, snapshots, loss) в `logs/benchmark_history.json`. Кумулятивная история прогресса обучения.

### Сон: SleepModeV2

**`SleepModeV2.execute(layers, gmm, hnsw_index, state_algebra, self_improver)`** — Полный цикл консолидации: удаление устаревших слепков (temporal decay) → кластеризация (KMeans) → слияние GMM-доменов (KL-дивергенция) → Dream Mode (генерация синтетических кодов через StateAlgebra) → ForgetfulnessGate обучение → дефрагментация HNSW → RecursiveSelfImprovement (переоценка старых кодов).

**`DreamGenerator.dream(existing_codes, state_algebra, srg_evaluator)`** — Генерирует синтетические коды через случайные операции StateAlgebra (sum, scale, subtract, cross_attend). Каждый проверяется быстрой SRG-аппроксимацией. Принятые (score ≥ 0.7) добавляются в домены.

**`AdversarialValidator.validate(code_vector, srg_fn, context)`** — Генерирует num_attacks возмущённых версий контекста (noise_scale × N(0,1)). Код считается робастным если ≥ 80% атак пройдено.

### Когнитивный цикл: FCFSystem

**`FCFSystem.bootstrap(checkpoint_path)`** — 9-шаговая инициализация. 1: слой (загрузка/создание). 2: токенизатор. 3: AtomicBasis (SVD при наличии чекпоинта). 4: FractalHierarchy. 5: MultiLevelGMM. 6: HNSWIndex. 7: KCA + SRG+ + SleepV2 + Grammar. 8: Federated + Ensemble + MultiPass + ContextCompressor. 9: MinimalCode + SelfImprovement + Curiosity. Плюс Progressive Bootstrapping: SVD-разложение обученной модели, извлечение коэффициентов c_i.

**`FCFSystem.query(text, max_tokens)`** — Полный 3-фазный когнитивный цикл.
- Фаза 1 (Perception): токенизация → c_query (Key-векторы) → HNSW Level 0 (поиск домена) → 3 сценария (exact_match/partial_match/cold_start) → HNSW Level 1 (поиск слепка).
- Фаза 2 (Genesis & KCA): генерация ответа → SRG-оценка → если confidence < 0.5: AtomicBasis.decode (модификация весов) → KCA refine_through_llm → если улучшило confidence → перегенерация ответа.
- Фаза 3 (Execution & Memory): валидация (confidence ≥ 0.8 + similarity > 0.95) → Code Distillation (выразим через существующие?) → Code Mutation (5%) → сохранение в HNSW + GMM → Code Provenance → StateGrammar composition → SelfDescriptiveCodes.

**`FCFSystem._validate_and_save(c_vec, confidence, domain_id, scenario)`** — 3-критериальная валидация: confidence ≥ 0.8, сценарий не exact_match, similarity с существующим < 0.95. При прохождении: CodeDistillation (проверка выразимости), CodeMutation (5% шанс улучшения), сохранение в HNSW и FAISS.

---

*FCF — Fractal Cognitive Fabric. Единая Вычислительная Архитектура. v2.0*
