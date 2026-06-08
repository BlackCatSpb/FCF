# FCF Architecture & Implementation Plan

## 1. Текущая архитектура (что есть сейчас)

### 1.1 Модули

| Модуль | Файл | Назначение |
|--------|------|------------|
| ConceptSkeleton | `concept_net.py` | 36,273 концепта из ConceptNet (имена + отношения) |
| ConceptTokenizer | `concept_tokenizer.py` | character-level BPE + WORD_OPEN/CLOSE границы + concept_id per token |
| ConceptSpace | `concept_space.py` | 128D векторы из SVD на 923k концепт→концепт переходах |
| SyntaxLattice | `syntax_lattice.py` | 2/3-gram n-граммы концептов из корпуса |
| CrystalGenerator | `crystal_generator.py` | Beam search + RRF(K=3) + STDP + hormonal modulation |
| HormonalSystem | `hormonal_system.py` | DA/5HT/NA/ACh модуляция параметров генерации |
| ConceptInductor | `concept_inductor.py` | Semantic resonance → новые meta-концепты (CID ≥ 10⁸) |
| POSTagger | `pos_tagger.py` | pymorphy3 обёртка: POS, features, agreement, POS_bigrams |

### 1.2 Поток данных (сейчас)

```
Вход: "на улице хорошая погода"
  │
resolve_anchor("на")   → (cid_на, 1.0)
resolve_anchor("улице") → orthographic → dice("улице","улица")=0.83 → (cid_улица, 0.83)
resolve_anchor("хорошая") → direct → (cid_хороший, 1.0)
resolve_anchor("погода") → direct → (cid_погода, 1.0)
  │
project_intent([на, улице, хорошая, погода])
  → intent_vec = mean(cid_на, cid_улица, cid_хороший, cid_погода)
  → closest_concept(k=5) → prefer non-query
  → intent_cid = ??? (может быть любой, semantic noise)
  │
generate(intent_cid)
  → beam search: concept → concept → concept (прыжки между 36k концептами)
  → текст: случайный набор концепт-связей из корпуса
```

**Проблема**: resolve_anchor даёт равный вес всем словам. "на" имеет тот же вес, что "погода". Intent centroid — среднее арифметическое шума и сигнала.

### 1.3 Ключевые баги (пофикшены)

| Баг | Фикс |
|-----|------|
| RRF K=60 → rank 1 vs 10 = 1.15× | K=3 → rank 1 vs 10 = 3.25× |
| theta_tau=5 → temp падает до 0.07 к слову 10 | theta_tau=12 + floor 15% |
| lateral inhibition seed=42 (всегда один набор) | динамический: `cid + _inhibition_step` |
| beam_width=3 + random inject | beam_width=5, без random inject |
| 4-gram (median count=1) | отключены, min_count=3 |
| repetition: только exact match | 6-window `0.05^count` + A→B→A→B `0.01` |
| 26% zero vectors (9,499 концептов) | ConceptNet neighbors + random fallback |

---

## 2. План реализации

### 2.1 Semantic Sieve Gate (ключевое изменение)

**Философия**: gate — не нейросеть, не transformer. Это **семантическое сито**, которое из входного потока слов выделяет единственную смысловую сущность — то, о чём запрос. Всё остальное — модификаторы/отношения вокруг неё.

```
Вход: ["на", "улице", "хорошая", "погода"]
  │
  │  Gate:
  │    ┌─── проекция каждого слова в concept space
  │    ├─── weighted centroid (вес: nouns > verbs > adj > function)
  │    ├─── поиск semantic attractor'а
  │    ├─── определение core vs modifier для каждого слова
  │    └─── определение connection strength modifiers → core
  │
  ├── CORE:     cid_погода (semantic attractor запроса)
  ├── MODIFIERS: {улица: {relation: located_at, strength: 0.7},
  │               хорошая: {relation: has_quality, strength: 0.9}}
  └── NOISE:    на (connection to nothing significant)
```

#### Реализация (symbolic, без PyTorch)

```python
class SemanticGate:
    """
    Семантическое сито: 
    1. Проецирует каждое слово в concept space (через существующий resolve_anchor)
    2. Определяет semantic type слова (core vs modifier vs noise) 
       НЕ через hardcoded POS, а через анализ:
       - расстояние до semantic centroid запроса
       - повторяемость слова в роли core в корпусе
       - connection strength к другим словам запроса
    3. Выделяет core concept (аттрактор)
    4. Строит modifier field вокруг core
    """
    
    def extract_core(words, cs, lattice):
        # 1. resolve всех слов
        anchors = [(w, cs.resolve_anchor(w)) for w in words]
        
        # 2. Вычисление semantic weight для каждого слова
        #    Вес = f(word_freq_as_core + semantic_breadth + connection_density)
        #    Слова, которые часто выступают core (существительные в корпусе)
        #    получают высокий core_score. Слова, которые всегда модифицируют
        #    (прилагательные, предлоги) — низкий.
        weights = {}
        for w, (cid, conf) in anchors:
            freq = lattice.concept_freq.get(cid, 0)
            # Чем выше freq как самостоятельного концепта, тем больше core_score
            core_score = min(freq / avg_freq, 1.0)  # 0..1
            weights[w] = core_score * conf
        
        # 3. Weighted centroid (core concept = weighted mean)
        #    Модификаторы и шум имеют низкий вес → не влияют на centroid
        centroid = weighted_mean(anchors, weights)
        
        # 4. Semantic attractor: ближайший core concept к centroid
        #    Если centroid достаточно близок к существующему → он
        #    Если centroid далёк от всех → создать новый core concept
        nearest = topk_similar(centroid, k=3, exclude_function_words=True)
        if cosine(centroid, nearest[0]) > THRESHOLD:
            core_cid = nearest[0]
        else:
            core_cid = create_concept(centroid)  # адаптивное расширение
        
        # 5. Определение modifier field
        modifiers = {}
        for w, (cid, conf) in anchors:
            if cid == core_cid:
                continue
            # connection strength = cosine similarity + co-occurrence
            conn = connection_strength(core_cid, cid, cs, lattice)
            if conn > NOISE_THRESHOLD:
                modifiers[cid] = {'strength': conn, 'word': w}
        
        return core_cid, modifier_field, centroid
    """
```

#### Свойства gate

- **Адаптивность**: не hardcoded список "это существительное". Gate учится из данных: если слово часто используется как semantic attractor → core_score растёт. Если всегда модифицирует → падает.
- **Никакой нейросети**: чистая геометрия vector space + частотная статистика. Веса не backprop, а кумулятивное накопление из наблюдений.
- **Query-зависимость**: gate смотрит connection strength между словами запроса. "Хорошая" сильнее связана с "погода" чем с "улица" → modifier погоды.
- **Расширяемость**: если запрос про новую сущность → создаётся новый core concept. Никакого фиксированного числа.

### 2.2 Токенизатор как Hierarchical Semantic Extractor

**Текущий** tokenizer: `char BPE → WORD_OPEN/BPE/WORD_CLOSE → concept_id per token`

**Новый** tokenizer — не просто разбиение строки, а **извлечение иерархической структуры**:

```
Текст
  │
  ├── Sentence split
  │     │
  │     ├── Word tokenization
  │     │     │
  │     │     ├── Morpheme analysis (замена char BPE)
  │     │     │   root → semantic core слова
  │     │     │   prefix → directional modifier
  │     │     │   suffix → aspectual/temporal modifier
  │     │     │   ending → grammatical role (case, number, gender, person)
  │     │     │
  │     │     └── Vector assembly:
  │     │         word_vector = root_core + prefix_shift + suffix_shift + ending_shift
  │     │
  │     ├── Phrase structure
  │     │     noun phrases: core + adjective field
  │     │     verb phrases: core + adverb field
  │     │     prepositional: relation marker
  │     │
  │     └── Sentence vector
  │         centroid of phrase vectors + inter-phrase connections
  │
  └── Hierarchical compression:
      words → phrases → sentence → paragraph → section
```

**Замена BPE на морфемный анализ**:
- Вместо `bpe.encode("хорошая") → ["хор", "ошая"]` (случайные subword границы)
- `pymorphy3.parse("хорошая") → root="хорош", ending="ая", POS=ADJ, case=nom, number=sg`
- `word_vector = root_vector("хорош") + ending_shift({"ая": ADJ_NOM_SG})`

#### Иерархическое сжатие (symbolic)

```python
def hierarchical_compress(vectors, levels):
    """
    vectors: список векторов (слова, фразы, предложения)
    levels: количество уровней сжатия
    
    Для каждого уровня:
      1. Группировка по semantic attractor'у
      2. centroid группы
      3. residuals = vectors - centroid
      4. Если residuals значимы → рекурсия
      5. Иначе: только centroid
    
    Returns: tree of {centroid, residuals, children}
    """
```

**Из старого кода**: `RecursiveTensorPotentialField.decompose(x) → [v₁..vₖ]` — адаптируется как группа слов → centroid + residuals. `compose()` — восстановление из centroid + residuals. Loss = `||compose(decompose(x)) - x||²`.

**Из старого кода**: `HierarchicalAdditiveField` — хранение как суммы суб-векторов с skip-connection гарантией восстановления.

### 2.3 Generator — не beam search, а field exploration

**Сейчас**: beam search: `[seed] → _branch() → candidates → prune → repeat`. Прыжки concept→concept.

**Новый**: 
```
Gate определил: core = cid_погода, modifiers = {улица, хорошая}

generate(core_cid, modifier_field):
    # 1. Определить все аспекты core concept
    aspects = decompose_core(core_cid)  # [аспект_1, ..., аспект_K]
    
    # 2. Для каждого шага генерации:
    #    - Выбрать аспект core concept (а не прыгать к другому core)
    #    - Выразить через modifier field
    #    - Подтвердить связь с запросом (gate verification)
    
    # 3. Критерий завершения:
    #    - Все значимые аспекты core выражены
    #    - Connection strength к запросу упала ниже порога
    #    - EOS (точка)
```

**Из старого кода**: `WordValenceField` — `left(core) ⊗ right(core)` предсказывает, что может стоять слева/справа от core. Адаптация: symbolic версия через connection strength из lattice.

**Из старого кода**: `TensorPotentialField.P[i][j][k]` — core concept i активирует силу связи j→k. Адаптация: `P[core][аспект_j][слово_в_поле]` = насколько слово_в_поле семантически релевантно для выражения аспекта_j core'а.

### 2.4 SyntaxLattice — Connection Strength Graph

**Сейчас**: `2/3-gram: prefix_tuple → Counter[next_cid]`. "После A вероятность B."

**Новый**: `connection(core_A, core_B) → {type, strength, context_field}`

```python
class ConnectionGraph:
    """
    Не n-граммы, а семантические связи между core concepts.
    
    connection_type определяется через environment слов:
    - "на улице погода" → located_at
    - "хорошая погода" → has_quality
    - "погода испортилась" → state_change
    
    strength = co-occurrence_count / total + cosine_similarity(core_A, core_B)
    """
    
    def get_connection(core_A, core_B):
        # Из lattice + vector space
        ngram_prob = predict([core_A, core_B])  # существующий predict
        cosine_sim = cosine(concept_vector(core_A), concept_vector(core_B))
        
        # Тип связи: по environment словам между cores
        env_words = words_between(core_A, core_B)  # предлоги, союзы
        relation_type = infer_relation(env_words)
        
        return {
            'strength': 0.6 * ngram_prob + 0.4 * cosine_sim,
            'type': relation_type,
            'context': env_words,
        }
```

### 2.5 Адаптивное количество концептов

**Сейчас**: фиксированные 36,273 концепта из ConceptNet. Новые meta-концепты на уровне 10⁸.

**Новый**: любой semantic attractor, достаточно далёкий от существующих, становится новым core concept.

```python
def find_or_create_attractor(centroid):
    nearest = topk_similar(centroid, k=1)
    if nearest and cosine(centroid, nearest[0]) > ATTRACTOR_THRESHOLD:
        return nearest[0].cid
    else:
        # Создать новый core concept
        new_cid = next_available_cid()
        concept_vectors[new_cid] = centroid
        cid_list.append(new_cid)
        concept_info[new_cid] = {
            'anchor': 'auto_' + str(new_cid),
            'satellites': [],
            'size': 1,
            'is_core': True,
            'birth_query': current_query,  # для traceability
        }
        return new_cid
```

### 2.6 Hormonal System — модуляция exploration radius

**Сейчас**: 5HT → temperature, NA → beam_width, DA → lr, ACh → plasticity.

**Новый**: добавить:
- DA → **gate strictness**: высокий DA → gate пропускает только сильные cores, низкий DA → gate пропускает больше кандидатов (exploration)
- NA → **exploration radius вокруг core**: высокий NA → близко к core, низкий NA → широкое поле
- 5HT → **modifier sensitivity**: высокий 5HT → только сильные modifiers, низкий → все modifiers включая слабые

### 2.7 Training Paradigm — декод текста в метаданные

**Ключевая идея**: обучение — не gradient descent весов, а **извлечение структуры из текста и организация семантического пространства**.

```
Текстовый корпус
  │
  ├── 1. Декод до метаданных:
  │     Для каждого предложения/документа:
  │       а) Морфемный анализ: root + affixes каждого слова
  │       б) Определение core concept (существительные, semantic attractors)
  │       в) Modifier field: adj→noun, adv→verb, prep→relation
  │       г) Иерархия: word → phrase → sentence
  │       д) Temporal markers: время глагола, порядок событий, длительность
  │       е) Connection strength: co-occurrence + cosine distance
  │
  ├── 2. Внесение метаданных:
  │     Каждое слово аннотируется:
  │       - его core concept (если слово = core)
  │       - его modifier target (если слово = modifier)
  │       - connection vectors к соседям
  │       - временная метка (relative position in narrative)
  │
  ├── 3. Организация семантических связей:
  │     Для каждого core concept:
  │       - какие modifiers вокруг него встречаются (и с какой частотой)
  │       - какие cores рядом (connection strength)
  │       - какие аспекты выражаются (decomposition)
  │       - какие temporal patterns (sequence order)
  │
  └── 4. Итог:
        concept vectors (128D) + connection graph + modifier fields + temporal grid
```

#### Что значит «декод до метаданных»

Не нейросетевая экстракция признаков, а **структурированное извлечение**:

```
Вход: "Вчера шёл сильный дождь, но к вечеру распогодилось."
  │
  ├── Sentence 1: "Вчера шёл сильный дождь"
  │     ├── CORE: дождь (сущ., subject)
  │     │     └── MODIFIERS: сильный {adj→noun, attr:quality, strength:0.9}
  │     ├── CORE: шёл (глаг., predicate)
  │     │     └── MODIFIERS: вчера {adv→verb, attr:time, strength:0.8}
  │     ├── CONNECTION: дождь ←subject→ шёл (strength:0.85)
  │     └── TEMPORAL: вчера (time_anchor = -1 day from narrative now)
  │
  ├── Sentence 2: "но к вечеру распогодилось"
  │     ├── CORE: распогодилось (глаг., predicate)
  │     │     └── MODIFIERS: к вечеру {prep→time, attr:time_boundary}
  │     ├── CONNECTION: sentence_1 →contrast→ sentence_2 (но)
  │     └── TEMPORAL: к вечеру (time_anchor = after sentence_1)
  │
  └── HIERARCHY:
        doc → [sent_1, sent_2]
        sent_1 → [{дождь}, {шёл}]
        {дождь} → [сильный]
        {шёл} → [вчера]
```

#### Чему модель «учится» на этом

Модель не оптимизирует loss. Она **накапливает статистику**:

| Что | Как хранится | Источник |
|-----|-------------|----------|
| Core→modifier patterns | `field[core_cid][mod_cid] = {freq, strength}` | Встречаемость в корпусе |
| Core→core connections | `connection[A][B] = {count, type_dist}` | Co-occurrence в предложениях |
| Аспекты core | `aspects[core_cid] = [cluster_1, ..., cluster_K]` | SVD на modifier vectors вокруг core |
| Temporal patterns | `temporal[cid_A][cid_B] = {time_delta, count}` | Порядок следования в тексте |
| gate core_score | `role_memory[word] = {core_count, mod_count, noise_count}` | Кумулятивно из всех вхождений |

#### Почему это не нейросеть

Нейросеть учит отображение `input → output` через backprop. Здесь:
- Нет функции потерь в классическом смысле
- Нет градиентов
- Нет эпох и батчей
- Есть **накопление структуры**: каждое прочтение текста добавляет/уточняет связи
- Есть **консолидация**: периодическое сжатие (SVD, кластеризация) для обобщения

Это ближе к построению семантической карты территории, а не к обучению аппроксиматора.

### 2.8 Temporal Markers — временная разметка

Temporal markers добавляют **измерение времени** в семантические связи:

```
core_концепт → {modifiers, connections, TEMPORAL_SLOT}

TEMPORAL_SLOT = {
    narrative_time:  (до/после/одновременно с другими событиями)
    absolute_time:   (вчера, сегодня, завтра, timestamp)
    duration:        (мгновенно/длительно/циклически)
    aspect:          (завершено/процесс/начинается)
}
```

#### Как temporal markers влияют на генерацию

```
Запрос: "какая была погода вчера?"
  │
Gate: core = погода, modifiers = {вчера}
  │
Temporal constraint: время = вчера (past, завершено)
  │
Generator: навигация вокруг core = погода
  │  └── filtered by temporal constraint:
  │      - было солнечно ✓ (past)
  │      - будет дождь ✗ (future — mismatch)
  │      - идёт снег ✗ (present — mismatch)
  │      - вчера было тепло ✓ (past + temporal word match)
  │
  → "Вчера было тепло и солнечно, без осадков."
```

#### Временная сетка

Temporal markers организуются не как линейная шкала, а как **поле связей**:

```
"Вчера шёл дождь. Сегодня выглянуло солнце."
  │
  ├── вчера → [дождь]
  ├── сегодня → [солнце]
  ├── CONNECTION: дождь →follows→ солнце (вчера→сегодня)
  └── TEMPORAL GRID:
        вчера ────────── дождь
          │                  │
          │   (затем)        │
          │                  │
        сегодня ──────── солнце
```

### 2.9 Очерёдность реализации

```
Phase 0: Semantic Sieve Gate
  - Реализовать SemanticGate.extract_core() 
  - Заменить project_intent() на gate
  - Интегрировать в generate()

Phase 1: Morpheme-based tokenizer
  - Заменить char BPE на pymorphy3 морфемный анализ
  - root → vector, affixes → shifts
  - HierarchicalAdditiveField symbolic версия

Phase 2: Connection Strength Graph
  - Дополнить SyntaxLattice методами connection_type
  - infer_relation() по environment словам
  - Интегрировать в RRF scoring

Phase 3: Field exploration generator
  - decompose_core() — разложение core на аспекты
  - Вместо beam search: exploration modifier_field
  - Gate verification каждого шага

Phase 4: Hierarchical compression
  - word → phrase → sentence → paragraph
  - centroid + residuals на каждом уровне
  - KV-кэширование сжатых представлений

Phase 5: Adaptive concept count
  - ATTRACTOR_THRESHOLD tuning
  - Автоматическое создание новых core concepts
  - Миграция старых 36k концептов
```

---

## 3. Сравнение: было → стало

| Аспект | Было (сейчас) | Станет |
|--------|---------------|--------|
| Единица генерации | concept ID (cid) | Core concept + modifier field |
| Токенизация | char BPE | morpheme analysis (root + affixes) |
| Intent | mean of all words (весь шум внутри) | Semantic Gate: core + modifiers (сито) |
| Gate | hardcoded POS list | Learned core_score из role_memory |
| Syntax | concept n-grams (последовательность) | connection strength graph (семантика) |
| Поиск | beam search с прыжками концепт→концепт | field exploration вокруг core (навигация) |
| Количество концептов | фикс 36k + meta (10⁸) | полностью адаптивное (аттракторы) |
| Связность текста | случайная (прыжки по 36k) | гарантирована (внутри поля core) |
| STDP | concept→concept vector shift | уточнение centroid/modifier geometry |
| Tokenizer output | flat [token_ids] | tree {core, modifiers, connections, time} |
| Обучение | gradient descent (не используется в symbolic) | декод текста → метаданные + орг. связей |
| Модель мира | частотная таблица n-грамм | семантическая карта с временной сеткой |
| Время | отсутствует | temporal markers как dimension связей |
| Роль модели | запомнить порядок концептов | запомнить как слова орбитируют вокруг ядер |
