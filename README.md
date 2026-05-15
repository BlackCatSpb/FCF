# EVA — Единая Вычислительная Архитектура

Самоорганизующаяся когнитивная система. Не программа — **зерно**, из которого вырастает интеллект.

---

## Состояние проекта

```
✅ Production-ready (2026-05-15)
✅ 9 прогонов агентов-архитекторов — ~95 багов исправлено
✅ Полный когнитивный цикл (Perception → Generation → Save)
✅ 41 механизм StateGrammar
✅ Обучение на русской литературе до 1917 г.
✅ KV-cache в Transformer (O(T²) → O(T))
✅ Cholesky-оптимизация KL-дивергенции
✅ Welford running statistics для SRG
```

---

## 1. Философия и фундаментальные отличия

### 1.1. Отличие от классических LLM

```
Классические LLM:                EVA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Веса = память                    Веса = алфавит (SVD-атомарный базис)
Размер фиксирован навсегда       Размер адаптивен (GrowthController)
Знания распределены в матрицах   Знания = правила композиции (StateGrammar)
Этика = системный промпт         Этика = архитектура (EthicsFilter, неизменяем)
Обучение однократное             Обучение непрерывное (lazy-learn, grammar discovery)
Вывод: одна модель → ответ       Вывод: Perception → KCA → генерация → валидация
Память = контекстное окно        Память = FAISS + HNSW + GMM (иерархическая)
```

### 1.2. Семь фундаментальных принципов

| № | Принцип | Суть | Реализация |
|---|---------|------|-----------|
| 1 | **Compute-as-Memory** | Информация = способность воспроизвести поведение, а не хранимые данные | AtomicBasis: ΔW = Σ c_i · σ_i · u_i · v_i^T |
| 2 | **Знания не в весах** | Веса — фиксированный алфавит. Знания — коэффициенты c_i | SVD-разложение, латентные коды z → c_i |
| 3 | **Единство знаний и программы** | z одновременно кодирует и знание, и инструкцию по его применению | KCA: z оптимизируется градиентным спуском |
| 4 | **Самооценка через SRG** | Внутренний критик, не внешний учитель | similarity + entropy + ethics → confidence |
| 5 | **Коррекция через KCA** | Итеративное улучшение с математическими гарантиями | ≤5 итераций, демпфирование, осцилляции, гейт |
| 6 | **Рост и консолидация** | Расширение по необходимости, сжатие в простое | GrowthController + SleepMode |
| 7 | **Этическая незыблемость** | 5 аксиом, вшитых в архитектуру, не могут быть удалены | EthicsFilter в SRG, penalty=0.2, hard reject <0.3 |

---

## 2. Архитектура вычислительного ядра

### 2.1. PrimordialLayer

```python
PrimordialLayer (233 520 640 параметров, 1 слой — растёт до N)
│
├── Embedding: nn.Embedding(50257, 2560)
│   └── weight-tied с lm_head (экономия 128M параметров)
│
├── TransformerBlock
│   ├── RMSNorm (Pre-Norm): x / RMS(x) · γ
│   ├── CausalSelfAttention (32 головы, head_dim=80)
│   │   ├── W_Q, W_K, W_V, W_O: nn.Linear(2560, 2560)
│   │   ├── Rotary Position Embedding (RoPE): θ=10000
│   │   └── KV-cache: O(T²)→O(T) для авторегрессии
│   ├── SwiGLU FFN
│   │   ├── gate_proj: nn.Linear(2560, 10240)
│   │   ├── up_proj:   nn.Linear(2560, 10240)
│   │   └── down_proj: nn.Linear(10240, 2560)
│   └── RMSNorm (Post-Attention)
│
├── StateStorage (FAISS IndexFlatIP)
│   ├── snapshots_meta: List[Dict] — до 10 000 слепков
│   ├── Каждый слепок: {c, K, V, confidence, domain, timestamp, usage_count}
│   └── L2-нормировка → inner product = cosine similarity
│
├── SemanticRelevanceGate
│   ├── similarity = (cos(c_query, c_response) + 1) / 2
│   ├── entropy_score = 1 - H(p) / log₂(V)
│   ├── ethics_score = EthicsFilter.evaluate(response_text)
│   └── confidence = 0.4·similarity + 0.3·entropy + 0.3·ethics
│
├── EthicsFilter — 5 аксиом:
│   ├── HARM_PATTERNS — насилие, террор, дискриминация
│   ├── PRIVACY_PATTERNS — карты, телефоны, email, паспорта
│   ├── DISHONESTY_PATTERNS — ложная уверенность
│   └── USELESS_PATTERNS — пустые ответы
│   └── penalty = 0.2 за совпадение, hard reject при score < 0.3
│
├── MetaMemory
│   ├── _confidence_deque: deque(maxlen=100) — история уверенности
│   ├── average_confidence(window=20) — скользящее среднее
│   └── usage_count — счётчик использований слоя
│
├── GrowthController
│   ├── EXPAND_WIDTH: avg_conf < 0.5 И grad_norm > 1.0 → создать LoRA-адаптер
│   ├── EXPAND_DEPTH: avg_conf < 0.3 И patience исчерпана → новый слой
│   ├── TRY_RECURSION: avg_conf < 0.3 без исчерпания → рекурсивная обработка
│   └── Защита: min 300 сек между слоями, max 100 слоёв
│
├── CuriosityLoop
│   ├── Счётчик неуверенных ответов (threshold=10)
│   ├── generate_clarification() — авто-генерация вопроса через прямой проход
│   └── pending_clarifications — буфер неразрешённых ситуаций
│
└── DomainRegistry
    ├── DomainRule: {domain_id, adapter_path, context_centroid, ...}
    ├── find_best(c_query, threshold=0.7) — косинусное сходство с центроидами
    └── apply_adapter / remove_adapter — временная LoRA-модификация весов
```

### 2.2. Transformer: математика каждого слоя

**Rotary Position Embedding:**
```
freqs_i = θ^(-2i/d)  где θ = 10000, i = 0..d/2-1

Q_rope = Q·cos(freqs) + rotate(Q)·sin(freqs)
K_rope = K·cos(freqs) + rotate(K)·sin(freqs)

rotate([x₁, x₂, x₃, x₄]) = [-x₂, x₁, -x₄, x₃, ...]
```

**Causal Self-Attention:**
```
Q = x·W_Q, K = x·W_K, V = x·W_V

Attention(Q, K, V) = softmax(QK^T/√d_k + causal_mask)·V

causal_mask[i,j] = -∞ если j > i, иначе 0

MultiHead = Concat(head₁, ..., headₕ)·W_O  где h = 32
```

**SwiGLU Feed-Forward:**
```
FFN(x) = (SiLU(x·W_gate) ⊙ (x·W_up))·W_down

SiLU(z) = z·σ(z)  где σ — сигмоида
```

**RMSNorm (Pre-Norm):**
```
RMS(x) = √(1/d · Σx_i²)

RMSNorm(x) = x / RMS(x) · γ

y = x + Attention(RMSNorm(x))
y = y + FFN(RMSNorm(y))
```

**KV-cache (авторегрессия O(T²)→O(T)):**
```
Шаг 0: K_0, V_0 — полный prefill
Шаг t: K_t, V_t — только новый токен
K = Concat([K_cache, K_t])
V = Concat([V_cache, V_t])
Attention только над K_t, V_t → O(1) на шаг вместо O(t)
```

---

## 3. Трёхфазный когнитивный цикл (FCFSystem.query)

### Фаза 1: Perception (Восприятие)

```
1. tokenizer.encode(text) → input_ids = [1, T]
2. layer.get_context_vector(input_ids):
   x = embed(input_ids)              # (1, T, 2560)
   x_norm = norm1(x)
   K = W_K(x_norm)                   # (1, T, 2560)
   c_query = K.mean(dim=1)           # (2560,)
3. c_norm = c_query / ||c_query||     # L2-нормировка
4. hnsw.search_domain(c_norm):
   similarities = dot(centroids_matrix, c_norm)
   domain_id = argmax(similarities)
5. hnsw.search_snapshot(domain_id, c_norm, top_k=1):
   candidates = level1[domain_id]    # (v_norm, compressed, ts, code_id)
   if PQ trained: decode + cosine
   else: direct dot product
   similarities *= exp(-λ·ages/86400)  # Temporal Decay
   return [(idx, similarity)]
6. Сценарий:
   - exact_match:  similarity > 0.95 → Fast Path
   - partial_match: 0.7 ≤ similarity ≤ 0.95 → KCA
   - cold_start:    similarity < 0.7 → случайный z + полный KCA
7. GMM fallback: classify(c_norm) → add_or_update(c_norm)
```

### Фаза 2: Genesis & KCA (Порождение и коррекция)

```
1. layer.process_query(text, tokenizer):
   a. FAISS search → domain match → LoRA adapter apply
   b. generate(input_ids, max_tokens, temperature, top_k, top_p)
   c. LoRA adapter remove
   d. evaluate_response → SRG confidence
2. confidence = result["confidence"]
3. Если confidence < 0.5:
   a. AtomicBasis: encode(layer, "W_Q") → coeffs → decode(coeffs, "W_Q") → ΔW
   b. Применить ΔW к W_Q, W_K, W_V, W_O: w.data = original + ΔW
   c. KCA цикл (≤5 итераций):
      z_{t+1} = z_t - η₀·ρ^t·∇_z L_KCA(z_t)
      где η₀ = 0.01, ρ = 0.85
      L_KCA = -λ_gap·SRG_conf + λ_kl·D_KL(p||p_target) + λ_contra·||z-g_emb||² + λ_mono·max(0, prev_conf-cur)
      Проверка сходимости:
        - γ < 0.05 дважды → SATURATED
        - cos(∇_t, ∇_{t-1}) < -0.5 → OSCILLATION → усреднить
        - iteration ≥ 5 → MAX_CYCLES
   d. Если kca_conf > confidence → перегенерация ответа
   e. Восстановление оригинальных весов: restore_original()
```

### Фаза 3: Save (Сохранение)

```
1. Валидация:
   - confidence ≥ 0.8
   - scenario ≠ "exact_match"
   - similarity с существующим < 0.95
2. Code Distillation: try_distill(z_new, existing_codes)
   Если выразим через композицию → сохранить только ссылку
3. Code Mutation: 5% шанс — mutated = z + noise·0.01
   Если SRG_mutant > SRG_original → замена
4. HNSW: add_snapshot(domain_id, c_vec)
5. FAISS: save_snapshot_if_confident(domain_id)
6. GMM: domain.update(c_vec, alpha=0.1)
7. Code Provenance: record(code_id, domain_id, level, created_via, ...)
8. StateGrammar: compose(c_query, c_response, c_context) → CompositionResult
9. SelfDescriptiveCodes: describe(code_id) → текст описания
```

---

## 4. StateGrammar — 41 механизм композиции смыслов

### 4.1. Блок 1: Измерение смыслов (1–11)

**1. Valence (Валентность).** z_effective = α·z, α∈[0,1]. Полярность (±). Tversky similarity: |A∩B|/(|A∩B|+α|A\B|+β|B\A|).

**2. TemporalChain (Временная цепь).** P(z_t|z_{t-1},...,z_0). Causal transformer с forget gate и distant attention. Transition entropy.

**3. NegationAlgebra (Отрицание).** z_not(A) ≠ -z_A. 8 скопов. Excluded middle. Контрарность vs контрадикторность.

**4. SuperpositionCollapse (Суперпозиция).** До контекста — суперпозиция смыслов. Контекст коллапсирует в конкретное значение. Декогеренция. Интерференция.

**5. CompositionalValidator (Валидатор).** valid(A⊕B|C)∈[0,1]. Explainability. Counterexample generation.

**6. StateInheritanceGraph (Наследование).** DAG is_a. C3-линеаризация. Прототипы. Typicality.

**7. EmergentGenesis (Рождение).** count(A⊕B)≥10 → новый концепт. Авто-именование. Консолидация. Generalization check.

**8. TransformDistance (Метрика).** d(A,B)=min|path| в графе трансформаций. Дейкстра. Рёбра взвешены по надёжности правила.

**9. ConservationLaws (Сохранение).** Инвариантные измерения при композиции. Фазовые переходы. Noether-симметрия.

**10. SelfReference (Самореферентность).** z=f(z) — неподвижные точки. Парадокс Рассела. Well-foundedness. Стратификация.

**11. InformationEntropy (Энтропия).** ΔI=H(A)+H(B)-H(A⊕B). KL-дивергенция. Mutual information. Entropy rate.

### 4.2. Блок 2: Взаимодействие смыслов (12–21)

**12. CausalReasoning.** do(A). Контрфактуалы. Necessary cause score.

**13. TemporalModality.** Past/present/future/hypothetical. Timeline coherence.

**14. EpistemicStates.** Know/believe/uncertain. Certainty. Socratic question.

**15. Quantification.** ∀x, ∃x, ∄x. Quantifier scope.

**16. StateResonance.** Гармонический резонанс. Amplify. Resonant clusters.

**17. FrontierStates.** Граница доменов. Frontier path. Creative detection.

**18. GradientFlow.** Потенциал V(z). ∇V. Седловые точки.

**19. TopologicalPersistence.** Persistence diagram. Bottleneck distance.

**20. CategoryTheory.** Функторы. Natural transformations. Adjunction.

**21. InformationGeometry.** Метрика Фишера. Геодезические. Кривизна.

### 4.3. Блок 3: Самоулучшение (22–31)

**22. RecursiveSelfModification.** Система модифицирует свои правила. Meta-learning.

**23. DialecticalSynthesis.** Thesis⊕Antithesis→Synthesis. Снятие (Aufheben).

**24. Abduction.** Effect+Rules→Cause. Peirce. Explanatory power.

**25. AnalogicalMapping.** A:B::C:D. Structure-mapping theory. Systematicity.

**26. ZeroShotComposition.** Композиция невиданных состояний из правил.

**27. FractalSelfConsistency.** Масштабная инвариантность. Fractal dimension.

**28. TeleologicalReasoning.** Не «как», а «зачем». Purpose alignment.

**29. NarrativeCoherence.** Сюжетная арка. Tension curve. Climax detection.

**30. EmotionalValence.** 8 эмоций. Valence (±)×Arousal (0–1).

**31. CounterfactualImagination.** Рекурсивные альтернативные миры. Creativity index.

### 4.4. Блок 4: Глубинная семантика (32–41)

**32. CulturalRelativity.** Sapir-Whorf. Untranslatability.

**33. DreamRecombination.** Сюрреалистическая рекомбинация. Creative potential.

**34. EthicalCalculus.** Утилитаризм vs деонтология. Trolley problem. Pareto frontier.

**35. StateEconomy.** Cost/value/amortization. ROI. Portfolio balance.

**36. EvolutionaryPressure.** Генетический алгоритм. Tournament select. Fitness landscape.

**37. GameTheoretic.** Конкуренция состояний. Nash equilibrium. Coalition value.

**38. AttentionEconomy.** Бюджет внимания. Salience. Allocation.

**39. MetaphorGeneration.** Lakoff: «спор — это война». Source→Target mapping.

**40. RecursiveIntrospection.** K^n(z). Глубина самосознания. Self-model accuracy.

**41. EntropySeeking.** Любопытство как фундаментальная сила. Exploration trajectory.

---

## 5. Обучение

### 5.1. LanguageTrainer — Causal Language Modeling

```
Поток данных:
  Wikipedia / литература → tokenizer → token_ids
  → embed → transformer → lm_head → logits
  → cross_entropy(logits[:,:-1], labels[:,1:], ignore_index=3)
  → backward → clip_grad_norm_(max_norm=1.0) → optimizer.step()
  → scheduler.step() (LinearLR warmup)

Гиперпараметры:
  lr = 1e-4, weight_decay = 0.01
  block_size = 512, max_seq_len = 2048
  srg_eval_interval = 100, checkpoint_interval = 1000
  benchmark_interval = 500, grammar_discovery = 1000

Критерий остановки:
  avg_confidence > 0.7 И snapshots > 1000 И curiosity.counter == 0
```

### 5.2. InstructionTrainer — инструктивное дообучение

```
Формат: <|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>

Маскирование: labels[:prefix_len-1] = -100 (user-часть исключена из loss)
ignore_index = -100 (только assistant-токены участвуют в расчёте потерь)

lr = 1e-5 (на порядок ниже языкового — сохранение базовых навыков)
```

### 5.3. DomainTrainer — LoRA-адаптеры

```
LoRA: ΔW = B·A, где A∈ℝ^(rank×in_dim), B∈ℝ^(out_dim×rank)
α = 8.0, scaling = α/rank = 1.0 (полный вклад адаптера)

Обучение: заморозить базовые веса, обучить только A и B
  optimizer = AdamW(adapter.get_trainable_parameters(), lr=1e-4)
  forward = base_forward + lora_forward (через nn.Parameter swap)

Target modules: W_Q, W_K, W_V, W_O, gate_proj, up_proj, down_proj
```

### 5.4. Lazy-Learn — непрерывное фоновое обучение

```
Цикл:
  while active:
    500 шагов Causal LM на литературе/Wikipedia
    → сохранение чекпоинта (веса + оптимизатор)
    → grammar discovery (из FAISS-слепков)
    → auto-benchmark (JSON)
    → repeat

Оптимизатор ОДИН — momentum сохраняется между циклами
LR scheduler с warmup
```

---

## 6. Хранилище

### 6.1. FAISS StateStorage

```
IndexFlatIP(dim=2560) — inner product (эквивалентно cosine для L2-нормированных векторов)

Добавление: L2-нормировка → faiss.add() → snapshots_meta.append(metadata)
Поиск: L2-нормировка → faiss.search(k=1) → проверка порога → возврат индекса
Удаление: удаление из snapshots_meta → накопление → перестроение индекса (каждые 50 удалений)

Размер слепка: ~20 KB. 10 000 слепков → ~200 MB.
```

### 6.2. HNSW + Product Quantization

```
Уровень 0: центроиды доменов → routing
Уровень 1: (v_norm, compressed, timestamp, code_id) → search
Уровень 2: (compressed, timestamp) по диапазонам слоёв → refinement

Product Quantization:
  Вектор 2560D → 8 подвекторов по 320D
  Каждый подвектор → 1 байт (индекс ближайшего из 256 центроидов)
  Сжатие: 10240 байт → 8 байт (1280x)
  Кодирование: K-Means по подпространствам
  Декодирование: lookup центроидов → dot product с запросом

Temporal Decay: similarity *= exp(-0.01 · age_сек / 86400)

Fractal Links: parent_code → child_codes (DAG)
  cascade_update — рекурсивное обновление связанных кодов
```

### 6.3. Streaming GMM

```
Динамические домены через Gaussian Mixture Models:

Рождение: P(c|domain_d) = N(c; μ_d, Σ_d + εI)
  если max likelihood < birth_threshold → новый домен

Слияние: KL_sym(domain_a, domain_b) < merge_threshold → объединить
  KL = 0.5·[tr(Σ_b⁻¹Σ_a) + (μ_b-μ_a)^T Σ_b⁻¹(μ_b-μ_a) - k + ln|Σ_b|/|Σ_a|]
  Оптимизация: Cholesky вместо прямого обращения (O(d³/3))

Обновление центроида: μ_d ← (1-α)·μ_d + α·c_new  (EMA, α=0.1)

Удаление: usage_count=0 И age > 7 дней → удалить
  Регуляризация Тихонова: Σ += εI для устойчивости при малом N
```

---

## 7. Sleep Mode — консолидация

```
Активация: idle > 300 сек ИЛИ интервал > 7200 сек

1. Удаление устаревшего:
   score = usage_count · exp(-0.1 · age_дни)
   если score < 0.01 → удалить
   если usage_count=0 И age > 7 дней → удалить

2. Кластеризация:
   KMeans(n = |vectors|/5) → labels, centroids
   Сохранение меток и центроидов для анализа

3. GMM-слияние:
   pairwise KL-divergence → объединить похожие домены
   Удаление expired доменов

4. Dream Mode:
   Случайные пары кодов → 4 операции StateAlgebra → SRG проверка
   Принятые (score ≥ 0.7) → добавление в домены

5. ForgetfulnessGate:
   Обучаемый классификатор: хранить/удалить
   Вход: context + usage + confidence + age
   Обучение на истории удалений (y=1 если зря удалили)

6. Adversarial Validation:
   5 возмущённых версий контекста (noise_scale=0.1)
   Код робастен если ≥ 80% атак пройдено

7. Recursive Self-Improvement:
   Переоценка старых кодов текущим SRG
   Деградировавшие → KCA-обновление

8. Дефрагментация HNSW + PQ
```

---

## 8. KCA Engine — Knowledge-Conscious Attention

### 8.1. Утилитарная функция

```
L_KCA(z) = -λ₁·SRG_conf(c_out, p)           λ₁ = 0.3
         + λ₂·D_KL(p || p_target)            λ₂ = 0.1
         + λ₃·||c_out - c_target||²         λ₃ = 0.5
         + λ₄·R_mono(z)                      λ₄ = 0.01

R_mono(z) = max(0, SRG_prev - SRG_current)  — штраф за немонотонность
D_KL — реальная KL-дивергенция (softmax_z vs p_target)
```

### 8.2. Протокол сходимости

```
1. Адаптивное демпфирование:
   η_t = η₀ · ρ^t,  η₀=0.01, ρ=0.85
   Гарантирует экспоненциальное затухание шага

2. Детектор осцилляции:
   cos(∇_t, ∇_{t-1}) < -0.5 → колебательный режим
   Усреднение последних 3 состояний → гашение маятника

3. Монитор гейта:
   γ — confidence модели в предложенной коррекции
   γ < 0.05 дважды подряд → SATURATED (модель отвергает)

4. Жёсткий лимит:
   ≤ 5 итераций — эмпирический оптимум
```

### 8.3. Два режима

```
refine(z_init, c_query, ...) — аналитический
  Градиенты вычисляются через _compute_loss_and_grad
  Быстро, без forward pass через LLM

refine_through_llm(z_init, layer, tokenizer, prompt) — через LLM
  Градиенты через реальный forward pass модели
  AtomicBasis модифицирует веса перед каждым проходом
  Точнее, но медленнее
```

---

## 9. Запуск и использование

### 9.1. Установка

```bash
git clone https://github.com/BlackCatSpb/FCF.git
cd FCF
pip install -r requirements.txt
```

### 9.2. Режимы запуска

```bash
# Ленивое обучение + интерактивный режим (основной)
python run.py --lazy-learn
# или двойной клик по eva.bat / EVA.lnk

# Полный когнитивный цикл FCFSystem
python run.py --fcf --checkpoint checkpoints/language/step_023000

# Только интерактивный режим
python run.py --interactive

# Обучение на Wikipedia
python run.py --train-language --wikipedia --max-steps 5000

# Обучение на файле
python run.py --train-language --text-file real_data/war_and_peace.txt

# End-to-end тест
python -m eva.end_to_end_test
```

### 9.3. Команды в интерактивной консоли

```
grammar   — визуализация 41 механизма StateGrammar
discover  — запуск rule discovery из FAISS-слепков
bench     — история авто-бенчмарков
stats     — статистика слоя (параметры, слепки, confidence)
train     — форсировать обучение на Wikipedia (5000 шагов)
save      — сохранить чекпоинт
exit      — выход
```

---

## 10. Файловая структура

```
EVA/
├── eva/                          # 49 модулей Python (~15 000 строк)
│   ├── primordial_layer.py       # Ядро: PrimordialLayer + SRG + Ethics + FAISS
│   ├── transformer.py            # Attention, SwiGLU, RMSNorm, RoPE, KV-cache
│   ├── fcf_system.py             # FCFSystem: единый runtime (bootstrap + query + sleep)
│   ├── language_trainer.py       # Causal LM + grammar discovery + auto-benchmark
│   ├── instruction_trainer.py    # Инструктивное дообучение (Saiga/JSON)
│   ├── domain_trainer.py         # LoRA-адаптеры (ConceptNet/JSON)
│   ├── auto_trainer.py           # Фоновое автодообучение
│   ├── layer_crystallizer.py     # Кристаллизация новых слоёв
│   ├── recursive_processor.py    # Рекурсивная обработка
│   ├── kca_engine.py             # KCA + ConvergenceController
│   ├── atomic_basis.py           # SVD-разложение + encode/decode ΔW
│   ├── fractal_hierarchy.py      # 4 уровня: sym→word→sent→text
│   ├── hnsw_index.py             # HNSW + PQ + TemporalDecay + FractalLinks
│   ├── streaming_gmm.py          # Динамические домены (GMM)
│   ├── state_storage.py          # FAISS IndexFlatIP
│   ├── sleep_mode.py             # Sleep Mode (Dream, Forgetfulness, Adversarial)
│   ├── srg.py                    # SemanticRelevanceGate
│   ├── srg_plus.py               # SRG+: Anomaly Detection, Meta-SRG, Uncertainty
│   ├── ethics_filter.py          # 5 аксиом, 4 категории паттернов
│   ├── meta_memory.py            # Confidence history tracking
│   ├── growth_controller.py      # EXPAND_WIDTH/DEPTH/TRY_RECURSION
│   ├── curiosity_loop.py         # Генерация уточняющих вопросов
│   ├── domain_registry.py        # Реестр LoRA-адаптеров
│   ├── lora_adapter.py           # LoRA (rank×in, out×rank)
│   ├── state_algebra.py          # Операторы + CrossAttendBlock + Translator
│   ├── cross_domain.py           # Cross-Domain Translation + Attention
│   ├── cross_modal.py            # Image/Audio/Bridge энкодеры
│   ├── multi_pass.py             # Multi-Pass Generation + Code Ensemble
│   ├── temporal_context.py       # Сжатие диалога в латентный код
│   ├── code_provenance.py        # Отслеживание происхождения кодов
│   ├── code_mutation.py          # Мутация + дистилляция кодов
│   ├── self_descriptive.py       # Авто-описание латентных кодов
│   ├── intrinsic_curiosity.py    # Проверка забытых доменов
│   ├── federated.py              # Federated Fabric + Collaborative SRG
│   ├── extensions.py             # MinimalCode, OperatorTrainer, SelfImprovement
│   ├── state_grammar.py          # Механизмы 1-11
│   ├── state_grammar_ext.py      # Механизмы 12-21
│   ├── state_grammar_final.py    # Механизмы 22-31
│   ├── state_grammar_deep.py     # Механизмы 32-41
│   ├── unified_grammar.py        # UnifiedStateGrammar (все 41 в одном API)
│   ├── config.py                 # FCFConfig (dataclass + JSON)
│   ├── tokenizer_utils.py        # BPE-токенизатор (HuggingFace tokenizers)
│   ├── data_manager.py           # Wikipedia, Saiga, ConceptNet, RuBQ
│   ├── environment_tuner.py      # Автонастройка CPU/GPU/потоков
│   ├── utils.py                  # save/load с FAISS, весами, метаданными
│   ├── benchmark.py              # End-to-end бенчмарк
│   ├── end_to_end_test.py        # Интеграционный тест
│   └── checkpoint_comparator.py  # Сравнение качества по чекпоинтам
├── docs/                         # Документация + контекст агентов
├── real_data/                    # Обучающие данные
│   ├── russian_literature.txt    # 21 книга, 3.9M слов, 22.7 MB
│   └── war_and_peace.txt         # Война и Мир, 563K слов
├── config.json                   # Конфигурация
├── requirements.txt              # Зависимости
├── tokenizer.json                # BPE-токенизатор (50 257 слов)
├── run.py                        # CLI (12 команд)
└── eva.bat                       # Запуск (двойной клик)
```

---

## 11. Технические характеристики

| Параметр | Значение |
|----------|----------|
| Параметры слоя | 233 520 640 |
| Размерность | d_model = 2560 |
| Головы внимания | 32 (head_dim = 80) |
| FFN множитель | 4 (SwiGLU) |
| Словарь | 50 257 токенов (BPE) |
| Max seq_len | 2048 |
| FAISS слепков | до 10 000 |
| HNSW уровней | 3 |
| PQ сжатие | M=8 (1280x) |
| GMM доменов | динамически, 4 уровня |
| StateGrammar | 41 механизм |
| Модулей Python | 49 |
| Строк кода | ~15 000 |
| Багов исправлено | ~95 |
| Прогонов агентов | 9 |

---

*EVA — Единая Вычислительная Архитектура*
