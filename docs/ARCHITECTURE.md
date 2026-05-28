# EVA Symbolic — Архитектура

> Символ ≡ координата в ℝ¹²⁸. Текст ≡ траектория. Знания ≡ рекурсивный тензор потенциалов + топология.

---

## 1. Фундаментальные принципы

### 1.1 Символ как координата
Каждый из 160 символов — не индекс в таблице, а **точка в 128-мерном координатном пространстве**. Позиция вычисляется не случайно — координаты получают из affinity к статистике co-occurrence в тексте, проецируя через MDS-подобную инициализацию.

### 1.2 Текст как траектория
Слово «привет» — не последовательность из 6 токенов, а **траектория из 6 точек** в ℝ¹²⁸. Предложение — последовательность траекторий слов. Всё — объекты одного типа, разница в масштабе.

### 1.3 Знания как топология + потенциалы
Модель (ядро ~1.14M params) — только навигатор. Знания хранятся отдельно:
- Координаты символов [160×128]
- Affinity-матрица [160×160]
- Рекурсивный тензор потенциалов [160×160×160] × K=6 уровней декомпозиции
- Траектории в TrajectoryStore
- Топология пути [160×160×3]

Модель можно заменить — знания останутся.

### 1.4 Инструкция как траектория
Генерация текста — не «предсказать следующий токен», а **исполнить инструкцию**: получить координатную траекторию → декодировать в символы. Трансформер — исполнитель, не предсказатель.

---

## 2. Компоненты системы

### 2.1 CharacterVocab — символьный словарь
160 токенов: PAD(0), UNK(1), BOS(2), EOS(3) + 152 печатных символа (кириллица, латиница, цифры, пунктуация) + 4 boundary-токена (`<W>`/157, `</W>`/158, `<S>`/159, `</S>`/160). `encode_with_boundaries()` оборачивает каждое слово в `<W>...</W>`, каждое предложение в `<S>...</S>`.

### 2.2 RecursiveTensorPotentialField — рекурсивный тензор потенциалов (4,293,383 params)

#### 2.2.1 Базовая структура
Трёхмерный тензор P[i][j][k] — сила связи j→k, активируемая символом i (4,096,000 params).
```
P[i][j][k] = P_i[j,k]  — 4M параметров, [V×V×V] для V=160
```

#### 2.2.2 Рекурсивная декомпозиция (+197,383 params)
Каждая координата x ∈ ℝ¹²⁸ декомпозируется на K=6 субвекторов:
```
decomp_proj: ℝ¹²⁸ → ℝ^(6·128) → reshape [6, 128]  — 98,304 params
gate_net:   ℝ¹²⁸ → ℝ^6 → softmax                  — 774 params

x → [v₁, v₂, ..., v₆], v_k ∈ ℝ¹²⁸
```

Каждый v_k — снова точка в ℝ¹²⁸ → квантизация к ближайшему символу → bias из base TPF.

#### 2.2.3 Батчевый BFS
```
Level 0: quantize(x) → P[idx] → bias_0
Level 1: decompose(x) → [v₁..v₆] → gate-filter → quantize each → P[idx] → bias_1 × depth_scale × gate
Level 2: decompose(each survivor) → ... → bias_2 × ... 
...
Max depth: 8, max paths: 4096
```
Все quantize — batched cdist, ни одного Python item(). N путей на уровне → один cdist.

#### 2.2.4 Gate filter
Каждый субвектор v_k взвешивается gate g_k = softmax(gate_net(x))_k.  
При g_k < 0.05 — ветка отбрасывается (экономия на BFS).  
Scale для bias на глубине d: `parent_scale × depth_scale × g_k`.

#### 2.2.5 Cycle detection
Если `||v_k - x||₂ < 0.01` — цикл. Вместо продолжения рекурсии — tensor product unfold:
```
bias = Σ P[idx_paths] ⊗ scale  (einsum 'iv,jv,kv→v' вдоль путей-циклов)
```

#### 2.2.6 Composition Loss (aux)
```python
def composition_loss(vectors):  # vectors: [B, L, D]
    subs, gates = decompose(vectors)     # [BL, K, D], [BL, K]
    recon = compose(subs)                # [BL, D]
    loss = MSE(recon, vectors)
    entropy = -Σ(gates · log(gates))     # gate entropy bonus
    return loss - 0.01 × entropy
```
Вес в total loss: **0.01**. Обучает decomp/compose быть взаимно-обратными. Gate entropy бонус (−0.01 × H) предотвращает коллапс к одному активному субвектору.

#### 2.2.7 Delegation к Base TPF
- `init_from_affinity()` → `base_tpf.init_from_affinity()`
- `update()` → `base_tpf.update()`
- `forward()` → `base_tpf()`

Все операции обучения/инициализации работают через Recursive → base. При resume из старого чекпоинта: `strict=False` + ручная миграция `tensor_potential.P → tensor_potential.base_tpf.P`.

#### 2.2.8 Полный bias для генерации
```python
bias = 0  # [V]
for depth, (P_depth, scale) in enumerate(zip(all_biases, all_scales)):
    bias += P_depth * scale
# P_depth = mean(P[syms_of_this_level, valid_context, :])
# scale = depth_scale^depth × Π(gates_along_path)
```

### 2.3 WordValenceField — словесная валентность (74K params)
Отображение координаты слова → матрица валентности [V, V].
- Outer-product декомпозиция: left_net(coord) × right_net(coord)
- bias для генерации: logits += WVF(word_coord) × 0.05

### 2.4 SentenceContextField — RBF-поле предложения (3 params)
Top-K центров из attention-весов → RBF-ядра с обучаемыми gamma.
Поле предложения: взвешенная сумма RBF + edge-векторы между словами.

### 2.5 StaticTopologyLayer — статическая матрица топологии
Тензор [160, 160, 3] — read-only bias:
- **Канал 0: affinity** — сила связи i→j
- **Канал 1: potential barrier** — высота барьера между i и j
- **Канал 2: forbidden** — 0/1 запрет

**Fast Path** (max 10K): кэш успешных траекторий с FIFO eviction. При генерации — поиск ближайшего центроида запроса.

### 2.6 TrajectoryStore — иерархическая память
4 уровня хранения: символьные координаты [L,D] → центроиды слов → connection-векторы → центроид предложения.
ConsolidationTransformer (Conv1d + gate, 6.5K params) — взвешенное усреднение шагов траектории.

### 2.7 SemanticRelevanceGate — семантический фильтр (0 params)
confidence = cosine_sim(coord, target) × cos_sim_curve × (1 - H/H_max) + ethics_bias. Пороговая фильтрация для генерации.

### 2.8 GradientFlowSolver — непрерывное рассуждение (0 params)
`dz/dt = -∇V(z) + η(t)`. Euler-Maruyama с детектором осцилляции (cos < -0.5 → демпфирование). Точки равновесия ||∇V|| < ε — ответы.

### 2.9 KCACycle — коррекция латентного кода (0 params)
Adam-оптимизация скрытого представления по L = -λ₁·SRG + λ₂·KL + λ₃·MSE. Экспоненциальное затухание lr.

### 2.10 Validation Suite — автоматическая валидация
Каждые 5000 шагов: генерация 7 seed'ов, тест на Анне Карениной (первые 100 предложений), SelfReflection (кривизна, confidence, пространственная эффективность).

---

## 3. Архитектура трансформера

### 3.1 CoordinateEmbedding
Координаты символов в ℝ¹²⁸ через `nn.Parameter`. Не nn.Embedding. Инициализация: ~N(0, 0.02), нормировка.

### 3.2 RoPE — Rotary Position Embeddings
Поворот вектора в 2D-подпространствах. θ = 10000.

### 3.3 MultiSubspaceEmbedding
Структурирует 128 измерений: dims 0–31 (symbol), знает о слове через CoordinateEmbedding. Без лишних проекций.

### 3.4 CoordinateDecoder
Линейный классификатор 128→160 + nearest-neighbour бонус + SubHSM group bias.

### 3.5 SubHSM — Subspace Hierarchical Softmax
4 группы по ~40 токенов. `group_classifier` → group_probs → bias на logits по группам. Group CE loss ×0.05.

### 3.6 AdaptiveFractalAttention
LevelController: `x.mean(dim=1) → gate_MLP → softmax → head_allocation`. Manifold bias: `exp(-||proj_2D(x_i) - proj_2D(x_j)||² / (2·scale²))` — вычисляется ОДИН раз per level (не per-head). Causal mask. CoordBias: L2-distance.

### 3.7 SGF — Subspace-Gated FFN
4 gate-вектора (128d), роутер 128→4 → softmax → blend. Subspace gate: different tokens processed differently.

### 3.8 Coordinate Residual Stream
Сквозной поток координат через все 6 слоёв. Element-wise gate merge + update. Информация не замывается.

### 3.9 TrajLoss — траекторная Auxiliary Loss
`MLP 128→64→128` предсказывает Δ-вектор следующей позиции. Target: embed(token) + pad(subspace(token)). MSE loss ×0.1.

### 3.10 Dilated KV-Cache
При генерации: для уровня dilation d — K/V каждого d-го токена.

---

## 4. Обучение

### 4.1 Данные
- **full_corpus_encoded.npy**: 106.5M токенов, boundary-разметка (680K предложений из full_corpus_ru.txt, 172 MB)
- Сплит на абзацы по `</S>`, каждый батч обрезается до границы предложения (`</S>`) или слова (`</W>`)

### 4.2 Тренировка (train_full_corpus.py)
- **Ядро**: 6 HybridFractalBlock слоёв, 128-dim, 32 heads → ~1.14M params
- **Потенциальные поля**: RecursiveTensorPotentialField (4.29M), WordValenceField (74K), SentenceContextField (3)
- **TrajPredictor**: +16.5K params
- **Всего**: ~5.49M params (ядро + поля)
- **Режим**: Resume c `full_latest.pt`, strict=False + миграция TPF.P → base_tpf.P
- **Батч**: B=8, seq=128, обрезка до `</S>` → `</W>`
- **Оптимизатор**: AdamW, lr=5e-3, cosine schedule, weight_decay=0.01
- **LR scheduler**: CosineAnnealing до 0 за 100K шагов
- **VRAM**: ~0.7 GB (NVIDIA GeForce MX550, 2.1 GB — 35% utilisation)
- **Скорость**: ~160 шагов/мин
- **Шагов**: 100 000

### 4.3 Loss
| Компонент | Функция | Вес |
|-----------|---------|-----|
| CE | cross_entropy(pred, target) | 1.0 |
| SubHSM group | cross_entropy(group_logits, group_target) | 0.05 |
| TrajLoss | MSE(pred_delta, actual_delta) | 0.1 |
| Composition | MSE(recon, identity) − 0.01·H(gates) | 0.01 |

### 4.4 Цикл обучения

**Каждые 50 шагов**: print loss, acc, traj_loss, group_loss, comp_loss.

**Каждые 500 шагов**:
1. Save checkpoint (`full_latest.pt`)
2. **Thinking Phase**:
   - Co-occurrence affinity (500 random samples × 128 tokens)
   - Init RecursiveTPF: `base_tpf.init_from_affinity(aff)`
   - Capture attention → `base_tpf.update()` (10 random blocks)
   - Extract trajectories → consolidate → store (5 random blocks)
   - Build topology from store (Fast Path cache)
   - Save TrajectoryStore (`trajectory_store_full.pkl`)

**Каждые 2500 шагов**: Random generation (3 случайных промта из корпуса, enhanced_generate с RecursiveTPF+WVF bias).

**Каждые 5000 шагов**: Full validation + **best checkpoint** (`full_best.pt`) по curvature.

### 4.5 Enhanced Generation Pipeline
```python
logits = CE_logits / temperature

# Recursive bias (BFS quantize на K=6, max_depth=8, max_cap=4096)
logits += RecursiveTPF.recursive_bias(x=hidden[-1], context=ids) * 0.1

# Word valence
logits += WVF.get_valence_bias(word_coord, context) * 0.05

# Adaptive repetition penalty (global frequency)
for t in ids: freq[t] += 1
for rank, t in enumerate(top-20 tokens):
    p *= 0.5 ** freq[t]  # каждый повтор режет вероятность вдвое

# Sampling
top-20 → softmax → multinomial
```

---

## 5. RecursiveTensorPotentialField — детальный дизайн

### 5.1 Мотивация
Плоский TPF [V×V×V] не различает контексты: bias одинаков для "кот" и "собака", если последний символ "а".  
Рекурсивная декомпозиция: hidden state трансформера → K субвекторов → каждый квантизуется к символу → bias от TPF для КАЖДОГО субвектора → взвешенная сумма.

### 5.2 Архитектура
```
x [128] ──→ decomp_proj [128 → K·128] ──→ reshape [K, 128] ──→ [v₁..v₆]
                │
                └── gate_net [128 → K → softmax] ──→ [g₁..g₆]

Каждый v_k:
  quantize → sym_id → P[sym_id] → bias_k
  gate-filter: g_k < 0.05 → отброс
  ||v_k - x|| < 0.01 → cycle → tensor product unfold
  иначе → recurse (max_depth=8)

Итог: bias = Σ_depth Σ_path (scale_path × P[sym_path])
```

### 5.3 Параметры
| Подкомпонент | Формула | Параметров |
|-------------|---------|-----------|
| base_tpf.P | [160,160,160] | 4,096,000 |
| decomp_proj.weight | [768, 128] | 98,304 |
| compose_proj.weight | [128, 768] | 98,304 |
| gate_net[0].weight + bias | [6, 128] + [6] | 768 + 6 = 774 |
| depth_scale | scalar | 1 |
| **Итого** | | **4,293,383** |

### 5.4 BFS Quantize
```python
for depth in range(1, max_depth + 1):
    N = frontier.shape[0]          # число путей на этом уровне
    syms = quantize(frontier)      # batched cdist: [N, 160] → [N]
    bias_all += P[syms].mean(valid) * scale
    
    subs, gates = decompose(frontier)     # [N, K, D], [N, K]
    subs_flat, gates_flat = flatten(N×K)
    keep = gates_flat > 0.05
    frontier = subs_flat[keep]            # новый frontier (меньше путей)
    frontier_scale = parent_scale × depth_scale × gate[keep]
```

### 5.5 Composition Loss
Обучает decomp/compose быть взаимно-обратными:
```
x → decomp → [v₁..v₆] → compose → x'
loss = ||x - x'||² - 0.01 × Σ g·log(g)
```
Gate entropy бонус: при равномерных гейтах H(g) ≈ 1.79, loss ≈ −0.0179.  
Наблюдаемое значение: `comp=-0.0178` — гейты равномерны, сеть изучает identity.

---

## 6. Метрики

| Модель | Параметры | Loss @ 50 | Acc @ 50 | VRAM |
|--------|-----------|-----------|----------|------|
| War & Peace (1.15M) | 1.15M | 2.34 | 32.7% | — |
| Full corpus (5.3M, flat TPF) | 5.3M | — | — | ~1.7 GB |
| Full corpus (5.49M, RecursiveTPF) | 5.49M | 1.24-1.89 | 49-64% | ~0.7 GB |

---

## 7. Отличия от LLM

| | LLM (GPT, LLaMA) | EVA |
|---|---|---|
| Символ | Индекс в таблице | Координата в ℝ¹²⁸ |
| Параметры | 1B–1.7T | **~5.5M** |
| Знания | В весах (неотделимы) | В топологии + RecursiveTPF |
| Bias | Нет (raw CE) | **RecursiveTPF (K=6, BFS) + WVF + SubHSM** |
| Bias-структура | Одноуровневая | **Рекурсивная, batched BFS quantize** |
| Память | Внутри модели | TrajectoryStore (внешняя) |
| Внимание | Multi-Head Self-Attention | **FractalConv2D + CoordBias + AdaptiveAttention** |
| Ретенция | Context window | **Рекурсивные потенциалы + траектории** |
| Интерпретация | Black box | **Траектория + рекурсивный bias по субвекторам** |
