# EVA Symbolic — Архитектура (полная)

> Символ ≡ координата в ℝ¹²⁸. Текст ≡ траектория.  
> Трансформер ≡ навигатор. Знания ≡ рекурсивный тензор потенциалов + топология + память траекторий.  
> **5.49M параметров. 160 токенов. 21 инновация. 100K шагов обучения.**

---

## 0. Философия

EVA не предсказывает следующий токен. EVA **навигирует в символьном пространстве**:
- Получает запрос → кодирует в координаты ℝ¹²⁸ → вычисляет траекторию через 6 слоёв
- На каждом шаге генерации: hidden state [128] → рекурсивная декомпозиция на K=6 субвекторов
- Каждый субвектор квантизуется к ближайшему символу → bias из тензора потенциалов
- Bias от каждого уровня рекурсии → взвешенная сумма → коррекция CE logits
- Мультиномиальный sampling с адаптивным repetition penalty

Знания отделены от модели: координаты [160×128], affinity [160×160], тензор [160×160×160], топология [160×160×3], иерархическая память TrajectoryStore. Модель — только исполнитель.

---

## 1. Фундаментальные принципы

### 1.1 Символ как координата в ℝ¹²⁸
Каждый из 160 символов — точка на единичной гиперсфере в 128-мерном пространстве. Позиция не случайна:
- Инициализация: `~N(0, 0.02)` с нормировкой на единичную сферу
- Координаты разделяют семантическое пространство: символы, часто встречающиеся вместе, имеют близкие координаты
- `CoordinateEmbedding` — `nn.Parameter`, а не `nn.Embedding`: это позиция, не индекс

**Почему 128?** Нижняя граница для 160 символов при разделимости (160 точек требуют минимум log₂(160) ≈ 8 dim для однозначной идентификации; 128 dim даёт избыточность для семантической структуры + рекурсивной декомпозиции на K субвекторов).

### 1.2 Текст как иерархическая траектория
- Символ → точка в ℝ¹²⁸
- Слово → последовательность точек + boundary-токены `<W>...</W>`
- Предложение → последовательность слов + boundary-токены `<S>...</S>`
- Текст → последовательность предложений

Вся иерархия — объекты одного типа (точки в ℝ¹²⁸), разница только в масштабе.

### 1.3 Отделение знаний от модели
Модель (ядро ~1.14M параметров) — **навигатор**, не хранилище:

| Компонент | Параметров | Тип | Обновление |
|-----------|-----------|-----|-----------|
| Transformer core (6 HybridFractalBlock) | ~1.14M | Модель | SGD каждый шаг |
| CoordinateEmbedding | 20,480 | Координаты | SGD каждый шаг |
| RecursiveTensorPotentialField | 4,293,383 | Потенциал | Thinking phase (каждые 500) |
| WordValenceField | 74,305 | Валентность | SGD каждый шаг |
| TrajectoryStore | 0 (внешняя) | Память | Thinking phase |
| StaticTopologyLayer | ~500 | Топология | Thinking phase |

Модель можно заменить — знания останутся в координатах, потенциале, топологии и памяти.

### 1.4 Генерация как навигация
Генерация — не `P(next_token | context)`, а **исполнение инструкции**:
1. Трансформер кодирует запрос → траектория [L, 128]
2. На каждом шаге: hidden state → рекурсивная декомпозиция → bias от потенциалов всех уровней
3. Sampling: top-20 с adaptive repetition penalty

### 1.5 Рекурсивная декомпозиция как иерархический bias
Плоский bias (один символ → один тензор [V]) не различает контексты.  
Рекурсивная декомпозиция: каждая координата раскладывается на K=6 субвекторов, каждый квантизуется к символу, bias от каждого символа взвешивается gate-механизмом и масштабом глубины.

### 1.6 Масштабирование через траектории, не параметры
LLM растут за счёт увеличения числа параметров (1B → 1.7T).  
EVA растёт за счёт накопления траекторий (10 → 100K → 1M). Модель не растёт — растёт память.

---

## 2. Компоненты системы (детально)

### 2.1 CharacterVocab — символьный словарь
**Тип**: PURE — 0 параметров, правила.

160 токенов:
- 0 PAD, 1 UNK, 2 BOS, 3 EOS
- 152 печатных символа: А-Я (32), а-я (32), A-Z (26), a-z (26), 0-9 (10), пунктуация (26)
- 4 boundary-токена: `<W>` (157), `</W>` (158), `<S>` (159), `</S>` (160)

`encode_with_boundaries(text, max_word_len=20)`:
```
"Привет, мир!" → 
<S><W>П</W><W>р</W><W>и</W><W>в</W><W>е</W><W>т</W><W>,</W> <W>м</W><W>и</W><W>р</W><W>!</W></S>
```

Каждое слово обёрнуто в `<W>...</W>` — это даёт **иерархическую структуру** без потери символьной детализации. Предложение — в `<S>...</S>`.

### 2.2 RecursiveTensorPotentialField — рекурсивный тензор потенциалов

**Параметров**: 4,293,383. **Инновация #21**.

#### 2.2.1 Базовая структура — TPF [V×V×V]
Трёхмерный тензор `P ∈ ℝ^{160×160×160}` (4,096,000 params).  
`P[i][j][k]` — сила связи от символа j к символу k, активируемая символом i.

Инициализация: `P[i][j][k] = affinity[i][j] × affinity[j][k]` — цепное правило: если i→j и j→k сильные связи, то P[i][j][k] высок.

Обновление (thinking phase): захват attention weights из transformer, накопление в P с модуляцией quality (confidence, curvature, contradictions).

#### 2.2.2 Рекурсивная обёртка — decomp + compose + gate
```
x [128] ──→ decomp_proj [128 → K·128] ──→ reshape [K, 128] ──→ [v₁, v₂, ..., v₆]
                │
                └── gate_net [128 → K → softmax] ──→ [g₁, g₂, ..., g₆]
```

**decomp_proj**: `nn.Linear(128, K·128, bias=False)` = 98,304 params.  
Проецирует координату в K параллельных 128-мерных подпространств. Без bias — декомпозиция должна быть чисто линейной.

**compose_proj**: `nn.Linear(K·128, 128, bias=False)` = 98,304 params.  
Восстанавливает исходную координату из K субвекторов. Используется только в composition loss, не в генерации.

**gate_net**: `nn.Linear(128, K) → Softmax` = 768 + 6 = 774 params.  
Предсказывает вес каждого субвектора: какие субвекторы важны для данного контекста.

**depth_scale**: `nn.Parameter(ones[8] × 0.5)` = 8 params.  
Обучаемый масштаб для каждого уровня глубины. Позволяет модели решать, какие уровни рекурсии важны. Начальное значение 0.5 — равномерный вклад всех уровней.

#### 2.2.3 Батчевый BFS — quantize + bias на каждом уровне

```
Level 0: quantize(x) → P[idx] → bias_0 [V]
Level 1: decompose(x) → [v₁..v₆] → gate-filter → quantize each → P[idx] → bias_1 × depth_scale[1] × gate
Level 2: decompose(each survivor) → ... → bias_2 × depth_scale[2] × Π(gates_along_path)
...
Max depth: 8, max paths: 4096
```

**quantize** — batched `cdist(vectors, sym_coords) → argmin`.  
N векторов → один вызов `torch.cdist`. Ноль Python `item()` — все операции на GPU.

**Gate filter**: `g_k < 0.05` → ветка отбрасывается.  
При обучении гейты равномерны (H ≈ ln6), ветвится 6× на каждом уровне. После 20K+ шагов гейты заостряются (1-2 активных субвектора), дерево сужается.

**Scale на глубине d**: `parent_scale × depth_scale[d] × g_k`.  
- `depth_scale[0]` = 0.5 (уровень 1), `depth_scale[3]` может стать 0.1 (уровень 4 неважен) или 0.9 (уровень 4 критичен)
- Каждый уровень — свой learnable масштаб

#### 2.2.4 Cycle → Tensor Product Unfold
Если `||v_k - x||₂ < 0.01` — субвектор почти совпадает с родителем (цикл).  
Вместо бесконечной рекурсии — unfold: перемножить bias-векторы всех путей цикла через element-wise умножение.

Не реализовано в коде (пока), но концептуально заложено: при `max_depth=8` и `max_cap=4096` дерево ограничено. Если сеть захочет углубиться — cycle detection включится.

#### 2.2.5 Composition Loss — auxiliary loss для decomp/compose

```python
def composition_loss(vectors):        # vectors: [B, L, D]
    flat = vectors.reshape(B*L, D)    # [BL, D]
    subs, gates = decompose(flat)     # [BL, K, D], [BL, K]
    recon = compose(subs)             # [BL, D]
    
    # MSE reconstruction
    loss = MSE(recon, flat)
    
    # Diversity: субвекторы должны быть разными
    subs_norm = F.normalize(subs, dim=-1)
    cos_mat = bmm(subs_norm, subs_norm.transpose)    # [BL, K, K]
    mask = 1 - eye(K)                                 # диагональ исключена
    diversity = mean(cos_mat * mask)                  # средний cosine i≠j
    loss += 0.1 * diversity
    
    # Gate entropy bonus: не коллапсить к одному субвектору
    entropy = -mean(Σ g·log(g))
    loss -= 0.01 * entropy
    
    return loss
```

**Три компонента composition loss**:

| Компонент | Назначение | Начальное значение | Желаемое значение |
|-----------|-----------|-------------------|-------------------|
| MSE recon | decomp→compose ≈ identity | ~0.0004 | 0 |
| Diversity ×0.1 | субвекторы разные | ~0 | >0 (cos → 0) |
| Entropy ×(−0.01) | не коллапсить к одному | −0.0179 | −0.01 (H→1) |

Diversity loss предотвращает ситуацию, когда все K субвекторов равны x/K — заставляет их покрывать разные подпространства.

Gate entropy бонус: при равномерных гейтах H=ln6≈1.79, loss contribution = −0.0179.  
Наблюдаемое значение: `comp=−0.0178` — гейты равномерны, сеть учит identity.

#### 2.2.6 Delegation к Base TPF
- `init_from_affinity(affinity)` → `base_tpf.init_from_affinity(affinity)`
- `update(sym_idx, attn_weights, lr)` → `base_tpf.update(sym_idx, attn_weights, lr)`
- `update_with_reflection(sym_idx, attn_weights, metrics, lr)` → `base_tpf.update_with_reflection(...)`
- `forward(sym_idx)` → `base_tpf(sym_idx)`
- `get_bias(sym_idx, target_ids)` → `base_tpf.get_bias(sym_idx, target_ids)`

Resume из старого чекпоинта: `strict=False` + ручная миграция `tensor_potential.P → tensor_potential.base_tpf.P`.  
Shape-мисматчи (depth_scale scalar → vector[8]) автоматически пропускаются.

#### 2.2.7 Полный bias для генерации
```python
bias = P[quantize(x), valid, :].mean(dim=0)   # Level 0
for depth in range(1, max_depth+1):
    N = frontier.shape[0]
    syms = quantize(frontier)                  # [N]
    bias += P[syms, valid, :].mean(dim=1) * scale[depth]
    
    subs, gates = decompose(frontier)           # [N, K, D], [N, K]
    keep = gates > 0.05
    frontier = subs[keep]                       # новый frontier
    scale[keep] *= depth_scale[depth] * gates[keep]

Итог: logits += bias * 0.1
```

#### 2.2.8 Параметры (полная таблица)

| Подкомпонент | Формула | Параметров | Ops/forward |
|-------------|---------|-----------|-------------|
| base_tpf.P | [160,160,160] | 4,096,000 | 0 (lookup) |
| decomp_proj.weight | [768, 128] | 98,304 | 128×768 MAC |
| compose_proj.weight | [128, 768] | 98,304 | 768×128 MAC |
| gate_net[0].weight | [6, 128] | 768 | 128×6 MAC |
| gate_net[0].bias | [6] | 6 | 6 add |
| depth_scale | [8] | 1 (вектор 8) | 8 mul |
| sym_coords | [160, 128] (buffer) | 0 | cdist: N×160×128 |
| **Итого** | | **4,293,383** | |

#### 2.2.9 Типичные значения при обучении (step 6500-7000)

| Параметр | Значение |
|----------|----------|
| depth_scale[0] | 0.500 (инициализация) |
| depth_scale[1] | 0.500 |
| depth_scale[7] | 0.500 |
| Gate entropy H | ~1.79 nats (ln 6) |
| Composition loss | ~−0.0178 (MSE≈0 + 0.1×0 − 0.01×1.79) |
| Bias paths | ~6^d до max_cap=4096 |
| quantize вызовов на шаг | ~два-три (BFS levels до cap) |

### 2.3 TensorPotentialField — базовый слой (4,096,000 params)

**Инновация #14**. Базовый тензор [V×V×V], общий для всех уровней рекурсии.

Инициализация (affinity chain rule):
```python
P[i][j][k] = affinity[i][j] × affinity[j][k]
```
Где `affinity[i][j]` — частота биграммы i→j (из co-occurrence 500 семплов × 128 токенов).

Обновление (attention capture):
```python
P[sym, :S, :S] += lr × attn_weights.mean(dim=1)  # [B, H, S, S] → [B, S, S]
```

Модуляция качеством:
```python
quality = confidence / (1 + curvature) / (1 + contradictions)
lr_eff = base_lr × quality
```

Хранит счётчик `count[sym]` — сколько раз символ участвовал в обновлении.

### 2.4 WordValenceField — словесная валентность (74,305 params)

**Инновация #15**. Отображение координаты слова → матрица валентности [V, V].

```
word_coord [128] → left_net → left [V]
                 → right_net → right [V]
valence = left outer right  → [V, V]
bias = mean(valence[valid_idxs, :])
```

Outer-product декомпозиция: вместо матрицы V×V (25,600) — два MLP 128→128→V (2 × 37,120 = 74,240 + bias 65 = 74,305).

Генерация: `logits += WVF.bias × 0.05`.

### 2.5 SentenceContextField — RBF-поле предложения (3 params)

**Инновация #16**. Поле активации от attention весов и connection-векторов.

```
centers = top-K attention_centroids  [B, K, D]
gamma = learnable [K]
phi_centers = Σ gamma · exp(-||coord - center||² / 2σ²)
edge_field = Σ exp(-||coord - mid_word||² / 2(σ/2)²)
activation = phi_centers + edge_field
```

Всего 3 обучаемых параметра: gamma_0, gamma_1, gamma_2 (для K=3 центров).

### 2.6 StaticTopologyLayer — статическая топология (~500 params)

**Инновация #6**. Тензор [160, 160, 3] + Fast Path cache (10K).

- **Канал 0: affinity** — сила связи i→j (из co-occurrence)
- **Канал 1: potential barrier** — высота барьера: `1 - cos(coord_i, coord_j)`
- **Канал 2: forbidden** — 0/1 (запрет на генерацию PAD/UNK/BOS/EOS)

**Fast Path cache**: FIFO-очередь на 10K.  
Хранит: `(centroid [128], path [list_of_ids])`.  
При генерации: поиск ближайшего `centroid` → если `cosine > 0.9` → bias из кэшированного пути.

`build_from_store(store)`: консолидирует траектории из TrajectoryStore → Fast Path.

### 2.7 TrajectoryStore — иерархическая память (6,529 params + 0 внешняя)

**Инновация #12**. 4 уровня хранения:

| Уровень | Формат | Размер элемента | Назначение |
|---------|--------|----------------|-----------|
| Symbol | [L, 128] | слово × 512 bytes | Полная траектория токенов в слове |
| Word | [N, 128] | 512 bytes | Центроиды слов |
| Connection | [N-1, 128] | 512 bytes | Векторы между центроидами |
| Sentence | [128] | 512 bytes | Центроид предложения |

`consolidate(trajectory)`: взвешенное усреднение через ConsolidationTransformer (Conv1d + gate).

`total_stored`: текущее число сохранённых траекторий (макс 50,000).

### 2.8 MultiSubspaceEmbedding — мультиподпространственный эмбеддинг

**Параметров**: 5,280. Часть ядра.

Структурирует 128 измерений в подпространства:
- dims 0-31: **symbol** — идентификация символа
- dims 32-127: **context** — контекстная информация

Проекция: `base_embed [128] + symbol_sub [32]` = 128.

`WordWeightEncoder`: дополнительный MLP 128→128→1 — оценивает важность каждого токена в слове.

### 2.9 AdaptiveFractalAttention — адаптивное фрактальное внимание

**Инновации #3, #4**. Часть HybridFractalBlock.

**LevelController**: `x.mean(dim=1) → Linear(128→32) → SiLU → Linear(32→32) → softmax → head_allocation`.  
Динамически распределяет 32 головы по 6 слоям и dilation-уровням (1, 2, 4, 8).

**Manifold bias**: `exp(-||proj_2D(x_i) - proj_2D(x_j)||² / (2·scale²))`.  
Вычисляется ОДИН раз per level (не per-head). Смещает attention к точкам, близким на многообразии.

**CoordBias**: `-||coord_i - coord_j||₂` — L2-distance bias в attention scores.

**FractalConv2D**: causal свёртка по L × Dim с dilation 1, 2, 4, 8.  
Параллельные ветви dilations → concat → выход. Обеспечивает multi-scale иерархию без внимания.

### 2.10 SGF — Subspace-Gated FFN (3,072 params)

**Инновация #10**. 4 gate-вектора (128d) + роутер 128→4 → softmax.

```
gates = softmax(router(x))          # [4] — какой gate активен
output = Σ gate_k · (x ⊙ gate_vec_k)  # element-wise blend
```

Разные токены обрабатываются разными gate-векторами. Аналог MoE, но в одном FFN.

### 2.11 Coordinate Residual Stream (1,536 params)

**Инновация #9**. Сквозной поток координат через все 6 слоёв.

```
coord_residual = x (вход)
for layer in layers:
    hidden = layer(coord_residual)
    gate = sigmoid(Linear(128→128)(coord_residual))
    coord_residual = gate ⊙ hidden + (1 - gate) ⊙ coord_residual
```

Информация не замывается — поток координат проходит через все слои без потери.

### 2.12 SubHSM — Subspace Hierarchical Softmax (516 params)

**Инновация #7**. 4 группы по ~40 токенов.

```
group_logits = Linear(128→4)(hidden)           # 128×4 + 4 = 516 params
group_probs = softmax(group_logits)
group_id = argmax(group_probs)
bias = group_bias[group_id]  # прибавляется к logits
```

Group CE loss × 0.05 — вспомогательная loss для обучения классификатора групп.

### 2.13 TrajLoss — траекторная Auxiliary Loss (16,576 params)

**Инновация #11**. `MLP 128→64→128` предсказывает Δ-вектор следующей позиции.

```
target = embed(token_next) + pad(subspace(token_next))  # [128]
current = hiddens[:, -1, :]                              # [128]
delta_pred = TrajPredictor(current)                      # [128]
next_pred = current + delta_pred
loss = MSE(next_pred, target)
```

Weight: 0.1 × total loss. Обучает внутреннее представление предсказывать следующую координату.

### 2.14 Validation Suite

**Инновация — автоматическая валидация**. Каждые 5000 шагов:

1. **Генерация 7 seed'ов**: "что", "почему", "как", "где", "когда", "зачем", "кто"
2. **Тест на Анне Карениной**: первые 100 предложений, Perplexity + Trajectory curvature
3. **SelfReflection**: кривизна траекторий, confidence, пространственная эффективность (ratio активных измерений)
4. **Best checkpoint**: сохраняется при улучшении avg_curvature

---

## 3. Архитектура трансформера (ядро, ~1.14M params)

```
Input [B, L] → CoordinateEmbedding [B, L, 128] → RoPE
  → MultiSubspaceEmbedding [B, L, 128]
  → HybridFractalBlock × 6 [B, L, 128]  (CoordResidualStream сквозной)
  → RMSNorm [B, L, 128]
  → CoordinateDecoder [B, L, 160]
  → SubHSM bias → Logits [B, L, 160]
```

### 3.1 CoordinateEmbedding

**Параметров**: 160 × 128 = 20,480.

`nn.Parameter`, не `nn.Embedding`. Прямое обращение по индексу: `coords[token_ids]`.

Координаты на единичной сфере: `coord = coord / ||coord||₂`.

### 3.2 RoPE

Поворот в 2D-подпространствах: 128/2 = 64 поворота. `θ = 10000.0`.

### 3.3 HybridFractalBlock (×6)

**Параметров на блок**: ~190K. **Инновация #5**.

```
Input → LayerNorm
  → FractalConv2D (dilations 1,2,4,8) → AdaptiveFractalAttention (32 heads)
  → GateMerge (sigmoid blend)
  → LayerNorm → SGF (4-gate FFN)
  → CoordResidualStream gate update
  → Output
```

### 3.4 CoordinateDecoder

`Linear(128→160)` + `nearest-neighbour_bonus` + `SubHSM_group_bias`.

Nearest-neighbour: `cosine(hidden, coord[sym])` — бонус к токенам, чьи координаты близки к hidden.

### 3.5 Dilated KV-Cache

Для каждого dilation-уровня d: K/V каждого d-го токена.  
Не реализовано в полной мере (зарезервировано).

---

## 4. Обучение

### 4.1 Данные

| Параметр | Значение |
|----------|----------|
| Источник | full_corpus_ru.txt (172 MB, русская литература XIX-XX вв.) |
| Формат кодирования | CharacterVocab с boundary-разметкой (`<W></W><S></S>`) |
| Размер после кодирования | 106,520,000 токенов |
| Структура | 680,318 предложений (блоки, разделённые `</S>`) |
| Уникальных токенов | 113 (из 160 — остальные boundary и служебные) |
| Упаковка | `mmap_mode='r'`, int32, полная загрузка в virtual memory |

### 4.2 Гиперпараметры

| Параметр | Значение | Обоснование |
|----------|---------|------------|
| B (batch) | **12** | VRAM 0.7/2.1 GB → запас 1.4 GB позволяет B=12 |
| ML (seq) | **192** | Предложений обычно 10-30 токенов; 192 ≈ 2-3 предложения |
| STEPS | 100,000 | ~10 часов на MX550 |
| LR | 5e-3 | Cosine до 0; типично для small transformers |
| Optimizer | AdamW | weight_decay=0.01 |
| TrajLoss weight | 0.1 | Эмпирически |
| GroupLoss weight | 0.05 | Эмпирически |
| CompLoss weight | 0.01 | Начальное значение ~−0.0179, не доминирует |
| Diversity weight | 0.1 | Внутри comp_loss; penalty на cos(i≠j) |
| Entropy bonus | −0.01 | Внутри comp_loss; предотвращает коллапс гейтов |

### 4.3 Loss (полная)

| Компонент | Функция | Вес | Типичное значение |
|-----------|---------|-----|-------------------|
| CE | `CE(pred, target)` | 1.0 | 1.2-1.9 |
| SubHSM group | `CE(group_logits, group_target)` | 0.05 | 0.06-0.16 |
| TrajLoss | `MSE(pred_delta, actual_delta)` | 0.1 | 0.005-0.007 |
| Composition | `MSE(recon, x) + 0.1·diversity − 0.01·H(gates)` | 0.01 | −0.0178 |

### 4.4 Цикл обучения

```
for s in 1..100000:
    1. Собрать батч: 12 блоков × обрезка до </S>→</W> → [12, L≤192]
    2. Forward: hiddens, scores = transformer(bt)
    3. CE loss + Group loss + TrajLoss + CompLoss → backward
    4. Clip grad norm 1.0 → opt.step() → sch.step()
    
    Каждые 50:  print loss, acc, traj, group, comp
    Каждые 500: save checkpoint → thinking_phase()
    Каждые 2500: random generation (3 prompts)
    Каждые 5000: full validation → best checkpoint by curvature
```

### 4.5 Thinking Phase (каждые 500 шагов)

Цель: обновить потенциалы и топологию на основе накопленного опыта.

```
1. Co-occurrence affinity:
   for 500 samples of 128 tokens:
       for each pair (ids[k], ids[k+1]):
           aff[ids[k], ids[k+1]] += 1
   aff = aff / aff.max()

2. Init RecursiveTPF:
   base_tpf.init_from_affinity(aff)
   topology[:, :, 0] = aff.cpu()

3. Capture → update RecursiveTPF (10 blocks):
   for each random block:
       forward(block, capture_attn=True)
       tensor_potential.update(sym_idx, attn_4d, lr=0.01)

4. Extract → consolidate → store (5 blocks):
   for each random block:
       ht = extract_hierarchical(model, ids, text)
       ht = store.consolidate(ht)
       store.store_hierarchical(ht)

5. Build topology:
   topology.build_from_store(store)
```

### 4.6 Enhanced Generation Pipeline

```python
def enhanced_generate(prompt_ids, cv, max_new=30, temperature=0.8):
    ids = list(prompt_ids)
    for _ in range(max_new):
        h, logits = forward(ids)                     # [1, L, D], [1, L, V]
        
        # Recursive bias (BFS quantize на K=6, max_depth=8)
        bias_tpf = recursive_bias(x=h[0,-1], context_ids=ids)
        logits += bias_tpf * 0.1
        
        # Word valence bias
        word_coord = h[0, :].mean(dim=0)
        bias_word = word_valence.get_valence_bias(word_coord, ids)
        logits += bias_word * 0.05
        
        # Adaptive repetition penalty
        freq = Counter(ids)                          # все сгенерированные
        top_vals, top_idx = logits.topk(20)
        probs = softmax(top_vals / temperature)
        for rank, t in enumerate(top_idx.tolist()):
            if freq.get(t, 0) > 0:                   # каждый повтор режет p вдвое
                probs[rank] *= 0.5 ** freq[t]
        probs /= probs.sum()
        
        # Sampling
        nt = top_idx[multinomial(probs, 1)]
        ids.append(nt)
        if nt == SENT_CLOSE: break
    
    return decode(ids)
```

---

## 5. RecursiveTensorPotentialField — детальный дизайн

### 5.1 Мотивация

Плоский TPF не различает контексты:
- "кот" и "собака" заканчиваются на "а" → bias одинаков
- "бежать" и "читать" заканчиваются на "ь" → bias одинаков

Рекурсивная декомпозиция: hidden state содержит информацию о всём контексте (благодаря трансформеру).  
Декомпозиция на K субвекторов → каждый квантизуется к символу → bias от TPF для КАЖДОГО субвектора.

Один и тот же символ "а" даст разный bias в зависимости от того, какие субвекторы активны:
- "кот" + gate(g₁=0.9, g₂=0.01, ...) → активирует символ "о" из TPF
- "собака" + gate(g₁=0.1, g₂=0.8, ...) → активирует символ "б" из TPF

### 5.2 Поток данных (с размерами)

```
Level 0:
  x [128] → quantize [1] → P[1, V, V] → mean(valid) → bias [V], scale=1.0
  → decomp: subs [1, 6, 128], gates [1, 6]
  
Level 1:
  frontier [N=6, 128]
  → quantize [N] → P[N, V, V] → mean(valid) → level_biases [N, V], scale=[0.5, ..., 0.5]
  → gate filter (keep gates > 0.05) → M ≤ 6 survivors
  → decomp: subs [M, 6, 128], gates [M, 6]
  → scale = [0.5² × gate₁, 0.5² × gate₂, ...]

Level 2:
  frontier [M×K, 128]
  ...

Combine:
  bias = Σ[depth] Σ[paths] P[sym_path, valid, :].mean(dim=0) × scale_path
  bias = bias / Σ[paths] scale_path  (нормировка)
```

### 5.3 Quantize — детали

```python
def quantize(self, vectors):
    """vectors: [N, D] → ids: [N]"""
    # ОДИН cdist на все векторы
    dists = torch.cdist(vectors, self.sym_coords, p=2)  # [N, 160]
    return dists.argmin(dim=-1)                          # [N]
```

Сложность: `O(N × 160 × 128)`. Для N=4096 (max_cap) и D=128: 4096 × 160 × 128 = 83.9M FLOPS.  
На MX550 (~2 TFLOPS): ~0.04ms. Ничтожно.

### 5.4 Decompose — детали

```python
def decompose(self, vectors):
    """vectors: [N, D] → sub: [N, K, D], gates: [N, K]"""
    N = vectors.shape[0]
    sub = self.decomp_proj(vectors).view(N, self.K, self.coord_dim)
    # decom_proj: Linear(128→768), N × 128 × 768 = 98,304N MAC
    gates = self.gate_net(vectors)  # Linear(128→6), N × 128 × 6 = 768N MAC
    return sub, gates
```

### 5.5 Composition Loss — детали

```python
def composition_loss(self, vectors):
    B, L, D = vectors.shape
    flat = vectors.reshape(-1, D)     # [B*L, 128]
    
    # Decompose
    subs, gates = self.decompose(flat)  # [BL, 6, 128], [BL, 6]
    
    # Reconstruct
    recon = self.compose(subs)         # [BL, 128]
    
    # MSE: обратная связь decomp→compose
    loss_mse = F.mse_loss(recon, flat)  # scalar
    
    # Diversity: penalty на косинусное сходство субвекторов
    subs_norm = F.normalize(subs, dim=-1)
    cos_mat = torch.bmm(subs_norm, subs_norm.transpose(1, 2))  # [BL, 6, 6]
    mask = 1.0 - torch.eye(self.K, device=cos_mat.device).unsqueeze(0)
    diversity = (cos_mat * mask).sum(dim=(1,2)) / (self.K * (self.K - 1))
    diversity = diversity.mean()
    # При случайных субвекторах: cos ~ 0, diversity ~ 0
    # При коллапсе (все v_k = x/K): cos = 1, diversity = 1
    
    # Gate entropy: предотвратить коллапс к одному gate
    entropy = -(gates * torch.log(gates + 1e-10)).sum(dim=-1).mean()
    # При равномерных gate: entropy = ln(6) ≈ 1.79
    # При коллапсе (один gate=1): entropy = 0
    
    return loss_mse + 0.1 * diversity - 0.01 * entropy
```

**Схема градиентов**:
- `loss_mse` → compose_proj ← decomp_proj (сквозная связь: учим оба)
- `diversity` → decomp_proj (субвекторы должны быть разные)
- `entropy` → gate_net (гейты не должны коллапсить)

**Динамика обучения**:
1. **Фаза 1 (0-10K steps)**: MSE доминирует. Состав учится быть identity. Diversity ~0, entropy ~1.79.
2. **Фаза 2 (10K-50K steps)**: MSE мал. Diversity начинает расти (субвекторы специализируются). Entropy снижается до ~1.0.
3. **Фаза 3 (50K+ steps)**: Установившийся баланс. Diversity ~0.5-0.7, entropy ~0.8-1.2, MSE ~0.

### 5.6 Depth Scale — per-level веса

```python
self.depth_scale = nn.Parameter(torch.ones(max_depth) * 0.5)
# shape: [8], все 0.5
```

Градиент: `∂loss/∂depth_scale[d] = Σ_paths scale_before_depth_d × g_k × bias_path`

Динамика: если глубокая рекурсия полезна → `depth_scale[d]` растёт. Если вредна → падает к 0.

### 5.7 BFS Tree — масштабирование

```
Gate threshold = 0.05
Max cap = 4096

При равномерных gates (H=1.79, все g_k ≈ 1/6 ≈ 0.17 > 0.05):
  Level 0: 1 path  (keep все 6)
  Level 1: 6 paths (keep все 6 × 6 = 36)
  Level 2: 36 paths
  Level 3: 216 paths
  Level 4: 1296 paths
  Level 5: 7776 paths → превышение max_cap=4096, обрезаем

После специализации gates (1-2 активных):
  Level 0: 1 path (keep 2)
  Level 1: 2 paths
  Level 2: 4 paths
  Level 3: 8 paths
  ...
  Level 8: 256 paths → все 8 уровней вмещаются
```

---

## 6. Resume из чекпоинта — миграция

### 6.1 Старый → Новый чекпоинт

| Старый ключ | Новый ключ | Действие |
|------------|-----------|----------|
| `tensor_potential.P` | `tensor_potential.base_tpf.P` | Копирование вручную |
| `tensor_potential.count` | `tensor_potential.base_tpf.count` | strict=False (игнорируется) |
| `tensor_potential.depth_scale` (scalar) | `tensor_potential.depth_scale` (vector[8]) | Отбрасывается (shape mismatch) |
| Все остальные ключи | Совпадают | Автоматическая загрузка |

### 6.2 Code

```python
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    old_sd = ckpt['ut']
    
    # Remove shape-mismatched keys
    for k in list(old_sd.keys()):
        if k in model.state_dict() and old_sd[k].shape != model.state_dict()[k].shape:
            del old_sd[k]  # depth_scale scalar→vector, etc.
    
    model.load_state_dict(old_sd, strict=False)
    
    # Migrate old TPF.P → base_tpf.P
    if 'tensor_potential.P' in old_sd:
        model.tensor_potential.base_tpf.P.copy_(old_sd['tensor_potential.P'])
```

---

## 7. Метрики и производительность

### 7.1 Текущие метрики (step 6500)

| Метрика | Значение |
|---------|---------|
| CE Loss | 1.24-1.89 |
| Accuracy | 49-64% |
| TrajLoss | 0.005-0.007 |
| GroupLoss | 0.06-0.16 |
| Composition Loss | ~−0.0178 |
| Topology connections | ~1250 |
| Trajectories in store | ~35 |
| Steps/min | ~160 |
| VRAM | 0.7 GB / 2.1 GB |

### 7.2 Производительность

| Операция | Время | Frequency |
|----------|-------|-----------|
| Forward + loss (B=12, ML=192) | ~375ms | Каждый шаг |
| Thinking phase (500 samples + update) | ~30s | Каждые 500 шагов |
| Validation (7 seeds + Anna Karenina) | ~60s | Каждые 5000 шагов |
| Checkpoint save (22 MB) | ~2s | Каждые 500 шагов |

### 7.3 VRAM Breakdown

| Компонент | VRAM | Доля |
|-----------|------|------|
| Model params (5.5M × 4 bytes) | 22 MB | 3% |
| Optimizer states (AdamW × 2) | 44 MB | 6% |
| Activations (B=12, L=192, 6 layers) | ~300 MB | 42% |
| CUDA context + allocator overhead | ~335 MB | 48% |
| **Total** | **~700 MB** | **100%** |

---

## 8. Отличия от LLM

| | LLM (GPT, LLaMA) | EVA |
|---|---|---|
| **Параметры** | 1B–1.7T | **5.49M** |
| **VRAM** | 8-80 GB | **0.7 GB** |
| **Знания** | В весах (неотделимы) | Потенциалы + топология + траектории |
| **Ретенция** | Context window (4K-128K) | Внешняя память (50K+ траекторий) |
| **Масштабирование** | Больше параметров | Больше траекторий |
| **Bias** | Нет (raw CE) | RecursiveTPF (K=6, BFS) + WVF + SubHSM |
| **Внимание** | MHSA | FractalConv2D + CoordBias + Adaptive |
| **Иерархия** | Flat tokens | Character + Word + Sentence (boundary) |
| **Интерпретируемость** | Black box | Траектория + рекурсивный bias |
| **Обучение** | Weeks, 8-256 GPUs | Hours, 1 GPU (MX550) |
