# EVA — Полный справочник параметров

## 1. Архитектура модели

### 1.1. Размерности

| Параметр | Значение | Смысл |
|----------|----------|-------|
| `d_model` / `D_MODEL` | 384 | Размер скрытого состояния (и координатного пространства). Каждый токен представлен вектором 384 float. Выбор: 6 масштабов × минимум 64 оси = 384. |
| `n_heads` / `N_HEADS` | 24 | Количество голов внимания. Каждая голова смотрит на свою проекцию Q,K,V размерности 16. |
| `head_dim` | 16 | Размерность одной головы. D_MODEL / N_HEADS = 384 / 24 = 16. |
| `n_groups` / `N_GROUPS` | 6 | Количество групп масштабов. 6 групп × 4 головы = 24. |
| `heads_per_group` / `HEADS_PER_GROUP` | 4 | Голов в одной группе. Каждая группа — свой масштаб: char, morph, word, phrase, sentence, discourse. |
| `group_dim` / `GROUP_DIM` | 64 | Выходная размерность группы. 4 × 16 = 64. Каждая группа имеет отдельный W_O: 64 → 384. |
| `n_layers` / `N_LAYERS` | 12 | Количество трансформер-блоков. Каждый слой: GroupedScaleAttention + SwiGLU FFN. |
| `d_ff` / `D_FF` | 512 | Размер скрытого слоя FFN (SwiGLU). 384 → 512 → 384. |
| `max_seq_len` / `MAX_SEQ_LEN` | 2048 | Максимальная длина последовательности для RoPE. В тренировке используется 64 (из-за VRAM). При генерации — до 2048. |
| `vocab_size` / `VOCAB_SIZE` | 4101 | BPE = 4096 токенов + 5 служебных (GAP, WO, WC, SO, SC). |
| `theta_min` / `THETA_MIN` | 500 | Минимальная частота MultiScaleRoPE (dim 0-1). Период = 2π × 500 ≈ 3140 позиций. |
| `theta_max` / `THETA_MAX` | 200000 | Максимальная частота MultiScaleRoPE (dim 382-383). Период = 2π × 200000 ≈ 1.26M позиций. |

### 1.2. Embedding

| Параметр | Значение | Смысл |
|----------|----------|-------|
| `BPEEmbedding.weight` | [4101, 384] | Таблица координат токенов. Каждый из 4101 токенов имеет точку в 384-мерном координатном пространстве. Инициализация: Normal(0, 0.02). |
| `BPEDecoder.linear` | [384, 4101] | Обратная проекция скрытого состояния → логиты. |
| `temperature` | [1] (learnable) | Температура декодера. Clamp [0.1, 10.0]. |

### 1.3. MultiScaleRoPE

| Параметр | Значение | Смысл |
|----------|----------|-------|
| `freqs` | [192] | ω_k = 1/θ_k для k = 0..191. θ_k = 500 × (200000/500)^(k/192). Не обучается. |
| `cos` / `sin` | [2048, 192] | Предвычисленные cos/sin для всех позиций. |

### 1.4. GroupedScaleAttention (на слой)

| Параметр | Значение | Смысл |
|----------|----------|-------|
| `to_q` | [384, 384] | Проекция Q. Единая для всех голов. |
| `to_k` | [384, 384] | Проекция K. Единая для всех голов. |
| `to_v` | [384, 384] | Проекция V. Единая для всех голов. |
| `w_os` | 6 × [64, 384] | 6 отдельных проекций W_O, по одной на группу масштаба. |
| `gate` | [6] (learnable) | Soft-веса групп. softmax → α_l ∈ ℝ⁶. Инициализация по слою. |
| `scale` | 1/√16 = 0.25 | Масштаб attention scores. |

#### Инициализация gate по слоям

```
Слой 0:  [0.70, 0.20, 0.10, 0.00, 0.00, 0.00]  # char-heavy
Слой 1:  [0.60, 0.25, 0.15, 0.00, 0.00, 0.00]
Слой 2:  [0.40, 0.30, 0.20, 0.10, 0.00, 0.00]
Слой 3:  [0.30, 0.30, 0.25, 0.15, 0.00, 0.00]
Слой 4:  [0.10, 0.30, 0.30, 0.20, 0.10, 0.00]  # word/phrase-heavy
Слой 5:  [0.05, 0.25, 0.30, 0.25, 0.15, 0.00]
Слой 6:  [0.00, 0.15, 0.25, 0.30, 0.25, 0.05]
Слой 7:  [0.00, 0.10, 0.20, 0.30, 0.30, 0.10]
Слой 8:  [0.00, 0.00, 0.10, 0.25, 0.40, 0.25]  # sentence/discourse-heavy
Слой 9:  [0.00, 0.00, 0.05, 0.20, 0.40, 0.35]
Слой 10: [0.00, 0.00, 0.00, 0.10, 0.35, 0.55]
Слой 11: [0.00, 0.00, 0.00, 0.05, 0.30, 0.65]
```

### 1.5. Residual Streams

| Параметр | Значение | Смысл |
|----------|----------|-------|
| `residual1` | [B, L, 384] | Поток char/morph — локальная структура. |
| `residual2` | [B, L, 384] | Поток word/phrase — синтаксис. |
| `residual3` | [B, L, 384] | Поток sentence/discourse — глобальная структура. |

Коэффициенты обновления:

```
Слои 0-3:  res1 += attn_out × 0.30, res2 += × 0.10, res3 += × 0.05
Слои 4-7:  res1 += attn_out × 0.10, res2 += × 0.30, res3 += × 0.15
Слои 8-11: res1 += attn_out × 0.05, res2 += × 0.15, res3 += × 0.30
```

---

## 2. AttractorField

### 2.1. Параметры

| Параметр | Значение | Смысл |
|----------|----------|-------|
| `coord_dim` | 384 | Размерность координат аттракторов = d_model. |
| `sigma` | 0.5 | Ширина RBF-ядра потенциала. exp(-||z-μ||² / 2σ²). |
| `eta` | 0.7 | Баланс в nxt_direction: η × притяжение к центру + (1-η) × рефрактер. |
| `max_attractors` | 10000 | Максимальное количество аттракторов (на каждый токен vocab). |
| `lr_center` | 0.01 | Learning rate для Hebbian update центра аттрактора. |
| `lr_refract` | 0.05 | Learning rate для рефрактерного вектора. |
| `creation_threshold` | 0.1 | Если расстояние до ближайшего аттрактора > этого — создать новый. |
| `decay` | 0.999 | Коэффициент затухания счётчика за каждый forward. |

### 2.2. Буферы

| Буфер | Размер | Смысл |
|-------|--------|-------|
| `centers` | [10000, 384] | Координаты центров аттракторов. |
| `counts` | [10000] | Счётчики посещений аттрактора (вес w_a). |
| `refractory` | [10000, 384] | Рефрактерные векторы (типичное направление выхода r_a). |
| `valid_mask` | [10000] | Boolean: какие аттракторы ещё живы (не удалены). |

### 2.3. Механизмы

**Hebbian update** (каждые UPDATE_ATTRACTORS_EVERY=10 шагов):
1. Для скрытого состояния h_t найти ближайший центр μ*
2. Если `||h_t - μ*|| > creation_threshold=0.1` → создать новый аттрактор
3. Иначе: w_* += 1, μ_* += (lr_center / w_*) · (h_t - μ_*)
4. Обновить r_*: r += lr_refract · ((h_{t+1} - h_t) - r)
5. Глобальное затухание: все w *= decay

**Потенциал**: P(z) = Σ_a w_a · exp(-||z - μ_a||² / 2σ²)

**Градиент**: ∇P(z) = -Σ_a (w_a/σ²) · exp(-||z - μ_a||²/2σ²) · (z - μ_a)

**nxt_direction**: nxt(z) = η · (μ* - z)/||μ* - z|| + (1-η) · r*/||r*||

---

## 2.4. HierarchicalAdditiveField (дополнение к AttractorField)

HAF — надстройка над AttractorField: то же Hebbian-хранение, но с иерархическим
аддитивным разложением каждого вектора на K суб-векторов.

Ключевое отличие от плоского AttractorField:
- AttractorField: z → nearest attractor (1 точка)
- HAF: z → decompose → [v₀,...,v_K] → каждый vₖ → nearest attractor (K точек)

Параметры:

| Параметр | Значение | Смысл |
|----------|----------|-------|
| `coord_dim` | 384 | Размерность = d_model. |
| `max_arity` | 8 | Максимум компонент разложения. |
| `max_depth` | 5 | Глубина рекурсивной иерархии. |
| `creation_threshold` | 0.1 | Порог создания аттрактора (наследуется от AttractorField). |

HAF добавляет ~298K обучаемых параметров (1.4% от модели) и использует
внутренний AttractorField(10000, 384) для хранения.

Архитектура последовательной декомпозиции:
```
r₀ = z
for k in 0..max_arity:
    stopₖ = sigmoid(W·rₖ + b)    → STOP при >0.5
    vₖ = MLP(rₖ + posₖ) + rₖ     → skip-connection (гарантия реконструкции)
    rₖ₊₁ = rₖ - vₖ                → обновление residual
return [v₀, v₁, ..., v_K]        → z ≈ Σvₖ (||r_K|| < 0.001)
```

---

## 3. Heads (головы)

### 3.1. TrajectoryBoundaryPredictor

| Параметр | Вход/Выход | Смысл |
|----------|-----------|-------|
| `mlp` | 384 → 256 → 1152 (3 × 384) | h → [end_coord, nxt_coord, conn_vector], каждый ℝ³⁸⁴. |

Компоненты выхода:
- `end` — предсказание координаты конца слова
- `nxt` — предсказание дельты до следующего токена: Δh = h_{t+1} - h_t
- `conn` — вектор связи между концом слова и началом следующего

### 3.2. BoundaryDetectionHead

| Параметр | Вход/Выход | Смысл |
|----------|-----------|-------|
| `mlp` | 384 → 64 → 3 | h → [word_start, word_inside, word_end] logits. |

Обучается на размеченном корпусе (60.2M токенов с метками 0/1/2).

### 3.3. BoundaryValidator

| Параметр | Вход/Выход | Смысл |
|----------|-----------|-------|
| `mlp` | 768 → 64 → 2 (softmax) | h + z_current → [word_boundary_prob, sentence_boundary_prob]. |

### 3.4. ConceptHead

| Параметр | Вход/Выход | Смысл |
|----------|-----------|-------|
| `mlp` | 384 → 64 → 32 → 1 (sigmoid) | h → concept_probability [0,1]. Важность слова как концепта. |

### 3.5. ContradictionHead

| Параметр | Вход/Выход | Смысл |
|----------|-----------|-------|
| `mlp` | 384 → 64 → 32 → 1 (sigmoid) | h → contradiction_probability [0,1]. Уровень противоречия в контексте. |

### 3.6. UncertaintyHead

| Параметр | Вход/Выход | Смысл |
|----------|-----------|-------|
| `mlp` | 384 → 64 → 384 (exp) | h → per-dim variance [384]. Предсказывает intrinsic uncertainty предсказания следующего токена. |

### 3.7. MetaWeighter

| Параметр | Вход/Выход | Смысл |
|----------|-----------|-------|
| `proj` | 384 → 64 | Проекция контекста. |
| `weight_net` | 64 → 3 | [w_know, w_conc, w_contr] — softmax. Веса трёх источников генерации. |
| `temperature` | [1] (learnable) | Температура warmup: 0.1 → 1.0 за 1000 шагов. |
| `_bias` | [3] | Bias [1.0, 0.0, 0.0] — предпочтение knowledge. |

### 3.8. ResidualHead

| Параметр | Вход/Выход | Смысл |
|----------|-----------|-------|
| `proj` | 1152 → 384 | [h_t, z_{t-1}, z_t] → 384. |
| `res_mlp` | 384 → 64 → 384 | Предсказание delta_z = z_t - z_{t-1}. |

### 3.9. WordValenceField

| Параметр | Вход/Выход | Смысл |
|----------|-----------|-------|
| `left_net` / `right_net` | 384 → 128 → 4101 | Два MLP: outer product → матрица валентности [4101, 4101]. |
| `valence_scale` | [1] | Скалярный множитель. |

### 3.10. HierarchicalAdditiveField

| Параметр | Вход/Выход | Смысл |
|----------|-----------|-------|
| `slot_net` | 384 → 384 → 384 | MLP (SiLU) с skip-connection: vₖ = MLP(rₖ) + rₖ. **295K params**. |
| `stop_head` | 384 → 1 | Sigmoid: вероятность остановки декомпозиции. **385 params**. |
| `slot_pos` | [8, 384] | Learnable positional bias — каждый слот имеет своё "направление". **3K params**. |
| `depth_scale` | [5] | Вес каждого уровня глубины. **5 params**. |
| `gs_temp` | [1] | Gumbel-Sigmoid temperature (learnable). **1 param**. |
| `attractors` | AttractorField(10000, 384) | Внутреннее хранилище аттракторов для компонентов. **0 trainable**. |

Общая архитектура:
```
decompose(z):
    r = z
    for k in 0..max_arity:
        stop = sigmoid(stop_head(r))
        if stop > 0.5 and k > 0: break
        v = slot_net(r + slot_pos[k]) + r   # skip-connection
        r = r - v
        parts.append(v)
    return parts   # z = Σ(parts) + r_K

store_hierarchical(z, depth=2):
    hebbian_update(z)
    for p in decompose(z):
        hebbian_update(p)
        if depth > 0: store_hierarchical(p, depth-1)
```

### 3.11. WordWeightEncoder (в subspace_coords.py)

---

## 4. Потери (Loss)

### 4.1. Гиперпараметры Phase 2

| Параметр | Значение | Смысл |
|----------|----------|-------|
| `W_CE` | 1.0 | Вес cross-entropy loss. Угадать следующий токен среди 4101. |
| `W_NXT` | 0.05 | Вес trajectory loss. MSE между предсказанным Δh и реальным. |
| `W_BOUNDARY` | 0.1 | Вес boundary loss. CE на метки границ (0/1/2). |
| `W_ALIGN` | 0.05 | Вес L_align. MSE между char_inside и word_importance. |
| `W_ATTRACTOR` | 0.01 | Вес attractor consistency loss + diversity loss. |
| `ATTRACTOR_WARMUP` | 1000 | С какого шага включать attractor loss. |
| `W_HAF` | 0.001 | Вес HAF multi-path loss (иерархическое разложение). |
| `HAF_WARMUP` | 1000 | С какого шага включать HAF loss. |

### 4.2. CE loss

```
mask = targets ∉ {0,1,2,3,4096,4099,4100}
L_CE = CE(logits[mask], targets[mask])
```

Special-токены маскируются: модель не учится их предсказывать.

### 4.3. NXT loss

```
Δh = h[:, 1:] - h[:, :-1]            # реальная дельта
L_NXT = MSE(nxt[:, :-1], Δh)          # предсказание траектории
```

### 4.4. Boundary loss

```
L_BOUNDARY = CE(boundary_logits[y>=0], y_labels[y>=0])
```

Обучает BoundaryDetectionHead на labelled корпусе.

### 4.5. L_align loss

```
char_inside = boundary_probs[..., 1]         # вероятность "inside"
word_pooled = boundary_end или mean(h)        # слово-центроиды
word_avg = mean(word_pooled, dim=-1).sigmoid() # [B, L]
L_ALIGN = MSE(char_inside, word_avg)
```

### 4.6. Attractor consistency loss

```
z_flat = h.reshape(-1, 384)
dists = cdist(z_flat, centers)
nearest = centers[dists.argmin(dim=-1)]
L_AC = MSE(z_flat, nearest.detach())
```

Gradient останавливается на nearest — только h притягивается к центру.

### 4.7. Attractor diversity loss

```
c_norm = normalize(centers, dim=-1)
cos_sim = c_norm @ c_norm.T
mask = 1 - eye(N)
L_DV = mean((cos_sim * mask)²)
```

### 4.8. HAF (HierarchicalAdditiveField) loss

HAF loss — multi-path consistency для иерархического аддитивного разложения.

На каждом шаге (после HAF_WARMUP=1000):
```
z_pooled = mean(h)             # [D] — глобальное скрытое состояние

# Два случайных пути декомпозиции (разный dropout)
parts_a = decompose(z, dropout=0.1)
parts_b = decompose(z, dropout=0.5)

L_recon = ||z - Σparts_a||² + ||z - Σparts_b||²     # реконструкция
L_cross = ||Σparts_a - Σparts_b||²                    # согласованность
L_sparsity = (K_a + K_b) / max_arity                  # мало компонент
L_diversity = cos(mean(parts_a), mean(parts_b))       # разные паттерны

L_HAF = W_recon · L_recon + W_cross · L_cross + W_sparsity · L_sparsity - W_div · L_div
```

Градиенты идут через HAF (slot_net, stop_head, slot_pos) в скрытые состояния.

---

## 5. Оптимизация

### 5.1. Оптимизатор

| Параметр | Значение | Смысл |
|----------|----------|-------|
| `optimizer` | AdamW | Adam с weight decay. |
| `lr` / `LR` | 3e-4 | Начальная скорость обучения. |
| `weight_decay` | 0.01 | L2-регуляризация на веса. |
| `gradient_clip` | 1.0 | Max norm градиента (clip_grad_norm_). |

### 5.2. Scheduler

| Параметр | Значение | Смысл |
|----------|----------|-------|
| `N_STEPS` | 200000 | Всего шагов. |
| `WARMUP` | 4000 | Число шагов linear warmup. |
| `start_factor` | 1e-4 | Фактор LR в начале warmup. LR_start = 1e-4 × 3e-4 = 3e-8. |
| `T_max` | 196000 | Период CosineAnnealing = N_STEPS - WARMUP. |

Динамика LR:
1. Шаги 0-4000: LR = 3e-8 → 3e-4 (линейно)
2. Шаги 4000-200000: LR = 3e-4 → 0 (косинус)

### 5.3. UPDATE_ATTRACTORS_EVERY

| Параметр | Значение | Смысл |
|----------|----------|-------|
| `UPDATE_ATTRACTORS_EVERY` | 10 | Hebbian update каждые 10 шагов (экономия compute). |

### 5.4. SAVE_EVERY / LOG_EVERY

| Параметр | Значение | Смысл |
|----------|----------|-------|
| `SAVE_EVERY` | 20000 | Чекпоинт каждые 20000 шагов. 10 чекпоинтов за Phase 2. |
| `LOG_EVERY` | 100 | Вывод метрик каждые 100 шагов. |

---

## 6. Данные

### 6.1. Размеры

| Файл | Размер | Токенов | Смысл |
|------|--------|---------|-------|
| `full_corpus_ru.txt` | 165 MB | — | Исходный русский текст. |
| `full_corpus_bpe.npy` | 224 MB | 29.4M | BPE-кодированный корпус (ID 4-4095). |
| `full_corpus_bpe_boundary.npy` | 459 MB | 60.2M | BPE + WO/WC маркеры: 29.4M + 12M пар × 2 ≈ 60.2M. |
| `full_corpus_bpe_labels.npy` | 57 MB | 60.2M | Метки границ: 0 (start), 1 (inside), 2 (end). |
| `bpe_tokenizer.json` | 0.3 MB | 4096 | HuggingFace BPE tokenizer. |

### 6.2. BPE-токены

| ID | Тип | Смысл |
|----|-----|-------|
| 0 | PAD | Padding. Маскируется из CE loss. |
| 1 | UNK | Unknown — токены, не попавшие в BPE. |
| 2 | BOS | Beginning of sequence. |
| 3 | EOS | End of sequence. |
| 4-4095 | BPE | Реальные subword-токены (3890 уникальных в корпусе). |
| 4096 | GAP | Filler для выравнивания. |
| 4097 | WO | Word Open — маркер начала слова (в boundary корпусе). |
| 4098 | WC | Word Close — маркер конца слова (в boundary корпусе). |
| 4099 | SO | Sentence Open — маркер начала предложения. |
| 4100 | SC | Sentence Close — маркер конца предложения. |

### 6.3. SPECIAL_IDS (маскируются из CE)

```
{0, 1, 2, 3, 4096, 4099, 4100}
PAD, UNK, BOS, EOS, GAP, SO, SC
```

WO (4097) и WC (4098) НЕ маскируются — модель должна учиться их предсказывать.

---

## 7. Конфигурация тренировки

### 7.1. Batch-параметры

| Параметр | MX550 | T4 Colab | Смысл |
|----------|-------|----------|-------|
| `B` (batch_size) | 8 | 64-96 | Размер батча. T4 16GB позволяет B=64-96. |
| `L` (seq_len) | 64 | 128-256 | Длина последовательности. T4: 128-256. |
| `B × L` | 512 | 8192-24576 | Токенов за шаг. |

### 7.2. Время

| Метрика | MX550 | T4 |
|---------|-------|-----|
| Steps/s | ~3 | ~20-30 |
| 200K steps | ~18 часов | ~2-3 часа |
| 20K steps | ~2 часа | ~15 минут |
| 1K steps | ~5 минут | ~40 секунд |

### 7.3. VRAM

| Компонент | Память |
|-----------|--------|
| Параметры (20.5M × 4 bytes) | 78 MB |
| Аттракторы (10000 × 384 × 4) | 15 MB |
| Активации (B=8, L=64) | ~200 MB |
| Активации (B=64, L=128) | ~6 GB |
| Градиенты | ~78 MB |
| Optimizer states | ~234 MB |
| Всего (B=8, L=64) | ~0.7 GB |
| Всего (B=64, L=128) | ~12 GB |

---

## 8. Генерация

### 8.1. Параметры generate_text

| Параметр | Значение | Смысл |
|----------|----------|-------|
| `temperature` | 0.8 | Температура softmax. >1 = более случайно, <1 = более детерминировано. |
| `max_new` | 128 | Макс. число новых токенов для генерации. |
| `use_attractors` | False | Использовать AttractorField.nxt_direction() вместо boundary_predictor. |
| `use_haf` | False | Использовать HAF.nxt_direction() — иерархическое направление через декомпозицию. Приоритет выше use_attractors. |
| `top_k` | 20 | Top-K sampling: из 20 самых вероятных. |
| `repetition_penalty` | count(t) × 0.5 | Штраф: каждый уже сгенерированный токен уменьшает свой logit. |

### 8.2. Три источника логитов

```
logits_know = decoder(z_pred)              # знание (из весов)
logits_conc = -||z_pred - E|| × (1+concept) # концепт (координатный)
logits_contr = -||z_pred - E|| × (1-contra×0.5) # контраргумент (координатный)
final = (w_know×know + w_conc×conc + w_contr×contr) / temperature
```

---

## 9. Трёхфазное обучение

### 9.1. Параметры фаз

| Параметр | Phase 1 | Phase 2 | Phase 3 |
|----------|---------|---------|---------|
| Данные | Char (155 токенов) | BPE (3890+5 токенов) | BPE + attractor |
| Loss | CE + nxt | CE + nxt + bd + align + attractor | CE + nxt + bd + align + attractor + KCA |
| Шагов | 60000 | 200000 | 200000+ |
| B | 8 | 8 (64 на T4) | 8 |
| Attractors | нет | Hebbian (каждые 10 шагов) | Hebbian + loss |
| Генерация | static | static | gradient ascent |

### 9.2. Phase 1 (завершена)

```
60K шагов, char-level
CE: 8.5 → 1.3 (плато)
acc: 0% → 58%
Вывод: 155 токенов слишком мало для meaningful обучения
```

### 9.3. Phase 2 (текущая)

```
200K шагов, BPE-level
CE: 8.6 → целевой ~1.0-1.5
b_acc: 35% → 92%+
attractors: динамический пул 200-4000
```

### 9.4. Phase 3 (план)

- Добавить KCA-цикл (итеративная коррекция латентного кода)
- AttractorField как полноценный компонент loss
- Learned prototype vectors

### 9.5. Phase 4 (план)

- Генерация через градиентный подъём ∇P(z)
- Langevin dynamics: z_{t+1} = z_t + η·∇P(z_t) + noise
- Замена top-k sampling на физическую динамику в поле аттракторов

---

## 10. Мониторинг (лог-строка)

```
[PHASE2 2100/200000] ce=2.99 nxt=0.60 bc=0.33
align=0.003 ac=0.002 dv=0.155 hf=0.000 hk=0 hr=0.000
acc=0.457 b_acc=0.859 att=3334 haf_att=5 lr=1.58e-04 | 974s (16.2min)
```

С HAF (после warmup):
```
[PHASE2 10500/200000] ce=2.42 nxt=0.34 bc=0.18
align=0.000 ac=0.005 dv=0.148 hf=0.003 hk=3 hr=0.001
acc=0.486 b_acc=0.918 att=2441 haf_att=85 lr=2.99e-04 | 4533s (75.5min)
```

| Метка | Полное имя | Диапазон | Что означает |
|-------|-----------|----------|--------------|
| ce | Cross-entropy loss | 0-∞ | Основная потеря. 8 = случайно, 1-2 = хорошее качество. |
| nxt | Trajectory MSE | 0-∞ | MSE дельты h. 0.5-0.6 = учится, <0.1 = отлично. |
| bc | Boundary CE | 0-∞ | CE границ слов. 1.1 = случайно, <0.1 = отлично. |
| align | L_align MSE | 0-∞ | Согласование уровней. <0.01 = согласовано. |
| ac | Attractor consistency | 0-∞ | MSE h→nearest center. 0 = все точки в центре (идеал). |
| dv | Attractor diversity | 0-1 | Средний квадрат косинуса между центрами. 0 = ортогональны. |
| hf | HAF loss | 0-∞ | Multi-path loss HAF. ~0.001-0.01 после warmup. |
| hk | HAF K | 0-8 | Число компонент декомпозиции. 2-4 = типично. |
| hr | HAF residual | 0-∞ | Норма остатка после decompose. <0.01 = хорошо. |
| acc | Token accuracy | 0-1 | Доля правильно угаданных токенов. 0.5 = 50%. |
| b_acc | Boundary accuracy | 0-1 | Доля правильно размеченных границ. >0.9 = отлично. |
| att | Attractor count | 0-10000 | Текущее число живых аттракторов (AttractorField). |
| haf_att | HAF attractors | 0-10000 | Текущее число аттракторов в HAF. |
| lr | Learning rate | 1e-7 — 3e-4 | Текущая скорость обучения. |
| steps/s | Шагов в секунду | 0-∞ | Производительность. ~3 на MX550, ~25 на T4. |

---

## 11. Параметры чекпоинта

```python
torch.save({
    'step': step,                    # номер шага
    'model_state': model.state_dict(),  # все веса + буферы (аттракторы входят)
    'optimizer_state': optimizer.state_dict(),  # состояние AdamW
}, path)
```

Размер чекпоинта: ~156 MB (20.5M params × 4 bytes × 2 + аттракторы + optimizer states).
Загрузка:
```python
state = torch.load(path, map_location=device, weights_only=True)
model.load_state_dict(state['model_state'])
# буферы AttractorField (centers, counts, refractory, valid_mask) восстанавливаются автоматически
```

---

## 12. Переменные окружения / код

| Параметр | Файл | Смысл |
|----------|------|-------|
| `DATA_IDS` | train_phase2.py:20 | Путь к BPE-boundary корпусу. |
| `DATA_LABELS` | train_phase2.py:21 | Путь к меткам границ. |
| `CKPT_DIR` | train_phase2.py:22 | Директория чекпоинтов. |
| `SPECIAL_IDS` | train_phase2.py:40 | Токены, маскируемые из CE. |
| `D_MODEL` | phase1_model.py:18 | Размерность модели (384). |
| `N_HEADS` | phase1_model.py:19 | Количество голов (24). |
| `VOCAB_SIZE` | phase1_model.py:25 | Размер словаря (4101). |
| `LAYER_GATE_INIT` | phase1_model.py:76-89 | Инициализация gate по слоям. |
