# EVA — Координатный ИИ

**Не next-token prediction. Навигация в 384-мерном координатном пространстве.**

EVA (Emergent Vector Architecture) — это иерархическая когнитивная архитектура, в которой символы языка представлены как точки в ℝ³⁸⁴, текст как траектория, трансформер как навигатор, а знания как поле аттракторов, формируемое по принципу Хебба.

Модель не угадывает следующий токен через softmax. Она строит карту символьного пространства, накапливает в ней потенциалы пересечений траекторий и генерирует текст через градиентный подъём по этому потенциалу.

**20.5M параметров. 4101 BPE-токен. 12 слоёв. 24 головы. 6 масштабов. 0.7 GB VRAM. Один GPU (MX550).**

---

## 1. Принцип координатной навигации

### 1.1 Проблема next-token prediction

LLM работают так: `P(token_t | tokens_{<t}) = softmax(hidden · W)`. Это статистическая аппроксимация условной вероятности. Модель не понимает, что такое символ — она знает только, какие токены чаще всего следуют друг за другом в обучающих данных.

У этого подхода три фундаментальные проблемы:
1. **Знания неотделимы от весов** — нельзя добавить новое знание без переобучения
2. **Контекстное окно конечно** — модель забывает за пределами окна
3. **Нет репрезентации смысла** — вектор токена — это просто индекс в lookup-таблице

### 1.2 Решение EVA: координатная навигация

EVA представляет каждый символ как точку в 384-мерном пространстве:

```
символ "а"  → z_а ∈ ℝ³⁸⁴
символ "б"  → z_б ∈ ℝ³⁸⁴
слово "дом" → траектория z_д → z_о → z_м
```

Пространство структурировано по 6 лингвистическим масштабам:

| Масштаб | Размерность | Что кодирует |
|---------|-------------|-------------|
| char    | 0-63        | Позиция символа в последовательности |
| morph   | 64-127      | Морфемная роль (корень, суффикс, окончание) |
| word    | 128-191     | Лексическое значение слова |
| phrase  | 192-255     | Синтаксическая роль в фразе |
| sentence | 256-319    | Смысл предложения |
| discourse | 320-383  | Роль в дискурсе |

Каждая координата `z` несёт информацию сразу обо всех шести уровнях. Модель учится **одновременно** предсказывать все шесть аспектов следующего символа.

### 1.3 Почему 384, а не 128 или 768?

- 6 масштабов × 64 dims = 384. Каждому масштабу нужно минимум 64 измерения, чтобы избежать коллапса треков (когда разные символы отображаются в одну точку).
- 384 — минимальная размерность, при которой round-trip тест показал кластеризацию (intra/inter ratio < 0.8 при случайной инициализации).
- Для MX550 с 2.1 GB VRAM: 384-dim позволяет B=8, L=64 (20.5M params, 705 MB).

---

## 2. MultiScaleRoPE — мультимасштабное позиционное кодирование

### 2.1 Принцип

Обычное RoPE (Rotary Position Embedding) применяет одинаковую частотную сетку ко всем головам внимания:

```
θ_k = 10000^(2k/d)   — одинаково для всех голов
```

В EVA каждому масштабу нужна своя частотная характеристика:
- **char**: высокие частоты (θ ≈ 500) — быстрые осцилляции, чувствительность к порядку символов
- **discourse**: низкие частоты (θ ≈ 200000) — медленные осцилляции, чувствительность к глобальной структуре

### 2.2 Реализация

```
θ_k = 500 · (200000/500)^(k/192)   для k = 0, 1, ..., 191
```

θ распределена логарифмически от 500 до 200000 по всем 192 парам измерений. Каждый head attention (через свои QKV веса) выбирает subset этого спектра.

Математически: для позиции pos и dim k:
```
RoPE(pos, k) = rotate_by_angle(pos · θ_k)
```

Где rotate — это поворот в 2D-плоскости (каждая пара dims образует комплексное число):
```
(x_{2k}, x_{2k+1}) → (x·cos - y·sin, y·cos + x·sin)
```

### 2.3 Почему логарифмическая шкала?

Лингвистические масштабы не линейны. Разница между char и morph меньше, чем между word и phrase. Логарифмическая шкала даёт равномерное покрытие всех масштабов.

---

## 3. GroupedScaleAttention — групповая attention по масштабам

### 3.1 Принцип

В стандартном Multi-Head Attention все головы однородны: каждая смотрит на все dims через свои QKV проекции. В EVA головы разделены на 6 групп по 4 головы, каждая группа обрабатывает свой масштаб:

```
Group 0 (char):      heads 0-3  → dims 0-63   → W_O: 64→384
Group 1 (morph):     heads 4-7  → dims 64-127  → W_O: 64→384
Group 2 (word):      heads 8-11 → dims 128-191 → W_O: 64→384
Group 3 (phrase):    heads 12-15→ dims 192-255 → W_O: 64→384
Group 4 (sentence):  heads 16-19→ dims 256-319 → W_O: 64→384
Group 5 (discourse): heads 20-23→ dims 320-383 → W_O: 64→384
```

Каждая из 6 групп имеет **отдельную W_O проекцию** (64→384). Это ключевое отличие от стандартного MHA, где одна W_O проекция (d_model→d_model).

### 3.2 Механика

```
Для слоя l, группы g:
1. Q_g = W_Q_g · x_norm     [B, L, 64]
2. K_g = W_K_g · x_norm     [B, L, 64]
3. V_g = W_V_g · x_norm     [B, L, 64]

4. Для каждой head h в группе:
     score_h = Q_h · K_h^T / √16 + causal_mask
     attn_h = softmax(score_h)
     out_h = attn_h · V_h     [B, L, 16]

5. group_out = concat([out_0, out_1, out_2, out_3])  [B, L, 64]
6. group_out = W_O_g(group_out)                      [B, L, 384]

7. output_l = Σ_g α_{l,g} · group_out_g
```

### 3.3 Soft Gating

α_{l,g} — это softmax от learnable gate-вектора слоя. Инициализация жёстко задана:

Слои 0-1: фокус на char/morph (α_char=0.65, α_morph=0.22). Ранние слои учатся базовой структуре символов.
Слои 2-5: постепенный переход к word/phrase. Средние слои строят лексические и синтаксические репрезентации.
Слои 6-9: доминирование phrase/sentence. Модель учится понимать предложения.
Слои 10-11: sentence/discourse (α_disc=0.60). Верхние слои работают с дискурсом.

### 3.4 Зачем это?

- **Разделение труда**: каждая группа специализируется на своём масштабе
- **Независимые W_O**: проекции не мешают друг другу — char-head не влияет на discourse-head
- **Плавный переход**: soft gating позволяет слоям постепенно переключать фокус, а не резко

---

## 4. Три residual stream

### 4.1 Принцип

В стандартном трансформере один residual stream: `x = x + attention(x); x = x + FFN(x)`. Вся информация смешивается в одном векторе.

В EVA три residual stream, каждый несёт информацию своего уровня иерархии:

```
r₁ = char/morph stream      — обновляется в нижних слоях (0-3)
r₂ = word/phrase stream     — обновляется в средних слоях (4-7)
r₃ = sentence/discourse stream — обновляется в верхних слоях (8-11)
```

### 4.2 Механика

```
На входе слоя l:
    attn_input = norm(x + r₁ + r₂ + r₃)
    attn_out = GroupedAttention(attn_input)
    
    x = x + attn_out
    x = x + SwiGLU(norm(x))
    
    # Обновление residual streams с разными α
    r₁ = r₁ + attn_out · α₁(l)
    r₂ = r₂ + attn_out · α₂(l)
    r₃ = r₃ + attn_out · α₃(l)
```

α распределены по закону: целевой stream получает α=0.3, соседние 0.1-0.15, дальний 0.05.

### 4.3 Зачем три stream, а не один?

Один residual stream — это "бутылочное горлышко": вся информация (от символов до дискурса) должна поместиться в один вектор.

Три streams — это три "полосы движения":
- r₁: локальная информация (какой символ, какая морфема)
- r₂: структурная информация (какое слово, какая фраза)
- r₃: глобальная информация (какое предложение, какой дискурс)

Слой может читать из всех трёх (через `x + r₁ + r₂ + r₃`), но писать преимущественно в свой.

---

## 5. AttractorField — Hebbian-поле аттракторов

### 5.1 Принцип

В биологических нейронных сетях знания хранятся не в весах синапсов (как в LLM), а в паттернах активности — аттракторах. **Принцип Хебба**: "neurons that fire together, wire together" — если два нейрона активируются одновременно, связь между ними усиливается.

EVA реализует этот принцип через AttractorField — поле аттракторов, которое накапливает статистику прохождения траекторий.

### 5.2 Структура аттрактора

Каждый аттрактор a = (μ_a, w_a, r_a, σ_a):

```
μ_a ∈ ℝ³⁸⁴     — центр аттрактора (среднее всех точек, прошедших через него)
w_a ∈ ℝ         — счётчик (сколько треков через него прошло)  
r_a ∈ ℝ³⁸⁴     — рефрактерный вектор (типичное направление выхода из аттрактора)
σ_a ∈ ℝ⁺       — ширина гауссианы (learnable в будущем)
```

### 5.3 Hebbian update

Каждые 10 шагов обучения:

```
1. Найти ближайший аттрактор для точки z:
     c = argmin_a ||z - μ_a||
   
2. Инкрементировать счётчик:
     w_c += 1
   
3. Сдвинуть центр к z (count-normalized EMA):
     μ_c += (lr_c / max(w_c, 1)) · (z - μ_c)
   
   lr_c / max(w_c, 1) — ключевая деталь:
   - Первый проход: lr_c / 1 = сильное обновление (быстрая адаптация)
   - 100-й проход: lr_c / 100 = слабое обновление (стабилизация)
   
4. Обновить рефрактер (направление выхода):
     if z_next is not None:
         r_c += lr_r · ((z_next - z) - r_c)
   
5. Decay всех счётчиков (забывание старых аттракторов):
     w_a *= decay   (decay = 0.999, однократно за forward)
```

### 5.4 Поле потенциала

```
P(z) = Σ_a w_a · exp(-||z - μ_a||² / 2σ²)
```

P(z) — это "ландшафт знаний". Чем выше P(z), тем больше траекторий проходило через эту точку. Аттракторы с большим w_a создают "горы" в ландшафте — это частотные паттерны (например, "в + Москва" → "в Москве").

### 5.5 Градиент потенциала

```
∇P(z) = -Σ_a (w_a / σ²) · exp(-||z-μ_a||²/2σ²) · (z - μ_a)
```

∇P(z) указывает направление к наиболее "знакомым" областям пространства. Это **естественный bias генерации**: двигаться туда, где плотность траекторий выше.

### 5.6 Генерация через поле

```
nxt(z) = η · (μ* - z) / ||μ* - z|| + (1-η) · r*
```

где μ* — центр ближайшего аттрактора, r* — его рефрактер.

Компоненты:
- η · (μ* - z): "притяжение" к центру аттрактора (консерватизм — оставаться в знакомой области)
- (1-η) · r*: "инерция" по типичному направлению выхода (креативность — продолжать паттерн)

### 5.7 Чем AttractorField лучше TensorPotentialField [V×V×V]?

TPF — тензор 4101×4101×4101 = 69 млрд элементов. Непрактично.

AttractorField:
- **Линейная память**: O(V · D) = 4101 × 384 = 1.6M элементов (вместо 69B)
- **Динамический**: аттракторы создаются и умирают (decay threshold)
- **Интерпретируемый**: каждый аттрактор — это кластер в координатном пространстве
- **Hebbian**: обучение без обратного распространения — просто счётчики и EMA

### 5.8 HierarchicalAdditiveField — иерархическая надстройка

AttractorField хранит точки, но не знает их внутренней структуры.
**HierarchicalAdditiveField (HAF)** раскладывает любой вектор z в сумму K суб-векторов:

```
z ≈ v₀ + v₁ + ... + v_K,        K ∈ [0, 8]
```

Каждый vₖ — самостоятельный концепт, рекурсивно разложимый. Sequential
decomposition: каждый шаг извлекает одну компоненту из остатка. Skip-connection
гарантирует реконструкцию с первого шага обучения. Multi-path loss (разные
dropout → разные разложения) даёт согласованность + разнообразие.

HAF превращает **угадывание** следующего токена в **навигацию по иерархии**:
текущий концепт раскладывается на под-концепты, каждый движется к своему
аттрактору, сумма указывает семантически осмысленное направление.

Добавляет ~298K параметров (1.4%). Включён с Phase 2 (W_HAF=0.001).

---

## 6. BoundaryDetectionHead — детекция границ

### 6.1 Принцип

Модель должна знать, где начинаются и заканчиваются слова, даже при BPE-токенизации (когда одно слово может быть разбито на несколько токенов).

### 6.2 Реализация

BoundaryDetectionHead — это простой MLP:

```
h [B, L, 384] → Linear(384→64) → SiLU → Linear(64→3) → logits [B, L, 3]
```

Три класса:
- **0 (word_start)**: первый BPE-токен слова или токен WORD_OPEN
- **1 (word_inside)**: середина multi-token слова
- **2 (word_end)**: последний BPE-токен слова, токен WORD_CLOSE, или пунктуация

Обучается через CrossEntropyLoss на labelled корпусе `full_corpus_bpe_labels.npy`.

### 6.3 Boundary corpus

Создаётся скриптом `create_boundary_labels.py`:

```
Исходный текст → предложения → слова
  для каждого слова:
    если пунктуация: BPE(пунктуация) → label=2 (word_end)
    если буквенное: WORD_OPEN + BPE(слово) с per-token labels + WORD_CLOSE
```

Статистика:
- 60.2M токенов
- 12.05M слов (каждое слово имеет WO и WC)
- 21.7M start, 11.8M inside, 26.8M end

---

## 7. TrajectoryBoundaryPredictor — предсказание траектории

### 7.1 Принцип

Модель должна предсказывать не только следующий токен (через CE loss), но и следующий шаг в координатном пространстве. Это аналог "движения" — откуда мы придём и куда направимся.

### 7.2 Реализация

```
h [B, L, D] → Linear(D→256) → SiLU → Linear(256→3D)
→ (end_coord, nxt_coord, conn_vector)
```

Три выхода:
- **end_coord** [B, L, D]: где заканчивается текущий элемент
- **nxt_coord** [B, L, D]: где начинается следующий элемент
- **conn_vector** [B, L, D]: вектор связи между ними

Loss: `MSE(nxt_coord, h_{t+1} - h_t)` — учится предсказывать delta траектории.

---

## 8. WordWeightEncoder — пулинг токенов в слова

### 8.1 Принцип

BPE-токены — это не слова. "Москве" может быть разбито на ["Моск", "ве"]. EVA нужна word-level репрезентация для понимания структуры предложения.

### 8.2 Реализация

```
1. Self-attention через токены → token_weights [B, L]
   (многослойная attention, каждая позиция attends ко всем предыдущим)

2. Boundary logits → softmax → p_start = prob[..., 0]
   word_ids = cumsum(p_start > 0.5)  # hard threshold

3. scatter_add_ token vectors into word centroids:
   for each position t:
       word_vecs[b, word_id] += h[b, t] * token_weights[b, t]
   
   word_vecs /= sum(token_weights per word)

4. Word importance: word_weights = Sigmoid(Linear(word_vecs))
```

WordWeightEncoder принимает boundary_logits от BoundaryDetectionHead (share весов, не дублирует вычисления).

---

## 9. ResidualHead — предсказание дельты координат

### 9.1 Принцип

Если модель может предсказать, на сколько изменится координата z при переходе к следующему токену — значит, она понимает семантический сдвиг.

### 9.2 Реализация

```
input: h_t (hidden), z_{t-1} (предыдущая координата), z_t (текущая координата)
concat = [h_t, z_{t-1}, z_t]  [B, L, 3·384]
delta_pred = Linear(concat) → SiLU → Linear(384→384) → Linear(384→384)

delta_true = z_t - z_{t-1}
loss = MSE(delta_pred, delta_true)
```

Высокий residual_error → высокая неопределённость → сигнал для thought loop (Phase 4).

---

## 10. MetaWeighter — динамическое смешивание источников

### 10.1 Принцип

При генерации у модели есть три источника информации:
1. **Knowledge**: logits от decoder (что обычно следует после этого контекста)
2. **Concept**: distance-based logits от ближайших символов в координатном пространстве
3. **Contradiction**: inverse-uncertainty weighted logits

MetaWeighter учится динамически смешивать их в зависимости от контекста:

### 10.2 Реализация

```
context = mean(h, dim=1)  [B, 384]
w = context → Linear(64) → SiLU → Linear(3) → softmax
→ [w_know, w_conc, w_contr]

final_logits = w_know · logits_know + w_conc · logits_conc + w_contr · logits_contr
```

Bias по умолчанию: knowledge (w_know > 0.5). Temperature растёт от 0.1 до 1.0 за warmup.

---

## 11. Loss функции

### 11.1 CE Loss (основной)

```
L_CE = CrossEntropy(logits, targets)  — на всех позициях, кроме special tokens
```

Только non-special токены (PAD, UNK, BOS, EOS, GAP, SENT_OPEN, SENT_CLOSE).

### 11.2 nxt Loss (траектория)

```
L_nxt = MSE(nxt_pred, h_{t+1} - h_t)  — delta trajectory prediction
```

Учит модель предсказывать движение в координатном пространстве.

### 11.3 Boundary Loss (границы)

```
L_boundary = CrossEntropy(boundary_logits, boundary_labels)  — 3 класса
```

Из pre-computed labels (full_corpus_bpe_labels.npy).

### 11.4 L_align (cross-level consistency)

```
align_loss = MSE(boundary_probs[inside], WordWeightEncoder.importance.sigmoid())
```

Согласует boundary detection (char-level inside probability) с word-level pooling.
Логика: если токен находится внутри слова по boundary detection, он должен иметь высокую важность для word pooling, и наоборот.

### 11.5 Полный loss

```
L = L_CE + 0.05 · L_nxt + 0.1 · L_boundary + 0.05 · L_align
  + 0.01 · (L_ac + L_dv) + 0.001 · L_haf
```

Где L_haf — multi-path loss иерархического разложения:
```
z_pooled = mean(h)   →   decompose(z_pooled) → [v₀,...,v_K]
L_haf = ||z - Σv||² + 0.05·||Σv_A - Σv_B||² + 0.005·(K_A+K_B)/8
```

---

## 12. Фазы обучения

### Phase 0 (Boundary Detection)

- **Данные**: full_corpus_encoded.npy (char tokens с WORD_OPEN/CLOSE)
- **Модель**: 128-dim char model + BoundaryDetectionHead
- **Loss**: BCE boundary
- **Шагов**: 5000
- **Результат**: веса BoundaryDetectionHead обучены (std=0.115)

### Phase 1 (Char-level Pre-training)

- **Данные**: full_corpus_ids.npy (char tokens 0-154)
- **Модель**: 384-dim / 12L / 24H (UnifiedMultidimensionalTransformerV2)
- **Loss**: CE + nxt
- **Шагов**: 60000 (остановлено на плато CE≈1.3)
- **Результат**: базовая репрезентация char в 384-dim (intra/inter ratio < 0.8, все 384 dims активны)

### Phase 2 (BPE Training + HAF)

- **Данные**: full_corpus_bpe_boundary.npy (60.2M BPE tokens с WO/WC)
- **Loss**: CE + nxt + boundary + L_align + attractor + HAF
- **Шагов**: 200000
- **Новое**: HierarchicalAdditiveField — иерархическое аддитивное хранение
- **W_HAF**: 0.001 (multi-path loss на разложение)
- **Статус**: IN PROGRESS (текущий запуск)

### Phase 3 (AttractorField Deepening)

- **AttractorField**: Hebbian update каждые 10 шагов
- **HAF**: рекурсивное хранение в attractors + multi-path loss
- **Loss**: CE + nxt + boundary + align + attractor + HAF + meta_KL + flow
- **Шагов**: 200000

### Phase 4 (Gradient Ascent Generation)

Генерация через градиентный подъём по P(z):

```
1. z_curr = model.get_last_coordinate(input_ids)
2. For _ in range(n_steps):
     grad = attractor_field.gradient(z_curr)
     z_curr += η · ∇P(z_curr) + noise·√(2·D·η)
3. next_token = argmin ||z_curr - embed_weights||
```

---

## 13. Данные

### 13.1 BPE Tokenizer

- HuggingFace tokenizers (Rust)
- ByteLevel pre-tokenizer
- Vocab: 4096 tokens (тренирован на full_corpus_ru.txt, 173 MB)
- Special tokens: `<PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3`
- Boundary tokens (поверх BPE vocab): 4096-4100

### 13.2 Корпуса

| Файл | Размер | Описание |
|------|--------|----------|
| `full_corpus_ru.txt` | 173 MB | Исходный текст из Wikipedia + ConceptNet |
| `full_corpus_ids.npy` | 757 MB | Char-level IDs (94.7M токенов, 0-154) |
| `full_corpus_bpe.npy` | 235 MB | BPE-encoded (29.4M токенов, 3890/4096 vocab) |
| `full_corpus_bpe_boundary.npy` | 482 MB | BPE + WO/WC markers (60.2M токенов) |
| `full_corpus_bpe_labels.npy` | 60 MB | Boundary labels (0=start, 1=inside, 2=end) |

---

## 14. Сравнение с альтернативами

| Характеристика | GPT-2 (124M) | GPT-2 distilled (50M) | EVA (20.5M) |
|---|---|---|---|
| Параметров | 124M | 50M | **20.5M** |
| VRAM | ~2 GB | ~1 GB | **0.7 GB** |
| Размерность | 768 | 512 | **384** |
| Слоёв | 12 | 6 | **12** |
| Голов | 12 | 8 | **24 (6 групп)** |
| Vocab | 50257 | 50257 | **4101** |
| Позиции | Learned | Learned | **MultiScaleRoPE** |
| Residual | 1 stream | 1 stream | **3 streams** |
| FFN | ReLU | ReLU | **SwiGLU** |
| Память | Отсутствует | Отсутствует | **AttractorField + HAF** |
| Генерация | argmax | argmax | **∇P(z) ascent + hierarchical** |
| Разложение | Нет | Нет | **Sequential additive decomposition** |
| Обучение | ~недель | ~дней | **~20 часов (1 GPU)** |

---

## 15. Как это работает в production

### 15.1 Обучение

```bash
# Настроить окружение
pip install torch numpy tokenizers

# Phase 2 (текущий этап)
python train_phase2.py

# С resume из Phase 1 чекпоинта:
python train_phase2.py --resume checkpoints/v4/phase1_step_60000.pt
```

### 15.2 Инференс

```python
from eva.symbolic.phase1_model import UnifiedMultidimensionalTransformerV2
from eva.symbolic.bpe_tokenizer import BPEVocab

model = UnifiedMultidimensionalTransformerV2(vocab_size=4101)
state = torch.load('checkpoints/v4/phase2_step_20000.pt', weights_only=True)
model.load_state_dict(state['model_state'], strict=False)  # strict=False для resume без HAF
model.eval()

cv = BPEVocab()
prompt = cv.encode('Жили-были')
text, _ = model.generate_text(prompt, cv, max_new=128, temperature=0.8)
# text → "Жили-были дед и баба..."
```

### 15.3 Eval

```bash
python eval_phase1.py --ckpt checkpoints/v4/phase2_step_20000.pt
```

Выводит:
- Примеры генерации
- Topology analysis (intra/inter расстояния между токенами)
- Активные размерности (dim variance по всем 384 dims)

---

*EVA — не LLM. Не статистический попугай. Это координатный навигатор, строящий карту символьного пространства и движущийся по ней градиентным подъёмом.*

*20.5 миллионов параметров. Один GPU. Двадцать часов. Ноль предобученных моделей. Полный цикл: символ → координата → траектория → аттрактор → генерация.*
