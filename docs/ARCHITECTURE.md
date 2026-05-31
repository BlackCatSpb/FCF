# EVA Architecture v4 — Technical Reference

## 0. Философия

### 0.1 Отказ от next-token prediction

Стандартные LLM решают задачу:

```
P(w_t | w_{<t}) = softmax(h_t · W)    где h_t — скрытое состояние, W — матрица декодера
```

Это **статистическая аппроксимация**. Модель не знает, что такое "смысл" — она знает только совместную встречаемость токенов.

EVA решает другую задачу:

```
z_t = f(token_t)                       — символ → координата
z_{t+1} = z_t + ∇P(z_t)               — следующая координата через градиент поля
token_{t+1} = argmin ||z_{t+1} - E||   — координата → ближайший символ
```

### 0.2 Почему координаты?

В природе смысл — это не метка, а положение в пространстве. Слова-синонимы находятся рядом, слова-антонимы — далеко, но в одном направлении. Глаголы и существительные занимают разные области.

### 0.3 Почему поле аттракторов?

Знания не должны быть "зашиты" в веса модели. Они должны быть отделимым словарём прецедентов — набором точек в пространстве с указанием, куда от них обычно движутся.

---

## 1. MultiScaleRoPE

### 1.1 Принцип

RoPE (Rotary Position Embedding) кодирует позицию tokens через вращение в 2D-плоскостях:

```
f(x_m, m) = R_θ_m · x_m

где R_θ_m — матрица вращения на угол m·θ для каждой пары измерений

θ_k = base^(2k/d) для стандартного RoPE
```

EVA модифицирует распределение θ:

```
θ_k = 500 · (200000 / 500)^(k/192)   для k = 0 .. 191
```

### 1.2 Свойства

- **θ_0 = 500** (dim 0-1): период 2π·500 ≈ 3140 позиций — высокая частота, различает соседние символы
- **θ_191 = 200000** (dim 382-383): период 2π·200000 ≈ 1.26M позиций — низкая частота, различает глобальные структуры
- **Логарифмическая шкала**: каждый dim получает θ, в 1.03 раза больше предыдущего

### 1.3 Реализация

```python
half = D // 2  # 192
k = torch.arange(half, dtype=torch.float32)
theta = THETA_MIN * (THETA_MAX / THETA_MIN) ** (k / half)
freqs = 1.0 / theta  # ω_k = 1/θ_k

pos = torch.arange(max_seq_len)
angles = pos[:, None] * freqs[None, :]  # [max_seq_len, half]
cos = angles.cos()
sin = angles.sin()

# apply: x → [x0·cos - x1·sin, x1·cos + x0·sin]
```

---

## 2. GroupedScaleAttention

### 2.1 Принцип: разделение по масштабам

В природе языка одновременно действуют процессы на разных масштабах:
- **Микро**: какие буквы образуют морфему (char→morph)
- **Мезо**: какие морфемы образуют слово (morph→word)  
- **Макро**: какие слова образуют предложение (word→sentence)
- **Глобальный**: как предложения связаны в дискурсе (sentence→discourse)

EVA выделяет 6 групп голов — по одной на каждый масштаб.

### 2.2 Архитектура

```
Q = W_Q · norm(x + r1 + r2 + r3)    # единый Q для всех групп
K = W_K · norm(x + r1 + r2 + r3)    # единый K для всех групп  
V = W_V · norm(x + r1 + r2 + r3)    # единый V для всех групп

Q, K, V = reshape(Q, [B, L, 24, 16])  # 24 heads × 16 dim

for group g in range(6):
    heads_g = Q[:, :, g*4:(g+1)*4]    # 4 heads for this group
    
    for head h in heads_g:
        score = head_Q · head_K^T / √16 + causal_mask
        attn = softmax(score)         # [B, L, L]
        out = attn · head_V           # [B, L, 16]
    
    group_out = concat(heads_out)     # [B, L, 64]
    group_out = W_O_g(group_out)      # [B, L, 384] — своя W_O на группу
    
out = Σ softmax(gate_l)_g · group_out_g
```

### 2.3 Per-layer gating

Gate-вектор каждого слоя `gate_l ∈ ℝ⁶` инициализирован по layer-specific bias:

```python
LAYER_GATE_INIT = {
    0:  [0.70, 0.20, 0.10, 0.00, 0.00, 0.00],   # char-heavy
    1:  [0.60, 0.25, 0.15, 0.00, 0.00, 0.00],
    2:  [0.40, 0.30, 0.20, 0.10, 0.00, 0.00],   # mix char/morph/word
    3:  [0.30, 0.30, 0.25, 0.15, 0.00, 0.00],
    4:  [0.10, 0.30, 0.30, 0.20, 0.10, 0.00],   # word/phrase-heavy
    5:  [0.05, 0.25, 0.30, 0.25, 0.15, 0.00],
    6:  [0.00, 0.15, 0.25, 0.30, 0.25, 0.05],   # phrase/sentence
    7:  [0.00, 0.10, 0.20, 0.30, 0.30, 0.10],
    8:  [0.00, 0.00, 0.10, 0.25, 0.40, 0.25],   # sentence/discourse
    9:  [0.00, 0.00, 0.05, 0.20, 0.40, 0.35],
    10: [0.00, 0.00, 0.00, 0.10, 0.35, 0.55],
    11: [0.00, 0.00, 0.00, 0.05, 0.30, 0.65],   # discourse-heavy
}
```

---

## 3. Residual Streams

### 3.1 Зачем?

В обычном трансформере:

```
x_l = x_{l-1} + Attention(LayerNorm(x_{l-1}))
x_l = x_l + FFN(LayerNorm(x_l))
```

Информация "смешивается" в одном векторе. Проблема: на слое 10 всё ещё нужно помнить, какой символ был на позиции 3, но вектор уже перезаписан discourse-информацией.

### 3.2 Три потока

```python
# Инициализация
residual1 = zeros(B, L, D)   # char/morph — локальная структура
residual2 = zeros(B, L, D)   # word/phrase — синтаксис
residual3 = zeros(B, L, D)   # sentence/discourse — глобальная структура

# Каждый слой
for l in range(12):
    attn_out = Attention(Norm(x + residual1 + residual2 + residual3))
    x = x + attn_out + FFN(Norm(x))
    
    # Обновление с layer-specific α
    residual1 += attn_out * α1(l)   # α1: 0.30 → 0.10 → 0.05
    residual2 += attn_out * α2(l)   # α2: 0.10 → 0.30 → 0.15
    residual3 += attn_out * α3(l)   # α3: 0.05 → 0.15 → 0.30
```

### 3.3 Динамика α

```
Layer 0-3:  char/morph learning  (α1=0.30, α2=0.10, α3=0.05)
Layer 4-7:  word/phrase learning (α1=0.10, α2=0.30, α3=0.15)
Layer 8-11: global structure     (α1=0.05, α2=0.15, α3=0.30)
```

---

## 4. AttractorField

### 4.1 Формальное определение

Поле аттракторов — это набор точек в ℝ³⁸⁴ с весами:

```
A = {(μ_a, w_a, r_a) | a = 0..N_attr}
```

Потенциал поля:

```
P(z) = Σ_a w_a · exp(-||z - μ_a||² / 2σ²)
```

### 4.2 Hebbian update

Алгоритм обновления при получении новой точки z ∈ ℝ³⁸⁴:

```
def hebbian_update(z, z_next=None):
    # 1. Find closest attractor
    c = argmin_a ||z - μ_a||
    
    # 2. Increment
    w_c += 1
    n = w_c  # total visits
    
    # 3. Count-normalized center EMA
    μ_c += (lr_center / n) · (z - μ_c)
    
    # 4. Refractory vector EMA
    if z_next is not None:
        r_c += lr_refract · ((z_next - z) - r_c)
    
    # 5. Global decay (once per forward)
    w_a *= decay
```

### 4.3 Свойства count-normalized update

Стандартная EMA: `μ += lr · (z - μ)` — каждый новый sample двигает центр одинаково.

Count-normalized: `μ += (lr/n) · (z - μ)` — ранние samples двигают сильно, поздние — слабо. Это даёт **сходимость**: μ_a → истинное среднее всех точек в кластере.

### 4.4 nxt_direction — направление генерации

```
nxt(z) = η · (μ* - z)_norm + (1-η) · r*_norm

где:
- μ* = центр ближайшего к z аттрактора
- r* = рефрактер этого аттрактора
- η = learnable (default 0.7)
- (·)_norm = normalize to unit vector
```

Компоненты:
- η·(μ*-z): **консервативное** притяжение — оставаться в области известного
- (1-η)·r*: **инновационное** продолжение — двигаться в типичном направлении выхода

### 4.5 gradient — градиент поля

```
∇P(z) = -Σ_a (w_a/σ²) · exp(-||z-μ_a||²/2σ²) · (z - μ_a)
```

Используется для gradient ascent генерации (Phase 4):

```
for step in range(n_steps):
    z += η · ∇P(z) + noise  # Langevin dynamics
```

---

## 5. Boundary Detection

### 5.1 Задача

Для последовательности BPE-токенов определить границы слов (начало/середина/конец).

### 5.2 BoundaryDetectionHead

```python
class BoundaryDetectionHead(nn.Module):
    def __init__(self, d_model=384):
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 64),   # проекция
            nn.SiLU(),                 # нелинейность
            nn.Linear(64, 3),         # [word_start, inside, word_end]
        )
    
    def forward(self, h):
        return self.mlp(h)  # [B, L, 3]
```

### 5.3 Boundary corpus

Создаётся скриптом `create_boundary_labels.py`:

```
Input: full_corpus_ru.txt (русский текст)

Для каждой строки:
  Разбить на предложения (по .!?… + заглавная)
  
  Для каждого предложения:
    Разбить на слова (по пробелам)
    
    Для каждого слова:
      Извлечь alpha-части (русские буквы) и punctuation
      
      Для alpha-части:
        BPE-encode → word_ids
        label: 
          - каждый id: start(0) если первый, inside(1) если середина, end(2) если последний
          - если 1 токен → end(2) (одновременно start и end)
      
      Для punctuation:
        BPE-encode → punct_id
        label: end(2)
```

### 5.4 Ограничения

- Не учитывает составные слова через дефис (разбиваются на части)
- Не различает SENT_OPEN/SENT_CLOSE (только WORD level)
- Все позиции имеют label (0, 1, или 2) — нет ignore_index

---

## 6. L_align — Cross-level Consistency

### 6.1 Принцип

Информация о границах слов должна быть согласована между:
- **char-level**: BoundaryDetectionHead — per-token "inside" probability
- **word-level**: WordWeightEncoder — per-word importance score

### 6.2 Loss

```
L_align = MSE(char_inside, word_importance_sigmoid)

где:
- char_inside = boundary_probs[..., 1]  — вероятность "внутри слова" для каждого токена  
  shape: [B, L]
- word_importance = Sigmoid(Linear(word_vec)).sigmoid()  — важность каждого слова
  shape: [B, L] — broadcast от word-level к token-level по word_id
```

### 6.3 Зачем?

- BoundaryHead учится на labelled данных (supervised)
- WordWeightEncoder учится на attention (unsupervised)
- L_align — bridge между ними: заставляет attention WordWeightEncoder согласовываться с supervised boundaries

---

## 7. Composite Loss

```
L = W_CE · L_CE 
  + W_NXT · L_nxt 
  + W_BOUNDARY · L_boundary 
  + W_ALIGN · L_align
  + W_ATTRACTOR · (L_ac + L_dv)
  + W_HAF · L_haf

Где:
W_CE = 1.0
W_NXT = 0.05
W_BOUNDARY = 0.1  
W_ALIGN = 0.05
W_ATTRACTOR = 0.01
W_HAF = 0.001
```

### 7.1 Gradients

```
∂L/∂θ_model = ∂L_CE/∂θ + 0.05·∂L_nxt/∂θ + 0.1·∂L_boundary/∂θ 
              + 0.05·∂L_align/∂θ + 0.01·∂L_AF/∂θ + 0.001·∂L_HAF/∂θ
```

L_boundary влияет только на BoundaryDetectionHead + скрытые состояния.
L_nxt влияет на TrajectoryBoundaryPredictor + скрытые состояния.
L_align влияет на оба — согласует их.
L_CE влияет на всю модель — основной сигнал.
L_HAF влияет на HierarchicalAdditiveField (slot_net, stop_head) + скрытые состояния.

### 7.2 Regularization

- Weight decay: 0.01 (AdamW)
- Gradient clipping: 1.0
- Temperature clamp: [0.1, 10.0]
- AttractorField decay: 0.999 per step

---

## 8. Generation Mechanics

### 8.1 Полный процесс

```
1. Вход: prompt_ids = [BOS, t1, t2, ..., tn]
2. h, logits, heads = model(prompt_ids, return_heads=True)
3. z_curr = h[0, -1]  — последняя координата

4. Если use_attractors и attractors > 0:
     nxt = attractor_field.nxt_direction(z_curr.unsqueeze(0))[0]
   Иначе:
     _, nxt, _ = boundary_predictor(h[:, -1:])  # [1, 1, D]
     nxt = nxt[0, 0]  # [D]

5. z_pred = z_curr + nxt  # [D]

6. logits_know = decoder(z_pred.unsqueeze(0).unsqueeze(0))[0, 0]

7. sym_coords = embed.weight  # [4101, 384]
   dists = -cdist(z_pred.unsqueeze(0), sym_coords, p=2)[0]
   
8. concept_score = heads['concept'][0, -1].item()
   contra_score = heads['contradiction'][0, -1].item()
   
   logits_conc = dists * (1.0 + concept_score)
   logits_contr = dists * (1.0 - contra_score * 0.5)

9. meta_w = heads['meta_weights'][0]  # [3]
   final = (meta_w[0]·logits_know + meta_w[1]·logits_conc + meta_w[2]·logits_contr) / temp

10. mask special tokens [0,1,2,3,GAP,157,158,159,160]
11. repetition penalty: final[t] -= count[t] * 0.5
12. top-20 → softmax → multinomial → next_token
13. append → repeat from step 2
```

### 8.2 Режимы generation

| Режим | nxt source | Когда |
|-------|-----------|-------|
| Standard | boundary_predictor | По умолчанию |
| Attractor | attractor.nxt_direction | Если накоплено >0 аттракторов |
| Hierarchical | haf.nxt_direction | Декомпозиция + аттракторы компонентов (Phase 3+) |
| Gradient | ∇P(z) ascent | Phase 4 |

### 8.3 HAF в генерации (перспектива)

```
1. z_curr — текущий вектор
2. decompose(z_curr) → [v₀, v₁, ..., v_K]  — разложить на компоненты
3. Для каждого vₖ: найти ближайший аттрактор → nxt_k
4. z_next = Σ(vₖ + nxt_k) / K  — собрать обратно
5. token_next = argmin ||z_next - E||
```

Преимущество: генерация учитывает иерархию концепта. Если z_curr = "пришёл",
decompose найдёт ["при-", "-шёл-", "-л"], каждый компонент двигается к своему
аттрактору, сумма даёт направление к семантически связанному следующему слову.

---

## 9. Parameter Schema

### 9.1 Embedding (1,574,784 params)

```
E ∈ ℝ^{4101 × 384}
init: Normal(0, 0.02)
```

### 9.2 Attention per layer (589,824 params per layer × 12 = 7,077,888)

```
Q: Linear(384, 384)      = 147,456
K: Linear(384, 384)      = 147,456
V: Linear(384, 384)      = 147,456
W_O × 6: 6 × Linear(64, 384) = 147,456
Total: 589,824
```

### 9.3 FFN per layer (589,824 params per layer × 12 = 7,077,888)

```
W_gate: Linear(384, 512) = 196,608
W_up:   Linear(384, 512) = 196,608
W_down: Linear(512, 384) = 196,608
Total: 589,824
```

### 9.4 Heads

| Head | Params | Formula |
|------|--------|---------|
| BoundaryDetection | 24,771 | 384·64 + 64·3 + biases |
| TrajectoryPredictor | 394,240 | 384·256 + 256 + 256·1152 + 1152 |
| ConceptHead | 26,689 | 384·64 + 64 + 64·32 + 32 + 32·1 + 1 |
| ContradictionHead | 26,689 | (same as Concept) |
| UncertaintyHead | 49,472 | 384·64 + 64 + 64·384 + 384 |
| ResidualHead | 484,608 | 3·384·384 + 384 + 384·64 + 64 + 64·384 + 384 |
| MetaWeighter | 24,771 | 384·64 + 64 + 64·3 + 3 |
| WordValence | 1,099,520 | 2 × (384·128 + 128 + 128·4101 + 4101) |
| BoundaryValidator | 49,408 | 2·384·64 + 64 + 64·2 + 2 |
| AttractorField | 0 | Buffers only |

### 9.5 Total: 20,473,064 (≈20.5M)

---

## 10. Training Config

### 10.1 Current (Phase 2)

```python
N_STEPS = 200000
B = 8
L = 64
LR = 3e-4
WARMUP = 4000
OPTIMIZER = AdamW(weight_decay=0.01)
SCHEDULER = SequentialLR([
    LinearLR(start_factor=1e-4, total_iters=4000),
    CosineAnnealingLR(T_max=196000),
], milestones=[4000])
CLIP_NORM = 1.0
```

### 10.2 Data flow

```
# per step:
idx = randint(0, N - L - 1, size=B)        # random offsets
batch_ids = stack([data[i:i+L] for i in idx])     # [B, L]
batch_labels = stack([labels[i:i+L] for i in idx]) # [B, L]
targets = stack([data[i+1:i+L+1] for i in idx])    # [B, L]

x = tensor(batch_ids)
y_labels = tensor(batch_labels)
targets = tensor(targets)

h, scores, weights, heads = model(x, return_heads=True)
ce_loss = CE(scores, targets)            # masked for specials
nxt_loss = MSE(nxt_pred, h_diff)          # trajectory
boundary_loss = CE(boundary_logits, y_labels)  # boundary
align_loss = MSE(char_inside, word_pooled)     # consistency
```

### 10.3 Metrics per step

```
ce=2.34 nxt=0.31 bc=0.13 align=0.0004 ac=0.002 dv=0.14
hf=0.003 hk=3 hr=0.001 acc=0.51 b_acc=0.95 att=2100 haf_att=85
```

- ce: CE loss (ожидаем 1.0-2.0 в начале)
- nxt: trajectory MSE (должен снижаться к ~0.01)
- bc: boundary loss (0.0 если нет WO/WC в данных, ∼0.1 если есть)
- align: L_align MSE (согласование char/word)
- ac: attractor consistency loss (MSE h→nearest center)
- dv: attractor diversity loss (ортогональность центров)
- hf: HAF multi-path loss (разложение + реконструкция)
- hk: HAF K — число компонент декомпозиции на шаге
- hr: HAF residual — норма остатка после декомпозиции
- acc: token prediction accuracy (∼50-60% на char, ∼30-40% на BPE)
- b_acc: boundary accuracy (80-95% с labelled corpus)
- att: количество аттракторов в AttractorField
- haf_att: количество аттракторов в HAF

---

## 11. HierarchicalAdditiveField

### 11.1 Мотивация

Стандартный AttractorField хранит точки в ℝ³⁸⁴, но не знает их внутренней
структуры. Концепт "пришёл" — это одновременно целое и сумма частей
("при-" + "-шёл-" + "-л"). HAF реализует **иерархическое аддитивное хранение**:

```
z ∈ ℝᴰ → decompose → [v₀, v₁, ..., v_K], K ∈ [0, max_arity], z ≈ Σ vₖ
```

Каждый vₖ — самостоятельный концепт, который тоже раскладывается. Нулевые
векторы — разделители между уровнями (аналог "0" в вашем примере 5 = 1+4 = 2+3).

### 11.2 Sequential Decomposition

В отличие от параллельного подхода (все компоненты из одного выстрела), HAF
использует **последовательное разложение** — каждый шаг объясняет остаток:

```
r₀ = z                              # residual = исходный вектор
for k = 0..max_arity:
    stopₖ = σ(W_stop · rₖ + b_stop)  # вероятность остановки
    vₖ  = MLP_slot(rₖ + posₖ) + rₖ   # skip-connection: vₖ ≈ rₖ на старте
    rₖ₊₁ = rₖ - vₖ                   # новый residual
    if stopₖ > 0.5: break            # STOP
return [v₀, v₁, ..., v_K]           # z = Σ vₖ (с точностью до r_K ≈ 0)
```

**Skip-connection** (MLP + r) — ключевое техническое решение:
- На инициализации MLP≈0 → v₀≈z → r₁≈0 → perfect reconstruction за 1 шаг
- Модель не тратит capacity на реконструкцию, сразу учится дробить осмысленно
- Гарантия: ||z - Σvₖ|| ≈ ||r_K|| < 0.001 после обучения

### 11.3 Variable Arity (STOP head)

Количество компонент K определяется контекстом через sigmoid-врата:

```python
stop_logit = W · r + b              # один Linear(384 → 1)
stop_prob = sigmoid(stop_logit)      # [0, 1]
# STOP когда stop_prob > 0.5
```

Во время обучения — **Gumbel-Sigmoid** для дифференцируемого дискретного выбора:

```python
gumbel = -log(-log(U + ε) + ε)
stop_noisy = sigmoid((stop_logit + gumbel) / τ)
stop_hard = (stop_noisy > 0.5).float()
stop = stop_hard + stop_soft - stop_soft.detach()  # straight-through
```

На первом шаге (k=0) STOP всегда заблокирован — нужен минимум 1 компонент.

### 11.4 Multi-Path Loss

Один концепт может быть разложен по-разному (как 5 = 1+4 = 2+3).
Разные dropout seeds → разные decomposition paths:

```
Path A (dropout=0.05): z → [v₀, v₁]    (2 компонента)
Path B (dropout=0.50): z → [w₀, w₁, w₂] (3 компонента)
Оба: Σ ≈ z
```

Loss:
```
L_haf = W_recon · (||z - Σvₖ||² + ||z - Σwₖ||²)    # реконструкция (≈0)
      + W_cross · ||Σvₖ - Σwₖ||²                     # согласованность путей
      + W_sparsity · (K_A + K_B) / max_arity          # мало компонент
      + W_diversity · cos(mean(v), mean(w))            # разные паттерны
```

### 11.5 Иерархия

Каждый vₖ — вектор в ℝ³⁸⁴ → рекурсивно раскладывается:

```
z (путь)
├── v₀ (глагольный корень)
│   ├── w₀ (приставка "при-")
│   ├── w₁ (корень "-шёл-")
│   └── w₂ (окончание "-л")
└── v₁ (вспомогательный смысл)
    ├── x₀
    └── x₁
```

Хранение: каждый узел → AttractorField (Hebbian update).

```
store_hierarchical(z, depth=2):
    1. hebbian_update(z)                    — сохранить z
    2. parts = decompose(z)                  — разложить
    3. for each p in parts:
         a. hebbian_update(p)               — сохранить компонент
         b. if depth > 0: recurse(p)        — рекурсия
```

### 11.6 Параметры

| Параметр | Значение | Смысл |
|----------|----------|-------|
| coord_dim | 384 | Размерность = d_model |
| max_arity | 8 | Максимум компонент в разложении |
| max_depth | 5 | Глубина рекурсивного дерева |
| creation_threshold | 0.1 | Порог создания нового аттрактора |

**slot_net**: Linear(384→384) → SiLU → Linear(384→384), 295K params.
**stop_head**: Linear(384→1), 385 params.
**slot_pos**: [8, 384] learnable positional embedding, 3072 params.
**depth_scale**: [5] learnable, 5 params.
**gs_temp**: [1] learnable Gumbel temperature, 1 param.
**attractors**: AttractorField(10000, 384) — 0 trainable, ~30 MB buffers.

Total HAF params: ~298K (1.4% от 20.5M модели).

### 11.7 Интеграция в модель

```python
# В __init__ модели:
self.haf = HierarchicalAdditiveField(coord_dim=d_model)

# В forward (update_attractors=True):
z_pooled = h.mean(dim=1)  # [B, D]
for b in range(min(B, 4)):
    self.haf.store_hierarchical(z_pooled[b], depth=2)

# В train loop (step > HAF_WARMUP):
z_pooled = h.mean(dim=(0, 1))  # [D] — глобальный mean
haf_loss = self.haf.multi_path_loss(z_pooled, n_paths=2)
total_loss += W_HAF * haf_loss  # W_HAF = 0.001
```

### 11.8 Связь с концептами и противоречиями

HAF даёт модели **явное представление** о структуре знания:
- Вместо "угадывания" следующего токена — декомпозиция текущего концепта
- Концепт ("понятие") = точка в пространстве с известным разложением
- Противоречие = компоненты, чьи аттракторы указывают в разные стороны
- Поиск = спуск по иерархии: z → decompose → sub-аттракторы → ближайшие концепты

Это делает ConceptHead и ContradictionHead не эвристиками, а
**измеримыми величинами**: density аттракторов вокруг z → concept_score,
dispersion направлений sub-аттракторов → contradiction_score.

---

## 12. Key Files

```
FCF/
├── train_phase1.py              — Phase 1: char-level (CE + nxt)
├── train_phase2.py              — Phase 2: BPE (CE + nxt + bd + align) ← текущий
├── eval_phase1.py               — Eval (generation + topology)
├── create_boundary_labels.py    — Boundary corpus builder
├── encode_bpe_corpus.py         — BPE corpus encoder
├── test_compile.py              — Verification tests
├── test_integrity.py            — Data integrity checks
│
├── eva/symbolic/
│   ├── phase1_model.py          — UnifiedMultidimensionalTransformerV2 (вкл. HAF)
│   ├── heads.py                 — All head modules
│   ├── subspace_coords.py       — WordWeightEncoder
│   ├── potential_fields.py      — AttractorField, WSentenceContextField,
│   │                              HierarchicalAdditiveField, etc.
│   └── bpe_tokenizer.py         — BPE wrapper (HuggingFace tokenizers)
│
├── real_data/
│   ├── full_corpus_ru.txt       — Raw text (173 MB)
│   ├── full_corpus_ids.npy      — Char-level (94.7M tokens)
│   ├── full_corpus_bpe.npy      — BPE (29.4M tokens)
│   ├── full_corpus_bpe_boundary.npy  — BPE + boundaries (60.2M tokens)
│   ├── full_corpus_bpe_labels.npy    — Boundary labels (0/1/2)
│   └── bpe_tokenizer.json       — Trained tokenizer
│
├── checkpoints/v4/
│   ├── phase1_step_20000.pt     — Phase 1 checkpoints
│   ├── phase1_step_40000.pt
│   └── phase1_step_60000.pt
│
└── docs/
    ├── ARCHITECTURE.md          — This file
    └── architecture_v4.md       — Original spec (historical)
```
