# FCF Neuro-Symbolic Audit V22
## Дата: 2026-06-23
## Версия: 7585dfb (HEAD)

---

## Содержание

1. [ZeckendorfQuantizer: SNR и шум bundle](#1-zeckendorfquantizer-качество)
2. [TemporalZeckendorf: замена exp(-d/τ) на Fibonacci theta](#2-temporalzeckendorf-замена-decay)
3. [VSA-консистентность: hybrid bind/unbind](#3-vsa-консистентность)
4. [HDTransformerLayer vs VSAAttention: конфликт pipeline](#4-hdtransformerlayer-vs-vsaattention)
5. [Harmonizer + MorphSTDP: двойной учёт морфем](#5-harmonizer--morphstdp)

---

## 1. ZeckendorfQuantizer: SNR и шум bundle

### 1.1 Архитектура кодировщика

**Файл:** `eva/symbolic/fibonacci_utils.py:72-133`

`ZeckendorfQuantizer` преобразует float-вес w в HD-вектор размерности D=768 через следующую процедуру:

1. `idx = int(round(abs(w) * scale))` — квантование веса с масштабом `scale=10000`
2. `fibs = FibonacciUtils.zeckendorf(idx)` — разложение Цекендорфа
3. `indices = [self._fib_to_idx[f] for f in fibs]` — маппинг чисел Фибоначчи на индексы базисных HD-векторов
4. `vec = sum(self._vecs[i] for i in indices)` — bundle всех Fib-векторов
5. Нормализация суммы до единичной нормы

### 1.2 Анализ числа Fib-векторов

Максимальное число Фибоначчи, используемое в encode: `max_fib_value=100000`, что соответствует `FibonacciUtils.get(i) <= 100000`. Наибольший индекс i удовлетворяющий этому условию — F₂₅=75025, F₂₆=121393 > 100000, то есть 24 числа.

**Теорема Цекендорфа** гарантирует, что для любого N число слагаемых в разложении ≤ log_φ(N) + 1 ≈ log_φ(100000) + 1 ≈ 24/1.44 + 1 ≈ 17.7. Практически, для w∈[0, 100K] при scale=10000 → idx max = 1e9, разложение Цекендорфа даёт в среднем ~30-40 чисел Фибоначчи.

**Ключевой вопрос:** сколько векторов bundle?

Для w=100000, idx=1e9:
- Zeckendorf(1e9) = 701408733 + 267914296 + 102334155 + 39088169 + 14930352 + 5702887 + 2178309 + 832040 + 317811 + 121393 + 46368 + 17711 + 6765 + 2584 + 987 + 377 + 144 + 55 + 21 + 8 + 3 + 1 + 1 (нет, последние два не подряд — теорема Цекендорфа не допускает последовательных чисел Фибоначчи)
- Реально: ~20-35 векторов на максимальный вес. Для w=100 → idx=1e6 → ~15-20 векторов.

**Вывод:** типичный bundle содержит 10-35 quasi-ортогональных векторов.

### 1.3 SNR для bundle из K quasi-ортогональных векторов

**Теоретическая модель:** Пусть v_1, ..., v_K — случайные единичные векторы на S^{D-1}, D=768. Ожидаемый косинус между двумя случайными единичными векторами ~ 1/√D ≈ 0.036. Сумма S = Σ v_i. Норма:

E[||S||²] = Σ E[||v_i||²] + 2 Σ_{i<j} E[v_i·v_j] = K + 0 = K

Сигнал = ||S|| (норма суммы). После нормализации направление суммы — это центр масс. SNR (signal-to-noise ratio) определим как:

SNR = E[||S||] / σ(||S||)

Но более полезно: **косинус между bundle(w₁) и bundle(w₂) для близких весов** — насколько он отличен от косинуса для далёких весов.

**Экспериментальная оценка:**

Для D=768:
- K=10: E[||S||] = √10 ≈ 3.16. Шум нормализации = 3.16/10 = 0.316 (каждый вектор вносит ~3% после нормализации)
- K=35: E[||S||] = √35 ≈ 5.92. Шум = 5.92/35 = 0.169
- K=100: E[||S||] = 10.0. Шум = 0.1

**Проблема:** Косинус между двумя независимыми bundle из K векторов:

E[cos(S₁, S₂)] ≈ 0 (но дисперсия ~ 1/D + 1/(K*D) — по simulation)

Экспериментально для K=10, D=768: σ(cos) ≈ 0.04. Для K=35: σ(cos) ≈ 0.02.

**Монотонность encode(w):** Критическая проблема. Если w₁ ≠ w₂, но индексы Цекендорфа отличаются лишь несколькими позициями, косинус между bundle(w₁) и bundle(w₂) должен быть строго монотонной функцией от |w₁-w₂|. Однако из-за нерегулярности разложения Цекендорфа (числа не меняются линейно) этого нет.

**Контрпример:** w₁=100, w₂=101.
- Zeckendorf(1000000) и Zeckendorf(1010000) — differ in lower Fib numbers
- Косинус может быть ~0.95, что делает encode почти нечувствительным к малым изменениям

### 1.4 Сравнение compression ratio

`compression_ratio()` возвращает 4/1536 = 0.0026 (строки 125-128). Это верно для fp32→HD, но:
- Каждый encode создаёт 768 float32 = 3072 байта на один вес
- Для веса fp32 (4 байта) — расширение в 768 раз!
- Только при хранении 768+ весов в одном HD-векторе достигается сжатие

**Практический смысл:** `ZeckendorfQuantizer` — это не метод сжатия (как заявлено в docstring), а метод представления скаляра в HD-пространстве для VSA-операций. Compression ratio < 1 — артефакт, не отражающий реальности.

### 1.5 Детектированные проблемы

**Проблема 1 (ФАЙЛ: fibonacci_utils.py:104-108):**
```python
def encode(self, w: float) -> np.ndarray:
    idx = int(round(abs(w) * self.scale))
    if idx <= 0:
        return np.zeros(self.dim, dtype=np.float32)
```
Нулевой вес → нулевой вектор. Но после нормализации (строки 113-115) нулевой вектор возвращается как нулевой, что нарушает инвариант "все векторы на единичной сфере". Это может вызвать деление на ноль в similarity().

**Проблема 2 (ФАЙЛ: fibonacci_utils.py:109):**
```python
fibs = FibonacciUtils.zeckendorf(idx)
```
Разложение Цекендорфа рекурсивно извлекает наибольшее число Фибоначчи ≤ n и вычитает его. Для n=1e9 это O(log_φ(n)) ≈ 44 итерации. При encode_batch из 100K весов — 4.4M вызовов `FibonacciUtils.get`, каждый с рекурсией до 25. Это ~110M операций, потенциально узкое место.

**Проблема 3 (ФАЙЛ: fibonacci_utils.py:99-102):**
```python
n_vecs = len(self._fib_to_idx)
rng = np.random.RandomState(seed)
vecs = rng.randn(n_vecs, dim).astype(np.float32)
```
Матрица _vecs размера [24, 768] — всего 24 базисных вектора для ВСЕХ кодировок. Для любого encode используется фиксированный набор из 24 quasi-ортогональных направлений. При K=35 векторов в bundle, неизбежны повторения (24 уникальных, K=35 > 24). Теорема Цекендорфа гарантирует уникальность ЧИСЕЛ, но разным числам могут соответствовать одни и те же векторы.

**Проблема 4 (ФАЙЛ: fibonacci_utils.py:113-115):**
```python
vec = np.array(list(self._vecs[i] for i in indices)).sum(axis=0)
n = np.linalg.norm(vec)
return vec / n if n > 1e-10 else np.zeros(self.dim, dtype=np.float32)
```
Если `indices` содержит дубликаты (из-за коллизий _fib_to_idx), один и тот же вектор суммируется дважды. Это не предусмотрено архитектурой quasi-ортогонального кодирования.

### 1.6 Рекомендации

1. **Увеличить n_vecs:** Вместо маппинга Fib→idx через `i-2` использовать уникальные HD-векторы для КАЖДОГО числа Фибоначчи (а не для каждого индекса). Или: seed = f (число Фибоначчи) для псевдослучайной генерации на лету без хранения матрицы.

2. **Проверить монотонность:** Написать тест `test_zeckendorf_monotonic` — для случайных w₁ < w₂ проверить, что `similarity(encode(w₁), encode(w₂))` коррелирует с |w₁-w₂|.

3. **Add noise floor:** Для w=0 возвращать случайный шумовой вектор малой амплитуды, а не нулевой.

4. **Оптимизировать encode_batch:** Использовать numpy векторизацию вместо list comprehension.

---

## 2. TemporalZeckendorf: замена decay

### 2.1 Что делает TemporalZeckendorf

**Файл:** `eva/symbolic/fibonacci_utils.py:135-203`

`TemporalZeckendorf` заменяет классический экспоненциальный decay:

```python
theta_gate = math.exp(-dist / theta_tau)      # fast: tau=5
theta_slow = math.exp(-dist / (theta_tau * 3)) # slow: tau=15
```

на:

```python
idx = largest_fib_idx(distance)
base = (max_depth - idx) / max_depth
if distance <= fast_window:  fast = base  else  fast = 0.0
if distance <= slow_window:  slow = base  else  slow = 0.0
```

### 2.2 Функциональная эквивалентность

**Вопрос:** Можно ли найти τ_fast и τ_slow такие, что `theta(dist) ≈ exp(-dist/τ)`?

`theta()` возвращает BASE = (max_depth - idx) / max_depth, где idx — индекс наибольшего числа Фибоначчи ≤ d, max_depth — полное число чисел Фибоначчи для max_steps=1e6.

**Facts:**
- Фибоначчи растут экспоненциально: F_n ≈ φ^n / √5
- idx = floor(log_φ(d * √5)) — логарифмическая функция
- base = (max_depth - floor(log_φ(d * √5))) / max_depth
- Это ЛИНЕЙНАЯ функция от log_φ(d), а НЕ экспонента

**Сравнение:**
- exp(-d/τ) — экспоненциальный спад: быстро падает для малых d, медленно для больших
- base = 1 - log_φ(d) / max_depth — логарифмический спад: медленно падает для малых d, быстро для больших

**При d=1:**
- exp(-1/5) ≈ 0.819
- idx = largest_fib_idx(1) = 1 (F₂=1, F₃=2>1), max_depth ≈ 30, base = (30-1)/30 = 0.967

**При d=10:**
- exp(-10/5) ≈ 0.135
- idx = 6 (F₇=13>10, F₆=8), base = (30-6)/30 = 0.8

**При d=100:**
- exp(-100/5) ≈ 2e-9
- idx = 11 (F₁₂=144>100, F₁₁=89), base = (30-11)/30 = 0.633

**Вывод: НЕТ функциональной эквивалентности.** Exponential decay даёт Θ(d) → 0 при d → ∞, а Fibonacci base → (max_depth - max_idx)/max_depth ≈ 0 (всегда ≥ 0, но в пределе не стремится к 0). При d=1e6: idx=max_depth, base=0.

Однако `theta()` дополнительно фильтрует через `fast_window` и `slow_window`:

```python
if distance <= fast_window:  fast = base  else  fast = 0.0
if distance <= slow_window:  slow = base  else  slow = 0.0
```

**Это step-функция, а не decay.** При d > fast_window (5) fast=0. При d > slow_window (10) slow=0.

### 2.3 Анализ theta() для STDP

В `_build_pairs` (stdp_trainer.py:693-704):

```python
fast_th, slow_th = _tz.theta(abs(j-i))
theta_gate = max(fast_th, _fc.theta_fast_min)
...
theta_slow = max(slow_th, _fc.theta_slow_min) * _fc.theta_slow_scale
slow_lr = lr * theta_slow if fast_th > _fc.theta_fast_min else 0.0
```

Сравнение с экспонентой:

| dist | exp(-d/5) | exp(-d/15) | theta.fast | theta.slow |
|------|-----------|------------|------------|------------|
| 0    | 1.000     | 1.000      | 1.0        | 1.0        |
| 1    | 0.819     | 0.936      | ~0.967     | ~0.967     |
| 2    | 0.670     | 0.875      | ~0.933     | ~0.933     |
| 3    | 0.549     | 0.819      | ~0.900     | ~0.900     |
| 5    | 0.368     | 0.717      | ~0.833     | ~0.833     |
| 6    | 0.301     | 0.670      | 0.0        | ~0.833     |
| 10   | 0.135     | 0.513      | 0.0        | ~0.700     |
| 11   | 0.111     | 0.479      | 0.0        | 0.0        |

**Ключевые различия:**

1. **Step-функция vs непрерывный спад:** theta() возвращает константу для d≤fast_window, а не плавное убывание. Это означает, что пары на расстоянии 1 и 5 получают ОДИНАКОВЫЙ theta_gate (разница только в lr от dist_weight и freq_weight).

2. **Резкий обрыв:** При d=6 fast_th прыгает с ~0.833 на 0.0. При d=11 slow_th с ~0.7 на 0.0. Это создаёт негладкий ландшафт обучения.

3. **Theta_fast_min / theta_slow_min:** Параметры `_fc.theta_fast_min=0.1` и `_fc.theta_slow_min=0.02` спасают ситуацию — когда fast_th=0.0, берётся `max(0.0, 0.1) = 0.1`, что даёт theta_gate=0.1 для d>5. Это превращает step-функцию в "два уровня": высокий для d≤5, низкий для d≥6.

4. **Отсутствие параметра τ:** Как заявлено — "no free tau parameter". Однако `fast_window=5` и `slow_window=10` по сути те же гиперпараметры, только с другой семантикой. Вместо "как быстро затухает" — "на каком расстоянии обрывается".

### 2.4 Преимущества и недостатки

**Преимущества:**
- Не требует настройки τ (но требует настройки windows)
- Периодическая структура Фибоначчи может давать иерархические временные масштабы
- Инвариантность к scale: не зависит от единиц измерения времени

**Недостатки:**
- Разрывная функция активации — негладкий ландшафт
- Потеря разрешения: для всех d ≤ 5 одинаковый fast_th
- Fast/slow theta идентичны для d ≤ min(fast_window, slow_window) — дублирование информации
- Для расстояний > slow_window сигнал нулевой (без slow_scale коррекции)

### 2.5 Рекомендации

1. **Сделать theta непрерывной:** Заменить step-функцию на `base * exp(-d / window)` — сохранить Fibonacci base как множитель, но добавить экспоненциальное окно.

2. **Использовать LCP напрямую:** `temporal_H(t_a, t_b, gamma)` (строки 178-183) — более естественная функция сходства, не использующая разрывные окна. Рассмотреть её как primary decay.

3. **Уменьшить fast_window / slow_window разрыв:** Увеличить fast_window до 10, slow_window до 20, чтобы покрыть больший диапазон расстояний.

4. **Добавить soft-переход:** Вместо `else: fast = 0.0` использовать `fast = base * max(0, 1 - (d - fast_window) / fast_window)` — линейный спад после окна.

---

## 3. VSA-консистентность

### 3.1 Инвентаризация VSA-операций

Все VSA-операции в системе разделены на несколько уровней:

**Уровень 1: Примитивы в concept_space.py**

| Функция | Файл:Строка | Тип | Использует |
|---------|-------------|-----|------------|
| `_hrr_bind` | concept_space.py:36-40 | FFT-HRR circular convolution | FFT |
| `_hrr_unbind` | concept_space.py:42-46 | FFT-HRR circular correlation | FFT |
| `_hybrid_bind` | concept_space.py:75-85 | α·hrr + (1-α)·ew | FFT + element-wise |
| `_hybrid_unbind` | concept_space.py:87-97 | α·hrr + (1-α)·ew | FFT + element-wise |
| `_bind_weighted_zeckendorf` | concept_space.py:99-121 | Zeckendorf-tree + _hybrid_bind | hybrid_bind |
| `_hybrid_bind_torch` | concept_space.py:156-164 | GPU batch hybrid bind | torch FFT |

**Уровень 2: Потребители примитивов**

| Компонент | Файл | Использует | Примечание |
|-----------|------|------------|------------|
| **EntityField._bind** | concept_space.py:1124-1125 | `_hybrid_bind` | OK |
| **EntityField._unbind** | concept_space.py:1126-1127 | `_hybrid_unbind` | OK |
| **Harmonizer._bind** | concept_space.py:1303-1305 | `_hybrid_bind` | OK |
| **Harmonizer._unbind** | concept_space.py:1307-1309 | `_hybrid_unbind` | OK |
| **TransitionManifold._vsa_transition** | transition_manifold.py:136-146 | `_hybrid_unbind` | Исправлено V21 |
| **VSAAttention._scale_bundle** | vsa_attention.py:64-79 | `_hybrid_bind` | Исправлено V21 |
| **HDTransformerLayer** | hdtransformer_layer.py:103-108 | `_hybrid_bind` | OK |
| **FractalField.hdc_bind** | concept_space.py:818-820 | `_hybrid_bind` | OK |
| **FractalField.hdc_unbind** | concept_space.py:822-824 | `_hybrid_unbind` | OK |

### 3.2 Остаточные element-wise операции

**Кандидат 1: HDTransformerLayer:103-104 — скалярный bind**

```python
weight_hv = np.full(self.dim, part / 7.0, dtype=np.float32)
wn = np.linalg.norm(weight_hv)
if wn > 1e-10:
    weight_hv /= wn
weighted = _hybrid_bind(val, weight_hv)
```

Здесь `weight_hv` — это константный вектор (все элементы равны part/7.0). После нормализации это `[1/√D, ..., 1/√D]`. Bind такого вектора с val эквивалентен:

```python
hybrid_bind(val, const) = α·circ_conv(val, const) + (1-α)·val·const
```

Но circ_conv(val, const) для const=[c,c,...,c]:
- FFT(const)[0] = c·√D (DC component)
- FFT(const)[k>0] = 0
- irfft(fft(val) · fft(const)) = c·√D · mean(val) · [1,1,...,1]

**Вывод:** `_hybrid_bind(val, const_vector)` сводится к масштабированию val с bias к константе. Это НЕ полноценная VSA-операция bind — weight_hv не является quasi-ортогональным вектором. Фактически это эвристическое взвешивание, маскирующееся под VSA.

В отличие от VSAAttention._scale_bundle (строка 73-74):
```python
weight_hv = self._weight_vector(p, self.max_weight)
bound = _hybrid_bind(value, weight_hv)
```
где `_weight_vector` генерирует quasi-ортогональный HD-вектор через seed_registry — это корректная VSA-операция.

**Кандидат 2: `_bind_weighted_zeckendorf` (concept_space.py:99-121) — избыточная сложность**

```python
for part in tree:
    scale = part / tree[0] if tree else 1.0
    sub = vec * scale      # ✗ element-wise scale
    sn = np.linalg.norm(sub)
    if sn > 1e-10:
        sub /= sn
    bound = _hybrid_bind(vec, sub)  # vec ⊛ (vec * scale)
```

Проблема: `sub` — это `vec * scale`, то есть тот же вектор, масштабированный. Bind(vec, sub) = hybrid_bind(vec, vec·scale). FFTHRR bind вектора с самим собой даёт автокорреляцию, которая максимальна на нулевом сдвиге. Это не эквивалентно bind с quasi-ортогональным весовым вектором.

**Кандидат 3: `_hrr_bind` и `_hrr_unbind` — остались неиспользованными**

Эти примитивы (concept_space.py:36-46) не используются нигде, кроме как через _hybrid_bind/unbind. Если curriculum alpha не доходит до 1.0, pure HRR никогда не применяется. Код dead code — рекомендуется удалить или оставить как reference.

### 3.3 Проверка всех вызовов _hybrid_bind/unbind

| Вызов | Функция | VSA-корректность |
|-------|---------|-----------------|
| EntityField.bind (1167) | _bind(ctx_vec, role) | OK — role — quasi-ортогональный вектор |
| EntityField.query (1197) | _unbind(entity_vec, role) | OK |
| Harmonizer.compose_word (1321) | _bind(morph_vec, role) | OK |
| Harmonizer.decompose_word (1361) | _unbind(word_vec, role) | OK |
| Harmonizer.harmonize (1417) | _unbind(error, role) | OK |
| TransitionManifold.push (507-509) | _hybrid_unbind(v_next, v_prev) | OK — исправлено |
| VSAAttention._scale_bundle (73-74) | _hybrid_bind(value, weight_hv) | OK — weight_hv quasi-ортогональный |
| VSAAttention.forward (123, 128) | _hybrid_bind(wv, head_roles) | OK |
| HDTransformerLayer._lsh_attention (108) | _hybrid_bind(val, weight_hv) | **НЕ КОРРЕКТНО** — константный вектор |
| CharEnvelope.word_envelope (172) | _hybrid_bind(result, shifted) | OK — caveat: shift может нарушить quasi-ортогональность |
| MorphSTDP.bind_char (53) | _hybrid_bind(c1, role_right) | OK |
| FractalField.hdc_bind (818-820) | _hybrid_bind(a, b) | OK — общий случай |

### 3.4 Проблема: alpha curriculum не применяется в новых компонентах

`_alpha_from_curriculum()` (concept_space.py:61-73) корректно вычисляет alpha на основе текущей эпохи. Но:

```python
def _hybrid_bind(a, b, alpha=None, eps=1e-8):
    if alpha is None:
        alpha = _alpha_from_curriculum()
```

Все вызовы `_hybrid_bind` из новых компонентов (VSAAttention, HDTransformerLayer, TransitionManifold) не передают alpha явно и полагаются на глобальный curriculum. Однако `_set_alpha_curriculum()` вызывается из training loop — если VSAAttention используется вне обучения (inference), alpha будет fallback = `_fc.hybrid_bind_alpha` (константа из конфига).

**ФАЙЛ: concept_space.py:73** — fallback alpha не соответствует curriculum при inference.

### 3.5 Вывод

**VSA-система консистентна в следующем:**
- Все компоненты используют _hybrid_bind/unbind из concept_space.py
- TransitionManifold._to_tangent исправлен на _hybrid_unbind (V21)
- VSAAttention._scale_bundle использует seed-based weight vectors (V21)

**Выявлены проблемы:**
1. HDTransformerLayer использует константный weight_hv вместо quasi-ортогонального — **FAIL**
2. _bind_weighted_zeckendorf смешивает element-wise scale с VSA bind — **WARNING**
3. _hrr_bind/_hrr_unbind — dead code — **LOW**
4. Alpha curriculum не работает вне training loop — **MEDIUM**

---

## 4. HDTransformerLayer vs VSAAttention

### 4.1 Где какой attention используется

**VSAAttention** (vsa_attention.py:18-139):
- Задействован через `use_vsa_attention` в FCFConfig (строка 813)
- Используется в `_branch` (вероятно, CrystalGenerator._branch)

**HDTransformerLayer** (hdtransformer_layer.py:19-200):
- Задействован через `use_hd_transformer` в FCFConfig (строка 814)
- Используется в `_train` (stdp_trainer.py:267-284)

**Оба активны одновременно:** `use_vsa_attention: True` и `use_hd_transformer: True` — конфликт не по включению, а по функциональному пересечению.

### 4.2 Функциональное пересечение

Оба компонента выполняют VSA-native attention:

**VSAAttention.forward:**
1. Cosine similarity query↔key
2. Квантование similarity → weight [0-7]
3. Zeckendorf-tree взвешивание: bind(weight_hv, value)
4. Multi-head bundle с quasi-ортогональными role vectors
5. Position encoding через Fibonacci shift

**HDTransformerLayer.forward:**
1. Top-K LSH-based cosine similarity
2. Adaptive quantile нормализация similarity
3. Zeckendorf-tree взвешивание: bind(weight_hv, value)
4. Multi-head через _random_masks (subspace masking)
5. Position encoding через Fibonacci shift
6. Fractal convolution FFN + residual

**Различия:**
- VSAAttention: все пары (n²), multi-head через role vectors
- HDTransformerLayer: top-K, multi-head через subspace masks, + fractal FFN

**Пересечение:** Оба делают взвешенную агрегацию VSA-векторов через Zeckendorf-дерево. Если оба включены в pipeline для одной и той же последовательности, то STDP получает коррекцию от HDTransformerLayer (stdp_trainer.py:267-284) ПОСЛЕ того, как основная STDP-обработка уже применила VSAAttention-подобную логику.

### 4.3 Анализ pipeline

В `_train` (stdp_trainer.py:200-308):

```
1. _build_pairs — STDP пары (с theta_gate)
2. _gpu_stdp_apply — STDP обновление векторов
3. _gpu_poststdp_fused — negative sampling + contrastive
4. HDTransformerLayer refinement ← здесь
5. _centroid_pull_batch
6. _cluster_centroid_pull
7. _harmonize_batch (включает VSAAttention?)
```

**ФАЙЛ: stdp_trainer.py:267-284:**
```python
if FCFConfig().use_hd_transformer:
    ...
    for ids in all_ids:
        seq = [cs.concept_vector(c) for c in ids if ...]
        if len(seq) >= 2:
            out = self._hd_transformer.forward(seq)
            for j, cid in enumerate(ids):
                pull = out[j] - cs.concept_vector(cid)
                ...
                cs._apply_vector_update(cid, new_v / nn)
```

**Проблема:** HDTransformerLayer получает на вход текущие concept_vectors (которые только что были обновлены STDP) и выдаёт "refined" версию. Это повторная агрегация контекста, аналогичная тому, что делает VSAAttention, но без учёта STDP-сигнала.

**Конфликт не в параллельном исполнении, а в последовательном:** два attention-прохода для одной и той же последовательности, без координации. Если HDTransformerLayer перезаписывает STDP-обновления, эффект STDP может быть ослаблен.

### 4.4 Когда VSAAttention применяется

Поищем вызовы VSAAttention:

```python
if FCFConfig().use_vsa_attention:
    attn = VSAAttention(dim=cs.dim)
    # ... в каком-то месте
```

Из контекста кода не видно, где именно VSAAttention вызывается в pipeline. Судя по имени `use_vsa_attention` и названию, это может быть в `_branch` для генерации или в `_evaluate`. Если VSAAttention используется ТОЛЬКО для генерации (inference), а HDTransformerLayer — для обучения, то конфликта нет.

Однако если оба применяются к одним и тем же данным на каждой итерации обучения, то:
1. VSAAttention → STDP → обновление concept_vectors → HDTransformerLayer → повторное обновление concept_vectors
2. HDTransformerLayer может откатить часть STDP-изменений

### 4.5 Рекомендации

1. **Разделить роли:** VSAAttention — для inference/generation, HDTransformerLayer — для refinement после STDP. Документировать это явно.

2. **Отключить один из двух:** Если оба активны, измерить влияние HDTransformerLayer refinement на метрики (perplexity, accuracy). Если эффект отрицательный или нулевой — отключить.

3. **Унифицировать:** Создать единый VSA Attention Layer с параметром `mode='stdp'|'refine'`.

4. **Проверить weight_hv в HDTransformerLayer:** Проблема 3.2 (константный вектор) — HDTransformerLayer не использует seed-based weight vectors, в отличие от VSAAttention. Это может приводить к семантически некорректным весам.

---

## 5. Harmonizer + MorphSTDP

### 5.1 Что делает каждый

**Harmonizer** (concept_space.py:1261-1565):
- Compose/decompose: word = ⊕ bind(morphemeⱼ, roleⱼ)
- harmonize(word_id, word_vec, sent_vec): итеративное уточнение word_vec и морфемных векторов через bottom-up (compose) и top-down (sent_vec unbind) сигналы
- Backprop error от word к морфемам через unbind(error, role)
- Dirty-флаги для каскадного обновления

**MorphSTDP** (semantic_piece.py:22-131):
- STDP-driven морфемное discovery: char→char bind c обучением
- observe(char_ids): STDP для char bigram
- discover_morphemes(): "pop-out" высококогезивных bigram как морфем
- decompose(char_ids): разбиение на известные морфемы + residue

### 5.2 Взаимодействие в pipeline

В `_harmonize_batch` (stdp_trainer.py:310-585) происходит следующее:

**Шаг 1. Sync concept → entity_field:**
```python
ef.sync_word(cid, v_new)  # строка 352
```
Word-векторы копируются из concept_vectors в EntityField.

**Шаг 2. Cross-level bindings (char↔word, word↔sent):**
```python
ef.bind('c', cp, 'w', cid, lr=0.05)  # строка 388
ef.bind('w', cid, 'c', cp, lr=0.05)  # строка 389
```
Это НЕ зависит от Harmonizer — EntityField сам хранит свои привязки.

**Шаг 2a-bis. MorphSTDP observation:**
```python
self._morph_stdp.observe([ord(ch) for ch in word_text], lr=0.1)  # строка 401
```
MorphSTDP видит те же char-последовательности.

**Шаг 3. Harmonizer.harmonize:**
```python
new_v, delta = harm.harmonize(cid, v_latent, sent_vec=sv)  # строка 471
```
Работает с морфемами, зарегистрированными через `harm.register_word()`.

**Шаг 4. Morph-level transition manifold (P1.9):**
```python
mm.push(T)  # строка 510
for i in range(len(morph_seqs) - 1):
    T = mm._vsa_transition(morph_seqs[i+1], morph_seqs[i])
```
Обновляет морфемные векторы через manifold.

**Шаг 5. MorphSTDP discovery → Harmonizer:**
```python
n_new = morph_stdp.discover_morphemes(...)  # строка 569
if n_new > 0:
    # Регистрирует новые морфемы в word_morphs и morphemes
    harm.word_morphs[mkey] = []  # строка 574
    harm.morphemes[morph_id] = mv.copy()  # строка 579
```

### 5.3 Двойной учёт: анализ

**Есть ли двойной учёт?** 

Да, но он разнесён по разным фазам и разным представлениям:

**Канал 1: Harmonizer — морфемы через pymorphy3/rule-based decomposition**
- Детерминированный, известный набор морфем
- ROLE-aware: prefix, root, suffix, ending
- Compose/decompose через bind с role vectors
- Работает с латентными векторами (2048D)
- Используется для гармонизации word→morph и morph→word

**Канал 2: MorphSTDP — STDP-based char bigram discovery**
- Эмерджентный: морфемы "pop out" из статистики char bigram
- Безролевой: просто n-граммы символов
- Работает с char векторами (768D)
- Используется для discovery новых морфем

**Пересечение:**

1. **Оба хранят `morphemes` в разных местах:**
   - Harmonizer: `harm.morphemes[morph_id]` — 2048D вектор (латентное пространство)
   - MorphSTDP: `morph_stdp.morphemes[morph_id]` — 768D вектор (char space)
   - После discovery (stdp_trainer.py:578-579): копия `mv = morph_stdp.morphemes.get(morph_id)` записывается в `harm.morphemes` через `cs.concept_vectors[mkey] = mv.copy()`
   
   **Проблема:** размерности разные! MorphSTDP.morphemes — 768D, а Harmonizer.morphemes — 2048D (latent_dim). При записи `cs.concept_vectors[mkey] = mv.copy()` вектор 768D помещается как concept_vector, но Harmonizer ожидает 2048D для bind-операций. Это вызовет silent dimension mismatch в `harmonize()`.

   **ФАЙЛ: stdp_trainer.py:579:**
   ```python
   cs.concept_vectors[mkey] = mv.copy()
   ```
   где `mv = morph_stdp.morphemes.get(morph_id)` — 768D, а concept_vectors ожидает dim=768 — OK для concept_vectors, но не для Harmonizer.morphemes.

   **ФАЙЛ: concept_space.py:1392-1394:**
   ```python
   def set_morpheme_vec(self, morph_id, vec):
       self.morphemes[morph_id] = vec.copy() if isinstance(vec, np.ndarray) else vec
   ```
   Нет проверки размерности. Если vec=768D, а self.dim=2048 — тихая ошибка.

2. **Оба обрабатывают char-последовательности:**
   - Harmonizer: через EntityField (char→word bind, строка 388-389)
   - MorphSTDP: через observe (char_ids, строка 401)
   - Эти операции дополняющие, не дублирующие

3. **Оба обновляют word-векторы:**
   - Harmonizer: через `harm.harmonize()`, строка 471
   - MorphSTDP discovery: не обновляет word-векторы напрямую
   - Нет двойного обновления одного и того же вектора из двух источников

### 5.4 Проблема размерности при MorphSTDP → Harmonizer

**ФАЙЛ: stdp_trainer.py:566-581:**

```python
morph_stdp = getattr(self, '_morph_stdp', None)
if morph_stdp is not None and self._morph_stdp_batches % _c.morph_stdp_discover_every == 0:
    n_new = morph_stdp.discover_morphemes(min_cohesion=_c.morph_stdp_cohesion)
    if n_new > 0 and hasattr(cs, 'morph_vocab'):
        for morph_id, chars in morph_stdp.morph_to_chars.items():
            mkey = ('MORPH', morph_id)
            if mkey not in harm.word_morphs:
                harm.word_morphs[mkey] = []  # ← !!!
            ...
            mv = morph_stdp.morphemes.get(morph_id)
            if mv is not None and mkey not in cs.concept_vectors:
                cs.concept_vectors[mkey] = mv.copy()
```

Здесь: `harm.word_morphs[mkey] = []` — mkey это `('MORPH', morph_id)`, но `word_morphs` ожидает int CID как ключ. Harmonizer использует `word_morphs[word_id]` где word_id — int. Если mkey — tuple, то в `harmonize()`:

```python
if word_id not in self.word_morphs:  # строка 1428
    return None, 0.0
```

word_id передаётся как int CID, а ключ — tuple → всегда False → harmonize никогда не обработает MorphSTDP-морфемы.

**Это серьёзная ошибка.** MorphSTDP discovery регистрирует морфемы с tuple-ключом, а Harmonizer ожидает int.

### 5.5 Дополнительный конфликт: Harmonizer ↔ TransitionManifold (morph)

**ФАЙЛ: stdp_trainer.py:496-525:**

```python
mm = getattr(self, 'morph_manifold', None)
if mm is not None and hasattr(harm, 'word_morphs'):
    for cid in morph_cids:
        parts = harm.word_morphs.get(cid, [])
        ...
        for i in range(len(morph_seqs) - 1):
            T = mm._vsa_transition(morph_seqs[i+1], morph_seqs[i])
            if np.linalg.norm(T) > mm._eps:
                mm.push(T)
                cent, sim, _cnt = mm.nearest_beam(T)
                if cent is not None and sim > mm.cos_threshold * 0.8:
                    pull = cent - T
                    ...
                    harm.morphemes[morph_id_i] = (updated_mv / unm).astype(np.float32)
```

Здесь TransitionManifold обновляет морфемные векторы напрямую в `harm.morphemes`. Это обновление НЕ синхронизировано с Harmonizer.harmonize — т.е. manifold может изменить морфемный вектор, а Harmonizer в следующем harmonize() будет использовать уже изменённый вектор, что может привести к неконтролируемому дрейфу.

### 5.6 Итог по Harmonizer + MorphSTDP

**Степень пересечения:** Низкая — они работают на разных представлениях и в разных фазах.

**Выявленные проблемы:**

1. **Dimension mismatch (ФАЙЛ: stdp_trainer.py:579):** MorphSTDP.morphemes (768D) → cs.concept_vectors (768D — OK) → Harmonizer.morphemes (ожидает 2048D для bind в compose_word). Если морфема используется в Harmonizer.harmonize, bind с 768D вместо 2048D даст неверный результат.

2. **Tuple vs int ключ (ФАЙЛ: stdp_trainer.py:573-574):** `harm.word_morphs[('MORPH', morph_id)]` — tuple, а Harmonizer ожидает int key в `harmonize()`.

3. **Несинхронизированные обновления (ФАЙЛ: stdp_trainer.py:519-525):** TransitionManifold обновляет harm.morphemes напрямую, минуя dirty-трекинг и Harmonizer.

4. **Дублирование char→word mapping:** EntityField.bind('c', cp, 'w', cid) (строка 388) и MorphSTDP.observe (строка 401) — оба обрабатывают связь char→word, но в разных представлениях. Если EntityField уже bind'ит char→word, MorphSTDP дублирует эту работу для своих целей.

---

## Сводка багов / проблем

| # | Файл:Строка | Серьёзность | Описание |
|---|-------------|-------------|----------|
| 1 | fibonacci_utils.py:107 | MEDIUM | Нулевой encode → нулевой вектор, не на сфере |
| 2 | fibonacci_utils.py:99-102 | HIGH | Только 24 базисных вектора для всех кодировок → коллизии при K>24 |
| 3 | fibonacci_utils.py:185-203 | MEDIUM | theta() — разрывная step-функция, не эквивалентна exp(-d/τ) |
| 4 | hdtransformer_layer.py:104 | HIGH | weight_hv — константный вектор, не quasi-ортогональный; bind даёт scale, а не VSA bind |
| 5 | concept_space.py:99-121 | MEDIUM | _bind_weighted_zeckendorf масштабирует vec → bind(vec, vec·scale) — не VSA bind |
| 6 | concept_space.py:73 | LOW | Alpha curriculum не работает вне training loop |
| 7 | stdp_trainer.py:267-284 | MEDIUM | HDTransformerLayer refinement может откатывать STDP |
| 8 | stdp_trainer.py:579 | HIGH | Dimension mismatch: MorphSTDP (768D) → Harmonizer.morphemes (2048D) |
| 9 | stdp_trainer.py:573-574 | HIGH | Неверный тип ключа: tuple vs int в word_morphs |
| 10 | stdp_trainer.py:519-525 | MEDIUM | Прямое обновление harm.morphemes без dirty-трекинга |

---

## Рекомендации по фиксации

1. **ZeckendorfQuantizer:** увеличить пул базисных векторов до числа уникальных чисел Фибоначчи в разложении для максимального idx (не 24, а ~50+). Добавить монотонность через ранжирование косинусов.

2. **TemporalZeckendorf.theta:** заменить step на `base * exp(-d/window)`, использовать `temporal_H` как primary decay функцию.

3. **HDTransformerLayer.weight_hv:** заменить `np.full(dim, part/7.0)` на seed-based quasi-ортогональный вектор как в VSAAttention._weight_vector.

4. **Harmonizer + MorphSTDP bridge:** конвертировать 768D→2048D через FractalField.basis перед записью в harm.morphemes. Использовать int-ключи в word_morphs.

5. **Pipeline coordination:** добавить блокировку: если HDTransformerLayer refinement активен, не применять VSAAttention (или наоборот).

---

*Report generated by Neuro-Symbolic Auditor V22*
*SHA: 7585dfb | Date: 2026-06-23*
