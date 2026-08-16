# FCF / EVA — Fractal Cognitive Field

**Экспериментальная языковая архитектура без трансформеров и без обратного распространения.**

Без attention, без backprop, без градиентного спуска. Память — это
векторное поле (VSA-суперпозиция), а не матрица. Обучение — локальные
правила (STDP, негативная выборка, контрастив), как на нейроморфном
чипе. Все числовые константы выведены из единой структурной константы
λ_d — ни одной эмпирической.

```
  ┌───────────────────────────────────────────────────────────────┐
  │                     FCF / EVA                                 │
  │   VSA-поле · STDP-обучение · λ_d-иерархия · без backprop      │
  │   нет attention · нет обратного распространения · без softmax │
  └───────────────────────────────────────────────────────────────┘
```

---

## Оглавление

1. [Три идеи](#1-три-идеи)
2. [Обзор потока данных](#2-обзор-потока-данных)
3. [Токенизация](#3-токенизация)
4. [Концепт-пространство](#4-концепт-пространство)
5. [Обучение без градиентов (STDP)](#5-обучение-без-градиентов-stdp)
6. [Коллокации и синтаксис](#6-коллокации-и-синтаксис)
7. [Морфология](#7-морфология)
8. [Генерация](#8-генерация)
9. [Нейромодуляция](#9-нейромодуляция)
10. [λ_d-иерархия в числах](#10-λ_d-иерархия-в-числах)
11. [Конфигурация](#11-конфигурация)
12. [Инференс-интерфейс](#12-инференс-интерфейс)
13. [Структура репозитория](#13-структура-репозитория)
14. [Статус](#14-статус)
15. [Наблюдения из обучения](#15-наблюдения-из-обучения)

---

## 1. Три идеи

### 1.1. Память — поле, не матрица

Каждый концепт — пара `(латентный_код, вектор)`:

```
z ∈ ℝ^{2048}    — латентный код (три подпространства, λ_d-взвешенные)
v ∈ 𝕊^{767}     — единичный вектор на гиперсфере (768D)
v = normalize(z · B)   — проекция через ортонормированный базис
```

`EntityField` — рекурсивное семантическое поле char↔morph↔word↔sent↔para
через VSA bind/unbind. Информация «свёрнута» в поле и обновляется
локально — контекстная история не хранится матрицей.

### 1.2. Обучение — локальные правила, не градиенты

Архитектура спроектирована так, как если бы обратное распространение
было запрещено, а единственным доступным вычислительным примитивом был
мемристорный кроссбар. Пять локальных правил:

- **STDP** — спайк-тайминговая пластичность по парам концептов;
- **негативная выборка** — раздвигание чужих концептов;
- **контрастив** — hard-негативы с cooc-масками;
- **морфемная гармонизация** — сборка слов из морфем;
- **TemporalZeckendorf** — временное затухание через разложение на
  непоследовательные числа Фибоначчи (заменило θ-decay).

Ни одного глобального градиента. FCF принципиально совместима с
нейроморфными чипами (Intel Loihi 2, IBM TrueNorth, НИИСИ, «Модуль»,
Курчатовский институт).

### 1.3. Структура — λ_d, не тюнинг

> **Каждый числовой коэффициент в архитектуре — weight, decay, window,
> threshold, learning rate — является позицией в иерархии λ_d^{−k},
> числом Фибоначчи F^(d)_n, или простой композицией этих двух.**

λ_d — действительный корень уравнения x^d = x^{d-1} + … + 1:

| d | λ_d | Название |
|---|-----|----------|
| 2 | 1.618… | φ — золотое сечение |
| 3 | 1.839… | Трибоначчи-константа |
| 4 | 1.928… | Тетраначчи-константа |
| d→∞ | 2.0 | Двоичный предел |

Если эмпирический коэффициент не имеет λ_d-вывода — это не «шум», а
указание на пропущенный структурный элемент («метод лакун»). Все 60+
коэффициентов выведены, 0 эмпирических констант.

**Дезайн-слоган:** FCF не «хранит и распространяет», а связывает,
разводит и собирает — bind/unbind в поле, пары в STDP, морфемы в слова.

---

## 2. Обзор потока данных

### 2.1 Обучение (train_full.py)

```
[token ids]  (WB v65536 BPE через sp_compat / morpheme-v1)
     ↓
[ConceptSpace: vocab_size × 768D сфера, 2048D латентный код]
     ↓
  batch × 100 строк, context_window=4:
     ↓
[STDP-обучение]  пары → STDP (GPU) → негативная выборка →
     контрастив → HDTransformerLayer-уточнение → centroid pull
     → решётка + HDC-память → CollocationMatrix (L2/L3) → гармонизация
     ↓
[SyntaxLattice: PPMI n-граммная решётка + HDC-резерв]
     ↓
каждые 10K строк: чекпойнт (state.pkl + meta.json) + тестовая генерация
```

### 2.2 Генерация (crystal_generator.py)

```
[seed-концепт]
     ↓
[beam search (width=5)]  ← 6 сигналов, RRF-взвешенные:
     graph 0.420 · syntax 0.259 · hdc 0.160 · vector 0.099 · prior 0.061
     ↓
[BMSSP: мультиисточниковый BFS по PPMI-графу] · [n-граммная решётка]
  · [HDC-память] · [векторная близость через секторный LSH] ·
  [CollocationMatrix] · [transition-manifold beam_score]
     ↓
[VSA-внимание (Zeckendorf-weighted bind)]
     ↓
[temperature softmax + top-p (nucleus)] → next token
```

Интерфейс стека:

```python
cs = ConceptSpace(vocab_size=sp.vocab_size(), dim=768)
lattice = SyntaxLattice(...)
gen = CrystalGenerator(cs, lattice, ...)
ids = gen.generate(seed_cid, max_words=30)
```

---

## 3. Токенизация

Токенизатор выбирается `EnvironmentResolver`/`load_piece_model` по
приоритету:

| Файл | Механизм | Vocab | Статус |
|---|---|---|---|
| `tokenizer.json` | HF ByteLevel BPE (WideBind v65536) | 65 536 | **основной** |
| `bpe_morph.model` | SentencePiece BPE | 65 536–256K | legacy |
| `morpheme_65k.json` | MorphemeTokenizer (`morpheme-v1`) | 65 536 | эксперимент |

### 3.1 SPCompat — адаптер HF BPE под API SentencePiece

FCF ожидает от токенизатора интерфейс SentencePieceProcessor.
`SPCompatTokenizer` (eva/symbolic/sp_compat.py) оборачивает HF ByteLevel
BPE: encode/decode, IdToPiece (с эмуляцией `▁` для ведущего пробела),
PieceToId, vocab_size, спецтокены 0-3 = `<|pad|>/<|bos|>/<|eos|>/<|unk|>`.
`load_piece_model` диспатчит по расширению: `tokenizer.json` → HF,
`format=morpheme-v1` → MorphemeTokenizer, `*.model` → sentencepiece.

### 3.2 Трёхуровневый MorphemeTokenizer (эксперимент)

Иерархическое «сито»: символы → морфемы → слова, где слово принимается
только если целиком собирается из словарных морфем:

```
id-пространство (65 536):
  0-3      спецтокены
  4-403    символы (топ-частотные + гарантированный математический набор)
  404-…    морфемы (до 30 000, min_freq≥3)
  …        слова (до 35 132, только валидные сборки)
```

Покрытие корпуса: слов как целое 44.5% употреблений, морфем 96.7%.
Сравнение (токенов/1k символов): WB-v65536 RU 246 / MATH 457;
MORPHEME-3lvl RU 545 / MATH 999. Маркер разбора `\u037E` во входе
игнорируется. Подробности — в `docs/TRAINING_JOURNAL.md`.

---

## 4. Концепт-пространство

`ConceptSpace` (eva/symbolic/concept_space.py) — 146K+ концептов на
768D-гиперсфере, латентные коды z∈ℝ²⁰⁴⁸ в `FractalField`.

### 4.1 FractalField

Код разделён на три подпространства с весами `φ² : φ : 1` (обобщение —
`λ_d² : λ_d : 1`):

```
z_c   содержание (медленно)     φ²
z_a   контекст (быстро)         φ
z_m   морфология (средне)       1
```

Вектор — каноническое представление; fractal-код пересчитывается как
проекция `v @ basis.T`. Обновления идут с масками подпространств
(CPU и батч-GPU), clamp сдвига `max_shift=0.5`.

### 4.2 EntityField

Рекурсивное семантическое поле char↔morph↔word↔sent↔para:

- VSA-операции: **bind** = FFT-HRR circular convolution (гибрид α=0.7
  с билинейным произведением), **unbind** = circular correlation,
  **permute** = циклический сдвиг;
- Векторы хранятся float16, выдаются float32;
- `EntityField.ensure` — случайный единичный вектор через центральный
  RNG-реестр;
- **learned fields** (`--learned-fields --field-bits 512`) — обучаемые
  поля (512 бит, field_gate=0.2) поверх статических.

### 4.3 Стабилизация

- **Латеральное торможение** — риманов градиент `sim·v − v_win`
  (топ-победитель отталкивает похожих);
- **Гомеостаз** — `concept_usage` EMA + `homeostatic_boost` (редкие
  концепты получают буст);
- **Групповая алгебра ℤ₈^d** — bind = convolution on group (FFT),
  permute = циклический сдвиг, bundle = суперпозиция;
- **Динамическая размерность** — grow/prune при насыщении/простаивании.

Сохранение: JSON + бинарный `.codes.npz` (коды, Harmonizer, EntityField)
с атомарным replace.

---

## 5. Обучение без градиентов (STDP)

`STDPTrainer` (eva/symbolic/stdp_trainer.py) — весь контур обучения.

### 5.1 Пары

Мета-колонки пары: I, J, PMI, DW (dist_weight), FW (freq_weight),
FIELD_W, SLOW, PREV_CID, NEXT_CID, ANTONYM.

```
lr = base_lr · freq_w · pmi_w · field_w
     × гормональная модуляция × θ-гейт
dist_weight = exp(−dist/2)
freq_weight = 1/(1+log(max_f)·scale)
```

- окно `context_window=4`, skip-2-граммы, полевое перекрытие;
- fast+slow STDP-пары через TemporalZeckendorf;
- антоним = отталкивание ×(−2) (словник `data/antonyms.json` +
  хардкод-фолбэк, перечитывается каждые 100 батчей).

### 5.2 Негативная выборка и контрастив (GPU)

```
негативная выборка: порог sim>0.1, neg_lr = avg_elr · 0.3 · ratio · (1+2·CE·field_gate)
контрастив: top-2000 hard-негативов, cooc-маски,
            field-aware cross-field репульсия
```

Антоним-пара в GPU-ядре также отталкивается ×(−2); scatter_add-фьюжн;
EMA ошибки предсказания концепта; `torch.compile` на Volta+/≥3GB.

### 5.3 Оценка

`_evaluate` (stdp_trainer.py): perplexity, vec_perplexity, acc@1,
vacc@1 на val-корпусе.

---

## 6. Коллокации и синтаксис

### 6.1 CollocationMatrix (branch_network.py)

```
colloc[s][t] = (1−β)·λ_d-prior·damping + β·PMI
β = (λ_d−1)/λ_d
prior = λ_d^(−α·dist·capacity),  damping = λ_d^(−decay)
```

- 4 уровня с λ_d (d=1..4), ёмкости из обобщённого ряда Фибоначчи;
- **Уровень 2 — ключи по cid** (STDP-safe: ключ не зависит от
  меняющегося вектора); уровни 1/3/4 — хэш-ключи от векторов;
- наблюдаются в `train_batch` (уровни 2 и 3) — это и есть
  метрики `colloc_l2` / `colloc_l3` в логе обучения;
- гормональное обучение при генерации (crystal_generator.py).

### 6.2 SyntaxLattice (syntax_lattice.py)

PPMI n-граммная решётка + HDC-резерв при нехватке данных, с
AMI-коррекцией. Потокобезопасна (`threading.Lock`, дефолт-дикты).

### 6.3 Transition Manifold (transition_manifold.py)

Паутина переходов: `unbind(B, A)` → кольцевой буфер → VSA-кластеризация
в «лучи» → `beam_score` при генерации (аналог residual stream в VSA).
Буфер ёмкости F₂₁=10946, cos-порог 0.8.

---

## 7. Морфология

### 7.1 Разметка (eva/morph.py)

`decompose_word` — pymorphy3 (primary) + rule-based фолбэк
(префикс+основа+окончание). Разделитель морфем **SEP = '\u037E'**:

```
при;нос;или   →   при | нос | или
```

Корпус `full_corpus_ru_morph.txt` (1.77GB) — морфемно размеченный
полный корпус.

### 7.2 MorphVocab (morph_vocab.py)

Персональные **Zeckendorf-пути** вместо общих эмбеддингов:

```
служебные слова      → ZCK(cid)
знаменательные       → ZCK(lemma_rank)[:12] + ZCK(form_rank)[:4]
                         (общий префикс у форм одной леммы)
BPE-фолбэк           → ZCK(cid)
```

Сборка через Natasha (Segmenter, NewsMorphTagger, NatMorph) по корпусу.
Служебные POS: ADP, CCONJ, SCONJ, PART, PRON, PUNCT.

### 7.3 Гармонизация

- **Harmonizer** — гармонизация слов в латентном 2048D с контекстной
  модуляцией; drift-skip при cos>0.95;
- **MorphSTDP** — открытие морфем по когезии (порог 0.6),
  `discover_morphemes` каждые 100 батчей.

---

## 8. Генерация

`CrystalGenerator` — **beam search по концептам** (width=5, max_words=30,
min_words=3, concept_temp=0.5).

### 8.1 Шесть сигналов (RRF-взвешивание)

| Сигнал | RRF-вес | Источник |
|---|---|---|
| graph | 0.420 | BMSSP — мультиисточниковый BFS по PPMI-графу (B=2.0, depth=5, topk=8) |
| syntax | 0.259 | n-граммная решётка PPMI |
| hdc | 0.160 | HDC-память (резерв решётки) |
| vector | 0.099 | векторная близость через секторный LSH (CAM: 4+10+20 бит, O(1)) |
| prior | 0.061 | CollocationMatrix λ_d-приоры |

Плюс: VSA-внимание (Zeckendorf-weighted bind), гомеостатический буст,
intent-centroid бонус, анти-повтор + блокировка n-грамм, field-mask
фильтр, temperature softmax + top-p (nucleus).

### 8.2 Ограничители

- length normalization, MMI-штраф за частотные токены;
- EOS по пунктуации;
- гормональная модуляция температуры и ширины луча.

---

## 9. Нейромодуляция

`HormonalSystem` — нейромодуляторы через λ_d-базлайны:

| Глубина k | Коэффициент | Значение |
|---|---|---|
| 1 | DA baseline (per-token) | λ^{−1} = 0.618 |
| 2 | ACh baseline (n-gram) | λ^{−2} = 0.382 |
| 3 | NA baseline (phrase) | λ^{−3} = 0.236 |
| 4 | 5HT baseline (sentence) | λ^{−4} = 0.146 |
| 6 | tonic_decay | 1−λ^{−6} = 0.944 |
| (λ−1)/2 | homeostatic, intent, neg_lr, na_confidence | 0.309 |

Окна — числа Фибоначчи: F₁₀=55 (recent window), F₁₆=987 (history_maxlen,
rrf_prior_freq_cap). Гормоны модулируют lr, температуру, ширину луча и
коллокации при генерации.

---

## 10. λ_d-иерархия в числах (d=2, φ = 1.618)

| Глубина k | Коэффициент | λ_d-формула | Значение |
|---|---|---|---|
| 1 | DA baseline, PMI slope | λ^{−1} | 0.618 |
| 1 | antirep_decay | ln(2)/λ | 0.428 |
| 2 | ACh baseline, PMI gate, ht_match_scale | λ^{−2} | 0.382 |
| 3 | NA baseline | λ^{−3} | 0.236 |
| 4 | 5HT baseline | λ^{−4} | 0.146 |
| 6 | tonic_decay | 1−λ^{−6} | 0.944 |
| (λ−1)/2 | homeostatic, intent, neg_lr, … | (λ−1)/2 | 0.309 |
| F_3 = 2 | target_boost_temp, hormonal_mod | 1/F_3 | 0.500 |
| F_4 = 3 | field_weight_cap, boredom_repeat | F_4 | 3 |
| F_5 = 5 | target_boost_scale, boredom_window | F_5 | 5 |
| F_8 = 21 | da_coherence, da_temp_min | 1/((F_5−1)F_5) | 0.050 |
| F_10 = 55 | hormone_recent_window | F_10 | 55 |
| F_16 = 987 | history_maxlen, rrf_prior_freq_cap | F_16 | 987 |
| F_19 | checkpoint-интервал | F_19 | 4181 |
| F_20 | cosine T0 | F_20 | 6765 |

Расписания тоже Фибоначчи: warmup F₁₆=987, eval fast/slow 987/1597.
`FormulaCoefficients.rebuild(lam, vocab, d)` пересчитывает все 60+
констант из λ_d при `use_fib_generalized=True`. Реализация:
`eva/symbolic/fcf_config.py`, `eva/symbolic/fibonacci_utils.py`.

---

## 11. Конфигурация

`FCFConfig` (eva/symbolic/fcf_config.py): dim=768, latent_dim=2048,
n_anchors=2048, path_levels=16, vocab_size=65 536 (фактический — из
токенизатора), seed=42. Адаптация гиперпараметров — 15 `ParamDef`
правил (full_lr, repel_strength, neg_samples, pmi_strength,
context_window, momentum_mu и др.).

Интеграционные флаги: use_morph_stdp, use_vsa_attention,
use_hd_transformer, use_temporal_zeckendorf, use_morph_manifold,
use_fib_generalized=True.

### Как использовать

```bash
# Обучение (resume при наличии чекпойнта)
py -3.12 train_full.py --resume --corpus real_data\corpus_1m.txt \
  --learned-fields --field-bits 512 --neg-samples 3 \
  --context-window 4 --pmi-gate 0.0 --gen-every 10000
# или просто: train.bat
```

| Флаг | Описание |
|---|---|
| `--fresh` | старт с нуля (очистка чекпойнтов) |
| `--resume` | продолжить с чекпойнта (meta.json) |
| `--corpus` | путь к корпусу (по умолч. full_corpus_ru_clean.txt) |
| `--max-lines` | ограничение числа строк |
| `--vocab-size` | переопределение размера словаря |
| `--field-bits` | битность learned-поля (512) |
| `--learned-fields` | обучаемые поля поверх статических |
| `--batch-size` | строк в батче (100) |
| `--neg-samples` | негативная выборка (3) |
| `--context-window` | окно пар (4) |
| `--base-lr` | базовый lr (0.001) |
| `--fluctuation-amp` | амплитуда флуктуации (0.003) |
| `--pmi-gate` | PMI-гейт (0.0) |
| `--gen-every` | чекпойнт+генерация каждые N строк (10 000) |
| `--gen-max-words` | лимит слов тестовой генерации (100) |
| `--qwen-seed` | Qwen-дистилляция сидов (.npy) |

Ctrl+C = мягкий стоп с сохранением чекпойнта (KeyboardInterrupt-
обработчик). Контроль resume: `checkpoints/meta.json` (lines/pairs),
`real_data/train_run.log`.

---

## 12. Инференс-интерфейс

```python
py -3.12 inference.py --prompt "мир"          # генерация от seed
py -3.12 inference.py --neighbours "мир"      # ближайшие концепты
py -3.12 inference.py --retrieve "запрос"     # RAG: центроид → top-k
py -3.12 inference.py --eval                  # PPL/acc + генерация по сидам
```

- **read-only**: чекпойнт копируется во временную папку (защита от
  file-lock); `latest` резолвится через `checkpoint_state.json`
  (`line//1000` → тег «Nk»);
- `run_eval` пишет `eval_{tag}.json` (PPL, acc, cos-метрики, генерации);
- HF-совместимый слой: `model/` (configuration_fcf, modeling_fcf,
  tokenization_fcf); REST API: `api/` (FastAPI).

---

## 13. Структура репозитория

```
FCF/
├── eva/
│   ├── symbolic/
│   │   ├── fcf_config.py            # FCFConfig + FormulaCoefficients + λ_d-rebuild
│   │   ├── fibonacci_utils.py       # λ_d, обобщённый ряд Фибоначчи, Zeckendorf
│   │   ├── concept_space.py         # Ядро: FractalField, EntityField, концепты, поле
│   │   ├── crystal_generator.py     # Движок генерации (beam, RRF, гормоны)
│   │   ├── stdp_trainer.py          # STDP, негативная выборка, контрастив, гармонизация
│   │   ├── branch_network.py        # CollocationMatrix (L2/L3, λ_d-приоры)
│   │   ├── syntax_lattice.py        # PPMI n-граммная решётка + HDC-резерв
│   │   ├── transition_manifold.py   # Паутина переходов (VSA-кластеризация)
│   │   ├── morph_vocab.py           # Zeckendorf-пути морфем (lemma/form rank)
│   │   ├── sp_compat.py             # SPCompatTokenizer: HF BPE под API SentencePiece
│   │   ├── morpheme_tokenizer.py    # Трёхуровневое сито (символы→морфемы→слова)
│   │   ├── semantic_piece.py        # CharEnvelope, MorphSTDP, LFU-эвикция
│   │   ├── vsa_attention.py         # VSA-внимание (Zeckendorf-weighted bind)
│   │   ├── hdtransformer_layer.py   # VSA-native однослойный трансформер
│   │   ├── hormonal_system.py       # Нейромодуляция через λ_d-базлайны
│   │   ├── fractal_encoding.py      # Fractal путевая адресация
│   │   ├── multi_level_encoder.py   # Многоуровневый энкодер
│   │   ├── dimension_coordinator.py # VRAM-оценщик
│   │   ├── adaptive_controller.py   # Адаптация гиперпараметров (ParamDef)
│   │   ├── adaptive_error_tracker.py# EMA-трекер ошибок
│   │   ├── alphabet_basis.py        # Базис алфавита (seed-эмбеддинги)
│   │   ├── checkpoint_manager.py    # Асинхронные чекпойнты
│   │   ├── federated.py             # Федеративная агрегация
│   │   ├── lsh_index.py             # Секторный LSH (CAM, O(1))
│   │   ├── parameter_optimizer.py   # Param cascade
│   │   ├── rng_registry.py          # Централизованный реестр RNG
│   │   ├── seed_registry.py         # Централизованный реестр сидов
│   │   └── experimental/            # VSAGrid, ResidueEncoder, VSAConvLayer…
│   ├── morph.py                     # decompose_word (pymorphy3 + rule-based)
│   └── agi_protocol.py
├── model/                           # HF-совместимый слой
├── api/                             # FastAPI REST
├── scripts/                         # диагностика, BPE, сиды, prepare_wiki
├── tests/                           # 391 тест (7 skip)
├── docs/                            # FIBONACCI_SEQUENCES, LANGUAGE_LAMBDA,
│                                    #   MATHEMATICAL_FOUNDATIONS, TRAINING_JOURNAL…
├── reports/                         # 5 отчётов V22 (2026-06-23)
├── real_data/                       # corpus_1m.txt, full_corpus_ru*.txt,
│                                    #   tokenizer.json, morph_vocab.json,
│                                    #   morpheme_65k.json, train_run.log
├── checkpoints/                     # state.pkl + meta.json + corpus_lines.txt
├── train_full.py · inference.py · eval_metrics.py
├── train.bat · train_fast.bat · train.ps1
└── README.md                        # этот файл
```

---

## 14. Статус

- ✅ **λ_d-иерархия**: 60+ коэффициентов выведены из λ_d через
  `FormulaCoefficients.rebuild()` — 0 эмпирических констант
- ✅ **Обучение без backprop**: STDP + негативная выборка + контрастив
  + морфемная гармонизация (GPU-ядра, torch.compile на Volta+)
- ✅ **Токенизация**: WB v65536 (HF ByteLevel BPE) через SPCompat;
  трёхуровневый MorphemeTokenizer как эксперимент (morpheme-v1)
- ✅ **CollocationMatrix**: L2 (STDP-safe, ключи по cid) и L3
  коллокации — метрики обучения
- ✅ **Генерация**: beam + 6 RRF-сигналов + VSA-внимание + top-p
- ✅ **Морфология**: MorphVocab (Zeckendorf-пути), decompose_word,
  Harmonizer, MorphSTDP
- ✅ **391 тест** (7 skip), 0 ошибок
- ✅ **Обучение идёт**: 250K/1M строк, 38.5M пар (см. §15 и журнал)

---

## 15. Наблюдения из обучения

Полный журнал — `docs/TRAINING_JOURNAL.md`. Актуальный прогон:
`corpus_1m.txt` (1 000 000 строк), resume с 150K, политика автора —
никаких вмешательств.

| строки | pairs | colloc_l2 | colloc_l3 | чекпойнт |
|---|---|---|---|---|
| 150 000 (resume) | 23 079 494 | 4 853 733 | 2 563 | state.pkl |
| 160 000 | 24 625 364 | 5 175 291 | 2 719 | 909 MB |
| 170 000 | 26 163 540 | 5 499 539 | 2 867 | 916 MB |
| 180 000 | ~27.6M | ~5.75M | ~3 000 | 931 MB |
| 190 000 | 29 251 870 | 6 148 975 | 3 198 | — |
| 210 000 | 32 316 788 | 6 794 637 | 3 527 | 945 MB |
| 230 000 | ~35.3M | ~7.42M | ~3 870 | 958 MB |
| 250 000 | 38 469 486 | 8 089 966 | 4 194 | 972 MB |

Скорость 5-7 l/s, batch 15-18s (стабильно, память не течёт),
ETA до конца корпуса ~43h.

**Динамика генераций** (seed в скобках повторяется моделью):

| чекпойнт | характер текста |
|---|---|
| 160K | цепочки коллокаций («национальной армии», «молодцом национальной»); мусор «012», «XLadar», «Warad0» |
| 170K | связные цепочки («большой королевский футбольный», «результаты уже чем второго состоялась»); мусор «386», «ME» |
| 180K | предложения с грамматикой («в пределах… составляет 28 года», «упомянут академический апостол», «римская богиня»); мусор «1825 Plan Sim», «DOS» |
| 210K | числа/даты в потоке («2016 с 25», «в США 28», «до 1991», «населения поселка»); повторяющаяся коллокация «в штат гуайлы…»; мусор «оахв», «пестрероузоненко» |
| 230K | глагольные формы и даты («в 1900 году учился», «1966 года», «2324 августа 2007 годах», «28 километра»); римские «II» в «мехр II степени»; мусор «святецено нань», «герир», «(;» |
| 250K | длинные связные структуры («получил место в россии», «с запада и стал род птиц семейства», «совета национального чемпионата», «ниже уровня области»); мусор «корнья», «центика», «гуномсвест» |

**Выводы**:
- Качественный сдвиг на ~180K: от цепочек к предложениям с
  предлогами/падежами; на 250K — многословные связные структуры.
- L3-коллокации ускоряются с ростом корпуса (2563→4194 за 100K строк).
- Мусор стабилен по типу: цифровые и латинские байтовые токены общего
  WB-v65536 словаря («012», «386», «ME», «DOS») — артефакт словаря, не
  корпуса; доля снижается по мере роста L2.
- Метрики монотонны: pairs ~17.2K/100 строк, colloc_l2 ~3.2K/100 строк,
  batch ~18s без деградации.

---

## 16. Лицензия и статус

**Экспериментально.** Архитектура активно развивается; API/конфиги могут
меняться; чекпойнты совместимы между версиями только при строгой
миграции state_dict.

> FCF — исследовательский проект нейроморфной языковой архитектуры:
> векторные символьные операции (VSA), локальные правила обучения
> (STDP), самоорганизующиеся регуляторы по λ_d — без attention, без
> backprop и без больших softmax-матриц. «FCF не нужен лучший GPU.
> FCF нужен мемристор.»

---

*FCF — исследовательский проект. Участие и вопросы приветствуются.*