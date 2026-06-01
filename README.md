# EVA — Emergent Vector Architecture

**Не next-token prediction. Навигация в 384-мерном координатном пространстве.**

EVA — иерархическая когнитивная архитектура. Символы языка — точки в ℝ³⁸⁴, текст — траектория, 6 статистических голов — компас, WeightTransformer (33K параметров) — штурман.

Данные: Война и Мир (27K предложений) + ConceptNet Russian (478K семантических tripleтов). Общий объём: 505K предложений, 14.7M токенов. Вес на диске: 21 MB.

---

## 1. Архитектура

### 1.1 CoordinatePacker — 384-мерная перфокарта

Каждый `h[t]` — 384-мерный вектор с детерминированной упаковкой:

```
┌─────────────┬──────────┬──────────────────────────┐
│ Поле        │  Биты    │  Назначение              │
├─────────────┼──────────┼──────────────────────────┤
│ TOKEN       │  0-12    │  token_id (0-8191)       │
│ POS_WORD    │  13-20   │  позиция в слове         │
│ LEN_WORD    │  21-28   │  длина слова             │
│ NUM_WORD    │  29-36   │  номер слова             │
│ POS_SENT    │  37-45   │  позиция в предложении   │
│ LEN_SENT    │  46-54   │  длина предложения       │
│ FLAGS       │  55-72   │  18 бинарных флагов      │
│ META        │  73-80   │  тип BPE-токена          │
│ CONTEXT     │  81-88   │  n-gram fingerprint      │
│ ID_MISC     │  89-96   │  text_id (8 бит, 0=WP, 1=ConceptNet)│
│ RESERVED    │ 97-383   │  для трансформера (287)  │
└─────────────┴──────────┴──────────────────────────┘
```

Значения: `+1.0` = бит установлен, `-1.0` = бит сброшен, `0.0` = зарезервировано.
Полная обратимость: `pack(unpack(h)) == h`.

### 1.2 HeadsEnsemble — 6 статистических голов

Детерминированные, data-driven, без обучения. Каждая голова — предвычисленный массив (V,) или (V, V) sparse:

| Голова       | Размерность          | Источник данных              | Смысл                                          |
|-------------|----------------------|------------------------------|------------------------------------------------|
| Morph       | morph_logprob[wl][pos] → array(V) | Морфология слов            | P(token | позиция в слове, длина слова)        |
| Syntax      | syntax_logprob[wn] → array(V)    | Синтаксис                   | P(token | номер слова в предложении)          |
| Transition  | log_prob_csr (VxV sparse)        | Биграммы токенов            | log P(token | prev_token)                     |
| Semantic    | semantic_sim (VxV sparse)        | Cosine similarity transitions| Семантическая близость токенов               |
| Concept     | concept_scores (V,)              | Частота × контекстное разнообразие | Важность токена как концепта         |
| Contra      | contra_penalty (VxV sparse)      | Взаимоисключающие пары       | Штраф за противоречие                       |

Все 6 голов за один векторизованный вызов: `score_all(context)` → weighted sum.

### 1.3 WeightTransformer

Учится взвешивать 6 голов динамически, в зависимости от контекста:

```
token_embed(8) + [word_len, pos_in_word, word_num, pos_in_sent, sent_len, flags]
  → Linear(14, 32) → ReLU → Linear(32, 6) → Softplus
```

- **33,486 параметров** — в 1000× меньше, чем embedding-слой LLM
- Self-training: генерирует текст → heads оценивают → трансформер учится предсказывать следующий токен по 6-D weighted score
- Best val_acc: 15.8% (rule-based baseline: 9.0%, +76% relative)

### 1.4 GenerationLoop

Детерминированный конвейер:

```
SENT_OPEN → WORD_OPEN → [heads score] → select → ... → WORD_CLOSE
  → [choose next word or SENT_CLOSE] → WORD_OPEN → ...
```

Ключевые механики:
- После WORD_CLOSE: выбор между WORD_OPEN (продолжить) и SENT_CLOSE (закончить)
- Sigmoid ramp на WORD_CLOSE (поз. 2-6): бонус растёт с длиной слова
- Температурная выборка (0.0 = argmax, 1.0 = uniform)
- Маски на SPECIAL-токенах (SENT_OPEN/CLOSE не генерируются как контент)

### 1.5 Multi-Text (text_id)

Dims 89-96 = 8 бит, до 255 различных текстов:
- `text_id = 0` — Война и Мир (27K предложений)
- `text_id = 1` — ConceptNet Russian (478K предложений)

Morph/syntax распределения — взвешенное среднее (WP × 2, CN × 1). Transition — суммарный CSR.

---

## 2. Data-Driven, не Neural

Вся семантика извлечена из данных, не из весов:

```
Данные (текст) → BPE → Сбор статистик → 6 предвычисленных head-массивов
  → HeadsEnsemble (numpy, 11.6K calls/s) → WeightTransformer (33K params)
```

**Никакого backpropagation на heads.** Никаких embedding-слоёв на 50M+ параметров.
WeightTransformer учится только взвешивать heads — это задача с 6-d выходом, не next-token prediction над 4101 токенами.

---

## 3. Autonomous Think Loop

4 фазы, бесконечный цикл:

```
┌─ THINK ──────────────────────┐
│  Generate ~45K токенов (15s) │
│  Наполнить train_buffer (10K)│
└──────────┬───────────────────┘
           ▼
┌─ ANALYZE ────────────────────┐
│  Concept clustering          │
│  Contradiction audit         │
└──────────┬───────────────────┘
           ▼
┌─ LEARN ──────────────────────┐
│  Self-train WeightTransformer│
│  на реальных + сгенерированных│
└──────────┬───────────────────┘
           ▼
┌─ OPTIMIZE ───────────────────┐
│  Save concept clusters to DB │
│  Save model checkpoint       │
└──────────────────────────────┘
```

Запуск: `python eva/core/think_loop.py --port 8383`
Веб-дашборд: `http://localhost:8383`

---

## 4. ConceptNet Integration

- Источник: `conceptnet.db` (10.25 GB, 34M assertions, 50 relations)
- Фильтр: Russian→Russian edges (480K / 34M)
- Шаблоны: `form_of`→«— форма слова», `is_a`→«— это», `related_to`→«связан с», и т.д.
- Результат: 478K естественных русских предложений, 29 MB текста
- Токенизация: BPE (boundary tokens 157-160 совместимы с WP)
- Хранилище: `real_data/v5/conceptnet/` (3.7 MB)

---

## 5. Сжатие данных (уникальное)

| Компонент          | До сжатия   | После сжатия | Коэффициент |
|-------------------|-------------|-------------|------------|
| Trajectory Store  | 10.6 GB     | — (удалён)  | ∞          |
| Heads metadata    | сырые счётчики| 7.7 MB     | ~1000×     |
| Transitions       | dense matrix| CSR sparse  | ~400×      |
| Morph/syntax      | full arrays | sparse V-dim | ~200×      |
| Concept clusters  | —           | 10 bins     | —          |

Подробно: см. [COMPRESSION.md](COMPRESSION.md).

---

## 6. Ключевые цифры

| Метрика                  | Значение                     |
|-------------------------|------------------------------|
| Размерность             | 384                          |
| Размер словаря          | 4101 (BPE)                   |
| Всего параметров        | 33,486 (WeightTransformer)   |
| Скорость heads          | 11,600 calls/s               |
| Скорость генерации      | ~2,600 tok/s                 |
| Всего предложений       | 504,804 (27K WP + 478K CN)   |
| Всего токенов           | 14,700,456                   |
| Всего слов              | 2,280,414                    |
| Уникальных переходов    | 80,149                       |
| Противоречий            | 9,012                        |
| Вес на диске            | 21 MB                        |
| VRAM                    | 0 MB (CPU-only, numpy)       |
| Запуск                  | `.\run_eva.ps1`              |

---

## 7. Интеграция с dashboard

Два источника данных:
- `/api/state` — JSON: фаза, счётчики, веса heads, история точности/скорости
- `/api/log` — JSON: последние события

UI: 9 карточек, 3 графика (heads distribution, accuracy trend, gen rate trend), лог событий.
Автообновление каждые 2.5 сек.

---

## 8. Файловая структура

```
FCF/
├── eva/
│   ├── core/
│   │   ├── think_loop.py      # Автономный цикл (4 фазы)
│   │   ├── dashboard.py        # Веб-дашборд (localhost:8383)
│   │   └── database.py         # Хранилище ("Хранилище")
│   └── symbolic/
│       ├── heads.py            # HeadsEnsemble (6 голов)
│       ├── weight_transformer.py # 33K-параметровый взвешиватель
│       ├── generation_loop.py  # Авторегрессивная генерация
│       ├── coordinate_packer.py # 384-мерная упаковка
│       ├── reserved_dims.py    # Заполнение dims 97-383
│       ├── bpe_tokenizer.py    # BPE-токенизатор (4101)
│       └── char_vocab.py       # Символьный словарь (legacy)
├── real_data/v5/
│   ├── hierarchical/           # WP-only (sentences, CSR, caches)
│   ├── conceptnet/             # CN-only (sentences, CSR, caches)
│   └── heads_meta.pkl          # Merged (7.7 MB)
├── models/
│   └── weight_transformer_best.pt # Checkpoint трансформера
├── build_conceptnet_text.py    # Извлечение CN→текст
├── build_conceptnet_trajectories.py # Токенизация CN + merge heads
├── build_hierarchical.py       # Построение hierarchical (legacy)
├── run_eva.ps1                 # Лаунчер
└── README.md                   # Этот файл
```

---

## 9. Запуск

```powershell
# Прямой запуск
python -X utf8 eva/core/think_loop.py --port 8383

# Через PowerShell (с дашбордом)
.\run_eva.ps1

# Открыть дашборд
start http://localhost:8383
```

Для остановки: `Ctrl+C`.
