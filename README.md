# FCF — Fractal Cognitive Fabric (ЕВА)

**Единая Вычислительная Архитектура** — самоорганизующаяся когнитивная система, начинающая с одного слоя со случайными весами и вырастающая в многослойную архитектуру с доменными LoRA-адаптерами, KCA-циклом и State Algebra.

---

## Философия

ЕВА — это не программа. Это **зерно**, из которого вырастает когнитивная структура. В момент запуска система не знает ни одного слова, не имеет предобученных весов, не содержит ни грамматики, ни фактов о мире. Она обладает лишь:

1. **Случайными матрицами** — первичным алфавитом потенциальных преобразований
2. **Неизменяемыми аксиомами** — этическим ядром (5 аксиом)
3. **Мета-принципами** — правилами самооценки, роста, экономии и консолидации

Всё остальное — язык, знания, способность к рассуждению — ЕВА выращивает в себе сама.

---

## Принципы

| Принцип | Описание |
|---------|----------|
| **Tabula Rasa** | Старт со случайных весов (Xavier/Glorot), без предобученных моделей |
| **Compute-as-Memory** | Нет отдельной памяти и процессора. Информация = матричное состояние |
| **Этическая незыблемость** | 5 аксиом, вшитых в SRG. Не могут быть изменены или удалены |
| **Самооценка** | Semantic Relevance Gate (SRG) оценивает каждый ответ по семантике, энтропии, этике |
| **Динамический рост** | В ширину (LoRA-адаптеры) и в глубину (новые слои) по метрикам |
| **Консолидация** | Sleep Mode: кластеризация, очистка, дистилляция, слияние слоёв |

---

## Архитектура

### Вычислительное ядро

```
PrimordialLayer (один слой на старте)
├── Embedding        (vocab_size × d_model)  weight-tied с LM Head
├── TransformerBlock
│   ├── CausalSelfAttention (32 головы, RoPE)
│   ├── SwiGLU FFN           (d_model × 4)
│   └── RMSNorm              (Pre-Norm)
├── StateStorage     (FAISS IndexFlatIP, до 10K слепков)
├── SRG              (w_sim=0.4, w_ent=0.3, w_eth=0.3)
├── EthicsFilter     (5 аксиом: не навреди, честность, приватность, истина, полезность)
├── MetaMemory       (confidence_history, usage_count)
├── GrowthController (EXPAND_WIDTH / EXPAND_DEPTH / TRY_RECURSION)
├── CuriosityLoop    (авто-генерация уточняющих вопросов)
└── DomainRegistry   (реестр LoRA-адаптеров по доменам)
```

### Параметры по умолчанию

| Параметр | Значение |
|----------|----------|
| d_model | 2560 |
| num_heads | 32 |
| head_dim | 80 |
| ff_mult | 4 (SwiGLU) |
| vocab_size | 50257 |
| max_seq_len | 2048 |
| Всего параметров | **233 520 640** |
| LoRA rank | 8 (286K параметров = 0.12%) |

---

## 7 этапов развития

### Пункт 1 — Первооснова
Создание `PrimordialLayer` со случайными весами, FAISS-хранилищем, SRG, EthicsFilter, CuriosityLoop и GrowthController.

### Пункт 2 — Самообучение языку
Автономный цикл Causal LM на текстовом корпусе. Критерий остановки: средняя уверенность SRG > 0.7, 1000+ слепков, счётчик любопытства = 0.

### Пункт 3 — Инструктивное дообучение
Обучение следованию инструкциям на парах «вопрос → ответ» (Saiga/JSON). Chat-шаблон `<|im_start|>`, маскирование user-части, пониженный learning rate (1e-5).

### Пункт 4 — Доменные правила (рост в ширину)
Создание LoRA-адаптеров для новых концептов. DomainRegistry хранит центроиды и адаптеры. При запросе определяется домен → подключается адаптер → генерация → отключение.

### Пункт 5 — Рост в глубину
Рекурсивная обработка (до 5 проходов через последний слой). При систематических провалах — кристаллизация нового слоя (копия последнего + специализация на failed_queries). Sequential pipeline слоёв.

### Пункт 6 — Sleep Mode (консолидация)
Фоновая оптимизация: HDBSCAN-кластеризация слепков, temporal decay, дистилляция LoRA в базовые веса, слияние избыточных слоёв, дефрагментация FAISS.

### Пункт 7 — Полноценный FCF
- **State Algebra** — композиция знаний: сумма, масштабирование, вычитание, кросс-аттеншн в латентном пространстве с проектором Proj_ℳ
- **KCA Engine** — итеративное уточнение с гарантированной сходимостью (демпфирование, детектор осцилляций, монитор гейта, ≤5 итераций)
- **HNSW Index** — иерархический поиск: уровень 0 (домены) → уровень 1 (слепки) → уровень 2 (слои)
- **Когнитивный цикл** — Восприятие → Порождение (KCA) → Исполнение и сохранение

---

## Структура проекта

```
FCF/
├── fcf/                        # Исходный код (26 модулей)
│   ├── primordial_layer.py     # PrimordialLayer — основной вычислительный элемент
│   ├── transformer.py          # CausalSelfAttention, SwiGLUFFN, RMSNorm, RoPE
│   ├── state_storage.py        # FAISS IndexFlatIP хранилище состояний
│   ├── srg.py                  # Semantic Relevance Gate (самооценка)
│   ├── ethics_filter.py        # 5 аксиом, 4 категории паттернов
│   ├── meta_memory.py          # Мета-память слоя (confidence_history)
│   ├── growth_controller.py    # EXPAND_WIDTH / EXPAND_DEPTH / TRY_RECURSION
│   ├── curiosity_loop.py       # Генерация уточняющих вопросов
│   ├── lora_adapter.py         # LoRA (A: rank×in, B: out×rank, 286K params)
│   ├── domain_registry.py      # Реестр доменных правил + поиск по центроидам
│   ├── domain_trainer.py       # Обучение LoRA-адаптеров под домены
│   ├── recursive_processor.py  # Рекурсивная обработка (max_depth=5)
│   ├── layer_crystallizer.py   # Кристаллизация новых слоёв
│   ├── sleep_mode.py           # Консолидация: кластеризация, очистка, дистилляция
│   ├── kca_engine.py           # KCA: итеративное уточнение + ConvergenceController
│   ├── state_algebra.py        # Операторы композиции + автоэнкодер Proj_ℳ
│   ├── hnsw_index.py           # Иерархический HNSW (3 уровня)
│   ├── config.py               # FCFConfig (dataclass + JSON)
│   ├── data_manager.py         # Wikipedia, Saiga, ConceptNet, RuBQ
│   ├── tokenizer_utils.py      # BPE-токенизатор (HuggingFace tokenizers)
│   ├── language_trainer.py     # Пункт 2: Causal LM обучение
│   ├── instruction_trainer.py  # Пункт 3: инструктивное дообучение
│   └── utils.py                # save/load с FAISS, весами, метаданными
│
├── docs/                       # Дизайн-документы (Пункты 1-7 + план реализации)
├── tests/                      # Тестовые данные
├── logs/                       # Логи обучения
├── checkpoints/                # Чекпоинты (игнорируются git)
├── domain_rules/               # LoRA-адаптеры доменов (игнорируются git)
├── config.json                 # Конфигурация
├── requirements.txt            # Зависимости
├── tokenizer.json              # BPE-токенизатор
├── fcf.bat                     # Запуск (двойной клик = интерактивный режим)
├── train_all.bat               # Полный пайплайн обучения
└── run.py                      # CLI точка входа
```

---

## Запуск

```bash
# Установка
pip install -r requirements.txt

# Инициализация PrimordialLayer
python run.py --init

# Полный пайплайн обучения (токенизатор → язык → инструкции → тест)
train_all.bat

# Интерактивный режим с загруженным чекпоинтом
python run.py --interactive --checkpoint checkpoints\instruction\final

# Обучение отдельного этапа
python run.py --train-language --text-file corpus.txt --max-steps 2000
python run.py --train-instruction --instructions-file instructions.json --max-steps 500

# Доменное обучение (ConceptNet или JSON)
python run.py --train-domain --data-file facts.json --domain-id my_domain

# Рост в глубину
python run.py --train-depth

# Консолидация (Sleep Mode)
python run.py --sleep

# Полный тест всех компонентов
python run.py --full-test
```

---

## Технологический стек

| Категория | Технология |
|-----------|-----------|
| ML-фреймворк | PyTorch 2.5+ |
| Векторная БД | FAISS (IndexFlatIP → HNSW) |
| Токенизатор | HuggingFace `tokenizers` (BPE) |
| Оптимизатор | AdamW |
| Кластеризация | HDBSCAN / KMeans |
| Логирование | Loguru |
| База знаний | ConceptNet (conceptnet-lite) |
| Python | 3.12+ |

---

## Состояние разработки

| Пункт | Компонент | Статус |
|-------|-----------|--------|
| 1 | PrimordialLayer, SRG, EthicsFilter, FAISS, Curiosity, Growth | ✅ Готово |
| 2 | LanguageTrainer (Causal LM) | ✅ Готово |
| 3 | InstructionTrainer (Saiga/JSON, маскирование) | ✅ Готово |
| 4 | LoRAAdapter, DomainRegistry, DomainTrainer | ✅ Готово |
| 5 | RecursiveProcessor, LayerCrystallizer | ✅ Готово |
| 6 | SleepMode (кластеризация, очистка, дистилляция, слияние) | ✅ Готово |
| 7 | KCAEngine, StateAlgebra, HNSWIndex | ✅ Готово |

---

## Отличия от EVA-Ai

| Аспект | EVA-Ai | FCF |
|--------|--------|-----|
| Модель | Предобученная Qwen3 4B (OpenVINO) | Случайные веса (PyTorch, Tabula Rasa) |
| Слои | Фиксированные 36 слоёв | 1 растущий слой |
| Хранилище | SQLite + HNSW (451K узлов) | FAISS IndexFlatIP |
| Этика | Отсутствует | 5 аксиом в EthicsFilter |
| Инференс | OpenVINO State API | PyTorch autograd |
| Рост | Онлайн-дообучение LoRA/GNN | Автономный GrowthController |
| Память | TCM + FractalGraphV2 + ScenarioTCM | FAISS StateStorage |
| Кодовая база | 200+ файлов, 50K строк | 26 модулей, ~7K строк |

---

## Лицензия

Приватный репозиторий. Все права защищены.

---

*FCF — Fractal Cognitive Fabric. Единая Вычислительная Архитектура. v0.1.0*
