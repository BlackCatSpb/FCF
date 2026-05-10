# FCF (Fractal Cognitive Fabric) — План реализации

> Основан на дизайн-документах FCF (Пункты 1–7) и анализе боевой реализации EVA-Ai.
> Дата: 2026-05-10

---

## 0. Стратегические решения

### 0.1. Что берём из FCF-дизайна (обязательно)

| Принцип | Реализация |
|----------|------------|
| **Tabula Rasa** | Старт со случайных весов Xavier/Glorot, без предобученных моделей |
| **Один растущий слой** | На старте — единственный `PrimordialLayer`, рост в ширину (LoRA) и глубину (новые слои) по метрикам |
| **FAISS-хранилище** | `IndexFlatIP` + `snapshots_meta` (как в Пункте 1.2) |
| **SRG** | Трёхкомпонентная оценка: семантика + энтропия + этика |
| **EthicsFilter** | Regex-фильтр с 5 аксиомами, неизменяемый |
| **CuriosityLoop** | Авто-генерация уточняющих вопросов через прямой проход |
| **GrowthController** | Явные сигналы EXPAND_WIDTH / EXPAND_DEPTH |
| **State Algebra** | Операторы в латентном пространстве (сумма, масштаб, вычитание, кросс-аттеншн) |
| **KCA-цикл** | Итеративное уточнение с гарантированной сходимостью |

### 0.2. Что адаптируем из EVA-Ai

| Компонент EVA-Ai | Как адаптируем в FCF |
|------------------|---------------------|
| **LayerwiseStateInjector** (OpenVINO State API) | Временно НЕ используем — FCF идёт через PyTorch. Но архитектура доступа к KV-кешу пригодится на этапе multi-layer |
| **HNSW-индекс** (вместо плоского FAISS) | Используем FAISS на старте, мигрируем на HNSW в Пункте 7 (иерархический поиск) |
| **FractalGraphV2** (SQLite + граф) | Не используем. FCF хранит состояния в FAISS + `snapshots_meta`, доменные правила — в `DomainRegistry` (in-memory + pickle) |
| **OnlineTrainer** (фоновое обучение) | Адаптируем для фонового дообучения LoRA-адаптеров и слоёв |
| **GraphCurator** (консолидация) | Берём логику кластеризации (HDBSCAN) и temporal decay для Sleep Mode |
| **KCA-детектор** (EVA-Ai `kca_integration.py`) | Адаптируем: в EVA-Ai хорошо сделана детекция лакун и осцилляций, протокол сходимости. Берём структуру `KCACorrection` и `ConvergenceController` |
| **ShadowLoRAManager** | Берём паттерн атомарной замены LoRA-адаптеров для доменных правил |
| **SQAM** (семантическая сигнатура) | Адаптируем как часть `CuriosityLoop` — для определения, о чём спрашивать |

### 0.3. Архитектурные константы

```python
d_model = 2560          # Размерность скрытого состояния
num_heads = 32          # Головы внимания
head_dim = 80           # d_model // num_heads
ff_mult = 4             # Множитель FFN (SwiGLU)
max_snapshots = 10000   # Лимит хранилища (растёт с ростом системы)
vocab_size = 50257      # Стандартный BPE (можно переопределить)
max_seq_len = 2048      # На старте; растёт до 262144
```

### 0.4. Стек технологий

| Категория | Выбор | Обоснование |
|-----------|-------|-------------|
| **ML-фреймворк** | PyTorch 2.5+ | Прямой доступ к градиентам, весам, кастомным циклам |
| **Векторная БД** | FAISS | `IndexFlatIP` → `IndexHNSWFlat` (Пункт 7) |
| **Токенизатор** | `tokenizers` (HuggingFace) | BPE с нуля на русском корпусе |
| **Оптимизатор** | AdamW | Стандарт для трансформеров |
| **Логирование** | `loguru` | Как в EVA-Ai |
| **Конфигурация** | JSON + dataclasses | `brain_config.json`-стиль |
| **NLP** | `nltk`, `spaCy` (ru_core_news_sm) | Для EthicsFilter и анализа |

---

## 1. Пункт 1 — Первооснова (PrimordialLayer)

### Цель
Создать единственный функциональный слой со случайными весами, способный:
- Выполнять прямой проход (Causal LM)
- Оценивать свои ответы через SRG + EthicsFilter
- Сохранять успешный опыт в FAISS
- Генерировать уточняющие вопросы (CuriosityLoop)
- Принимать решение о росте (GrowthController)

### 1.1. Структура файлов

```
FCF/
├── fcf/
│   ├── __init__.py
│   ├── primordial_layer.py    # PrimordialLayer (TransformerBlock + все компоненты)
│   ├── transformer.py         # TransformerBlock, CausalSelfAttention, SwiGLU-FFN, RMSNorm
│   ├── state_storage.py       # StateStorage (FAISS + snapshots_meta)
│   ├── srg.py                 # SemanticRelevanceGate
│   ├── ethics_filter.py       # EthicsFilter (regex + аксиомы)
│   ├── meta_memory.py         # MetaMemory (confidence_history)
│   ├── growth_controller.py   # GrowthController (EXPAND_WIDTH / EXPAND_DEPTH)
│   ├── curiosity_loop.py      # CuriosityLoop (генерация вопросов)
│   ├── data_manager.py        # DataManager (Wikipedia, Saiga, RuBQ)
│   ├── config.py              # FCFConfig, константы
│   └── utils.py               # save/load, инициализация Xavier
├── config.json                # Конфигурация системы
├── requirements.txt
└── run.py                     # Точка входа
```

### 1.2. Реализация PrimordialLayer

```python
class PrimordialLayer(nn.Module):
    """Единственный вычислительный элемент ЕВА на старте."""
    
    def __init__(self, config: FCFConfig):
        super().__init__()
        self.config = config
        
        # Transformer
        self.transformer = TransformerBlock(
            d_model=config.d_model,
            num_heads=config.num_heads,
            ff_mult=config.ff_mult
        )
        
        # FAISS StateStorage
        self.state_storage = StateStorage(
            dim=config.d_model,
            max_snapshots=config.max_snapshots
        )
        
        # Самооценка
        self.srg = SemanticRelevanceGate(config)
        
        # Мета-память слоя
        self.meta = MetaMemory()
        
        # Контроллер роста
        self.growth = GrowthController(config)
        
        # Любопытство
        self.curiosity = CuriosityLoop(config)
        
        # LM Head (из TransformerBlock или отдельный)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.W_O.weight  # weight tying
    
    def forward(self, x, attention_mask=None):
        return self.transformer(x, attention_mask)
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=128, 
                 temperature=0.8, top_k=50, top_p=0.9):
        """Авторегрессионная генерация."""
        ...
```

### 1.3. TransformerBlock

Взять архитектуру из EVA-Ai `fcp_core/hybrid_layer.py` как референс, но упростить:

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_mult=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Causal Self-Attention
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        # SwiGLU Feed-Forward
        self.gate_proj = nn.Linear(d_model, d_model * ff_mult, bias=False)
        self.up_proj = nn.Linear(d_model, d_model * ff_mult, bias=False)
        self.down_proj = nn.Linear(d_model * ff_mult, d_model, bias=False)
        
        # RMSNorm
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        
        self._init_weights()
```

### 1.4. EthicsFilter — адаптация из FCF-дизайна + расширение

Взять базовый regex-фильтр из Пункта 1.4 и расширить:

```python
class EthicsFilter:
    """Неизменяемый этический фильтр на 5 аксиомах."""
    
    # 5 аксиом (из FCF New.txt, раздел 4.1)
    AXIOMS = {
        "non_harm": "Не навреди",
        "honesty": "Будь честен",
        "privacy": "Уважай приватность",
        "truth": "Стремись к истине",
        "usefulness": "Будь полезен"
    }
    
    HARM_PATTERNS = [...]       # Пункт 1.4 FCF + дополнения
    PRIVACY_PATTERNS = [...]    # Пункт 1.4 FCF + дополнения
    DISHONESTY_PATTERNS = [...] # Новое: маркеры неуверенности без оговорок
    USELESS_PATTERNS = [...]    # Новое: пустые/отвлечённые ответы
    
    PENALTY_PER_MATCH = 0.2
    ETHICS_THRESHOLD = 0.3     # Ниже — безусловное отклонение
    
    def evaluate(self, text: str) -> Tuple[float, Dict[str, float]]:
        """Возвращает (общий скор, {аксиома: скор})."""
        ...
```

### 1.5. Чек-лист Пункта 1

- [x] `TransformerBlock` со случайными весами (Xavier)
- [x] `StateStorage` на FAISS `IndexFlatIP`
- [x] `SemanticRelevanceGate` (w_sim=0.4, w_ent=0.3, w_eth=0.3)
- [x] `EthicsFilter` (5 аксиом, regex)
- [x] `MetaMemory` (confidence_history, usage_count)
- [x] `GrowthController` (width_threshold=0.5, depth_threshold=0.3)
- [x] `CuriosityLoop` (генерация вопросов через прямой проход)
- [x] `DataManager` (Wikipedia, Saiga, RuBQ — загрузчики)
- [x] `save()` / `load()` — сериализация всех компонентов
- [x] Тест: прямой проход на случайном тензоре не падает

---

## 2. Пункт 2 — Самообучение языку

### Цель
Научить единственный слой предсказывать следующее слово (Causal LM) на русской Википедии.
Без внешней разметки. Система сама решает, когда остановиться.

### 2.1. Данные

Использовать `DataManager.load_wikipedia()` из FCF-дизайна:
```python
from datasets import load_dataset
wiki = load_dataset('wikipedia', '20220301.ru', split='train', streaming=True)
```

Блоки по 512 токенов. `input_ids` = первые 511, `labels` = сдвиг на 1.

### 2.2. Цикл обучения

```python
def train_language(self, data_iterator, max_steps=None):
    optimizer = AdamW(self.parameters(), lr=1e-4)
    step = 0
    srg_eval_interval = 100
    checkpoint_interval = 1000
    curiosity_threshold = 10
    
    stopped = False
    while not stopped:
        batch = next(data_iterator)
        
        # Forward
        logits = self(batch['input_ids'])
        loss = F.cross_entropy(logits, batch['labels'])
        
        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        step += 1
        
        # SRG evaluation every N steps
        if step % srg_eval_interval == 0:
            confidence = self._evaluate_generation()
            self.meta.record(confidence)
            
            if confidence > 0.8:
                self._save_snapshot()
            
            if confidence < 0.6:
                self.curiosity.counter += 1
            else:
                self.curiosity.counter = 0
            
            if self.curiosity.should_ask():
                question = self.curiosity.generate_clarification(...)
                # На этом этапе — только в лог
        
        # Checkpoint
        if step % checkpoint_interval == 0:
            self.save(f"checkpoints/step_{step}")
        
        # Критерий остановки
        stopped = self._check_stop_criterion()
```

### 2.3. Критерий остановки (из FCF Пункта 2.4)

```python
def _check_stop_criterion(self):
    return (
        self.meta.average_confidence(window=500) > 0.7 and
        len(self.state_storage.snapshots_meta) > 1000 and
        self.curiosity.counter == 0   # не срабатывал 500+ итераций
    )
```

### 2.4. Ожидаемый результат
- Слой порождает грамматически связный текст
- 1000+ успешных слепков в FAISS
- Средняя уверенность SRG > 0.7

---

## 3. Пункт 3 — Инструктивное дообучение

### Цель
Научить слой следовать инструкциям, используя датасет Saiga.

### 3.1. Данные

```python
saiga = DataManager.load_saiga()  # IlyaGusev/saiga2_70b_lora
```

Формат: `<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>`

Маскирование user-части (как в FCF Пункте 3.2 и EVA-Ai `fcp_pipeline.py`).

### 3.2. Отличия от Пункта 2

| Параметр | Пункт 2 | Пункт 3 |
|----------|---------|---------|
| Данные | Wikipedia (сырой текст) | Saiga (инструкции) |
| Learning rate | 1e-4 | 1e-5 |
| Маскирование | Нет | User-часть исключена из loss |
| Порог сохранения слепка | 0.8 | 0.8 |
| Критерий остановки | Те же 3 условия | Те же 3 условия |

### 3.3. Буфер неразрешённых ситуаций

Новое: `pending_clarifications` — список для накопления неудачных примеров (из FCF Пункта 3.3.6). Будет использован в активном обучении.

---

## 4. Пункт 4 — Доменные правила и рост в ширину

### Цель
При столкновении с новым концептом создавать LoRA-адаптер, а не переучивать базовые веса.

### 4.1. DomainRegistry

Ключевая структура данных (в отличие от EVA-Ai, где домены хранятся в FractalGraphV2):

```python
@dataclass
class DomainRule:
    domain_id: str
    lora_A: np.ndarray       # (d_model, rank)
    lora_B: np.ndarray       # (rank, d_model)
    context_centroid: np.ndarray  # (d_model,)
    usage_count: int = 1
    confidence_history: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

class DomainRegistry:
    def __init__(self):
        self.rules: Dict[str, DomainRule] = {}
        self.centroids_matrix: Optional[np.ndarray] = None  # Для быстрого поиска
    
    def find_best(self, c_query: np.ndarray, threshold: float = 0.7) -> Optional[str]:
        """Найти домен по косинусному сходству с центроидом."""
        ...
    
    def add(self, rule: DomainRule):
        ...
    
    def apply_adapter(self, domain_id: str, layer_weights: Dict[str, torch.Tensor]):
        """W = W + lora_B @ lora_A (временно, на время запроса)."""
        ...
```

### 4.2. LoRA-адаптер

В отличие от EVA-Ai (где 4 фиксированных адаптера), в FCF адаптеры создаются динамически:

```python
class LoRAAdapter:
    def __init__(self, d_model: int, rank: int = 8):
        # Инициализация: A ~ N(0, 0.02), B ~ zeros
        self.A = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.B = nn.Parameter(torch.zeros(rank, d_model))
        self.alpha = 1.0  # Масштаб вклада
    
    def forward(self, W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        delta = (self.B @ self.A) * (self.alpha / self.rank)
        return F.linear(x, W + delta)
```

### 4.3. Обнаружение нового домена

```python
def process_query(self, query: str) -> str:
    c_query = self._compute_context_vector(query)
    
    # 1. Точный поиск в StateStorage
    snapshot_idx = self.state_storage.search(c_query, threshold=0.95)
    if snapshot_idx >= 0:
        return self._generate_from_snapshot(snapshot_idx)
    
    # 2. Поиск домена
    domain_id = self.domain_registry.find_best(c_query, threshold=0.7)
    if domain_id:
        self.domain_registry.apply_adapter(domain_id, self.transformer)
        response = self.generate(query)
        self.domain_registry.remove_adapter(domain_id, self.transformer)
        return response
    
    # 3. Новый домен — создать адаптер
    signal = self.growth.evaluate(self.meta, gradient_norm)
    if signal == "EXPAND_WIDTH":
        return self._create_new_domain(query, c_query)
    
    return self.generate(query)  # fallback
```

### 4.4. Источники структурированных знаний

```python
# Wikidata через RuBQ 2.0
rubq = DataManager.load_rubq("rubq_2.0.jsonl")

# ConceptNet (русская часть)
conceptnet = DataManager.load_conceptnet_ru("conceptnet.db")
```

---

## 5. Пункт 5 — Рост в глубину

### Цель
Когда одного слоя (даже с LoRA) недостаточно — создать новый слой.

### 5.1. Сигнал EXPAND_DEPTH

В отличие от EVA-Ai (где 36 слоёв фиксированы), в FCF слой создаётся при:

```python
def evaluate(self, meta: MetaMemory, gradient_norm: float) -> str:
    avg = meta.average_confidence()
    
    # Систематически низкая уверенность + исчерпана рекурсия
    if avg < self.depth_threshold and len(meta.confidence_history) >= self.patience:
        return "EXPAND_DEPTH"
    
    if avg < self.width_threshold and gradient_norm > self.gradient_threshold:
        return "EXPAND_WIDTH"
    
    return "NO_GROWTH"
```

### 5.2. Рекурсивная обработка (как в FCF Пункте 5.3)

Перед созданием слоя — попытка «додумать»:

```python
def recursive_process(self, x, max_depth=5):
    for i in range(max_depth):
        x = self.forward(x)
        response = self.generate_from_state(x)
        confidence = self.srg.evaluate(...)
        
        if confidence > 0.7:
            return response, confidence  # Успех
    
    return None, 0.0  # Рекурсия не помогла
```

### 5.3. Кристаллизация нового слоя

Новый слой = копия последнего (как в FCF Пункте 5.4) + дообучение на failed_queries:

```python
def crystallize_layer(self):
    last_layer = self.layers[-1]
    new_layer = PrimordialLayer(self.config)
    
    # Копируем веса
    new_layer.load_state_dict(last_layer.state_dict())
    
    # Пустое хранилище для нового слоя
    new_layer.state_storage = StateStorage(dim=self.config.d_model)
    new_layer.domain_registry = DomainRegistry()
    
    # Специализация на проваленных запросах
    if self.failed_queries:
        self._finetune_layer(new_layer, self.failed_queries, steps=100)
    
    self.layers.append(new_layer)
```

### 5.4. Маршрутизация (sequential pipeline)

```python
def forward_all(self, x):
    for layer in self.layers:
        x = layer(x)
    return x
```

### 5.5. Защита от взрывного роста (из FCF Пункта 5.5)

- `MAX_LAYERS = 100`
- `MIN_LAYER_INTERVAL = 300` секунд (5 минут)
- `MAX_RECURSION_DEPTH = 5`
- `QUERY_TIMEOUT = 5` секунд

---

## 6. Пункт 6 — Sleep Mode (Консолидация)

### Цель
В периоды простоя — оптимизировать архитектуру, удалить избыточность.

### 6.1. Условия активации (из FCF + адаптация EVA-Ai)

```python
class SleepMode:
    IDLE_TIMEOUT = 300       # 5 минут без запросов
    SLEEP_INTERVAL = 7200    # 2 часа планово
    
    def should_sleep(self):
        return (
            time.time() - self.last_query_time > self.IDLE_TIMEOUT or
            time.time() - self.last_sleep_time > self.SLEEP_INTERVAL
        )
```

### 6.2. Этапы консолидации

#### 6.2.1. Кластеризация слепков (адаптация из EVA-Ai GraphCurator)

```python
def cluster_snapshots(self):
    # Извлечь все векторы из FAISS
    vectors = self._get_all_vectors()
    
    # HDBSCAN кластеризация
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='cosine')
    labels = clusterer.fit_predict(vectors)
    
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # Шум
        cluster_vectors = vectors[labels == cluster_id]
        centroid = cluster_vectors.mean(axis=0)
        
        # Оставить лучший слепок, удалить остальные
        best_idx = self._best_in_cluster(labels, cluster_id)
        self._prune_cluster(labels, cluster_id, keep=best_idx, centroid=centroid)
```

#### 6.2.2. Удаление устаревшего

```python
def remove_stale(self):
    now = time.time()
    for meta in list(self.state_storage.snapshots_meta):
        age = now - meta['timestamp']
        score = meta['usage_count'] * math.exp(-0.1 * age / 86400)  # λ=0.1, дни
        if score < self.MIN_SCORE:
            self._remove_snapshot(meta)
```

#### 6.2.3. Дистилляция LoRA в базовые веса

Взять логику из EVA-Ai `GraphCurator`:

```python
def distill_domain(self, domain_id: str):
    rule = self.domain_registry.rules[domain_id]
    if rule.usage_count < 100 or avg_confidence(rule) < 0.8:
        return  # Недостаточно стабилен
    
    # Дообучить базовые веса с регуляризацией
    self._distill_to_base_weights(rule)
    
    # Удалить адаптер
    del self.domain_registry.rules[domain_id]
```

#### 6.2.4. Слияние избыточных слоёв

```python
def merge_redundant_layers(self):
    for i in range(len(self.layers) - 1):
        for j in range(i + 1, len(self.layers)):
            kl_div = self._compute_output_kl(self.layers[i], self.layers[j])
            if kl_div < 0.1:
                self._merge_layers(i, j)  # Усреднение матриц, слияние хранилищ
                break
```

### 6.3. Интеграция с EVA-Ai паттернами

| EVA-Ai компонент | Как используем |
|-----------------|----------------|
| `GraphCurator._async_cleanup_garbage()` | Асинхронная очистка слепков |
| `GraphCurator.integrate_with_fg2_decay()` | Temporal decay для FAISS-метаданных |
| `HDBSCAN` кластеризация | Вместо k-Means (не требует задания числа кластеров) |

---

## 7. Пункт 7 — Полноценный FCF

### Цель
Интегрировать State Algebra, KCA-цикл и иерархический HNSW.

### 7.1. State Algebra (из FCF Пункта 7.2)

```python
class StateAlgebra:
    def __init__(self, k: int = 256):
        self.k = k  # Размерность латентного кода
        self.projector = Projector(k)  # Proj_ℳ: лёгкий автоэнкодер
    
    def sum(self, z_A, z_B):
        return self.projector(z_A + z_B)
    
    def scale(self, z, alpha):
        return self.projector(alpha * z)
    
    def subtract(self, z_A, z_B, beta=1.0, tau=1.0):
        clamped = torch.clamp(z_A - beta * z_B, -tau, tau)
        return self.projector(clamped)
    
    def cross_attend(self, z_A, z_B):
        combined = torch.cat([z_A, z_B], dim=-1)
        return self.projector(self.cross_attn(combined))
```

### 7.2. KCA-цикл (адаптация из EVA-Ai `kca_integration.py`)

Структура `KCACorrection` из EVA-Ai уже хороша:

```python
@dataclass
class KCACorrection:
    gap_embedding: np.ndarray
    contra_embedding: np.ndarray
    total_correction: np.ndarray
    gate_value: float
    layer_idx: int
    confidence: float

class KCAEngine:
    def __init__(self, config):
        self.max_iterations = 5
        self.rho = 0.85
        self.convergence = ConvergenceController(config)
    
    def refine(self, z_init, context, graph_nodes) -> Tuple[np.ndarray, float]:
        z = z_init
        for i in range(self.max_iterations):
            eta = self.config.eta0 * (self.rho ** i)  # Адаптивное демпфирование
            
            delta_W = self.decoder(z)  # z → матрицы
            output = self.forward_with(delta_W, context)
            
            loss = self.utility_function(output, graph_nodes)
            grad = self.compute_gradient(z, loss)
            
            z_new = z - eta * grad
            
            status, z_out = self.convergence.check(z_new, z, self.gate_value, i)
            if status != "CONTINUE":
                return z_out, self.convergence.final_confidence
            
            z = z_new
        
        return z, self.srg.evaluate(...)
```

### 7.3. Иерархический HNSW (из FCF Пункта 7.4)

Переход с FAISS `IndexFlatIP` на `IndexHNSWFlat` с 3 уровнями:

```
Уровень 0: центроиды всех доменов      → IndexHNSWFlat(d_model, 32)
Уровень 1: слепки внутри домена        → N индексов по числу доменов
Уровень 2: слепки по диапазонам слоёв  → подындексы (1–8, 9–16, 17–24, 25–32)
```

### 7.4. Трёхфазный когнитивный цикл

Адаптировать из EVA-Ai `fcp_pipeline.py` (строки 800–1200):

```python
def cognitive_cycle(self, query: str) -> str:
    # Фаза 1: Восприятие и поиск
    tokens = self.tokenizer.encode(query)
    c_query = self.compute_context_vector(tokens)
    stored_state = self.hnsw_search(c_query)
    
    if stored_state and stored_state.similarity > 0.95:
        return self.generate_from_stored(stored_state)
    
    # Фаза 2: Порождение и коррекция (KCA)
    if stored_state and stored_state.similarity > 0.7:
        z = self.interpolate(stored_state.z, c_query)
    else:
        z = self.cold_start(c_query)
    
    z_optimal, confidence = self.kca.refine(z, query, graph_nodes)
    
    # Фаза 3: Исполнение и сохранение
    response = self.generate_from_z(z_optimal)
    
    if confidence > 0.7:
        self.save_state(c_query, z_optimal, confidence)
    
    return response
```

---

## 8. Дорожная карта

| Этап | Срок | Результат | Зависимости |
|------|------|-----------|-------------|
| **Пункт 1** | 2 недели | PrimordialLayer: слой + SRG + FAISS + Ethics + Curiosity | — |
| **Пункт 2** | 2–3 недели | Самообучение на Wikipedia, уверенность > 0.7 | Пункт 1 |
| **Пункт 3** | 1–2 недели | Инструктивное дообучение на Saiga | Пункт 2 |
| **Пункт 4** | 3 недели | DomainRegistry, LoRA-адаптеры, Wikidata/ConceptNet | Пункт 3 |
| **Пункт 5** | 3 недели | Рекурсивная обработка, кристаллизация слоёв, sequential pipeline | Пункт 4 |
| **Пункт 6** | 2 недели | Sleep Mode: кластеризация, очистка, дистилляция, слияние | Пункт 5 |
| **Пункт 7** | 3 недели | State Algebra, KCA-цикл, иерархический HNSW, cognitive cycle | Пункт 6 |
| **Тестирование** | 2 недели | End-to-end тесты, бенчмарки | Пункт 7 |
| **Итого** | **16–18 недель** | Полноценный FCF | |

---

## 9. Что НЕ переносим из EVA-Ai (осознанные решения)

| Компонент EVA-Ai | Почему не берём |
|-----------------|-----------------|
| **OpenVINO** | FCF идёт через PyTorch для прямого контроля градиентов и весов |
| **FractalGraphV2 (SQLite)** | FCF хранит состояния в FAISS — это ближе к архитектуре «матрица как память» |
| **36 фиксированных слоёв Qwen3** | FCF растёт сам, начиная с одного слоя |
| **Flask Web GUI** | Пока не нужно; FCF — сначала консольный/Core API |
| **Двойная событийная система** (EventBus + EventSystem) | Избыточно для FCF; достаточно одного EventBus |
| **CoreBrain God Object** (10+ миксинов) | Архитектурная ошибка EVA-Ai — не повторять |
| **TCM / ScenarioTCM** | FCF использует FAISS-хранилище как краткосрочную память |

---

## 10. Что УЛУЧШАЕМ в FCF относительно EVA-Ai

| Проблема EVA-Ai | Решение в FCF |
|-----------------|--------------|
| **God Object** (CoreBrain) | Чёткое разделение: `PrimordialLayer` владеет своими компонентами, `FCFSystem` — только оркестрация слоёв |
| **Дублирование кода** (38 мёртвых модулей) | Монорепо без мёртвого кода с первого дня |
| **Разные размерности GNN/LoRA/LLM** | Единая размерность `d_model=2560` через всю систему |
| **Отсутствие этики** | `EthicsFilter` встроен в SRG с первого пункта |
| **Hardcoded пути** | Вся конфигурация в `config.json` |

---

## 11. Структура репозитория (конечная)

```
FCF/
├── fcf/
│   ├── __init__.py
│   ├── config.py                 # FCFConfig
│   ├── primordial_layer.py       # PrimordialLayer
│   ├── transformer.py            # TransformerBlock, Attention, SwiGLU
│   ├── state_storage.py          # StateStorage (FAISS)
│   ├── srg.py                    # SemanticRelevanceGate
│   ├── ethics_filter.py          # EthicsFilter (5 аксиом)
│   ├── meta_memory.py            # MetaMemory
│   ├── growth_controller.py      # GrowthController
│   ├── curiosity_loop.py         # CuriosityLoop
│   ├── data_manager.py           # DataManager (Wikipedia, Saiga, RuBQ, ConceptNet)
│   ├── domain_registry.py        # DomainRegistry + DomainRule
│   ├── lora_adapter.py           # LoRAAdapter
│   ├── domain_trainer.py         # Обучение LoRA-адаптеров
│   ├── recursive_processor.py    # Рекурсивная обработка (Пункт 5)
│   ├── layer_crystallizer.py     # Кристаллизация новых слоёв
│   ├── sleep_mode.py             # SleepMode (консолидация)
│   ├── cluster_engine.py         # HDBSCAN кластеризация
│   ├── distillation.py           # Дистилляция LoRA → базовые веса
│   ├── state_algebra.py          # StateAlgebra (сумма, масштаб, вычитание, кросс-аттеншн)
│   ├── projector.py              # Proj_ℳ (автоэнкодер)
│   ├── kca_engine.py             # KCAEngine + ConvergenceController
│   ├── hnsw_index.py             # Иерархический HNSW (3 уровня)
│   ├── cognitive_cycle.py        # Трёхфазный когнитивный цикл
│   ├── event_bus.py              # EventBus (один, не два как в EVA-Ai)
│   ├── online_trainer.py         # Фоновый тренер (адаптация из EVA-Ai)
│   ├── tokenizer_utils.py        # BPE токенизатор
│   └── utils.py                  # save/load, Xavier init, helper-функции
├── tests/
│   ├── test_transformer.py
│   ├── test_state_storage.py
│   ├── test_srg.py
│   ├── test_ethics_filter.py
│   ├── test_growth_controller.py
│   ├── test_curiosity_loop.py
│   ├── test_domain_registry.py
│   ├── test_lora_adapter.py
│   ├── test_recursive_processor.py
│   ├── test_sleep_mode.py
│   ├── test_kca_engine.py
│   ├── test_state_algebra.py
│   └── test_cognitive_cycle.py
├── config.json
├── requirements.txt
├── run.py
└── README.md
```

---

## 12. Первый запуск (цель Пункта 1)

```bash
# Установка
pip install -r requirements.txt

# Запуск — создание PrimordialLayer и проверка работоспособности
python run.py --init

# Ожидаемый вывод:
# [FCF] PrimordialLayer создан: d_model=2560, num_heads=32
# [FCF] StateStorage: FAISS IndexFlatIP (2560d)
# [FCF] SRG: w_sim=0.4, w_ent=0.3, w_eth=0.3
# [FCF] EthicsFilter: 5 аксиом загружено
# [FCF] GrowthController: width_thr=0.5, depth_thr=0.3
# [FCF] CuriosityLoop: threshold=10
# [FCF] Тестовый прямой проход: OK (input=(1,16), output=(1,16,50257))
# [FCF] Система готова к самообучению (Пункт 2)
```

---

*Документ создан на основе:*
- *FCF: Пункты 1–7 (дизайн-документы)*
- *FCF New.txt (консолидированный том)*
- *EVA-Ai: анализ боевой реализации (200+ файлов, 50K строк)*
