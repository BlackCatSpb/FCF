# Архитектурный план — Hardcode → адаптивная динамическая архитектура

## Философия

Максимум адаптивности, минимум хардкода. Всё, что может быть динамическим, должно быть динамическим — в пределах корректной работы VSA-архитектуры.

---

## Итоги анализа хардкода

**~200+ хардкод-значений.** Категории:

| Категория | Кол-во | Критичность |
|-----------|--------|-------------|
| Размерности (768/2048/1024) | ~15 файлов | Критичная |
| Seeds (28 вхождений, 4 разных seed) | 28 | Высокая |
| Subspace ratios + architectural thresholds | ~20 | Высокая |
| Формульные коэффициенты (RRF, гормоны, θ, PMI) | ~50 | Средняя |
| Пути к файлам | ~25+ | Средняя |
| Дефолты функций (дублируют конфиг) | ~30 | Низкая |

---

## Фаза 0: EnvironmentResolver — единый источник путей

### Проблема
Имя модели `bpe_ru_146k.model` — **10× дублируется** в 8 файлах. 3 разных паттерна конструирования пути. При смене модели нужно менять 8 файлов.

### Решение
`EnvironmentResolver` — единый класс для всех путей:

```python
class EnvironmentResolver:
    def __init__(self, base_dir=None, model_name=None, data_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = data_dir or os.environ.get('FCF_DATA_DIR',
                                                   os.path.join(self.base_dir, 'real_data'))
        self.model_name = model_name or os.environ.get('FCF_MODEL_NAME')

    @property
    def sp_model_path(self):
        # Авто-детект: если model_name не задан, ищем первый *.model в data_dir
        if self.model_name:
            return os.path.join(self.data_dir, self.model_name)
        models = glob(os.path.join(self.data_dir, '*.model'))
        return models[0] if models else None

    @property
    def morph_vocab_path(self): ...
    @property
    def concept_space_path(self): ...
    @property
    def syntax_lattice_path(self): ...
    @property
    def antonyms_path(self): ...
    @property
    def qwen_knowledge_path(self): ...
```

### Затрагиваемые файлы
`fcf_config.py`, `morph_vocab.py`, `crystal_generator.py`, `concept_space.py`, `stdp_trainer.py`, `qwen_knowledge.py`, `modeling_fcf.py`, `tokenization_fcf.py`, `nn_check.py`, `check_dim.py`, `checkpoint_manager.py`, `syntax_lattice.py`

### Риски
- Совместимость с HF `modeling_fcf.py` — пути должны резолвиться относительно `pretrained_model_name_or_path`
- Fallback: если файл не найден → понятная ошибка с именем ожидаемого файла

---

## Фаза 1: AdaptiveDimensionResolver — динамические размерности

### Проблема
4 жёстких размерности: `vec_dim=768`, `latent_dim=2048`, `entity_dim=2048`, `harm_dim=2048`. entity_dim == harm_dim == latent_dim (все 2048) — дублирование.

### Решение

**Единственный вход:** `vocab_size` от SentencePiece.

```python
class AdaptiveDimensionResolver:
    def __init__(self, vocab_size, vram_limit_mb=2048):
        self.vocab_size = vocab_size

        # SNR требование: D >= 10 * log2(V) для надёжного unbind
        self.min_vec_dim = int(10 * math.log2(vocab_size))

        # VRAM лимит: V * D * 6.34 bytes (vectors + codes)
        self.max_vec_dim = int(vram_limit_mb * 1024 * 1024 / (vocab_size * 6.34))

        # vec_dim: ближайшая степень 2 в диапазоне [min, max]
        self.vec_dim = self._power_of_2(min(self.max_vec_dim, 2048))

        # latent_dim: vec_dim * ratio (2.67 для 768→2048)
        self.latent_ratio = 2.67  # начальный, адаптируется
        self.latent_dim = int(self.vec_dim * self.latent_ratio)

        # entity_dim == harm_dim == latent_dim
        self.entity_dim = self.latent_dim

    @property
    def grid_shape(self):
        return VSAGrid.factorize(self.vec_dim)
```

**Проверка GPU-совместимости:**
- `_vecs_t: (V, D, fp16)` — авто-размер
- `_codes_t: (V, latent_dim, bf16)` — авто-размер
- `_fused_buf: (min(V, 4096), D+1)` — авто-размер
- `basis.T @ basis ≈ eye(D)` — уже параметризовано

**Отказ от entity_dim / harm_dim:**
- Harmonizer и EntityField работают напрямую с latent_dim
- Projection (768→2048) в EntityField убирается — все уровни в одном пространстве
- `_to_dim()` — упрощается до identity (или assert)

### Риски
1. **VSAGrid.factorize(D)** — протестирован только для 768 и 64. Для произвольного D (например 512, 1024) может не найти разложение на множители ≤ 8. Нужен fallback: `factorize_with_padding(dim)` — дополняет до ближайшего раскладываемого.
2. **GPU VRAM** — на 2GB карте при V=146K и D=1024: `(146000*1024*2 + 146000*2730*2) ≈ 1.1 GB`. Помещается, но впритык. Нужен флаг `vram_limit_mb` с запасом.

---

## Фаза 2: SeedRegistry — единый источник seed

### Проблема
28 вызовов `RandomState()`, 4 разных seed (42, 0, 1, 7). При смене master_seed воспроизводимость ломается.

### Решение
```python
class SeedRegistry:
    def __init__(self, master_seed=42):
        self.master_seed = master_seed
        self._requested = {}  # name → seed

    def get(self, name):
        if name in self._requested:
            return self._requested[name]
        seed = (self.master_seed + hash(name)) % (2**31 - 1)
        self._requested[name] = seed
        return seed

    def rng(self, name):
        return np.random.RandomState(self.get(name))
```

**Именованные seed:**
- `basis` — FractalField basis init
- `field_bits` — FractalField W_proj init
- `entity_roles` — EntityField role vectors
- `entity_proj` — EntityField projection
- `harm_roles` — Harmonizer role vectors
- `fluctuate` — FractalField fluctuation
- `rng` / `item_rng` / `inhibit_rng` — ConceptSpace RNGs
- `rescued` — Cluster repulsion fallback
- `morph_init` — Morpheme vector init
- `residue` — ResidueEncoder init
- `vsa_utils` — Random masks
- `vsa_attention` — Head roles
- `semantic_bootstrap` — Bootstrap RNG
- `lsh_index` — LSH permutation tables
- (`test_rng*` — для тестов, через seed registry с `test_` prefix)

**Правило:** `rng(name)` → `RandomState(master_seed + hash(name))`. Для тестов: можно переопределить `master_seed` для воспроизводимости конкретного сценария.

---

## Фаза 3: AdaptiveArchitectureController — динамические соотношения

### Проблема
`l_c/l_a/l_m` = 60%/25%/15% — жёсткая формула `*3//5 / //4 / остаток`. Пороги роста/прунинга — числовые константы.

### Решение

**Начальные соотношения — из конфига:**
```python
@dataclass
class SubspaceConfig:
    l_c_ratio: float = 0.6    # ~φ²/(φ²+φ+1)
    l_a_ratio: float = 0.25   # ~φ/(φ²+φ+1)
    l_m_ratio: float = 0.15   # ~1/(φ²+φ+1)
```

**Адаптивный контроллер:**
```python
class AdaptiveArchitectureController:
    def __init__(self, config, dim_resolver):
        self.ratios = SubspaceConfig()
        self.dim_resolver = dim_resolver

    def update(self, codes):
        """Обновить соотношения на основе статистик кодов."""
        # Density z_c
        z_c_active = np.mean(np.abs(codes[:, :self.l_c]) > 1e-4, axis=1)
        mean_density = np.mean(z_c_active)

        # Если плотность ниже цели → увеличить l_c (больше ёмкости identity)
        if mean_density < self.l1_target_density * 0.5:
            self.ratios.l_c_ratio = min(self.ratios.l_c_ratio * 1.05, 0.75)

        # Если плотность выше цели → уменьшить l_c
        elif mean_density > self.l1_target_density * 2.0:
            self.ratios.l_c_ratio = max(self.ratios.l_c_ratio * 0.95, 0.3)

        # Пересчитать l_a и l_m
        remaining = 1.0 - self.ratios.l_c_ratio
        self.ratios.l_a_ratio = remaining * 0.6  # 60% остатка на attention
        self.ratios.l_m_ratio = remaining * 0.4  # 40% на meta
```

**Динамические пороги:**
```python
@property
def density_threshold_grow(self):
    """Авто-порог: 90-й перцентиль плотности всех концептов."""
    densities = per_concept_density(all_codes)
    return np.quantile(densities, 0.9)

@property
def density_threshold_prune(self):
    """Авто-порог: 10-й перцентиль."""
    return np.quantile(densities, 0.1)
```

### Затрагиваемые константы
- `l_c/l_a/l_m` ratios (6 вхождений)
- `_density_threshold_grow = 0.15`
- `_density_threshold_prune = 0.01`
- `l1_target_density = 0.08`
- `_growth_factor = 1.5`
- `_sector_depths = [4, 10, 20]`

---

## Фаза 4: Formula coefficients → config + adaptive

### 4.1 RRF weights (генерация)

**Сейчас:** `graph=0.7, syntax=0.15, hdc=0.10, vec=0.15, prior=0.02` — жёстко в `crystal_generator.py:824-833`.

**Решение:**
```python
@dataclass
class RRFConfig:
    graph_weight: float = 0.7
    syntax_weight: float = 0.15
    hdc_weight: float = 0.10
    vector_weight: float = 0.15
    prior_weight: float = 0.02
    adaptation_rate: float = 0.01  # скорость адаптации весов

# Адаптация: на валидации считаем acc@1 каждой ручки
def adapt_rrf_weights(weights, signal_accuracies, temperature=3.0):
    """Softmax взвешивание по качеству каждой ручки."""
    logits = np.array([w * acc * temperature
                      for w, acc in zip(weights, signal_accuracies)])
    return softmax(logits)
```

### 4.2 Гормональная система

**Сейчас:** десятки числовых коэффициентов в формулах `hormonal_system.py:30-190`.

**Решение:** вынести ВСЕ числа в `hormonal_coeffs` config:
```python
@dataclass
class HormonalCoeffs:
    # Baselines
    da_baseline: float = 0.5
    ht_baseline: float = 0.5
    na_baseline: float = 0.3
    ach_baseline: float = 0.5

    # Decays
    tonic_decay: float = 0.95
    phasic_decay: float = 0.7

    # DA formula
    da_coherence_strength: float = 0.05
    da_extrinsic_match: float = 0.5
    da_extrinsic_mismatch: float = -0.3
    da_novelty_strength: float = 0.2

    # 5HT formula
    ht_target_adapt: float = 0.3
    ht_match_scale: float = 0.4

    # NA formula
    na_baseline_part: float = 0.2
    na_surprise_scale: float = 0.5
    na_confidence_scale: float = 0.3

    # ACh formula
    ach_novelty_scale: float = 0.5
    ach_uncertainty_scale: float = 0.3
    ach_drift_up: float = 0.15
    ach_drift_down: float = 0.1

    # Modulation
    da_temperature_strength: float = 0.9
    na_beam_strength: float = 0.5
```

Сами формулы остаются в коде (они — алгоритм, не конфиг). Коэффициенты — из config.

### 4.3 Прочие формульные константы

- **θ-распад:** `theta_tau=12.0, tau*3.0` — уже в config, нормально
- **PMI mapping:** `pmi/2.0 + 0.2, clamp(2.0)` — вынести в `pmi_mapping_slope, pmi_mapping_intercept, pmi_mapping_max`
- **Anti-repetition:** `exp(-0.3 * count)` — `antirep_penalty=0.3`
- **Edge weight (graph):** `max(0.20, 1.0 - min(ppmi/8.0, 1.0) * 0.7)` — `edge_weight_min=0.20, edge_ppmi_max=8.0, edge_weight_strength=0.7`
- **Target boost:** `5.0 * (1.0 - theta * 0.5)` — `target_boost_scale=5.0, target_boost_decay=0.5`
- **Novelty freq cap:** `/50` — `novelty_freq_cap=50`

---

## Фаза 5: VSA-native SemanticPiece — многоуровневая токенизация

### Проблема
SentencePiece BPE — чёрный ящик, не знающий про VSA-уровни. Harmonizer пытается восстановить морфемы через хардкод-списки префиксов.

### Решение

**BPE остаётся для первичной токенизации (инициализация).** VSA-уровни надстраиваются поверх:

```
BPE (инициализация)
    ↓ CID
Character VSA (CharEnvelope)
    ↓ STDP char→char bind
Morph VSA (новый: обучение морфем через STDP)
    ↓ STDP morph→word bind
Word VSA (Harmonizer, без хардкода)
    ↓ EntityField
Sentence VSA
    ↓ EntityField
Paragraph / Knowledge Cluster VSA
```

**Char→Morph обучение:**
- После BPE токенизации: каждый CID → текст → последовательность Unicode codepoints
- CharEnvelope строит char vectors
- STDP на char bi-grams: `bind(c₁, ρ(c₂))` → "soft morph token"
- Если char→char bind стабильно высок (когезия) → фиксируется как morph vector
- Popout через LSH: похожие char n-grams → один morph vector

**Morph→Word обучение:**
- Harmonizer получает морфемы не из хардкод-списка, а из STDP char→morph
- `word = compose([(morph₁, ROOT), (morph₂, SUFFIX), ...])`
- Роль-векторы (ROOT/PREFIX/SUFFIX) — обучемые, не фиксированные
- STDP тюнит: word_vector → pull к compose(word) → backprop ошибки на morph vectors

**Cross-level генерация:**
- `generate(topic=<cluster_vec>)` → top-down bias на sent level
- `generate(sent=<sent_vec>)` → top-down bias на word level
- EntityField binds: любой уровень может быть стартовой точкой
- Остальные уровни достраиваются через cross-level unbind

### Зависимости
- `CharEnvelope` — уже есть, доработка к STDP-обучению char→char
- `Harmonizer` — уже есть, убрать хардкод-декомпозицию, заменить на STDP
- `EntityField` — уже есть, обучение char↔morph (сейчас только char↔word)
- BPE — остаётся для seed vectors

### Риски
- **Качество морфем без хардкода:** STDP может не найти русские морфемы в char-последовательностях. Нужен warm-start: инициализировать morph vectors из существующего `_decompose_word`, потом дообучать.
- **Vocab explosion:** каждый char n-gram → morph vector → 146K + десятки тысяч. Нужен LSH-index для дедупликации (уже есть `LSHIndex`).
- **Скорость:** char-level STDP на каждом батче — дорого. Нужен sampling: только для новых/редких CIDs.

---

## Фаза 6: Единообразие дефолтов

### Проблема
Все функции дублируют дефолты из конфига: `pmi_strength=1.0, neg_samples=1, ...`. При изменении конфига сигнатуры не синхронизируются.

### Решение
```python
# Везде:
def train_from_text(self, text, base_lr=None, ...):
    base_lr = base_lr if base_lr is not None else self.config.get('learning_rate', 0.1)
    pmi_strength = pmi_strength if pmi_strength is not None else self.config.get('pmi_strength', 1.0)
    ...
```

Паттерн: `None → config.get(key, fallback)`. Никаких жёстких дефолтов в сигнатурах, кроме `None`.

---

## Итоговый порядок реализации

| Фаза | Что | Затрагивает файлов | Сложность | Зависимости |
|------|-----|--------------------|-----------|-------------|
| **0** | EnvironmentResolver (пути) | 12 | ☆☆ | Нет |
| **1** | AdaptiveDimensionResolver | 15 | ☆☆☆ | Фаза 0 (путь к SP) |
| **2** | SeedRegistry | 28 | ☆ | Фаза 1 (DimsResolver как singleton) |
| **3** | AdaptiveArchitectureController | 6 | ☆☆☆ | Фаза 1 |
| **4** | Formula coefficients | ~15 | ☆☆ | Фаза 0, 3 |
| **5** | SemanticPiece (VSA tokenizer) | ~10 | ☆☆☆☆☆ | Фазы 0-4 |
| **6** | Единообразие дефолтов | ~20 | ☆ | Фаза 0 |

**Рекомендуемый порядок:** 0 → 1 → 2 → 6 → 3 → 4 → 5

Фазы 0-2-6 — "housekeeping" (можно быстро), фазы 3-4 — "гибкость" (средне), фаза 5 — "архитектурная перестройка" (долго).

---

## Пограничные случаи и тестирование

### Для Фазы 1 (размерности)
- `vocab_size < 1000`: минимальный D = 64, но SNR ≈ √64 = 8 — мало для надёжного unbind. **Решение:** `min_vec_dim = max(64, int(10*log2(V)))` + warning если D < 128
- `vocab_size > 1M`: D > 200, VRAM лимит. **Решение:** `max_vec_dim` по VRAM
- GPU VRAM < 1GB: **Решение:** принудительный CPU fallback + warning

### Для Фазы 2 (seeds)
- `hash(name)` может дать коллизию? Проверка: `hash('basis')`, `hash('entity_roles')` — все разные. Добавить assert в `get()`
- Воспроизводимость тестов: `SeedRegistry(master_seed=test_seed)` в conftest

### Для Фазы 3 (adaptive ratios)
- Если все концепты имеют density=0 (cold start): `quantile(densities, 0.9)` может быть 0. **Решение:** fallback на начальные проценты, если концептов < 100

### Для Фазы 5 (SemanticPiece)
- BPE модель другого языка: механизм char→morph→word остаётся (работает на Unicode), но начальные морфемы из BPE будут другими
- Пустой словарь: STDP не может обучиться без данных. **Решение:** BPE bootstrapping обязателен
- Очень редкий чар (Unicode > 0xFFFF): суррогатные пары — CharEnvelope должен их обрабатывать

---

## Критерии готовности

- [ ] Все 294 тестов проходят после каждой фазы
- [ ] `FCFConfig` — единственное место с дефолтами
- [ ] `EnvironmentResolver` — единственное место с путями
- [ ] `AdaptiveDimensionResolver` вычисляет размерности один раз при старте
- [ ] `SeedRegistry` — все RandomState через него
- [ ] `AdaptiveArchitectureController` активен через N батчей
- [ ] Формульные коэффициенты читаются из config
- [ ] BPE + VSA-уровни работают в связке
- [ ] Дефолты в сигнатурах — только `None`
