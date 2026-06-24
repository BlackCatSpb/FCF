# Архитектурный план — Hardcode → адаптивная динамическая архитектура

## Философия

Максимум адаптивности, минимум хардкода. Всё, что может быть динамическим, должно быть динамическим — в пределах корректной работы VSA-архитектуры.

---

## Статус завершённых фаз

| Фаза | Статус | Коммит |
|------|--------|--------|
| **0** EnvironmentResolver (пути) | ✅ | Phase 0 |
| **1** AdaptiveDimensionResolver + _hrr_bind fix | ✅ | Phase 1 |
| **2** SeedRegistry (28 RNG) | ✅ | Phase 2 |
| **3** AdaptiveArchitectureController | ✅ | Phase 3 |
| **4** FormulaCoefficients датакласс | ✅ | Phase 4 |
| **5** SemanticPiece (VSA токенизация) | ✅ | Phase 5 |
| **6** Defaults pattern (None → config) | ✅ | Phase 6 |

---

## Фаза 7: Оставшийся хардкод (Phase 7)

### Выводы повторного аудита (24.06.2026)

**Ключевое открытие:** `FormulaCoefficients` содержит **все** необходимые коэффициенты (35 полей для гормонов, 10 для STDP, 15 для RRF/θ/PMI), но файлы `hormonal_system.py` и `stdp_trainer.py` **их не читают** — используют хардкод. Это не просто "значения не в конфиге", а **семантический баг**: конфиг есть, он игнорируется.

**Итого:** ~120 значений в ~15 файлах, из них ~30 — баги (конфиг есть, не читается), ~40 — нет полей в `FCFConfig`, ~50 — скрипты/дублирование.

| Категория | Кол-во | Приоритет |
|-----------|--------|-----------|
| Баги: конфиг есть, код не читает | ~30 | **P0** |
| Нет полей в FCFConfig | ~40 | **P1** |
| Scripts / дублирование | ~50 | **P2** |

---

## P0: Чтение существующего конфига (2 файла, ~30 значений)

### P0-A: `hormonal_system.py` не читает FormulaCoefficients

**Файл:** `eva/symbolic/hormonal_system.py`

**Симптом:** все 35 коэффициентов в `FormulaCoefficients` (строки 398–430 `fcf_config.py`) существуют, но `HormonalSystem.__init__` и `update()` используют хардкод.

**Что менять:**
```python
class HormonalSystem:
    def __init__(self, config=None):
        _fc = (config.formula if config else FCFConfig().formula)
        self.dopamine = _fc.da_baseline       # вместо 0.5
        self.serotonin = _fc.ht_baseline      # вместо 0.5
        self.noradrenaline = _fc.na_baseline  # вместо 0.3
        self.acetylcholine = _fc.ach_baseline # вместо 0.5
        self.tonic_decay = _fc.tonic_decay    # вместо 0.95
        self.phasic_decay = _fc.phasic_decay  # вместо 0.7

    def update(self, ..., _fc=None):
        _fc = _fc or FormulaCoefficients()
        da_coherence = _fc.da_coherence_strength       # вместо 0.05
        da_curiosity = novelty * _fc.da_curiosity_strength  # вместо * 0.4
        da_mastery = max(0, delta_match) * _fc.da_mastery_strength  # вместо * 0.5
        ...
```

**Файлы для изменений:** `hormonal_system.py`, `crystal_generator.py` (передача config в HormonalSystem)

### P0-B: `stdp_trainer.py` STDP формулы не читают FormulaCoefficients

**Файл:** `eva/symbolic/stdp_trainer.py`

**Симптом:** `_build_pairs` использует `freq_weight_log_scale=0.15`, `field_weight_log_scale=2.0`, `field_weight_cap=3.0`, `field_weight_floor=0.1`, `hormonal_mod_baseline=0.5`, `hormonal_mod_scale=0.5` — все есть в `FormulaCoefficients` (строки 363–373 fcf_config.py), но код читает `_fc` только для PMI.

**Что менять:**
```python
_fc = FormulaCoefficients()
# freq_weight:
freq_weight = 1.0 / (1.0 + math.log(...) * _fc.freq_weight_log_scale)
# field_weight:
fw = min(1.0 + math.log(overlap + 1) * _fc.field_weight_log_scale, _fc.field_weight_cap) if overlap > 0 else _fc.field_weight_floor
# hormonal modulation:
lr *= (_fc.hormonal_mod_baseline + gen.hormones.acetylcholine * _fc.hormonal_mod_scale)
```

---

## P1: Добавить поля в FCFConfig + читать (4 файла, ~40 значений)

### P1-A: `crystal_generator.py` — Graph search и Branch параметры

**Новые поля в FCFConfig:**
```python
# Graph search defaults
graph_search_B: float = 1.2
graph_search_max_candidates: int = 30
graph_search_max_depth: int = 5
graph_search_connections_topk: int = 8
graph_search_syn_preds_limit: int = 80
graph_search_hdc_k: int = 30
graph_search_hdc_score_min: float = 0.05
graph_search_sector_k: int = 40
graph_search_sector_depth: int = 1
graph_search_focal_k: int = 20
graph_search_focal_sample_size: int = 500
graph_search_sim_threshold: float = 0.05

# Branch params
branch_antirep_window: int = 6
branch_n_candidates_base: int = 15
branch_overlap_log_scale: float = 0.1
branch_conf_scale: float = 0.5
branch_adaptive_bw_min_ratio: float = 0.5
branch_eos_punct: str = ".,!?;:…—–"
```

**Файлы:** `fcf_config.py`, `crystal_generator.py`

### P1-B: `parameter_optimizer.py` — Plateau/Metric/Detector

**Новые поля в FCFConfig:**
```python
# MetricBuffer defaults
metric_maxlen_primary: int = 10
metric_maxlen_secondary: int = 8
metric_maxlen_tiny: int = 6

# Plateau detection
plateau_patience: int = 3
plateau_rel_thresh_default: float = 0.005
plateau_rel_thresh_ppl: float = 0.002
plateau_rel_thresh_acc1: float = 0.02

# ParameterOptimizer defaults
opt_flat_threshold: float = 0.002
opt_cos_trend_window: int = 5
opt_full_stuck_threshold: int = 5
opt_toward_default_rate: float = 0.03
opt_inh_threshold_fallback: float = 0.1

# PlateauDetector defaults
detector_window: int = 100
detector_patience: int = 20
detector_threshold_std: float = 0.5
detector_min_decay: float = 0.1
detector_recovery_factor: float = 0.05
detector_ema_alpha: float = 0.05
detector_decay_per_step: float = 0.01
```

**Файлы:** `fcf_config.py`, `parameter_optimizer.py`

### P1-C: `adaptive_controller.py` — SubspaceConfig + update()

**Новые поля в FCFConfig (дублируют SubspaceConfig + internal):**
```python
# SubspaceConfig defaults
subspace_l_c_ratio: float = 0.6
subspace_l_a_ratio: float = 0.25
subspace_l_m_ratio: float = 0.15
subspace_density_threshold_grow: float = 0.15
subspace_density_threshold_prune: float = 0.01
subspace_l1_target_density: float = 0.08
subspace_growth_factor: float = 1.5
subspace_sector_depths: list = field(default_factory=lambda: [4, 10, 20])

# AdaptiveController update()
subspace_density_epsilon: float = 1e-4
subspace_density_history_maxlen: int = 10000
subspace_warmup_updates: int = 10
subspace_adjust_up_rate: float = 1.03
subspace_adjust_up_max: float = 0.75
subspace_adjust_down_rate: float = 0.97
subspace_adjust_down_min: float = 0.3
subspace_redistribute_a_ratio: float = 0.6
subspace_redistribute_m_ratio: float = 0.4
```

**Файлы:** `fcf_config.py`, `adaptive_controller.py`

### P1-D: `concept_space.py` — Capacity лимиты

```python
# FractalField defaults
fractal_hdc_memory_max: int = 20000
fractal_init_z_c_active_pct: float = 0.03
fractal_init_z_c_active_min: int = 8
fractal_init_z_a_scale: float = 0.01
fractal_init_z_m_scale: float = 0.001
fractal_init_field_n_anchors: int = 1024
fractal_l1_density_window: int = 100
fractal_l1_adjust_rate: float = 0.1
fractal_l1_lambda_cap: float = 0.1
```

При этом `hdc_memory_max` уже используется как лимит — заменить чтением из config.

---

## P2: Централизация и скрипты (~50 значений)

### P2-A: Special token IDs (3 файла)

```python
# В FCFConfig:
special_pad_id: int = 0
special_bos_id: int = 1
special_eos_id: int = 2
```

Убрать глобальные `_BOS_ID = 1`, `_EOS_ID = 2` из `crystal_generator.py:47-48`.
В `configuration_fcf.py:31-33` и `tokenization_fcf.py:28-30` читать из config.

### P2-B: SeedRegistry — оставшиеся прямые вызовы

- `concept_space.py:364`: `np.random.RandomState(cid * 137 + 42)` → `_R.rng('init_concept')`
- `concept_space.py:472`: `np.random.RandomState(42 + self._capacity_growths)` → `_R.rng('field_collapse')`
- `crystal_generator.py:89`: `RNGRegistry(master_seed=42)` → читать `config.global_seed`

### P2-C: Скрипты

- `train_full.py`: `random.seed(42)` → `config.global_seed`; `N_FIELD_BITS = 512` → `config.field_bits`
- `inference.py`: `sample_size=500` → config; default seeds
- `filter_corpus.py`: regex thresholds (4, 2, 3, 10, 0.5) — если используются в тренировке, то в config
- `eval_checkpoint.py`, `visualize.py`: `seed=42` → config

---

## Итоговый порядок реализации Phase 7

| Шаг | Что | Файлы | Сложность | Зависимости |
|-----|-----|-------|-----------|-------------|
| **P0-A** | hormonal_system.py читает FC | 2 | ☆ | Нет (config уже есть) |
| **P0-B** | stdp_trainer.py читает FC | 2 | ☆ | P0-A |
| **P1-A** | crystal_generator graph/branch params | 2 | ☆☆ | Нет |
| **P1-B** | parameter_optimizer plateau/metric | 2 | ☆☆ | Нет |
| **P1-C** | adaptive_controller subspace params | 2 | ☆ | Нет |
| **P1-D** | concept_space capacity params | 2 | ☆ | Нет |
| **P2-A** | Special token IDs | 4 | ☆ | Нет |
| **P2-B** | Seeds через Registry | 3 | ☆ | Нет |
| **P2-C** | Scripts consolidation | 5 | ☆☆ | P2-A/B |

**Рекомендуемый порядок:** P0-A → P0-B → P1-A → P1-B → P1-C → P1-D → P2-A → P2-B → P2-C

**Критическая зависимость:** P0-A не зависит ни от чего; P0-B можно параллельно.

---

## Пограничные случаи

### HormonalSystem без config
`HormonalSystem()` создаётся в `crystal_generator.py:94` без config. Решение: `HormonalSystem(config=None)` → `FCFConfig().formula`.

### ParameterOptimizer MetricBuffer разных размеров
10/8/6 для разных метрик — не случайные числа, а эвристики под частоту обновления. Решение: NamedTuple в config: `metric_maxlen_map: dict = field(default_factory=lambda: {'mean_cos': 10, 'std_cos': 10, 'vec_ppl': 8, ...})`.

### SubspaceConfig дублирование
`SubspaceConfig` в `adaptive_controller.py` — отдельный dataclass, дублирующий часть FCFConfig. Решение: убрать `SubspaceConfig`, читать напрямую из `FCFConfig.subspace_*`.

### _BOS_ID/_EOS_ID в crystal_generator как глобалы
После централизации — убрать `_BOS_ID = 1`, `_EOS_ID = 2` на уровне модуля. Заменить на `self.config.special_bos_id`. В `generate()` и `_encode_input()` — проверка через config.

---

## Критерии готовности Phase 7

- [ ] Все тесты проходят (294 + новые)
- [ ] `hormonal_system.py` не содержит хардкод-коэффициентов — все из `FormulaCoefficients`
- [ ] `stdp_trainer.py` не содержит хардкод-коэффициентов STDP — все из `FormulaCoefficients`
- [ ] `crystal_generator.py` graph search читает все параметры из config
- [ ] `parameter_optimizer.py` plateau/metric читает все пороги из config
- [ ] `adaptive_controller.py` не имеет `SubspaceConfig` — читает из `FCFConfig`
- [ ] `concept_space.py` capacity лимиты читает из config
- [ ] Special token IDs — одно место `FCFConfig`, а не 3 файла
- [ ] Оставшиеся прямые `RandomState()` — через SeedRegistry

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
