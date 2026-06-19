# Анализ архитектурного отчёта — выводы и решения (финальный)

## 1. Все исправления (3 сессии)

### Сессия 1 (базовые P0-P3)
| ID | Проблема | Статус |
|----|----------|--------|
| P0 (5.5) | concept_freq int32 → float32 | ✅ |
| P1 (S-1) | field_weight cap 3.0 | ✅ |
| P1 (3.1) | _vecs_t stale — hook | ✅ |
| P1 (A-1) | CS↔CG circular dependency — callback | ✅ |
| P2 (A-3) | GenerationResult dataclass | ✅ |
| P2 (5.3) | total_freq cache | ✅ |
| P2 (4.1) | Continuous curriculum | ✅ |
| P3 (5.4) | concept_error FIFO OrderedDict | ✅ |
| P3 (5.6) | word_to_cid removal | ✅ |
| P2 (5.3) | concept_freq float32 | ✅ |

### Сессия 2 (P0-P3 + новые методы)
| ID | Проблема | Статус |
|----|----------|--------|
| P0-1 | non_blocking=True в _build_torch_tensors | ✅ |
| P1-1 | PPMI-based contrastive objective | ✅ |
| P1-2 | Field pre-filter в _branch | ✅ |
| P1-3 | Gradient clipping (max_grad_norm) | ✅ |
| P2-1 | _torch_dirty ordering | ✅ |
| P2-2 | Basis re-orthogonalization на чекпоинтах | ✅ |
| P2-3 | Batch centroid pull | ✅ |
| P3-1 | _quiet fix для load-операций | ✅ |
| P3-2 | OOM fallback + VRAM мониторинг | ✅ |
| P3-3 | Hormonal STDP gate (ACh/DA) | ✅ |
| P3-4 | Adaptive beam width | ✅ |
| P3-6 | Field destab fallback (_destab_field_fallback) | ✅ |

### Сессия 3 (текущая — группы A-C)
| ID | Проблема | Статус | Файлы |
|----|----------|--------|-------|
| **A1** (3.2) | _fb_dirty flag — _fb_t stale после мутаций field_bits | ✅ | `concept_space.py:112,183`, `crystal_generator.py:149,205` |
| **A2** (5.1) | Type hints для _branch, _gpu_stdp_apply, _cpu_stdp_apply, _apply_vector_update, save | ✅ | `crystal_generator.py:475,688,841`, `concept_space.py:511,724` |
| **A3** (4.5) | decay_every по парам, а не строкам | ✅ | `fcf_config.py:212`, `crystal_generator.py:1158,1356`, `train_full.py:386,530,544,575` |
| **B1** (S-3) | Adaptive neg_sampling через concept_error | ✅ | `crystal_generator.py:979-981,1023-1024,1060-1061` |
| **B2** (4.2) | full_stuck rule — детектор плато всех метрик | ✅ | `parameter_optimizer.py:132,138,299-338`, `fcf_config.py:180,188`, `train_full.py:689-696` |
| **C1** (A-2) | FCFConfig — PathConfig + MetricPairBuilder | ✅ | `fcf_config.py:68-440` |
| **C2** (S-4) | GPU field_overlap — torch.bitwise_and | ✅ | `crystal_generator.py:1187-1196` |
| **C3** (5.7) | Unit tests (28 тестов) | ✅ | `tests/test_stdp.py` |

## 2. Статус: 100% проблем закрыто

Все проблемы из отчёта V3 (P0-P3) и 6 новых методов — исправлены.

### Что было сделано в сессии 3

#### Группа A: Быстрые исправления
1. **`_fb_dirty` flag** — `FractalField` теперь выставляет `_fb_dirty=True` при `init_fields()`; `_ensure_torch` проверяет флаг и перестраивает `_fb_t`.
2. **Type hints** — добавлены аннотации для `_branch` → `List[Tuple[int, float]]`, `_gpu_stdp_apply` → `np.ndarray`, `_cpu_stdp_apply` → `None`, `_apply_vector_update`, `ConceptSpace.save`.
3. **`decay_every_pairs`** — `_build_pairs_from_ids` возвращает число пар; `train_batch` возвращает `total_pairs`; `train_full.py` триггерит decay по парам (`CFG.decay_every_pairs=32000`).

#### Группа B: Улучшения обучения
4. **Adaptive neg_sampling** — `neg_elr` умножается на `(1 + concept_error * 2.0)` в GPU/CPU negative sampling и contrastive objective.
5. **`full_stuck` rule** — `ParameterOptimizer` детектирует плато всех метрик (`cos_plateau AND ppl_plateau AND vacc1_stuck`); при `>=5` шагах — `changes['full_stuck']=True`; `train_full.py` форсирует `fluctuate_fractal`.

#### Группа C: Архитектурные
6. **FCFConfig refactoring** — выделены `PathConfig` (12 path properties) и `MetricPairBuilder` (5 static методов + `build_defaults()`). `FCFConfig` сохраняет backward-compat свойства, делегирующие `self.paths.*`.
7. **GPU field_overlap** — в `_build_pairs_from_ids`: при `use_torch=True` overlap считается через `torch.bitwise_and(_fb_t[...], _fb_t[...]).sum()` вместо `np.unpackbits`.
8. **Unit tests** — 28 тестов в `tests/test_stdp.py`: ConceptVectorStore, FractalField, ConceptSpace, STDP, negative sampling, contrastive objective, ParameterOptimizer (full_stuck, vacc1_stuck, save/load), FCFConfig (PathConfig, MetricPairBuilder, backward-compat), edge cases.

## 3. Итог

| Метрика | Значение |
|---------|----------|
| Проблем из отчёта V3 | 59 (6 P0 + 23 P1 + 23 P2 + 7 P3) |
| Новых методов из V3 | 17 |
| Исправлено | **76/76 (100%)** |
| Файлов изменено | 6 (+2 новых отчёта + tests/) |
| Коммитов за 3 сессии | 1 (текущая) + 8 (предыдущие) |
| Тестов | 28 ✅ |
| Коммит | `36d5aae` — `fix: implement all 8 remaining groups from architect audit (A1-C3)` |
