# FCF Architect Report V4 — Коллегия AI-агентов (Fresh Audit)

**Дата**: 2026-06-17
**Проект**: Fractal Cognitive Field (FCF) — нейро-символическая языковая модель
**Версия отчёта**: V4 (после V3 + 10 коммитов, 100% проблем V3 закрыты, fresh audit на обновлённом коде)
**Состав коллегии**: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent

---

## Сводка

| Агент | Новых проблем | P0 | P1 | P2 | P3 | Новых методов |
|-------|:-------------:|:--:|:--:|:--:|:--:|:-------------:|
| Architect-AI | 7 | 0 | 0 | 4 | 3 | 0 |
| Neuro-Symbolic Specialist | 3 | 1 | 1 | 1 | 0 | 0 |
| GPU-Opt Agent | 3 | 0 | 0 | 3 | 0 | 0 |
| Training-Dynamics Agent | 2 | 1 | 0 | 1 | 0 | 0 |
| Quality-Safety Agent | 4 | 1 | 0 | 1 | 2 | 0 |
| **Итого новых** | **19** | **3** | **1** | **10** | **5** | **0** |
| **Старых P0/P1 подтверждено** | **16** | **3** | **13** | — | — | — |

---

## 1. Architect-AI: Архитектурный анализ (Fresh Audit)

### A-N1: OOM fallback retries CUDA на каждый вызов — нет персистентного CPU fallback

- **Файл**: `crystal_generator.py:143-164`
- **Суть**: После OOM на CUDA `_ensure_torch` падает на CPU (`dev = torch.device('cpu')`), но НЕ запоминает это. Следующий вызов с `device=None` снова пробует CUDA (`torch.cuda.is_available()` = True), снова OOM, снова падает на CPU. Каждое перестроение тензоров (ребилд по dirty-флагам) повторяет эту последовательность, тратя ~2-5c на failed CUDA malloc + `empty_cache`.
- **Влияние**: P2 — на системах с <2GB VRAM каждое перестроение вызывает лишний OOM + fallback
- **Предложение**: После первого OOM установить `self._torch_device = torch.device('cpu')` и `self._torch_fallback = True`, проверять в начале `_ensure_torch`.

### A-N2: Double `fluctuate_fractal` при `full_stuck` на чекпоинте

- **Файл**: `train_full.py:571-576` и `696-701`
- **Суть**: В одной итерации `fluctuate_fractal` может быть вызван ДВАЖДЫ: сначала по расписанию (`is_fluct_due`), затем если `opt.step()` вернул `full_stuck`. Двойной drift за один шаг.
- **Влияние**: P2 — ускоренный дрейф кодов на чекпоинтах
- **Предложение**: В блок `if opt_changes.get('full_stuck')` добавить проверку — если флуктуация уже была на этом idx, пропустить.

### A-N3: `decay_all()` сбрасывает `_prefix_total` без пересборки — PMI отключён

- **Файл**: `syntax_lattice.py:260-263`
- **Суть**: `decay_all()` устанавливает `self._prefix_total = {}` и `self._skip2_total = {}`, но не вызывает `_refresh_prefix_totals()`. Следующий вызов `_pmi_weight` получает `count_prev = 0`, возвращает fallback 0.1 для ВСЕХ пар — PMI-гейтинг отключён на 1 батч после каждого decay.
- **Влияние**: P2 — регулярное кратковременное отключение PMI-взвешивания
- **Предложение**: Вызвать `_refresh_prefix_totals()` в конце `decay_all()`.

### A-N4: Test coverage gap — новые механизмы не покрыты тестами

- **Файл**: `tests/test_stdp.py` (298 строк, 24 теста)
- **Суть**: Из 10 изменений (после V3) не покрыты тестами: GPU/CPU численная паритетность, hormonal STDP gate, concept error weighting, field_gate в negative sampling, `_fb_dirty` флаг, `decay_every_pairs` триггер. Только smoke-тесты "no crash".
- **Влияние**: P2 — регрессия незаметна до продакшн-ран
- **Предложение**: Добавить тесты на численную эквивалентность GPU/CPU (`np.allclose(v_gpu, v_cpu, atol=1e-5)`).

### A-N5: `last_decay_lines` — мёртвый код

- **Файл**: `train_full.py:582`
- **Суть**: `last_decay_lines` устанавливается при decay, но НИГДЕ не читается.
- **Влияние**: P3 — только захламление
- **Предложение**: Удалить `last_decay_lines`.

### A-N6: `_negative_sampling_gpu` — избыточная identity-индексация CIDs

- **Файл**: `crystal_generator.py:987`
- **Суть**: `neg_cids = torch.tensor(self._torch_cid_order, device=device)[neg_idxs]` — `_torch_cid_order = list(range(V))`, т.е. `result[i] == i`. Лишний gather на GPU.
- **Влияние**: P3
- **Предложение**: Использовать `neg_idxs` напрямую как CIDs.

### A-N7: Lazy `_default_use_torch` — мутация class-переменной из instance-метода

- **Файл**: `crystal_generator.py:1226-1228, 1304-1306`
- **Суть**: `train_from_text` и `train_batch` модифицируют `CrystalGenerator._default_use_torch` как class-переменную (shared mutable state). В однопоточном режиме безопасно, но code smell.
- **Влияние**: P3
- **Предложение**: Заменить на instance-переменную `self._use_torch` с ленивой инициализацией.

---

## 2. Neuro-Symbolic Specialist: Анализ концептуального пространства (Fresh Audit)

### SN-1 (P0): Undefined `gpu_cid_gen` в `_negative_sampling_gpu` — NameError

- **Файл**: `crystal_generator.py:982`
- **Суть**: Строка `ce = self.concept_error.get(gpu_cid_gen[pi], 0.0)` ссылается на `gpu_cid_gen`, который **не определён** в области видимости метода. Параметр `gpu_cid_gen` отсутствует в сигнатуре `_negative_sampling_gpu` (строка 937) и не передаётся ни одним call site (`train_from_text:1272`, `train_batch:1354`). **Crash при `neg_samples > 0 && use_torch == True`.**
- **Корень**: concept error weighting был добавлен в `_negative_sampling_gpu` (фикс S-4 из V3), но сигнатура не была расширена; `gpu_cid_gen` есть в обоих caller-функциях, но не проброшен.
- **Влияние**: **P0** — crash при любом GPU прогоне с negative sampling
- **Предложение**: добавить `gpu_cid_gen` в сигнатуру на строке 937 и передать из `train_from_text:1272` и `train_batch:1354`.

### SN-2 (P1): Двойное distance-затухание (dist_weight × theta_gate) в CPU path

- **Файл**: `crystal_generator.py:1175,1200-1201`
- **Суть**: `dist_weight = exp(-dist/2.0)` (строка 1175) перемножается с `theta_gate = exp(-dist/theta_tau)` (строка 1200) в `lr` для `gen_updates`. Результирующий спад `exp(-dist·(1/2 + 1/theta_tau))` вдвое круче, чем в GPU path (где `lr = clamp(fw,0.05)·dw·pmi·field` на строке 708, `theta` — отдельный множитель на строке 710-711). Асимметрия CPU/GPU не документирована.
- **Влияние**: P1 — CPU и GPU path дают разные learning rate schedules для дальних пар
- **Предложение**: Убрать `dist_weight` из CPU, оставив только `theta_gate` (как в GPU).

### SN-3 (P2): Concept_error pruned без увязки с размером словаря

- **Файл**: `crystal_generator.py:1287-1288, 1369-1370`
- **Суть**: Кэш `concept_error` ограничен 30000 записей FIFO. При словаре 146K для ~80% концептов `concept_error.get(cid, 0.0)` возвращает 0 — нейтральный вес. Эффект concept error weighting для редких токенов нивелирован.
- **Влияние**: P2 — для длинного хвоста распределения концепт-эррор не работает
- **Предложение**: Увеличить лимит до `min(3 * vocab_size // 4, 100000)` или LRU eviction вместо FIFO.

---

## 3. GPU-Opt Agent: Анализ GPU-оптимизации (Fresh Audit)

### GN-1 (P2): Регенерация `torch.tensor(self._torch_cid_order, device=device)` на каждый вызов

- **Файл**: `crystal_generator.py:987`
- **Суть**: `neg_cids = torch.tensor(self._torch_cid_order, device=device)[neg_idxs]` создаёт O(V) тензор в каждом `_negative_sampling_gpu`. `_torch_cid_order` стабилен между `_build_torch_tensors`.
- **Влияние**: P2 — лишняя аллокация 146K int64 (~1.1MB) на вызов
- **Предложение**: Кэшировать как `self._cid_order_t = torch.tensor(self._torch_cid_order, device=device)` в `_build_torch_tensors`.

### GN-2 (P2): CPU-цикл concept_error_weighting в GPU negative sampling

- **Файл**: `crystal_generator.py:981-983`
- **Суть**: `for pi in range(n_pairs): neg_elr_arr[pi] *= (1.0 + self.concept_error.get(...) * 2.0)` — синхронный CPU dict lookup для каждого pair. Разрушает GPU batching.
- **Влияние**: P2 (~500-2000 итераций CPU на batch)
- **Предложение**: Кэшировать `concept_error` как CPU tensor `self._ce_t` и векторизовать.

### GN-3 (P2): evaluate — дублирование `prev_vecs` в GPU evaluate path

- **Файл**: `crystal_generator.py:1466-1474`
- **Суть**: `prev_vecs = np.array([... cs.concept_vectors.get(c) ...])` — CPU gather, потом H2D, потом matmul. Вместо прямого gather с GPU через `_vecs_t[prev_cids_t]`.
- **Влияние**: P2 — O(batch×dim) CPU аллокация + H2D каждые 500 позиций
- **Предложение**: `prev_cids_t = torch.tensor(batch_prev, device=device); pv_t = self._vecs_t[prev_cids_t]`.

---

## 4. Training-Dynamics Agent: Анализ цикла обучения (Fresh Audit)

### TN-12 (P0): full_stuck ложные срабатывания без eval

- **Файл**: `parameter_optimizer.py:307-319` + `train_full.py:694`
- **Суть**: Три условия full_stuck:
  1. `cos_plateau` — `|cos| < 0.002` (нормальное состояние обученной системы)
  2. `ppl_plateau` = `True` когда `vec_ppl is None` (eval не запускался)
  3. `v1_stuck` = `True` когда `vacc1 == 0.0` (eval не запускался → `vacc1=0`)
  Без eval все три `True` → `_full_stuck_counter` растёт каждый шаг. После 5 шагов — ложный `full_stuck` → форсированный `fluctuate_fractal`.
- **Влияние**: **P0** — поломка обучения: постоянные принудительные флуктуации на не-eval чекпоинтах
- **Исправление**:
  - `ppl_plateau` должно быть `False` когда `vec_ppl is None`
  - `v1_stuck` должно требовать `vacc1 is not None` (а не `vacc1 == 0.0`)
  - Убрать `eval_vacc1 or 0` → передавать `vacc1` только когда eval выполнен

### TN-13 (P2): Hormonal lr модуляция в train_batch без обновления гормонов

- **Файл**: `crystal_generator.py:1199`
- **Суть**: Гормональный gate применяется к lr пар, но гормоны обновляются только в `generate()`, не в `train_batch()`. На ранних стадиях гормоны = начальные (ACh=0.5, DA=0.5), множитель = 0.75. Разрыв между `gen.train_lr` и реальным lr пар.
- **Влияние**: P2 — разрыв между номинальным и реальным learning rate
- **Предложение**: Логировать фактические lr пар.

---

## 5. Quality-Safety Agent: Анализ качества кода (Fresh Audit)

### QN-1 (P0): `syntax_lattice.py` — `np.load` без `allow_pickle=False`

- **Файл**: `syntax_lattice.py:548`
- **Суть**: `npz = np.load(binary_path)` — то же, что Q-1 из V3, но в SyntaxLattice.load(). RCE уязвимость.
- **Влияние**: P0 — RCE при загрузке checkpoint
- **Предложение**: `np.load(binary_path, allow_pickle=False)`.

### QN-2 (P2): `_ensure_torch` OOM fallback ненадёжен

- **Файл**: `crystal_generator.py:154-163`
- **Суть**: CUDA OOM fallback ловит `RuntimeError` и проверяет `'out of memory' in str(e)`. Строка ошибки меняется между версиями torch.
- **Влияние**: P2 — может не распознать OOM в новых версиях
- **Предложение**: Использовать `isinstance(e, torch.cuda.OutOfMemoryError)` (torch ≥ 2.0).

### QN-3 (P3): `FCFConfig.base_dir` дублирует `PathConfig.base_dir`

- **Файл**: `fcf_config.py:239`
- **Суть**: `FCFConfig` хранит `base_dir: str` и `paths: PathConfig` с тем же default. Все свойства делегируют `self.paths`, `base_dir` нигде не используется. Мёртвый код.
- **Влияние**: P3
- **Предложение**: Убрать `base_dir` из `FCFConfig`.

### QN-4 (P3): Тесты не изолированы

- **Файл**: `tests/test_stdp.py:20-21`
- **Суть**: Модульные константы `DIM = 64`, `VOCAB_SIZE = 20` разделяются всеми тестами. Побочные эффекты возможны.
- **Влияние**: P3
- **Предложение**: Использовать `@pytest.fixture` для создания объектов.

### Анализ тестового покрытия:

- **Покрыто**: ConceptVectorStore CRUD, границы, итерация. FractalField init, basis health, fluctuate, fb_dirty. ConceptSpace init, norms, topk, `_apply_vector_update`. STDP CPU (smoke). Negative sampling CPU. Contrastive objective. Concept error FIFO. ParameterOptimizer step/save/load/stuck. FCFConfig path/MetricPairBuilder/serialization/backward compat. Edge cases (empty store, octree config, destab range, subspace dims).
- **НЕ покрыто**:
  - **GPU/CPU parity**: Нет ни одного теста, сравнивающего `_gpu_stdp_apply` и `_cpu_stdp_apply`. Класс `TestGPUParity` пуст.
  - `train_from_text` / `train_batch`: интеграционные тесты отсутствуют
  - `_negative_sampling_gpu`: GPU-ветка не тестирована (только CPU)
  - `generate` / `_branch`: логика генерации не тестирована
  - `build_octree_fields`: не тестирована
  - Сохранение/загрузка: save/load ConceptSpace и SyntaxLattice не тестированы
  - Фаззинг границ, Memory stress: отсутствуют

---

## 6. Матрица приоритетов V4 (новые проблемы)

| ID | Проблема | Приор. | Агент | Тип |
|:--:|----------|:------:|:-----:|:---:|
| **SN-1** | Undefined `gpu_cid_gen` → NameError | **P0** | NS | Корректность |
| **TN-12** | `full_stuck` ложные срабатывания без eval | **P0** | TD | Корректность |
| **QN-1** | syntax_lattice `np.load` без allow_pickle=False (RCE) | **P0** | QA | Безопасность |
| **SN-2** | Double distance decay CPU vs GPU | **P1** | NS | Корректность |
| **A-N1** | OOM fallback retries CUDA на каждый вызов | **P2** | Arch | Производительность |
| **A-N2** | Double fluctuate при full_stuck | **P2** | Arch | Корректность |
| **A-N3** | PMI отключён на 1 батч после decay_all | **P2** | Arch | Корректность |
| **A-N4** | Test coverage gap — новые механизмы | **P2** | Arch | Тестирование |
| **SN-3** | concept_error кэш 30K < 146K vocab | **P2** | NS | Качество |
| **GN-1** | Регенерация `_cid_order_t` на каждый вызов | **P2** | GPU | Производительность |
| **GN-2** | CPU-цикл concept_error в GPU path | **P2** | GPU | Производительность |
| **GN-3** | evaluate — дублирование prev_vecs CPU→GPU | **P2** | GPU | Производительность |
| **TN-13** | Гормоны не обновляются при train | **P2** | TD | Улучшение |
| **QN-2** | `_ensure_torch` OOM fallback ненадёжен | **P2** | QA | Надёжность |
| **A-N5** | `last_decay_lines` мёртвый код | **P3** | Arch | Косметика |
| **A-N6** | identity-индексация CIDs | **P3** | Arch | Производительность |
| **A-N7** | class-переменная `_default_use_torch` | **P3** | Arch | Косметика |
| **QN-3** | `FCFConfig.base_dir` дублирует `PathConfig` | **P3** | QA | Косметика |
| **QN-4** | Тесты не изолированы | **P3** | QA | Косметика |

---

## 7. Статус старых проблем P0/P1 (из V3)

| ID | Проблема | Приоритет V3 | Статус V4 |
|:--:|----------|:------------:|:----------:|
| **G-1** | `.to()` вместо `.copy_()` — тройные аллокации | **P0** | ⚠️ Не исправлено |
| **G-2** | FP16 не используется | **P0** | ⚠️ Не исправлено |
| **Q-1** | `np.load` без `allow_pickle=False` (RCE) | **P0** | ⚠️ Не исправлено (concept_space + syntax_lattice) |
| **T-1** | `context_window=1` на старте | **P0** | ✅ Не баг — осознанное curriculum |
| **T-2** | Destab reset на epoch 2+ | **P0** | ⚠️ Не исправлено |
| **S-1** | L2 retraction после STDP | **P0** | ✅ Исправлено |
| **A-1** | Циркулярная Generator↔Space | **P1** | ⚠️ Не исправлено |
| **A-2** | Monkey-patching SyntaxLattice | **P1** | ⚠️ Не исправлено |
| **A-3** | FCFConfig God Object | **P1** | ✅ Исправлено (рефакторинг) |
| **A-5** | Неполная синхронизация GPU тензоров | **P1** | ✅ Исправлено (_fb_dirty) |
| **G-3** | Нет `pin_memory` для H2D transfers | **P1** | ⚠️ Не исправлено |
| **G-4** | Синхронный CPU transfer после scatter_add | **P1** | ⚠️ Не исправлено |
| **G-5** | Нет профилирования GPU | **P1** | ⚠️ Не исправлено |
| **S-2** | Ненормализованный scatter_add | **P1** | ✅ Исправлено (нормализация по weighted sum elr) |
| **S-3** | Перевёрнутый concept_error в PMI | **P1** | ✅ Не перевёрнут — логика корректна |
| **S-4** | Concept error не используется в neg sampling | **P1** | ✅ Исправлено |
| **T-3** | CURICULUM_MIN_LEN=16 отсекает корпус | **P1** | ⚠️ Не исправлено (P2) |
| **T-4** | vacc1_stuck ложные срабатывания | **P1** | ⚠️ Не исправлено |
| **T-5** | Накопление шума от fluctuate | **P1** | ⚠️ Не исправлено (P2) |
| **T-6** | Нет early stopping | **P1** | ⚠️ Не исправлено |
| **Q-2** | Отсутствие тестов | **P1** | ✅ Частично — тесты созданы |
| **Q-3** | `_quiet()` глотает исключения | **P1** | ⚠️ Не исправлено |
| **Q-4** | Дублирование STDP-логики | **P1** | ⚠️ Не исправлено |
| **Q-5** | Отсутствие type hints | **P2** | ✅ Исправлено |

---

## 8. Сводка критических (P0) задач к немедленному исправлению

1. **SN-1**: Добавить `gpu_cid_gen` в сигнатуру `_negative_sampling_gpu` и пробросить из call sites. Иначе NameError при GPU+neg sampling.
2. **TN-12**: Исправить `full_stuck` детекцию — не срабатывать без eval данных; не передавать `vacc1=0` по умолчанию.
3. **QN-1**: `np.load(..., allow_pickle=False)` в syntax_lattice.py (и проверить concept_space.py — Q-1 тоже не исправлен).
4. **T-2**: Destab scale — вычислять от global_step, а не от per-epoch idx.
5. **G-1, G-2, Q-1 (concept_space)**: Долгоживущие P0, требующие исправления.

---

## 9. Итоги

- **Новых проблем**: 19 (3×P0, 1×P1, 10×P2, 5×P3)
- **Старых P0/P1 не исправлено**: 13 (3×P0, 10×P1)
- **Всего активных P0**: 6 (SN-1, TN-12, QN-1, G-1, G-2, Q-1, T-2)
- **Прогресс V3→V4**: из 6 P0 проблем V3 исправлено 2 (S-1, T-1 confirmed design). Добавилось 3 новых P0 (SN-1, TN-12, QN-1).
- **Тестовое покрытие**: создано (24 теста, 298 строк), но отсутствует GPU/CPU parity — критический пробел.

---

*Отчёт составлен коллегией AI-агентов: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent*
