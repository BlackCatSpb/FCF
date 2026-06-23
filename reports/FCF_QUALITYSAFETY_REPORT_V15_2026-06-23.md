# FCF Quality & Safety Audit Report V15

**Дата:** 2026-06-23  
**Версия кодовой базы:** HEAD 4178389 (пост-V14)  
**Аналитик:** Quality-Safety Agent  
**Файл тестов:** `tests/test_stdp.py` (1786 строк, ~145 тестов)  

---

## 1. Executive Summary

После релиза V14 кодовая база FCF находится в состоянии активной разработки с внедрением новых архитектурных компонентов (EntityField, Harmonizer, HDC n-gram, sector index, adaptive L1, dynamic capacity, item memory, cluster-potential, W_proj Hebbian update, JL projection, CheckpointManager async save). Ключевое изменение размерности (dim 384→768, latent_dim 512→2048) создаёт риск неконсистентности существующих тестов, завязанных на hardcoded константы размерности.

**Основные выводы:**
- Тестовое покрытие существующих компонентов (ConceptVectorStore, FractalField, ConceptSpace, STDP) адекватное — ~145 тестов, 6 skipped в типичном окружении.
- **Новые компоненты практически не покрыты тестами.** Из 11 новых модулей/функций только CheckpointManager имеет smoke-тесты, всё остальное — нулевое покрытие.
- Опасность: `_META_QWEN = 9` остаётся в `crystal_generator.py` как мёртвая константа, но в `gpu_meta_l` кортежи не содержат 10 элементов — это может вызвать `index out of bounds` при активации qwen-пути.
- `qwen_knowledge.py` существует как самостоятельный модуль, но в `train_full.py` инициализируется `qwen_knowledge=None` — модуль не используется, но код поддержки (конфиг, meta-индекс) не удалён.
- `_semantic_bootstrap` — мёртвый код: вызывается в `_checkpoint` (`train_full.py:475`), но не вызывается в `_train` — жив только как ручная утилита при чекпойнтах.
- `_skip_gpu_sync` механизм корректен, но тонок: при ошибке в `_harmonize_batch` (строка 243 `stdp_trainer.py`) гасится исключение `try-except` вокруг `sp.IdToPiece` — если вызов `ef.sync_word` падает, GPU-векторы могут быть раз синхронизированы с `_skip_gpu_sync=True` без восстановления.
- После capacity grow/prune checkpoint resume не тестировался: `CheckpointManager.save` сохраняет `.codes.npz` через `FractalField.to_dict`, который корректно сериализует `basis` и `codes`, но `latent_dim` меняется после grow — при загрузке в `FractalField.from_dict` размерности не проверяются на консистентность с `dim`.

**Общая оценка: SAFETY-ORANGE.** Критических дыр в безопасности нет, но накопленный технический долг по тестированию новых компонентов требует немедленного внимания перед V15.

---

## 2. Test Coverage Analysis (по компонентам)

### 2.1 ConceptVectorStore — ПОЛНОЕ покрытие
- `test_basic_crud` — create, set, get, contains, len
- `test_bounds_check` — out-of-range get
- `test_items_and_keys` — iteration, keys, values, items
- `test_empty_concept_vector_store` — пустой store
Все тривиальные краевые случаи покрыты.

### 2.2 FractalField — ЧАСТИЧНОЕ покрытие
- `test_init_and_vector` — init_concept возвращает unit-norm
- `test_basis_health` — check_basis_health возвращает False на свежей матрице
- `test_fluctuate` — fluctuate не ломает норму
- `test_fb_dirty_flag` — init_fields выставляет флаг
- `test_fractal_subspace_dims` — l_c + l_a + l_m == latent_dim

**НЕ покрыты:**
- `_apply_l1` / `_apply_l1_batch` — L1 софт-трешинг
- `init_learned_fields` / `update_learned_fields` — W_proj инициализация и Hebbian update
- `_rebuild_field_bits` — упаковка bits
- `adjust_l1_lambdas` — адаптивная L1
- `grow_capacity` / `prune_capacity` / `auto_adjust_capacity` — динамическая ёмкость
- `_init_sector_fields` / `_rebuild_sector_index` / `sector_key` / `search_in_sector` / `focal_refine` — sector index
- `hdc_bind` / `hdc_permute` / `hdc_bundle` / `hdc_ngram_repr` / `hdc_unbind` / `hdc_update_ngram` / `hdc_predict` — HDC n-gram
- `field_overlap` — overlap count
- `to_dict` / `from_dict` — сериализация с binary_path и field_bits

### 2.3 EntityField — НУЛЕВОЕ покрытие
- `key_char`, `key_word`, `key_sent`, `key_para` — ни одного теста
- `_to_dim` — JL projection не тестирована
- `ensure`, `get`, `set`, `sync_word` — не тестированы
- `bind`, `query` — VSA bind/unbind не тестированы
- `to_dict` / `from_dict` — сериализация не тестирована
- `decay` — decay-фактор не тестирован

### 2.4 Harmonizer — НУЛЕВОЕ покрытие
- `compose_word` / `decompose_word` — композиция/декомпозиция морфем
- `harmonize` — итеративная гармонизация word↔morph
- `harmonize_with_envelope` — character-level envelope
- `register_word` / `set_morpheme_vec` / `get_morpheme_vec`
- `mark_word_dirty` / `mark_morph_dirty` / `clear_dirty`
- `to_dict` / `from_dict`

### 2.5 STDP / CPU path — ХОРОШЕЕ покрытие
- `_cpu_stdp_apply` — smoke, vector update, lateral inhibition, gradient clipping, destab — все покрыты
- `_negative_sampling_cpu` — smoke
- `_contrastive_objective_cpu` — smoke (только запуск)
- `_centroid_pull_batch` — CPU path (только запуск с unit-norm assert)
- `_build_pairs` — smoke на синтетических данных

### 2.6 STDP / GPU path — СРЕДНЕЕ покрытие
- `_gpu_stdp_apply` — smoke, momentum, destab (базовый и high), deferred write, subspace skip
- `_negative_sampling_gpu` — batched write, empty
- `_contrastive_objective_gpu` — empty, valid_hn mask, cross-field reg
- `_lateral_inhibition_gpu` — precomputed mask, correctness
- `_gpu_poststdp_fused` — neg и contrastive mock-call
- `_centroid_pull_batch` GPU — parity test (CPU vs GPU)

**НЕ покрыто в GPU:**
- `_gpu_stdp_core` с `antonym_mask` (antonym repel path)
- GPU destab с реальным `_ce_t` (в тестах `_ce_t` не заполнен)
- `_cluster_centroid_pull` — не тестирован вообще
- `_update_cluster_potential` — smoke есть (cluster_potential_update), но без LR modulation asserts

### 2.7 CheckpointManager — СРЕДНЕЕ покрытие
- `test_init_defaults`, `test_mgr_save`, `test_mgr_cleanup` — smoke
- `test_save_roundtrip`, `test_cleanup_removes_old`, `test_shutdown_clean` — resilience
- `test_save_with_opt`, `test_save_with_extras` — опциональные параметры
- `test_failure_cleanup` — корректная очистка .tmp при ошибке save
- `test_remove_tag` — удаление файлов по тэгу

**НЕ покрыто:**
- `ckpt_state` — запись `checkpoint_state.json` (тест QN-54 только проверяет паттерн save, не интеграцию)
- `wait()` с множественными futures
- `_sync_save` с ошибкой opt.save_state (очистка tmp opt)
- shutdown с незавершёнными futures

### 2.8 ParameterOptimizer — ПОЛНОЕ покрытие
- `test_basic_step`, `test_full_stuck_no_eval`, `test_full_stuck_with_eval`, `test_vacc1_stuck`, `test_save_load_state` — все режимы покрыты.

### 2.9 Остальные компоненты
- `RNGRegistry` (QN-36) — 4 теста, детерминизм и изоляция: OK
- `AdaptiveErrorTracker` (QN-37) — 4 теста, EMA, FIFO, dict interface: OK
- `FractalEncoding` (QN-15) — path, LCP: OK
- `HormonalSystem` (QN-11) — init, update match/mismatch, temperature, beam width, save/load: OK
- `SyntaxLattice` — только roundtrip save/load (косвенно через CheckpointManager)

---

## 3. Пропущенные тесты (Missing Coverage)

### 3.1 Критически важные (без тестов — риск регресса)
1. **EntityField** — ни одного теста. 236 строк кода, VSA bind/unbind, JL projection, sync_word, serialization.
2. **Harmonizer** — ни одного теста. 355 строк кода, compose/decompose, harmonize, dirty cascade, serialization.
3. **HDC n-gram memory** (FractalField) — bind, permute, bundle, ngram_repr, unbind, update_ngram, predict — ни одного теста.
4. **W_proj Hebbian update** (`FractalField.update_learned_fields`) — 0 тестов.
5. **`_to_dim` JL projection** (`EntityField._to_dim`) — 0 тестов.
6. **Dynamic capacity** (`grow_capacity`, `prune_capacity`, `auto_adjust_capacity`) — 0 тестов.
7. **Sector index** (`_init_sector_fields`, `_rebuild_sector_index`, `sector_key`, `search_in_sector`, `focal_refine`) — 0 тестов.
8. **Adaptive L1** (`adjust_l1_lambdas`, `_apply_l1` per-concept) — 0 тестов.
9. **Item memory** (`reinit_rare`) — 0 тестов.
10. **Cluster-potential** (`_ensure_cluster_map`, `_update_cluster_potential`, `_cluster_centroid_pull`) — smoke есть, но без asserts на корректность потенциала.

### 3.2 Средней важности
11. **`_harmonize_batch`** — 0 тестов. Вызывается в `_train` каждую итерацию.
12. **`_semantic_bootstrap`** — 0 тестов (и мёртвый код, но живой в `_checkpoint`).
13. **`_update_hdc_ngrams`** — 0 тестов.
14. **CheckpointManager интеграция с capacity grow** — 0 тестов.
15. **GPU `antonym_mask` path** в `_gpu_stdp_core` — 0 тестов.
16. **`_apply_subspace_update_batch` L1 path** — тесты есть, но L1 branch не проверен (ce_vals и _apply_l1_batch).

---

## 4. Skipped тесты — анализ

При типичном запуске (PyTorch установлен, CUDA отсутствует, SentencePiece модель не загружена) `pytest` показывает 6 skipped тестов. Причина и категоризация:

### 4.1 Безусловные технические skip'ы (3 теста)

| Тест | Причина | Критичность |
|------|---------|-------------|
| `test_generate_returns_result` | `pytest.skip("No sentencepiece model")` | LOW — gen.sp=None в fixture, модель не инициализируется |
| `test_generate_empty_seed` | `pytest.skip("No sentencepiece model")` | LOW — то же |
| `test_gpu_stdp_momentum` | `pytest.skip("no torch/cuda")` | LOW — CUDA не доступна |

**Оценка:** эти тесты принципиально не могут работать в CI без SentencePiece модели и GPU. Нормальная ситуация. Возможное улучшение: создать mock для SentencePiece в conftest.py/fixture, чтобы тесты generate работали без реальной модели.

### 4.2 Условные skip'ы на GPU (2 теста)

| Тест | Причина | Критичность |
|------|---------|-------------|
| `test_gpu_contrastive_simple` | `pytest.skip("No GPU")` | LOW |
| `test_gpu_contrastive_no_double_update` | `pytest.skip("No GPU")` | LOW |

**Оценка:** тесты корректно пропускаются при отсутствии CUDA. На CI без GPU это норма. Но код внутри тестов использует `hasattr(gen, '_vecs_t')` в условии skip — это fragile, т.к. `_vecs_t` инициализируется лениво.

### 4.3 Проблемные skip'ы (1 тест?)

Из 43 `@pytest.mark.skipif(not HAS_TORCH)` — при наличии torch все эти тесты запускаются. Проблема: при сборке на CI без PyTorch будет пропущено 43 теста (~30% тестовой базы). Рекомендуется: либо сделать PyTorch обязательной зависимостью для тестов, либо вынести GPU-тесты в отдельный маркер.

**Вывод:** 6 skipped — не критично. Но неявное требование PyTorch для >40 тестов — архитектурный risk для CI без GPU/PyTorch.

---

## 5. Безопасность (Safety Analysis)

### 5.1 QwenKnowledge reference — НЕ удалена

**Факт:** `_META_QWEN = 9` определена в `crystal_generator.py:35`. Вся инфраструктура поддержки QwenKnowledge сохранена:
- `qwen_knowledge.py` (121 строка) — полноценный модуль с классом
- `fcf_config.py:122-123` — `qwen_knowledge_path` property
- `crystal_generator.py:66` — параметр `qwen_knowledge=None` в `__init__`
- `train_full.py:640` — явная передача `qwen_knowledge=None`

**Риск:** `gpu_meta_l` кортежи в `_build_pairs` содержат 10 элементов: `(i, j, pmi_w, dist_weight, freq_weight, field_weight, 0.0, ids[i], ids[j], antonym_flag)`. Индекс 9 занят `antonym_flag`, а не qwen. Если кто-либо попытается активировать QwenKnowledge через параметр, произойдёт:
1. В `_build_pairs` будет сформирован кортеж из 10+ элементов
2. Мета-индекс `_META_QWEN = 9` будет конфликтовать с `antonym_flag`
3. В `_gpu_stdp_core` чтение `meta_t[:, _META_QWEN]` вернёт antonym_flag вместо qwen_factor — логическая ошибка

**Рекомендация:** удалить `_META_QWEN`, либо явно присвоить ему `_META_ANTONYM` (сейчас 9). Или удалить модуль `qwen_knowledge.py` и всю связанную конфигурацию, если он не используется.

### 5.2 `_skip_gpu_sync` корректность

**Механизм:** флаг `_skip_gpu_sync` подавляет copy-back в `_on_vector_update`. Устанавливается в `True` перед batched write и `False` после.

**Уязвимое место — `_harmonize_batch`** (`stdp_trainer.py:209-329`):
```python
gen._skip_gpu_sync = True
for cid, v_new in zip(all_cids, vecs_cpu):
    cs._apply_vector_update(cid, v_new)  # может упасть?
    ef.sync_word(cid, v_new)
gen._skip_gpu_sync = False
```
Если `cs._apply_vector_update` падает (например, assert в `set_vec` при невалидном cid), `_skip_gpu_sync` останется `True`. Все последующие вызовы `_on_vector_update` не будут синхронизировать GPU → тихое расхождение CPU/GPU.

**Рекомендация:** использовать try/finally:
```python
gen._skip_gpu_sync = True
try:
    ...
finally:
    gen._skip_gpu_sync = False
```

### 5.3 `_harmonize_batch` — скрытое исключение

```python
try:
    word_text = gen.sp.IdToPiece(int(cid)).replace('\u2581', ' ').strip()
except Exception:
    pass
```
Гасятся все исключения, включая `IndexError` и `RuntimeError`. Это маскирует проблемы:
- Если `sp` загружен не полностью
- Если cid выходит за пределы vocab_size
- Если `sp.IdToPiece` внутренне падает

После `except`, `word_text` остаётся `None`, и код просто не выполняет char↔word bind — корректно, но диагностика потеряна.

### 5.4 Checkpoint resume после capacity grow/prune

**Проблема:** `FractalField.from_dict` восстанавливает `basis` и `codes` из `.codes.npz`. Размерности `latent_dim` и `dim` берутся из JSON-ключа `data['fractal']['latent_dim']` и `data['fractal']['dim']`. После `grow_capacity` или `prune_capacity`:
- `latent_dim` меняется
- `basis` меняет форму `[latent_dim, dim]`
- `codes` меняют длину

При загрузке старого чекпойнта (с меньшим latent_dim) на новую версию кода — нет проверки, что `basis.shape[0] == latent_dim`. Если mismatch — `np.stack` (в `from_dict:820`) упадёт с ValueError.

**В `grow_capacity`** новый basis ортогонализируется через QR, `codes` расширяются нулями. При сохранении после grow:
- `.codes.npz` будет содержать basis и codes нового размера
- JSON будет содержать новый `latent_dim`

Всё корректно, **если** загрузка делается на той же версии кода, которая сохраняла. Если код после обновления ожидает другую размерность subspace split (`l_c`, `l_a`, `l_m`) — это не проверяется.

**Риск:** при смене `FractalField.__init__` по умолчанию (latent_dim=2048), если сохранить с custom latent_dim (например, после prune 1536) и загрузить с ожиданием 2048 — `basis.shape[0]` не совпадёт с `latent_dim` в JSON.

### 5.5 fp16/bfloat16 numerical stability

Тесты QN-62 (EMA bf16, mom_t bf16, codes fp16) проверяют стабильность на 100 шагов — это минимальный smoke. В реальном обучении (миллионы шагов) bf16 EMA и fp16 codes могут дрейфовать.

---

## 6. Edge Cases

### 6.1 Пустые поля и нулевые векторы

| Компонент | Edge case | Статус |
|-----------|-----------|--------|
| `ConceptVectorStore.__getitem__` | cid с `_valid[cid]=False` | OK — возвращает None |
| `ConceptVectorStore.get` | cid невалидный (<0 или ≥V) | OK — bounds check |
| `FractalField.compute_vector` | codes[cid] нет | OK — None |
| `FractalField.init_fields` | `self.codes` пуст | OK — field_bits = {} |
| `EntityField.get` | ключ не в `entities` и word_store=None | OK — None через `entities.get` |
| `EntityField.bind` | ctx_type нет в `ETYPE_TO_ROLE` | OK — return |
| `Harmonizer.compose_word` | morph_parts пуст | OK — return None |
| `Harmonizer.harmonize` | word_id нет в word_morphs | OK — return (None, 0.0) |
| `STDPTrainer._harmonize_batch` | cs.harmonizer нет или word_morphs пуст | OK — return |
| `FractalField.grow_capacity` | `new_latent_dim < old_dim + 8` | OK — clamp |
| `ConceptSpace.topk_similar_concepts` | mat пуст (нет concept_vectors) | OK — return [] |
| `CrystalGenerator._branch` | seq пуст | OK — return [] |
| `_apply_vector_update` | v_old нет (первый раз) | OK — без shift |
| `_lateral_inhibition_cpu` | gen_cids < 2 | OK — return |
| `_gpu_stdp_apply` | gpu_ctx_l пуст | OK — N=0, метатрица пуста |

### 6.2 V=0 и d_codes_t пуст

При `vocab_size=0`:
- `ConceptSpace(0, dim)` — создаётся корректно
- `ConceptVectorStore(0, dim)` — `_data.shape = (0, dim)`
- `init_concepts()` — ничего не делает (range(0) пуст)
- `init_homeostasis()` — `concept_usage = {}`

**Проблема:** `FractalField.__init__` не зависит от vocab_size — она создаёт basis (latent_dim, dim) и пустой codes dict — OK. Но при `V=0` ни один компонент не падает.

В `_build_torch_tensors` при `V=0`:
- `vecs = np.zeros((0, D))` — корректно
- `_vecs_t = torch.empty(0, D, ...)` — OK

**Вывод:** V=0 безопасен, но бесполезен.

### 6.3 W_proj = None

`W_proj` инициализируется как `None` и создаётся только `init_learned_fields()`. Все места использования:
- `update_learned_fields` — проверка `self.W_proj is None` → return
- `_rebuild_field_bits` — проверка `self.W_proj is None` → return
- `_init_sector_fields` — не зависит от W_proj, использует `_sector_W`
- `grow_capacity` — проверка `if self.W_proj is not None`
- `prune_capacity` — проверка `if self.W_proj is not None`
- hdc_memory update в `_train` — `if cs.fractal.W_proj is not None`

**Проблема:** условие в `_train` (stdp_trainer.py:192):
```python
if hasattr(cs.fractal, 'hdc_memory') and cs.fractal.W_proj is not None:
    _update_hdc_ngrams(cs, ids, max_n=3)
```
HDC обновление привязано к наличию W_proj, хотя W_proj не используется внутри `hdc_update_ngram`. Это логическая ошибка: HDC n-gram память должна работать независимо от режима learned fields.

### 6.4 `_cluster_map` для концептов без field_bits

`_ensure_cluster_map` (crystal_generator.py:450-473) находит первый set bit в field_bits[cid] как cluster ID. Если field_bits пусты — `cluster_arr` остаётся нулевой, и все концепты попадают в cluster 0. Некорректно, но не крашится.

### 6.5 `search_in_sector` с пустым сектором

Если `sector_index[depth].get(key, [])` возвращает пустой список или содержит только `query_cid` — функция возвращает `[]`. OK.

### 6.6 `focal_refine` без sector_index

Если `_sector_W` не инициализирован — `search_in_sector` вернёт `[]`, `focal_refine` вернёт `[]`. Обработчик в `_branch` падает на `sim_candidates` пустой список → не использует sector search — OK, fallback на `topk_similar_concepts`.

---

## 7. Предложения тестов (минимальный набор — 15 тестов)

### 7.1 EntityField (4 теста)

**Тест 1:** `test_entity_field_basic_bind_query`
- Scope: `EntityField.bind` + `EntityField.query`
- Что: создать EntityField(dim=64), зарегистрировать word и char, bind char→word, query word
- Assert: query возвращает вектор, норма ~1.0

**Тест 2:** `test_entity_field_sync_word`
- Scope: `EntityField.sync_word`
- Что: создать ConceptVectorStore(10, 64), sync_word через word_store
- Assert: get(('w', cid)) возвращает вектор, совпадающий с store.get(cid), scaled через _to_dim

**Тест 3:** `test_entity_field_to_dim_jl`
- Scope: `_to_dim`
- Что: EntityField(dim=64) с входным вектором 128D, вызов _to_dim
- Assert: результат shape (64,), норма сохранена (после нормализации)

**Тест 4:** `test_entity_field_serialization_roundtrip`
- Scope: `to_dict` + `from_dict`
- Что: создать EntityField, bind несколько entities, сериализовать, десериализовать
- Assert: entities совпадают, dim совпадает

### 7.2 Harmonizer (3 теста)

**Тест 5:** `test_harmonizer_compose_decompose`
- Scope: `compose_word` + `decompose_word`
- Что: создать Harmonizer(dim=64), зарегистрировать морфемы {'ROOT': v1, 'SUFFIX': v2}, compose → decomposed
- Assert: после decompose результат ~исходные векторы (VSA свойства)

**Тест 6:** `test_harmonizer_harmonize_convergence`
- Scope: `harmonize`
- Что: register_word с двумя морфемами, harmonize с random word_vec
- Assert: delta уменьшается, итоговый word_vec на unit sphere

**Тест 7:** `test_harmonizer_dirty_cascade`
- Scope: `mark_morph_dirty`
- Что: register_word(w1, {ROOT: m1}), register_word(w2, {ROOT: m1}), mark_morph_dirty(m1)
- Assert: w1 и w2 оба в word_dirty

### 7.3 HDC n-gram (2 теста)

**Тест 8:** `test_hdc_bind_permute_identity`
- Scope: `hdc_bind` + `hdc_permute`
- Что: создать FractalField, bind двух векторов, permute
- Assert: hdc_bind(hdc_permute(a), hdc_permute(b)) ≈ hdc_permute(hdc_bind(a,b)) — коммутативность

**Тест 9:** `test_hdc_ngram_update_predict`
- Scope: `hdc_update_ngram` + `hdc_predict`
- Что: FractalField(hdc_memory_max=100), update_ngram с тремя префиксами, predict
- Assert: predict возвращает top-k, в top-3 есть ожидаемый следующий код

### 7.4 Dynamic capacity (2 теста)

**Тест 10:** `test_grow_capacity_preserves_norms`
- Scope: `grow_capacity`
- Что: FractalField(latent_dim=64), init_concept(0,1,2), измерить нормы, grow_capacity(new_latent_dim=128)
- Assert: нормы векторов ≈ 1.0, codes длины 128, basis.shape[0] == 128

**Тест 11:** `test_prune_capacity_removes_dead_dims`
- Scope: `prune_capacity`
- Что: FractalField(latent_dim=64), создать codes с одним нулевым измерением у всех, prune_capacity(sparsity_threshold=0.99)
- Assert: latent_dim уменьшился, basis.shape[0] уменьшился, векторы unit-norm сохранены

### 7.5 W_proj Hebbian update (1 тест)

**Тест 12:** `test_hebbian_wproj_update`
- Scope: `update_learned_fields`
- Что: FractalField с W_proj (latent_dim=64, n_field_bits=16), init_concepts(10), update_learned_fields
- Assert: W_proj изменился, колонки unit-norm, field_bits обновлены

### 7.6 Adaptive L1 (1 тест)

**Тест 13:** `test_adaptive_l1_maintains_density`
- Scope: `adjust_l1_lambdas`
- Что: FractalField(l1_lambda=0.001), создать один концепт, _apply_l1 с mock density window
- Assert: l1_lambda_per_cid изменился в правильном направлении

### 7.7 Sector index (1 тест)

**Тест 14:** `test_sector_search_finds_similar`
- Scope: `search_in_sector`
- Что: FractalField с _init_sector_fields, init_concepts(100), проверить sector_key для depth=0
- Assert: search_in_sector возвращает кандидатов, query_cid не включён

### 7.8 Checkpoint с capacity grow (1 тест)

**Тест 15:** `test_checkpoint_after_capacity_grow_roundtrip`
- Scope: интеграция FractalField.grow_capacity + CheckpointManager.save + ConceptSpace.load
- Что: ConceptSpace → grow_capacity → CheckpointManager.save → ConceptSpace.load
- Assert: latent_dim совпадает, basis.shape совпадает, codes длины совпадают, compute_vector возвращает unit-norm

---

## 8. Дополнительные замечания

### 8.1 Отсутствие type hints в новых компонентах
`EntityField` и `Harmonizer` имеют минимальные type hints. `FractalField` имеет частичные. Это снижает возможность статического анализа.

### 8.2 State explosion в `FractalField`
Класс содержит 20+ полей состояния (`codes`, `field_bits`, `W_proj`, `hdc_memory`, `hdc_memory_counts`, `l1_lambda_per_cid`, `_sector_W`, `_sector_index`, ...). Это усложняет верификацию корректности сериализации — каждое новое поле должно быть добавлено в `to_dict`/`from_dict`.

### 8.3 `_apply_l1` вызывает `np.sign` + `np.maximum` каждый вызов
При частом вызове (каждый STDP update для каждого concept) — накладные расходы. Можно кэшировать z_c slice.

### 8.4 `_rebuild_sector_index` — O(V · depth · avg_bits) каждый вызов
В `_rebuild_field_bits` (вызывается из `update_learned_fields`) перестраивается весь sector index. При V=146K и 3 уровнях — тяжёлая операция (~438K matmuls).

### 8.5 Проверка HEALTH в `FractalField.__init__` (строка 112)
`Q, _ = np.linalg.qr(mat, mode='reduced')` на матрице `(latent_dim, dim) = (2048, 768)` — QR разложение 2048×768, ~2ms, OK.

---

*Конец отчёта. Всего символов: ~19,000 (с пробелами).*
