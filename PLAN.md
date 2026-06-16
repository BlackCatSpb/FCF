# План доработок (Planned Fixes)

Основание: AUDIT.md (40+ issues) + верификация всех claims против кода (2026-06-16).
Уже исправлено: B1, C2/C3, B5, B3, R1, E2.

---

## P0 — Critical

### P0-1: Missing KMeans import
**Файл:** `concept_space.py:13-15`  
**Проблема:** `KMeans` используется в `pq_train()` (L1035), но не импортирован — `NameError` в runtime.  
**Статус:** Жив, но `pq_train()` — мёртвый код (никем не вызывается).  
**Fix:** Либо `from sklearn.cluster import KMeans`, либо удалить `pq_train/pq_encode/pq_decode` как мёртвый код.

### P0-2: Дублирование train_from_text / train_batch
**Файлы:** `crystal_generator.py:563-1004` и `1007-1400`  
**Проблема:** ~80% идентичной логики (pair building, GPU STDP, centroid pull, lateral inhibition). Изменения рассинхронизируются.  
**Fix:** Рефакторинг: вынести общую логику в приватные методы (`_build_pairs`, `_gpu_stdp`, `_centroid_pull`, `_lateral_inhibition`), оставить только различия (contrastive loop).

---

## P1 — High

### P1-1: Hardcoded paths (19 файлов)
**Проблема:** `C:\Users\black\...` в `sys.path.insert(0, ...)` — не портабельно, утекает username.  
**Файлы:** `eval_checkpoint.py:2`, `eval_metrics.py:6`, `inference.py:8`, `train_full.py:11`, `train.ps1:7`, `train.bat:2`, `train_fast.bat:2`, `run_train.bat:2`, `morph_vocab.py:266`, `crystal_generator.py:1583,1589`, `concept_space.py:1430`, `syntax_lattice.py:738,743,746,751,756`, `concept_inductor.py:356`  
**Fix:** Заменить на `os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))`.

### P1-2: ARCHITECTURE.md полностью устарел
**Файл:** `ARCHITECTURE.md`  
**Проблема:** Пишет 36K×128D (реально 146K×384D), перечисляет мёртвые модули как активные, не упоминает fractal_encoding, morph_vocab, parameter_optimizer.  
**Fix:** Полный rewrite или удаление.

### P1-3: GPU negative sampling — Python-цикл
**Файл:** `crystal_generator.py:885-910`, `1293-1314`  
**Проблема:** GPU-путь делает `.item()` + Python-цикл по парам для negative sampling. Это сводит на нет выгоду GPU.  
**Fix:** Векторизовать negative sampling на GPU: `randint(0, vocab_size, (N, neg_samples))` + gather + sims = всё на тензорах.

### P1-4: train.ps1 — мёртвые параметры
**Файл:** `train.ps1`  
**Проблема:** `-Resume`, `-QuickTest`, `-MaxLines` не передаются в `train_full.py`. `EVA_MAX_LINES` не читается.  
**Fix:** Либо убрать мёртвые флаги, либо передавать их в `train_full.py`.

### P1-5: eval_checkpoint.py — старая BPE модель
**Файл:** `eval_checkpoint.py:10`  
**Проблема:** `bpe_ru_32k.model` вместо `bpe_ru_146k.model`.  
**Fix:** Заменить на `CFG.bpe_model_path` или `bpe_ru_146k.model`.

---

## P2 — Medium

### P2-1: ach_phasic всегда 0
**Файл:** `hormonal_system.py:37,143`  
**Проблема:** `self.ach_phasic *= self.phasic_decay` (0.7) — никогда не устанавливается в ненулевое значение.  
**Fix:** Либо удалить, либо имплементировать механизм phasic ACh.

### P2-2/P2-3/P2-4/P2-11: Мёртвый код
- `pos_tagger.py:132-137` — `pos_transition_score()`, никем не вызывается
- `crystal_generator.py:129-140` — `_semantic_delta()`, никем не вызывается
- `concept_space.py:791-843` — `fractal_stdp()`, никем не вызывается
- `concept_space.py:505-522` — `build_anchor_matrix()`, никем не вызывается
- `concept_space.py:1003-1176` — `pq_train/pq_encode/pq_decode`, никем не вызываются
- `concept_space.py:1259-1319` — `contrastive_spread()`, никем не вызывается
**Fix:** Удалить или закомментировать с пометкой.

### P2-5: train.ps1 мёртвые параметры
(дубль P1-4)

### P2-6: URL_TLDS мёртвая константа
**Файл:** `filter_corpus.py:15` — `{'.reggi'}` объявлен, не используется.  
**Fix:** Удалить или использовать.

### P2-7: Неиспользуемые импорты в fcf_config.py
**Файл:** `fcf_config.py:8` — `math` и `re` не используются.  
**Fix:** Удалить.

### P2-8: Множественные EMA обновления per batch
**Файл:** `crystal_generator.py:700-705, 819-822`  
**Проблема:** При дублирующихся gen_cid, `concept_error` обновляется несколько раз за batch.  
**Fix:** Дедуплицировать обновления — `unique(gen_cids)`.

### P2-9: Double np.abs
**Файл:** `concept_space.py:922-923`  
**Проблема:** `np.abs(all_codes)` вычисляется дважды.  
**Fix:** `abs_codes = np.abs(all_codes)`, переиспользовать.

### P2-10: import cdist внутри contrastive_spread — и не используется
**Файл:** `concept_space.py:1275`  
**Проблема:** import внутри dead code, причём `cdist` не используется.  
**Fix:** Удалить (вместе с методом, если удаляем dead code).

### P2-13: Доступ к _data/_valid извне
**Файлы:** `inference.py:109-110`, `eval_metrics.py:93,96,98`, `concept_space.py:878,978-979`  
**Проблема:** Прямой доступ к приватным аттрибутам `ConceptVectorStore`.  
**Fix:** Добавить публичные свойства `.data`/`.valid` в `ConceptVectorStore`.

---

## P3 — Low

### P3-1: Stale comment "10K lines trained"
**Файл:** `eval_checkpoint.py:60`  
**Fix:** Удалить или актуализировать.

### P3-2: Расхождение checkpoint_state.json vs _train_status.json
**Файлы:** `real_data/checkpoint_state.json` (line=6000), `_train_status.json` (line=5984)  
**Fix:** Устранить расхождение (возможно, разный момент сохранения).

### P3-6: ngrams[4] orphan
**Файл:** `syntax_lattice.py:96` — `ngrams[4] = {}` инициализирован, но при `max_n=3` не заполняется.
**Fix:** Убрать из `__init__` или динамически создавать.

### P3-10: Tokenization теряет BPE-информацию
**Файл:** `model/tokenization_fcf.py:42-52`  
**Проблема:** `_tokenize` возвращает `[str(i) for i in ids]` — исходный текст токена теряется.  
**Fix:** Возвращать `(token_text, token_id)` парой.

---

## Архитектурные

### A1: Две параллельных имплементации (modeling_fcf vs eva/symbolic)
**Fix:** Определить primary API, вторичный сделать обёрткой.

### A2: _archive/ — ~2000 строк мёртвого кода
**Fix:** Удалить директорию.

### A3: GPU/CPU split — two implementations diverging
(дубль P0-2, но шире — различаются lateral inhibition formula, centroid pull, etc.)
**Fix:** Единая implementation с селектором `use_torch`.

### A4: Epoch resume — fragile
**Файл:** `train_full.py:478-484`
**Fix:** Сохранять точный line number + epoch, а не полагаться на `start_line >= len(lines)-1`.

### A5: Config duplication (FCFConfig vs configuration_fcf.py)
**Fix:** `configuration_fcf.py` должен читать из `fcf_config.py`, а не дублировать defaults.

### A6: Corpus path not tracked in git
**Fix:** `.gitignore` исключает .txt в real_data/. Рассмотреть git LFS или smaller sample.

---

## Error Handling

### E1: _semantic_delta norm safety
(dead code — не фиксить, удалить)

### E3: _quiet swallows exceptions
**Файл:** `train_full.py:539`  
**Fix:** Перехватывать exception и логировать, а не глушить.

### E4: rng.choice может упасть при sample_k > valid
**Файл:** `eval_metrics.py:96-97`  
**Fix:** `sample_k = min(sample_k, len(valid_candidates))`.

### E5: Empty data в inference
**Файл:** `inference.py:113`  
**Fix:** Проверка `len(valid) > 0` перед matmul.

---

## Code Quality

### Q1: Redundant query_words default
**Файл:** `crystal_generator.py:195` (inference.py уже делает split)
**Fix:** Убрать из crystal_generator, положиться на caller.

### Q2: Module-level `import torch`
**Файл:** `crystal_generator.py:18`  
**Fix:** Lazy import: `_torch = None`, при первом `use_torch=True` делать `import torch`.

### Q3-Q4: sys.path.insert с абсолютными путями
(дубль P1-1)

### Q5: hasattr/setattr в hormonal_system.py
**Файл:** `hormonal_system.py:74,80-81,109-110`  
**Fix:** Инициализировать в `__init__`.

### Q7: TeeOut file handle leak
**Файл:** `train_full.py:48-63`  
**Fix:** Добавить `__del__` или context manager.

### Q8-Q9: Code duplication (save sequence)
**Файл:** `train_full.py:674-693`  
**Fix:** Вынести в `_final_save()`.

---

## Security

### S1: HTML/JS embedded in Python string
**Файл:** `train_full.py:374-471`  
**Fix:** Вынести в отдельный HTML-файл, читать `open(...).read()`.

### S2: API без rate limiting / auth
**Файл:** `api/main.py:45-66`  
**Fix:** Добавить rate limiting (slowapi) + optional API key.

---

## Очерёдность выполнения

### Phase 1 (перед следующим --fresh тренингом)
- [ ] P1-1: Заменить все hardcoded paths на относительные
- [ ] P1-5: Исправить BPE модель в eval_checkpoint.py
- [ ] P2-8: Дедуплицировать concept_error обновления
- [ ] P2-9: Double np.abs
- [ ] E4: sample_k safety
- [ ] E5: empty data guard
- [ ] P2-7: Удалить неиспользуемые импорты
- [ ] P3-6: ngrams[4] fix

### Phase 2 (качество кода)
- [ ] P0-2: Рефакторинг train_from_text / train_batch
- [ ] P2-13: Публичные свойства _data/_valid
- [ ] Q2: Lazy torch import
- [ ] Q5: hasattr/setattr fix
- [ ] Q7: TeeOut file handle
- [ ] Q8-Q9: Устранить дублирование save

### Phase 3 (чистка)
- [ ] P0-1: Удалить мёртвый PQ код (или добавить import)
- [ ] P2-2/3/4/10/11: Удалить мёртвый код
- [ ] P2-6: Удалить URL_TLDS
- [ ] P3-1: Stale comment
- [ ] A2: Удалить _archive/

### Phase 4 (архитектура)
- [ ] P1-2: Rewrite ARCHITECTURE.md
- [ ] P1-3: Векторизовать GPU negative sampling
- [ ] P1-4: Починить train.ps1
- [ ] A1/A3: Объединить GPU/CPU пути
- [ ] A4: Epoch resume fix
- [ ] A5: Устранить config duplication
- [ ] S1: HTML в отдельный файл
- [ ] S2: API security

---

*Last updated: 2026-06-16*
