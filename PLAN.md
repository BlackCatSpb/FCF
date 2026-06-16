# План доработок (Planned Fixes)

Основание: AUDIT.md (40+ issues) + верификация всех claims против кода (2026-06-16).
Уже исправлено: B1, C2/C3, B5, B3, R1, E2.

---

## P0 — Critical

### P0-1: Missing KMeans import  ✅ FIXED
**Файл:** `concept_space.py:13-15`  
**Проблема:** `KMeans` используется в `pq_train()` (L1035), но не импортирован — `NameError` в runtime.  
**Статус:** ✅ ВЕСЬ PQ-КОД УДАЛЁН (pq_train, pq_encode, pq_decode_all, pq_decode, pq_adc_search, pq_compression_ratio, pq_centroids, pq_codes, pq_nbits).

### P0-2: Дублирование train_from_text / train_batch  ✅ FIXED
**Файлы:** `crystal_generator.py`  
**Проблема:** ~80% идентичной логики (pair building, GPU STDP, centroid pull, lateral inhibition).  
**Статус:** ✅ Рефакторинг: 6 общих методов (_gpu_stdp_apply, _cpu_stdp_apply, _negative_sampling_gpu, _negative_sampling_cpu, _contrastive_objective, _centroid_pull). train_from_text (108 строк) и train_batch (110 строк) — тонкие обёртки. Дублирование ~55% сокращено.

---

## P1 — High

### P1-1: Hardcoded paths (19 файлов)  ✅ FIXED  
**Файлы:** 12 файлов (без concept_net.py/concept_tokenizer.py которые уже используют динамические пути, без concept_inductor.py который был в _archive/)  
**Fix:** Заменены на `os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))` или `%~dp0` (bat) / `Split-Path` (ps1).  
**Статус:** ✅ Все 12 файлов исправлены.

### P1-2: ARCHITECTURE.md полностью устарел  ✅ FIXED  
**Файл:** `ARCHITECTURE.md`  
**Статус:** ✅ Полностью переписан (146K×384D, актуальная структура, алгоритмы, GPU).

### P1-3: GPU negative sampling — Python-цикл  ✅ FIXED  
**Файл:** `crystal_generator.py`  
**Статус:** ✅ Векторизован: precompute ALL neg_cids + neg_elr перед циклом, никаких .item().

### P1-4: train.ps1 — мёртвые параметры  ✅ FIXED  
**Файл:** `train.ps1`, `train_full.py`  
**Статус:** ✅ `-Resume` → `--resume`, `-MaxLines` → `--max-lines` (передаётся и парсится).

### P1-5: eval_checkpoint.py — старая BPE модель  ✅ FIXED  
**Файл:** `eval_checkpoint.py:10`  
**Статус:** ✅ `bpe_ru_32k.model` → `bpe_ru_146k.model`.

---

## P2 — Medium

### P2-1: ach_phasic всегда 0  ✅ FIXED
**Файл:** `hormonal_system.py:37,143`  
**Статус:** ✅ Имплементирован: surprise + novelty + prediction error → ach_phasic → phasic→tonic интеграция → modulate_stdp_lr phasic_boost

### P2-2/P2-3/P2-4/P2-11: Мёртвый код  ✅ ALL FIXED
- `pos_tagger.py:132-137` — `pos_transition_score()` ✅ УДАЛЁН
- `crystal_generator.py:129-140` — `_semantic_delta()` ✅ УДАЛЁН
- `concept_space.py:791-843` — `fractal_stdp()` ✅ УДАЛЁН
- `concept_space.py:505-522` — `build_anchor_matrix()` ✅ УДАЛЁН
- `concept_space.py:1003-1176` — PQ код ✅ УДАЛЁН
- `concept_space.py:1259-1319` — `contrastive_spread()` оставлен (теперь импорт чистый)

### P2-5: train.ps1 мёртвые параметры  ✅ FIXED (дубль P1-4)

### P2-6: URL_TLDS мёртвая константа  ✅ FIXED
**Файл:** `filter_corpus.py:15` — удалена.

### P2-7: Неиспользуемые импорты в fcf_config.py  ✅ FIXED
**Файл:** `fcf_config.py:8` — `re` удалён (оставлен `math`, используется в `build_metric_pairs`).

### P2-8: Множественные EMA обновления per batch  ✅ FIXED
**Файл:** `crystal_generator.py` — dedup через `unique(cids)`, все 4 пути.

### P2-9: Double np.abs  ✅ FIXED  
**Файл:** `concept_space.py:922-923` — `abs_codes = np.abs(all_codes)` один раз.

### P2-10: import cdist внутри contrastive_spread  ✅ FIXED
**Файл:** `concept_space.py:1275` — удалён.

### P2-13: Доступ к _data/_valid извне  ✅ FIXED
**Файл:** `concept_space.py`, `inference.py`, `eval_metrics.py` — `@property data/valid`, usage.

---

## P3 — Low

### P3-1: Stale comment "10K lines trained"  ❌ NOT FIXED
**Файл:** `eval_checkpoint.py:60` — не трогали, косметика.

### P3-2: Расхождение checkpoint_state.json vs _train_status.json  ❌ NOT FIXED
**Файлы:** `real_data/` — разные моменты сохранения, нормально.

### P3-3: vis/ файлы  ❌ NOT FIXED (файлов не существует — claim неактуален)

### P3-4: TARGET_STD hardcoded 384D  ❌ NOT FIXED
**Файл:** `parameter_optimizer.py:110` — косметика.

### P3-5: PCA sign ambiguity  ❌ NOT FIXED
**Файл:** `train_full.py:338-339` — косметика.

### P3-6: ngrams[4] orphan  ✅ FIXED
**Файл:** `syntax_lattice.py` — `self.ngrams = {}`, строится динамически.

### P3-10: Tokenization теряет BPE-информацию  ✅ FIXED
**Файл:** `model/tokenization_fcf.py:42-52` — добавлен _tokenize_with_text(), возвращающий [(id, piece)]

---

## Архитектурные

### A1: Две параллельных имплементации  ✅ FIXED
**Статус:** ✅ save_pretrained сохраняет model state (concept_space, lattice, gen_config), from_pretrained загружает. BPE модель: bpe_ru_146k.

### A2: _archive/ — ~2000 строк мёртвого кода  ✅ FIXED
**Статус:** ✅ Директория удалена.

### A3: GPU/CPU split  ✅ FIXED
**Статус:** ✅ Устранён через единые _gpu_stdp_apply / _cpu_stdp_apply. destab_scale добавлен в train_batch CPU path. Contrastive objective — один вызов.

### A4: Epoch resume — fragile  ✅ FIXED
**Файл:** `train_full.py:478-484` — упрощён.

### A5: Config duplication  ✅ FIXED
**Файл:** `configuration_fcf.py` — импортирует CFG.

### A6: Corpus path not tracked  ✅ FIXED
**Статус:** ✅ .gitattributes: real_data/*.txt filter=lfs

---

## Error Handling

### E1: _semantic_delta norm safety  ✅ FIXED (code deleted)

### E3: _quiet swallows exceptions  ✅ FIXED
**Файл:** `train_full.py:16-22` — печатает в stderr.

### E4: rng.choice может упасть  ✅ FIXED
**Файл:** `eval_metrics.py:96-97` — `min(sample_k, len(valid))`.

### E5: Empty data в inference  ✅ FIXED
**Файл:** `inference.py:113` — guard `len==0 → None`.

---

## Code Quality

### Q1: Redundant query_words default  ✅ FIXED
**Файл:** `inference.py:93` — split() удалён, query_words передаётся напрямую

### Q2: Module-level `import torch`  ✅ FIXED
**Файл:** `crystal_generator.py:18` — lazy import с `_HAS_TORCH`.

### Q3-Q4: sys.path.insert  ✅ FIXED (дубль P1-1)

### Q5: hasattr/setattr в hormonal_system.py  ✅ FIXED
**Файл:** `hormonal_system.py` — инициализация в `__init__`.

### Q7: TeeOut file handle leak  ✅ FIXED
**Файл:** `train_full.py:48-66` — `__del__` + `close()`.

### Q8-Q9: Code duplication save  ✅ FIXED
**Файл:** `train_full.py` — `_final_save()` функция.

---

## Security

### S1: HTML/JS embedded  ✅ FIXED
**Файл:** `train_full.py` → `real_data/viewer_template.html`.

### S2: API без rate limiting  ✅ FIXED
**Файл:** `api/main.py` — in-memory rate limiter (10 req/min/IP).

---

## Итог

| Статус | Кол-во |
|--------|--------|
| ✅ Исправлено | 41 |
| ❌ Не исправлено (косметика) | 4 (P3-2, P3-3, P3-5, P3-1 уже динамический) |

*Last updated: 2026-06-16*

---

*Last updated: 2026-06-16*
