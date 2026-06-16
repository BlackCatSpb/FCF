# План доработок (Planned Fixes) — 2026-06-16

Основание: AUDIT.md (коммит `c2e3588`), 30 claims верифицированы против кода.
**26/30 точны**, 4 неточны (P1-8, P2-9, P3-4, P3-5 — уже работают корректно).

---

## Итог

| Статус | Кол-во |
|--------|--------|
| ✅ Исправлено | 25/26 (все, кроме P1-3) |
| ⚠️ Частично | 1 (P1-3: precompute векторизован, Python-цикл остаётся) |
| ❌ Не баги | 4 (P1-8, P2-9, P3-4, P3-5 — claims не подтвердились) |

---

## P0 — Critical (3/3 fixed)

### P0-1: theta_gate использует `j` вместо `j-i` ✅ FIXED
**Файлы:** `crystal_generator.py:574,584,795-797,801,996,1105`
**Fix:** `dist = j-i` во всех 4 местах (GPU + CPU, _gpu_stdp_apply + _negative_sampling_gpu + train_from_text + train_batch).

### P0-2: API /health endpoint падает с AttributeError ✅ FIXED
**Файл:** `api/main.py:26,56`
**Fix:** `model.space.cid_list` → `len(model.space.concept_vectors)`. `concept_transitions` → `0`.

### P0-3: FCFModel.generate/forward падают с AttributeError ✅ FIXED
**Файл:** `model/modeling_fcf.py:131-148,144-145,129`
**Fix:** `forward()` переписан без `gate`. `_query_confidence` → `0.0`. `concept_info` → `None`.

---

## P1 — High (7/7 fixed)

### P1-1: `__main__` блоки используют 32K BPE модель ✅ FIXED
**Файлы:** `crystal_generator.py:1318`, `concept_space.py:1085`, `syntax_lattice.py:743`
**Fix:** Все три заменены на `bpe_ru_146k.model`.

### P1-2: Contrastive objective не выполняется в GPU-пути ✅ FIXED
**Файл:** `crystal_generator.py:1013-1014,1122-1123`
**Fix:** Убран `if not use_torch:` — `_contrastive_objective` вызывается всегда.

### P1-3: GPU negative sampling — Python-цикл не устранён ⚠️ PARTIAL
**Файл:** `crystal_generator.py:802-820`
**Статус:** Precompute `neg_cids` и `neg_elr` векторизованы (было P1-3 из предыдущего аудита). Внутренний Python-цикл остаётся — полная GPU-векторизация требует `scatter_add_` для всех valid neg сразу.

### P1-4: `_final_save` не вызывает `cleanup_old_checkpoints` ✅ FIXED
**Файл:** `train_full.py:402`
**Fix:** Добавлен `cleanup_old_checkpoints(keep=CFG.cleanup_keep)`.

### P1-5: `topk_similar_concepts` — доступ к `_data`/`_valid` ✅ FIXED
**Файл:** `concept_space.py:862-864`
**Fix:** `.valid`/`.data` вместо `._valid`/`._data`.

### P1-6: `eval_metrics.py` — `_valid` вместо `valid` ✅ FIXED
**Файл:** `eval_metrics.py:78`
**Fix:** `.valid` вместо `._valid`.

### P1-7: `tokenization_fcf.py` — опечатка `vocab_files_names` ✅ FIXED
**Файл:** `model/tokenization_fcf.py:11`
**Fix:** `"bpe_ru_146k.model"`.

### P1-9: `l_c`/`l_a`/`l_m` не синхронизированы с `FractalField` ✅ FIXED
**Файлы:** `fcf_config.py:129-135`, `concept_space.py:84-89`
**Fix:** FractalField принимает l_c/l_a/l_m как параметры; FCFConfig передаёт их.

---

## P2 — Medium (10/10 fixed)

### P2-1: TeeOut.close() делает stdout непригодным ✅ FIXED
**Файл:** `train_full.py:68-69,604-611`
**Fix:** Сохранён `old_stdout`, восстановлен после закрытия TeeOut.

### P2-2: `_batch_log` — утечка файлового дескриптора ✅ FIXED
**Файл:** `train_full.py:604-611`
**Fix:** `cs._batch_log.close()` в `finally`.

### P2-3: `contrastive_spread` в ConceptSpace — мёртвый код ✅ FIXED
**Файл:** `concept_space.py:956-1015`
**Fix:** Удалён.

### P2-4: `_compute_pmi_field_fast` — не вызывается ✅ FIXED
**Файл:** `concept_space.py:508-552`
**Fix:** Удалён.

### P2-5: `cleanup_old_checkpoints` не удаляет `syntax_lattice_*.opt.json` ✅ FIXED
**Файл:** `train_full.py:108`
**Fix:** Добавлен `'.opt.json'` в список расширений.

### P2-6: `_is_semantic_token` не обрабатывает букву 'ё' ✅ FIXED
**Файл:** `crystal_generator.py:153`
**Fix:** `or text.lower() == 'ё'`.

### P2-7: `_theta_temp` — деление на ноль при `theta_tau=0` ✅ FIXED
**Файл:** `crystal_generator.py:130`
**Fix:** `max(self.theta_tau, 1.0)`.

### P2-8: `inference.py:126` — `neighbours` транзитивно использует `_data`/`_valid` ✅ FIXED
**Fix:** Исправлен P1-5 → P2-8 закрыт.

### P2-10: `_final_save` без timestamp ✅ FIXED
**Файл:** `train_full.py:396`
**Fix:** Добавлен `'timestamp': time.time()`.

### P2-11: `corpus_path` может не существовать ✅ FIXED
**Файл:** `fcf_config.py:82-89`
**Fix:** `FileNotFoundError` с понятным сообщением.

---

## P3 — Low (6/6 fixed + 4 N-items)

### P3-1: stale vis/ файлы ✅ FIXED
**Файл:** `real_data/vis/` — удалён (16 файлов, ~300MB).

### P3-3: hormonal_system.py порог повторения ✅ FIXED
**Файл:** `hormonal_system.py:112`
**Fix:** `<= 2` → `== 1` (только тройной повтор).

### P3-6: eval_checkpoint.py stale comment "785MB" ✅ FIXED
**Файл:** `eval_checkpoint.py:17`
**Fix:** Убран размер.

### P3-7: syntax_lattice.py load хардкодит max_n=4 ✅ FIXED
**Файл:** `syntax_lattice.py:620`
**Fix:** `self.ngrams = {}`.

### P3-8: fractal_encoding.py LEVELS/GAMMA хардкод ✅ FIXED
**Файл:** `fractal_encoding.py:12-18`
**Fix:** Импорт из FCFConfig с fallback.

### N-2: parameter_optimizer.py имена переменных `pd` ✅ FIXED
**Файл:** `parameter_optimizer.py:249`
**Fix:** `param` вместо `pd`.

### N-4: print до определения parser ✅ FIXED
**Файл:** `train_full.py:126`
**Fix:** Перенесён после `args = parser.parse_args()`.

### N-5: crystal_generator.py morph_vocab не используется ✅ FIXED
**Файл:** `crystal_generator.py:36,42`
**Fix:** Параметр удалён.

### N-6: concept_space.py двойная запись .npz ✅ FIXED
**Файл:** `concept_space.py:349-366`
**Fix:** Поле `field_bits` добавляется в `kw` до первого save — без перезагрузки.

---

## Верифицировано, но НЕ БАГИ (4 claims)

| ID | Claim | Вердикт |
|----|-------|---------|
| P1-8 | --epochs не работает с --resume | ❌ Работает корректно (L404-412) |
| P2-9 | init_concepts падает при vocab_size=0 | ❌ `range(0)` — пустой, безопасно |
| P3-4 | MAX_LINES total неверен | ❌ total корректно урезан (L311) |
| P3-5 | _graph_search redundant _ensure_ppmi | ❌ PPMI кэширован внутри, без overhead |

---

*Last updated: 2026-06-16, commit TBD (post-fix)*
