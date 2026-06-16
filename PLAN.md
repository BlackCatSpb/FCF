# План доработок (Planned Fixes) — 2026-06-16

Основание: AUDIT.md (коммит `c2e3588`), 30 claims верифицированы против кода.
**26/30 точны**, 4 неточны (P1-8, P2-9, P3-4, P3-5 — уже работают корректно).

---

## P0 — Critical (3 issues)

### P0-1: theta_gate использует `j` вместо `j-i`
**Файлы:** `crystal_generator.py:574,584,795,799,994,1104`
**Проблема:** `theta = exp(-min(j,5)/tau)` — использует абсолютную позицию `j`, а не расстояние `j-i`. Для длинных предложений конец получает меньший LR, чем начало, при одинаковых расстояниях.
**Fix:** `dist = abs(j-i)`; `theta = exp(-min(dist,5)/max(theta_tau,1))` — 4 места.

### P0-2: API /health endpoint падает с AttributeError
**Файл:** `api/main.py:26,56`
**Проблема:** `model.space.cid_list` (строка 26) и `model.space.concept_transitions` (строка 56) — удалены при рефакторинге.
**Fix:** `len(model.space.concept_vectors)` вместо `cid_list`. `0` вместо `concept_transitions` (или вычислить).

### P0-3: FCFModel.generate/forward падают с AttributeError
**Файл:** `model/modeling_fcf.py:148,157,207-209`
**Проблема:** `self.generator.gate` удалён; `_query_confidence` не определён; `concept_info` не существует.
**Fix:** Заменить на рабочие заглушки: `forward` → возвращать `FCFOutput` без gate. `generate` → удалить `_query_confidence` и `concept_info.intent_anchor`.

---

## P1 — High (7 issues)

### P1-1: `__main__` блоки используют 32K BPE модель
**Файлы:** `crystal_generator.py:1318`, `concept_space.py:1085`, `syntax_lattice.py:743`
**Проблема:** Три отладочных `__main__` блока используют `bpe_ru_32k.model` или `bpe_ru.model` вместо `bpe_ru_146k.model`.
**Fix:** Заменить на `bpe_ru_146k.model`.

### P1-2: Contrastive objective не выполняется в GPU-пути
**Файл:** `crystal_generator.py:1012-1013,1122-1123`
**Проблема:** `if not use_torch: self._contrastive_objective(...)` — GPU-путь полностью пропускает contrastive push.
**Fix:** Реализовать GPU-версию contrastive или выполнять на CPU с корректными `gen_updates` после GPU-шага.

### P1-3: GPU negative sampling — Python-цикл не устранён
**Файл:** `crystal_generator.py:802-820`
**Проблема:** После precompute neg_cids/neg_elr остаётся двойной Python-цикл с per-element numpy-операциями.
**Fix:** Собрать все valid neg-индексы в тензор, вычислить сдвиги одним matmul, применить через `scatter_add_`.

### P1-4: `_final_save` не вызывает `cleanup_old_checkpoints`
**Файл:** `train_full.py:390-400`
**Проблема:** При KeyboardInterrupt и финальном сохранении старые чекпоинты не чистятся.
**Fix:** Добавить `cleanup_old_checkpoints(keep=CFG.cleanup_keep)` в `_final_save`.

### P1-5: `topk_similar_concepts` — доступ к `_data`/`_valid`
**Файл:** `concept_space.py:862-864`
**Проблема:** Использует `self.concept_vectors._data` и `self.concept_vectors._valid` вместо публичных `.data`/`.valid`.
**Fix:** Заменить на `.data`/`.valid`.

### P1-6: `eval_metrics.py` — `_valid` вместо `valid`
**Файл:** `eval_metrics.py:78`
**Проблема:** `sum(cs.concept_vectors._valid)` — приватный атрибут.
**Fix:** Заменить на `.valid`.

### P1-7: `tokenization_fcf.py` — опечатка `vocab_files_names`
**Файл:** `model/tokenization_fcf.py:11`
**Проблема:** `vocab_files_names` (пропущена 'i') — HF не найдёт файлы. Значение `bpe_ru.model` неактуально.
**Fix:** `vocab_files_names = {"spm_file": "bpe_ru_146k.model"}`.

### P1-9: `l_c`/`l_a`/`l_m` не синхронизированы с `FractalField`
**Файлы:** `fcf_config.py:129-135`, `concept_space.py:87-89`
**Проблема:** Дублированная логика вычисления — при изменении `latent_dim` в конфиге FractalField не синхронизируется.
**Fix:** Передавать `l_c/l_a/l_m` из FCFConfig в FractalField.

---

## P2 — Medium (10 issues)

### P2-1: TeeOut.close() делает stdout непригодным
**Файл:** `train_full.py:68,601-603`
**Проблема:** `sys.stdout.close()` в `finally` не восстанавливает `sys.__stdout__`.
**Fix:** `sys.stdout = sys.__stdout__` после закрытия TeeOut.

### P2-2: `_batch_log` — утечка файлового дескриптора
**Файл:** `train_full.py:471-475`
**Проблема:** `cs._batch_log = open(...)` — никогда не закрывается.
**Fix:** Закрывать в `_final_save` или через context manager.

### P2-3: `contrastive_spread` в ConceptSpace — мёртвый код (60 строк)
**Файл:** `concept_space.py:956-1015`
**Проблема:** Метод нигде не вызывается.
**Fix:** Удалить.

### P2-4: `_compute_pmi_field_fast` — не вызывается
**Файл:** `concept_space.py:508-552`
**Проблема:** `build_octree_fields` полностью заменил PMI-подход.
**Fix:** Удалить.

### P2-5: `cleanup_old_checkpoints` не удаляет `syntax_lattice_*.opt.json`
**Файл:** `train_full.py:107-109`
**Проблема:** Цикл удаления syntax_lattice не обрабатывает `.opt.json`.
**Fix:** Добавить `'.opt.json'` в список расширений.

### P2-6: `_is_semantic_token` не обрабатывает букву 'ё'
**Файл:** `crystal_generator.py:154`
**Проблема:** Диапазон `'а' <= text <= 'я'` исключает 'ё' (U+0451).
**Fix:** `or text.lower() == 'ё'`.

### P2-7: `_theta_temp` — деление на ноль при `theta_tau=0`
**Файл:** `crystal_generator.py:131`
**Проблема:** `exp(-word_num / theta_tau)` — нет `max(..., 1.0)`.
**Fix:** `self.theta_tau` → `max(self.theta_tau, 1.0)`.

### P2-8: `inference.py:126` — `neighbours` транзитивно использует `_data`/`_valid`
**Файл:** `inference.py:126`
**Проблема:** Вызов `topk_similar_concepts` (см. P1-5).
**Fix:** Исправить P1-5, это закроет и P2-8.

### P2-10: `_final_save` без timestamp
**Файл:** `train_full.py:395`
**Проблема:** `ckpt = {'epoch': epoch, 'line': total_lines}` — нет timestamp.
**Fix:** Добавить `'timestamp': time.time()`.

### P2-11: `corpus_path` может не существовать
**Файл:** `fcf_config.py:82-83`
**Проблема:** Нет проверки существования файла корпуса.
**Fix:** Добавить `os.path.exists()` + понятную ошибку.

---

## P3 — Low (6 issues + 5 N-items)

### P3-1: stale vis/ файлы (16 файлов, ~300MB)
**Файл:** `real_data/vis/`
**Fix:** Удалить старые `points_*.json`.

### P3-2: PCA sign ambiguity
**Файл:** `train_full.py:349`
**Fix:** Косметика — можно игнорировать.

### P3-3: hormonal_system.py порог повторения неточен
**Файл:** `hormonal_system.py:112-113`
**Fix:** Уточнить условие `<=2`.

### P3-6: eval_checkpoint.py stale comment "785MB"
**Файл:** `eval_checkpoint.py:19`
**Fix:** Удалить или актуализировать.

### P3-7: syntax_lattice.py load хардкодит max_n=4
**Файл:** `syntax_lattice.py:620`
**Fix:** Использовать динамическое построение.

### P3-8: fractal_encoding.py LEVELS/GAMMA хардкод
**Файл:** `fractal_encoding.py:12-13`
**Fix:** Импортировать из FCFConfig.

### N-1: FAST mode print после парсинга аргументов
**Файл:** `train_full.py:130-131`
**Fix:** Перенести в более подходящее место.

### N-2: parameter_optimizer.py имена переменных `pd`
**Файл:** `parameter_optimizer.py:249`
**Fix:** Переименовать в `param`.

### N-3: train_full.py проверка cs.H
**Файл:** `train_full.py:182-185`
**Fix:** Уже корректно через `hasattr`.

### N-4: print до определения parser
**Файл:** `train_full.py:112`
**Fix:** Перенести print после parser.

### N-5: crystal_generator.py morph_vocab не используется
**Файл:** `crystal_generator.py:36-41`
**Fix:** Удалить или закомментировать.

### N-6: concept_space.py двойная запись .npz
**Файл:** `concept_space.py:349-366`
**Fix:** Записывать field_bits сразу, без перезагрузки.

---

## Верифицировано, но НЕ БАГИ (4 claims)

| ID | Claim | Вердикт |
|----|-------|---------|
| P1-8 | --epochs не работает с --resume | ❌ Работает корректно (L404-412) |
| P2-9 | init_concepts падает при vocab_size=0 | ❌ `range(0)` — пустой, безопасно |
| P3-4 | MAX_LINES total неверен | ❌ total корректно урезан (L311) |
| P3-5 | _graph_search redundant _ensure_ppmi | ❌ PPMI кэширован внутри, без overhead |

---

## Очерёдность выполнения

### Phase A (Runtime crashes — делать сейчас)
- [ ] P0-1: theta_gate `j` → `j-i` (4 места)
- [ ] P0-2: /health endpoint — убрать `cid_list`, `concept_transitions`
- [ ] P0-3: FCFModel — убрать `gate`, `_query_confidence`, `concept_info`

### Phase B (Functional correctness)
- [ ] P1-2: Contrastive objective в GPU-пути
- [ ] P1-5 + P2-8: `topk_similar_concepts` → `.data`/`.valid`
- [ ] P1-6: `eval_metrics.py` → `.valid`
- [ ] P2-7: `_theta_temp` div by zero guard
- [ ] P2-6: 'ё' в `_is_semantic_token`

### Phase C (Cleanup & maintenance)
- [ ] P1-1: `__main__` BPE модели (3 файла)
- [ ] P1-7: tokenization_fcf.py typo + BPE path
- [ ] P2-3: `contrastive_spread` dead code
- [ ] P2-4: `_compute_pmi_field_fast` dead code
- [ ] N-5: morph_vocab unused param

### Phase D (Training pipeline quality)
- [ ] P1-3: Полная GPU-векторизация negative sampling
- [ ] P1-4: cleanup_old_checkpoints в _final_save
- [ ] P1-9: l_c/l_a/l_m sync
- [ ] P2-1: TeeOut stdout restore
- [ ] P2-2: _batch_log leak
- [ ] P2-5: .opt.json cleanup
- [ ] P2-10: timestamp в _final_save
- [ ] P2-11: corpus_path guard

### Phase E (Cosmetic)
- [ ] P3-1: удалить stale vis/ файлы
- [ ] P3-3: уточнить порог повторения
- [ ] P3-6: stale comment
- [ ] P3-7: ngrams load
- [ ] P3-8: LEVELS/GAMMA sync
- [ ] N-1, N-2, N-4, N-6

---

*Last updated: 2026-06-16, verified against commit c2e3588*
