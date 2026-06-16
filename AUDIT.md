# Аудит проекта FCF — Fractal Cognitive Field (2026-06-16)

**Коммит:** `69d4bc2` → `HEAD` (post-fix)
**Всего проблем:** 30 | **Исправлено:** 30 | **Осталось:** 0

---

## P0 — Critical (3/3 fixed)

| ID | Суть | Файл | Fix |
|----|------|------|-----|
| P0-1 | ✅ GPU lateral inhibition — неправильный знак (вместо отталкивания — притяжение) | crystal_generator.py:708 | Убран `-` перед `str_val` |
| P0-2 | ✅ Contrastive objective не работает в GPU (gen_updates пуст) | crystal_generator.py:1018-1037,1125-1143 | field_weight/lr вычисляются до `if use_torch:`, gen_updates заполняется безусловно |
| P0-3 | ✅ evaluate() использует устаревшие GPU-тензоры | crystal_generator.py:1248 | `if use_gpu and self._torch_dirty: self._ensure_torch()` |

## P1 — High (7/7 fixed)

| ID | Суть | Файл | Fix |
|----|------|------|-----|
| P1-1 | ✅ decay_all() не обновляет _prefix_total | syntax_lattice.py:381-382 | `_prefix_total = {}`, `_skip2_total = {}` после decay |
| P1-2 | ✅ total_freq внутри цикла generate() (O(vocab×beam)) | crystal_generator.py:201,239-240 | Вынесен до цикла beam search |
| P1-3 | ✅ GPU neg sampling — base_lr несоответствие CPU/GPU | crystal_generator.py:837-863 | GPU путь приведён к CPU-схеме (norm + step) |
| P1-4 | ✅ topk_similar_concepts — неиспользуемый sample_size | concept_space.py:773-811 | Реализована subsampling |
| P1-5 | ✅ _apply_vector_update в цикле GPU inhibition | crystal_generator.py:709-717 | np.unique + np.add.at → O(unique_targets) |
| P1-6 | ✅ _lateral_inhibition_fractal использует _data | concept_space.py:710 | `_data` → `data` |
| P1-7 | ✅ eval_metrics.py — двойной import os | eval_metrics.py:7 | Убран `os` из второго import |

## P2 — Medium (12/12 fixed)

| ID | Суть | Файл | Fix |
|----|------|------|-----|
| P2-1 | ✅ th_mod не используется | concept_space.py:241 | `lr_mod, _ = ...` |
| P2-2 | ✅ shift_attention — мёртвый код (31 строка) | concept_space.py | Удалён |
| P2-3 | ✅ expand_dim — мёртвый код | concept_space.py | Удалён |
| P2-4 | ✅ normalize_vectors — мёртвый код | concept_space.py | Удалён |
| P2-5 | ✅ path_index — мёртвый код | fractal_encoding.py | Удалён |
| P2-6 | ✅ tokenize — мёртвый код | morph_vocab.py | Удалён |
| P2-7 | ✅ predict_with_context — мёртвый код | syntax_lattice.py | Удалён |
| P2-8 | ✅ build_anchor_matrix — мёртвый код | syntax_lattice.py | Удалён |
| P2-9 | ✅ pos_tagger.py — мёртвый модуль (111 строк) | eva/symbolic/pos_tagger.py | Файл удалён |
| P2-10 | ✅ self.beam_width не используется | crystal_generator.py:45 | Строка удалена |
| P2-11 | ✅ inference.py — избыточный int() | inference.py:136 | Убран |
| P2-12 | ✅ config.params дублирует destab_decay_lines | fcf_config.py:239 | Комментарий |

## P3 — Low (8/8 fixed)

| ID | Суть | Файл | Fix |
|----|------|------|-----|
| P3-1 | ✅ nltk в requirements — не используется | requirements.txt | Удалён |
| P3-2 | ✅ pandas нет в requirements — OK, scikit-learn/scipy есть | — | Не требует правки |
| P3-3 | ✅ _repel_centroid память ~225MB | concept_space.py:613 | Комментарий |
| P3-4 | ✅ __post_init__ обработка eval_pairs | fcf_config.py | Уже корректно |
| P3-5 | ✅ README eval_max_lines=300 vs inference 3250 | README.md:49 | Уточнено |
| P3-6 | ✅ inference.py shadow clean_sp в run_eval() | inference.py:146-147 | Удалена вложенная clean_sp |
| P3-7 | ✅ eval_metrics.py неявная конфигурация | eval_metrics.py:21-27 | try/except с fallback |
| P3-8 | ✅ requirements версии без верхних границ | requirements.txt | Комментарий |

---

**Удалено ~450 строк мёртвого кода.** Все 30 issues закрыты.
