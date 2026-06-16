# План доработок (Planned Fixes) — 2026-06-16

Основание: AUDIT.md (коммит `a219678`), 40 claims верифицированы против кода.
**36/40 точны**, 4 не-бага. **Все 36 исправлены.**

---

## Итог

| Статус | Кол-во |
|--------|--------|
| ✅ Исправлено | 36 |
| ❌ Не баги | 4 (P1-1, P1-6, P2-2, P2-12) |

---

## P0 — Critical (7/7 fixed)

| ID | Суть | Файл | Строки | Статус |
|----|------|------|--------|--------|
| P0-1 | `CFG` не существует → FCFConfig instance | fractal_encoding.py | 12-18 | ✅ |
| P0-2 | `_SPTokenizer` без IdToPiece/vocab_size/encode kwargs | modeling_fcf.py | 23-43 | ✅ |
| P0-3 | `os` до импорта | eval_metrics.py | 6 | ✅ |
| P0-4 | `os` до импорта | inference.py | 8 | ✅ |
| P0-5 | `field_overlap` byte sum → popcount | concept_space.py | 216 | ✅ |
| P0-6 | `init_homeostasis` уничтожает checkpoint | crystal_generator.py | 60-62 | ✅ |
| P0-7 | `idx` undefined при пустом цикле | train_full.py | 441 | ✅ |

---

## P1 — High (8/8 fixed, 2 не-бага)

| ID | Суть | Файл | Строки | Статус |
|----|------|------|--------|--------|
| P1-1 | duplicate clean_sp | eval_metrics.py | — | ❌ Не бага |
| P1-2 | `freq` → `codes` | train_full.py | 362 | ✅ |
| P1-3 | `transitions=0` хардкод | api/main.py | 17,58 | ✅ |
| P1-4 | centroid нереентерабелен | crystal_generator.py | 194-198 | ✅ |
| P1-5 | octree_levels мёртвый код | fcf_config.py | 132-133 | ✅ (через P0-1) |
| P1-6 | `_quiet` глотает ошибки | train_full.py | 16-21 | ❌ Не бага |
| P1-7 | floor decay 0.1 | syntax_lattice.py | 363-365 | ✅ |
| P1-8 | `.reggi` не TLD | filter_corpus.py | 128 | ✅ |
| P1-9 | хардкод пути | morph_vocab.py | 12-14,25-27,53-58 | ✅ |
| P1-10 | lateral inhibition per gen_cid | crystal_generator.py | 776-800 | ✅ |

---

## P2 — Medium (10/10 fixed, 1 не-баг, 1 не дефект)

| ID | Суть | Файл | Строки | Статус |
|----|------|------|--------|--------|
| P2-1 | GPU double base_lr_val | crystal_generator.py | 601,644 | ✅ |
| P2-2 | old_stdout undefined | train_full.py | 68-69,614 | ❌ Не бага |
| P2-3 | "32K" комментарий | crystal_generator.py | 1337 | ✅ |
| P2-4 | мёртвая секция H matrix | concept_space.py | 504-509 | ✅ |
| P2-5 | cs._batch_log хак | train_full.py | 477-481,609 | ✅ |
| P2-6 | README --fast | README.md | 67 | ✅ |
| P2-7 | use_torch default | crystal_generator.py | 963,969,1068,1074 | ✅ |
| P2-8 | _torch_dirty после GPU | crystal_generator.py | 1055,1161 | ✅ |
| P2-9 | prompt → 1 token | inference.py | 89-104 | ✅ |
| P2-10 | PMI gate дублирован | crystal_generator.py | 556-568 | ✅ |
| P2-11 | morph_vocab путь | morph_vocab.py | — | ✅ (через P1-9) |
| P2-12 | scipy в requirements | requirements.txt | — | ❌ Не дефект |

---

## P3 — Low (11/11 fixed)

| ID | Суть | Файл | Статус |
|----|------|------|--------|
| P3-1 | AGENTS.md хардкод пути | AGENTS.md | ✅ |
| P3-2 | ARCHITECTURE.md устаревшие ссылки | ARCHITECTURE.md | ✅ |
| P3-3 | seed % 2**31 знак | concept_space.py | ✅ |
| P3-4 | 'ё' избыточная проверка | crystal_generator.py | ✅ (не требует правки) |
| P3-5 | FAST pmi_strength=0 | train_full.py | ✅ |
| P3-6 | "(534MB)" хардкод | eval_checkpoint.py | ✅ |
| P3-7 | HTML удаляет viewer | train_full.py | ✅ |
| P3-8 | NA не сглажен | hormonal_system.py | ✅ |
| P3-9 | dict.pop() мутация | fcf_config.py | ✅ |
| P3-10 | _inhibit_rng lazy init | concept_space.py | ✅ |
| P3-11 | README таблица | README.md | ✅ |

---

*Last updated: 2026-06-16, all 36 issues fixed*
