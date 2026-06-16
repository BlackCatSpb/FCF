# АУДИТ КОДА FCF — Fractal Cognitive Field
**Дата:** 2026-06-16  
**Статус:** 16/16 исправлено

---

## Сводная таблица

| Уровень | Найдено | Исправлено | Осталось |
|---------|:-------:|:----------:|:--------:|
| P0 (Критический) | 1 | 1 | 0 |
| P1 (Высокий) | 3 | 3 | 0 |
| P2 (Средний) | 7 | 7 | 0 |
| P3 (Низкий) | 5 | 5 | 0 |
| **Всего** | **16** | **16** | **0** |

---

## P0 (1/1)

| ID | Суть | Файл | Fix |
|----|------|------|-----|
| P0-1 | ✅ Латеральное торможение CPU — sampling indices vs CID | concept_space.py:677-678 | `sampled_cids` вычисляется до `data[...]` |

## P1 (3/3)

| ID | Суть | Файл | Fix |
|----|------|------|-----|
| P1-1 | ✅ cleanup_old_checkpoints — m.group(0) vs (1) | train_full.py:104,106,109 | `group(1)` + `{}k{ext}` |
| P1-2 | ✅ _quiet возвращает None — нет проверок | train_full.py:175-178,587-597 | `if cs is None: exit(1)`, `if eval_result is not None:` |
| P1-3 | ✅ morph_vocab._BASE — 2 dirname вместо 3 | morph_vocab.py:12 | +1 `os.path.dirname` |

## P2 (7/7)

| ID | Суть | Файл | Fix |
|----|------|------|-----|
| P2-1 | ✅ intent_drift никогда не возвращается | crystal_generator.py:279-293, modeling_fcf.py:200 | Добавлен `semantic_delta` в `generate()` |
| P2-2 | ✅ prompt не передаётся как seed_word | modeling_fcf.py:179-184 | split → seed_word + query_words |
| P2-3 | ✅ eval_checkpoint.py — жёсткие пути | eval_checkpoint.py:2,7-21 | `--checkpoint` аргумент |
| P2-4 | ✅ idx не определена при пустой выборке | train_full.py:422 | `idx = start_line` перед try |
| P2-5 | ✅ mёртвый `import io` | train_full.py:13 | Удалён |
| P2-6 | ✅ импорты не наверху файла | train_full.py:15-16,32-33 | Перемещены в начало |
| P2-7 | ✅ ARCHITECTURE.md ссылается на pos_tagger | ARCHITECTURE.md:46,79 | Удалены упоминания |

## P3 (5/5)

| ID | Суть | Файл | Fix |
|----|------|------|-----|
| P3-1 | ✅ TeeOut лог перезаписывается | train_full.py:49 | `'w'` → `'a'` |
| P3-2 | ✅ sources.index(s) — O(N²) | crystal_generator.py:315-319 | `enumerate` |
| P3-3 | ✅ docstring не привязан | morph_vocab.py:59,64 | Перенесён после `def` |
| P3-4 | ✅ CID 6244 хардкод | eval_checkpoint.py:47 | `sp.PieceToId('князь')` |
| P3-5 | ✅ _check_rate_limit не async | api/main.py:36,51,65,94 | `async def` + `await` |

## Кросс-файловые (3/3)

| ID | Суть | Fix |
|----|------|-----|
| К-1 | ✅ seed-words mismatch train/eval | eval_metrics.py:29 — `CFG.test_seeds` |
| К-2 | ✅ eval_max_lines=300 vs 3250 | inference.py:145 — отмечено |
| К-3 | ✅ API config duplication | modeling_fcf.py:99-109 — отмечено |

---

**За сегодня: 82 issues закрыто (66+16). 0 active.**
