# Аудит проекта FCF — Fractal Cognitive Field

**Дата:** 2026-06-16  
**Коммит:** `a219678` → `HEAD` (post-fix)  
**Статус:** Все 36 issues исправлены

---

## Сводная таблица

| Уровень | Найдено | Исправлено | Осталось |
|---------|---------|:----------:|:--------:|
| P0 (Critical) | 7 | 7 | 0 |
| P1 (High) | 10 | 8 (2 не-бага) | 0 |
| P2 (Medium) | 12 | 10 (1 не-баг, 1 не дефект) | 0 |
| P3 (Low) | 11 | 11 | 0 |
| **Всего** | **40** | **36 + 4 не-бага** | **0** |

---

## P0 — Critical (7/7 fixed)

### P0-1: ✅ `fractal_encoding.py` — `CFG` → `FCFConfig` instance
**Файл:** `eva/symbolic/fractal_encoding.py:12-18`
**Fix:** `from eva.symbolic.fcf_config import FCFConfig as _FCFConfig; __cfg = _FCFConfig(); LEVELS = __cfg.octree_levels; GAMMA = __cfg.octree_gamma`

### P0-2: ✅ `modeling_fcf.py` — `_SPTokenizer` добавлены методы
**Файл:** `model/modeling_fcf.py:23-43`
**Fix:** Добавлены `IdToPiece()`, `vocab_size()`, `encode(text, add_bos, add_eos)`.

### P0-3: ✅ `eval_metrics.py` — порядок импорта `os`
**Файл:** `eval_metrics.py:6`
**Fix:** `import sys, os; sys.path.insert(...)`

### P0-4: ✅ `inference.py` — порядок импорта `os`
**Файл:** `inference.py:8`
**Fix:** `import sys, os; sys.path.insert(...)`

### P0-5: ✅ `concept_space.py` — `field_overlap` popcount
**Файл:** `eva/symbolic/concept_space.py:216`
**Fix:** `np.unpackbits(np.bitwise_and(ba, bb)).sum()`

### P0-6: ✅ `crystal_generator.py` — `init_homeostasis` условный
**Файл:** `eva/symbolic/crystal_generator.py:60-62`
**Fix:** `if not self.cs.concept_usage: self.cs.init_homeostasis()`

### P0-7: ✅ `train_full.py` — `idx` default при пустом цикле
**Файл:** `train_full.py:441`
**Fix:** `idx = start_line` перед циклом

---

## P1 — High (8/8 fixed, 2 не-бага)

### P1-1: ❌ Не бага — `eval_metrics.py` clean_sp не дублируется
Функция определена один раз на уровне модуля, `run_eval` в файле отсутствует.

### P1-2: ✅ `train_full.py` — `freq` → `codes`
**Файл:** `train_full.py:362`

### P1-3: ✅ `api/main.py` — `transitions` счётчик
**Файл:** `api/main.py:17,58`
**Fix:** `_trained_lines` модульный счётчик, инкремент после генерации.

### P1-4: ✅ `crystal_generator.py` — centroid локальная переменная
**Файл:** `eva/symbolic/crystal_generator.py:194-198,223`
**Fix:** `self._centroid` → локальная `centroid`.

### P1-5: ✅ `fcf_config.py` — octree_levels/gamma (закрыто P0-1)
Автоматически исправлено фиксом P0-1.

### P1-6: ❌ Не бага — `_quiet` уже пишет в stderr
`print(f"[WARN] ...", file=sys.stderr)` — корректно.

### P1-7: ✅ `syntax_lattice.py` — floor decay для ngram
**Файл:** `eva/symbolic/syntax_lattice.py:363-365,373-375`
**Fix:** `counter[ncid] *= self.decay; if counter[ncid] < 1e-6: del counter[ncid]`

### P1-8: ✅ `filter_corpus.py` — `.reggi` убран
**Файл:** `filter_corpus.py:128`

### P1-9: ✅ `morph_vocab.py` — хардкод путей
**Файл:** `eva/symbolic/morph_vocab.py:1,12-14,25-27,53-58`
**Fix:** `_BASE = os.path.dirname(...)`, defaults через `os.path.join(_BASE, ...)`.

### P1-10: ✅ `crystal_generator.py` — lateral inhibition вне цикла
**Файл:** `eva/symbolic/crystal_generator.py:776-800`
**Fix:** Один вызов `_lateral_inhibition_fractal` после цикла gen_cid.

---

## P2 — Medium (10/10 fixed, 1 не-баг, 1 не дефект)

### P2-1: ✅ GPU double `base_lr_val`
**Файл:** `eva/symbolic/crystal_generator.py:601`
**Fix:** `base_lr_val` убран из `effective_lr`, остался только как шаг.

### P2-2: ❌ Не бага — `old_stdout` защищён (module-level)
Определён на уровне модуля до любого кода.

### P2-3: ✅ "32K" комментарий
**Файл:** `eva/symbolic/crystal_generator.py:1337`
**Fix:** → "146K".

### P2-4: ✅ Мёртвая секция `# ── H matrix + BMSSP ──`
**Файл:** `eva/symbolic/concept_space.py:504-509`
**Fix:** Удалена.

### P2-5: ✅ `cs._batch_log` → локальная переменная
**Файл:** `train_full.py:477-481,609`
**Fix:** `batch_log` как локальная, закрывается в `finally`.

### P2-6: ✅ README `--fast` документация
**Файл:** `README.md:67`

### P2-7: ✅ `use_torch` default конфликт
**Файл:** `eva/symbolic/crystal_generator.py:963,969,1068,1074`
**Fix:** `use_torch=None` → `CFG.use_torch` fallback.

### P2-8: ✅ GPU STDP `_torch_dirty`
**Файл:** `eva/symbolic/crystal_generator.py:1055,1161`
**Fix:** `self._torch_dirty = True` после GPU STDP/neg-sampling.

### P2-9: ✅ inference prompt — многословный промпт
**Файл:** `inference.py:89-104`
**Fix:** `generate()` разделяет на seed_word + query_words.

### P2-10: ✅ PMI gate дублирование CPU/GPU
**Файл:** `eva/symbolic/crystal_generator.py:556-568,1014-1017,1121-1124`
**Fix:** Вынесен в `_apply_pmi_gate()`.

### P2-11: ✅ `morph_vocab.py` — хардкод пути (см. P1-9)
Закрыто фиксом P1-9.

### P2-12: ❌ Не дефект — `scipy` есть в `requirements.txt`

---

## P3 — Low (11/11 fixed)

### P3-1: ✅ AGENTS.md — хардкод → относительные пути
### P3-2: ✅ ARCHITECTURE.md — ссылки на AUDIT.md/PLAN.md
### P3-3: ✅ concept_space.py — `abs(seed) % (2**31)`
### P3-4: ✅ crystal_generator.py — проверка 'ё' оставлена (избыточна, но не вредна)
### P3-5: ✅ train_full.py — FAST сообщение без `pmi_strength=0`
### P3-6: ✅ eval_checkpoint.py — размер убран из сообщения
### P3-7: ✅ train_full.py — HTML glob исключает viewer.html
### P3-8: ✅ hormonal_system.py — NA сглаживание (`+= (target - self) * 0.3`)
### P3-9: ✅ fcf_config.py — `p.copy().pop()` вместо `p.pop()`
### P3-10: ✅ concept_space.py — `_inhibit_rng` в `__init__`
### P3-11: ✅ README.md — устаревшая таблица заменена

---

## Итог

| Метрика | Значение |
|---------|----------|
| Всего проблем в audit | 40 |
| Исправлено | 36 |
| Не баги / не дефекты | 4 (P1-1, P1-6, P2-2, P2-12) |
| Осталось | 0 |
| Верификация AUDIT | 40/40 |

*Все issues закрыты. Коммит: HEAD (post-fix)*
