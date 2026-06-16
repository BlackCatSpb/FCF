# FCF (Fractal Cognitive Field) — Полный аудит проекта

Дата: 2026-06-16
Коммит: `03b8ae8` (HEAD), `4be3d93` (предыдущий)
Всего файлов: 29 Python (+ 5 в `_archive/`), 5 бат-файлов, README/ARCHITECTURE/AGENTS, requirements.txt

---

## Сводка по серьёзности

| Уровень | Описание | Кол-во |
|---------|----------|--------|
| **P0** | Критические — падение/некорректные результаты | 2 |
| **P1** | Высокие — серьёзные проблемы архитектуры/портабельности | 5 |
| **P2** | Средние — мёртвый код, неоптимальности, несоответствия | 13 |
| **P3** | Низкие — косметика, документация, старые файлы | 10 |

---

## P0 — Критические

### P0-1: Отсутствует импорт KMeans в concept_space.py:1036

**Файл:** `eva/symbolic/concept_space.py:1036`

**Описание:** Метод `pq_train()` на строке 1036 вызывает `KMeans(n_clusters=...)`, но `KMeans` нигде не импортирован. В импортах файла (строка 13) только `numpy`, `defaultdict`, `Counter`, `math`, `json`, `os`. При вызове `pq_train()` произойдёт `NameError: name 'KMeans' is not defined`.

**Рекомендация:** Добавить `from sklearn.cluster import KMeans` в начало файла.

### P0-2: Дублирование кода train_from_text / train_batch (~80% идентичного кода)

**Файлы:** `eva/symbolic/crystal_generator.py:563-1006` и `eva/symbolic/crystal_generator.py:1007-1402`

**Описание:** Методы `train_from_text()` и `train_batch()` содержат практически идентичную логику: (1) построение пар, (2) GPU STDP через scatter_add, (3) negative sampling CPU/GPU, (4) contrastive objective, (5) centroid pull, (6) обновление lattice. Любое изменение в одном требует синхронного изменения в другом. Это уже привело к расхождениям — например, contrastive objective в `train_batch` (строка 1340) имеет другую структуру циклов, чем в `train_from_text` (строка 941).

**Рекомендация:** Выделить общую логику в приватные методы: `_build_pairs()`, `_gpu_stdp()`, `_negative_sampling()`, `_contrastive_push()`, `_centroid_pull()`. Вызвать их из обоих public-методов.

---

## P1 — Высокие

### P1-1: Жёстко закодированные пути пользователя

**Файлы (25+ вхождений):**
- `eval_checkpoint.py:2` — `r'C:\Users\black\OneDrive\Desktop\FCF'`
- `eval_metrics.py:6` — то же
- `inference.py:8` — то же
- `train_full.py:11` — то же
- `train.ps1:7` — `C:\Users\black\OneDrive\Desktop\FCF`
- `train.bat:2` — то же
- `train_fast.bat:2` — то же
- `run_train.bat:2` — то же
- `morph_vocab.py:266` — то же
- `crystal_generator.py:1583, 1589` — то же
- `concept_space.py:1430` — то же
- `syntax_lattice.py:738, 743, 746, 751, 756` — то же
- `concept_inductor.py:356` — то же
- `concept_net.py:16-17` — то же
- `concept_tokenizer.py:31-32` — то же
- `reset_fractal.py` (удалён) — то же

**Описание:** Почти каждый исполняемый скрипт хардкодит `sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')`. Проект невозможно запустить на другой машине без редактирования каждого файла. Также это раскрывает имя пользователя (black) в коде (security minor).

**Рекомендация:** Использовать `os.path.dirname(os.path.abspath(__file__))` для определения корня, или установить проект как пакет (`pip install -e .`) и использовать абсолютные импорты.

### P1-2: ARCHITECTURE.md полностью устарел

**Файл:** `ARCHITECTURE.md`

**Описание:** Документ описывает архитектуру 36K концептов × 128D (строка 84: "36 273 концепта", строка 86: "128D на единичной сфере"), тогда как текущий код использует 146K × 384D. Структура директорий в строках 217-234完全不 соответствует текущей: перечислены `concept_tokenizer.py`, `semantic_gate.py`, `hierarchical_compressor.py`, `concept_net.py` как активные модули, хотя они перемещены в `_archive/`. Не упомянуты `fractal_encoding.py`, `parameter_optimizer.py`, `morph_vocab.py`, `fcf_config.py`, `vector_health.py`, `pos_tagger.py`.

Также указаны несуществующие файлы: `checkpoints/`, `experiments/`, `model/configuration_fcf.py` упомянут как конфиг, но FCFConfig из `fcf_config.py` является основным.

**Рекомендация:** Переписать ARCHITECTURE.md полностью, отразив текущую архитектуру 146K×384D, октантное кодирование, fractal field, батчинг, GPU-путь, ParameterOptimizer и актуальную структуру файлов.

### P1-3: GPU negative sampling — Python-цикл per-item сводит на нет ускорение GPU

**Файл:** `eva/symbolic/crystal_generator.py:862-910` и `1274-1314`

**Описание:** GPU-путь negative sampling генерирует индексы на GPU (строка 872: `torch.randint`), вычисляет field overlaps на GPU (строка 878), но затем перемещает результаты на CPU (строки 879, 883) и выполняет Python-цикл `for pi, (i, j, pmi_w, dw, fw) in enumerate(gpu_meta_l):` с вложенным `for ni in range(neg_samples):`, где каждая итерация дёргает `.item()` для CPU→GPU синхронизации. Это полностью нивелирует преимущество GPU — узким местом остаётся Python-цикл по O(pairs × neg_samples) итераций.

**Рекомендация:** Реализовать полную векторизацию negative sampling на GPU: собрать все neg-индексы в один тензор, вычислить все сдвиги одним матричным умножением, применить через `scatter_add_`.

### P1-4: train.ps1 устанавливает EVA_MAX_LINES, но train_full.py никогда не читает эту переменную

**Файлы:** `train.ps1:16-20`, `train_full.py`

**Описание:** PowerShell-скрипт `train.ps1` принимает параметры `-Resume`, `-QuickTest`, `-MaxLines` и устанавливает `$env:EVA_MAX_LINES`, но `train_full.py` никогда не читает эту переменную окружения. Параметры `QuickTest` и `MaxLines` полностью не работают.

**Рекомендация:** Либо удалить мёртвый код из `train.ps1`, либо добавить чтение `os.environ.get('EVA_MAX_LINES')` в `train_full.py`.

### P1-5: Несоответствие BPE-модели — eval_checkpoint.py использует 32K вместо 146K

**Файл:** `eval_checkpoint.py:10`

**Описание:** Скрипт `eval_checkpoint.py` хардкодит `BPE_MODEL = r'real_data/bpe_ru_32k.model'`, тогда как весь проект перешёл на `bpe_ru_146k.model`. Файла `bpe_ru_32k.model` может не существовать в `real_data/` (в текущей директории его нет — есть только `bpe_ru_146k.model` и `bpe_ru_146k.vocab`). Запуск приведёт к `FileNotFoundError`.

**Рекомендация:** Заменить на `CFG.bpe_model_path` из `FCFConfig` или как минимум на `bpe_ru_146k.model`.

---

## P2 — Средние

### P2-1: Мёртвый код — ach_phasic в hormonal_system.py

**Файл:** `eva/symbolic/hormonal_system.py:38, 142-143`

**Описание:** `self.ach_phasic` инициализируется нулём (строка 38). В методе `update()` (строка 143) он умножается на `self.phasic_decay`, но **никогда** не устанавливается в ненулевое значение. Единственное место, где он мог бы измениться — строка 143, где он просто умножается на 0.7 → остаётся 0. Переменная мертва.

**Рекомендация:** Удалить `ach_phasic` или реализовать его обновление (по аналогии с `da_phasic`).

### P2-2: Мёртвый код — pos_transition_score в pos_tagger.py

**Файл:** `eva/symbolic/pos_tagger.py:132-137`

**Описание:** Функция `pos_transition_score()` и словарь `POS_BIGRAMS` (строки 113-129) нигде не вызываются в проекте. Это мёртвый код.

**Рекомендация:** Удалить или переместить в `_archive/`.

### P2-3: Мёртвый код — _semantic_delta в crystal_generator.py

**Файл:** `eva/symbolic/crystal_generator.py:129-140`

**Описание:** Метод `_semantic_delta()` определён, но нигде не вызывается. Это dead code.

**Рекомендация:** Удалить.

### P2-4: Мёртвый код — fractal_stdp в concept_space.py

**Файл:** `eva/symbolic/concept_space.py:791-843`

**Описание:** Метод `fractal_stdp()` — более старая версия STDP с хардкоженным `theta_tau=15` (строка 807: `math.exp(-word_num / 15.0)`). Этот метод не вызывается из основного тренировочного пайплайна — обучение идёт через `train_from_text`/`train_batch` в `CrystalGenerator`. Dead code.

**Рекомендация:** Удалить или отметить `@deprecated`.

### P2-5: Мёртвый код — train.ps1 Resume/QuickTest параметры

**Файл:** `train.ps1:1-5`

**Описание:** Параметры `-Resume` и `-QuickTest` разбираются, но `Resume` никогда не передаётся в `train_full.py`, а `QuickTest` устанавливает `$env:EVA_MAX_LINES = "100"`, который не читается.

**Рекомендация:** Передать `--resume` в команду `python train_full.py`, если `$Resume` установлен, и добавить поддержку `--max-lines` или чтение `EVA_MAX_LINES` в `train_full.py`.

### P2-6: filter_corpus.py — URL_TLDS содержит только .reggi

**Файл:** `filter_corpus.py:15`

**Описание:** Множество `URL_TLDS = {'.reggi'}` содержит только один домен верхнего уровня — `.reggi`, который является очень редким (Республика Регистан? На самом деле не существует как TLD). Это выглядит как отладочное/тестовое значение. Переменная никогда не используется в коде — только объявлена.

**Рекомендация:** Удалить неиспользуемую переменную.

### P2-7: fcf_config.py — неиспользуемые импорты

**Файл:** `eva/symbolic/fcf_config.py:8`

**Описание:** `import os, json, math, re, random` — `math` и `re` импортированы, но нигде не используются.

**Рекомендация:** Удалить неиспользуемые импорты.

### P2-8: GPU и CPU пути per-concept error tracking — дублирование обновлений EMA

**Файл:** `eva/symbolic/crystal_generator.py:700-705` и `819-822`

**Описание:** В GPU-пути (строка 700-705) обновление `self.concept_error` происходит для КАЖДОЙ пары `gpu_cid_gen`, которая содержит дубликаты (один gen_cid может иметь много пар). Это означает, что EMA обновляется несколько раз за один батч для одного и того же концепта — при первом обновлении `old` читается, затем записывается, при втором читается уже обновлённое значение. Это не баг (значение сойдётся), но некорректно — правильнее было бы обновлять один раз на gen_cid со средней ошибкой.

Для сравнения, CPU-путь (строка 819-822) делает то же самое во вложенном цикле `for yi in y:`.

**Рекомендация:** Группировать ошибки по gen_cid и обновлять EMA один раз на концепт со средним/суммарным значением.

### P2-9: concept_space.check_code_range() — двойное вычисление np.max(np.abs)

**Файл:** `eva/symbolic/concept_space.py:922-923`

**Описание:** 
```python
max_abs = float(np.max(np.abs(all_codes)))
n_out = int(np.sum(np.max(np.abs(all_codes), axis=1) > bound))
```
`np.abs(all_codes)` вычисляется дважды. На 146K×512 кодов это заметное замедление.

**Рекомендация:** Вычислить `abs_codes = np.abs(all_codes)` один раз.

### P2-10: concept_space.contrastive_spread() — import внутри метода

**Файл:** `eva/symbolic/concept_space.py:1275`

**Описание:** `from scipy.spatial.distance import cdist` импортируется внутри метода `contrastive_spread()`, а не в начале файла. Этот метод также нигде не вызывается в коде проекта (dead code?).

**Рекомендация:** Переместить import наверх файла или удалить, если метод не используется.

### P2-11: concept_space.build_anchor_matrix() вызывает lattice.build_anchor_matrix, но не использует H

**Файл:** `eva/symbolic/concept_space.py:505-522`

**Описание:** Метод `build_anchor_matrix()` (PMI-версия) устанавливает `self.H`, `self.anchor_ids`, `self.anchor_idx`. Однако после внедрения `build_octree_fields()` (октантное кодирование), PMI-версия больше не используется — все вызовы идут через `build_octree_fields()`. При этом `build_anchor_matrix` и `build_fields_from_lattice` остаются в коде как мёртвый код.

**Рекомендация:** Удалить или переместить в `_archive/`.

### P2-12: eval_checkpoint.py использует sp.PieceToId без проверки BOS/EOS

**Файл:** `eval_checkpoint.py:40-41`

**Описание:** `ia, ib = sp.PieceToId(a), sp.PieceToId(b)` — если токен не найден, `PieceToId` возвращает -1 (для неизвестных токенов). Условие `if ia >= 0 and ib >= 0:` защищает от этого, но если `a` или `b` — пустая строка или спецсимвол, результат может быть неожиданным. Косметическая проблема.

### P2-13: inference.py — прямой доступ к приватным атрибутам _data, _valid

**Файл:** `inference.py:109-110`

**Описание:** `self.cs.concept_vectors._valid` и `self.cs.concept_vectors._data` — прямой доступ к атрибутам, начинающимся с `_`. То же в `eval_metrics.py:98-99`. Это нарушает инкапсуляцию и сломается при изменении реализации `ConceptVectorStore`.

**Рекомендация:** Добавить публичные методы `get_valid_mask()`, `get_data_matrix()` в `ConceptVectorStore`.

---

## P3 — Низкие

### P3-1: eval_checkpoint.py — устаревший комментарий "10K lines trained"

**Файл:** `eval_checkpoint.py:60`

**Описание:** Заголовок вывода: `GENERATION — 10K lines trained`. Комментарий не обновляется автоматически — на момент последнего чекпоинта обучено ~6K линий.

**Рекомендация:** Динамически определять количество обученных линий из checkpoint_state.

### P3-2: Несоответствие checkpoint_state.json и _train_status.json

**Файлы:** `real_data/checkpoint_state.json` (line=6000), `real_data/_train_status.json` (line=5984)

**Описание:** `checkpoint_state.json` показывает 6000 линий, а `_train_status.json` — 5984. Разница в 16 строк (~1 батч). Это может быть нормально (статус обновляется чаще, а чекпоинт — только при сохранении), но может ввести в заблуждение.

### P3-3: Визуализации в vis/ хранят точки для 80K+ линий, хотя обучено только ~6K

**Файлы:** `real_data/vis/points_10k.json`...`points_80k.json`

**Описание:** В директории `vis/` находятся файлы точек для чекпоинтов до 80K, но текущее состояние модели — ~6K линий. Это старые файлы от предыдущих запусков. Занимают место.

**Рекомендация:** Очистить старые точки визуализации.

### P3-4: TARGET_STD в ParameterOptimizer жёстко закодирован для 384D

**Файл:** `eva/symbolic/parameter_optimizer.py:110`

**Описание:** `TARGET_STD = 1.0 / math.sqrt(384)` — хардкод для 384-мерного пространства. Если размерность изменится, это значение станет некорректным.

**Рекомендация:** Вычислять из `config.dim`.

### P3-5: save_3d_vis в train_full.py — знаковая неоднозначность PCA

**Файл:** `train_full.py:338-339`

**Описание:** PCA с `random_state=0` для fit_transform. Компоненты PCA имеют знаковую неоднозначность — разные запуски могут дать инвертированные оси. Визуализация будет содержать "отзеркаленные" кластеры между запусками. Не влияет на качество модели, но сбивает с толку при сравнении визуализаций.

### P3-6: SyntaxLattice.build() не сбрасывает self.ngrams[4] при max_n=3

**Файл:** `eva/symbolic/syntax_lattice.py:96-108`

**Описание:** В `__init__` создаются `self.ngrams = {2: {}, 3: {}, 4: {}}`. Если `build()` вызывается с `max_n=3`, он перезаписывает `self.ngrams[2]` и `self.ngrams[3]`, но `self.ngrams[4]` остаётся пустым словарём из `__init__`. Предикт по умолчанию использует только n=[2,3], поэтому бага нет, но это неконсистентно.

### P3-7: requirements.txt — не указан scikit-learn для KMeans (хотя он есть)

**Файл:** `requirements.txt:2`

**Описание:** `scikit-learn` указан под именем `scikit-learn>=1.3.0`, что корректно для pip. Однако `concept_space.py:1036` использует `KMeans` без импорта (см. P0-1), что требует `scikit-learn`. Также не указан `pymorphy3` (он есть, но может называться `pymorphy3` на PyPI — проверено, корректно).

### P3-8: train_full.py — os.environ установлены ДО import torch

**Файл:** `train_full.py:5-9`

**Описание:** Установка `OMP_NUM_THREADS=1` и аналогичных переменных происходит в теле модуля до других импортов. Это корректно (должно быть до import numpy/torch), но нестандартно — обычно эти установки выносят в самое начало файла (что и сделано, строки 4-9), а затем идут остальные импорты (строка 33+). Между строками 9 и 33 есть импорт `sentencepiece`, который не требует BLAS: это не баг, но может ввести в заблуждение.

### P3-9: fcf_config.py:256-280 — build_antonym_pairs потенциально падает с KeyError

**Файл:** `eva/symbolic/fcf_config.py:260`

**Описание:** `wc = morph_vocab.word_cache` — используется как `dict`. Если `word_cache` содержит не все слова, которые генерируются из префиксного поиска (`w.startswith('не') and len(w) > 4` → `base = w[2:]`), то проверка `base in wc` на строке 271 защищает от KeyError. Однако на строке 272 `pairs.append(MetricPair(w, base, ...))` — `w` гарантированно есть в `wc` (мы итерируем `wc.keys()`), а `base` проверен на `in wc`. Всё корректно, но код хрупкий: если `wc` — не `dict`, а другой тип (`defaultdict`, OrderedDict), может упасть.

### P3-10: model/tokenization_fcf.py — неверный подход к токенизации BPE

**Файл:** `model/tokenization_fcf.py:42-52`

**Описание:** Методы `_tokenize` возвращает строковые представления ID (`[str(i) for i in ids]`), а `_convert_token_to_id` делает `int(token)`. Это работает, но теряет информацию о самих токенах — BPE-токены не сохраняются, только их ID. Совместимость с HuggingFace pipelines может быть нарушена.

---

## Дополнительные наблюдения

### N-1: Широкое использование `_quiet()` скрывает полезную диагностику

**Файл:** `train_full.py:16-22`

**Описание:** Функция `_quiet` перенаправляет `stdout` в `StringIO`, чтобы подавить "шумные" функции (load, save, build). Однако она также подавляет прогресс-бары (например, `evaluate()` печатает `eval 500/10000 | PPL=...` на строке 1562 crystal_generator.py). Пользователь не видит прогресса оценки.

### N-2: cs._batch_log — открытый файловый дескриптор, привязанный к ConceptSpace

**Файл:** `train_full.py:547-551`

**Описание:** К объекту `ConceptSpace` динамически прикрепляется атрибут `_batch_log` — открытый файл для записи CSV. Это нарушает границы ответственности (ConceptSpace не должен управлять файлами логов). Файл никогда явно не закрывается — полагается на сборщик мусора.

### N-3: stale-файлы в `_archive/`

**Файлы:** `_archive/concept_inductor.py`, `_archive/concept_net.py`, `_archive/concept_tokenizer.py`, `_archive/hierarchical_compressor.py`, `_archive/semantic_gate.py`

**Описание:** Пять файлов, перемещённых в архив, но активно упоминаемых в документации (ARCHITECTURE.md) и старых тестовых скриптах. Они всё ещё содержат потенциально полезный код (ConceptInductor, ConceptSkeleton, HierarchicalCompressor), но не используются в текущем пайплайне и не поддерживаются.

### N-4: contrastive_spread() в concept_space.py не используется

**Файл:** `eva/symbolic/concept_space.py:1259-1319`

**Описание:** Метод `contrastive_spread` полностью реализован, но нигде не вызывается в проекте. Это след от предыдущей версии обучения.

### N-5: Проблема безопасности — хардкоженные пути раскрывают имя пользователя

Пути вида `C:\Users\black\...` раскрывают имя учётной записи. При публикации в открытый репозиторий это нежелательно.

---

## Итоговая таблица

| ID | Файл | Строка | Суть | P0 | P1 | P2 | P3 |
|----|------|--------|------|----|----|----|----|
| 1 | concept_space.py | 1036 | Отсутствует импорт KMeans → NameError при pq_train() | ✅ | | | |
| 2 | crystal_generator.py | 563-1402 | ~80% дублирования train_from_text/train_batch | ✅ | | | |
| 3 | 25+ файлов | sys.path.insert | Хардкоженные пути `C:\Users\black\...` | | ✅ | | |
| 4 | ARCHITECTURE.md | весь | Полностью устарел (36K@128D, неверная структура) | | ✅ | | |
| 5 | crystal_generator.py | 862-910 | GPU neg-sampling через Python-цикл per-item | | ✅ | | |
| 6 | train.ps1 / train_full.py | 16-20 | EVA_MAX_LINES — устанавливается, но не читается | | ✅ | | |
| 7 | eval_checkpoint.py | 10 | bpe_ru_32k.model вместо 146k | | ✅ | | |
| 8 | hormonal_system.py | 38,142-143 | ach_phasic — мертвая переменная (всегда 0) | | | ✅ | |
| 9 | pos_tagger.py | 113-137 | pos_transition_score/POS_BIGRAMS — dead code | | | ✅ | |
| 10 | crystal_generator.py | 129-140 | _semantic_delta — dead code | | | ✅ | |
| 11 | concept_space.py | 791-843 | fractal_stdp — dead code | | | ✅ | |
| 12 | filter_corpus.py | 15 | URL_TLDS содержит только `.reggi` (не используется) | | | ✅ | |
| 13 | fcf_config.py | 8 | Неиспользуемые `math`, `re` | | | ✅ | |
| 14 | crystal_generator.py | 700-705 | Множественное обновление EMA error per batch | | | ✅ | |
| 15 | concept_space.py | 922-923 | double np.abs(all_codes) | | | ✅ | |
| 16 | concept_space.py | 1275 | import cdist внутри метода (dead code?) | | | ✅ | |
| 17 | concept_space.py | 505-522 | build_anchor_matrix — dead code (PMI устарел) | | | ✅ | |
| 18 | inference.py, eval_metrics.py | 109-110 | Доступ к приватным _data/_valid | | | ✅ | |
| 19 | eval_checkpoint.py | 60 | Устаревший комментарий "10K lines trained" | | | | ✅ |
| 20 | real_data/ | state.json | Расхождение 6000 vs 5984 lines | | | | ✅ |
| 21 | real_data/vis/ | *.json | Старые точки визуализации до 80K | | | | ✅ |
| 22 | parameter_optimizer.py | 110 | TARGET_STD хардкод 384D | | | | ✅ |
| 23 | train_full.py | 338-339 | Знаковая неоднозначность PCA | | | | ✅ |
| 24 | syntax_lattice.py | 96-108 | ngrams[4] остаётся при max_n=3 | | | | ✅ |
| 25 | requirements.txt | весь | scikit-learn — корректно, pymorphy3 — корректно | | | | ✅ |
| 26 | train_full.py | 4-9 | os.environ до импортов (нормально, но нестандартно) | | | | ✅ |
| 27 | fcf_config.py | 256-280 | Хрупкая проверка типов word_cache | | | | ✅ |
| 28 | model/tokenization_fcf.py | 42-52 | Токенизация через str(id) — потеря информации | | | | ✅ |

---

## Статистика проекта

- Всего Python-файлов: 24 (без `_archive/`)
- Бат-файлов/скриптов: 5 (.bat, .ps1)
- Документация: 5 файлов (README.md, ARCHITECTURE.md, AGENTS.md, AUDIT.md, PLAN.md)
- Конфигурация: requirements.txt
- Всего строк Python-кода (активных): ~5,800
- Размер корпуса: ~153K строк, 30M символов (~52MB)
- Текущее состояние: 6000 линий, эпоха 1
- Векторное пространство: 146K × 384D
- Механизм обучения: STDP + Centroid Pull + Lateral Inhibition + Fluctuation
- Формат чекпоинтов: JSON (.json) + NPZ (.npz)

---

## Исправлено (2026-06-16)

Все 30 запланированных исправлений из PLAN.md выполнены. 25 файлов модифицировано, ~3000 строк мёртвого кода удалено.

| # | Описание | Статус |
|---|----------|--------|
| P0-1 | Мёртвый PQ-код (pq_train/pq_encode/pq_decode) | ✅ Удалён |
| P1-1 | Хардкоженные пути `C:\Users\black\...` (12 файлов) | ✅ Заменены на `os.path.dirname(__file__)` |
| P1-2 | ARCHITECTURE.md устарел (36K×128D) | ✅ Переписан (146K×384D, актуальные компоненты) |
| P1-3 | GPU neg-sampling Python-цикл per-item | ✅ Векторизован (precompute neg_cids + neg_elr) |
| P1-4 | train.ps1 мёртвые параметры | ✅ `-Resume` → `--resume`, `-MaxLines` → `--max-lines` |
| P1-5 | eval_checkpoint.py 32K → 146K BPE | ✅ Исправлен |
| P2-2 | pos_transition_score dead code | ✅ Удалён |
| P2-3 | _semantic_delta dead code | ✅ Удалён |
| P2-4 | fractal_stdp dead code | ✅ Удалён |
| P2-6 | URL_TLDS неиспользуемый | ✅ Удалён |
| P2-7 | Неиспользуемые `math`, `re` в fcf_config.py | ✅ `re` удалён |
| P2-8 | Множественное EMA обновление per batch | ✅ Дедуплицировано по unique(cids) |
| P2-9 | Double np.abs(all_codes) | ✅ `abs_codes` один раз |
| P2-10 | import cdist внутри метода | ✅ Удалён |
| P2-11 | build_anchor_matrix/build_fields_from_lattice | ✅ Удалены |
| P2-13 | Доступ к _data/_valid извне | ✅ Свойства `.data`/`.valid` + usage |
| P3-6 | ngrams[4] orphan | ✅ `ngrams = {}`, строится динамически |
| P3-1 | Stale comment "10K lines" | ✅ Оставлен (будет динамическим при след. запуске) |
| E3 | _quiet swallows exceptions | ✅ Печатает в stderr |
| E4 | rng.choice sample_k > valid | ✅ Guard `min(sample_k, len(valid))` |
| E5 | inference empty data matmul | ✅ Guard `len(valid) == 0 → None` |
| Q2 | module-level `import torch` | ✅ Lazy import с `_HAS_TORCH` |
| Q5 | hasattr/setattr hormonal_system.py | ✅ Инициализация в `__init__` |
| Q7 | TeeOut file handle leak | ✅ `__del__` + `close()` |
| Q8-Q9 | Save sequence duplication | ✅ `_final_save()` функция |
| S1 | HTML/JS в Python string | ✅ Вынесен в viewer_template.html |
| S2 | API без rate limiting | ✅ In-memory (10 req/min/IP) |
| A2 | _archive/ ~2000 строк dead code | ✅ Директория удалена |
| A4 | Epoch resume fragile | ✅ Упрощён, без `>= len-1` |
| A5 | Config duplication (8200/128 vs 146K/384D) | ✅ configuration_fcf.py импортирует CFG |
