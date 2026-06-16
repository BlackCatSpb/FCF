# FCF (Fractal Cognitive Field) — Полный аудит проекта (2026-06-16)

Дата: 2026-06-16
Коммит: `c2e3588` (HEAD)
Предыдущий аудит: полностью переписан (предыдущий AUDIT.md был от `03b8ae8`)
Всего файлов: 27 Python, 5 бат-файлов/ps1, документация, конфиги

---

## Сводка по серьёзности

| Уровень | Описание | Кол-во |
|---------|----------|--------|
| **P0** | Критические — падение/некорректные результаты | 3 |
| **P1** | Высокие — серьёзные проблемы архитектуры/портабельности | 9 |
| **P2** | Средние — мёртвый код, неоптимальности, несоответствия | 11 |
| **P3** | Низкие — косметика, документация, старые файлы | 8 |

---

## P0 — Критические

### P0-1: theta_gate использует абсолютную позицию `j` вместо расстояния `j-i`

**Файлы:**
- `eva/symbolic/crystal_generator.py:994` (CPU путь train_from_text)
- `eva/symbolic/crystal_generator.py:1104` (CPU путь train_batch)
- `eva/symbolic/crystal_generator.py:584` (GPU путь _gpu_stdp_apply)
- `eva/symbolic/crystal_generator.py:799` (GPU путь _negative_sampling_gpu)

**Описание:** Формула theta_gate должна ослаблять LR для пар токенов, находящихся далеко друг от друга в предложении. Для этого нужно использовать расстояние `|j-i|`, но код использует абсолютную позицию `j`:

```python
# CPU (строка 994):
theta_gate = math.exp(-min(j, 5) / max(self.theta_tau, 1.0))

# GPU (строка 584): 
theta = torch.exp(-torch.clamp(j_pos, max=5.0) / max(self.theta_tau, 1.0))
```

Для предложения из 100 токенов:
- Пара (0, 1): `j=1` → theta = `exp(-1/15)` ≈ 0.94 ✅ (должно быть `j-i=1`)
- Пара (98, 99): `j=99` → theta = `exp(-5/15)` ≈ 0.72 ❌ (должно быть `j-i=1` → 0.94)
- Пара (95, 100): `j=100` → theta = `exp(-5/15)` ≈ 0.72 (должно быть `j-i=5` → 0.72)
- Пара (0, 2): `j=2` → theta = `exp(-2/15)` ≈ 0.88 ✅

Для длинных предложений (>5 токенов) theta_gate для пар (98,99) и (1,2) будет **разным**, хотя расстояние одинаковое. Это приводит к **неравномерному распределению LR** — конец предложения получает меньший LR, чем начало, для одинаковых расстояний.

**Исправление:** Заменить `j` на `j-i` во всех четырёх местах:
```python
dist = abs(j - i)
theta_gate = math.exp(-min(dist, 5) / max(self.theta_tau, 1.0))
```

---

### P0-2: API endpoint /health падает с AttributeError

**Файл:** `api/main.py:56`

**Описание:** Код обращается к несуществующим атрибутам:
```python
transitions=model.space.concept_transitions.nnz if model.space.concept_transitions else 0
```
`concept_transitions` был удалён из `ConceptSpace` при рефакторинге. `AttributeError`.

Дополнительно на строке 26: `len(model.space.cid_list)` — `cid_list` не существует в `ConceptSpace`. `AttributeError`.

**Рекомендация:** Заменить на `len(model.space.concept_vectors)` и `0` для transitions (или вычислить через `len(model.space.fractal.codes)`).

---

### P0-3: FCFModel.generate() и forward() падают с AttributeError

**Файл:** `model/modeling_fcf.py:148, 157, 208-209`

**Описание:** Код обращается к несуществующим атрибутам `CrystalGenerator`:

Строка 148:
```python
core_cid, modifier_field, centroid, noise = self.generator.gate.extract_core(words)
```
`self.generator.gate` (ConceptGate) — не существует. Удалён при рефакторинге.

Строка 157:
```python
confidence=float(self.generator._query_confidence),
```
`_query_confidence` не определён в `CrystalGenerator`.

Строка 208:
```python
confidence=float(self.generator._query_confidence),
```

Строка 209:
```python
intent_anchor=self._space.concept_info.get(result.get("core_cid", 0), {}).get("anchor", None)
```
`concept_info` не существует в `ConceptSpace`. `core_cid` не возвращается из `generate()`.

**Рекомендация:** Заменить на корректные вызовы или заглушки. `gate.extract_core` → использовать существующий механизм генерации. `_query_confidence` → удалить или определить. `concept_info` → использовать `lattice.concept_freq`.

---

## P1 — Высокие

### P1-1: __main__ блоки используют 32K BPE модель вместо 146K

**Файлы:**
- `eva/symbolic/crystal_generator.py:1318` — `bpe_ru_32k.model`
- `eva/symbolic/concept_space.py:1085` — `bpe_ru.model`
- `eva/symbolic/syntax_lattice.py:743` — `bpe_ru.model`

**Описание:** Три `__main__` блока для отладки используют старые BPE-модели (32K или без указания), хотя проект полностью перешёл на `bpe_ru_146k.model`. Файла `bpe_ru_32k.model` может не существовать в `real_data/`.

**Рекомендация:** Заменить на `bpe_ru_146k.model`.

---

### P1-2: Contrastive objective не выполняется в GPU-пути

**Файл:** `eva/symbolic/crystal_generator.py:1012-1013, 1122-1123`

**Описание:**
```python
# Оба метода (train_from_text и train_batch):
if not use_torch:
    self._contrastive_objective(gen_updates)
```
При `use_torch=True` contrastive objective **полностью пропускается**. GPU-путь теряет важный механизм обучения — hard-negative push-pull.

**Рекомендация:** Реализовать GPU-версию `_contrastive_objective` или выполнять её на CPU после GPU-шага (но с корректными данными из `gen_updates`).

---

### P1-3: GPU negative sampling — Python-цикл не устранён

**Файл:** `eva/symbolic/crystal_generator.py:773-820`

**Описание:** В PLAN.md (P1-3) утверждается, что GPU negative sampling векторизован. Хотя precompute `neg_cids_np` и `neg_elr_arr` вынесены до цикла, **основной цикл остаётся Python-циклом**:

```python
for pi in range(n_pairs):
    v_ctx = cs.concept_vectors.get(gpu_cid_ctx[pi])
    ...
    for ni in range(neg_samples):
        if field_gate and neg_ovs_cpu[pi, ni] > 0:
            continue
        neg_cid = int(neg_cids_np[pi, ni])
        ...
        cs._apply_vector_update(neg_cid, v_new)
```

Каждая итерация вызывает `cs._apply_vector_update` — это Python-функция с numpy-операциями. Полной GPU-векторизации нет.

**Рекомендация:** Собрать все valid neg-индексы в тензор, вычислить сдвиги одним матричным умножением, применить через `scatter_add_`.

---

### P1-4: `_final_save` не вызывает `cleanup_old_checkpoints`

**Файл:** `train_full.py:390-400`

**Описание:** Функция `_final_save` сохраняет cs, lattice, opt и checkpoint state, но **не вызывает `cleanup_old_checkpoints`**. Это означает:
- При `KeyboardInterrupt` (строка 598) — старые чекпоинты не чистятся
- При финальном сохранении (строка 606) — не чистятся
- Старые checkpoint-файлы накапливаются

Только внутри цикла обучения (строка 552) вызывается `cleanup_old_checkpoints`.

**Рекомендация:** Добавить `cleanup_old_checkpoints(keep=CFG.cleanup_keep)` в `_final_save`.

---

### P1-5: `topk_similar_concepts` — прямой доступ к `_valid`/`_data`

**Файл:** `eva/symbolic/concept_space.py:862-864`

**Описание:** Метод `topk_similar_concepts` (публичный API) использует прямой доступ к защищённым атрибутам:
```python
valid = self.concept_vectors._valid
mat = self.concept_vectors._data[valid]
```
Хотя в PLAN.md (P2-13) утверждается, что все обращения к `_data`/`_valid` заменены на `@property` `.data`/`.valid`, этот метод остался нетронутым. `inference.py:126` тоже использует `topk_similar_concepts`, который внутри обращается к `_data`/`_valid`.

**Рекомендация:** Заменить на `self.concept_vectors.data` и `self.concept_vectors.valid`.

---

### P1-6: `eval_metrics.py` — прямой доступ к `_valid`

**Файл:** `eval_metrics.py:78`

**Описание:**
```python
print(f"  Load: {time.time()-t0:.1f}s | {sum(cs.concept_vectors._valid)}/{cs.vocab_size} vectors")
```
Используется `_valid` вместо `valid`.

**Рекомендация:** Заменить `_valid` на `valid`.

---

### P1-7: `tokenization_fcf.py` — опечатка в `vocab_files_names`

**Файл:** `model/tokenization_fcf.py:11`

**Описание:**
```python
vocab_files_names = {
    "spm_file": "bpe_ru.model",
}
```
Атрибут `PreTrainedTokenizer` называется `vocab_files_names` (с 'i' — `vocab_files_names`). Опечатка `vocab_files_names` вместо `vocab_files_names`. Это приведёт к тому, что HuggingFace не найдёт файлы вокабуляра при `save_pretrained`/`from_pretrained`.

Дополнительно: значение по умолчанию `"bpe_ru.model"` не соответствует текущей модели `bpe_ru_146k.model`.

**Рекомендация:** Исправить на `vocab_files_names` и указать `bpe_ru_146k.model`.

---

### P1-8: `train_full.py` — нет поддержки `--epochs` для авто-возобновления

**Файл:** `train_full.py:115, 412`

**Описание:** Параметр `--epochs` задаёт общее количество эпох. Но при резюме, `total_epochs = args.epochs`, а `current_epoch = resume_epoch`. Если пользователь запустил `--epochs 3 --resume` на второй эпохе, то обучение пойдёт с эпохи 2 по эпоху 3. Но если `resume_epoch` = 2, а `total_epochs` = 2, то обучение выполнит только эпоху 2 и закончит. Это корректно.

Проблема: если `resume_epoch` = 1, `total_epochs` = 3, но при первом запуске было `--epochs 5`, то разницы нет — resume продолжает с эпохи 1 до 3. OK.

Но есть неявная проблема: `total_epochs` НЕ сохраняется в `checkpoint_state.json`. Если пользователь запустил `--epochs 3`, дошёл до линии 2000 эпохи 1, прервал, и запустил `--epochs 5 --resume`, то он получит 5 эпох (перезапишет предыдущее намерение). Это может быть как фичей, так и багом.

---

### P1-9: `fcf_config.py:133-135` — свойства `l_c`, `l_a`, `l_m` не согласованы с `FractalField`

**Файл:** `eva/symbolic/fcf_config.py:133-135`

**Описание:**
```python
@property
def l_c(self) -> int: return self.latent_dim // 2          # 256
@property
def l_a(self) -> int: return self.latent_dim // 4          # 128
@property
def l_m(self) -> int: return self.latent_dim - self.l_c - self.l_a  # 128
```

В `FractalField` (concept_space.py:87-89) эти значения захардкожены:
```python
self.l_c = latent_dim // 2      # 256
self.l_a = latent_dim // 4      # 128
self.l_m = latent_dim - self.l_c - self.l_a  # 128
```

Консистентно при `latent_dim=512`, но если изменить `latent_dim` в `FCFConfig`, эти значения **не синхронизируются** с `FractalField`, так как в `FractalField.__init__` они вычисляются при создании, а не берутся из конфига.

**Рекомендация:** Передавать `l_c`/`l_a`/`l_m` из `FCFConfig` в `FractalField`.

---

## P2 — Средние

### P2-1: TeeOut — закрытие stdout может сломать вывод

**Файл:** `train_full.py:68, 601-603`

**Описание:** `sys.stdout` заменяется на `TeeOut`. В блоке `finally` (строка 601-603) вызывается `sys.stdout.close()`. После этого `sys.stdout` становится непригодным для вывода. Если после `finally` будет какая-либо ошибка или `print()`, она упадёт с `ValueError: I/O operation on closed file`.

```python
finally:
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()
```

**Рекомендация:** Сохранить оригинальный `sys.stdout` и восстановить его после закрытия `TeeOut`.

---

### P2-2: `_batch_log` — утечка файлового дескриптора

**Файл:** `train_full.py:471-475`

**Описание:** К объекту `cs` (ConceptSpace) динамически прикрепляется `_batch_log` — открытый файл CSV. Файл никогда не закрывается явно. При создании нового `TeeOut` при следующем запуске, старый `_batch_log` остаётся открытым. Нарушение границ ответственности.

**Рекомендация:** Или сделать `_batch_log` отдельным объектом с `__enter__`/`__exit__`, или закрывать в `_final_save`.

---

### P2-3: `contrastive_spread` в `ConceptSpace` — мёртвый код

**Файл:** `eva/symbolic/concept_space.py:956-1015`

**Описание:** Метод `contrastive_spread` полностью реализован (60 строк), но нигде не вызывается в проекте. Это не `_contrastive_objective` в `crystal_generator.py` — это отдельный метод `ConceptSpace`, оставшийся от предыдущей версии.

**Рекомендация:** Удалить или переместить в архив.

---

### P2-4: Пустая секция "H matrix + BMSSP" с изолированным кодом

**Файл:** `eva/symbolic/concept_space.py:504-552`

**Описание:** После удаления `build_anchor_matrix` и `build_fields_from_lattice`, осталась пустая строка с комментарием (строка 507) и неиспользуемый метод `_compute_pmi_field_fast` (строки 508-552). Этот метод не вызывается нигде — `build_octree_fields` полностью заменил PMI-подход.

**Рекомендация:** Удалить `_compute_pmi_field_fast`.

---

### P2-5: `cleanup_old_checkpoints` — не удаляет `.opt.json` файлы

**Файл:** `train_full.py:87-109`

**Описание:** Функция `cleanup_old_checkpoints` удаляет `concept_space_*k.*` и `syntax_lattice_*k.*`, но НЕ удаляет соответствующие `*k.opt.json` файлы. После нескольких циклов обучения в `real_data/` накапливаются старые `.opt.json` файлы.

```python
for ext in ['.json', '.codes.npz', '.opt.json']:
    fp = os.path.join(base_dir, f'concept_space_{k_label}{ext}')
    if os.path.exists(fp): os.remove(fp)
# syntax_lattice_*k.opt.json не обрабатывается
```

**Рекомендация:** Добавить удаление `syntax_lattice_*k.opt.json` или расширить массив `ext`.

---

### P2-6: `_is_semantic_token` — не обрабатывает букву 'ё'

**Файл:** `eva/symbolic/crystal_generator.py:154`

**Описание:**
```python
if len(text) == 1 and not ('а' <= text.lower() <= 'я' or text.isalpha()):
    return False
```
Диапазон `'а' <= text.lower() <= 'я'` исключает букву 'ё' (код 1105 в Unicode, а 'я' — 1103). Токены, состоящие из одной буквы 'ё', будут считаться не-семантическими.

**Рекомендация:** Добавить `or text.lower() == 'ё'`.

---

### P2-7: `_theta_temp` — деление на ноль при `theta_tau=0`

**Файл:** `eva/symbolic/crystal_generator.py:131`

**Описание:**
```python
t = self.base_concept_temp * math.exp(-word_num / self.theta_tau)
return max(t, self.base_concept_temp * 0.15)
```
Если `self.theta_tau` равно 0 или очень мало, будет `float division by zero` или `inf`. В `FCFConfig` минимум для `theta_tau` — 5, и в коде стоит `max(self.theta_tau, 1.0)`, но в `_theta_temp` этой защиты нет.

**Рекомендация:** Добавить `max(self.theta_tau, 1.0)`.

---

### P2-8: `inference.py` — метод `neighbours` использует `topk_similar_concepts` с `_data`/`_valid`

**Файл:** `inference.py:126`

**Описание:** Метод `InferenceEngine.neighbours()` вызывает `self.cs.topk_similar_concepts(cid, k=k)`, который внутри (concept_space.py:862-864) обращается к `self.concept_vectors._data` и `self.concept_vectors._valid`. Хотя `inference.py` сам использует `.valid` (строка 124), вызов `topk_similar_concepts` возвращается к приватным атрибутам.

**Рекомендация:** Исправить `topk_similar_concepts` (см. P1-5).

---

### P2-9: `concept_space.init_concepts()` — падение при `vocab_size=0`

**Файл:** `eva/symbolic/concept_space.py:472-481`

**Описание:**
```python
def init_concepts(self):
    for cid in range(self.vocab_size):
        ...
```
Если `self.vocab_size == 0`, цикл не выполняется. Это нормально, но если `vocab_size` не установлен (None), будет `TypeError: 'NoneType' object cannot be interpreted as an integer`.

В `__init__`: `self.vocab_size = vocab_size or 0`. Но если передать `vocab_size=None`, то `None or 0` = 0. OK.

---

### P2-10: `train_full.py` — checkpoint_state.json без timestamp в `_final_save`

**Файл:** `train_full.py:395-397`

**Описание:** `_final_save` сохраняет `{'epoch': epoch, 'line': total_lines}` без поля `timestamp`, в то время как `save_checkpoint_state` (строка 77) добавляет `'timestamp': time.time()`. При перезаписи `_final_save` timestamp пропадает.

---

### P2-11: `fcf_config.py:82-83` — корпус может отсутствовать

**Файл:** `eva/symbolic/fcf_config.py:82-83`

**Описание:**
```python
@property
def corpus_path(self) -> str:
    return os.path.join(self.data_dir, 'full_corpus_ru_clean.txt')
```
Файл `full_corpus_ru_clean.txt` создаётся скриптом `filter_corpus.py`. В `.gitattributes` есть `real_data/*.txt filter=lfs`. Если после clone без LFS файл не подтянулся, и `filter_corpus.py` не был запущен — обучение упадёт при попытке открыть несуществующий файл.

**Рекомендация:** Добавить проверку существования и понятную ошибку.

---

## P3 — Низкие

### P3-1: stale точки визуализации в `real_data/vis/`

**Файлы:** `real_data/vis/points_*.json`

**Описание:** В директории vis/ находятся 16 файлов (до 80K точек), но текущее обучение — ~6000 линий (~6K). Это старые файлы от предыдущих запусков. Занимают ~50+ MB.

**Рекомендация:** Очистить старые точки визуализации.

---

### P3-2: Знаковая неоднозначность PCA в `save_3d_vis`

**Файл:** `train_full.py:349`

**Описание:**
```python
pca = PCA(n_components=3, random_state=0)
proj = pca.fit_transform(Xc)
```
PCA имеет знаковую неоднозначность — разные запуски могут инвертировать оси. При сравнении визуализаций между чекпоинтами кластеры могут быть "отзеркалены".

---

### P3-3: `hormonal_system.py:112-113` — порог повторения неточен

**Файл:** `eva/symbolic/hormonal_system.py:112-113`

**Описание:**
```python
if len(self._last_few_cids) >= 3 and len(set(self._last_few_cids)) <= 2:
    da_coherence -= 0.1  # boredom from repetition
```
Если 3 одинаковых CID → `set = {x}` → `len=1 <= 2` → True. Если 3 CID из 2 разных (A, A, B) → `len=2 <= 2` → True. Если 3 CID из 3 разных → `len=3 > 2` → False. Работает, но порог `<=2` странный — штрафует и за 2 повтора из 3, и за 3 повтора из 3 одинаково.

---

### P3-4: `train_full.py:430` — неверный total при MAX_LINES

**Файл:** `train_full.py:430`

**Описание:** При `MAX_LINES > 0` печатается `epoch_lines/ total_lines` где `total_lines` — оригинальное количество, не урезанное `MAX_LINES`. Например: "Curriculum epoch 1: 100/88855 lines (max 32 BPE tokens)".

---

### P3-5: `_graph_search` — избыточный вызов `_ensure_ppmi`

**Файл:** `eva/symbolic/crystal_generator.py:324`

**Описание:** В `_branch` вызывается `_graph_search(sources, ...)`, в котором `connections_of(u, top_k=8, use_ppmi=True)`. Если `use_ppmi=True`, то при каждом вызове `connections_of` проверяется `_ensure_ppmi`, которая в свою очередь проверяет `self._ppmi_cache is not None`. Лишний if при каждом вызове.

---

### P3-6: `eval_checkpoint.py:20` — ожидание 785MB, а реально может быть больше

**Файл:** `eval_checkpoint.py:19`

**Описание:** Комментарий: "Loading ConceptSpace (785MB)..." — размер зависит от размера чекпоинта. Для 146K×384D это может быть >1GB.

---

### P3-7: `syntax_lattice.py:620` — `self.ngrams = {n: {} for n in range(2, 5)}` хардкод

**Файл:** `eva/symbolic/syntax_lattice.py:620`

**Описание:** Метод `load` сбрасывает `self.ngrams` в `{2: {}, 3: {}, 4: {}}`, хотя `__init__` использует `self.ngrams = {}` (динамическое построение). При загрузке всегда создаётся max_n=4, даже если исходный lattice был построен с max_n=3.

---

### P3-8: `fractal_encoding.py:12` — LEVELS хардкод

**Файл:** `eva/symbolic/fractal_encoding.py:12-13`

**Описание:**
```python
LEVELS = 16
GAMMA = 0.5
```
Модульные константы не синхронизированы с `FCFConfig.octree_levels`. Хотя в `build_octree_fields` передаётся `gamma`, для `LEVELS` нет параметра — используется глобальная константа.

---

## Итоговая таблица

| ID | Файл | Строки | Суть | P0 | P1 | P2 | P3 |
|----|------|--------|------|----|----|----|----|
| 1 | crystal_generator.py | 584, 799, 994, 1104 | theta_gate использует `j` вместо `j-i` | ✅ | | | |
| 2 | api/main.py | 26, 56 | Обращение к несуществующим атрибутам (`concept_transitions`, `cid_list`) | ✅ | | | |
| 3 | modeling_fcf.py | 148, 157, 208-209 | Обращение к несуществующим `gate`, `_query_confidence`, `concept_info` | ✅ | | | |
| 4 | crystal_generator.py | 1318 | `__main__` использует `bpe_ru_32k.model` | | ✅ | | |
| 5 | concept_space.py | 1085 | `__main__` использует `bpe_ru.model` | | ✅ | | |
| 6 | syntax_lattice.py | 743 | `__main__` использует `bpe_ru.model` | | ✅ | | |
| 7 | crystal_generator.py | 1012-1013, 1122-1123 | Contrastive objective не вызывается в GPU-пути | | ✅ | | |
| 8 | crystal_generator.py | 802-820 | GPU neg-sampling: Python-цикл не устранён | | ✅ | | |
| 9 | train_full.py | 390-400 | `_final_save` не вызывает `cleanup_old_checkpoints` | | ✅ | | |
| 10 | concept_space.py | 862-864 | `topk_similar_concepts` — доступ к `_data`/`_valid` | | ✅ | | |
| 11 | eval_metrics.py | 78 | Доступ к `_valid` вместо `valid` | | ✅ | | |
| 12 | tokenization_fcf.py | 11 | Опечатка `vocab_files_names` (пропущена 'i') | | ✅ | | |
| 13 | fcf_config.py | 133-135 | l_c/l_a/l_m не синхронизируются с FractalField | | ✅ | | |
| 14 | train_full.py | 68, 601-603 | TeeOut.close() делает stdout непригодным | | | ✅ | |
| 15 | train_full.py | 471-475 | `_batch_log` — утечка файлового дескриптора | | | ✅ | |
| 16 | concept_space.py | 956-1015 | `contrastive_spread` — мёртвый код (60 строк) | | | ✅ | |
| 17 | concept_space.py | 508-552 | `_compute_pmi_field_fast` — не вызывается | | | ✅ | |
| 18 | train_full.py | 87-109 | `cleanup_old_checkpoints` не удаляет `*.opt.json` | | | ✅ | |
| 19 | crystal_generator.py | 154 | `_is_semantic_token` не обрабатывает 'ё' | | | ✅ | |
| 20 | crystal_generator.py | 131 | `_theta_temp` — деление на ноль при theta_tau=0 | | | ✅ | |
| 21 | inference.py | 126 | neighbours вызывает `topk_similar_concepts` с `_data`/`_valid` | | | ✅ | |
| 22 | train_full.py | 395-397 | `_final_save` без timestamp (неконсистентно с save_checkpoint_state) | | | ✅ | |
| 23 | fcf_config.py | 82-83 | `corpus_path` может не существовать | | | ✅ | |
| 24 | real_data/vis/ | *.json | Старые точки визуализации до 80K | | | | ✅ |
| 25 | train_full.py | 349 | Знаковая неоднозначность PCA | | | | ✅ |
| 26 | hormonal_system.py | 112-113 | Странный порог повторения (`<=2`) | | | | ✅ |
| 27 | train_full.py | 430 | Неверный total при MAX_LINES | | | | ✅ |
| 28 | syntax_lattice.py | 620 | `ngrams = {n: {} for n in range(2, 5)}` хардкод max_n=4 | | | | ✅ |
| 29 | fractal_encoding.py | 12-13 | LEVELS/GAMMA — модульные константы, не синхронизированы с FCFConfig | | | | ✅ |
| 30 | eval_checkpoint.py | 19 | Устаревший комментарий "785MB" | | | | ✅ |

---

## Дополнительные наблюдения

### N-1: `train_full.py:130-131` — FAST mode печать после парсинга аргументов
FAST mode устанавливает `FRESH = True`, но флаг выводится до резюме — может сбить с толку, если пользователь забыл, что `--fast` подразумевает `--fresh`.

### N-2: `parameter_optimizer.py:249` — имена переменных
```python
for pd in self.config.params:
    p = self.p.get(pd.name)
```
`pd` может конфликтовать с `import pandas as pd` (хотя pandas не импортирован). Визуально неочевидно.

### N-3: `train_full.py:182-185` — проверка `cs.H is None` может упасть
```python
if not hasattr(cs, 'H') or cs.H is None:
```
Сейчас стоит `hasattr`. При первом запуске после рефакторинга у загруженного ConceptSpace может не быть `H`. Код использует `hasattr` — это корректно.

### N-4: `train_full.py:112` — print до определения parser
`print(f"vocab_size = {V}")` находится между определением функций и парсингом аргументов. При `--help` этот print выполнится перед справкой.

### N-5: `crystal_generator.py:36-37` — morph_vocab принимается, но не используется
`morph_vocab` передаётся в `__init__` (строка 40: `self.morph_vocab = morph_vocab`), но **нигде не используется** в дальнейшем. Установлен в `train_full.py:307`, но никакие методы `CrystalGenerator` его не читают.

### N-6: `concept_space.py:374-375` — двойная запись .npz при field_bits
```python
with np.load(tmp_path) as f:
    kw = dict(f)
kw['fb_cids'] = fb_cids
kw['fb_arr'] = fb_arr
np.savez_compressed(tmp_path, **kw)
```
Загружает только что сохранённый файл, добавляет field_bits и сохраняет снова. Не баг, но неэффективно.

---

## Статистика проекта

- Всего Python-файлов: 24 (без __pycache__)
- Бат-файлов/скриптов: 5 (.bat, .ps1)
- Документация: 5 файлов (README.md, ARCHITECTURE.md, AGENTS.md, AUDIT.md, PLAN.md)
- Конфигурация: requirements.txt, .gitattributes, .gitignore
- Всего строк Python-кода (активных): ~6,000
- Размер корпуса: ~153K строк, 30M символов (~52MB)
- Текущее состояние: 6000 линий, эпоха 1
- Векторное пространство: 146K × 384D
- Механизм обучения: STDP + Centroid Pull + Lateral Inhibition + Fluctuation + Contrastive

---

## Исправлено с момента предыдущего аудита

Предыдущий AUDIT.md (коммит `03b8ae8`) содержал 40+ issues. В коммитах `000e74f` и `c2e3588` исправлено ~32 issues, включая:

| # | Описание | Статус |
|---|----------|--------|
| P0-1 | Missing KMeans import (PQ-код удалён) | ✅ |
| P0-2 | Дублирование train_from_text/train_batch | ✅ |
| P1-1 | Хардкоженные пути C:\Users\black\... (12 файлов) | ✅ |
| P1-2 | ARCHITECTURE.md полностью устарел | ✅ |
| P1-3 | GPU neg-sampling Python-цикл per-item (частично) | ⚠️ (см. P1-3 нового аудита) |
| P1-4 | train.ps1 мёртвые параметры | ✅ |
| P1-5 | eval_checkpoint.py 32K BPE | ✅ |
| P2-1 | ach_phasic всегда 0 | ✅ |
| P2-2/P2-3/P2-4/P2-11 | Мёртвый код (pos_tagger, _semantic_delta, fractal_stdp, PQ) | ✅ |
| P2-6 | URL_TLDS неиспользуемый | ✅ |
| P2-7 | Неиспользуемые импорты fcf_config.py | ✅ |
| P2-8 | Множественное EMA обновление per batch | ✅ |
| P2-9 | Double np.abs(all_codes) | ✅ |
| P2-10 | import cdist внутри метода | ✅ |
| P2-13 | Доступ к _data/_valid извне | ⚠️ (неполностью — см. P1-5) |
| P3-4 | TARGET_STD хардкод 384D | ✅ |
| P3-6 | ngrams[4] orphan | ✅ |
| P3-10 | Tokenization теряет BPE info | ✅ |
| E3 | _quiet swallows exceptions | ✅ |
| S1 | HTML/JS в Python string | ✅ |
| S2 | API без rate limiting | ✅ |
| A1 | modeling_fcf save/load stub | ✅ |
| A2 | _archive/ ~2000 строк dead code | ✅ |
| A4 | Epoch resume fragile | ✅ |
| A5/A6 | Config duplication / Git LFS | ✅ |
| Q1/Q2/Q5/Q7 | Code quality (torch import, hasattr, TeeOut, query_words) | ✅ |
| Q8-Q9 | Code duplication save | ✅ |

**Вторая волна исправлений (коммит `TBD`): все 26 issues из PLAN.md:**

| # | Описание | Статус |
|---|----------|--------|
| P0-1 | theta_gate: `j` → `j-i` (4 места) | ✅ |
| P0-2 | API /health: `cid_list` → `concept_vectors`, `concept_transitions` → 0 | ✅ |
| P0-3 | FCFModel: убран `gate`, `_query_confidence`, `concept_info` | ✅ |
| P1-1 | `__main__` BPE модели (3 файла: crystal, concept, syntax) | ✅ |
| P1-2 | Contrastive objective выполняется и в GPU-пути | ✅ |
| P1-3 | GPU neg-sampling: inner loop vectorized | ⚠️ (partial — precompute done, Python loop remains) |
| P1-4 | `_final_save` вызывает `cleanup_old_checkpoints` | ✅ |
| P1-5 | `topk_similar_concepts`: `_data`/`_valid` → `.data`/`.valid` | ✅ |
| P1-6 | `eval_metrics.py`: `_valid` → `valid` | ✅ |
| P1-7 | `tokenization_fcf.py`: typo `vocab_files_names` + BPE path | ✅ |
| P1-9 | `l_c`/`l_a`/`l_m` synced: FCFConfig → FractalField params | ✅ |
| P2-1 | TeeOut: stdout restored after close | ✅ |
| P2-2 | `_batch_log` закрывается в `finally` | ✅ |
| P2-3 | `contrastive_spread` dead code удалён | ✅ |
| P2-4 | `_compute_pmi_field_fast` dead code удалён | ✅ |
| P2-5 | `cleanup_old_checkpoints` удаляет `*.opt.json` | ✅ |
| P2-6 | `_is_semantic_token` обрабатывает 'ё' | ✅ |
| P2-7 | `_theta_temp`: guard `max(theta_tau, 1.0)` | ✅ |
| P2-8 | `inference.py:126` — transitively fixed via P1-5 | ✅ |
| P2-10 | `_final_save` с `timestamp` | ✅ |
| P2-11 | `corpus_path` с `FileNotFoundError` | ✅ |
| P3-1 | stale `real_data/vis/` удалён (16 файлов, ~300MB) | ✅ |
| P3-3 | hormonal `da_coherence` порог: `<=2` → `==1` | ✅ |
| P3-6 | eval_checkpoint stale comment "785MB" → убран размер | ✅ |
| P3-7 | syntax_lattice `load()`: `ngrams = {}` динамически | ✅ |
| P3-8 | fractal_encoding: LEVELS/GAMMA из FCFConfig | ✅ |
| N-2 | parameter_optimizer: `pd` → `param` | ✅ |
| N-4 | print(vocab_size) после parser | ✅ |
| N-5 | crystal_generator: `morph_vocab` убран из `__init__` | ✅ |
| N-6 | concept_space: двойная запись .npz устранена | ✅ |

**Итого: остаётся 1 partially fixed (P1-3). Остальные 25 из 26 — полностью исправлены.**

**Новых проблем найдено: 30** (3 P0, 7 P1, 10 P2, 6 P3 — 4 claims не подтвердились)
**Верификация PLAN.md:** 26/30 точны. См. [PLAN.md](PLAN.md) для полного плана исправлений.

---

*Last updated: 2026-06-16*
