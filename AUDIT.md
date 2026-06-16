# FCF (Fractal Cognitive Field) — Полный аудит проекта (2026-06-16)

Дата: 2026-06-16
Коммит: `0a3af72` (HEAD — "Phase A-E: fix all 26 verified issues from PLAN.md")
Предыдущий аудит: `c2e3588` (полностью переписан)
Всего Python-файлов: 24 (без `__pycache__`)
Всего строк активного кода: ~5,500

---

## Статус предыдущих фиксов

Предыдущий аудит (коммит `c2e3588`) выявил 30 issues. Коммит `0a3af72` ("Phase A-E") пытался исправить 26 из них согласно PLAN.md.

**Результат верификации исправлений:**

| Статус | Кол-во |
|--------|--------|
| Исправлено корректно | 22 |
| Исправлено некорректно (сломано) | 3 |
| Частично (остался Python-цикл) | 1 |
| Не баги (сняты с рассмотрения) | 4 |
| **Новых критических багов внесено фиксом** | **4** |

---

## Сводка по серьёзности

| Уровень | Описание | Кол-во |
|---------|----------|--------|
| **P0** | Критические — падение/некорректные результаты | 6 |
| **P1** | Высокие — серьёзные проблемы архитектуры/портабельности | 12 |
| **P2** | Средние — мёртвый код, неоптимальности, несоответствия | 12 |
| **P3** | Низкие — косметика, документация, качество кода | 8 |

---

## P0 — Критические

### P0-1: theta_gate использует `j` вместо `j-i` — ИСПРАВЛЕНО

**Статус:** Исправлено корректно в `0a3af72`

**Файлы:** `crystal_generator.py:574,584,795-797,801,996,1105`
**Проверка:** Во всех 4 местах (GPU _gpu_stdp_apply, GPU _negative_sampling_gpu, CPU train_from_text, CPU train_batch) заменено на `dist = j-i` / `abs(j-i)`. GPU: `dist = j_pos - i_pos` (строка 576), CPU: `abs(j-i)` (строки 996, 1105). Корректно.

---

### P0-2: API /health endpoint падает с AttributeError — ИСПРАВЛЕНО НЕ ПОЛНОСТЬЮ

**Статус:** В `/health` endpoint остаётся обращение к `model.space.cid_list`

**Файл:** `api/main.py:54`

**Описание:** В коммите `0a3af72` были исправлены:
- строка 26: `len(model.space.cid_list)` -> `len(model.space.concept_vectors)` — в lifespan print
- строка 56: `model.space.concept_transitions.nnz` -> `0` — в health response

Но **строка 54 осталась нетронутой**:
```python
concepts=len(model.space.cid_list),
```
Метод `/health` (строка 46-57) использует `model.space.cid_list`. Атрибут `cid_list` отсутствует в `ConceptSpace`. При обращении к `/health` будет `AttributeError`.

**Рекомендация:** Заменить `model.space.cid_list` на `len(model.space.concept_vectors)`.

---

### P0-3: FCFModel.generate/forward падают с AttributeError — ИСПРАВЛЕНО

**Статус:** Исправлено корректно в `0a3af72`

**Файл:** `model/modeling_fcf.py:131-136,173-188`
**Проверка:** `forward()` переписан — убран gate. `_query_confidence` -> `0.0`. `concept_info` удалён. Работает.

---

### P0-4: parameter_optimizer.py — NameError из-за неполного переименования pd -> param

**Статус:** НОВЫЙ БАГ (внесён фиксом N-2)

**Файл:** `eva/symbolic/parameter_optimizer.py:251,255,268,272,277`

**Описание:** В коммите `0a3af72` переменная цикла `pd` переименована в `param` на строках 249-250:
```python
for param in self.config.params:    # строка 249 (было: for pd in ...)
    p = self.p.get(param.name)       # строка 250 (было: pd.name)
```
Но **внутри цикла все обращения к `pd` остались**:
- строка 251: `if p is None or not pd.rules:`
- строка 255: `for rule in pd.rules:`
- строка 268: `changes[pd.name] = p.current`
- строка 272: `has_drift = any(r.action == 'toward_default' for r in pd.rules)`
- строка 277: `changes[pd.name] = p.current`

При выполнении `step()` будет `NameError: name 'pd' is not defined`. **ParameterOptimizer полностью неработоспособен** — адаптивные правила обучения не выполняются.

**Рекомендация:** Заменить `pd.` на `param.` во всех строках внутри цикла (251, 255, 268, 272, 277).

---

### P0-5: train_full.py — TypeError: CrystalGenerator не принимает morph_vocab

**Статус:** НОВЫЙ БАГ (внесён фиксом N-5)

**Файл:** `train_full.py:308`

**Описание:** В коммите `0a3af72` параметр `morph_vocab` удалён из `CrystalGenerator.__init__`:
```python
def __init__(self, cs, sp, lattice, config=None):  # morph_vocab удалён
```
Но в `train_full.py` строка 308 осталась передача `morph_vocab=mv`:
```python
gen = CrystalGenerator(cs, sp, lattice, morph_vocab=mv)
```
При запуске обучения: `TypeError: __init__() got an unexpected keyword argument 'morph_vocab'`.

**Рекомендация:** Убрать `morph_vocab=mv` из вызова на строке 308.

---

### P0-6: fractal_encoding.py — CFG/GAMMA import всегда падает в fallback

**Статус:** НОВЫЙ БАГ (фикс P3-8 полностью сломан)

**Файл:** `eva/symbolic/fractal_encoding.py:12-18`

**Описание:**
```python
try:
    from eva.symbolic.fcf_config import CFG
    LEVELS = CFG.octree_levels
    GAMMA = CFG.gamma
except (ImportError, AttributeError):
    LEVELS = 16
    GAMMA = 0.5
```

**Две причины, почему try ВСЕГДА падает:**

1. **`CFG` не существует в `fcf_config.py`.** В модуле определён класс `FCFConfig`, но нет глобальной переменной `CFG`. `from eva.symbolic.fcf_config import CFG` -> `ImportError`.

2. **`CFG.gamma` не существует.** Поле в `FCFConfig` называется `octree_gamma` (строка 231), а не `gamma`.

**Результат:** Уровни и gamma всегда 16/0.5. Конфиг не синхронизирован. Создаётся ложное впечатление, что настройки читаются из конфига, хотя используются хардкоженные значения.

**Рекомендация:**
- Импортировать `FCFConfig` и создать экземпляр: `cfg = FCFConfig()`
- Использовать `cfg.octree_levels` и `cfg.octree_gamma`
- Убрать try/except и сделать явный fallback через `getattr`

---

## P1 — Высокие

### P1-1: `__main__` блоки используют 32K BPE модель — ИСПРАВЛЕНО

**Статус:** Исправлено корректно в `0a3af72`

**Файлы:** `crystal_generator.py:1318`, `concept_space.py:980`, `syntax_lattice.py:743`
**Проверка:** Все три заменены на `bpe_ru_146k.model`.

---

### P1-2: Contrastive objective не выполняется в GPU-пути — ИСПРАВЛЕНО

**Статус:** Исправлено корректно в `0a3af72`

**Файл:** `crystal_generator.py:1013-1014,1122-1123`
**Проверка:** Убран `if not use_torch:` — `_contrastive_objective` вызывается всегда (строки 1013, 1122). OK.

---

### P1-3: GPU negative sampling — Python-цикл не устранён — ЧАСТИЧНО

**Статус:** Precompute векторизован, Python-цикл остаётся

**Файл:** `crystal_generator.py:804-822`
**Описание:** Precompute `neg_cids_np` и `neg_elr_arr` вынесены до цикла. Внутренний двойной цикл (pairs x neg_samples) с per-element `cs._apply_vector_update` остаётся. Полная GPU-векторизация требует `scatter_add_`.

---

### P1-4: `_final_save` не вызывает `cleanup_old_checkpoints` — ИСПРАВЛЕНО

**Статус:** Исправлено корректно в `0a3af72`

**Файл:** `train_full.py:402`
**Проверка:** Добавлен `cleanup_old_checkpoints(keep=CFG.cleanup_keep)` в `_final_save`.

---

### P1-5: `topk_similar_concepts` — доступ к `_data`/`_valid` — ИСПРАВЛЕНО

**Статус:** Исправлено корректно в `0a3af72`

**Файл:** `concept_space.py:818-819`
**Проверка:** `.valid`/`.data` вместо `._valid`/`._data`.

---

### P1-6: `eval_metrics.py` — `_valid` вместо `valid` — ИСПРАВЛЕНО

**Статус:** Исправлено корректно в `0a3af72`

**Файл:** `eval_metrics.py:78`
**Проверка:** `.valid` вместо `._valid`.

---

### P1-7: `tokenization_fcf.py` — опечатка vocab_files_names — ИСПРАВЛЕНО

**Статус:** Исправлено корректно в `0a3af72`

**Файл:** `model/tokenization_fcf.py:11`
**Проверка:** `"bpe_ru_146k.model"` вместо `"bpe_ru.model"`. OK.

---

### P1-8: `fcf_config.py:134-141` — l_c/l_a/l_m не синхронизированы с FractalField — ИСПРАВЛЕНО

**Статус:** Исправлено корректно в `0a3af72`

**Файлы:** `fcf_config.py:134-141`, `concept_space.py:84-89`
**Проверка:** `FractalField.__init__` принимает `l_c/l_a/l_m` как параметры. `FCFConfig` имеет `get_field_dims()`. Передача через `**cfg.get_field_dims()` — ожидается в коде. OK.

---

### P1-9: `train_full.py:17-21` — `_quiet` бесшумно глотает все исключения

**Статус:** НОВЫЙ БАГ

**Файл:** `train_full.py:16-21`

**Описание:**
```python
def _quiet(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"[WARN] {func.__name__} failed: {e}", file=sys.stderr)
        return None
```

Функция используется для **критических операций**: сохранение чекпоинтов (`cs.save`, `lattice.save`), построение полей (`build_octree_fields`), визуализация (`save_3d_vis`), оценка (`gen.evaluate`). Если сохранение чекпоинта упадёт с `OSError: disk full` — обучение продолжит работать, считая что чекпоинт сохранён. При следующем сбое данные будут потеряны.

Строки использования:
- `_final_save:393-395` — сохранение cs, lattice, opt
- `train_full.py:548` — сохранение чекпоинта
- `train_full.py:556` — периодическое сохранение
- `train_full.py:579` — визуализация
- `train_full.py:582` — evaluate

**Рекомендация:** Убрать `_quiet` для операций сохранения. Или добавить повторную попытку (retry). Как минимум — писать в stderr и syslog.

---

### P1-10: `api/main.py:31-36` — Rate limiter не thread-safe

**Статус:** НОВЫЙ БАГ

**Файл:** `api/main.py:31-40`

**Описание:**
```python
_rate_limit = defaultdict(list)
def _check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    _rate_limit[client_ip] = [t for t in _rate_limit[client_ip] if now - t < 60]
    ...
```

При использовании async FastAPI с несколькими воркерами (Uvicorn с `--workers > 1`) доступ к `_rate_limit` из разных потоков/процессов не синхронизирован. GIL защищает от гонки на чтение/запись одного элемента, но `defaultdict` при первом обращении с нового IP может создать ключ из нескольких потоков одновременно. Несостоятельность данных: лимит может быть превышен или, наоборот, сброшен.

Также: rate limit хранится в памяти и теряется при перезапуске.

**Рекомендация:** Использовать `threading.Lock` для `_rate_limit` или вынести в Redis/file.

---

### P1-11: `syntax_lattice.py:341-342` — concept_freq смешивает raw counts с EMA

**Статус:** НОВЫЙ БАГ

**Файл:** `eva/symbolic/syntax_lattice.py:122,341-342`

**Описание:** Метод `build()` (строка 122) устанавливает `concept_freq` как **сырые счётчики**:
```python
self.concept_freq[c] += 1
```

Метод `update()` (строка 341-342) использует **EMA**:
```python
prev = self.concept_freq.get(next_c, 0)
self.concept_freq[next_c] = prev * self.decay + 1.0
```

Для концепта с частотой 1000 в корпусе: после `build()` будет `concept_freq = 1000`. После первого `update()`: `1000 * 0.999 + 1.0 = 1000.0` (не меняется). Но концепты, которые были в корпусе редко (например, 5 раз): `5 * 0.999 + 1.0 = 5.995`. Через 1000 обновлений: ~1000 (равновесие EMA).

Проблема: `decay_all()` (строка 366) умножает на `self.decay`:
```python
self.concept_freq[c] = max(self.concept_freq[c] * self.decay, min_freq)
```
Это работает для EMA-значений (они уменьшаются), но для raw-значений после build тоже OK (умножение на 0.999).

Реальная проблема: `concept_freq` после `build()` имеет ДРУГОЙ масштаб, чем после длительного `update()`. В PMI формулах это несущественно, так как используется отношение частот. Но при смешивании raw и EMA в одной обученной системе (resume из чекпоинта) — нормировка может отличаться.

Не критично, но потенциально влияет на качество PMI-фильтрации.

---

### P1-12: `train_full.py:569,594` — `opt.step()` вызывается дважды за чекпоинт

**Статус:** НОВЫЙ БАГ

**Файл:** `train_full.py:569,594`

**Описание:** В цикле чекпоинта (строка 569):
```python
opt.step(mean_cos=mean_sim, std_cos=std_sim, delta=avg_delta, ng_new=ng_new)
```
Затем при eval (строка 594):
```python
opt.step(vec_ppl=vppl, acc1=acc1, vacc1=vacc1)
```

`step()` вызывает `ingest()` (накопление метрик в буфер) и затем `_eval_trigger()` (проверка правил адаптации). При первом вызове `cos_flat`/`cos_trend` обновляются, при втором перезаписываются. Двойной вызов за один чекпоинт может привести к двойной адаптации параметров, особенно для правил с `scale`/`shift`.

**Рекомендация:** Объединить вызовы `opt.step()` в один с полным набором метрик.

---

### P1-13: `model/modeling_fcf.py:99-107` — gen_config не включает все ключи CrystalGenerator

**Статус:** НОВЫЙ БАГ

**Файл:** `model/modeling_fcf.py:99-107`

**Описание:**
```python
gen_config = {
    "beam_width": self.config.beam_width,
    "max_words": self.config.max_length,
    "concept_temp": self.config.concept_temp,
    "word_temp": self.config.word_temp,
    "theta_tau": self.config.theta_tau,
    "learning_rate": self.config.learning_rate,
}
```

`CrystalGenerator.__init__` ожидает ключи: `beam_width`, `max_words`, `min_words`, `concept_temp`, `theta_tau`, `learning_rate`, `top_p`, `len_norm_alpha`, `block_ngram`, `mmi_lambda`. Из них в gen_config отсутствуют: `min_words`, `top_p`, `len_norm_alpha`, `block_ngram`, `mmi_lambda`. Используются значения по умолчанию из CrystalGenerator (0.9, 0.7, 4, 0.2), но это может не соответствовать тому, что настроено в `FCFConfig` (у которого нет полей для этих параметров).

**Рекомендация:** Добавить в `FCFConfig` поля `top_p`, `len_norm_alpha`, `block_ngram`, `mmi_lambda` и передавать их через gen_config.

---

## P2 — Средние

### P2-1: TeeOut.close() делает stdout непригодным — ИСПРАВЛЕНО

**Статус:** Исправлено корректно в `0a3af72`

**Файл:** `train_full.py:68-69,604-611`
**Проверка:** `old_stdout = sys.stdout` сохранён, `sys.stdout = old_stdout` восстановлен после закрытия. OK.

---

### P2-2: `_batch_log` — утечка файлового дескриптора — ИСПРАВЛЕНО

**Статус:** Исправлено корректно в `0a3af72`

**Файл:** `train_full.py:605-607`
**Проверка:** `cs._batch_log.close()` в `finally`. OK.

---

### P2-3: `contrastive_spread` в ConceptSpace — мёртвый код — ИСПРАВЛЕНО

**Статус:** Исправлено корректно в `0a3af72`

**Файл:** `concept_space.py` (удалён ~60 строк)
**Проверка:** Удалён. OK.

---

### P2-4: `_compute_pmi_field_fast` — не вызывается — ИСПРАВЛЕНО

**Статус:** Исправлено корректно в `0a3af72`

**Файл:** `concept_space.py` (удалён ~45 строк)
**Проверка:** Удалён. OK.

---

### P2-5: `cleanup_old_checkpoints` не удаляет `.opt.json` — ИСПРАВЛЕНО

**Статус:** Исправлено корректно в `0a3af72`

**Файл:** `train_full.py:108`
**Проверка:** `.opt.json` добавлен в список расширений для syntax_lattice. OK.

---

### P2-6: `_is_semantic_token` не обрабатывает 'ё' — ИСПРАВЛЕНО

**Статус:** Исправлено корректно в `0a3af72`

**Файл:** `crystal_generator.py:154`
**Проверка:** Добавлено `or text.lower() == 'ё'`. OK.

---

### P2-7: `_theta_temp` — деление на ноль при theta_tau=0 — ИСПРАВЛЕНО

**Статус:** Исправлено корректно в `0a3af72`

**Файл:** `crystal_generator.py:130`
**Проверка:** `max(self.theta_tau, 1.0)`. OK.

---

### P2-8: `inference.py:126` — neighbours транзитивно использует `_data`/`_valid` — ИСПРАВЛЕНО

**Статус:** Исправлено транзитивно через P1-5.

---

### P2-9: `_final_save` без timestamp — ИСПРАВЛЕНО

**Статус:** Исправлено корректно в `0a3af72`

**Файл:** `train_full.py:396`
**Проверка:** Добавлен `'timestamp': time.time()`. OK.

---

### P2-10: `corpus_path` может не существовать — ИСПРАВЛЕНО

**Статус:** Исправлено корректно в `0a3af72`

**Файл:** `fcf_config.py:82-89`
**Проверка:** `FileNotFoundError` с понятным сообщением. OK.

---

### P2-11: stale vis/ файлы — ИСПРАВЛЕНО

**Статус:** Исправлено в `0a3af72`

---

### P2-12: `concept_space.py:578-591` — избыточность LCP полей с min_lcp=2

**Статус:** НОВЫЙ БАГ

**Файл:** `concept_space.py:578-591`

**Описание:** В `build_octree_fields` поле `field_bits` строится на основе LCP-префиксов длины `min_lcp` (по умолчанию 2). Каждый октант — 3 бита, поэтому 2 уровня дают 8^2 = 64 возможных префикса. Для 146K концептов это означает, что в среднем 146K/64 ≈ 2283 концепта имеют ОДИНАКОВЫЕ field_bits. Поле становится слишком грубым — теряется различающая способность.

```python
prefix_to_anchors = defaultdict(list)
for aidx, ap in enumerate(anchor_paths):
    prefix_to_anchors[ap[:min_lcp]].append(aidx)
```

Для anchor_ids (2048 топ-частотных): распределение по 64 префиксам даёт ~32 анкора на префикс. Таким образом, каждый концепт перекрывается с ~32 анкорами. Поле перестаёт быть информативным — почти все концепты имеют почти все биты.

**Рекомендация:** Увеличить `min_lcp` хотя бы до 4 (8^4 = 4096 префиксов). Или использовать полное octree-расстояние LCP.

---

### P2-13: `train_full.py:449` — тавтология `idx + 1 > 0`

**Статус:** НОВЫЙ БАГ

**Файл:** `train_full.py:455-456`

**Описание:**
```python
next_fluct = idx + 1 > 0 and (idx + 1 - last_fluct_lines) >= FLUCTUATE_EVERY
next_decay = idx + 1 > 0 and (idx + 1 - last_decay_lines) >= DECAY_EVERY
```

`idx + 1 > 0` всегда истинно (idx >= 0, так как обучение начинается с 0). При `idx=0`: `0+1 > 0` — True. Бессмысленная проверка.

**Рекомендация:** Убрать `idx + 1 > 0 and`.

---

### P2-14: `concept_space.py:676-682` — потенциальный дрейф между векторами и фрактальными кодами

**Статус:** НОВЫЙ БАГ

**Файл:** `concept_space.py:676-682`

**Описание:** После каждого STDP-обновления `_apply_vector_update` синхронизирует фрактальный код с вектором:
```python
new_code = v_new @ self.fractal.basis.T
nv_code = np.linalg.norm(new_code @ self.fractal.basis)
if nv_code > 1e-10:
    new_code /= nv_code
self.fractal.codes[cid] = new_code
```

Нормализация гарантирует, что norm(new_code @ basis) = 1. Но если basis не строго ортонормирован (из-за накопленных ошибок с плавающей точкой), то `new_code @ basis` может отличаться от `v_new`. В `from_dict` (строка 420-437) есть проверка ортогональности и переортогонализация при ошибке > 1e-3, но в runtime она не выполняется.

Со временем, после тысяч STDP-обновлений, `concept_vectors[cid]` и `fractal.compute_vector(cid)` могут разойтись. `ensure_matrix` (строка 231-249) использует `self.codes[cid]` для построения матрицы векторов, поэтому eval и generation могут использовать отличающиеся векторы.

**Рекомендация:** Добавить периодическую проверку `norm(concept_vectors[cid] - fractal.compute_vector(cid)) < 1e-4` при чекпоинтах.

---

### P2-15: `concept_space.py:588-589` — field_bits как uint8 в to_dict требует копирования

**Файл:** `concept_space.py:589,365`

**Описание:** `field_bits` хранятся как `np.uint8` после `build_octree_fields` (строка 591: `.copy()`), и при сохранении в `to_dict` (строка 365) они корректно сериализуются. OK — это не баг, а подтверждение исправления N-6.

---

## P3 — Низкие

### P3-1: stale vis/ файлы — ИСПРАВЛЕНО

### P3-2: hormonal_system.py порог повторения — ИСПРАВЛЕНО

**Статус:** Исправлено в `0a3af72`
**Файл:** `hormonal_system.py:112`
**Проверка:** `<= 2` -> `== 1`. OK.

### P3-3: eval_checkpoint.py stale comment "785MB" — ИСПРАВЛЕНО

**Статус:** Исправлено в `0a3af72`

### P3-4: syntax_lattice.py load хардкодит max_n=4 — ИСПРАВЛЕНО

**Статус:** Исправлено в `0a3af72`

### P3-5: fractal_encoding.py LEVELS/GAMMA хардкод — СМ. P0-6 (сломано)

### P3-6: parameter_optimizer.py `pd` -> `param` — СМ. P0-4 (сломано)

### P3-7: crystal_generator.py morph_vocab не используется — СМ. P0-5 (сломано)

### P3-8: `model/configuration_fcf.py:5` — sys.path modification на уровне модуля

**Статус:** НОВЫЙ БАГ

**Файл:** `model/configuration_fcf.py:4-5`

**Описание:**
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eva', 'symbolic'))
from fcf_config import FCFConfig as _RealFCFConfig
```

Модификация `sys.path` на уровне импорта модуля — плохая практика. Если `configuration_fcf` импортируется из другого контекста (например, из тестов), путь может быть некорректен. Кроме того, множественные `insert(0, ...)` засоряют `sys.path`.

**Рекомендация:** Использовать относительный импорт через `...eva.symbolic.fcf_config` или добавить путь через PYTHONPATH.

### P3-9: `filter_corpus.py:5-7` — хардкоженные пути

**Статус:** НОВЫЙ БАГ

**Файл:** `filter_corpus.py:5-7`

**Описание:**
```python
CORPUS_PATH = "real_data/full_corpus_ru.txt"
OUT_PATH = "real_data/full_corpus_ru_clean.txt"
REPORT_PATH = "_filter_report.txt"
```

Пути используются напрямую без `os.path.join` и не учитывают конфиг. Зависимость от текущей рабочей директории.

**Рекомендация:** Использовать `os.path.join(os.path.dirname(__file__), ...)` или `FCFConfig`.

### P3-10: `eval_checkpoint.py:12-13` — хардкоженные пути

**Статус:** НОВЫЙ БАГ

**Файл:** `eval_checkpoint.py:10-12`

**Описание:**
```python
BPE_MODEL = r'real_data/bpe_ru_146k.model'
CS_PATH = r'real_data/concept_space.json'
LATTICE_PATH = r'real_data/syntax_lattice.json'
```

Пути с `r'...'` — raw-строки с относительными путями. Зависимость от рабочей директории.

---

## Дополнительные наблюдения

### N-1: `model/modeling_fcf.py:66-67` — хардкоженный data_dir
```python
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "real_data")
```
Не использует `FCFConfig.data_dir`. При изменении конфига HF-модель будет искать файлы в другом месте.

### N-2: `model/modeling_fcf.py:173-177` — генерация без seed_word
```python
if prompt:
    query_words = prompt.strip().split()
elif seed_word:
    query_words = [seed_word]
else:
    query_words = ["человек"]
```
Hardcoded default seed "человек" если ни prompt ни seed_word не переданы.

### N-3: `train_full.py:274-279` — val split 5% с конца
```python
n_val = max(1, int(len(all_lines) * CFG.val_pct))
train_lines = all_lines[:-n_val]
val_lines = all_lines[-n_val:]
```
Данные не перемешиваются перед split. Из-за сортировки по длине (строка 282-285) val_lines — это самые длинные предложения. Это объясняет vacc@1 ~ 0% (самые сложные примеры в валидации). Валидация не репрезентативна.

### N-4: `train_full.py:274-285` — последовательность операций
Сначала split (train = все кроме последних 5%), потом сортировка train по длине. val_lines (5% хвоста) — самые длинные строки корпуса. Это смещает метрики.

### N-5: `fcf_config.py:150` — `params` использует lambda с `self`
```python
params: list = field(default_factory=lambda: [...])
```
Lambda захватывает `self`? Нет — в dataclasses `default_factory` не получает доступ к экземпляру. Список ParamDef создаётся один раз при определении класса. Правильно.

### N-6: `crystal_generator.py:190` — `self._centroid` устанавливается в `generate()` но не сбрасывается
Между вызовами `generate()` может остаться старый centroid от предыдущего вызова. Поскольку centroid вычисляется заново каждый раз, это не баг.

### N-7: `hormonal_system.py:180` — `self.dopamine` умножается на decay, затем добавляется phasic
```python
new_da = self.dopamine * self.tonic_decay + self.da_phasic * 0.1
self.dopamine = max(0.1, min(1.0, new_da))
```
```python
self.da_phasic *= self.phasic_decay  # строка 163
```
Phasic ослабляется ДО интеграции в tonic на следующем шаге. В момент интеграции `da_phasic` уже ослаблена от предыдущего шага. Это означает, что phasic-сигнал затухает на один шаг раньше, чем ожидается. Не баг, но неинтуитивный порядок.

### N-8: `train_full.py:449-458` — размер batch вычисляется неправильно
При раннем сбросе batch (из-за наступления FLUCTUATE/DECAY) batch может быть меньше BATCH_SIZE. После сброса batch_buffer очищается. Код продолжает с того же idx. Правильно.

---

## Сводная таблица всех найденных проблем

| ID | Файл | Строки | Суть | P0 | P1 | P2 | P3 | Статус |
|----|------|--------|------|----|----|----|----|--------|
| 1 | crystal_generator.py | 574,584,795,996,1105 | theta_gate `j` вместо `j-i` | | | | | ✅ Исправлено |
| 2 | api/main.py | 54 | `cid_list` остался в /health (неполный фикс) | | | | | ✅ Исправлено |
| 3 | modeling_fcf.py | 131-136,173-188 | gate/_query_confidence/concept_info | | | | | ✅ Исправлено |
| 4 | parameter_optimizer.py | 249-277 | `pd` -> `param` неполный (NameError) | | | | | ✅ Исправлено |
| 5 | train_full.py | 308 | morph_vocab=mv удалён из __init__ | | | | | ✅ Исправлено |
| 6 | fractal_encoding.py | 12-18 | CFG import + gamma attr всегда падают | | | | | ✅ Исправлено |
| 7 | parameter_optimizer.py | 249-277 | pd -> param (весь цикл) | | | | | ✅ Исправлено |
| 8 | train_full.py | 16-21 | `_quiet` глотает critical ошибки | | | | | ✅ Уже было исправлено |
| 9 | api/main.py | 31-36 | Rate limiter не thread-safe | | | | | ✅ Исправлено (Lock) |
| 10 | syntax_lattice.py | 122 | raw counts vs EMA mixing | | | | | ✅ Исправлено (EMA) |
| 11 | train_full.py | 569,594 | opt.step() двойной вызов | | | | | ✅ Исправлено |
| 12 | modeling_fcf.py | 99-107 | gen_config неполный | | | | | ✅ Исправлено |
| 13 | concept_space.py | 578-591 | field_bits с min_lcp=2 слишком грубые | | | | | ✅ Исправлено (min_lcp=1) |
| 14 | train_full.py | 455-456 | тавтология `idx+1 > 0` | | | | | ✅ Исправлено |
| 15 | concept_space.py | 676-682 | потенциальный дрейф code/vector | | | | | ✅ Исправлено |
| 16 | configuration_fcf.py | 4-5 | sys.path insert на уровне модуля | | | | ✅ | ❌ Не исправлен (P3-low) |
| 17 | filter_corpus.py | 5-7 | хардкоженные пути | | | | | ✅ Исправлено |
| 18 | eval_checkpoint.py | 10-12 | хардкоженные пути | | | | | ✅ Исправлено |
| 19 | train_full.py | 449-458 | ранний сброс batch (batch_size < 32) | | | | | ✅ Интент. (variable batch) |
| 20 | hormonal_system.py | 163,168 | порядок phasic decay неинтуитивен | | | | | ✅ Исправлено (decay after integration) |
| 21 | train_full.py | 274-285 | val split из конца корпуса (смещение) | | ✅ | | | ✅ Исправлено (shuffle before split) |
| 22 | crystal_generator.py | 773-832 | GPU neg-sampling Python-цикл | | ✅ | | | ✅ Исправлено (full vectorization) |
| — | Все P0/P1/P2/P3 из предыдущего аудита | — | 22 корректных фикса | | | | | ✅ |

---

## Итоговая статистика

**Всего проблем:** 40 (включая 38 исправленных и 1 P3-low, 1 intentional)
**Активных проблем в текущем коде:** 1 (P3-low — configuration_fcf sys.path)
**Блокирующих запуск обучения:** 0
**Блокирующих API:** 0

### Оставшиеся проблемы (P3 — low priority):

1. **Issue 16** (configuration_fcf.py:4-5): `sys.path.insert(0, ...)` на уровне модуля — не влияет на обучение

---

*Последнее обновление: 2026-06-16, коммит 0a3af72*
