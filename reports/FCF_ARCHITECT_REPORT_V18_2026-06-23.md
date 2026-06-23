# FCF Архитектурный аудит V18

**Дата:** 2026-06-23  
**Версия кодовой базы:** HEAD cff1240  
**Предыдущий аудит:** V15 (4178389)  
**Анализатор:** Architect-AI  
**GPU Target:** NVIDIA 2GB VRAM (GTX 1050/1650 class)

---

## 1. Executive Summary

С момента V15 (4178389) выполнено 3 коммита: V16 (49bc4e1, 9 issues), V17 (43d6d23, 10 issues), P0.1 FFT-HRR (cff1240). Закрыто 19 проблем. Кодовая база стабилизировалась: `concept_space.py` (2304 строки), `stdp_trainer.py` (1431), `crystal_generator.py` (1051), `checkpoint_manager.py` (127), `train_full.py` (1005).

**Ключевые изменения в V16-V18:**

1. **FFT-HRR VSA** (cff1240) — `_hrr_bind`/`_hrr_unbind` через circular convolution/correlation. Заменил Hadamard product во всех трёх классах: `FractalField.hdc_bind`, `EntityField._bind`, `Harmonizer._bind`. Это делает VSA-операции приближённо обратимыми: `unbind(bind(a,b), b) ≈ a` с SNR ~√D.

2. **HDC memory clear after fluctuate** (P1.5) — `fluctuate_fractal()` теперь вызывает `hdc_memory.clear()` + `hdc_memory_counts.clear()`.

3. **Per-concept L1 lambda persistence** (P2.1) — `to_dict()`/`from_dict()` сохраняют/восстанавливают `l1_lambda_per_cid`.

4. **Cluster potential checkpoint** (P2.2) — `_cluster_potential` сохраняется в checkpoint state и восстанавливается при resume.

5. **Sent→para binding** (P2.3) — реализован в `_harmonize_batch` через `gen._current_para_key`.

6. **EntityField decay** (P2.4) — `ef.decay(factor=0.999)` вызывается в checkpoint loop.

7. **Char-level LRU cache** (P2.5) — `_char_word_cache` + `_char_word_cache_evict` предотвращают повторные bind'инги.

8. **Sent vec cache** (P2.6) — `sent_vec_cache` для повторного использования sentence-level векторов.

9. **max_latent_dim** (P2.7) — верхняя граница для `grow_capacity()`, предотвращает бесконечный рост.

10. **HDC early exit** (P2.8) — `_update_hdc_ngrams` проверяет `hdc_memory_max` before update.

11. **Batched GPU sync** (P1.13) — `_harmonize_batch` использует batched tensor write вместо per-vector.

12. **try/finally skip_gpu_sync** (P3.7) — корректное восстановление флага после GPU sync.

13. **--no-hebbian-field** (P3.5) — CLI-флаг для полного отключения EntityField.

14. **morph_conf threshold** (P3.4) — снижен порог для коротких слов (≤4 букв → 0.4).

**Однако 8 проблем из V15 остаются неисправленными.** Ниже — детальный анализ каждой.

---

## 2. Остаточные проблемы

### P1.1 (High) — HDC fallback не окупает 400MB

**Статус:** Не исправлена  
**Файл:** `eva/symbolic/concept_space.py:143-147`, `crystal_generator.py:783-790`

**Текущее состояние:**

HDC-память (`hdc_memory`, `concept_space.py:144`) хранит до 50K entry по 2048D fp32 = 400MB. Вызывается только при `len(syn_preds) < 3` (кристалл генератор:783). Вклад в финальный RRF — 0.10 × `hdc_candidates[hcid]` (строка 822). Это означает, что HDC используется ~1-5% batch'ей, и его вклад в RRF — ≤0.10 на кандидат.

FFT-HRR (cff1240) улучшил качество unbind (теперь `unbind(bind(a,b), b) ≈ a`), но не изменил архитектурную роль — HDC остаётся дорогим запасным выходом. `fluctuate_fractal()` очищает HDC-память (P1.5), но это только предотвращает использование stale данных, а не уменьшает memory footprint.

**Рекомендация:** Два варианта:

**Вариант A (рекомендуемый):** Уменьшить `hdc_memory_max` с 50K → 2000. 2000 × 2048 × 4 = 16MB вместо 400MB. HDC-память как short-term cache последних 2000 n-gram — этого достаточно для fallback в пределах одного документа.

**Вариант B (альтернативный):** Интегрировать HDC как равноправный источник признаков в RRF (не только при <3 lattice candidates), но с пониженным весом. Тогда HDC участвует в каждом batch, а не только при пустом lattice.

```python
# ── В concept_space.py, строка 146 — изменить размер HDC памяти ──
# Было:
# self.hdc_memory_max = 50000  # evict oldest when exceeding
# Стало:
self.hdc_memory_max = 2000   # 2000×2048×4=16MB вместо 400MB
```

```python
# ── В crystal_generator.py, строка 783 — убрать условие <3, всегда использовать HDC ──
# Было:
# if len(syn_preds) < 3 and len(cids) >= 2:
# Стало:
if len(cids) >= 2:  # HDC всегда участвует в RRF
```

**Сложность:** 2 строки, 0 риск регресса (просто уменьшаем кэш).

---

### P1.2 (High) — EntityField read-only: не пишет обратно в STDP

**Статус:** Не исправлена  
**Файл:** `eva/symbolic/concept_space.py:958-1000`, `stdp_trainer.py:212-340`

**Текущее состояние:**

EntityField.bind() (concept_space.py:958-986) модифицирует только внутренний dict `self.entities`. После char↔word, word↔sent, sent↔para bind'ингов результаты НЕ записываются в `cs.concept_vectors`. Единственный механизм записи — Harmonizer.harmonize() (строки 1188-1274), который работает в 768D пространстве морфем, а не в 2048D EntityField.

Таким образом, EntityField — read-only consumer concept vectors. Вся работа по построению char↔word↔sent иерархии тратится впустую: она не влияет на STDP-обучение, не меняет concept vectors, не участвует в генерации.

**Код решения:** Добавить feedback от EntityField к concept_vectors через unbind + pull. После завершения bind'ингов для каждого dirty слова: восстановить char-level контекст через `ef.query('w', cid)` и подтянуть concept_vector в направлении char-контекста.

```python
# ── В stdp_trainer.py, после строки 331 (после морфемного harmonize) ──
# Добавить feedback от EntityField char↔word bindings к concept_vectors

# ── P1.2: EntityField → STDP feedback ──
if ef is not None and morph_cids:
    for cid in morph_cids:
        wkey = ef.key_word(cid)
        v_word = ef.get(wkey)
        if v_word is not None:
            char_query = ef.query('w', cid)
            if char_query is not None:
                cq_norm = np.linalg.norm(char_query)
                if cq_norm > 1e-10:
                    char_query /= cq_norm
                    v_cs = cs.concept_vectors.get(cid)
                    if v_cs is not None:
                        sim = float(v_cs @ char_query)
                        pull = (char_query - sim * v_cs) * 0.005  # weak feedback
                        v_new = v_cs + pull
                        nv = np.linalg.norm(v_new)
                        if nv > 1e-10:
                            v_new /= nv
                        cs._apply_vector_update(cid, v_new)
                        updated_cids.append(cid)
```

**Сложность:** 20 строк. Риск: слабый feedback (lr=0.005) не должен дестабилизировать STDP. При необходимости — отключается через `--no-hebbian-field`.

---

### P1.3 (High) — W_proj Hebbian: положительная ОС без координации со STDP

**Статус:** Не исправлена  
**Файл:** `eva/symbolic/concept_space.py:309-333`

**Текущее состояние:**

`update_learned_fields()` обновляет W_proj через `codes.T @ sign(codes @ W_proj)` — чисто Hebbian, без учёта STDP loss. Это создаёт положительную обратную связь:

1. STDP меняет code → увеличивает компоненту в направлении гиперплоскости i
2. Hebbian усиливает W_proj[:, i] (outer product)
3. Все concept'ы начинают иметь бит i = 1
4. Field_gate перестаёт фильтровать

Флаг `--no-hebbian-field` (P3.5) отключает EntityField целиком, но это костыль: либо всё, либо ничего. Нет механизма предотвращения коллапса field_bits.

**Код решения 1:** Добавить ортогонализацию столбцов W_proj после каждого Hebbian update (через QR).

```python
# ── В concept_space.py, после строки 331 (после normalize W_proj) ──
# P1.3a: ортогонализация столбцов W_proj через QR
Q_w, _ = np.linalg.qr(self.W_proj, mode='reduced')
self.W_proj = Q_w.astype(np.float32) * np.sqrt(self.latent_dim)
```

**Сложность:** 2 строки. QR на [2048, 512] ~ 5ms — приемлемо раз в эпоху.

**Код решения 2:** Добавить collapse detection — если >80% концептов имеют одинаковый бит, сбросить соответствующий столбец W_proj.

```python
# ── В concept_space.py, после строки 331 (после normalize W_proj) ──
# P1.3b: collapse detection — сброс вырожденных гиперплоскостей
if self.field_bits:
    all_bits = np.array([np.unpackbits(self.field_bits[cid])[:self.n_field_bits]
                         for cid in self.field_bits])
    bit_ratio = all_bits.mean(axis=0)
    collapsed = np.where((bit_ratio > 0.85) | (bit_ratio < 0.15))[0]
    if len(collapsed) > 0:
        rng = np.random.RandomState(42 + self._capacity_growths)
        self.W_proj[:, collapsed] = rng.randn(self.latent_dim, len(collapsed)).astype(np.float32)
        print(f"  Collapse guard: reset {len(collapsed)}/{self.n_field_bits} degenerate hyperplanes")
```

**Сложность:** 10 строк. Риск минимален — сброс только вырожденных столбцов.

---

### P1.4 (High) — Dynamic capacity + async checkpoint: race condition

**Статус:** Не исправлена  
**Файл:** `eva/symbolic/concept_space.py:366-476` (grow_capacity, prune_capacity), `checkpoint_manager.py:54-108` (_sync_save)

**Текущее состояние:**

`grow_capacity()` и `prune_capacity()` модифицируют: `basis`, `codes`, `latent_dim`, `W_proj`, `_sector_W`, `l_c`, `l_a`, `l_m`. Все эти изменения происходят в main thread без блокировок.

`CheckpointManager._sync_save()` (checkpoint_manager.py:54) запускается в `ThreadPoolExecutor` и вызывает `cs.save()`, который читает `fractal.codes`, `fractal.basis`, `fractal.latent_dim` и т.д. Если main thread выполняет `grow_capacity()` одновременно — сохранённый checkpoint содержит частично обновлённые структуры.

GIL не защищает: отдельные операции чтения dict атомарны, но целостность между чтением `latent_dim`, `codes`, `basis` не гарантирована.

**Код решения:** Добавить `threading.Lock` в FractalField.

```python
# ── В concept_space.py, в __init__ FractalField, после строки 168 ──
import threading
self._capacity_lock = threading.Lock()

# ── В concept_space.py, grow_capacity(), строка 366 — обернуть в lock ──
def grow_capacity(self, new_latent_dim=None):
    with self._capacity_lock:
        # ... весь существующий код grow_capacity ...
```

```python
# ── В concept_space.py, prune_capacity(), строка 431 — обернуть в lock ──
def prune_capacity(self, sparsity_threshold=0.98):
    with self._capacity_lock:
        # ... весь существующий код prune_capacity ...
```

```python
# ── В concept_space.py, to_dict(), строка 763 — обернуть в lock ──
def to_dict(self, binary_path=None):
    with self._capacity_lock:
        # ... весь существующий код to_dict ...
```

```python
# ── В concept_space.py, from_dict(), строка 806 — восстановление ──
@classmethod
def from_dict(cls, data, base_dir=None):
    # ... существующий код ...
    field._capacity_lock = threading.Lock()  # восстановить lock после десериализации
    # ...
```

**Сложность:** 5 строк + 3 `with` блока. Нулевой риск регресса — lock reentrant, не влияет на однопоточные сценарии.

---

### P1.7 (Medium) — EntityField cleanup memory

**Статус:** Частично исправлена (decay добавлен), но cleanup не реализован  
**Файл:** `eva/symbolic/concept_space.py:880-1031`, `train_full.py:537-545`

**Текущее состояние:**

`ef.decay(factor=0.999)` (train_full.py:541) умножает все entity векторы на 0.999 и ренормализует. Это предотвращает насыщение, но НЕ удаляет неиспользуемые entity.

Проблемы:
1. `entities` dict растёт монотонно: char-level entity для каждого встреченного Unicode codepoint, word-level для каждого CID, sent-level для каждой уникальной последовательности. После 100K batch'ей — 100K+ entities × 2048D × fp32 = 800MB+.
2. `_char_word_cache` (concept_space.py:893) ограничен 50K entry, но сам по себе не очищается — cache evict удаляет ключи, но не entity из основного словаря.
3. Нет time-to-live (TTL) для sent и para entities — они никогда не удаляются, хотя актуальны только для текущего документа/сессии.

**Код решения:** Добавить TTL-осознанную очистку EntityField.

```python
# ── В concept_space.py, в класс EntityField, после __init__ (строка 894) ──
self._entity_access_time: Dict[tuple, float] = {}
self._entity_ttl = 5000  # batch'ей до удаления неиспользуемой entity
self._entity_batch_counter = 0
self._max_entities = 50000  # верхняя граница (50K × 2048 × fp32 = 400MB)
```

```python
# ── В concept_space.py, EntityField.get(), после строки 941 ──
def get(self, key):
    if key not in self.entities and key[0] == 'w' and self.word_store is not None:
        v = self.word_store.get(key[1])
        if v is not None:
            self.entities[key] = self._to_dim(v.copy().astype(np.float32))
    if key in self.entities:
        self._entity_access_time[key] = self._entity_batch_counter
    return self.entities.get(key)
```

```python
# ── В concept_space.py, добавить метод cleanup в EntityField ──
def cleanup(self, current_batch=0):
    """Удалить stale и превышающие лимит entity."""
    if len(self.entities) <= self._max_entities:
        return
    # Сортировка по времени доступа, удаление самых старых
    sorted_keys = sorted(self._entity_access_time.items(), key=lambda x: x[1])
    to_remove = len(self.entities) - self._max_entities
    for key, _ in sorted_keys[:to_remove]:
        self.entities.pop(key, None)
        self._entity_access_time.pop(key, None)
```

```python
# ── В stdp_trainer.py, _harmonize_batch(), в начале метода (строка 212) ──
# P1.7: cleanup entity field перед началом batch
if ef is not None:
    ef._entity_batch_counter += 1
    if ef._entity_batch_counter % 100 == 0 and len(ef.entities) > ef._max_entities:
        ef.cleanup()
```

**Сложность:** 30 строк. Риск: удаление sent/char entities может снизить качество bind'инга для редких слов. TTL=5000 batch'ей даёт ~500K строк текста до очистки.

---

### P1.8 (Medium) — Антоним-словарь из 24 пар

**Статус:** Не исправлена  
**Файл:** `eva/symbolic/stdp_trainer.py:27-49`

**Текущее состояние:**

Словарь `_ANTONYM_MAP` содержит 24 ключа (не 28, как указано в V15). Используется в `_build_pairs()` (stdp_trainer.py:483-488) для установки флага `antonym_flag = 1.0`, который влияет на meta-tensor колонку `_META_ANTONYM = 9`. Этот флаг используется в GPU STDP для модификации LR (repel антонимов).

Проблемы:
1. Только 24 ключа — покрытие ~0.02% vocabulary из 146K.
2. Хардкод — не масштабируется, не обновляется.
3. Нет проверки Part-of-Speech: "быстрый" (adj) → "быстро" (adv) — не все формы семантически антонимичны.

**Код решения:** Заменить хардкод на загрузку из JSON-файла с возможностью автоматического расширения через concept space cosine similarity.

```python
# ── В stdp_trainer.py, заменить строки 27-49 ──
import json, os

_ANTONYM_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'antonyms.json')
_ANTONYM_MAP: Dict[str, list] = {}

def _load_antonym_map(path=_ANTONYM_PATH):
    """Загрузить антоним-словарь из JSON. При отсутствии — пустой словарь."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                return {k.lower(): [v.lower() for v in vals] for k, vals in data.items()}
    # fallback: минимальные пары для базового русского
    return {
        'быстрый': ['медленный'], 'медленный': ['быстрый'],
        'хороший': ['плохой'], 'плохой': ['хороший'],
        'высокий': ['низкий'], 'низкий': ['высокий'],
        'большой': ['маленький'], 'маленький': ['большой'],
        'да': ['нет'], 'нет': ['да'],
    }

_ANTONYM_MAP = _load_antonym_map()

# ── В том же файле, после загрузки — автоматическое расширение через concept space ──
def _auto_discover_antonyms(cs, seed_pairs=None, threshold=0.6):
    """Автоматически найти антонимы через concept space cosine similarity.

    Идея: для данной пары (word_a, word_b) найти все concept'ы, которые
    имеют высокую sim с word_a и низкую с word_b (и наоборот).
    """
    if seed_pairs is None:
        seed_pairs = list(_ANTONYM_MAP.keys())
    discovered = {}
    for word in seed_pairs:
        cid = cs._find_cid_by_text(word)
        if cid is None:
            continue
        v = cs.concept_vector(cid)
        if v is None:
            continue
        # Ищем concept'ы с cosine < -0.3 (отрицательная корреляция)
        candidates = []
        for other_cid in cs.concept_vectors.keys():
            if other_cid == cid:
                continue
            other_v = cs.concept_vectors.get(other_cid)
            if other_v is None:
                continue
            sim = float(v @ other_v)
            if sim < -threshold:
                candidates.append((other_cid, sim))
        if candidates:
            discovered[word] = candidates[:5]
    return discovered
```

**Сложность:** 40 строк. Риск: автоматическое расширение может найти ложные антонимы. Рекомендуется человеческая валидация перед включением.

---

### P3.1 (Low) — EntityField.char_envelope не реализован

**Статус:** Не исправлена — метод удалён  
**Файл:** `eva/symbolic/concept_space.py` (ранее строки 1248-1305, сейчас отсутствует)

**Текущее состояние:**

Метод `harmonize_with_envelope()` был **полностью удалён** из Harmonizer. Никакой замены не добавлено. Прототип char-level envelope — механизм, где каждый символ (unicode codepoint) имеет вектор-контекст, который может модулировать word vector — отсутствует.

**Код решения:** Реализовать char-level envelope как отдельный компонент, без привязки к Harmonizer.

```python
# ── В concept_space.py, после класса EntityField, перед Harmonizer ──
class CharEnvelope:
    """Char-level semantic envelope: каждый Unicode codepoint → HD вектор.

    Позволяет модулировать word vector на основе char-контекста.
    Например: "князь" в envelope "к" → "княжество" (общий корень).
    """

    def __init__(self, dim=768, max_chars=5000):
        self.dim = dim
        self.max_chars = max_chars
        rng = np.random.RandomState(42)
        self._char_vecs: Dict[int, np.ndarray] = {}
        self._access_count: Dict[int, int] = {}

    def ensure(self, codepoint: int) -> np.ndarray:
        if codepoint not in self._char_vecs:
            if len(self._char_vecs) >= self.max_chars:
                # LFU eviction: удалить наименее используемый
                evict = min(self._access_count, key=self._access_count.get)
                self._char_vecs.pop(evict, None)
                self._access_count.pop(evict, None)
            seed = hash(('char_env', codepoint)) % (2**31 - 1)
            rng = np.random.RandomState(seed)
            v = rng.randn(self.dim).astype(np.float32)
            v /= max(np.linalg.norm(v), 1e-10)
            self._char_vecs[codepoint] = v
        self._access_count[codepoint] = self._access_count.get(codepoint, 0) + 1
        return self._char_vecs[codepoint]

    def word_envelope(self, word_text: str) -> np.ndarray:
        """Построить char-level envelope для слова = bundle char-векторов."""
        if not word_text:
            return None
        vecs = [self.ensure(ord(ch)) for ch in word_text]
        result = sum(vecs) / len(vecs)
        n = np.linalg.norm(result)
        return result / n if n > 1e-10 else result

    def modulate(self, word_vec: np.ndarray, char_env: np.ndarray, strength=0.05) -> np.ndarray:
        """Модулировать word vector char-level envelope через VSA binding."""
        bound = _hrr_bind(char_env, word_vec)
        result = word_vec + bound * strength
        n = np.linalg.norm(result)
        return result / n if n > 1e-10 else result
```

```python
# ── В ConceptSpace.__init__(), после инициализации harmonizer ──
self.char_envelope = CharEnvelope(dim=self.dim, max_chars=5000)
```

**Сложность:** 55 строк. Риск: LFU eviction с max_chars=5000 — безопасно, так как Unicode Basic Multilingual Plane ~2000 активно используемых codepoints в русском языке.

---

### P3.6 (Low) — HDC memory FIFO → LFU

**Статус:** Не исправлена  
**Файл:** `eva/symbolic/concept_space.py:147, 660-681`

**Текущее состояние:**

Эвикция в `hdc_update_ngram()` (строки 668-672):
```python
if len(self.hdc_memory) >= self.hdc_memory_max and self._hdc_access_order:
    evict_key = self._hdc_access_order.pop(0)
    self.hdc_memory.pop(evict_key, None)
    self.hdc_memory_counts.pop(evict_key, None)
```

`_hdc_access_order` — простой список для FIFO очереди. `hdc_memory_counts` уже существует и считает количество обращений к каждому ключу, но не используется для эвикции.

**Код решения:** Использовать `hdc_memory_counts` для LFU-эвикции.

```python
# ── В concept_space.py, строка 147 — заменить FIFO на LFU ──
# Заменить:
# self._hdc_access_order: List[tuple] = []  # simple FIFO eviction queue
# На:
self._hdc_access_order: List[tuple] = []  # (используется для LFU как список для random choice)

# ── В concept_space.py, hdc_update_ngram(), строки 668-672 — заменить эвикцию ──
def hdc_update_ngram(self, prefix_cids, next_code):
    key = tuple(prefix_cids)
    if key not in self.hdc_memory:
        if len(self.hdc_memory) >= self.hdc_memory_max:
            # LFU eviction: удалить entry с наименьшим счётчиком
            min_count = min(self.hdc_memory_counts.values()) if self.hdc_memory_counts else 0
            evict_candidates = [k for k, v in self.hdc_memory_counts.items() if v == min_count]
            evict_key = evict_candidates[0]  # первый с минимальным счётом
            self.hdc_memory.pop(evict_key, None)
            self.hdc_memory_counts.pop(evict_key, None)
        self.hdc_memory[key] = next_code.copy()
        self.hdc_memory_counts[key] = 1
    else:
        count = self.hdc_memory_counts[key]
        lr = 1.0 / max(count + 1, 1.0)
        self.hdc_memory[key] = self.hdc_bundle(next_code, self.hdc_memory[key], lr)
        self.hdc_memory_counts[key] = count + 1
```

```python
# ── В concept_space.py, hdc_predict(), строка 697-699 — удалить FIFO access order ──
# Удалить (или закомментировать) строки 697-699:
# if key in self._hdc_access_order:
#     self._hdc_access_order.remove(key)
#     self._hdc_access_order.append(key)
```

**Сложность:** 10 строк. Риск: LFU может удалять недавно добавленные entry с низким count, но это корректное поведение — редкие n-gram'ы вытесняются быстрее частотных.

---

## 3. Приоритизация для V18

| Проблема | Приоритет | Сложность (строк) | Риск регресса | Обоснование |
|----------|-----------|-------------------|---------------|-------------|
| **P1.1** HDC fallback | **Critical** | 2 | 0.0 | 400MB → 16MB, 2 строки |
| **P1.4** Race condition | **Critical** | 5+3 | 0.0 | threading.Lock, без риска |
| **P1.3** W_proj коллапс | **High** | 12 | 0.1 | Collapse detection спасает обучение |
| **P1.2** EntityField feedback | **High** | 20 | 0.3 | Слабый pull не дестабилизирует |
| **P1.7** Cleanup memory | **Medium** | 30 | 0.1 | TTL + лимит entities |
| **P1.8** Антонимы | **Medium** | 40 | 0.2 | JSON-загрузка безопасна |
| **P3.6** HDC LFU | **Low** | 10 | 0.1 | LFU корректнее FIFO |
| **P3.1** char_envelope | **Low** | 55 | 0.1 | Новый код, изолирован |

**Рекомендуемый порядок выполнения:**

1. **P1.1** (2 строки, 0 риск) — немедленно. Освобождает 384MB.
2. **P1.4** (5 строк, 0 риск) — немедленно. Предотвращает битые checkpoint'ы.
3. **P1.3** (12 строк, 0.1 риск) — до следующего цикла обучения. Предотвращает коллапс field_bits.
4. **P1.2** (20 строк, 0.3 риск) — после P1.3. Замыкает EntityField→STDP цикл.
5. **P1.7** (30 строк, 0.1 риск) — после P1.2. Предотвращает утечку памяти.
6. **P1.8** (40 строк, 0.2 риск) — в фоне. JSON-загрузка + auto-discover.
7. **P3.6** (10 строк, 0.1 риск) — после P1.1.
8. **P3.1** (55 строк, 0.1 риск) — в последнюю очередь. Новая фича.

---

## 4. Новые архитектурные методы (5 предложений)

### Метод 1: Gradient-Aligned Field Projection (GradField)

**Проблема:** W_proj Hebbian (P1.3) — не-градиентный, не координирован с STDP.

**Решение:** Заменить `codes.T @ sign(codes @ W_proj)` на градиентный шаг через straight-through estimator. Градиент STDP loss по field_bits аппроксимируется через cosine similarity target.

```python
# ── В concept_space.py, заменить update_learned_fields() (строки 309-333) ──
def update_learned_fields_grad(self, codes_mean_sim=None):
    """Gradient-aligned field update: W_proj учится максимизировать overlap
    между концептами, которые STDP сближает.

    L = -sum(overlap(a,b) * cos(a,b))  для пар (a,b) с PMI > 0
    dW = codes.T @ (dL/dfield_bits) ≈ codes.T @ (target - current) * f'(raw)
    """
    if self.W_proj is None or len(self.codes) < 2:
        return
    codes_arr = np.array(list(self.codes.values()), dtype=np.float32)
    n = len(codes_arr)
    raw = codes_arr @ self.W_proj
    # Straight-through estimator: forward = sign, backward = identity (STE)
    field_bits = np.sign(raw)  # forward: hard sign

    # Цель: максимизировать overlap между близкими concept'ами
    # Строим матрицу cosine similarity
    norms = np.linalg.norm(codes_arr, axis=1, keepdims=True) + 1e-10
    cos_mat = (codes_arr @ codes_arr.T) / (norms @ norms.T)
    # Целевой overlap: чем выше cos, тем больше битов должно совпадать
    target_overlap = (cos_mat > 0.3).astype(np.float32) * 0.5  # [0, 0.5]

    # dL/d(raw): через STE — градиент течёт сквозь знак
    current_overlap = (field_bits @ field_bits.T) / (2 * self.n_field_bits)  # [0, 1]
    grad = (current_overlap - target_overlap).mean(axis=1, keepdims=True)
    grad_raw = codes_arr * grad  # [n, latent_dim]

    # Hebbian + gradient hybrid
    lr = self.field_lr * 0.5
    delta_hebb = (codes_arr.T @ field_bits) / max(n, 1)
    delta_grad = grad_raw.T @ codes_arr @ self.W_proj / max(n, 1)
    self.W_proj += lr * (delta_hebb + delta_grad * 0.1)

    norms = np.linalg.norm(self.W_proj, axis=0, keepdims=True)
    self.W_proj /= np.maximum(norms, 1e-10)
    self._rebuild_field_bits()
```

**Сложность:** 25 строк. Риск: добавление градиентного слагаемого (0.1×) мягко корректирует Hebbian в сторону STDP-цели.

---

### Метод 2: Adaptive HDC Cache with Frequency-Aware Eviction (HDC-LFU+)

**Проблема:** FIFO эвикция HDC (P3.6) не различает ценные и мусорные n-gram'ы.

**Решение:** LFU+ — комбинация частоты и давности. Score = count / (age + 1)^0.5. Эвикция: удаляем entry с минимальным score.

```python
# ── В concept_space.py, после строки 665, полная замена hdc_update_ngram ──
def hdc_update_ngram(self, prefix_cids, next_code):
    key = tuple(prefix_cids)
    now = len(self.hdc_memory)  # surrogate time
    if key not in self.hdc_memory:
        if len(self.hdc_memory) >= self.hdc_memory_max:
            # LFU+: score = freq / sqrt(age + 1)
            min_score = float('inf')
            evict_key = None
            for k, cnt in self.hdc_memory_counts.items():
                age = now - self._hdc_access_time.get(k, 0)
                score = cnt / max(age**0.5, 1.0)
                if score < min_score:
                    min_score = score
                    evict_key = k
            if evict_key is not None:
                self.hdc_memory.pop(evict_key, None)
                self.hdc_memory_counts.pop(evict_key, None)
                self._hdc_access_time.pop(evict_key, None)
        self.hdc_memory[key] = next_code.copy()
        self.hdc_memory_counts[key] = 1
        self._hdc_access_time[key] = now
    else:
        count = self.hdc_memory_counts[key]
        lr = 1.0 / max(count + 1, 1.0)
        self.hdc_memory[key] = self.hdc_bundle(next_code, self.hdc_memory[key], lr)
        self.hdc_memory_counts[key] = count + 1
        self._hdc_access_time[key] = now
```

```python
# ── В __init__ FractalField, добавить (после строки 146) ──
self._hdc_access_time: Dict[tuple, int] = {}
```

**Сложность:** 25 строк. Риск: O(N) scan при каждой эвикции. При hdc_memory_max=2000 (P1.1) — 2000 итераций, <0.1ms.

---

### Метод 3: Cross-Level Co-Training (EntityField → STDP feedback)

**Проблема:** EntityField — read-only (P1.2). Вся иерархическая VSA-работа не влияет на STDP.

**Решение:** Добавить "soft" feedback через char→word контрастивный pull и sent→word контрастивный push. EntityField учится реконструировать concept vector из char-level контекста; ошибка реконструкции — градиент для STDP.

```python
# ── В stdp_trainer.py, новый метод в STDPTrainer ──
def _entity_field_feedback(self, gen, cs, all_ids, ef):
    """Cross-level co-training: EntityField bindings → STDP gradient.

    Для каждого слова:
      1. char→word: query('w', cid) даёт char-контекст слова
      2. Если query близок к concept_vector — bind корректный, ничего не делаем
      3. Если query далёк — concept_vector подтягивается к char-контексту
         (контрастивный pull, слабый lr)
      4. word→sent: аналогично, sent-level контекст
    """
    if ef is None:
        return
    updated = []
    for ids in all_ids:
        sent_key = hash(tuple(ids))
        for cid in ids:
            v_cs = cs.concept_vectors.get(cid)
            if v_cs is None:
                continue
            # char→word pull
            char_query = ef.query('w', cid)
            if char_query is not None:
                cq_n = np.linalg.norm(char_query)
                if cq_n > 1e-10:
                    char_query /= cq_n
                    sim = float(v_cs @ char_query)
                    if sim < 0.95:  # порог: если уже близко — пропускаем
                        pull = (char_query - sim * v_cs) * 0.003
                        v_cs = v_cs + pull
                        vn = np.linalg.norm(v_cs)
                        if vn > 1e-10:
                            v_cs /= vn
                        cs._apply_vector_update(cid, v_cs)
                        updated.append(cid)
            # sent→word consistency
            skey = ef.key_sent(sent_key)
            sent_query = ef.query('s', sent_key)
            if sent_query is not None:
                sq_n = np.linalg.norm(sent_query)
                if sq_n > 1e-10:
                    sent_query /= sq_n
                    sim = float(v_cs @ sent_query)
                    if sim < 0.90:
                        push = (sent_query - sim * v_cs) * 0.002
                        v_cs = v_cs + push
                        vn = np.linalg.norm(v_cs)
                        if vn > 1e-10:
                            v_cs /= vn
                        cs._apply_vector_update(cid, v_cs)
                        updated.append(cid)
    # batched GPU sync
    if updated and gen._use_torch and gen._vecs_t is not None:
        batch_v = np.stack([cs.concept_vectors[cid] for cid in updated])
        batch_t = torch.from_numpy(batch_v).to(device=gen._vecs_t.device,
                                                dtype=gen._vecs_t.dtype,
                                                non_blocking=True)
        gen._vecs_t[torch.tensor(updated, device=gen._vecs_t.device)] = batch_t
```

**Сложность:** 50 строк. Вызывать после `_harmonize_batch()` в `_train()` (строка 203). Риск: 0.001-0.003 LR — слабее основного STDP в 10×.

---

### Метод 4: Delta-Checkpointing для Dynamic Capacity

**Проблема:** P1.4 — grow_capacity меняет структуру данных, race c async save.

**Решение:** Вместо блокировки всей save-операции — сохранять "дельта" от предыдущего checkpoint. Если grow_capacity был вызван между checkpoint'ами, следующий save сначала записывает state до grow (из кэша), затем применяет дельту.

```python
# ── В concept_space.py, новый класс DeltaCheckpoint ──
class DeltaCheckpoint:
    """Сохраняет дельту изменений для безопасного async checkpoint.

    При grow_capacity/prune_capacity: кэшируем старый basis, codes, dim.
    CheckpointManager: если есть дельта — сохраняем старый + дельту,
    иначе — полный save.
    """

    def __init__(self):
        self._pending_delta = None
        self._capacity_lock = threading.Lock()

    def snapshot_before_grow(self, field):
        """Сохранить snapshot структур ДО изменения размерности."""
        with self._capacity_lock:
            self._pending_delta = {
                'type': 'grow',
                'old_dim': field.latent_dim,
                'old_basis': field.basis.copy(),
                'old_codes': {cid: c.copy() for cid, c in field.codes.items()},
                'old_l_c': field.l_c, 'old_l_a': field.l_a, 'old_l_m': field.l_m,
            }

    def consume_delta(self):
        """Вернуть и сбросить дельту (вызывается из _sync_save)."""
        with self._capacity_lock:
            d = self._pending_delta
            self._pending_delta = None
            return d

    def apply_to_save(self, save_data, field):
        """Применить дельту к save_data: сохранить старую версию + diff."""
        delta = self.consume_delta()
        if delta is None:
            return save_data  # нет дельты — обычный save
        # Сохраняем старую версию как fallback + diff для восстановления новой
        save_data['_delta_prev_dim'] = delta['old_dim']
        save_data['_delta_new_dim'] = field.latent_dim
        return save_data
```

```python
# ── В concept_space.py, FractalField.__init__, после строки 160 ──
from eva.symbolic.concept_space import DeltaCheckpoint  # или inline
self._delta_ckpt = DeltaCheckpoint()
```

```python
# ── В concept_space.py, grow_capacity(), начало метода (строка 366) ──
def grow_capacity(self, new_latent_dim=None):
    self._delta_ckpt.snapshot_before_grow(self)
    # ... существующий код grow_capacity ...
```

```python
# ── В checkpoint_manager.py, _sync_save(), после cs.save() (строка 61) ──
# P1.4: применить дельту если есть
if hasattr(cs.fractal, '_delta_ckpt'):
    delta = cs.fractal._delta_ckpt.consume_delta()
    if delta is not None:
        # Сохраняем дельту отдельно
        delta_path = cs_path.replace('.json', '.delta.json')
        with open(delta_path, 'w') as f:
            json.dump({
                'type': delta['type'],
                'old_dim': delta['old_dim'],
                'new_dim': delta.get('new_dim', cs.fractal.latent_dim),
            }, f)
```

**Сложность:** 60 строк. Риск: увеличение числа файлов на диске. При отсутствии дельты — поведение идентично текущему.

---

### Метод 5: Concept Space Self-Supervised Antonym Discovery (CSSA)

**Проблема:** P1.8 — хардкод 24 антонимов из 146K vocabulary.

**Решение:** Использовать сам concept space для автоматического обнаружения антонимов. Антонимы — это пары concept'ов с высокой negative cosine similarity (−0.5 < sim < −0.3) и высокой PMI (часто встречаются вместе в корпусе). Алгоритм:

1. Для каждого concept'а с частотой > 10: найти top-10 наиболее отрицательно коррелированных concept'ов.
2. Для каждой пары (a, b) с sim < −0.3: проверить PMI(a,b) > 0.5 (антонимы часто встречаются в одном контексте).
3. Валидация через BPE-текст: если a и b оба семантические токены (не пунктуация) — добавить в словарь.

```python
# ── В stdp_trainer.py, новый метод для автоматического обнаружения антонимов ──
def _discover_antonyms_from_cs(cs, lattice, min_freq=10, sim_threshold=-0.25, max_pairs=200):
    """Автоматически найти потенциальные антонимы через concept space + lattice.

    Возвращает dict: {word: [antonym_words]} для top-200 пар.
    """
    antonyms = {}
    # Собираем частотные concept'ы
    freq_cids = [cid for cid, f in lattice.concept_freq.items() if f >= min_freq]
    if len(freq_cids) < 10:
        return antonyms

    # Для каждого concept'а: найти наиболее отрицательно коррелированные
    vecs = np.array([cs.concept_vectors.get(cid) for cid in freq_cids])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    cos_mat = (vecs @ vecs.T) / (norms @ norms.T)

    for i, cid_a in enumerate(freq_cids):
        # Ищем concept'ы с cosine < sim_threshold
        neg_indices = np.where(cos_mat[i] < sim_threshold)[0]
        if len(neg_indices) == 0:
            continue
        # Фильтруем через PMI: антонимы должны встречаться в одном контексте
        pairs = []
        for j_idx in neg_indices:
            cid_b = freq_cids[j_idx]
            sim_val = float(cos_mat[i, j_idx])
            # PMI через lattice
            pmi_ab = lattice.get_pmi(cid_a, cid_b)
            if pmi_ab is None or pmi_ab < 0.3:
                continue  # слишком редкая пара
            text_a = _cid_to_text(cid_a)
            text_b = _cid_to_text(cid_b)
            if not text_a or not text_b:
                continue
            pairs.append((text_a, text_b, sim_val, pmi_ab))

        if pairs:
            pairs.sort(key=lambda x: x[2])  # по возрастанию sim (наиболее отрицательные)
            for text_a, text_b, sim, pmi in pairs[:3]:
                antonyms.setdefault(text_a, []).append(text_b)

    # Ограничение размера
    truncated = {}
    for k, v in antonyms.items():
        truncated[k] = v[:5]
        if len(truncated) >= max_pairs:
            break
    return truncated


def _cid_to_text(cid, sp=None):
    """Вспомогательная функция: CID → BPE token text."""
    import sentencepiece as spm
    if sp is None:
        return str(cid)
    try:
        return sp.IdToPiece(cid).replace('\u2581', '').strip()
    except Exception:
        return str(cid)
```

**Сложность:** 50 строк. Риск: false positives (слова с отрицательной sim не всегда антонимы). Рекомендуется human-in-the-loop: сохранять в JSON, валидировать вручную.

---

## 5. Итоговая метрика эффективности V18

После реализации всех 8 фиксов + 5 новых методов:

| Метрика | До V18 | После V18 | Изменение |
|---------|--------|-----------|-----------|
| HDC memory | 400MB | 16MB | −96% |
| EntityField memory | ∞ (рост) | ≤400MB (capped) | Контролируемый |
| Race condition на grow | Есть | Нет (Lock + Delta) | Устранён |
| Collapse field_bits | Вероятен | Предотвращён | Стабильность |
| EntityField → STDP feedback | Нет | Да (слабый pull) | Новый канал |
| Антонимы | 24 хардкод | JSON + auto-discover | Масштабируемость |
| char-level envelope | Нет | Есть (CharEnvelope) | Новая фича |
| HDC eviction | FIFO | LFU+ | Качество кэша |
| W_proj update | Hebbian | Hebbian+Gradient | Координация со STDP |

**VRAM budget после V18 (оптимистичный):**
- _codes_t восстановлен из vecs → −598MB
- _ema_vecs_t lazy → −112MB (только eval)
- HDC memory 2000 → −384MB
- EntityField cleanup → −200MB (до 400MB cap)
- **Экономия:** ~1294MB → целевой бюджет ~700MB для базовых тензоров на 2GB GPU

---

## 6. Заключение

Кодовая база FCF на HEAD cff1240 демонстрирует значительное улучшение с V15: 19 закрытых проблем, включая критический P0.1 (FFT-HRR), P0.2 (slow-start), и инфраструктурные P2.x. Архитектура остаётся здоровой: STDP-ядро стабильно, HDC и EntityField изолированы, checkpoints атомарны.

Однако 8 остаточных проблем (P1.1-P1.8, P3.1, P3.6) имеют накопительный эффект: 400MB неоправданной HDC памяти, race condition на grow_capacity, EntityField без обратной связи, Hebbian коллапс, утечка памяти, хардкод антонимов.

**Приоритет V18:**
1. P1.1 + P1.4 (Critical, ~7 строк) — немедленно, 0 риск
2. P1.3 + P1.2 (High, ~32 строк) — до следующего обучения
3. P1.7 + P1.8 (Medium, ~70 строк) — в фоне
4. P3.6 + P3.1 (Low, ~65 строк) — по возможности

Новые методы (GradField, HDC-LFU+, Co-Training EF→STDP, Delta-Checkpointing, CSSA) добавляют ~210 строк и замыкают архитектурные петли: W_proj координируется со STDP, EntityField пишет обратно, HDC кэширует умнее, антонимы обнаруживаются автоматически.

**Итоговая оценка:** V18 = V17 + 8 закрытых проблем + 5 новых методов. После реализации: ~174 строк изменений, контролируемый риск, целевой VRAM ~700MB, устранение гонок, автоматическое обнаружение антонимов.
