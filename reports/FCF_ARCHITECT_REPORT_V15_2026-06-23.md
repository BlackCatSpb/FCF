# FCF Архитектурный аудит V15

**Дата:** 2026-06-23  
**Версия кодовой базы:** HEAD 4178389 (V14 → V15)  
**Анализатор:** Architect-AI  
**GPU Target:** NVIDIA 2GB VRAM (GTX 1050/1650 class)

---

## 1. Executive Summary

Кодовая база FCF (Free Crystal Field) после V14 представляет собой гибридную систему, в которой нейросимволический STDP-движок (ядро) обрастает вспомогательными подсистемами — HDC/VSA n-gram памятью, learnable field projection (W_proj), 3-уровневым иерархическим sector index, EntityField с Harmonizer, CheckpointManager, Minesweeper cluster-potential. Фундаментальная архитектура остаётся здоровой: STDP-обучение на concept vectors (768D) через lattice PMI + contrastive + negative sampling — единственный путь обновления весов. Однако новые подсистемы вносят серьёзные риски для целевой среды (2GB GPU).

**Ключевые выводы:**

1. **P0: VRAM-коллапс.** 146K vocab × 768D × fp16 = 224MB для _vecs_t. С добавлением _codes_t (146K × 2048 × fp16 = 598MB), _mom_t (146K × 768 × bf16 = 112MB), _fb_t (146K × 64B × uint8 = 9MB), _fused_buf, _ce_t, _ema_vecs_t — суммарный бюджет ~1100MB только на тензоры. Harmonizer (EntityField, morphemes) добавляет ещё ~200-600MB в RAM/VRAM. На 2GB GPU это оставляет <300MB для CUDA context, batch buffers, workspace — критический дефицит.

2. **P1: Дублирование HDC vs SyntaxLattice.** `FractalField.hdc_predict()` и `SyntaxLattice.predict()` решают одну задачу (предсказание следующего токена по контексту) разными методами (VSA binding vs PMI-ngram). В `_branch()` HDC используется только как fallback при <3 кандидатах от lattice — это означает, что HDC-память не окупает свои 50K entry × 2048D × fp32 = 400MB overhead.

3. **P1: EntityField — избыточная сложность.** Четыре уровня (char ↔ morph ↔ word ↔ sent ↔ para) с VSA bind/unbind на каждое вхождение — O(4 × sent_len × batch_size) операций. На 146K vocabulary каждая sync_word — проекция 768D → 2048D через случайную матрицу (3.1M парам). При batch_size=32 это 300ms+ на batch только на EntityField.

4. **P2: W_proj Hebbian update конфликтует с STDP.** `update_learned_fields()` обновляет W_proj через outer product `code @ sign(code @ W_proj)`. Это **независимый** от STDP процесс, который меняет field_bits (а значит, field_gate в STDP) без координации с concept_error или градиентами STDP. Возможен режим, когда Hebbian update усиливает одни и те же гиперплоскости, делая field_gate неинформативным (все концепты попадают в один сектор).

5. **P2: Dynamic capacity (grow/prune) небезопасен для checkpoint resume.** `grow_capacity()` меняет `latent_dim`, `basis`, `codes`, `W_proj`, `_sector_W` — все одновременно. Если процесс прерван (Ctrl+C, OOM) во время `grow_capacity()`, структура `codes` оказывается в несогласованном состоянии: часть кодов имеет старую размерность, часть — новую. CheckpointManager сохраняет асинхронно — race condition между сохранением и `grow_capacity()`.

6. **P3: Концептуальные дыры.** Не реализовано: (a) char-level envelope из концепции EntityField (concept_space.py:1248-1305 — заглушка), (b) PARA-level привязка (sent→para bind не вызывается в _harmonize_batch), (c) decay для EntityField вызывается только в коде, но не в training loop.

---

## 2. Архитектурный анализ новых компонентов

### 2.1 HDC/VSA n-gram memory (FractalField)

**Реализация:** `FractalField.hdc_memory` (dict[tuple, ndarray]), `hdc_bind`, `hdc_permute`, `hdc_bundle`, `hdc_ngram_repr`, `hdc_predict`, `hdc_update_ngram`.

**Архитектурная роль:** "третий глаз" для предсказания — альтернатива SyntaxLattice (PMI) и STDP (vector similarity). Использует VSA binding: `ρ^{n-1}(w1) ⊙ ρ^{n-2}(w2) ⊙ ... ⊙ wn` для кодирования n-gram, unbind для запроса следующего токена.

**Оценка целостности:**
- HDC memory полностью изолирована от STDP — не конфликтует, но и не интегрирована. Единственная точка соединения — `hdc_predict` в `_branch()`, где она вызывается лишь при `len(syn_preds) < 3` (т.е. когда PMI-предсказание почти ничего не дало). Это "запасной выход", а не равноправный компонент.
- HDC использует `codes` (latent vectors, 2048D) как носители — это означает, что HDC-память чувствительна к `fluctuate_fractal()`, который меняет все коды. После fluctuate HDC-память становится мусором (bundled representations построены из старых кодов).
- `hdc_memory_max = 50000` — FIFO эвикция. При 146K vocabulary и batch=32 за 2000 batch'ей память полна, но полезных n-gram'ов может быть значительно меньше.
- **Проблема:** FIFO эвикция не различает полезные и мусорные n-gram'ы. Частотные n-gram'ы ("в", "на", "и") вытесняются так же, как редкие семантические пары.

**Метрика эффективности:** Предполагаемая доля в предсказаниях <1% (только при пустом lattice). Стоимость: 50K × 2048 × 4 байта = 400MB RAM + O(n) per batch на _update_hdc_ngrams.

### 2.2 Learnable field projection (W_proj)

**Реализация:** `FractalField.W_proj` (np.ndarray [latent_dim, n_field_bits]), `init_learned_fields`, `update_learned_fields`, `_rebuild_field_bits`.

**Архитектурная роль:** Замена статической octree field projection на обучаемую. Каждый столбец W_proj — гиперплоскость, field_bit — знак проекции кода на эту гиперплоскость.

**Оценка целостности:**
- **Hebbian update — вне STDP.** W_proj обновляется через `codes.T @ sign(codes @ W_proj)`. Это не-градиентный, не-контрастивный процесс. Он усиливает направления, в которых коды уже имеют ненулевую проекцию — положительная обратная связь.
- **Риск коллапса:** Если коды всех концептов сконцентрированы в одном регионе latent space (что типично при малом L1 λ и большом latent_dim), Hebbian update будет усиливать одни и те же гиперплоскости для всех концептов. Полевые биты перестанут различать концепты.
- **Размер:** W_proj [2048, 512] × fp32 = 4MB — незначительно на фоне остального.
- **Частота обновления:** `update_learned_fields` вызывается раз в эпоху (в `train_full.py:782-783`). Это означает, что field_bits обновляются редко, а STDP использует stale field_bits между эпохами.
- **Секторный индекс (3 уровня):** три независимые W_proj уровня (4, 10, 20 бит) — общая размерность 34 бита. `_rebuild_sector_index` итеративно обходит ВСЕ коды (146K) и вычисляет префикс для каждого — O(V × total_bits) = 5M операций. При каждом `_rebuild_field_bits` (т.е. при каждом Hebbian update). Не оптимизировано для частых вызовов.

### 2.3 EntityField + Harmonizer

**Реализация:** `EntityField` (concept_space.py:840-1004), `Harmonizer` (1006-1361), `_harmonize_batch` (stdp_trainer.py:209-329).

**Архитектурная роль:** Рекурсивное семантическое поле: char ↔ word ↔ sent ↔ para через VSA bind/unbind. Harmonizer — частный случай для morpheme ↔ word.

**Детальный анализ сложности:**

1. **EntityField.dim = 2048** (равен latent_dim, а не 768 dim). Каждый entity — 2048D fp32 = 8KB.
2. **sync_word** — при каждом вызове: проекция 768D → 2048D через `_proj` [2048, 768] = 6.3M операций. В _harmonize_batch синхронизируются ВСЕ cids из фокуса (146K в худшем случае).
3. **char ↔ word binding:** для каждого слова (BPE токена) — итерация по символам. Средний BPE токен ~4 символа. Для batch=32 × 25 токенов = 800 токенов × 4 символа = 3200 char-level bind'ингов. Каждый bind: 2 × (VSA multiply + normalize) = 2 × 2048 + normalization.
4. **sent binding:** для каждой уникальной последовательности — HDC ngram_repr от слов (2048D) + bind каждого слова к sent.
5. **Harmonizer:** dirty-слов может быть >10K. `harmonize()` делает до 5 итераций, каждая: compose_word (N морфем × 2 × 2048), unbind, error + backprop.

**Итоговая стоимость:** ~50-200ms на batch в зависимости от количества dirty-слов. На CPU (после OOM fallback) это может быть >2s per batch.

**Архитектурный вопрос:** EntityField.DIM = 2048 (latent_dim) vs ConceptSpace.DIM = 768. Почему? Проекция 768→2048 через случайную матрицу — дешёвый JL-стиль, но:
- `_to_dim` создаёт новую матрицу [2048, 768] при каждом изменении входной размерности (но вход всегда 768, так что один раз).
- Обратная проекция (из EntityField в codes) **не реализована**. EntityField — read-only consumer concept vectors. Он не влияет на STDP.

### 2.4 CheckpointManager

**Реализация:** `checkpoint_manager.py`, 127 строк. ThreadPoolExecutor с max_workers=1.

**Архитектурная роль:** Асинхронное сохранение состояния без блокировки training loop.

**Оценка безопасности:**
- **Race condition:** `save()` вызывается из training loop (train_full.py:461-466). Thread делает `cs.save()` и `lattice.save()` — обе операции читают CPU-структуры, которые могут быть модифицированы main thread'ом во время сохранения (особенно `fractal.codes`, `concept_vectors`). Нет блокировок.
- **GIL защита:** CPython GIL защищает отдельные операции чтения dict, но не целостность между cs.save и lattice.save. Если lattice изменится между ними — чекпоинт будет несогласован.
- **Cleanup:** `_cleanup_old()` удаляет файлы на основе `_saved_tags`. Если shutdown происходит во время cleanup — файлы могут быть удалены, пока main thread пытается их загрузить.
- **Тем не менее:** для однопоточного режима обучения (GIL + синхронные вызовы) риск минимален. Реальная проблема — `grow_capacity()` + async save.

### 2.5 Minesweeper cluster-potential system

**Реализация:** `_cluster_map`, `_cluster_potential`, `_update_cluster_potential`.

**Архитектурная роль:** Инвертированный LR modulation: высокий concept error → увеличение learning rate для кластера. "Minesweeper" — редкие/трудные концепты получают больше обучения.

**Оценка:**
- Инновационная идея: кластер = первый установленный field_bit. Это даёт 2048 кластеров.
- LR modulation: `target = 1.0 + (mean_ce - 0.5) × 0.4` — simple linear mapping.
- **Проблема:** `_cluster_potential` умножает LR в `_gpu_stdp_core` (line 707). Если cluster_potential варьируется от 0.8 до 1.2, это лишь ±20% — эффект может быть незаметен на фоне других модуляций (theta, PMI, field_gate).
- `_cluster_update_counter` увеличивается на каждый batch, но `_update_cluster_potential()` вызывается только каждый `_cluster_update_every=50` батчей.

---

## 3. Дублирование и конфликты

### 3.1 FractalField (HDC prediction) vs SyntaxLattice (PMI-based prediction)

| Аспект | HDC (FractalField) | SyntaxLattice |
|--------|-------------------|---------------|
| Механизм | VSA bind/bundle/unbind | PMI-weighted n-gram counts |
| Память | 50K entries × 2048D (400MB) | Словари: concept_freq, ngrams[2-4], skip2 |
| Скорость | O(V × D) per unbind | O(1) dict lookup per pair |
| Интеграция | Fallback при <3 lattice candidates | Основной источник _branch |
|Обновление| _update_hdc_ngrams (каждый batch) | lattice.update (каждый batch) |

**Вывод:** HDC-prediction — это дорогой и неточный дубликат SyntaxLattice. Lattice работает с дискретными счётчиками (PMI), HDC — с непрерывными векторами. Для предсказания следующего токена в BPE-пространстве счётчики PMI дают более стабильные результаты, чем VSA unbind (который страдает от шума в кодах после fluctuate).

**Рекомендация:** Сделать HDC не fallback при пустом lattice, а источникам дополнительных признаков для RRF в _branch (как graph_candidates). Это хотя бы окупит 400MB памяти.

### 3.2 W_proj Hebbian vs STDP subspace update

**Конфликт:** STDP через subspace update меняет коды (z_c, z_a, z_m) с разными LR для каждой подсистемы. W_proj Hebbian update меняет field_bits (которые влияют на field_gate, field_weight, contrastive objective). Возможный цикл:

1. STDP меняет код A → увеличивает компоненту в направлении гиперплоскости i
2. Hebbian update усиливает W_proj[:, i] (outter product code @ sign)
3. Усиленная гиперплоскость i теперь сильнее влияет на field_bits
4. Field_gate даёт больший вес парам, где оба концепта имеют бит i = 1
5. STDP ещё сильнее "затягивает" коды в это направление

**Результат:** коллапс field_bits — все коды имеют одинаковый паттерн битов. Field_gate перестаёт фильтровать.

### 3.3 EntityField vs Harmoner

**Дублирование:** Оба используют VSA bind/unbind с quasi-ортогональными role vectors. Harmonizer имеет 6 ролей (ROOT, PREFIX, SUFFIX, ENDING, WORD_POS, WORD_ROLE), EntityField — 5 (CHAR, MORPH, WORD, SENT, PARA). Harmonizer оперирует в 768D (dim), EntityField — в 2048D (latent_dim).

**Проблема:** Harmonizer и EntityField — два независимых VSA-движка. Harmonizer влияет на concept vectors (через `_apply_vector_update`), EntityField — нет. EntityField — читатель, Harmonizer — писатель. Разная размерность усложняет обмен информацией (нужна _to_dim проекция 768→2048).

---

## 4. Новые проблемы

### P0 — Критические (блокирующие для 2GB GPU)

**P0.1: VRAM budget превышен**

| Тензор | Размерность | Тип | VRAM |
|--------|------------|-----|------|
| _vecs_t | [146K, 768] | fp16 | 224 MB |
| _codes_t | [146K, 2048] | fp16 | 598 MB |
| _mom_t | [146K, 768] | bf16 | 112 MB |
| _ce_t | [146K] | fp32 | 0.6 MB |
| _fb_t | [146K, 64] | uint8 | 9 MB |
| _fused_buf | [4096, 769] | fp32 | 12 MB |
| _ema_vecs_t | [146K, 768] | bf16 | 112 MB |
| _basis_t | [2048, 768] | fp32 | 6 MB |
| CUDA context + workspace | — | — | ~300 MB |
| **Итого** | | | **~1374 MB** |
| Harmonizer morphemes (10K × 768 × fp32) | | | ~30 MB RAM |
| EntityField entities (50K × 2048 × fp32) | | | ~400 MB RAM |
| HDC memory (50K × 2048 × fp32) | | | ~400 MB RAM |

**Вывод:** на 2GB GPU остаётся ~600MB для batch buffers. `_gpu_stdp_core` создаёт временные тензоры: `pair_delta` (N × 768), `fused_src` (N × 768), `scatter_add` buffer. При N=32K pairs (batch=32, window=5) — это ещё ~150MB.

**P0.2: Async checkpoint + dynamic capacity race**

`grow_capacity()` и `prune_capacity()` меняют размерность `latent_dim`, `basis`, `codes`, `W_proj`, `_sector_W`. Если `CheckpointManager._sync_save` запущен в threading pool и читает `cs.fractal.codes` в момент, когда main thread выполняет `grow_capacity()`, сохранённый чекпоинт будет битым (или с partial new dims).

### P1 — Высокие

**P1.1: EntityField вызывает ~400MB дополнительной RAM**

EntityField.entities хранит все entity-векторы: chars (2000+), words (146K), sents (уникальные последовательности в batch), paras. В типичном сценарии ~30-40K entities. Каждый вектор 2048D fp32 = 8KB. 40K × 8KB = 320MB. На 2GB GPU это RAM, не VRAM, но при OOM fallback на CPU — это 25% от доступной системной памяти.

**P1.2: HDC memory не очищается после fluctuate**

`fluctuate_fractal()` меняет все latent codes. HDC memory (hdc_memory) содержит bundled representations, построенные из старых кодов. Они становятся невалидными. Нигде в `fluctuate_fractal()` или `_sync_after_fluctuate()` нет вызова `hdc_memory.clear()`.

**P1.3: Sector index rebuild — O(V × total_bits) при каждом Hebbian update**

`_rebuild_sector_index()` проходится по ВСЕМ 146K кодам и вычисляет `code @ hstack(W_levels)`. Это ~146K × 2048 × 34 FMA ≈ 10 GFLOP. При вызове раз в эпоху — приемлемо. При вызове каждые N batch'ей (если `field_lr` высокий) — заметная задержка.

### P2 — Средние

**P2.1: per-concept L1 lambda не сохраняется в чекпоинт**

`l1_lambda_per_cid`, `l1_density_window` — атрибуты FractalField. `to_dict()` сохраняет `codes` и `basis`, **но не** adaptive L1 state и density window. После resume: все per-concept L1 lambdas сбрасываются к глобальному `l1_lambda`. Адаптация начинается заново.

**P2.2: Antonym dictionary — хардкод вместо данных**

stdp_trainer.py:27-49 — 24 пары русских антонимов. Это не масштабируется. Нет механизма загрузки из файла или автоматической генерации.

**P2.3: Cluster potential не сохраняется в чекпоинт**

`_cluster_potential` и `_cluster_map` — runtime-кэш в CrystalGenerator. При resume они пересоздаются с нуля (через `_ensure_cluster_map` и reset `_cluster_potential = None`). Требуется 50 batch'ей для восстановления потенциала.

### P3 — Низкие

**P3.1: EntityField.char_envelope (envelope) не реализован**

`concept_space.py:1248-1305` — метод `harmonize_with_envelope`. Он принимает `envelope: dict`, но нигде не создаётся. char-level envelope — обещанная, но не реализованная фича.

**P3.2: PARA-level binding не вызывается**

В `_harmonize_batch` (stdp_trainer.py:258-302) есть code для sent→para bind ("if para boundaries exist"), но логика определения параграфов отсутствует. PARA-level всегда пропускается.

**P3.3: EntityField.decay не вызывается**

`EntityField.decay()` существует (concept_space.py:998-1003), но нигде не вызывается в training loop. Векторы в entities никогда не затухают — старые bindings накапливаются.

**P3.4: Не используется total_freq_cache в GPU path**

`_get_total_freq()` кэшируется в `_total_freq_cache`. GPU path пересоздаёт `_total_freq_t` при каждой синхронизации, но не использует кэш (сравните `_sync_freq_tensors` line 188 vs CPU path).

---

## 5. Рекомендации

### 5.1 VRAM оптимизация (P0.1 — критично)

1. **Убрать _codes_t как отдельный тензор.** Восстанавливать codes из vecs через `vecs @ basis.T` на GPU. Это добавит ~0.5ms per batch на matmul, но сэкономит 598MB VRAM. Vecs_t (224MB) + рекомпьютация дешевле, чем хранение обоих.

2. **Перевести _vecs_t в bf16** вместо fp16. bf16 имеет ту же экспоненту, что fp32 — меньше риск underflow при ERIC (error-based learning rate modulation). Потеря мантиссы не критична для 768D cosine similarity.

3. **Убрать HDC memory** (P1.2). Либо редуцировать до 1000 entries. Fallback при <3 lattice candidates не стоит 400MB.

4. **Сделать _ema_vecs_t опциональным** — только для eval режима. В training mode — None. Экономия 112MB.

5. **Итого:** _codes_t removed (-598MB) + _ema_vecs_t lazy (-112MB) + HDC memory reduced (-390MB) = ~710MB freed. Реальный бюджет: ~664MB для базовых тензоров. На 2GB GPU: 1360MB свободно для batch buffers и CUDA context.

### 5.2 W_proj безопасность (P2)

1. **Добавить ортогонализацию столбцов W_proj** после каждого Hebbian update (уже есть нормализация, но не ортогонализация). Это предотвратит коллапс гиперплоскостей.

2. **Перемешать знаки field_bits случайно** если `mean_overlap > 0.8` (80% концептов имеют одинаковый бит). Сбросить проблемные столбцы W_proj в случайное направление.

3. **Сделать Hebbian update опциональным** — `--no-hebbian-field` флаг. При нестабильном обучении Hebbian можно выключить.

### 5.3 Checkpoint безопасность (P0.2)

1. **Thread lock для grow/prune.** Использовать `threading.Lock` вокруг `grow_capacity()`/`prune_capacity()` и `CheckpointManager._sync_save()`.

2. **Pre/post validation:** перед сохранением проверять `len(codes) == len(fractal.codes)` и соответствие размерностей.

### 5.4 Упрощение EntityField (P1.1)

1. **EntityField.dim → 768** (совпадает с concept vector dim). Убрать _to_dim проекцию. VSA binding в 768D вместо 2048D снижает память в 2.7× (148MB вместо 400MB).

2. **Отложить sync_word** — обновлять EntityField только при чекпоинтах (каждые 5000 lines), а не при каждом batch.

3. **Убрать PARA-level** до реализации реальной параграфной логики.

---

## 6. Предложения новых методов (3-5)

### Метод 1: Gradient-Based Field Projection (замена Hebbian)

**Проблема:** W_proj Hebbian update — не-градиентный, не координирован с STDP.

**Решение:** Заменить Hebbian на градиентный метод через straight-through estimator (STE):
```
field_bit = sign(code @ W_proj)  # forward: hard sign
grad_W = code.T @ (dL/d(field_bit))  # backward: STE
W_proj -= lr * grad_W
```
Где `dL/d(field_bit)` — градиент от STDP loss по field_bit'ам, аппроксимированный через cosine similarity target:
```
L_field = -sum(overlap(a,b) * cos(vec_a, vec_b) for STDP pair (a,b))
```
Это свяжет field_projection с STDP-целью: field_bits будут оптимизироваться для максимизации overlap между концептами, которые STDP сближает.

**Сложность:** ~2 matmul per field update (так же, как Hebbian). Преимущество: field_bits становятся согласованы с STDP.

### Метод 2: Adaptive HDC Cache (вместо FIFO)

**Проблема:** HDC memory использует FIFO эвикцию, не различает частотные и редкие n-gram'ы.

**Решение:** Заменить на LFU (Least Frequently Used) с frequency-aware eviction:
```python
# При обновлении: увеличиваем счётчик, если ключ существует
self.hdc_freq[key] = self.hdc_freq.get(key, 0) + 1
# При эвикции: удаляем entry с (1 / freq + age_weight * time_since_access)
evict_score = 1.0 / (self.hdc_freq[key] + 1) + 0.01 * (now - self.hdc_access_time[key])
# Храним top-K по score (минимальный score → уходит)
```

Это гарантирует, что редкие семантические n-gram'ы (начало князь → великий) сохраняются дольше, чем частотные (а → также).

### Метод 3: Cross-Level Co-Training (EntityField → STDP feedback)

**Проблема:** EntityField — read-only потребитель concept vectors. Harmonizer может писать, EntityField — нет.

**Решение:** Добавить feedback от EntityField к STDP через контрастивные градиенты:
```python
# В _harmonize_batch: после char↔word binding
for cid in morph_cids:
    v_word = ef.get(ef.key_word(cid))
    v_char_query = ef.query('w', cid)  # unbind: superposition of chars
    if v_word is not None and v_char_query is not None:
        # Если unbind даёт вектор, отличный от concept_vector,
        # concept_vector должен подстроиться (pull)
        char_context = v_char_query / max(np.linalg.norm(v_char_query), 1e-10)
        sim = float(v_word @ char_context)
        grad = (char_context - sim * v_word) * 0.01  # weak pull
        cs._apply_vector_update(cid, v_word + grad)
```

Это замкнёт цикл: char-level контекст влияет на векторы концептов.

### Метод 4: Memory-Efficient Dynamic Capacity с Progressive Resizing

**Проблема:** `grow_capacity()` меняет latent_dim ×1.5 (step function), создаёт ~3GFLOP ре-ортогонализации.

**Решение:** Progressive resizing через внешние (external) dimension buckets:
- Вместо изменения `latent_dim` на месте, поддерживаем пул "отложенных" базисных векторов.
- При росте: не расширяем коды немедленно, а выделяем `extra_basis: ndarray [extra_n, dim]`.
- Compute_vector: `v = code @ basis + extra_code @ extra_basis` (concatenate).
- При чекпоинте: merge extra в основной basis и пересчитываем коды через SVD `[basis; extra_basis] → Q`.
- Преимущество: grow — O(n) (просто добавляем extra), merge — O(V × dim × extra_n) только при чекпоинте.

```python
# Grow: O(1) memory allocation
def grow_capacity_fast(self, extra_n=64):
    rng = np.random.RandomState(42 + self._capacity_growths)
    new_dims = rng.randn(extra_n, self.dim).astype(np.float32)
    residual = new_dims - new_dims @ self.basis.T @ self.basis
    Q_new, _ = np.linalg.qr(residual, mode='reduced')
    self.extra_basis = Q_new.astype(np.float32)
    self.extra_codes = {cid: np.zeros(extra_n) for cid in self.codes}
    self.latent_dim += extra_n  # logical size
```

### Метод 5: L1-Adaptive Sparsity через Gumbel-Softmax (вместо hand-tuned threshold)

**Проблема:** per-concept L1 lambda настраивается эвристически: `density > target → λ↑, density < target×0.5 → λ↓`. Это медленное ручное правило.

**Решение:** Заменить hard threshold на Gumbel-Softmax relaxation для выбора активных размерностей z_c:
```python
# Для каждого концепта z_c имеет n_candidates вероятностных масок
logits = self.gate_predictor(code)  # [n_candidates, 2] — active/inactive per dim
gates = F.gumbel_softmax(logits, tau=0.5, hard=False)[:, 1]  # soft binary mask
z_c = z_c * gates
# Loss: -lambda * gates.sum()  (sparsity pressure)
# + lambda_target * (gates.mean() - target_density)^2  (target density)
```

Gumbel-Softmax даёт дифференцируемые бинарные маски — можно обучать per-concept gates через градиентный спуск, а не эвристические правила.
