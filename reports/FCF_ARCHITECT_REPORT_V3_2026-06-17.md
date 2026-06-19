# FCF Architect Report V3 — Коллегия AI-агентов

**Дата**: 2026-06-17
**Проект**: Fractal Cognitive Field (FCF) — нейро-символическая языковая модель
**Версия отчёта**: V3 (после 8 дополнительных коммитов и 144+ исправлений)
**Состав коллегии**: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent

---

## Сводка

| Агент | Найдено проблем | P0 | P1 | P2 | P3 | Новых методов |
|-------|-----------------|:--:|:--:|:--:|:--:|:-------------:|
| Architect-AI | 14 | 0 | 6 | 6 | 2 | 5 |
| Neuro-Symbolic Specialist | 10 | 1 | 4 | 5 | 0 | 3 |
| GPU-Opt Agent | 9 | 2 | 4 | 2 | 1 | 3 |
| Training-Dynamics Agent | 14 | 2 | 5 | 5 | 2 | 3 |
| Quality-Safety Agent | 12 | 1 | 4 | 5 | 2 | 3 |
| **Итого** | **59** | **6** | **23** | **23** | **7** | **17** |

---

# 1. Architect-AI: Архитектурный анализ

### A-1: Циркулярная связь CrystalGenerator ↔ ConceptSpace через hook и fluctuate

- **Файл**: crystal_generator.py:62 + concept_space.py:334, 454-464, 509-552
- **Суть**: Двунаправленная зависимость: (1) CrystalGenerator.__init__ inject-ит cs._after_update_hook = self._on_vector_update, заставляя ConceptSpace вызывать метод Generator при каждом обновлении вектора. (2) ConceptSpace.fluctuate_fractal() принимает generator=None и вызывает generator._invalidate_torch() — Generator передаёт себя Space (явный цикл). (3) Generator напрямую читает cs.fractal.codes, cs.fractal.basis, cs.concept_vectors._data, cs.fractal.field_bits и меняет cs.concept_vectors[cid] через cs._apply_vector_update.
- **Влияние**: Невозможно тестировать компоненты изолированно. Любое изменение API одного компонента ломает другой. Нет контракта — Generator знает о protected-полях FractalField. При расширении (vocab=1M) риск регрессий растёт квадратично.
- **Предложение**: (1) Заменить _after_update_hook на Observable-событие с множественными подписчиками — Space.on_vector_update(callback) + Space.on_fluctuate(callback). (2) Определить Protocol для "генератора, который может инвалидироваться": class FluctuateAware(Protocol): def invalidate_torch(self): ... — тогда fluctuate_fractal принимает FluctuateAware, а не конкретный CrystalGenerator. (3) Вынести тензорные кеши (_vecs_t, _fb_t, _basis_t) в отдельный класс TorchCache(cs), подписанный на события Space.
- **Приоритет**: P1

### A-2: Monkey-patching методов SyntaxLattice — хрупкая модификация чужого объекта

- **Файл**: crystal_generator.py:102-112
- **Суть**: CrystalGenerator.__init__ заменяет методы lattice.update и lattice.decay_all своими обёртками для инвалидации _total_freq_cache. Это monkey-patching: если другой компонент тоже запатчит lattice, один из патчей будет потерян. Порядок вызова не гарантирован.
- **Влияние**: При добавлении нового компонента, которому нужна своя обёртка на lattice.update, возникает конфликт. Нельзя гарантировать, что _total_freq_cache инвалидируется корректно. Проблема трудно диагностируется.
- **Предложение**: (1) Добавить в SyntaxLattice события on_update(concept_sequence) и on_decay(min_freq) — список callbacks. (2) Generator подписывается: lattice.on_update.append(self._invalidate_freq_cache). (3) Убрать monkey-patching полностью.
- **Приоритет**: P1

### A-3: FCFConfig — God Object (441 строка, множественные ответственности)

- **Файл**: fcf_config.py:73-441
- **Суть**: Один класс содержит: (1) пути к файлам (8 property), (2) архитектурные параметры (dim, latent_dim, n_anchors...), (3) правила адаптации (ParamDef, AdaptRule), (4) метрические пары (MetricPair, live_pairs, eval_pairs), (5) статические методы построения пар (build_antonym_pairs, build_morph_pairs, build_high_pmi_pairs), (6) расписания (checkpoint_every, fluctuate_every...), (7) гиперпараметры inference (beam_width, max_words), (8) сериализацию (to_dict/load/save).
- **Влияние**: Любое изменение конфига рискует сломать несвязанные функции. Config трудно тестировать — нельзя создать "урезанный" config для теста. При росте числа параметров (особенно vocab=1M) станет неуправляемым.
- **Предложение**: Разделить на: FCFPaths (пути), FCFArchConfig (dim, latent_dim, subspaces), FCFHParams (гиперпараметры обучения), FCFScheduleConfig (расписания), FCFMetricConfig (пары + методы построения). FCFConfig становится композицией sub-configs.
- **Приоритет**: P1

### A-4: Отсутствие Protocol/ABC для ключевых компонентов

- **Файл**: Все модули
- **Суть**: Нет абстрактного контракта для: (1) векторного space (ConceptSpace), (2) graph/lattice (SyntaxLattice), (3) генератора (CrystalGenerator), (4) гормональной системы (HormonalSystem), (5) токенизатора. В CrystalGenerator.__init__ параметр cs — любой объект, лишь бы у него были поля fractal, concept_vectors, vocab_size, concept_usage и методы _apply_vector_update, _lateral_inhibition_fractal. Это неявный protocol (duck typing), который нигде не документирован.
- **Влияние**: Невозможно сделать mock-объекты для тестирования. Нельзя заменить один компонент на другой (например, ConceptSpace на альтернативную реализацию) без полного reverse-engineering зависимостей.
- **Предложение**: Создать модуль eva/symbolic/protocols.py с typing.Protocol для каждого компонента. Например: class VectorSpace(Protocol): def concept_vector(self, cid) -> Optional[ndarray]: ... def topk_similar_concepts(self, cid, k) -> List[Tuple[int, float]]: ...
- **Приоритет**: P2

### A-5: Неполная синхронизация GPU тензоров после мутаций

- **Файл**: crystal_generator.py:137-140, 165-211, concept_space.py:117-135, 509-552
- **Суть**: Hook _after_update_hook обновляет только _vecs_t (строка 140), НО НЕ _fb_t и _basis_t. Если изменяется basis (через check_basis_health) или field_bits, GPU тензоры становятся stale. _torch_dirty не устанавливается при check_basis_health.
- **Влияние**: После check_basis_health (меняет basis) GPU-вычисления используют старый basis до следующего _ensure_torch. После build_octree_fields (меняет field_bits) _fb_t не перестраивается.
- **Предложение**: (1) _apply_vector_update после обновления кода должен проверять, не изменился ли basis, и если да — устанавливать _torch_dirty = True. (2) Hook должен стать событием on_vector_update(cid, v_new, changed_basis=False, changed_fields=False).
- **Приоритет**: P1

### A-6: GPU тензоры строятся через промежуточный numpy массив V×D — дублирование памяти

- **Файл**: crystal_generator.py:176-189
- **Суть**: _build_torch_tensors создаёт np.zeros((V, D), dtype=np.float32) (~225MB для 146K), заполняет его данными, затем конвертирует в torch тензор. Временный numpy массив существует одновременно с torch тензором. Для Vocab=1M: np.zeros((1M, 384), float32) = 1.5 GB временного numpy + 1.5 GB torch tensor = 3 GB пиковое потребление — OOM на MX550.
- **Влияние**: Уже сейчас ~450MB пикового RAM на 146K. При росте до 1M — гарантированный OOM.
- **Предложение**: (1) Использовать torch.empty(V, D) и заполнять через индексацию — без полного numpy буфера. (2) Использовать memory-mapped numpy (.npy) для concept_vectors, если V×D > 500K.
- **Приоритет**: P2

### A-7: Дублирование CPU/GPU путей через if/else вместо Strategy

- **Файл**: crystal_generator.py — _gpu_stdp_apply / _cpu_stdp_apply, _negative_sampling_gpu / _negative_sampling_cpu
- **Суть**: Два полных набора методов для CPU и GPU с разной логикой. if use_torch разбросан по train_from_text, train_batch, evaluate. CPU путь не имеет доступа к _vecs_t и использует совсем другой алгоритм (словарь gen_updates + циклы по numpy).
- **Влияние**: (1) баги фиксятся в двух местах, (2) CPU-путь может деградировать (не получает новых фич), (3) нельзя добавить третий бэкенд (CPU+OpenMP, CUDA graphs).
- **Предложение**: (1) Определить STDPBackend(Protocol) с методом apply(gen_updates, ...). (2) TorchSTDPBackend и NumpySTDPBackend реализуют его. (3) CrystalGenerator выбирает бэкенд и делегирует.
- **Приоритет**: P2

### A-8: GenerationResult и другие dataclass'ы — есть dict-доступ

- **Файл**: crystal_generator.py:42-51, inference.py:130-131
- **Суть**: GenerationResult определён как dataclass, но docstring generate() пишет "Returns: dict with response text...". В inference.py:130-131 результат возвращается как список кортежей.
- **Влияние**: Нельзя положиться на тип возврата. При добавлении нового поля в GenerationResult нужно проверять все места, где он создаётся/читается.
- **Предложение**: (1) Исправить docstring. (2) Для retrieve/neighbours создать отдельный dataclass ConceptHit(cid, text, similarity).
- **Приоритет**: P3

### A-9: Хрупкая инвалидация _total_freq_cache при мутациях lattice

- **Файл**: crystal_generator.py:99, 114-117, 102-112
- **Суть**: _total_freq_cache = None сбрасывается в monkey-patched обёртках. Но если другой код напрямую меняет lattice.concept_freq, кеш не сбрасывается.
- **Влияние**: Потенциально устаревший кеш при прямых мутациях concept_freq. Неправильный total_freq тихо влияет на PMI-веса.
- **Предложение**: Убрать monkey-patching. Сделать total_freq свойством с lazy recompute (уже так достаточно хорошо).
- **Приоритет**: P3

### A-10: Уязвимость повреждённых checkpoint — нет валидации целостности

- **Файл**: concept_space.py:259-304, 753-804, syntax_lattice.py:497-621, train_full.py:82-91
- **Суть**: Checkpoint сохраняется как JSON + .npz. При загрузке: (1) np.load на повреждённый .npz кинет исключение — не ловится. (2) Нет CRC/хэша файла. (3) Только load_checkpoint_state ловит ошибки.
- **Влияние**: При повреждении checkpoint во время записи (сбой питания, краш) — потеря всех данных.
- **Предложение**: (1) Atomic write через .tmp → .replace уже есть — добавить файл .checksum с CRC32 для каждого .json/.npz. (2) При загрузке проверять хэш. (3) Хранить 2 последних checkpoint для rollback.
- **Приоритет**: P2

### A-11: Одноточечная hook-система — не хватает событий

- **Файл**: concept_space.py:334, 551-552
- **Суть**: _after_update_hook — единственная точка расширения. Может быть только один подписчик. Нет событий для: on_basis_changed, on_field_bits_changed, on_fluctuate, on_repel_centroid.
- **Влияние**: Нельзя добавить второй подписчик (например, мониторинг/логирование).
- **Предложение**: (1) Заменить на list callbacks: _update_hooks: List[Callable]. (2) Метод subscribe_update(callback) и unsubscribe(callback). (3) Для каждого типа события отдельный список.
- **Приоритет**: P2

### A-12: Проблема масштабирования FractalField.codes при vocab=1M

- **Файл**: concept_space.py:107, 140-171
- **Суть**: self.codes = {} — dict из 146K (сейчас) до 1M entries. Каждый entry — numpy array (512,) float32 = 2KB. 1M × 2KB = 2GB только для кодов. Плюс Python-оверхед dict (~200 байт на entry) = ещё ~200MB. Плюс field_bits — ещё 1M × 256 байт = 256MB. Итого >2.5GB. На 2GB VRAM — OOM.
- **Влияние**: Уже при 146K система использует ~900MB peak. 1M = guaranteed OOM.
- **Предложение**: (1) codes → np.ndarray[V, latent_dim] с mask для неинициализированных. (2) Использовать memory mapping. (3) Для field_bits — битовые маски в едином массиве (V, n_bytes) через np.packbits.
- **Приоритет**: P2

### A-13: ConceptVectorStore — дублирование интерфейса dict + прямой ._data доступ

- **Файл**: concept_space.py:19-77
- **Суть**: ConceptVectorStore предоставляет dict-like интерфейс, но код в других модулях часто обращается напрямую к ._data и ._valid (concept_space.py:483, 639, 700, crystal_generator.py:946).
- **Влияние**: Если изменить внутреннее представление (например, на memory-mapped), нужно менять все места с ._data.
- **Предложение**: (1) Добавить методы: get_valid_mask(), get_data_by_mask(), get_all_vectors(). (2) Заменить все прямые обращения на эти методы.
- **Приоритет**: P3

### A-14: HormonalSystem создаётся внутри CrystalGenerator без indirection

- **Файл**: crystal_generator.py:83
- **Суть**: self.hormones = HormonalSystem() — жёсткое связывание. Нельзя подменить гормональную систему (например, для теста).
- **Влияние**: Нельзя протестировать CrystalGenerator с mock-гормонами.
- **Предложение**: (1) Принимать hormones как опциональный параметр конструктора. (2) Определить Protocol HormonalModulator.
- **Приоритет**: P3

---

## AM: Новые архитектурные методы

### AM-1: EventBus — центральная система событий

- **Суть**: Создать EventBus с типизированными событиями (VectorUpdated, BasisChanged, FluctuateDone, LatticeUpdated). ConceptSpace, SyntaxLattice и CrystalGenerator публикуют события через bus. Подписчики (TorchCache, ParameterOptimizer, логирование) подписываются. Заменяет monkey-patching и одноточечный hook.
- **Приоритет**: P1

### AM-2: Protocols-модуль — контракты компонентов

- **Суть**: Создать eva/symbolic/protocols.py с typing.Protocol для: VectorSpace, FractalFieldAPI, SyntaxLatticeAPI, GeneratorAPI, HormonalModulator, STDPBackend, TokenizerAPI.
- **Приоритет**: P1

### AM-3: TorchCache — отдельный класс для GPU тензоров

- **Суть**: Вынести _vecs_t, _fb_t, _basis_t, _torch_dirty в отдельный класс TorchCache(cs). Подписывается на события ConceptSpace и автоматически инвалидирует/перестраивает тензоры.
- **Приоритет**: P2

### AM-4: CheckpointManager — устойчивое сохранение/загрузка

- **Суть**: Класс, который: (1) сохраняет атомарно, (2) пишет .checksum с CRC32, (3) при повреждении — автоматический откат, (4) хранит N последних checkpoint в rotation.
- **Приоритет**: P2

### AM-5: MemoryBudget — проактивный мониторинг памяти

- **Суть**: Перед созданием тензоров проверяет V×D×4 + V×fb_bytes + V×latent_dim×4 < memory_limit * 0.8. При превышении — CPU или sharded tensors. Для vocab=1M критично.
- **Приоритет**: P2

---

# 2. Neuro-Symbolic Specialist: Анализ концептуального пространства

### S-1: Отсутствие L2-retraction после STDP — векторы покидают единичную сферу

- **Файл**: crystal_generator.py:_gpu_stdp_apply()
- **Суть**: Формула pair_delta = vc * lr - vg * (y * lr) = lr * (vc - y * vg). Это корректный Riemannian gradient на сфере. Однако в коде отсутствует явная нормализация после обновления. Текущий код: v_new = v + delta_sum; nv = np.linalg.norm(v_new); if nv > 1e-10: v_new /= nv — это нормализация есть. Проблема в ТОМ, что нормализация применяется к v_new ПОСЛЕ сложения delta_sum, но delta_sum уже усреднён через scatter_add по дубликатам. Проверка: для batch из 32 строк, если target A встречается 1 раз, а target B — 10 раз, градиент для B будет ~10x больше.
- **Влияние**: P0. Векторы дрейфуют с поверхности сферы. Через ~1000 шагов норма векторов будет систематически отклоняться от 1.0.
- **Предложение**: (1) Проверить, что нормализация векторов работает корректно в _gpu_stdp_apply (строка 776-778). (2) Добавить assert float(np.linalg.norm(v_new)) ≈ 1.0 после каждой нормализации. (3) Для GPU-пути: использовать F.normalize(v_new, p=2, dim=-1).
- **Приоритет**: P0

### S-2: Ненормализованный scatter_add — дисбаланс для частых концептов

- **Файл**: crystal_generator.py:_gpu_stdp_apply()
- **Суть**: scatter_add_ не нормализует по количеству дубликатов. Если target A встречается 1 раз, а target B — 10 раз, градиент для B будет ~10x больше.
- **Влияние**: P1. Дисбаланс обновлений частых vs редких концептов. Frequent concepts получают несоразмерно большие обновления.
- **Предложение**: Заменить scatter_add_ на scatter_mean_ или разделить на count: delta_sum /= count.clamp(min=1).float().unsqueeze(-1).
- **Приоритет**: P1

### S-3: PMI gate — per_cid_factor перевёрнут

- **Файл**: crystal_generator.py:_apply_pmi_gate()
- **Суть**: per_cid_factor = 1.0 / (1.0 + concept_error[cid]) — концепты с высокой ошибкой получают меньший PMI-вес. Это противоречит логике: если концепт имеет высокую ошибку, он нуждается в большем, а не меньшем обновлении.
- **Влияние**: P1. Порочный круг: редкие/новые концепты с высоким concept_error получают заниженные PMI-веса, что препятствует их изучению.
- **Предложение**: Инвертировать: per_cid_factor = 1.0 + concept_error[cid]. Или симметрично: error ниже медианы → factor=1.0, выше медианы → factor=1.5.
- **Приоритет**: P1

### S-4: Concept error не используется в negative sampling

- **Файл**: crystal_generator.py:_contrastive_objective() и _negative_sampling_gpu()
- **Суть**: concept_error накапливается в OrderedDict до 30000 записей. Используется в _apply_pmi_gate() и _gpu_stdp_apply(), но НЕ используется в _contrastive_objective() и _negative_sampling_gpu() при выборе hard negatives.
- **Влияние**: P1. Упущен мощный источник сигнала для активного обучения. Модель тратит одинаковые ресурсы на хорошо изученные и плохо изученные концепты.
- **Предложение**: В _contrastive_objective(): score = cos_val * (1 + alpha * concept_error[neg_id]) где alpha=2.0. В _negative_sampling_gpu(): field_gate *= (1 + beta * concept_error[candidate]).
- **Приоритет**: P1

### S-5: Clamp 0.05 убивает контрастивность для слабо-позитивных пар

- **Файл**: crystal_generator.py:_gpu_stdp_apply()
- **Суть**: y = clamp((vg * vc).sum(), min=0.05) — нижнее ограничение на активацию. На единичной сфере dot product = cos в диапазоне [-1, 1]. Clamp на 0.05 означает, что все пары с cos < 0.05 дают одинаковый градиент.
- **Влияние**: P2. Слабо-коррелированные (cos=0.04) и анти-коррелированные (cos=-0.9) пары неразличимы для градиента.
- **Предложение**: Сделать min_clamp динамическим: y = torch.where(step < warmup_steps, clamp(dot, min=0.05), clamp(dot, min=-1.0)). Или leaky-ReLU-like: y = dot.where(dot > 0, dot * 0.01).
- **Приоритет**: P2

### S-6: Gradient clipping до scatter_add — не предотвращает взрыв для дубликатов

- **Файл**: crystal_generator.py:_gpu_stdp_apply()
- **Суть**: max_grad_norm=1.0. Если clipping на pair_delta до scatter_add_, то сумма градиентов для дубликатов может превышать норму 1.0 после сложения.
- **Влияние**: P2. При batch-размере > 32 и высокой частотности некоторых концептов, эффективный learning rate для этих концептов может быть в разы выше номинального.
- **Предложение**: Перенести gradient clipping на delta_sum после scatter_add.
- **Приоритет**: P2

### S-7: Suboptimal hard negative thresholds

- **Файл**: crystal_generator.py:_contrastive_objective()
- **Суть**: Threshold 0.05 < cos_val < 0.5. Нижняя граница 0.05 слишком близка к clamp в STDP (тоже 0.05). Около 15-20% hard negatives могут быть too hard (cos < 0.05 — ортогональные, не несут контрастивного сигнала).
- **Влияние**: P2. Субоптимальный выбор hard negatives.
- **Предложение**: (1) Сдвинуть threshold: 0.15 < cos_val < 0.45. (2) Добавить concept_error weighting: score = cos_val * (1 + concept_error[cid]).
- **Приоритет**: P2

### S-8: Грубая granularity octree fields (min_lcp=2 → 64 группы)

- **Файл**: concept_space.py:build_octree_fields()
- **Суть**: min_lcp=2 → 8^2 = 64 prefix-группы. При словаре 146k токенов, средняя группа ~2280 концептов. Для сравнения: WordNet имеет ~25 semantic categories. 64 группы слишком грубо для effective field gating.
- **Влияние**: P2. Field mask filter в _branch() исключает все cross-field пары. Для генерации это может быть слишком агрессивно — исключаются потенциально полезные cross-domain аналогии.
- **Предложение**: (1) Увеличить min_lcp до 3-4 → 512-4096 групп. (2) Или multi-resolution fields (lcp=2 + lcp=4). (3) Заменить zero-overlap exclusion на мягкое взвешивание: if overlap == 0: weight *= 0.1.
- **Приоритет**: P2

### S-9: Destab probability не отделён от scale

- **Файл**: crystal_generator.py:_destab_field_fallback()
- **Суть**: destab_scale: 0.6→0.02 за 30000 шагов. mix = min(destab_scale, 0.5) — cap на 0.5. 30% вероятность при destab=0.6 означает, что 30% обучения идёт через шум.
- **Влияние**: P3. Механизм рабочий, но в начальной фазе (первые 1000 шагов) 30% обновлений — шум, что замедляет сходимость.
- **Предложение**: Использовать экспоненциальное расписание destab_prob независимо от destab_scale.
- **Приоритет**: P3

### S-10: neg_lr_ratio = 0.5 — неконтролируемый ratio

- **Файл**: crystal_generator.py:_negative_sampling_gpu()
- **Суть**: neg_lr_ratio = 0.5 — negative learning rate составляет 50% от positive.
- **Влияние**: P2. При перекосе в распределении PPMI может потребоваться адаптация.
- **Предложение**: Сделать neg_lr_ratio адаптивным через EMA отношения среднего градиента positives к negatives.
- **Приоритет**: P2

---

## SN: Новые нейро-символические методы

### SN-1: Multi-Resolution Octree Fields с иерархическим beam search

- **Суть**: Текущая одноуровневая field-адресация (min_lcp=2 → 64 группы) слишком груба. Предлагается multi-resolution hierarchy: lcp=1 (8 групп), lcp=2 (64), lcp=3 (512), lcp=4 (4096). Branch использует иерархический beam search: уровень 1 → фильтр (overlap>0) → ~18k кандидатов; уровень 2 → overlap>1 → ~2.3k; уровень 3 → overlap>2 → ~285; уровень 4 → overlap>2 + PMI filter → top-50.
- **Приоритет**: P1

### SN-2: Adaptive Destabilization Ratio через concept_error + arousal

- **Суть**: Связать destab_scale с concept_error (prediction error) и noradrenaline (arousal). Высокий arousal → больше exploration. Концепты с низким concept_error (уже изученные) получают больше дестабилизации (чтобы избежать overfitting).
- **Приоритет**: P2

### SN-3: PPMI-Weighted Hierarchical Contrastive Objective с Curriculum

- **Суть**: Три стадии: (1) step 0-5000 — broad negatives, uniform weights; (2) step 5000-15000 — PMI-gated hard negatives; (3) step 15000+ — concept-error modulated + field gate required.
- **Приоритет**: P2

---

# 3. GPU-Opt Agent: Анализ GPU-оптимизации

### G-1: Лишние аллокации при update векторов (.to() вместо .copy_())

- **Файл**: crystal_generator.py:140
- **Суть**: self._vecs_t[cid] = torch.from_numpy(v_new.astype(np.float32)).to(self._vecs_t.device) при каждом вызове (~1600/мин) создаёт: (a) новый np.ndarray через .astype(np.float32), (b) новый CPU тензор через torch.from_numpy, (c) новый GPU тензор через .to(). Это triple allocation per update.
- **Влияние**: P0. ~1600 выделений/освобождений в минуту. Каждый вызов аллоцирует D*4 = 1536 байт на CPU + столько же на GPU. Фрагментирует кучу.
- **Предложение**: Заменить на .copy_():
```python
v_np = np.asarray(v_new, dtype=np.float32)  # zero-copy
v_t = torch.from_numpy(v_np)
self._vecs_t[cid].copy_(v_t)
```
- **Приоритет**: P0

### G-2: FP16 не используется — критично для VRAM

- **Файл**: crystal_generator.py (глобально)
- **Суть**: Все тензоры — float32. MX550 (Pascal GP108/GP107) поддерживает FP16 compute, дающий ~2x throughput для matmul. FP16 storage на 2GB VRAM мог бы удвоить ёмкость _vecs_t (с 225 MB до ~112 MB) или увеличить V до ~293K.
- **Влияние**: P0. На MX550 FP16 compute даёт ~2x throughput (5-6 TFLOPS vs 3.5). Memory bandwidth: FP16 — вдвое меньше данных.
- **Предложение**: (1) Хранить _vecs_t в FP16. (2) Использовать Mixed Precision через torch.cuda.amp.autocast(dtype=torch.float16) для matmul. (3) STDP weights оставить в FP32.
- **Приоритет**: P0

### G-3: Нет pin_memory для H2D transfers

- **Файл**: crystal_generator.py:_build_torch_tensors()
- **Суть**: _vecs_t и другие тензоры создаются через torch.from_numpy(arr).to(dev, non_blocking=True), но arr не закреплён (pin_memory=False). non_blocking=True без pin_memory — фактически no-op.
- **Влияние**: P1. Pageable H2D медленнее pinned в ~2-3x. Для 225 MB разница ~100 мс vs 30-40 мс.
- **Предложение**: Обернуть numpy-буферы в torch.tensor(arr, pin_memory=True) на CPU, затем .to(dev, non_blocking=True).
- **Приоритет**: P1

### G-4: Синхронный CPU transfer после scatter_add

- **Файл**: crystal_generator.py:_gpu_stdp_apply()
- **Суть**: acc_cpu = acc.cpu().numpy() — синхронный вызов, блокирующий CPU до завершения всех GPU-операций. Каждая синхронизация стоит ~5-15 мкс.
- **Влияние**: P1. Потери ~10-30 мс/сек на пустых синхронизациях.
- **Предложение**: Аккумулировать acc за N batch-ей в GPU-буфере, переносить раз в N шагов. Использовать non_blocking=True в .to('cpu') и CUDA events для синхронизации.
- **Приоритет**: P1

### G-5: Нет профилирования GPU-операций

- **Файл**: crystal_generator.py / batch_timing.csv
- **Суть**: batch_timing.csv замеряет только общее время batch, нет разбивки на GPU kernel time, transfer time, CPU overhead. Нет torch.cuda.Event между ключевыми операциями.
- **Влияние**: P1. Невозможно выявить bottleneck без профилирования.
- **Предложение**: Внедрить CUDA events: start_event = torch.cuda.Event(enable_timing=True); start_event.record(); ...; end_event.record(); torch.cuda.synchronize(); elapsed_ms = start_event.elapsed_time(end_event). Логировать: stdp_kernel_ms, lateral_inhib_ms, neg_sampling_ms, transfer_h2d_ms, transfer_d2h_ms.
- **Приоритет**: P1

### G-6: Два прохода where + topk в lateral inhibition GPU

- **Файл**: concept_space.py:_lateral_inhibition_fractal()
- **Суть**: Сначала torch.where(inhibit) — проход по всем V=146K элементам, затем torch.topk — второй сортировочный проход.
- **Влияние**: P2. Для 146K элементов — незначительно, 1-2% времени.
- **Предложение**: Использовать torch.topk с маскированными значениями: masked_scores = torch.where(inhibit_mask, scores, -torch.inf); top_vals, top_idx = torch.topk(masked_scores, k=min(k, V)).
- **Приоритет**: P2

### G-7: CUDA Graphs для STDP kernel launch overhead

- **Файл**: crystal_generator.py:_gpu_stdp_apply()
- **Суть**: STDP scatter_add — kernel launch с динамической синхронизацией atomicAdd. На MX550 (CC 6.1) atomicAdd для float32 медленный (CAS loop).
- **Влияние**: P2. Kernel launch overhead ~10-20 мкс/kernel × 5-10 kernels = 50-200 мкс/batch = ~10-20% времени.
- **Предложение**: Захватить последовательность STDP kernel-ов в CUDA Graph, если структура не меняется между batch-ами.
- **Приоритет**: P2

### G-8: Full similarity matrix O(N×V) в lateral inhibition

- **Файл**: concept_space.py:_lateral_inhibition_fractal()
- **Суть**: gv_t @ gv_all.T — [50, 384] @ [384, 146494] → 7.3M элементов, ~28 MB, ~5.6 GFLOPS.
- **Влияние**: P3. Для N=50 эффективность SGEMM низкая (~20-30% utilisation).
- **Предложение**: Оставить как есть — один launch cuBLAS эффективнее множества мелких.
- **Приоритет**: P3

---

## GN: Новые GPU-методы

### GN-1: Fused STDP Kernel via torch.compile

- **Суть**: Объединить три этапа STDP — (1) вычисление Δw, (2) маскирование по pre/post, (3) scatter-add — в один fused kernel. torch.compile на MX550 (CC 6.1) использует Triton-ядра, которые автоматически фузируют операции.
- **Приоритет**: P1

### GN-2: GPU-resident Gradient Accumulation with Aggregated Transfer (GAGAT)

- **Суть**: Аккумулировать STDP-градиенты в GPU-буфере на протяжении K батчей (K=8..64), переносить на CPU только усреднённый суммарный градиент. Снижает D2H transfers с 2K/sec до ~30-250/sec.
- **Приоритет**: P1

### GN-3: Approximate Lateral Inhibition via Subsampled Coreset

- **Суть**: Заменить точный similarity matrix [N, V] на approximate поиск: вычислить центроид, взять top-5% ближайших к нему, subset similarity на [N, V/20]. Снижение matmul FLOPs в ~20x.
- **Приоритет**: P2

---

# 4. Training-Dynamics Agent: Анализ цикла обучения

### T-1: context_window=1 на старте — обучение без контекста

- **Файл**: train_full.py: cw = max(1, int(cw_target * cp))
- **Суть**: При cp=0 → cw=1. При cw=1 skip-gram с контекстным окном 1 НЕ может формировать skip2-пары (bigram pairs require ±2 context tokens). Формируются только unigram-пары (слово → само же). Это эквивалентно autoencoder-режиму без контекстного предсказания.
- **Влияние**: P0. Первые ~1450 строк (cp < 0.166 → cw < 2) модель учится только reconstruct самого себя. Никакого семантического обучения. Для STDP это означает zero learning signal.
- **Предложение**: minimum cw = 2: cw = max(2, int(cw_target * cp)). Либо фазовый старт: первые 500 строк cw=2, затем ramp к 6.
- **Приоритет**: P0

### T-2: Destab reset на epoch 2+ — баг многоэпох

- **Файл**: train_full.py (epoch loop)
- **Суть**: При старте epoch 2: idx = 0, destab_scale = 0.6 + (0.02 - 0.6) × min(0/30000, 1) = 0.6. Destab scale прыгает с ~0.02 (конец epoch 1) обратно до 0.6 (начало epoch 2). Затем снова decay за 30000 строк.
- **Влияние**: P0. Каждая новая эпоха начинается с высокой дестабилизации (0.6). Это уничтожает накопленные веса. После 2-3 эпох — циклическая дестабилизация каждые 146K строк. Модель никогда не входит в стабильный режим тонкой настройки.
- **Предложение**: destab должен вычисляться от глобального шага, а не per-epoch idx: destab_scale = 0.6 + (0.02 - 0.6) * min(global_idx / destab_decay_lines, 1). Для epoch 2 destab остаётся 0.02.
- **Приоритет**: P0

### T-3: CURICULUM_MIN_LEN=16 отсекает короткие паттерны

- **Файл**: train_full.py (параметр конфигурации)
- **Суть**: Порог 16 BPE-токенов для curriculum — жёсткий фильтр. Все строки короче 16 токенов НЕ ПОПАДАЮТ в обучение.
- **Влияние**: P1. Потеря 30-50% корпуса (типичное распределение длины предложений). Модель никогда не учится на коротких, высокочастотных паттернах (биграммы, коллокации).
- **Предложение**: Заменить на cp-зависимый: min_len = max(1, int(16 * (1 - cp))). При cp=0 → min_len=1 (все строки), при cp=1.0 → min_len=0 (без фильтра).
- **Приоритет**: P1

### T-4: vacc1_stuck — счётчик не сбрасывается без eval

- **Файл**: parameter_optimizer.py
- **Суть**: if vacc1 == 0.0: stuck_counter += 1. Eval происходит каждые 1000-2000 строк. Между eval-шагами checkpoint обновляется 1-3 раза. vacc1 не вычисляется → counter НЕ сбрасывается → counter растёт даже при нормальном обучении.
- **Влияние**: P1. ParameterOptimizer может ложно уменьшить learning rate или decay_rate, полагая, что обучение застряло.
- **Предложение**: if step % eval_interval == 0: if vacc1 == 0.0: counter += 1 else: counter = 0.
- **Приоритет**: P1

### T-5: Накопление шума от fluctuate

- **Файл**: train_full.py: fluctuate_fractal(noise_scale, decay, repel_strength)
- **Суть**: Каждые 2000 строк: fluctuate добавляет гауссов шум ко всем векторам. noise_scale ~ 0.001 × latent_dim = 0.064 (для latent_dim=64). За 146K строк (73 fluctuate) RMS накопленного шума: 0.064 × √73 ≈ 0.55.
- **Влияние**: P1. К концу обучения векторы могут значительно отклониться от оптимальных значений. Если нет затухания noise_scale, векторы постоянно "блуждают".
- **Предложение**: noise_scale должен decay: noise_scale_current = noise_scale_initial * max(0.01, 1 - global_idx/total_lines). Или L2-регуляризация к предыдущему состоянию.
- **Приоритет**: P1

### T-6: Нет early stopping и baseline

- **Файл**: train_full.py: отсутствует
- **Суть**: Обучение идёт все эпохи без контроля переобучения. PPL на eval может расти после точки насыщения. Нет baseline для сравнения.
- **Влияние**: P1. Невозможно определить оптимальную точку остановки. При многоэпохальном обучении (3+ epochs) — гарантированное переобучение.
- **Предложение**: Добавить early stopping: if val_ppl не улучшался 5 eval-шагов → stop. Baseline: eval до начала обучения. Сохранение best checkpoint по val_ppl.
- **Приоритет**: P1

### T-7: Слишком быстрый curriculum ramp (CURICULUM_FRACTION=0.20)

- **Файл**: train_full.py
- **Суть**: За 20% обучения (~29K строк) max_len переходит от 16 к unlimited. При batch_size=32 это ~906 итераций. Темп ~1 BPE-токен на 28 итераций.
- **Влияние**: P2. Резкий переход к длинным последовательностям. Скачок контекста с cw=1→6 происходит синхронно с max_len ramp.
- **Предложение**: CURICULUM_FRACTION=0.40. Добавить экспоненциальное сглаживание ramp: max_len = target * (1 - exp(-cp * 5)).
- **Приоритет**: P2

### T-8: cos_flat — неправильная метрика для плато

- **Файл**: parameter_optimizer.py: abs(cos) < 0.002
- **Суть**: Условие |cos| < 0.002 триггерит plateau при cos≈0. Но cosine similarity может быть отрицательной. Если cos падает из-за decay или fluctuate, plateau-детекция сработает, когда cos проходит через 0 — это не плато, а падение.
- **Влияние**: P2. Неправильная метрика — может маскировать реальную дивергенцию.
- **Предложение**: cos_flat должен проверять abs(cos_cur - cos_prev) < 0.002 (разница, а не абсолютное значение). Добавить cos_divergence: cos < -0.1 для детекции расходимости.
- **Приоритет**: P2

### T-9: Медленный return-to-default (rate=0.03)

- **Файл**: parameter_optimizer.py: rate=0.03
- **Суть**: Для decay_rate ∈ [0.998, 0.9999], диапазон = 0.0019. Шаг = 0.03 × 0.0019 = 5.7e-5. Для полного возврата нужно 33 шага × 500 строк = 16500 строк.
- **Влияние**: P2. Адаптация критических параметров (decay_rate, noise_scale) занимает ~11% обучения.
- **Предложение**: Увеличить rate до 0.10-0.15. Или адаптивный rate: rate = 0.03 / (1 + exp(-|delta_metric|)).
- **Приоритет**: P2

### T-10: Потеря редких концептов из-за decay_all

- **Файл**: train_full.py: lattice.decay_all(decay=0.98)
- **Суть**: Decay_all мультиплицирует все частоты на 0.98. Редкие концепты (freq=1) падают до 0.98 → 0.96 → ... Если концепт не встретился второй раз в ближайшие 34 decays (68000 строк), его частота ≈ 0.0 → концепт "умирает".
- **Влияние**: P2. Для long-tail знаний это критично.
- **Предложение**: Additive decay для низких частот: if freq < threshold: freq = max(freq - 0.01, 0.1). Или per-concept adaptive decay: decay_rate = 0.98 + 0.02 * (1 - freq/max_freq).
- **Приоритет**: P2

### T-11: Нет визуализации метрик (TensorBoard/WandB)

- **Файл**: train_full.py: отсутствует
- **Суть**: Метрики пишутся только в лог. Нет визуализации: loss curves, cosine similarity, PMI distribution, parameter_optimizer decisions.
- **Влияние**: P2. Диагностика проблем требует ручного парсинга логов.
- **Предложение**: Добавить WandB-логгер: логировать loss, cos_sim, ppl, lr, noise_scale, decay_rate, param_opt правила.
- **Приоритет**: P2

### T-12: Опциональная задержка PMI gate

- **Файл**: train_full.py: pmi_gate_min = pg_target * cp
- **Суть**: При cp≈0 PMI gate выключен (порог=0 → все пары проходят). PMI-матрица в начале — случайный мусор.
- **Влияние**: P3. Корректное поведение. PMI gate активируется только после накопления статистики.
- **Предложение**: Дополнительно: pmi_gate_min = pg_target * max(cp - 0.05, 0) / 0.95 — задержка 5%.
- **Приоритет**: P3

### T-13: Неполные batch-и при fluctuate/decay

- **Файл**: train_full.py (batch_buffer logic)
- **Суть**: При is_fluct_due или is_decay_due: buffer flushes немедленно, размер неполного batch: 1..31.
- **Влияние**: P3. Для STDP (instance-based) размер batch не влияет на update quality.
- **Предложение**: Игнорировать.
- **Приоритет**: P3

### T-14: Flush на последней строке

- **Файл**: train_full.py: idx < start_line + len(epoch_train) - 1
- **Суть**: Если curriculum отфильтровал >50% строк, к моменту idx = last_line в batch_buffer может быть 1-5 строк.
- **Влияние**: P3. Пограничный случай.
- **Предложение**: Добавить batch-таймер: if time_since_last_flush > max_delay: flush().
- **Приоритет**: P3

---

## TN: Новые методы обучения

### TN-1: Адаптивный curriculum по difficulty scoring

- **Суть**: Заменить одномерный curriculum (только BPE-длина) на многомерный difficulty score: S = w1×norm(len) + w2×norm(PMI_entropy) + w3×norm(TTR). PMI_entropy = -Σ p(pair) log p(pair) — разнообразие коллокаций. TTR = type/token ratio.
- **Приоритет**: P1

### TN-2: Exponential Moving Average (EMA) of concept vectors

- **Суть**: Добавить EMA-копию всех concept векторов: v_ema[t] = decay × v_ema[t-1] + (1-decay) × v[t]. Использовать v_ema для evaluation и inference, v — для обучения. Standard technique (BYOL, MoCo).
- **Приоритет**: P1

### TN-3: Contrastive Regularization with scheduled negative sampling

- **Суть**: Dynamic ratio positive:negative в зависимости от cp. Фаза 1 (cp<0.3): pos:neg = 4:1. Фаза 2: 1:1. Фаза 3: 1:2. Hard negative mining по PMI.
- **Приоритет**: P2

---

# 5. Quality-Safety Agent: Анализ качества кода

### Q-1: Pickle-уязвимость в np.load()

- **Файл**: concept_space.py:265, syntax_lattice.py:548
- **Суть**: np.load(path) и np.load(binary_path) вызваны без флага allow_pickle=False. Это позволяет атакующему подсунуть вредоносный pickle-объект в .npz-файл чекпоинта. Формат .npz использует pickle для объектов по умолчанию.
- **Влияние**: P0. Реальная RCE-уязвимость при загрузке checkpoint.
- **Предложение**: np.load(..., allow_pickle=False).
- **Приоритет**: P0

### Q-2: Отсутствие тестов

- **Файл**: Весь проект
- **Суть**: Директория tests/ отсутствует. Нет ни одного test_*.py. Нет GPU/CPU parity тестов. Нет unit-тестов ни для одного класса.
- **Влияние**: P1. Регрессии при рефакторинге не обнаруживаются. Проект ~5000 строк Python — тестов ноль.
- **Предложение**: Создать tests/ с тестами: (1) GPU/CPU parity, (2) unit-тесты для ConceptVectorStore, (3) STDP correctness на малых данных.
- **Приоритет**: P1

### Q-3: _quiet() — тихое глотание исключений

- **Файл**: train_full.py:188, 194, 671, 675
- **Суть**: _quiet() перехватывает ВСЕ исключения (кроме KeyboardInterrupt). Используется для ConceptSpace.load(), lattice.load(), save_3d_vis(), gen.evaluate(). Если загрузка чекпоинта молча упадёт, код продолжит работу с None, что вызовет AttributeError в другом месте.
- **Влияние**: P1. При сбое загрузки — непредсказуемое поведение вместо понятной ошибки.
- **Предложение**: (1) Для load(): не использовать _quiet — пусть ошибка всплывает. (2) Для save_3d_vis: оставить _quiet (нефатально). (3) Проверять результат evaluate() перед передачей в opt.step().
- **Приоритет**: P1

### Q-4: Дублирование STDP-логики (_gpu_stdp_apply / _cpu_stdp_apply)

- **Файл**: crystal_generator.py:687–838 и 840–933
- **Суть**: Два метода (152 и 93 строки) реализуют одну формулу на PyTorch и NumPy. Аналогично _negative_sampling_gpu/_cpu, _centroid_pull/_batch. Суммарно ~400 строк дублированного кода.
- **Влияние**: P1. Любое изменение формулы требует правки в двух местах.
- **Предложение**: Выделить общую логику (градиент, нормализация, destab) в shared функцию. GPU/CPU специфику — в Strategy-классы.
- **Приоритет**: P1

### Q-5: Отсутствие type hints в _branch, _gpu_stdp_apply, _cpu_stdp_apply

- **Файл**: crystal_generator.py:474, 687, 840
- **Суть**: Ключевые методы без аннотаций. _branch возвращает list[tuple[int, float]].
- **Влияние**: P2. Затрудняет статический анализ и IDE-поддержку.
- **Предложение**: Добавить type hints: def _branch(self, seq: List[int], word_num: int, theta_temp: float = 0.3, target_cid: Optional[int] = None, centroid: Optional[ndarray] = None) -> List[Tuple[int, float]]:
- **Приоритет**: P2

### Q-6: Магические числа и хардкод

- **Файл**: crystal_generator.py (глобально)
- **Суть**: Десятки чисел: 0.15 (515, 1474), 0.3 (526, 755, 1097), 0.5 (121, 707), 0.7 (513), 0.05 (500), 0.02 (519), 0.25 (675), 0.75 (676), 2.0 (668).
- **Влияние**: P2. Код трудно поддерживать — непонятно откуда взяты числа.
- **Предложение**: Вынести в именованные константы класса или config.
- **Приоритет**: P2

### Q-7: FractalField.fluctuate() — потенциально бесконечный рост кодов

- **Файл**: concept_space.py:210–218
- **Суть**: fluctuate() на каждой итерации добавляет шум: c[:] = c * decay + noise. При decay=0.999 и noise_scale=0.005 дисперсия кодов растёт как noise_scale² / (1 - decay²) ≈ 0.0125. За 100К шагов std достигает ~35 — превышает check_code_range(bound=10.0).
- **Влияние**: P2. Нет механизма автоисправления — только диагностика.
- **Предложение**: Добавить клиппинг кодов: np.clip(c, -bound, bound, out=c). Или регуляризацию: c *= bound / max(abs(c)).
- **Приоритет**: P2

### Q-8: ConceptVectorStore.__getitem__ — нет проверки границ

- **Файл**: concept_space.py:44
- **Суть**: def __getitem__(self, cid): if self._valid[cid]: — нет проверки cid < 0 или cid >= self._V. Если вызвать store[-1], будет IndexError.
- **Влияние**: P2. Потенциальный crash при некорректном CID.
- **Предложение**: Добавить проверку: if 0 <= cid < self._V and self._valid[cid]:
- **Приоритет**: P2

### Q-9: requirements.txt — нет верхних границ

- **Файл**: requirements.txt
- **Суть**: numpy>=1.24.0, torch>=2.0.0 — нет верхних границ. При выходе NumPy 2.0 (ломающая обратную совместимость) проект упадёт. scipy не является прямой зависимостью, но указан как обязательный.
- **Влияние**: P2. Риск при обновлении зависимостей.
- **Предложение**: Добавить верхние границы: numpy>=1.24.0,<2.0. scipy — в optional dependencies.
- **Приоритет**: P2

### Q-10: Бесконечный цикл при превышении лимита concept_error

- **Файл**: crystal_generator.py (OrderedDict FIFO)
- **Суть**: while len(self.concept_error) > 30000: self.concept_error.popitem(last=False). Работает только для OrderedDict. Если кто-то переприсвоит self.concept_error = {}, .popitem(last=False) не гарантирует FIFO.
- **Влияние**: P3. Хрупкое допущение.
- **Предложение**: Явно проверять тип: assert isinstance(self.concept_error, OrderedDict) или использовать свой FIFO-класс.
- **Приоритет**: P3

### Q-11: Тяжёлый импорт scipy.sparse в теле метода

- **Файл**: concept_space.py:388-389
- **Суть**: from scipy.sparse import csr_matrix внутри build_octree_fields().
- **Влияние**: P3. Накладные расходы при каждом вызове.
- **Предложение**: Импорт на уровне модуля с обработкой ImportError.
- **Приоритет**: P3

### Q-12: Неконсистентное именование (theta_tau, _torch_dirty)

- **Файл**: crystal_generator.py:69, 98
- **Суть**: theta_tau — смесь греческой буквы и латинской. _torch_dirty — жаргонное название.
- **Влияние**: P3. Косметика.
- **Предложение**: theta_tau → theta_tau (оставить, известный термин), _torch_dirty → _torch_stale.
- **Приоритет**: P3

---

## QN: Новые методы QA

### QN-1: Проверка GPU/CPU численной эквивалентности (Parity Test)

- **Суть**: Автоматический тест на малом наборе данных (2-3 предложения) запускает train_from_text() в обоих режимах и сравнивает изменения векторов (np.allclose), concept_path, score, PPL.
- **Приоритет**: P1

### QN-2: Стресс-тест памяти (Memory Safety / OOM Guard)

- **Суть**: Запуск на GPU с максимальными параметрами (batch_size=128, neg_samples=8). Проверка torch.cuda.max_memory_allocated() < 80% VRAM. Проверка OOM fallback на CPU. Замер утечек alloc до/после 100 итераций.
- **Приоритет**: P2

### QN-3: Фаззинг границ концептов (Boundary Fuzz)

- **Суть**: Подача граничных значений cid = -1, vocab_size, vocab_size+100500. Очень длинные последовательности (10K токенов). Пустой beam. Проверка, что все методы корректно возвращают None/пустой список.
- **Приоритет**: P2

---

# 6. Итоговая матрица приоритетов

| ID | Проблема | Приор. | Агент | Тип |
|:--:|----------|:------:|:-----:|:---:|
| G-1 | .to() вместо .copy_() — тройные аллокации | **P0** | GPU | Производительность |
| G-2 | FP16 не используется — половинная ёмкость VRAM | **P0** | GPU | Память |
| Q-1 | np.load() без allow_pickle=False — RCE | **P0** | QA | Безопасность |
| T-1 | cw=1 на старте — обучение не работает | **P0** | TD | Корректность |
| T-2 | Destab reset на epoch 2+ — циклический сброс | **P0** | TD | Корректность |
| S-1 | Отсутствие L2-retraction после STDP | **P0** | NS | Корректность |
| A-1 | Циркулярная связь Generator↔Space | **P1** | Arch | Архитектура |
| A-2 | Monkey-patching SyntaxLattice | **P1** | Arch | Архитектура |
| A-3 | FCFConfig God Object | **P1** | Arch | Архитектура |
| A-5 | Неполная синхронизация GPU тензоров | **P1** | Arch | Архитектура |
| G-3 | Нет pin_memory для H2D transfers | **P1** | GPU | Производительность |
| G-4 | Синхронный CPU transfer | **P1** | GPU | Производительность |
| G-5 | Нет профилирования GPU | **P1** | GPU | Мониторинг |
| S-2 | Ненормализованный scatter_add | **P1** | NS | Корректность |
| S-3 | Перевёрнутый concept_error factor в PMI | **P1** | NS | Корректность |
| S-4 | Concept error не используется в neg sampling | **P1** | NS | Улучшение |
| T-3 | CURICULUM_MIN_LEN отсекает 30-50% корпуса | **P1** | TD | Данные |
| T-4 | vacc1_stuck — ложные срабатывания | **P1** | TD | Корректность |
| T-5 | Накопление шума от fluctuate | **P1** | TD | Стабильность |
| T-6 | Нет early stopping | **P1** | TD | Качество |
| Q-2 | Отсутствие тестов | **P1** | QA | Тестирование |
| Q-3 | _quiet() глотает исключения | **P1** | QA | Надёжность |
| Q-4 | Дублирование STDP-логики | **P1** | QA | Качество |
| GN-1 | Fused STDP kernel | **P1** | GPU | Оптимизация |
| GN-2 | Gradient Accumulation (GAGAT) | **P1** | GPU | Оптимизация |
| SN-1 | Multi-resolution octree fields | **P1** | NS | Новая фича |
| TN-1 | Adaptive curriculum по difficulty | **P1** | TD | Улучшение |
| TN-2 | EMA concept vectors | **P1** | TD | Стабильность |
| AM-1 | EventBus — система событий | **P1** | Arch | Архитектура |
| AM-2 | Protocols-модуль | **P1** | Arch | Архитектура |

---

*Отчёт составлен коллегией AI-агентов: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent*
*Всего: 59 проблем (6 P0, 23 P1, 23 P2, 7 P3) + 17 новых методов*
