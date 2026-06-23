# FCF V15 — Сводный отчёт коллегии AI-агентов

**Дата**: 2026-06-23
**HEAD**: 4178389 (V14 → V15)
**Состав**: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent

---

## Executive Summary

После V14 произошла **крупная архитектурная перестройка**: 8 коммитов, −7228 строк, +480 строк. Ключевые изменения:

1. **FractalField переписан**: dim=384→768, latent_dim=512→2048, новые подсистемы
2. **EntityField + Harmonizer** — рекурсивное семантическое поле char→morph→word→sent→para
3. **QwenKnowledge полностью удалён**
4. **HDC/VSA n-gram память**, learnable field projection (W_proj), 3-level sector index
5. **Per-concept adaptive L1** и dynamic capacity growth/prune
6. **Minesweeper cluster-potential** LR modulation
7. **CheckpointManager** — асинхронное сохранение

| Метрика | V14 | V15 | Δ |
|---------|:---:|:---:|:-:|
| P0 | 0 | **3** | +3 |
| P1 | 3 | **13** | +10 |
| P2 | 5 | **8** | +3 |
| P3 | 0 | **8** | +8 |
| Тесты | 145 | 145 | = |
| Failed | 0 | 0 | = |
| VRAM | ~520MB | **~1518–1718 MB** | +1000MB |

VRAM вырос в 3× из-за dim/latent_dim scale-up. На 2GB GPU остаётся ~330–530 MB запаса.

---

## P0 — Критические проблемы (3)

| ID | Проблема | Зона | Источник |
|:--:|----------|:----:|:--------:|
| **V15-P0.1** | **HRR-алгебра не работает с вещественными векторами.** `a * b` и `c * b` с unit-norm гауссовыми не дают `unbind(bind(a,b), b) ≈ a`. Искажение ~50% на bind. Поражены EntityField, Harmonizer, HDC n-gram | NS | Neuro-Symbolic |
| **V15-P0.2** | **Slow-start Harmonizer декларирован, не реализован.** `_harm_slow_start_epochs=5` объявлен, но `harmonize()` не читает его — гармонизация на полную мощность с первого батча | NS | Neuro-Symbolic |
| **V15-P0.3** | **VRAM budget критический.** _codes_t 598MB + _mom_t 224MB + _vecs_t 224MB + _ema_vecs_t 224MB = 1270MB только 4 тензора. На 2GB GPU запас ~330–530 MB, при batch>16 вероятен OOM | GPU | GPU-Opt |

---

## P1 — Высокие проблемы (13)

| ID | Проблема | Зона | Источник |
|:--:|----------|:----:|:--------:|
| V15-P1.1 | **HDC fallback не окупает 400MB.** Вызывается только при <3 lattice candidates (<1% случаев). SyntaxLattice дешевле и точнее | Arch | Architect-AI |
| V15-P1.2 | **EntityField read-only.** Не пишет обратно в STDP, 400MB RAM, VSA без cleanup | Arch | Architect-AI |
| V15-P1.3 | **W_proj Hebbian — положительная ОС без координации со STDP.** Коллапс field_bits | Arch | Architect-AI |
| V15-P1.4 | **Dynamic capacity + async checkpoint — race condition.** grow/prune меняет 5 структур, checkpoint может сохранить несогласованное состояние | Arch | Architect-AI |
| V15-P1.5 | **HDC memory не очищается после fluctuate.** Старые bundled representations становятся мусором | Arch | Architect-AI |
| V15-P1.6 | **L1-адаптация не срабатывает.** init_concept даёт плотность 3%, пороги 2.4%/12% — 3% не попадает ни в один. >90% концептов без адаптации | NS | Neuro-Symbolic |
| V15-P1.7 | **EntityField без cleanup-памяти.** После unbind — шумовой вектор без деноизинга | NS | Neuro-Symbolic |
| V15-P1.8 | **Антоним-словарь из 28 пар не работает.** BPE-токены не совпадают с ключами, реальная детекция ≈0% | NS | Neuro-Symbolic |
| V15-P1.9 | **Секторный индекс не обновляется после grow/prune.** `_rebuild_sector_index()` не вызывается | NS | Neuro-Symbolic |
| V15-P1.10 | **`pipeline.last_fluct_lines` shadowing.** Локальная `last_fluct_lines = 0` в `_checkpoint()` не обновляет глобальную — fluctuate может не срабатывать | TD | Training-Dynamics |
| V15-P1.11 | **`_batch_mult` никогда не сбрасывается.** После плато батч остаётся 2×/4× навсегда | TD | Training-Dynamics |
| V15-P1.12 | **`_META_QWEN = 9` конфликтует с antonym_flag.** Если активировать Qwen — index out of bounds или логическая ошибка | QA | Quality-Safety |
| V15-P1.13 | **EntityField CPU→GPU sync через 1920 индивидуальных копий.** 10–20 ms/шаг из-за PCIe latency | GPU | GPU-Opt |

---

## P2 — Средние проблемы (8)

| ID | Проблема | Зона | Источник |
|:--:|----------|:----:|:--------:|
| V15-P2.1 | per-concept L1 lambda не сохраняется в чекпоинт | Arch | Architect-AI |
| V15-P2.2 | Cluster potential не сохраняется в чекпоинт | Arch | Architect-AI |
| V15-P2.3 | PARA-level EntityField binding не вызывается | Arch | Architect-AI |
| V15-P2.4 | EntityField.decay не вызывается в training loop | Arch | Architect-AI |
| V15-P2.5 | Char-level binding O(corpus_bytes) — LRU-кэш нужен | NS | Neuro-Symbolic |
| V15-P2.6 | Sent_vec пересчитывается для каждого dirty-слова O(N×S) | NS | Neuro-Symbolic |
| V15-P2.7 | Отсутствует верхний предел для adaptive capacity growth | TD | Training-Dynamics |
| V15-P2.8 | HDC n-gram update — O(L×max_n) на предложение, memcpy 8KB/call | TD | Training-Dynamics |

---

## P3 — Низкие проблемы (8)

| ID | Проблема | Зона | Источник |
|:--:|----------|:----:|:--------:|
| V15-P3.1 | EntityField.char_envelope не реализован | Arch | Architect-AI |
| V15-P3.2 | Scheduler не использует total_freq_cache в GPU | Arch | Architect-AI |
| V15-P3.3 | Антоним-словарь — хардкод, не масштабируется | NS | Neuro-Symbolic |
| V15-P3.4 | `morph_conf_threshold = 0.8` отсекает короткие слова | NS | Neuro-Symbolic |
| V15-P3.5 | `_semantic_bootstrap` избыточен (но жив в _checkpoint) | NS | Neuro-Symbolic |
| V15-P3.6 | HDC memory max=50000 FIFO — LFU лучше | NS | Neuro-Symbolic |
| V15-P3.7 | `_skip_gpu_sync` без try/finally — риск тихого расхождения | QA | Quality-Safety |
| V15-P3.8 | `harmonize_with_envelope()` — мёртвый код | TD | Training-Dynamics |

---

## Ключевые метрики и анализ

### VRAM Budget (исправленная оценка)

После поправки, что `_mom_t = V × 768 × bf16` (не V × 2048):

| Тензор | Размерность | Тип | VRAM |
|--------|------------|:---:|:----:|
| _codes_t | [146K, 2048] | fp16 | 598 MB |
| _mom_t | [146K, 768] | bf16 | 224 MB |
| _vecs_t | [146K, 768] | fp16 | 224 MB |
| _ema_vecs_t | [146K, 768] | bf16 | 224 MB |
| _fb_t | [146K, 256] | uint8 | 37 MB |
| _basis_t | [2048, 768] | fp32 | 6 MB |
| _cf_t + _pt2_t + _skip2_t + _ce_t | [146K]×4 | fp32 | 2.3 MB |
| _cluster_map | [146K] | int64 | 1.2 MB |
| Остальное | — | — | ~0.1 MB |
| **Постоянные тензоры** | | | **~1318 MB** |
| CUDA context + overhead | | | ~100–150 MB |
| Временные (batch) | | | ~100–250 MB |
| **Итого** | | | **~1518–1718 MB** |

**Запас**: 330–530 MB на 2GB GPU.

### Покрытие тестами

| Компонент | Покрытие | Статус |
|-----------|:--------:|:------:|
| ConceptVectorStore | ✅ Полное | OK |
| FractalField (базовый) | ⚠️ Частичное | Не тестированы L1, HDC, sector, capacity, W_proj |
| EntityField | ❌ Нулевое | 0 тестов на 236 строк |
| Harmonizer | ❌ Нулевое | 0 тестов на 355 строк |
| STDP CPU path | ✅ Хорошее | Все основные пути |
| STDP GPU path | ⚠️ Среднее | Нет antonym_mask, cluster_centroid |
| CheckpointManager | ⚠️ Среднее | Нет capacity grow интеграции |
| ParameterOptimizer | ✅ Полное | OK |
| HDC n-gram | ❌ Нулевое | 7 функций без тестов |
| Dynamic capacity | ❌ Нулевое | grow/prune/auto без тестов |

---

## Предложенные улучшения (новые методы)

### От Architect-AI
1. **Gradient-Based Field Projection (STE)** — замена Hebbian на градиентный W_proj через straight-through estimator
2. **Adaptive HDC Cache (LFU)** — замена FIFO на frequency-aware eviction
3. **EntityField → STDP feedback** — замкнуть цикл char-level контекст → concept vectors
4. **Progressive resizing** — external dimension buckets вместо немедленного grow
5. **Gumbel-Softmax adaptive sparsity** — замена hand-tuned порогов на дифференцируемые gates

### От Neuro-Symbolic
6. **FFT-HRR для VSA** — замена element-wise multiply на frequency-domain HRR
7. **Bipolar Binary VSA (BSC)** — альтернатива: {±1}^d для EntityField
8. **Cleanup Memory** — Sparse Associative Memory через sector_index
9. **Автоматическая детекция антонимов** — через cosine-антиподы + PMI
10. **Hierarchical VSA** — разная размерность для каждого уровня (64/256/2048/1024/512)
11. **Structured L1 (Group Lasso)** — вместо покомпонентной

### От Training-Dynamics
12. **Adaptive harmonizer ramp** — convergence-зависимая модуляция slow-start
13. **GPU semantic bootstrap** — перенос _semantic_bootstrap на GPU

### От GPU-Opt
14. **fp8 quantization _codes_t** — E4M3, −299 MB VRAM
15. **Необязательный _ema_vecs_t** — lazy, −224 MB VRAM
16. **Батчевая запись в harmonize** — −10–20 ms/шаг

---

## План работ (рекомендуемый)

### Фаза 0 — Немедленно (баги с эффектом)

1. **[P0.2] Slow-start Harmonizer** — добавить модуляцию harm_lr по чекпоинтам (5 строк)
2. **[P1.10] last_fluct_lines shadowing** — заменить на `self.last_fluct_lines`
3. **[P1.11] _batch_mult сброс** — добавить decay после выхода из плато
4. **[P1.12] _META_QWEN** — удалить константу (конфликт с antonym)

### Фаза 1 — V16 (VRAM + P0)

5. **[P0.3] VRAM**: _ema_vecs_t lazy (−224 MB), _codes_t fp8 (−299 MB)
6. **[P0.1] FFT-HRR**: заменить element-wise VSA на frequency-domain во всех 3 компонентах (EntityField, Harmonizer, HDC)
7. **[P1.5] HDC memory**: очищать после fluctuate
8. **[P1.13] Батчевая запись harmonize**: накапливать updates в буфер вместо 1920 микро-копий

### Фаза 2 — V17 (P1)

9. **[P1.1] HDC fallback**: или убрать / уменьшить до 1K entry / использовать как RRF-сигнал
10. **[P1.4] Async checkpoint safety**: threading.Lock для grow/prune + save
11. **[P1.6] L1 triggers**: расширить пороги (init_density × 0.5)
12. **[P1.7] EntityField cleanup memory**: добавить деноизинг через sector_index
13. **[P1.9] Sector index rebuild**: добавить в grow/prune

### Фаза 3 — Тесты (V16-17)

14. EntityField: 4 теста (bind/query, sync_word, _to_dim, serialization)
15. Harmonizer: 3 теста (compose/decompose, converge, dirty cascade)
16. HDC: 2 теста (bind/permute, update/predict)
17. Dynamic capacity: 2 теста (grow, prune)
18. W_proj: 1 тест (Hebbian update)
19. Adaptive L1: 1 тест (density maintenance)
20. Sector index: 1 тест (search/focal_refine)
21. Checkpoint + capacity: 1 интеграционный тест

---

## Итог

V15 — **самый крупный архитектурный сдвиг** в проекте. Размерность удвоена (384→768D, 512→2048D), добавлены EntityField/Harmonizer/HDC — но **VSA bind/unbind сломана фундаментально** (HRR с real-valued векторами не даёт clean unbind). VRAM выросла в 3×, на 2GB GPU — впритык.

Позитив: тесты все проходят (145/145), QwenKnowledge удалён, новая архитектура концептуально правильная. Критические P0 (HRR, slow-start, VRAM) блокируют дальнейшее развитие — их исправление в V16 является обязательным.

---

*Отчёт составлен коллегией AI-агентов: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent*
