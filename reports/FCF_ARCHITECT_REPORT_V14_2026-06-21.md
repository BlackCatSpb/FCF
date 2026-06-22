# FCF V14 Architect Audit — 2026-06-21

## Состояние V13

**139 passed, 0 failed, 3 skipped** — первый раз 0 failed.

Все 4 P1, 6 P2 закрыты. 13 новых тестов (destab, neg sampling, bf16, cleanup API).
Исправлены: AM-96-1 (fp16→bf16), SN-53 (int64→int32), B4 (double-write fix),
G-69 (_codes_t fp16, −150MB), G-72 (lazy CPU sync), TN-47/32/44/48, SN-54,
G-66 (pure-tensor + torch.compile), инкрементальный prefix_total, полный async checkpoint.

---

## AM-100+: Что осталось сделать

### 1. Высокий приоритет (производительность/корректность)

| ID | Проблема | Файл | Суть | Предложение |
|----|----------|------|------|-------------|
| AM-100 | `_sync_after_fluctuate` создаёт `np.zeros((V, latent_dim))` | `crystal_generator.py:356-358` | Каждый флукту — полный numpy array 146K×512 (299MB), потом PCIe копия. | Писать напрямую в `_codes_t` через batched tensor assign, избегая numpy. |
| AM-101 | `_lateral_inhibition_gpu` — per-element loop (gi in range(n)) | `stdp_trainer.py:594` | Не векторизован финальный write-back, каждый gi → отдельный check `gi_mask.any()`. | Полная векторизация: один batched scatter_add для всех inhibit-обновлений. |
| AM-102 | `_contrastive_objective_gpu` — per-concept loop (for i in range(ng)) | `stdp_trainer.py:868` | Каждая итерация i делает `v_local = g_vecs[i].clone()` + conditional checks. | Вынести conditional per-concept logic в маски: `valid_reg[i]`, `valid_hn[i]` — уже есть, но v_local update не векторизован. |
| AM-103 | `_negative_sampling_gpu` — per-concept loop (for gi, gen_cid in enumerate) | `stdp_trainer.py:695` | Аналогично: gi loop с `neg_mask = mask[gi]`, собирает `_neg_updates`. | Все обновления можно собрать в batched tensor, затем один write. |
| AM-104 | Нет CUDA streams для overlap PCIe/compute | `crystal_generator.py` | `copy_(..., non_blocking=True)` без streams — синхронизация всё равно блокирует. | Создать 2 CUDA streams: stream1 для H2D копий, stream2 для compute. |
| AM-105 | Нет gradient accumulation | `stdp_trainer.py` | Каждый batch — независимый forward/backward. Нет аккумуляции через несколько micro-batches. | Добавить `accum_steps`: аккумулировать acc/elr в `_fused_buf`, делить на `accum_steps` при записи. |
| AM-106 | _build_torch_tensors O(V·D) CPU overhead | `crystal_generator.py:267-278` | При каждом rebuild (fluctuate, dirty) — полный проход по 146K codes → numpy array. | Использовать `_codes_t.float() @ _basis_t` на GPU вместо CPU loop. Уже частично в `_sync_after_fluctuate`, но `_build_torch_tensors` всё ещё старый. |

### 2. Средний приоритет (precision / safety / debuggability)

| ID | Проблема | Файл | Суть | Предложение |
|----|----------|------|------|-------------|
| AM-107 | `_mom_t` bf16 — малая точность для momentum | `crystal_generator.py:319` | bf16 имеет 7 bits mantissa — малые градиенты (<1e-3) теряются. | `_mom_t` → fp16 или хранить в fp32 с кастомным `_mom_scale` (per-tensor lossless downcast). |
| AM-108 | `_codes_t` fp16 → fp32 roundtrip loss | `stdp_trainer.py:637` | `_codes_t[cids_t] = new_codes.to(torch.float16)` — fp16 точности 3.3 знака. При малых LR (<0.01) код может не обновиться. | Хранить `_codes_t` в fp32 (extra 300MB на 146K×512?) — дорого. Лучше: хранить fp16, но accumulate updates в fp32 и синхрить раз в K шагов. |
| AM-109 | Нет bounds checks на GPU tensor access | `stdp_trainer.py:410-412` | `gen._cf_t[prev_cid_t]` — если prev_cid >= V, будет silent error. | Добавить `torch.clamp` или assert. |
| AM-110 | `_ce_t` обновляется в двух местах | `stdp_trainer.py:452` и `460` | GPU loop обновляет `gen._ce_t[unique_gen]` напрямую, CPU fallback — через `gen.concept_error.update()`. Двойной source of truth. | Убрать CPU fallback: всегда писать в `_ce_t` на GPU, на CPU читать `.cpu()`. |
| AM-111 | `qwen_knowledge.inject_qwen_knowledge` — dead code | `qwen_knowledge.py:98-109` | Функция — `pass` с docstring. Инлайн уже сделан в `_build_pairs:250-251`. | Удалить. |

### 3. Низкий приоритет (рефакторинг / тесты)

| ID | Проблема | Файл | Суть | Предложение |
|----|----------|------|------|-------------|
| AM-112 | Нет end-to-end training теста | — | Нет теста, который запускает train_batch + checkpoints + resume. | Добавить `test_end_to_end.py`: small vocab, 10 lines, 2 epochs, verify metrics. |
| AM-113 | Нет теста fluctuate + generation | — | `fluctuate_fractal` меняет вектора, но generation должна работать. | Добавить тест: fluctuate → generate → нормы единичные. |
| AM-114 | Нет `_sync_after_fluctuate` теста | — | Проверить, что после fluctuate GPU тензоры консистентны с CPU. | Тест: fluctuate → `_sync_after_fluctuate` → `_sync_dirty_cpu` → compare norms. |
| AM-115 | `hormonal_system.py:48` — _last_few_cids как instance attr | `hormonal_system.py:52` | Инициализирован в __init__ как `[]`, но в `save/load` восстанавливается. OK, но reset() не очищает. | `reset()` должен заново вызывать `__init__()` — уже есть. |
| AM-116 | Нет теста на GPU field_gate интеграцию | — | field_gate threshold как float параметр — протестирован только CPU path. | Добавить `test_gpu_field_gate_threshold`. |
| AM-117 | `train_full.py:730` — _batch_timing CSV открыт на всё обучение | `train_full.py:728` | CSV file handle живёт всю тренировку (сотни тысяч строк). Утечка ресурса, если не закрыт. | Использовать `with` или закрывать при каждой записи. |
| AM-118 | `batch_log` может быть не определён при `KeyboardInterrupt` | `train_full.py:731,817-820` | `if 'batch_log' in locals()` — хрупко. | Использовать контекстный менеджер. |

### 4. Feature-level suggestions (архитектурные)

| ID | Предложение | Обоснование |
|----|-------------|-------------|
| AM-119 | **Gradient checkpointing** для GPU STDP | `_gpu_stdp_core` держит промежуточные тензоры (vc, vg, pair_delta, fused_src). На 2GB GPU каждый байт на счету. Recompute градиента вместо хранения. |
| AM-120 | **Static batch size** вместо dynamic growth | `_fused_buf` растёт динамически (4096 → n*2). Аллокация в цикле — overhead. Лучше выделить max(N_pairs_per_batch) однажды. |
| AM-121 | **CPU-offload для EMA** | `_ema_vecs_t` (bf16) обновляется каждый batch. На CPU это async thread. |
| AM-122 | **Per-concept LR** вместо единого base_lr | Разные концепты имеют разную частоту. `lr *= 1/log(freq+1)` — уже есть freq_weight, но не для каждого gen_cid индивидуально. |
| AM-123 | **Remove `gen_updates` dict from GPU path** | Сейчас GPU STDP строит `gen_updates` (CPU dict) параллельно с `gpu_ctx_l/gpu_tgt_l`. Это dead data для GPU path — нужен только для contrastive_objective CPU fallback. GPU contrastive использует `_gpu_elr_avg` / `_gpu_unique_gen`. |
| AM-124 | **Конфигурируемый latent_dim** | `FractalField.__init__` hardcoded на `latent_dim=512` в `ConceptSpace.__init__:329`. Должен браться из `FCFConfig`. |
| AM-125 | **Отдельный evaluate на GPU без CPU fallback** | `evaluate()` использует `cs.batch_dot()` (Python loop). GPU evaluate есть неявно через `_contrastive_objective_gpu`, но не для метрик. |

---

## Итого

**138 passed → планируется 155+** после закрытия AM-100..125.

| Тип | Кол-во | Экономия |
|-----|--------|----------|
| Performance (GPU) | 6 | ~30-50% batch time |
| Precision fixes | 3 | ~5% quality improvement |
| Safety | 2 | Устранение silent crashes |
| Testing | 6 | +30 новых тестов |
| Refactoring | 8 | −200 строк |
| Features | 5 | Качество генерации |

Основной bottleneck сейчас — per-element loops на GPU (inhibition, contrastive, neg sampling).
Векторизация даст 2-3× ускорение batched STDP.
