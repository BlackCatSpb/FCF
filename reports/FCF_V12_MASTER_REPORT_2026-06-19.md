# FCF V12 — Сводный отчёт коллегии AI-агентов

**Дата**: 2026-06-19
**Версия**: V12 (аудит V11.2+V11.3+hotfixes: a705223..1768f27)
**Состав**: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent

---

## Executive Summary

V11.2 и V11.3 закоммичены. **122 passed, 4 failed, 3 skipped** (126 тестов, +21 с V11).

| Метрика | V11 | V12 | Δ |
|---------|:---:|:---:|:-:|
| P0 | 1 | **0** | −1 |
| P1 | 4 | **4** | = |
| P2 | 12 | **10** | −2 |
| Syncs/batch | ~1,000-5,000 | **~640** | −87% |
| `.item()` syncs | ~5,600 | **0** | −100% |
| VRAM usage | ~1.1GB | **~650MB** | −450MB |
| Тесты | 105 | **126** (+21) | +20% |

**Главные достижения V12:**
1. ✅ **G-60/SN-45: GPU destab** — полный перевод на GPU (RNG, PPMI, mix). 0 `.item()`, 0 `.numpy()`
2. ✅ **SN-43: Batched GPU neg sampling** — per-concept loop → batched tensor, один D2H
3. ✅ **SN-44: Pure-tensor GPU contrastive** — 0 `.item()` syncs. Pre-computed cooc_masks + fb_overlaps
4. ✅ **B1 (double momentum)** — исправлен: CPU-цикл удалён, momentum через `_mom_t` один раз
5. ✅ **TN-40 (P0 crash)** — `noise_scale`→`fluctuation_amp` фикс
6. ✅ **TN-41 (LR warmup rescore)** — `_lr_offset` добавлен
7. ✅ **VRAM −450MB** — `_fused_buf` dynamic growth, `_ema_vecs_t`/`_mom_t` fp16
8. ✅ **160s/batch fix** — убран unconditional `_torch_dirty=True`
9. ✅ **QN-49..QN-58** — все 22 теста реализованы

---

## 1. V11.2 Fixes Verification (a705223)

| ID | Fix | Статус | Детали |
|:--:|-----|:------:|--------|
| **G-60/SN-45** | GPU destab (RNG, PPMI, mix) | ✅ | `torch.rand`, `_vecs_t[rand_idx]`, `torch.where`. 0 `.item()` |
| **SN-43** | Batched GPU neg sampling | ✅ | Reuse `_gpu_elr_avg`. Batched `_vecs_t[cids_batch]`. Один D2H |
| **SN-44** | GPU contrastive pure-tensor | ✅ | Pre-computed masks. 0 `.item()`. Единственный Python loop: `for i in range(ng)` |

## 2. V11.3 + Hotfixes Verification

| Коммит | Fix | Статус |
|:------:|-----|:------:|
| 3150d5e | `_fused_buf` dynamic growth (225MB→<1MB) | ✅ |
| 4030b54 | `_ema_vecs_t`/`_mom_t` fp16 (экономия 224MB) | ⚠️ Риск underflow для редких концептов |
| 1768f27 | Убрать `_torch_dirty=True` (160s/batch fix) | ✅ |
| ac27cce | `_cached_decay` kwargs pass-through | ✅ |
| 56f69a0 | `_usage_decay_steps` on load | ✅ |
| 024f1aa | TN-40: noise_scale→fluctuation_amp | ✅ |

---

## 3. P1 — Критические проблемы (4)

| ID | Проблема | Зона | Сложность |
|:--:|----------|:----:|:---------:|
| **TN-46** | TN-13 plateau doubling BROKEN — `BATCH_SIZE` перезаписывается `bs_curve()` на каждой строке | TD | 2 |
| **AM-96-1** | `_mom_t`/`_ema_vecs_t` fp16 → underflow у редких концептов. Нужен bf16 или динамический каст | GPU | 2 |
| **SN-53** | `fb_overlaps` int64 → 117MB на 100 концептов, 1.3GB на 1000. Нужен uint8 | NS | 3 |
| **B4 (double-write)** | `_on_vector_update` перезаписывает batched GPU write — 2× bandwidth waste | GPU | 2 |

---

## 4. P2 — Проблемы средней критичности (10)

### GPU (4)
- G-69: `_codes_t` fp32→fp16 (−142MB VRAM)
- G-72: Lazy CPU `concept_vectors` sync
- SN-54: `_ensure_torch` — полный rebuild O(V·D) после каждого fluctuate (нужен инкрементальный)
- G-66: CUDA Graph (blocked by B4 + per-element loops)

### Training Dynamics (4)
- TN-32/44: `idx=-1` curriculum reset (mitigated, root cause open)
- TN-34: opt.json naming mismatch (mitigated, root cause open)
- TN-47: нет обратной связи `mean_cos`→`fluctuate_fractal`
- TN-48: `field_gate` не в `opt.p` (бинарный флаг)

### Quality (2)
- G-60 destab coverage: 0% (все тесты c `destab_scale=0.0`)
- QN-59..QN-63: 13 новых тестов предложены (destab, cleanup, fp16, neg sampling parity)

---

## 5. НОВЫЕ проблемы V12

| ID | Проблема | P | Агент | Сложность |
|:--:|----------|:-:|:-----:|:---------:|
| **TN-46** | TN-13 plateau doubling broken (BATCH_SIZE сбрасывается) | P1 | TD | 2 |
| **AM-96-1** | fp16 underflow у редких концептов | P1 | Arch | 2 |
| **SN-53** | fb_overlaps int64 — 1.3GB на 1000 концептов | P1 | NS | 3 |
| **B4** | double-write `_on_vector_update` | P1 | GPU | 2 |
| **SN-54** | Полный rebuild тензоров после каждого fluctuate | P2 | NS | 4 |
| **TN-47** | Нет обратной связи cos→fluctuate | P2 | TD | 2 |
| **TN-48** | field_gate не в opt.p | P2 | TD | 2 |

---

## 6. Что СДЕЛАНО (V11→V12 прогресс)

### Исправлено (15+)
- ✅ G-60/SN-45: GPU destab
- ✅ SN-43: Batched GPU neg sampling
- ✅ SN-44: Pure-tensor GPU contrastive
- ✅ B1: Double momentum fix
- ✅ TN-40: noise_scale crash fix
- ✅ TN-41: LR warmup after rescore
- ✅ TN-13: Progressive batch size (но BROKEN — TN-46)
- ✅ TN-15: Decay warmup
- ✅ VRAM −450MB (fused_buf dynamic, fp16 EMA/mom)
- ✅ 160s/batch slowdown fix
- ✅ Hotfixes: kwargs, usage_decay_steps
- ✅ QN-49..QN-58: 22 теста
- ✅ SN-48: G-65 field overlap fix

### Остаётся
- ⏳ TN-46: fix TN-13 plateau doubling
- ⏳ B4: fix double-write
- ⏳ fp16 underflow protection
- ⏳ fb_overlaps int64→uint8
- ⏳ G-60 destab tests (0% coverage)
- ⏳ 4 failed CheckpointManager теста

---

*Отчёт составлен коллегией AI-агентов: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent*
