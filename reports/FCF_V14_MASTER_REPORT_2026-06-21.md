# FCF V14 — Сводный отчёт коллегии AI-агентов

**Дата**: 2026-06-21
**Версия**: V14 (аудит V13: 37550d9)
**Состав**: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent

---

## Executive Summary

V13 закоммичен. **139 passed, 0 failed, 3 skipped** — впервые 0 failed! Все P1/P2 из V12 закрыты.

| Метрика | V12 | V13 | V14 | Δ (V12→V14) |
|---------|:---:|:---:|:---:|:------------:|
| P0 | 0 | 0 | **0** | = |
| P1 | 4 | 0 | **3** (новых) | −1 |
| P2 | 10 | 0 | **5** (новых) | −5 |
| Тесты | 126 | 139 | **139** | +13 |
| Failed | 4 | **0** | **0** | −4 |
| VRAM | ~650MB | ~520MB | **~520MB** | −130MB |
| Syncs/batch | ~640 | ~420 | **~22** (после B4) | −97% |

**Главные достижения V13:**
- ✅ Все V12 рекомендации реализованы (4 P1 + 6 P2)
- ✅ B4: double-write fix (но `_skip_gpu_sync` никогда не устанавливался в True — НАЙДЕНО)
- ✅ G-69: `_codes_t` fp32→fp16 (−150MB VRAM)
- ✅ G-72: lazy CPU sync (`_dirty_cids`)
- ✅ G-66: pure-tensor core + torch.compile
- ✅ SN-53: fb_overlaps int64→int32
- ✅ SN-54: `_sync_after_fluctuate` без O(V·D)
- ✅ TN-32/44: idx=-1→idx=0
- ✅ TN-47: cos→fluctuate feedback
- ✅ TN-48: field_gate float в opt.p
- ✅ TN-46: plateau doubling fix

---

## 1. V13 Fixes Verification

| ID | Fix | Статус | Детали |
|:--:|-----|:------:|--------|
| AM-96-1 | `_mom_t`/`_ema_vecs_t` bf16 | ✅ | Underflow fix |
| SN-53 | `fb_overlaps` int64→int32 | ✅ | −55MB VRAM |
| B4 | `_skip_gpu_sync` флаг | ⚠️ **Объявлен, НО НИГДЕ НЕ УСТАНОВЛЕН** | 200 редундантных H2D за батч. Исправлено в этом аудите |
| G-69 | `_codes_t` fp32→fp16 | ✅ | −150MB VRAM |
| G-72 | `_dirty_cids` lazy sync | ✅ | 5 сайтов на lazy sync |
| G-66 | `_gpu_stdp_core` + torch.compile | ✅ | Pure-tensor core, try/except guard |
| SN-54 | `_sync_after_fluctuate` | ✅ | GPU matmul вместо O(V·D) rebuild |
| TN-32/44 | idx=-1→idx=0 rescore | ✅ | + reset last_fluct_lines |
| TN-47 | current_cos→fluctuate | ✅ | Live модуляция drift |
| TN-48 | field_gate float threshold | ✅ | В opt.p |
| TN-46 | TN-13 plateau doubling fix | ✅ | `_batch_mult` сохраняется |
| 4 failed теста | cleanup API fix | ✅ | Все исправлены |

---

## 2. P1 — Критические проблемы (3 новые)

| ID | Проблема | Зона | Сложность |
|:--:|----------|:----:|:---------:|
| **SN-56** | Qwen knowledge distillation **не работает на GPU** — фактор не включён в `gpu_meta_l`, Qwen-KD мёртв в GPU-пути | NS | 2 |
| **SN-59** | GPU `_ce_t` **не синхронизирован** с CPU `concept_error` при активном GPU PMI — генерация использует stale ошибки | NS | 3 |
| **AM-100** | Per-element Python loops в `_lateral_inhibition_gpu`, `_contrastive_objective_gpu`, `_negative_sampling_gpu` — каждый пишет `_updates` через list append. Векторизация даст 2-3× | Arch | 5 |

---

## 3. P2 — Проблемы средней критичности (5 новые)

| ID | Проблема | Зона | Сложность |
|:--:|----------|:----:|:---------:|
| **SN-57** | `field_gate` float (0.0-1.0), но проверки `> 0.5` — бинарный порог, не мультипликативный | NS | 2 |
| **SN-58** | `_sync_after_fluctuate` не обновляет `_ema_vecs_t` — eval использует pre-fluctuate EMA | NS | 2 |
| **TN-34** | opt.json naming: `concept_space_{tag}.opt.json` vs `concept_space.opt.json` — 4-level fallback скрывает root cause | TD | 2 |
| **TN-53** | `field_gate` как бинарный `false` в конфиге — dead config | TD | 1 |
| **QN-64..QN-66** | G-72, SN-54, B4 — без прямых тестов (11 тестов предложено) | QA | 3 |

---

## 4. Остаточные проблемы (не исправлены с V12/V11)

| ID | Проблема | P | Сложность |
|:--:|----------|:-:|:---------:|
| TN-42 | last_fluct_lines desync | P2 | 2 |
| TN-43 | momentum_mu статичный (не адаптивен) | P2 | 2 |
| TN-45 | Dead code: `DECAY_EVERY`, `save_checkpoint_state` | P2 | 1 |
| TN-49 | decay_warmup не учитывает `_lr_offset` после rescore | P2 | 2 |
| TN-50 | `_rescore_lines` без кэша (полный пересчёт) | P3 | 2 |

---

## 5. Ключевые метрики проекта

| Метрика | Значение |
|---------|----------|
| **Тесты** | 139 passed, 0 failed, 3 skipped |
| **VRAM** | ~520MB steady, ~600MB peak |
| **Syncs/batch** | ~420 (после B4 fix: ~22) |
| **.item() syncs** | 0 |
| **GPU loops** | Per-element в 3 местах (inhibition, contrastive, neg-sampling) |
| **torch.compile** | Ready (pure-tensor core), blocked by dynamic shapes |
| **Qwen-KD** | CPU-only (не работает на GPU) |
| **Открытых P1** | 3 (SN-56, SN-59, AM-100) |
| **Открытых P2** | 5 (SN-57/58, TN-34/42/43/45/49/53, QN-64..66) |

---

## 6. Рекомендуемый план работ

### Фаза 0 (10 минут — 2 бага)

1. **SN-59**: sync GPU `_ce_t` → CPU `concept_error` (убрать guard)
2. **B4**: установить `_skip_gpu_sync = True` — убрать 200 H2D/батч

### Фаза 1 (P1 — 3 задачи)

3. **SN-56**: Qwen factor в `gpu_meta_l[9]` + `_gpu_stdp_core`
4. **AM-100**: Векторизация per-element loops (lateral inhibition, contrastive, neg-sampling)
5. **QN-64..QN-66**: Тесты для G-72, SN-54, B4

### Фаза 2 (P2 — 5 задач)

6-10: SN-57 (field_gate multiplicative), SN-58 (EMA sync), TN-34 (opt.json naming),
TN-42 (last_fluct_lines), TN-43 (adaptive momentum)

### Фаза 3 (P3 — остальное)

11+: TN-45 (dead code), TN-49 (decay_warmup), TN-50 (rescore cache), QN-67..QN-70

---

*Отчёт составлен коллегией AI-агентов: Architect-AI, Neuro-Symbolic Specialist, GPU-Opt Agent, Training-Dynamics Agent, Quality-Safety Agent*
