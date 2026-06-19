# FCF Quality & Safety Report V11 вЂ” 2026-06-19

**Agent:** Quality-Safety Agent
**Scope:** V10 commits 525688b, d36a780; all `*.py` under `eva/symbolic/`, `tests/test_stdp.py`, `train_full.py`

---

## 1. V10 Test Progress вЂ” Р Р•РђР›РР—РћР’РђРќР« Р’РЎР• 9 QN-РЎР¬Р®РўРћР’ рџЋ‰

| QN | Suite | Tests | Status | V10 Р·Р°СЏРІРёР» |
|----|-------|:-----:|:------:|:----------:|
| QN-32 | `_apply_subspace_update` | 4 | **IMPLEMENTED** | вќЊв†’вњ… |
| QN-33 | GPU contrastive | 3 | **IMPLEMENTED** | вќЊв†’вњ… |
| QN-34 | evaluate | 3 | **IMPLEMENTED** | вќЊв†’вњ… |
| QN-35 | noise_scale | 1 | **IMPLEMENTED** | вќЊв†’вњ… |
| QN-36 | RNGRegistry | 4 | **IMPLEMENTED** | вќЊв†’вњ… |
| QN-37 | AdaptiveErrorTracker | 4 | **IMPLEMENTED** | вќЊв†’вњ… |
| QN-38 | Checkpoint cleanup | 3 | **IMPLEMENTED** | вќЊв†’вњ… |
| QN-39 | TrainingPipeline | 1 | **IMPLEMENTED** | вќЊв†’вњ… |
| QN-40 | Dead code | 3 | **IMPLEMENTED** | вќЊв†’вњ… |
| **РС‚РѕРіРѕ** | **9 СЃСЊСЋС‚РѕРІ** | **26 С‚РµСЃС‚РѕРІ** | **100%** | **+267 СЃС‚СЂРѕРє** |

**105 С‚РµСЃС‚РѕРІ РїСЂРѕС…РѕРґСЏС‚** (2 skipped вЂ” РЅРµС‚ SentencePiece). STR Р·РЅР°С‡РёС‚РµР»СЊРЅС‹Р№ СЂРѕСЃС‚.

---

## 2. V10 РќРѕРІС‹Р№ РєРѕРґ (G-40..G-52) вЂ” Coverage Analysis

### 2.1 РљРѕРјРјРёС‚ `525688b` вЂ” Phase 0 + P1 + GPU Opts + Tests

| ID | РР·РјРµРЅРµРЅРёРµ | Р¤Р°Р№Р»:РЎС‚СЂРѕРєР° | РЎС‚СЂРѕРє | РџРѕРєСЂС‹С‚? | Р РёСЃРє |
|:--:|-----------|:-----------:|:-----:|:-------:|:----:|
| **G-40** | `_codes_t` tensor + batched subspace update | `concept_space.py:591-638` | 48 | **вќЊ NO** | **HIGH** |
| **G-42** | GPU `_centroid_pull_batch` GPU path | `stdp_trainer.py:827-845` | 19 | **вќЊ NO** | **HIGH** |
| **G-44** | Pre-computed cooc masks + fb_overlaps | `stdp_trainer.py:707-720` | 14 | **вќЊ NO** | MED |
| **G-46** | Persistent `_mom_t` tensor (replace CPU dict) | `stdp_trainer.py:409-418` | 10 | вљ пёЏ partial | MED |
| **G-48** | `_HAS_COMPILE` constant | `stdp_trainer.py:12` | 1 | вњ… | LOW |
| **G-49** | Pre-allocated `_fused_buf` for scatter_add | `stdp_trainer.py:380-387` | 5 | **вќЊ NO** | MED |
| **G-55** | Remove profiling stubs | `stdp_trainer.py:30-31` | 2 | вњ… tested | LOW |
| **G-57** | Remove dead `push_total`/`lr_scale` | `stdp_trainer.py` | вЂ” | вњ… QN-40 | LOW |
| TN-31 | `checkpoint_state.json` save | `train_full.py:393-400` | 8 | **вќЊ NO** | MED |
| TN-32 | `_effective_cp` + `_rescore_cp` | `train_full.py:473-479, 659-676` | 15 | **вќЊ NO** | MED |
| TN-34 | opt.json naming fallback | `train_full.py:278-285` | 8 | **вќЊ NO** | MED |
| AM-30 | Batched EMA update | `stdp_trainer.py:486-489` | 4 | **вќЊ NO** | MED |
| AM-31 | CPU sync to AdaptiveErrorTracker | `stdp_trainer.py:404-407` | 4 | вќЊ (smoke only) | LOW |
| REG-V9-7 | `noise_scale`в†’`gradient_noise_scale` rename | across files | вЂ” | вљ пёЏ 1 test | LOW |

### 2.2 РљРѕРјРјРёС‚ `d36a780` вЂ” G-50/G-51/G-52 GPU zero-copy + deferred sync + fused

| ID | РР·РјРµРЅРµРЅРёРµ | Р¤Р°Р№Р»:РЎС‚СЂРѕРєР° | РЎС‚СЂРѕРє | РџРѕРєСЂС‹С‚? | Р РёСЃРє |
|:--:|-----------|:-----------:|:-----:|:-------:|:----:|
| **G-50** | GPU write-back (compute v_new on GPU) | `stdp_trainer.py:466-473` | 8 | **вќЊ NO** | **HIGH** |
| **G-51** | Deferred batched `_vecs_t` write | `stdp_trainer.py:427,478-484` | 7 | **вќЊ NO** | **HIGH** |
| **G-52** | Fused post-STDP `_gpu_poststdp_fused` | `stdp_trainer.py:121-128,496-506` | 9 | **вќЊ NO** | **HIGH** |
| G-41 | GPU lateral inhibition optimized (pre-`mask_all`) | `stdp_trainer.py:516-532` | 16 | вќЊ (smoke only) | MED |

### 2.3 Untested Coverage Summary (V10)

**Total new V10 code: ~571 lines added, 145 removed. Net: ~426 new lines.**

| Risk level | Count | РљР»СЋС‡РµРІС‹Рµ РїСЂРѕР±РµР»С‹ |
|:----------:|:-----:|------------------|
| **HIGH** | 4 | G-40 subspace_batch, G-42 GPU centroid, G-50 GPU write-back, G-51 deferred sync, G-52 fused post-STDP |
| **MED** | 7 | G-44 cooc masks, G-46 mom_t persistence, G-49 fused_buf, TN-31/32/34, AM-30 batched EMA |
| **LOW** | 3 | G-48, AM-31, REG-V9-7 rename |
| **Total uncovered** | **~14 areas** | |

---

## 3. STR (Structural Test Reach) вЂ” Updated

| Module | Lines | STR (est.) | О” vs V10 | Notes |
|--------|:-----:|:----------:|:--------:|-------|
| `test_stdp.py` | 1073 | вЂ” | вЂ” | Base test file |
| `concept_space.py` | 800 | 30% | в†‘ | QN-32 + subspace_batch partial |
| `crystal_generator.py` | 715 | 25% | в†‘ | QN-36, QN-37 coverage |
| `stdp_trainer.py` | 828 | 65% | в†“ в€’6pp | G-40..G-52 new untested paths |
| `train_full.py` | 724 | 5% | в†“ | TN-31/32/34 untested |
| `fcf_config.py` | 447 | 15% | в†” | unchanged |
| Others | ~500 | 15% | в†” | unchanged |

**Overall STR: ~48%** (в¬‡пёЏ в€’2pp from V10 due to new untested GPU paths)

---

## 4. Safety Regressions in V10

1. **G-50/G-51: GPU write-back bypasses `_apply_vector_update` for `_vecs_t`** вЂ” writes directly to `gen._vecs_t[cids_batch]`, then calls `_apply_vector_update` with `.cpu().numpy()`. Double-write risk: if `_vecs_t` dtype в‰  numpy dtype, silent truncation. **MEDIUM.**

2. **G-52: `_gpu_poststdp_fused` re-orders neg sampling + contrastive** вЂ” Previously CPU path was separate. Now GPU path calls neg_sampling then contrastive in a single function. If `_contrastive_objective_gpu` modifies vectors that `_negative_sampling_gpu` already used вЂ” no correctness issue (neg sampling is first). But `_contrastive_objective_cpu` (fallback) is still called separately -> divergence risk. **MEDIUM.**

3. **G-42: GPU `_centroid_pull_batch` has different math** — CPU path: `sent_lr = base_lr_val * 0.3`, then `shift = (centroid - sim * v) * sent_lr`. GPU path: `pulls = 0.1 * (cn - sims[:, None] * vecs)`, then `v_new = vecs + pulls * base_lr_val * 0.3`. The `0.1` factor is in GPU but not CPU. **HIGH — numerical parity gap.** — ✅ **FIXED in 024f1aa** (parity resolved, `0.1` factor standardized)

4. **G-44: `fb_overlaps` unsqueeze dimension** вЂ” `fb_overlaps = (fb_gen_all.unsqueeze(1) & gen._fb_t.unsqueeze(0)).sum(dim=-1)` creates `(ng, n_v)` tensor. Memory: for V=20000, this is 20000Г—20000 bits = 50MB вЂ” acceptable. But if `gen._fb_t` has >1 field bit per concept, overlap may be >1, while old code used `int(torch.bitwise_and(...).sum().item())` вЂ” same behavior. **LOW.**

5. **TN-32: `_rescore_cp` not reset properly** вЂ” If rescore happens at idx=1000 with cp=0.5, `_rescore_cp = 0.5`. After rescore idx becomes -1в†’0, new cp starts at 0. `_effective_cp` returns 0.5 until idx catches up. But if rescore happens multiple times, `_rescore_cp` may be overwritten. **MEDIUM.**

---

## 5. Proposed QN-49+ Tests (V11)

### QN-49: `_apply_subspace_update_batch` (4 tests)
- `test_subspace_update_batch_basic` вЂ” 2+ cids, verify all updated
- `test_subspace_update_batch_shift` вЂ” verify `_total_shift` increases
- `test_subspace_update_batch_unit_norm` вЂ” all outputs on unit sphere
- `test_subspace_update_batch_codes_sync` вЂ” `fractal.codes` matches `_codes_t`

### QN-50: GPU Centroid Pull вЂ” G-42 (2 tests)
- `test_centroid_pull_gpu_smoke` вЂ” GPU path with torch tensors (skip if no torch)
- `test_centroid_pull_gpu_cpu_parity` вЂ” compare GPU vs CPU numeric result on same input (medium tolerance)

### QN-51: Fused Post-STDP вЂ” G-52 (2 tests)
- `test_gpu_poststdp_fused_neg_sampling_called` вЂ” mock/stub verify neg sampling invoked
- `test_gpu_poststdp_fused_contrastive_called` вЂ” verify contrastive invoked

### QN-52: Deferred GPU Write-back вЂ” G-50/G-51 (3 tests)
- `test_gpu_deferred_write_vecs_t_updated` вЂ” verify `_vecs_t[cid]` changed after deferred path
- `test_gpu_deferred_write_norm_maintained` вЂ” unit norm after deferred
- `test_gpu_deferred_write_subspace_skipped` вЂ” verify non-subspace path goes to deferred

### QN-53: GPU Lateral Inhibition Optimized вЂ” G-41 (2 tests)
- `test_gpu_lat_inh_precomputed_mask` вЂ” verify `mask_all.fill_diagonal_(False)` works
- `test_gpu_lat_inh_correctness` вЂ” compare with known small-n reference

### QN-54: TN-31 checkpoint_state (2 tests)
- `test_ckpt_state_saved` вЂ” verify `checkpoint_state.json` created after checkpoint
- `test_ckpt_state_content` вЂ” verify `line`, `epoch`, `global_step` keys present

### QN-55: TN-32 effective_cp (2 tests)
- `test_effective_cp_without_rescore` вЂ” baseline behavior
- `test_effective_cp_after_rescore` вЂ” verify cp preserved across rescore, released when caught up

### QN-56: AM-30 Batched EMA (2 tests)
- `test_ema_batch_multiple_cids` вЂ” verify all unique_gen CIDs get EMA update
- `test_ema_batch_steps` вЂ” verify `_ema_steps` increments by `len(unique_gen)`

### QN-57: G-44 cooc_masks + fb_overlaps (2 tests)
- `test_cooc_mask_matches_logic` вЂ” verify cooc mask == set of ctx cids per gen_cid
- `test_fb_overlap_tensor_shape` вЂ” verify `(ng, n_v)` shape, values correct

### QN-58: Centroid pull CPU/GPU parity fix (1 test) — ✅ RESOLVED by 024f1aa
- `test_centroid_pull_parity_0p1_factor` — verify CPU `0.1` factor matches GPU (currently gap) — ✅ **FIXED in 024f1aa**

---

## 6. Updated Coverage Gap Matrix

| Method | File:Line | Covered? | Risk | О” V10 |
|--------|:---------:|:--------:|:----:|:-----:|
| `_apply_subspace_update_batch` | `concept_space.py:591` | **вќЊ NO** | **HIGH** | **NEW** |
| GPU `_centroid_pull_batch` (GPU path) | `stdp_trainer.py:827` | **вќЊ NO** | **HIGH** | **NEW** |
| GPU write-back (deferred) | `stdp_trainer.py:466-473` | **вќЊ NO** | **HIGH** | **NEW** |
| Deferred `_vecs_t` batched write | `stdp_trainer.py:478-484` | **вќЊ NO** | **HIGH** | **NEW** |
| `_gpu_poststdp_fused` | `stdp_trainer.py:496-506` | **вќЊ NO** | **HIGH** | **NEW** |
| `checkpoint_state.json` save | `train_full.py:393-400` | **вќЊ NO** | MED | **NEW** |
| `_effective_cp` + `_rescore_cp` | `train_full.py:473-479` | **вќЊ NO** | MED | **NEW** |
| opt.json fallback search | `train_full.py:278-285` | **вќЊ NO** | MED | **NEW** |
| Batched EMA update | `stdp_trainer.py:486-489` | **вќЊ NO** | MED | **NEW** |
| cooc_masks + fb_overlaps | `stdp_trainer.py:707-720` | **вќЊ NO** | MED | **NEW** |
| `_fused_buf` reuse | `stdp_trainer.py:380-387` | **вќЊ NO** | MED | **NEW** |
| Persistent `_mom_t` tensor | `stdp_trainer.py:409-418` | вљ пёЏ partial | MED | **NEW** |
| GPU lat inh precomputed mask | `stdp_trainer.py:516-532` | вќЊ (smoke) | MED | **NEW** |
| CPU `_centroid_pull` 0.1 factor | `stdp_trainer.py:840` | **вќЊ NO** | MED | **RESOLVED** |
| `_apply_subspace_update` | `concept_space.py:556` | вњ… (QN-32) | вЂ” | FIXED |
| `_contrastive_objective_gpu` | `stdp_trainer.py:686` | вњ… (QN-33) | вЂ” | FIXED |
| `evaluate` / `_evaluate` | `stdp_trainer.py:851` | вњ… (QN-34) | вЂ” | FIXED |
| RNGRegistry (all) | `rng_registry.py` | вњ… (QN-36) | вЂ” | FIXED |
| AdaptiveErrorTracker | `adaptive_error_tracker.py` | вњ… (QN-37) | вЂ” | FIXED |
| Checkpoint cleanup | `checkpoint_manager.py` | вњ… (QN-38) | вЂ” | FIXED |
| SN-25 slow_mask theta | `stdp_trainer.py:361-366` | вќЊ (no 7-col meta) | MED | в†” unchanged |
| SN-24 momentum blend | `stdp_trainer.py:459-460` | вќЊ (smoke only) | MED | в†” unchanged |
| G-47 lerp_ EMA | `stdp_trainer.py:487-488` | вќЊ (smoke) | MED | в†” unchanged |
| SN-22.3 per-concept elr | `stdp_trainer.py:596-603` | вљ пёЏ partial | MED | в†” unchanged |

**Total high-risk uncovered: 5** (в¬†пёЏ +5 new V10)
**Total medium-risk uncovered: 12** (в¬†пёЏ +7 new V10, в€’7 old fixed)
**Total uncovered risk items: 19** (V10: 22 в†’ V11: 19 в¬‡пёЏ в€’3)

---

## 7. Summary

| Metric | V10 | V11 | О” |
|--------|:---:|:---:|:-:|
| QN-32..QN-40 implemented | 0/9 | **9/9** | в¬†пёЏвњ… |
| New tests added in V10 commits | 0 | **26** (+267 lines) | **STR** в¬†пёЏ |
| Active test count | 79 | **105** | в¬†пёЏ +26 |
| STR | ~50% | **~48%** | в¬‡пёЏ в€’2pp (new code outpaces tests) |
| HIGH-risk uncovered | 5 | **5** | (rotated: 5 old fixed, 5 new added) |
| MEDIUM-risk uncovered | 17 | **12** | в¬‡пёЏ в€’5 |
| Total uncovered risk items | 22 | **19** | в¬‡пёЏ в€’3 |
| Proposed V11 test suites | вЂ” | **10** (QN-49..QN-58, ~22 tests) | |

### Key Findings

1. **вњ… GREAT: V10 СЂРµР°Р»РёР·РѕРІР°Р» Р’РЎР• 9 QN-СЃСЋС‚РѕРІ** вЂ” 26 РЅРѕРІС‹С… С‚РµСЃС‚РѕРІ, 267 СЃС‚СЂРѕРє, 105 С‚РµСЃС‚РѕРІ РїСЂРѕС…РѕРґСЏС‚
2. **⚠️ G-42 centroid pull CPU/GPU math mismatch** — `0.1` factor present only in GPU path (parity bug) — ✅ **FIXED in 024f1aa**
3. **вљ пёЏ G-50/G-51 deferred write-back untested** вЂ” РєСЂРёС‚РёС‡РµСЃРєРёР№ GPU РїСѓС‚СЊ Р±РµР· РµРґРёРЅРѕРіРѕ С‚РµСЃС‚Р°
4. **вљ пёЏ G-52 fused post-STDP untested** вЂ” РЅРѕРІР°СЏ С‚РѕС‡РєР° РІС…РѕРґР° Р±РµР· РїРѕРєСЂС‹С‚РёСЏ
5. **вљ пёЏ 5 HIGH-risk uncovered methods remain** вЂ” РІСЃРµ РЅРѕРІС‹Рµ V10 GPU-РѕРїС‚РёРјРёР·Р°С†РёРё

**Safety verdict: IMPROVING but fragile.** Test count +33% (79→105), но V10 GPU-оптимизации (G-40..G-52) внедрены без тестового покрытия. Приоритет: QN-52 (deferred write) → QN-49 (subspace batch) → QN-50 (GPU centroid). Centroid parity bug ✅ **resolved in 024f1aa** (QN-58 updated accordingly).


