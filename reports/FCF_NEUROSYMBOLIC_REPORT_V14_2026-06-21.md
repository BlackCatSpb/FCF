# FCF Neuro-Symbolic Audit V14 — 2026-06-21

**Auditor:** Neuro-Symbolic Specialist  
**Commit:** `37550d9` (V13: P1+P2 полный цикл)  
**Scope:** `stdp_trainer.py`, `concept_space.py`, `crystal_generator.py`

---

## V13 Fix Verification

### SN-53: `fb_overlaps` int64 → int32 ✅

- **File:** `stdp_trainer.py:824` `_contrastive_objective_gpu`
- `fb_overlaps` shape `(ng, n_v)` stores count of overlapping field bits. Max value = `n_anchors` ≤ 1024 ≪ 2³¹. int32 saves 55MB at peak (ng=100).
- **Verdict:** Correct. No precision risk. All downstream `.gather()` and comparisons work identically.

### SN-54: `_sync_after_fluctuate` — без O(V·D) CPU rebuild ✅

- **File:** `crystal_generator.py:341–380`
- Replaces `_invalidate_torch()` → full `_build_torch_tensors`. New path:
  1. CPU numpy gather of `codes` (O(V·latent_dim))
  2. `_codes_t.copy_()` (O(V·latent_dim) PCIe)
  3. GPU matmul `_codes_t @ basis_t` + normalize (O(V·latent_dim·D) GPU)
- Old path did step 1 + per-concept CPU matmul + PCIe transfer of full `_vecs_t`.
- **Verdict:** Correct. Moves the heavy matmul to GPU. Per-concept loop is gone.
- **Residual issue:** Step 1 still iterates all ~146K codes with Python `for` — ~5ms per fluctuate. Acceptable.

### TN-48: `field_gate` bool → float ✅

- **All occurrences** `stdp_trainer.py` changed from `if field_gate:` to `if field_gate > 0.5`.
- Backward-compatible: old `True` → 1.0 > 0.5, old `False` → 0.0 ≤ 0.5.
- **Verdict:** Correct. Enables gradual field gating.

### G-72: lazy CPU sync (`_dirty_cids`) ✅

- `crystal_generator.py:123` — `_dirty_cids: Set[int]` tracks CIDs modified on GPU.
- All GPU update paths (`_gpu_stdp_apply`, `_lateral_inhibition_gpu`, `_negative_sampling_gpu`, `_contrastive_objective_gpu`, `_centroid_pull_batch`) now call `gen._dirty_cids.update(cids_batch)` instead of per-CID `cs._apply_vector_update`.
- Sync triggered at: `train_from_text()` → `_sync_dirty_cpu()`, `train_batch()` → `_sync_dirty_cpu()`, `_evaluate()` → `_sync_dirty_cpu()`.
- **Verdict:** Correct. Reduces PCIe transfers from O(N·D) to O(1·D) per batch.

---

## Remaining Issues

### SN-56: Qwen knowledge distillation отстутствует в GPU path

**Severity:** Medium  
**File:** `stdp_trainer.py:250–251` vs `stdp_trainer.py:419`  
**Problem:**  
Qwen knowledge factor `gen.qwen_knowledge.get_factor(ids[i], ids[j])` is only applied on the CPU path (line 251: `lr *= gen.qwen_knowledge.get_factor(...)` before building `gen_updates`). The GPU path stores raw components in `gpu_meta_l` and recomputes LR in `_gpu_stdp_core:419`, but the Qwen factor is **not included** in any metadata field and not recomputed on GPU.

**Impact:**  
Training with `qwen_knowledge` enabled + GPU path ignores Qwen distillation entirely. CPU path applies it correctly.

**Fix:**  
Either (a) store Qwen factor as field index 9 in `gpu_meta_l`, or (b) pre-multiply into `field_weight` or `pmi_w` before metadata capture.

---

### SN-57: `field_gate` float → binary threshold loss

**Severity:** Low  
**File:** `stdp_trainer.py:239, 632, 691, 737, 804`  
**Problem:**  
`field_gate` is now a float `[0.0, 1.0]`, but all guards use `if field_gate > 0.5` — a binary threshold. When `field_gate = 0.3` (weak field gating), the `> 0.5` check skips all field-gated logic (concept_error reweighting in neg-sampling, contrastive objective, field_weight computation). A `True` boolean in V12 would have applied it fully.

**Impact:**  
Partial field gating (`0.0 < field_gate < 0.5`) silently degenerates to no field gating. The gradient of `field_gate` through the training objective is lost.

**Fix:**  
Replace binary thresholds with multiplicative scaling:  
```python
# Instead of:
if field_gate > 0.5:
    contr_lr *= (1.0 + ce * 2.0)
# Use:
contr_lr *= (1.0 + ce * 2.0 * field_gate)
```

---

### SN-58: EMA stale after `_sync_after_fluctuate`

**Severity:** Low  
**File:** `crystal_generator.py:341–380`  
**Problem:**  
`_sync_after_fluctuate` updates `_vecs_t` and `_codes_t` but leaves `_ema_vecs_t` unchanged. During evaluation, `_sync_ema()` copies the stale pre-fluctuate EMA into `_vecs_t`, producing vectors that don't reflect recent fluctuate/STDP updates.

**Impact:**  
Evaluation accuracy may be biased toward pre-fluctuate state. On the other hand, EMA is intended as a stable reference — the question is whether fluctuate should affect EMA.

**Fix:**  
Add `_ema_vecs_t` refresh to `_sync_after_fluctuate`:
```python
if self._ema_vecs_t is not None:
    self._ema_vecs_t.copy_(vecs_gpu.to(torch.bfloat16), non_blocking=True)
```

---

### SN-59: GPU `_ce_t` desync from CPU `concept_error` in GPU PMI mode

**Severity:** High  
**File:** `stdp_trainer.py:450–460`  
**Problem:**  
`_gpu_stdp_core` updates `gen._ce_t[unique_gen]` on GPU (line 452). The CPU sync to `gen.concept_error` (AdaptiveErrorTracker) only happens when `gen._cf_t is None` (line 458). When GPU PMI is active (`gen._cf_t is not None`), the CPU `concept_error` is **never updated** from GPU STDP. Meanwhile, CPU-side `_apply_pmi_gate` (during generation `_branch`) reads from `gen.concept_error`, which is stale.

**Impact:**  
Generation PMI gate uses stale concept error values. PMI filtering during beam search is degraded when GPU training is active.

**Fix:**  
Always sync `avg_err` to CPU `concept_error`, regardless of `_cf_t`:
```python
avg_err_cpu = avg_err.cpu().numpy()
for gi, gen_cid in enumerate(unique_gen):
    gen.concept_error.update(gen_cid, float(avg_err_cpu[gi]))
```
Move this outside the `if gen._cf_t is None` guard.

---

### SN-60: Gradient explosion risk in `_gpu_stdp_core` freq_weight (near-zero freq)

**Severity:** Low  
**File:** `stdp_trainer.py:413`  
**Problem:**  
```python
fa_t = gen._cf_t[prev_cid_t]; fb_t = gen._cf_t[next_cid_t]
fw_t = 1.0 / (1.0 + torch.log(torch.max(...).clamp(min=1)) * 0.15)
```
When `fa_t = fb_t = 0` (never-seen concept), `clamp(min=1)` gives `freq_weight = 1.0` — same as if the concept had frequency 1. This is the same behavior as CPU path (`max(max(fa, fb), 1)`), so no regression. No fix needed.

---

### SN-61: `_sync_dirty_cpu` — redundant code round-trip

**Severity:** Low  
**File:** `crystal_generator.py:396–407`  
**Problem:**  
`_sync_dirty_cpu` calls `cs._apply_vector_update(cid, v_new)` which:
1. Stores vector (correct)
2. Re-derives fractal code: `new_code = v_new @ basis.T` (line 562)
3. Normalizes code (line 564-565)
4. Stores code in `fractal.codes[cid]`
5. Sets `fractal._matrix_dirty = True`

Steps 2–5 are redundant — GPU STDP only changed the vector, and the fractal code derived from `v_new @ basis.T` is the same as the pre-existing code (since code → vector is linear, but the inverse projection `v @ basis.T` doesn't equal the original code if basis is non-square). Actually, `basis` is `(latent_dim, dim)` with `latent_dim > dim`, so `code = v @ basis.T` is a projection back to latent space and WILL differ from the original code. This round-trip fundamentally changes the representation.

**Impact:**  
Each CPU sync after GPU STDP performs a latent projection that may drift codes. Over thousands of batches, this can accumulate. Not a bug per se — it's how the system works — but worth monitoring.

---

### SN-62: `_apply_subspace_update_batch` — missing `_codes_t` writeback (V13 fixed)

**Severity:** Fixed in V13  
**File:** `concept_space.py:636` (V13 diff)  
**Fix:** `gen._codes_t[cids_t] = new_codes.to(torch.float16)` was missing in V12. Added in V13. ✅

---

## New Proposals (SN-63+)

### SN-63: Fuse Qwen factor into GPU meta via index 9

Add Qwen factor as `gpu_meta_l` field index 9, read in `_gpu_stdp_core`, multiply into LR. This unifies CPU/GPU Qwen support.

### SN-64: Batched `_sync_dirty_cpu` with pre-allocated buffer

Replace per-CID `for` loop with one large `cpu().numpy()` + batched `_apply_vector_update` using numpy operations. Currently `_sync_dirty_cpu` calls `_apply_vector_update` per CID, each doing code re-derivation + norm + dict insert. Could batch.

### SN-65: `_gpu_stdp_core` NaN guard for `gradient_noise_scale`

Line 441: `acc += torch.randn_like(acc) * gradient_noise_scale * (elr_grouped[:, None] / elr_grouped.max().clamp(min=1))`. When `elr_grouped.max() → ∞` (rare), the division is safe due to clamp. But when all `elr_grouped = 0`, `ratio = 0/1 = 0`, noise is zeroed — correct. No action needed.

---

## Summary

| Issue | Status | Severity |
|-------|--------|----------|
| SN-53: fb_overlaps int64→int32 | ✅ Fixed | Low |
| SN-54: _sync_after_fluctuate | ✅ Fixed | High |
| TN-48: field_gate bool→float | ✅ Fixed | Medium |
| G-72: lazy CPU sync | ✅ Fixed | High |
| **SN-56: Qwen missing from GPU** | **❌ Open** | **High** |
| **SN-57: field_gate binary threshold** | **❌ Open** | **Medium** |
| **SN-58: EMA stale after fluctuate** | **❌ Open** | **Low** |
| **SN-59: GPU _ce_t → CPU desync** | **❌ Open** | **High** |
| SN-60: freq_weight edge case | ✅ Acceptable | Low |
| SN-61: code round-trip drift | ⚠️ Monitor | Low |
| SN-62: _codes_t writeback | ✅ Fixed (V13) | High |

**Critical:** Fix SN-56 (Qwen GPU) and SN-59 (_ce_t sync) before V14 release.  
**Recommended:** Fix SN-57 for `field_gate` gradual support.  
**Optional:** SN-58, SN-64 (optimization).
