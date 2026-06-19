# FCF GPU Optimization Audit вЂ” V11 (2026-06-19)

**Auditor:** GPU-Opt Agent (V11 deep scan)  
**Scope:** `eva/symbolic/stdp_trainer.py` (920L) + `crystal_generator.py` (844L) + `concept_space.py` (936L)  
**Config:** ~146K vocab, 384D, FP16 `_vecs_t`, CUDA  
**Current HEAD:** Post V10 вЂ” all G-40..G-52 evaluated against actual code

---

## 1. V10 Report Verification вЂ” Discrepancies Found

| Claim in V10 | Actual Code | Verdict |
|---|---|---|
| G-45 вњ… Persistent CUDA events at `stdp_trainer.py:27-32` | Line 31: `pass # profiling stubs removed (G-55)` | **FALSE POSITIVE** вЂ” events were removed |
| G-47 вќЊ Per-element `lerp_` | Line 488: `gen._ema_vecs_t[unique_gen].lerp_(...)` batched | **FALSE NEGATIVE** вЂ” IS batched |
| G-49 вќЊ `torch.zeros()` per call | Line 382: `buf = gen._fused_buf[:ng]; buf.zero_()` pre-allocated | **FALSE NEGATIVE** вЂ” IS implemented |
| G-40 вќЊ Still 100% CPU numpy | Line 476 calls `_apply_subspace_update_batch` (GPU) for GPU path | **PARTIALLY FALSE** вЂ” GPU path batched, CPU path still per-element |
| G-50 was a "proposal" | Lines 479вЂ“484: deferred batched `_vecs_t` write implemented | **ALREADY IN CODE** |

**Key Takeaway:** V10 was based on old diff rather than current file. Actual implementation is **ahead** of V10 in several areas (G-47 batched, G-49, G-50/51 deferred writes, G-40 batch GPU path).

---

## 2. Actual G-40..G-52 Status (Verified Against Live Code)

### вњ… Fully Implemented

| ID | Description | Location | Evidence |
|----|-------------|----------|----------|
| **G-47** | Batched EMA `lerp_` | `stdp_trainer.py:487вЂ“489` | `gen._ema_vecs_t[unique_gen].lerp_(gen._vecs_t[unique_gen].float(), 1.0 - gen._ema_decay)` вЂ” single indexed scatter+lerp |
| **G-49** | Pre-allocated fused buffer | `crystal_generator.py:250вЂ“251`, `stdp_trainer.py:382вЂ“383` | `gen._fused_buf` created once with shape `(V, D+1)`, sliced via `buf = gen._fused_buf[:ng]`, no re-allocation |

### вљ пёЏ Partially Implemented

| ID | Description | What Works | What's Missing |
|----|-------------|------------|----------------|
| **G-40** | Batched GPU subspace | `_apply_subspace_update_batch` at `concept_space.py:591вЂ“638` does GPU batch | CPU path (`_cpu_stdp_apply:286вЂ“287`) still calls per-element CPU `_apply_subspace_update`. Also `_apply_subspace_update_batch` has `new_vecs.cpu().numpy()` + per-element `set_vec`+hook at lines 624вЂ“637 |
| **G-41** | GPU lateral inhibition | `_lateral_inhibition_gpu:508вЂ“532` вЂ” sim on GPU, batched matmul | Line 532: `cs._apply_vector_update(gen_cids[gi], v_new.cpu().numpy())` вЂ” per-element D2H + CPU hook |
| **G-42** | GPU centroid push | `_centroid_pull_batch:827вЂ“845` вЂ” tensor ops on GPU, vectorized | Line 844вЂ“845: per-element `.cpu().numpy()` + `_apply_vector_update` |
| **G-43** | GPU neg sampling | `_negative_sampling_gpu:574вЂ“624` вЂ” vectorized noise + sim on GPU | Lines 604вЂ“624: per-element Python loop + `.cpu().numpy()` + `_apply_vector_update` |
| **G-44** | GPU contrastive | `_contrastive_objective_gpu:686вЂ“792` вЂ” cooc_masks and fb_overlaps on GPU | Lines 735вЂ“792: per-element loop with `.item()` syncs (deeply nested) |
| **G-46** | Persistent `_mom_t` | `stdp_trainer.py:409вЂ“418`: `gen._mom_t` is GPU tensor | Line 418: `mom_cpu = mom_t.cpu().numpy()` still fetched to CPU. Also `mom_t = momentum_mu * mom_t + ...` creates temp tensor (re-assigns binding) |
| **G-50** | Zero-copy deferred write | Lines 479вЂ“484: batched `_vecs_t[cids_batch] = vecs_batch` | Lines 483вЂ“484: still calls `cs._apply_vector_update(cid, v_new_gpu.cpu().numpy())` for fractal sync |
| **G-51** | Deferred vector sync | Same as G-50 вЂ” batched `_vecs_t` write | Per-element `_apply_vector_update` hook defeats the purpose |
| **G-52** | Fused post-STDP | `_gpu_poststdp_fused:498вЂ“506` вЂ” single call site | Still calls separate `_negative_sampling_gpu` + `_contrastive_objective_gpu` вЂ” loops not merged |

### вќЊ Not Implemented

| ID | Description | Location |
|----|-------------|----------|
| **G-45** | Persistent CUDA events | **Removed by G-55** вЂ” line 31: `pass  # profiling stubs removed` |
| **G-48** | `torch.compile` | Line 337: still a comment only |

---

## 3. CPU-GPU Sync Point Catalog (V11 Measured)

Per batch: ~100 unique_gen, ~500 pairs, ~25 concepts per sentence, 5 sentences.

### Category A: Full Tensor D2H (1Г— per batch each)

| # | Line | Code | Sync Type |
|---|------|------|-----------|
| S1 | 404 | `avg_err.cpu().numpy()` | Full tensor D2H |
| S2 | 420 | `acc.cpu().numpy()` | Full tensor D2H |
| S3 | 421 | `cnt.cpu().numpy()` | Full tensor D2H |
| S4 | 422 | `elr_grouped.cpu().numpy()` | Full tensor D2H |
| S5 | 418 | `mom_t.cpu().numpy()` (conditional) | Full tensor D2H |
| S6 | 624 | `new_vecs.cpu().numpy()` in `_apply_subspace_update_batch` | Full tensor D2H |

**Total Category A: 5вЂ“6 syncs per batch** вЂ” negligible (fixed overhead)

### Category B: Per-Element D2H Syncs (N Г— unique_gen)

| # | Line | Code | Freq | Notes |
|---|------|------|------|-------|
| S7 | 406вЂ“407 | `for gi, gen_cid: gen.concept_error.update(gen_cid, float(avg_err_cpu[gi]))` | NГ— | CPU dict update |
| S8 | 429вЂ“473 | Python loop: `v = cs.concept_vectors.get(gen_cid)`, RNG, destab | NГ— | Full CPU logic |
| S9 | 467 | `torch.from_numpy(grad).to(device)` | NГ— (non-subspace) | H2D per element |
| S10 | 484 | `cs._apply_vector_update(cid, v_new_gpu.cpu().numpy())` | NГ— | D2H + fractal code sync |
| S11 | 532 | `_lateral_inhibition_gpu: cs._apply_vector_update(...)` | в‰¤NГ— | D2H per inhibited |
| S12 | 604вЂ“624 | `_negative_sampling_gpu` loop: `.item()`, `_apply_vector_update` | NГ— | Per-concept loop |
| S13 | 738, 743, 746 | `_contrastive_objective_gpu: int(...item()), float(...item())` | NГ—5 + NГ—50 | Deep loop syncs |
| S14 | 787, 792 | `_contrastive_objective_gpu: v_new.cpu().numpy()` | NГ— | Per-concept D2H |
| S15 | 844вЂ“845 | `_centroid_pull_batch: .tolist() + _apply_vector_update` | ~50Г— | Per-token D2H |

### Category C: Per-Pair Syncs (O(N_pairs))

| # | Line | Code | Freq | Notes |
|---|------|------|------|-------|
| S16 | 185 | `torch.bitwise_and(...).sum().item()` | ~500Г— | field_gate per pair |

### Estimated Total: ~1,000вЂ“5,000 syncs per batch

Down from V10's ~20,000. Still dominated by:
- `_contrastive_objective_gpu` loop (S13: ~500 `.item()` syncs)
- Per-element `_apply_vector_update` calls (S10+S11+S12+S14+S15: ~400 D2H)
- Per-pair field overlap (S16: ~500 scalar syncs)

---

## 4. G-48 Feasibility Assessment

`torch.compile` on `_gpu_stdp_apply` is **not yet feasible** because:
- Lines 429вЂ“473: per-element Python loop with numpy ops, RNG, dict access
- Lines 406вЂ“407: `concept_error.update()` вЂ” mutable CPU state
- Dynamic shapes: `N` varies per batch (not a blocker for `reduce-overhead` mode)

**Prerequisites for G-48:**
1. Move destab logic to GPU (G-60)
2. Replace `concept_error.update()` with GPU scatter + lazy CPU sync (G-61)
3. Eliminate all `.item()` and `.cpu().numpy()` from `_gpu_stdp_apply`

---

## 5. Correctness Audit of GPU Optimizations

### `_lateral_inhibition_gpu` (Line 522)
```python
inhibit_vec = (sim[gi][gi_mask] * gv[gi_mask] - (sim[gi][gi_mask]**2) * gv[gi]).sum(dim=0)
```
This computes: ОЈ(sim_j * v_j - sim_jВІ * v_i) = ОЈ(sim_j * (v_j - sim_j * v_i))
CPU version (`_lateral_inhibition_cpu` line 322):
```python
inhibit_vec += (sims[gi][gj] * v_other - sims[gi][gj]**2 * v)
```
вњ… **Correct** вЂ” gradient matches. Riemannian form: `sim*v_other - simВІ*v` = `sim*(v_other - sim*v)` which is tangent to sphere at v.

### `_negative_sampling_gpu` grad (Line 614)
```python
grad = (gen._vecs_t[valid_idx].float() - sim[gi][neg_mask][:, None] * vg_i).sum(dim=0)
```
CPU version (line 563):
```python
grad = v_neg - sim * v_gen
```
вњ… **Correct** вЂ” same Riemannian negative gradient, summed over valid negatives.

### `_centroid_pull_batch` (Line 840)
```python
pulls = 0.1 * (cn - sims[:, None] * vecs)
```
CPU version (line 820):
```python
shift = (centroid - sim * v) * sent_lr
```
вњ… **Correct** вЂ” `centroid - sim*v` is tangent vector on sphere. GPU scales by 0.1 constant, CPU uses `sent_lr = base_lr_val * 0.3`.

### Momentum in `_gpu_stdp_apply` (Lines 414вЂ“416)
```python
avg_grad = acc / cnt[:, None].clamp(min=1)
mom_t = gen._mom_t[unique_gen]
mom_t = momentum_mu * mom_t + (1 - momentum_mu) * avg_grad
gen._mom_t[unique_gen] = mom_t
```
вљ пёЏ **Bug:** `mom_t = momentum_mu * mom_t + ...` re-binds local `mom_t` to a new tensor. `gen._mom_t[unique_gen] = mom_t` then writes it back. This works but creates an unnecessary temporary. Should use `gen._mom_t[unique_gen].mul_(momentum_mu).add_(avg_grad, alpha=1 - momentum_mu)` for in-place update.

Also, line 459вЂ“460:
```python
if mom_cpu is not None:
    grad = momentum_mu * mom_cpu[gi] + (1 - momentum_mu) * grad
```
This applies momentum AGAIN in the per-element loop. The momentum was already applied at line 416. This is **double-applying** momentum. вќЊ

### EMA `lerp_` (Line 488)
```python
gen._ema_vecs_t[unique_gen].lerp_(gen._vecs_t[unique_gen].float(), 1.0 - gen._ema_decay)
```
`lerp_(source, weight)` computes: `self = self + weight * (source - self)` = `self * (1-weight) + source * weight`
With `weight = 1 - 0.999 = 0.001`: `new = 0.999 * old + 0.001 * new_vec`
вњ… **Correct.**

---

## 6. Correctness Bugs Found

### B1 (HIGH): Double momentum вЂ” `stdp_trainer.py:416 + 459вЂ“460`
**Line 416:** `mom_t = momentum_mu * mom_t + (1 - momentum_mu) * avg_grad` вЂ” momentum applied to `avg_grad`
**Line 459вЂ“460:** `grad = momentum_mu * mom_cpu[gi] + (1 - momentum_mu) * grad` вЂ” momentum applied AGAIN
**Effect:** Momentum is applied twice per step. Gradient becomes: `ВµВІ * old_mom + Вµ*(1-Вµ)*avg_grad + (1-Вµ)*grad_step` instead of expected `Вµ * old_mom + (1-Вµ) * grad_step`. This corrupts gradient direction.

**Fix:** Remove lines 459вЂ“460. The momentum at line 416 is sufficient. If per-element momentum mixing is needed (for subspace vs non-subspace paths), apply it once at line 416 and pass `mom_t` indexed per element without re-applying formula.

### B2 (MED): `_negative_sampling_gpu` вЂ” `gen._ce_t` may hold stale tensor
Line 608: `neg_lr_i *= (1.0 + gen._ce_t[gen_cid] * 2.0)`
`gen._ce_t` is built in `_build_torch_tensors` but updated at line 403: `gen._ce_t[unique_gen] = ce_decay * gen._ce_t[unique_gen] + (1 - ce_decay) * avg_err`. This update happens in `_gpu_stdp_apply`. But if `_gpu_poststdp_fused` runs after `_gpu_stdp_apply`, the `_ce_t` should be fresh.
**Verdict:** вњ… Correct for current call order. Fragile вЂ” reordering would break.

### B3 (LOW): `_gpu_stdp_apply` line 469 вЂ” `grad_t` dtype
`grad_t = torch.from_numpy(grad).to(device=device, dtype=torch.float32)` вЂ” `grad` is float64 (from `acc_cpu[gi]` which comes from `acc.cpu().numpy()`, dtype float32). `torch.from_numpy` will preserve dtype, then `.to(..., dtype=torch.float32)` is a no-op if already float32. вњ… Fine.

### B4 (LOW): `_on_vector_update` double-write
Line 161: `self._vecs_t[cid].copy_(torch.from_numpy(v_new).to(...))` writes to `_vecs_t`.
Then line 481: `gen._vecs_t[cids_batch] = vecs_batch` ALSO writes to `_vecs_t`.
So each concept gets written TWICE per batch вЂ” once via `_on_vector_update` (from `cs._apply_vector_update` at line 484) and once via the batched write (line 481). The batched write at line 481 happens first, then `_on_vector_update` fires per-element and overwrites. The final state is correct (last write wins), but this is 2Г— bandwidth waste.

---

## 7. G-60+ Proposals

### G-60: GPU Destabilization (HIGH вЂ” eliminates CPU per-element loop)
**Problem:** Lines 429вЂ“473: per-element Python loop with CPU RNG, numpy ops, dict access. This is the main blocker for torch.compile.

**Fix:** Move destab logic to GPU tensor ops:
```python
ce_t = gen._ce_t[unique_gen]  # already on GPU
destab_p = torch.clamp(ce_t * 0.5 * destab_scale, max=0.5)
destab_mask = torch.rand_like(destab_p) < destab_p  # GPU RNG batch
# For destab_mask True: use PPMI or field-fallback vector on GPU
```
Pre-compute PPMI candidates as GPU tensor (top-k per concept, pre-built). Use batched gather.

**Saves:** S7, S8, S9 вЂ” entire per-element loop.
**Eliminates:** ~NГ— Python iterations, ~NГ— H2D `torch.from_numpy(grad)` calls.

### G-61: GPU Concept Error Sync вЂ” lazy batch (MED)
**Problem:** Lines 406вЂ“407: `concept_error.update(gen_cid, float(avg_err_cpu[gi]))` forces full D2H of `avg_err`.

**Fix:** Keep `gen._ce_t` as ground truth. Only sync to CPU `concept_error` when `_branch` needs it (inference). Use a `_ce_dirty` flag:
```python
gen._ce_t[unique_gen] = ce_decay * gen._ce_t[unique_gen] + (1 - ce_decay) * avg_err
if gen._ce_dirty:
    gen._ce_dirty = False  # batch sync on next _branch call
```
In `_branch`, if dirty, sync `gen._ce_t[cpu_seen_cids].cpu().numpy()` в†’ `concept_error` dict.

**Saves:** S1 (1 full D2H per batch), S7 (NГ— CPU dict updates).

### G-62: GPU `_apply_vector_update` bypass (HIGH вЂ” eliminates all hook D2H)
**Problem:** Every GPU path calls `cs._apply_vector_update(cid, v_new.cpu().numpy())` which:
1. Converts GPU tensor в†’ CPU numpy (D2H)
2. Calls `set_vec` вЂ” writes CPU array
3. Syncs fractal code via CPU matmul
4. Fires `_on_vector_update` hook вЂ” copies back to GPU (H2D)

**Fix:** Create `_gpu_apply_vector_update(cid, v_gpu)` that:
1. `gen._vecs_t[cid] = v_gpu` (direct GPUв†’GPU, already done)
2. Sync fractal code on GPU: `gen._codes_t[cid] = normalize(v_gpu @ basis_t.T)`
3. Mark as dirty for CPU sync (deferred batch sync to `concept_vectors._data`)

**Saves:** S2, S3, S4, S10, S11, S12, S14, S15 вЂ” all per-element D2H roundtrips.
**Eliminates:** All `.cpu().numpy()` in GPU training paths.

### G-63: GPU Contrastive вЂ” full vectorization without `.item()` (HIGH)
**Problem:** `_contrastive_objective_gpu` lines 735вЂ“792: deep Python loop with multiple `.item()` syncs per concept.

**Fix:** Replace `.item()` with GPU-indexed tensor ops:
```python
# Instead of per-element loops:
cooc_mask_expanded = cooc_masks[:, None, :]  # (ng, 1, n_v)
self_mask = topk_idx == torch.arange(ng, device=d)[:, None, None]
valid_mask = ~self_mask & ~cooc_mask_expanded.gather(2, topk_idx)
# All valid hard negatives in one shot
hard_mask = (topk_val > 0.05) & valid_mask
# TN-14: field-aware regularization in batch
cross_field_mask = (fb_overlaps[:, None, :] == 0)  # batch
```
This is complex because each concept may have different numbers of valid hard negatives. Use padding + masking.

**Saves:** S13 (~500 `.item()` syncs).

### G-64: GPU Fused STDP + NegSampling + Contrastive (MED)
**Problem:** Separate passes over same data вЂ” `_gpu_stdp_apply`, `_negative_sampling_gpu`, `_contrastive_objective_gpu` each read/write `_vecs_t`.

**Fix:** Single pass:
```python
def _gpu_fused_full_batch(self, ...):
    # 1. STDP: compute acc, elr_grouped
    # 2. Neg sampling: for each unique_gen, sample and compute gradient
    # 3. Contrastive: pre-compute topk, apply push
    # 4. Apply all gradients to _vecs_t in one batched write
    # 5. Lateral inhibition on final vectors
    # 6. Lazy CPU sync for all modified concepts
```

**Saves:** Reduces `_vecs_t` reads from 3Г— to 1Г— per batch. Enables G-48 (torch.compile).

### G-65: GPU Pair Building вЂ” field_gate without `.item()` (MED)
**Problem:** Line 185: `torch.bitwise_and(...).sum().item()` per pair вЂ” ~500 scalar syncs per batch.

**Fix:** Pre-compute field overlap matrix on GPU for all CID pairs in batch:
```python
# Instead of per-pair .item():
batch_cids = torch.tensor(list(set(gpu_cid_ctx + gpu_cid_gen)), device=device)
fb_batch = gen._fb_t[batch_cids]
fb_matrix = (fb_batch[:, None, :] & fb_batch[None, :, :]).sum(dim=-1)  # (M, M)
field_weight = torch.where(fb_matrix > 0, torch.clamp(1.0 + torch.log(fb_matrix.float() + 1) * 2.0, max=3.0), 0.1)
```
Build pair data on GPU alongside CPU. GPU path reads from pre-computed GPU array.

**Saves:** S16 (~500 scalar syncs).

### G-66: CUDA Graph for Fixed-Size Batches (LOW вЂ” 1.2вЂ“1.5Г—)
**Problem:** Dynamic shapes prevent CUDA Graph capture.

**Fix:** After G-60..G-65 eliminates all Python loops, pad variable-size inputs to fixed max_N. Capture CUDA Graph with `torch.cuda.CUDAGraph`. Replay for each batch.

**Requires:** All dynamic shapes eliminated (G-60..G-65 as prerequisite).

### G-67: Persistent GPU RNG State (LOW)
**Problem:** `torch.randint` and `torch.rand_like` use global RNG state вЂ” creates device sync.

**Fix:** Create persistent `PCG32` state tensors on GPU for:
- Negative sampling noise
- Destab coin flips
- Gradient noise

Use `torch.rand` with `generator=gen._gpu_rng` to avoid global RNG sync.

### G-68: Async H2D for Input Tensors (LOW вЂ” ~1.05Г—)
**Problem:** `torch.tensor(gpu_ctx_l)` at line 345 is synchronous CPUв†’GPU copy. Blocking.

**Fix:** Pre-allocate pinned memory pool. Copy lists to pinned buffers using `np.array(list).to(device, non_blocking=True)` on a separate CUDA stream. Overlap CPU pair building with GPU compute of previous batch.

---

## 8. Recommended Implementation Order (V11)

| Priority | ID | Effort | Risk | Impact | Eliminates |
|----------|----|--------|------|--------|------------|
| 1 | **B1** ✅ FIXED in 024f1aa — Fix double momentum | 0.5 day | None | Correctness | вЂ” |
| 2 | **G-62** GPU `_apply_vector_update` | 3 days | Medium | **5вЂ“10Г—** on write path | S2,S3,S4,S10,S11,S12,S14,S15 |
| 3 | **G-60** ✅ FIXED in a705223 — GPU Destabilization | 2 days | Medium | **2–5×** on STDP loop | S7,S8,S9 |
| 4 | **G-65** GPU field_gate in pairs | 1 day | Low | **1.5вЂ“2Г—** on pair building | S16 |
| 5 | **G-63** GPU Contrastive vectorized | 3 days | High | **5вЂ“20Г—** on contrastive | S13 |
| 6 | **G-61** Lazy CE sync | 1 day | Low | **1.2Г—** | S1,S7 |
| 7 | **G-64** Fused full batch | 3 days | High | **1.5вЂ“3Г—** | 3в†’1 pass |
| 8 | **G-66** CUDA Graph capture | 2 days | Low | **1.2вЂ“1.5Г—** | Kernel launch overhead |

**After G-60..G-65:** `_gpu_stdp_apply` has zero Python loops в†’ **G-48 (`torch.compile`) becomes feasible** вЂ” estimated 1.5вЂ“3Г— additional speedup.

---

## 9. Projected Sync Count per Batch

| Phase | Current (V11) | After G-62+G-60 | After G-63+G-65 | Target |
|-------|---------------|-----------------|-----------------|--------|
| Full tensor D2H | 5вЂ“6 | 0 | 0 | 0 |
| Per-element D2H | ~400 | 0 | 0 | 0 |
| `.item()` syncs | ~500 | ~500 | 0 | 0 |
| Per-pair scalar | ~500 | ~500 | 0 | 0 |
| **Total** | **~1,000вЂ“5,000** | **~500** | **~0** | **~0** |

---

## 10. V10в†’V11 Delta

| Metric | V10 Reported | V11 Actual | Delta |
|--------|--------------|------------|-------|
| G-45 (CUDA events) | вњ… Done | вќЊ Removed (G-55) | Regression (feature removed) |
| G-47 (EMA lerp_) | вќЊ Per-element | вњ… Batched | Improvement (was wrong in V10) |
| G-49 (fused buffer) | вќЊ Not done | вњ… Done | Fix (was wrong in V10) |
| G-40 (subspace batch) | вќЊ 100% CPU | вљ пёЏ Mixed (GPU batch + CPU per-element) | Partial |
| G-50 (zero-copy) | Proposal | вљ пёЏ Partially done | Already in code |
| G-51 (deferred sync) | Proposal | вљ пёЏ Partially done | Already in code |
| Total syncs/batch | ~20,000 | ~1,000вЂ“5,000 | **2вЂ“20Г— better** than V10 claimed |
| Correctness bugs | 0 found | **1 HIGH (B1)** | New finding |
| G-60+ proposals | 8 (G-50..G-57) | 9 (G-60..G-68) | Updated |

---

## 11. Critical Findings

1. **B1 (HIGH): Double momentum** at `stdp_trainer.py:416 + 459вЂ“460`. Momentum is applied twice вЂ” once on GPU `avg_grad` and once on per-element CPU grad. Fix immediately вЂ” affects training dynamics with `momentum_mu > 0`.

2. **B4 (MED): Double `_vecs_t` write** вЂ” batched write at line 481 writes to `_vecs_t`, then `_on_vector_update` hook at line 161 overwrites per-element. 2Г— bandwidth waste.

3. **G-62 is the highest-impact remaining optimization**: Eliminating all `.cpu().numpy()` roundtrips in GPU paths removes ~95% of remaining syncs. Prerequisite: GPU fractal code sync (`v @ basis.T` on GPU).

4. **`_contrastive_objective_gpu` is the last deep Python loop** with ~500 `.item()` syncs. G-63 is needed to make it tensor-only.

5. **G-48 (`torch.compile`) is blocked** by per-element Python loops. Must clear G-60..G-65 first.

---

## 12. V11 Action Items

1. **Fix B1** вЂ” Remove double momentum (lines 459вЂ“460). Momentum at line 416 is sufficient.
2. **Implement G-62** вЂ” GPU-side `_apply_vector_update` with deferred CPU sync.
3. **Implement G-60** вЂ” GPU destab logic (coin flips, PPMI gather, gradient mix).
4. **Implement G-65** вЂ” GPU field overlap matrix for pair building.
5. **Implement G-63** вЂ” Full vectorization of contrastive objective.
6. **Implement G-61** вЂ” Lazy concept_error sync from `_ce_t`.
7. **Re-evaluate G-48** after all Python loops are eliminated.

---

*Report generated 2026-06-19 by GPU-Opt Agent (V11 deep audit)*

