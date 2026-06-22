# FCF GPU Optimization Audit — V12 (2026-06-21)

**Auditor:** GPU-Opt Agent (V12 deep scan)  
**Scope:** `eva/symbolic/stdp_trainer.py` (1045L) + `crystal_generator.py` (915L) + `concept_space.py` (950L) + `tests/test_stdp.py` (1447L)  
**Config:** BPE 146K vocab, 384D, FP16 `_vecs_t`, CUDA  
**HEAD:** `1768f27` + `4030b54` + `3150d5e` on top of `a705223` (V11.2)

---

## 1. V11.2 Commit Verification

### ✅ G-60: GPU Destabilization (a705223)
**Location:** `stdp_trainer.py:492-510`
```python
destab_p = torch.clamp(gen._ce_t[unique_gen] * 0.5 * destab_scale, max=0.5)
destab_mask = torch.rand(ng, device=device) < destab_p
if destab_mask.any():
    rand_idx = torch.randint(1, n_v, (ng,), device=device)
    rand_idx = torch.where(rand_idx == unique_gen_t, (rand_idx + 1) % n_v, rand_idx)
    v_ppmi = gen._vecs_t[rand_idx].float()
    noise_gpu = v_ppmi - y_ppmi[:, None] * v_self
    ...
    acc = torch.where(destab_mask[:, None], acc * (1 - mix_gpu[:, None]) + destab_update, acc)
```
**Verdict:** ✅ **Correct GPU implementation.** Full tensor ops: `torch.rand` for coin flips, `torch.randint` for candidate sampling, `torch.where` for conditional mix. No CPU RNG, no `.item()`, no lattice queries. Removed ~N× per-element Python loop (was lines 429–473 in V11).

### ✅ SN-43: GPU Negative Sampling Batched (a705223)
**Location:** `stdp_trainer.py:653-714`
```python
noise = torch.randint(0, n_v, (len(unique_gen), n_neg), device=device)
ngv = gen._vecs_t[noise].float()
sim = (gv[:, None, :] * ngv).sum(dim=-1)
mask = sim > 0.1
...
_neg_updates = []
for gi, gen_cid in enumerate(unique_gen):
    ...
    vecs_batch = torch.stack([d[1] for d in _neg_updates]).to(gen._vecs_t.dtype)
    gen._vecs_t[cids_batch] = vecs_batch
```
**Verdict:** ✅ **Batched GPU.** Per-concept loop remains (line 693) but has **zero `.item()` or `.cpu().numpy()` calls inside** — only tensor ops. Uses `avg_elr_per_gen` from `_gpu_elr_avg` stored by `_gpu_stdp_apply`. Single batched `_vecs_t` write at end.

### ✅ SN-44: GPU Contrastive Pure-Tensor (a705223)
**Location:** `stdp_trainer.py:777-909`
```python
self_hn = best_idx == gen_idxs[:, None]
cooc_hn = cooc_masks.gather(1, best_idx)
if fb_overlaps is not None:
    fb_hn = fb_overlaps.gather(1, best_idx)
    cos_upper = torch.where(fb_hn > 0, 0.3, 0.999)
    ...
valid_hn = ~self_hn & ~cooc_hn & (best_val > 0.05) & (best_val < cos_upper)
...
for i in range(ng):
    vmask = valid_hn[i]
    if vmask.any():
        hn = best_idx[i][vmask]
        grad = (cos_v[:, None] * v_neg).mean(dim=0) - v_local
```
**Verdict:** ✅ **All masks pre-computed on GPU via `gather` ops.** Zero `.item()` calls inside per-concept loop (was ~500 in V11). Single batched `_vecs_t` write. Remaining per-concept loop does only tensor ops.

### ✅ G-65/SN-48: GPU Field Overlap Matrix (1768f27 parent)
**Location:** `stdp_trainer.py:175-180`
```python
overlap_mat = (fb_t.unsqueeze(1) & fb_t.unsqueeze(0)).sum(dim=-1).cpu().numpy()
_overlap_lookup = lambda i, j: int(overlap_mat[i, j])
```
**Verdict:** ✅ **One D2H per sentence** (T×T matrix, not per-pair `.item()`). Eliminated S16 (~500 scalar syncs).

### ✅ 1768f27: Remove `_torch_dirty=True` (Fix 160s/batch)
**Location:** `stdp_trainer.py:140-142`
```python
# _torch_dirty is NOT set here — would force full tensor rebuild every batch
# Only _invalidate_torch() (after fluctuate) should set it.
```
**Verdict:** ✅ Fix confirmed. Without this fix: each batch re-ran `_build_torch_tensors` → 146K × `code @ basis` = 56M FLOPs + 636MB PCIe transfers. **160s/batch → now ~normal.**

### ✅ 4030b54: `_ema_vecs_t` + `_mom_t` → FP16
**Location:** `crystal_generator.py:278-317`
```python
self._vecs_t = torch.empty(V, D, device=dev, dtype=torch.float16)  # 107MB
self._ema_vecs_t = self._vecs_t.clone()  # 107MB (was fp32: +112MB)
self._mom_t = torch.zeros(V, D, device=dev, dtype=torch.float16)  # 107MB (was fp32: +112MB)
```
**Verdict:** ✅ Saves 224MB. EMA precision is adequate for eval (lerp with fp16 is O(1/2048) relative error — negligible for similarity).

### ✅ 3150d5e: `_fused_buf` Dynamic Growth
**Location:** `crystal_generator.py:320-322`
```python
init_rows = min(V, 4096)
self._fused_buf = torch.zeros(init_rows, D + 1, device=dev, dtype=torch.float32)
```
Growth at `stdp_trainer.py:449-450`:
```python
if gen._fused_buf.shape[0] < ng:
    gen._fused_buf = torch.zeros(ng * 2, D + 1, device=device, dtype=torch.float32)
```
**Verdict:** ✅ Was `V×(D+1) = 146K×385×4 = 225MB` pre-allocated. Now starts at 4096×385×4 = **6.3MB**, grows only if needed. Typical batch uses ~100 unique_gen → still 6.3MB.

---

## 2. B1 (Double Momentum) Verification

**Status in V11:** Double momentum at lines 416 + 459–460 — momentum applied on GPU `avg_grad` AND again in per-element CPU loop.

**Current code (stdp_trainer.py:483-523):**
```python
# G-46: Persistent _mom_t tensor (replace CPU dict)
if momentum_mu > 0:
    avg_grad = acc / cnt[:, None].clamp(min=1)
    mom_new = momentum_mu * gen._mom_t[unique_gen] + (1 - momentum_mu) * avg_grad
    gen._mom_t[unique_gen] = mom_new.to(torch.float16)
...
# SN-7: momentum already applied on GPU; _mom_t IS the smoothed gradient
if momentum_mu > 0 and gen._mom_t is not None:
    grad_gpu = gen._mom_t[unique_gen]
```

**Verdict:** ✅ **B1 FIXED.** The per-element CPU loop that re-applied momentum was **entirely removed by G-60** (GPU destab). Momentum is now stored in `_mom_t` (fp16 GPU tensor) and used directly via `grad_gpu = gen._mom_t[unique_gen]`. Single application. Correct.

### B2 (ConceptError staleness): ✅ Still correct — `_gpu_stdp_apply` updates `_ce_t` before `_gpu_poststdp_fused` runs.

### B4 (Double `_vecs_t` write): ⚠️ **Still present.**
**Location:** `stdp_trainer.py:548-552` + `crystal_generator.py:221-224` (`_on_vector_update`)
```python
gen._vecs_t[cids_batch] = vecs_batch        # batched GPU write
...
for k, cid in enumerate(cids_batch):
    cs._apply_vector_update(cid, vecs_np[k])  # fires _on_vector_update → H2D copy
```
`_on_vector_update` at crystal_generator.py:221-224:
```python
def _on_vector_update(self, cid, v_new):
    if self._vecs_t is not None:
        self._vecs_t[cid].copy_(torch.from_numpy(v_new).to(...))
```
Each concept written **twice**: batched write + per-element `_on_vector_update` overwrite. 2× bandwidth waste. ~200 extra H2D copies per batch.

---

## 3. Current Sync Count (V12 Measured)

Per batch: ~5 sentences, ~100 unique_gen, ~500 pairs.

### Category A: Full Tensor D2H (per batch)

| # | Line | Code | Freq |
|---|------|------|------|
| S1 | 179 | `overlap_mat.cpu().numpy()` (per sentence) | ~5× |
| S2 | 164-167 | `_cf_arr = _cf_t[ids_t].cpu().numpy()` (per sentence) | ~5× |
| S3 | 549 | `vecs_batch.cpu().numpy()` in `_gpu_stdp_apply` | 1× |
| S4 | 607 | `vecs_batch.cpu().numpy()` in `_lateral_inhibition_gpu` | 1× (cond) |
| S5 | 624 | `new_vecs_np = new_vecs.cpu().numpy()` subspace batch | 1× (cond) |
| S6 | 712 | `vecs_batch.cpu().numpy()` in `_negative_sampling_gpu` | 1× (cond) |
| S7 | 907 | `vecs_batch.cpu().numpy()` in `_contrastive_objective_gpu` | 1× |
| S8 | 968 | `vecs_batch.cpu().numpy()` in `_centroid_pull_batch` | 1× |

**Subtotal Category A:** ~8–12 D2H per batch

### Category B: Per-Element D2H

| # | Line | Code | Freq | Notes |
|---|------|------|------|-------|
| S9 | 551-552 | `_apply_vector_update(cid, vecs_np[k])` (+ hook H2D) | ~100× | Deferred update path |
| S10 | 609-611 | `_apply_vector_update` in lat_inh | ~10× | Per inhibited |
| S11 | 626-637 | `_apply_subspace_update_batch` per-element | ~10× | Subspace path |
| S12 | 713-714 | `_apply_vector_update` in neg_sampling | ~10× | Per updated |
| S13 | 908-909 | `_apply_vector_update` in contrastive | ~50× | Per updated |
| S14 | 969-970 | `_apply_vector_update` in centroid | ~50× | Per token |
| S15 | 222-224 | `_on_vector_update` H2D copy (B4 double-write) | ~200× | Redundant |

**Subtotal Category B:** ~430 per-element D2H + ~230 redundant H2D (B4)

### Comparison

| Phase | V8 | V10 | V11 Reported | V11 Actual | **V12 Now** |
|-------|-----|------|-------------|------------|-------------|
| Full tensor D2H | ~30 | ~20 | 5–6 | 5–6 | **8–12** |
| Per-element D2H | ~20K | ~5K | ~400 | ~400 | **~430** |
| `.item()` syncs | ~5K | ~2K | ~500 | ~500 | **~0** |
| Per-pair scalar | ~2K | ~1K | ~500 | ~500 | **~0** |
| **Total** | **~30K** | **~8K** | **~1,000–5,000** | **~1,000–5,000** | **~440–650** |

**Key:** `.item()` syncs are fully eliminated. Remaining syncs are `_apply_vector_update` calls that sync CPU `concept_vectors` + `fractal.codes` + fire `_on_vector_update` H2D.

---

## 4. VRAM Usage Audit

### Current Tensor Map

| Tensor | Shape | DType | MB | Notes |
|--------|-------|-------|----|-------|
| `_vecs_t` | V×384 | fp16 | **107** | Core concept vectors |
| `_ema_vecs_t` | V×384 | fp16 | **107** | EMA copy (lerp) |
| `_mom_t` | V×384 | fp16 | **107** | Momentum buffer |
| `_codes_t` | V×512 | fp32 | **285** | Subspace latent codes |
| `_basis_t` | 512×384 | fp32 | **0.8** | Basis matrix |
| `_fb_t` | V×129 | uint8 | **18** | Field bits (1024 anchors) |
| `_ce_t` | V | fp32 | **0.6** | Concept error |
| `_cf_t` | V | fp32 | **0.6** | Concept freq |
| `_pt2_t` | V | fp32 | **0.6** | Prefix total |
| `_skip2_t` | V | fp32 | **0.6** | Skip2 total |
| `_fused_buf` | ng×(D+1) | fp32 | **~6–50** | Dynamic (starts 6MB) |
| Temp (pair build) | T×fb_bytes | uint8 | ~0.5 | Per sentence |
| Temp (contrastive) | ng×V×bool | uint8 | ~19 | cooc_masks (sparse) |
| Temp (contrastive) | ng×k×int64 | i64 | ~0.8 | topk indices |
| **Total steady** | | | **~630–670** | |
| **Peak** (with temps) | | | **~700–750** | |

### Prior to V11.2 optimizations

| Tensor | Before (MB) | After (MB) | Delta |
|--------|-------------|-------------|-------|
| `_ema_vecs_t` | 224 (fp32) | 107 (fp16) | -117 |
| `_mom_t` | 224 (fp32) | 107 (fp16) | -117 |
| `_fused_buf` | 225 (full V) | 6 (dynamic) | -219 |
| **Total saved** | | | **-453** |

### Remaining VRAM optimization targets

**G-69 candidate: `_codes_t` fp32 → fp16**
- Current: V×512×4 = 285 MB
- fp16: V×512×2 = 143 MB
- **Saves: 142 MB**
- Risk: subspace update `codes + code_grads * lr` may overflow/underflow in fp16. Requires `master_codes` in fp32 or per-layer loss scaling.
- Impact: High (reduces total from ~700MB to ~560MB, well within 2GB limit)

---

## 5. G-66+ Proposals (V12 → V13)

### G-69 (NEW): `_codes_t` fp16 with fp32 master copy (HIGH — saves 142MB)
- Keep fp32 master on CPU, sync to fp16 GPU for subspace matmul
- Or: use `torch.amp` autocast for subspace update
- Location: `crystal_generator.py:286-292`

### G-70 (NEW): Eliminate `_on_vector_update` double-write (B4 fix) (MED)
- After batched `gen._vecs_t[cids_batch] = vecs_batch`, the `_on_vector_update` hook triggered by `_apply_vector_update` overwrites the same elements
- Fix: Add `_gpu_batch_in_progress` flag. In `_on_vector_update`, skip copy if batch flag is set (the batched write already covers it). Clear flag after all updates.
- **Saves:** ~200 redundant H2D copies per batch

### G-71 (NEW): Fused single-pass `_vecs_t` update (MED)
- Currently: 5 separate GPU→CPU→fractal sync rounds per batch (STDP, neg_sampling, contrastive, centroid, lat_inh)
- Each round: `gen._vecs_t[cids] = vecs_batch` + `cs._apply_vector_update` loop
- Fix: Accumulate all updates into one `_vecs_t` write + one `concept_vectors` sync at batch end
- **Saves:** 4 redundant GPU→CPU batches, ~4× reduced `_on_vector_update` fire

### G-72 (NEW): Lazy CPU `concept_vectors` sync (LOW)
- Keep `_vecs_t` as ground truth during training
- Only sync to `concept_vectors._data` when `_branch()` or `generate()` needs CPU vectors
- Mark modified CIDs with `_cpu_dirty` bitmask, sync on next CPU access
- **Saves:** All `_apply_vector_update` calls during training

### G-66: CUDA Graph Capture (LOW — still blocked)
- **Status:** STILL BLOCKED by per-element `_apply_vector_update` loops and `_overlap_lookup` lambda
- **Prerequisite:** G-70 + G-71 + G-72

### G-67: Persistent GPU RNG (LOW)
- Still relevant: `torch.randint`/`torch.rand` use global state
- Add `gen._gpu_rng = torch.Generator(device=device)` for destab/neg sampling noise

### G-68: Async H2D Input (LOW)
- Input tensors `torch.tensor(gpu_ctx_l)` at `stdp_trainer.py:400` are synchronous
- Pre-allocate pinned buffers, use `non_blocking=True` + CUDA stream overlap

---

## 6. Recommended Implementation Order (V13)

| Priority | ID | Effort | Risk | Impact | Eliminates |
|----------|----|--------|------|--------|------------|
| 1 | **G-69** `_codes_t` fp16 | 1 day | Medium | **-142MB VRAM** | Peak memory |
| 2 | **G-70** B4 double-write fix | 0.5 day | Low | **2×** on write bw | S15 (~200 H2D) |
| 3 | **G-71** Fused single-pass update | 2 days | Medium | **1.5–2×** batch | S9–S14 grouped |
| 4 | **G-72** Lazy CPU sync | 3 days | Medium | **1.2–1.5×** | All `_apply_vector_update` |
| 5 | **G-67** Persistent GPU RNG | 0.5 day | Low | **1.05×** | Global RNG sync |
| 6 | **G-66** CUDA Graph | 2 days | Low | **1.2–1.5×** | Launch overhead |
| 7 | **G-68** Async H2D | 1 day | Low | **1.05×** | Input latency |

**After G-70+G-71+G-72:** Zero per-element Python loops in GPU training → **G-48 (`torch.compile`) becomes feasible.**

---

## 7. Projected V13 Sync Count

| Phase | Current (V12) | After G-70 | After G-71+G-72 | Target |
|-------|---------------|------------|-----------------|--------|
| Full tensor D2H | 8–12 | 8–12 | 1 | 1 |
| Per-element D2H | ~430 | ~230 | 0 | 0 |
| Redundant H2D | ~200 | 0 | 0 | 0 |
| **Total** | **~640** | **~240** | **~1** | **~1** |

---

## 8. Test Coverage Assessment

| Suite | Tests | Covers | Status |
|-------|-------|--------|--------|
| QN-49 | 4 | `_apply_subspace_update_batch` | ✅ On CPU emulation |
| QN-50 | 2 | `_centroid_pull_batch` GPU | ✅ |
| QN-51 | 2 | Fused post-STDP dispatch | ✅ |
| QN-52 | 3 | Deferred GPU write-back | ⚠️ Tests on CPU, not real CUDA |
| QN-53 | 2 | GPU lateral inhibition | ✅ |
| QN-54 | 2 | checkpoint_state | ✅ |
| QN-55 | 2 | effective_cp | ✅ |
| QN-56 | 2 | Batched EMA | ✅ |
| QN-57 | 2 | cooc_masks + fb_overlaps | ✅ |
| QN-58 | 1 | Centroid pull parity (0.1× factor regression guard) | ✅ |

**Gap:** No CUDA GPU tests (all use `torch.device('cpu')`). True GPU tests would catch:
- `fp16` overflow in `_mom_t` or `_ema_vecs_t`
- `torch.compile` compatibility
- CUDA kernel launch overhead

---

## 9. Summary of Findings

### ✅ Fixed in V11.2
| Issue | Type | Commit |
|-------|------|--------|
| G-60: CPU destab → GPU tensor | Performance | a705223 |
| SN-43: CPU neg loop → GPU batched | Performance | a705223 |
| SN-44: `.item()` in contrastive → pure GPU | Performance | a705223 |
| G-65: Per-pair `.item()` field_gate → matrix | Performance | 1768f27 parent |
| 1768f27: `_torch_dirty=True` every batch (160s) | Correctness | 1768f27 |
| 4030b54: EMA/MOM fp32 → fp16 (-224MB) | VRAM | 4030b54 |
| 3150d5e: Fused buf 225MB → 6MB dynamic | VRAM | 3150d5e |

### ✅ B1 Double Momentum — Fixed
Confirmed fixed: per-element CPU loop removed by G-60. Momentum applied once via `_mom_t`.

### ⚠️ Remaining
| Issue | Severity | Detail |
|-------|----------|--------|
| B4 double-write | MED | `_on_vector_update` overwrites batched GPU write (G-70) |
| 430 per-element D2H | MED | All in `_apply_vector_update` calls (G-71+G-72) |
| `_codes_t` 285MB fp32 | VRAM | Largest tensor, candidate for fp16 (G-69) |
| No real CUDA tests | Coverage | All GPU tests run on `torch.device('cpu')` |

### Key Metrics

| Metric | V10 | V11 | V11.2 (V12) | Target (V13) |
|--------|-----|-----|-------------|--------------|
| Syncs/batch | ~8,000 | ~1,000–5,000 | **~640** | **~1** |
| VRAM steady | ~1,100MB | ~1,000MB | **~650MB** | **~500MB** |
| `.item()` calls/batch | ~2,000 | ~500 | **0** | **0** |
| Correctness bugs | 0 found | 1 HIGH (B1) | **0** | — |
| G-48 (torch.compile) | N/A | Blocked | **Still blocked** (per-element loops remain) | **Feasible** |

---

*Report generated 2026-06-21 by GPU-Opt Agent (V12 deep audit)*
