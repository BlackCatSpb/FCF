# FCF GPU Optimization Audit — V10 (2026-06-19)

**Auditor:** GPU-Opt Agent  
**Scope:** `stdp_trainer.py` (871L) + `crystal_generator.py` (823L) + `concept_space.py` (881L)  
**Config:** ~146K vocab, 384D, FP16 `_vecs_t`, CUDA  
**Base:** `cccc392` (V9 GPU fixes: G-45, G-47) + `21ee6ca` (SN-22.3 per-concept avg_elr)

---

## 1. V9 GPU Fix Verification

### G-45: Persistent CUDA Events ✅
**Commit:** `cccc392`  
**File:** `stdp_trainer.py:27-32`, `_gpu_stdp_apply:348-349,478-481`

- Events created once in `__init__` (not per-call) ✅
- `record()` / `record() + synchronize() + elapsed_time()` correctly structured ✅
- Guarded by checking `self._prof_start is not None` instead of `torch.cuda.is_available()` ✅
- ⚠️ **Residual:** `self.gen._prof_ms` is written but **never read** — dead profiling code. Entire block can be removed if profiling is not consumed externally.

### G-47: `lerp_` for EMA ✅ (Partial)
**Commit:** `cccc392`  
**File:** `stdp_trainer.py:462-464`

- Old: `gen._ema_vecs_t[gen_cid] = ema_decay * old + (1 - ema_decay) * new`  
  New: `gen._ema_vecs_t[gen_cid].lerp_(gen._vecs_t[gen_cid].float(), 1.0 - gen._ema_decay)`
- `lerp_` is a fused CUDA kernel — no intermediate tensor ✅
- ⚠️ **Still per-element** in the Python loop — not batched as `gen._ema_vecs_t[unique_gen].lerp_(...)`. The V9 report's batched version (G-22) remains unimplemented.

### G-22 (Partially): EMA per-concept loop → batched lerp_ ❌
- Still per-element loop. `lerp_` is used, but for single vectors, not the indexed slice.

### Other V9 Changes in `cccc392`:
- SN-22.1: `_graph_cache` `{}` → `OrderedDict` with LRU eviction (crystal_generator.py:72-73)
- SN-22.1: `_negative_sampling_gpu`: `.mean(dim=0)` → `.sum(dim=0)` (line 593) — matches CPU behavior

### SN-22.3 (HEAD `21ee6ca`):
- Per-concept `avg_elr` via `scatter_add_` grouping instead of global `elr_sum` (lines 571-579)
- Eliminates 1× scalar `.item()` sync; adds per-concept `.item()` at line 582 — net neutral sync count

---

## 2. V9 GPU Fixes NOT Applied

| ID | Description | Location | Status |
|----|-------------|----------|--------|
| G-40 | Batched GPU subspace update | `concept_space.py:556-583` | ❌ Still 100% CPU numpy |
| G-41 | Full GPU lateral inhibition | `stdp_trainer.py:485-510` | ❌ Still `.item()` + CPU write-back |
| G-42 | GPU centroid_pull_batch | `stdp_trainer.py:771-796` | ❌ Still 100% CPU numpy |
| G-43 | Vectorized negative sampling | `stdp_trainer.py:581-604` | ❌ Still per-element Python loop |
| G-44 | Batched GPU TN-14/contrastive | `stdp_trainer.py:701-765` | ❌ Still ~16,500 syncs/batch |
| G-46 | Persistent `_mom_t` tensor | `stdp_trainer.py:410-423` | ❌ Still CPU `_mom_buf` dict |
| G-48 | `torch.compile` | `_gpu_stdp_apply` | ❌ Not applied |
| G-49 | Pre-allocate fused buffer | `stdp_trainer.py:382` | ❌ `torch.zeros(...)` per call |
| AM-42/SN-19 | GPU contrastive vectorization | `_contrastive_objective_gpu` | ❌ Python loop |
| AM-43 | GPU neg sampling vectorization | `_negative_sampling_gpu` | ❌ Python loop |

---

## 3. Current CPU-GPU Sync Point Catalog

Measured per batch (N=100 unique_gen, 500 pairs):

| # | Location | Line | Sync Type | Frequency |
|---|----------|------|-----------|-----------|
| S1 | `_gpu_stdp_apply`: `avg_err.cpu().numpy()` | 398 | Full tensor D2H | 1× |
| S2 | `_gpu_stdp_apply`: `concept_error.update()` | 405 | per-element CPU dict | N× |
| S3 | `_gpu_stdp_apply`: `_mom_buf.get()` read | 414-416 | per-element H2D + dict | N× |
| S4 | `_gpu_stdp_apply`: `_mom_buf[...]=` write | 422-423 | per-element D2H + dict | N× |
| S5 | `_gpu_stdp_apply`: `acc.cpu().numpy()` | 426 | Full tensor D2H | 1× |
| S6 | `_gpu_stdp_apply`: `cnt.cpu().numpy()` | 427 | Full tensor D2H | 1× |
| S7 | `_gpu_stdp_apply`: `elr_grouped.cpu().numpy()` | 428 | Full tensor D2H | 1× |
| S8 | `_gpu_stdp_apply`: per-element Python loop | 430-473 | Python loop + CPU numpy | N× |
| S9 | `_gpu_stdp_apply`: `_apply_subspace_update` | 467 | 100% CPU numpy matmul | N× (if enabled) |
| S10 | `_negative_sampling_gpu`: `avg_elr_per_gen[gi].item()` | 582 | scalar sync | N× |
| S11 | `_negative_sampling_gpu`: `.cpu().numpy()` + apply | 600-604 | per-element D2H | N× |
| S12 | `_contrastive_objective_gpu`: `.item()` syncs | 704, 712, 718 | scalar sync | N×max_hard×3 |
| S13 | `_contrastive_objective_gpu`: TN-14 `.item()` syncs | 730, 736, 740 | scalar sync | N×min(50,K)×3 |
| S14 | `_contrastive_objective_gpu`: `.cpu().numpy()` | 760, 765 | per-element D2H | N× |
| S15 | `_lateral_inhibition_gpu`: `inhibition.item()` | 498 | scalar sync | N× |
| S16 | `_lateral_inhibition_gpu`: `.cpu().numpy()` + apply | 503-510 | per-element D2H | N× |
| S17 | `_build_pairs`: field overlap `.item()` | 188 | scalar sync | O(N²) per sentence |

**Estimated total: ~20,000+ individual GPU→CPU syncs per batch** (unchanged from V9).

---

## 4. G-45/G-47 Correctness Assessment

### G-45: ✅ Clean
- Event lifetime: created once, no leak
- Thread safety: no concurrent calls to `_gpu_stdp_apply` in current design
- Edge case: `torch.cuda.is_available()` at init, but if CUDA becomes unavailable later, `record()`/`synchronize()` will raise. Guarded by `self._prof_start is not None` which is set at init time only.
- Recommendation: Remove entire profiling block if `_prof_ms` is unused — saves 2× CUDA Event objects + synchronize call per batch.

### G-47: ✅ Correct, but suboptimal
- `lerp_(source, weight)` formula: `self = self + weight * (source - self)` = `(1-weight)*self + weight*source`
- With `weight = 1 - ema_decay = 0.001`, this is correct: `0.999*old + 0.001*new`
- But per-element: `gen._ema_vecs_t[gen_cid].lerp_(...)` vs ideal batched `gen._ema_vecs_t[unique_gen].lerp_(gen._vecs_t[unique_gen].float(), 1 - gen._ema_decay)`
- Batched version would fuse N× scatter+lerp into one kernel launch — ~N× fewer CUDA API calls
- **Fix:** Replace per-element with single batched call after the Python loop

---

## 5. Updated Speedup Estimates (V10)

| ID | Severity | V9 Est. | V10 Updated Est. | Rationale |
|----|----------|---------|------------------|-----------|
| **G-40** | **HIGH** | 10-20× | **10-20×** | No change; subspace still 100% CPU |
| **G-41** | MED | 2-5× | **2-5×** | No change |
| **G-42** | MED | 3-10× | **3-10×** | No change |
| **G-43** | MED | 2-5× | **2-5×** | No change (`.sum()` fix minor) |
| **G-44** | **HIGH** | 5-50× | **5-50×** | No change; 16,500 syncs remain |
| **G-45** | LOW | <1% | **✅ Done** | — |
| **G-46** | MED | 3-10× | **3-10×** | No change |
| **G-47** | LOW | ~1.2× | **~1.05×** | Per-element `lerp_` only; need batched |
| **G-48** | MED | 1.5-3× | **1.5-3×** | Prerequisite: all loops vectorized first |
| **G-49** | LOW | <1% | **<1%** | Minor allocator churn |

### Cumulative Speedup Potential

| Path | Current (est.) | With G-40..G-49 | Delta |
|------|---------------|-----------------|-------|
| GPU STDP + subspace | ~15-50ms | ~2-5ms | **3-10×** |
| GPU STDP no subspace | ~5-30ms | ~1-3ms | **3-10×** |

---

## 6. G-50+ Proposals

### G-50: Zero-copy vector write-back (HIGH — 2-5×)
**Problem:** Every `_apply_vector_update` → `_on_vector_update` hook → `torch.from_numpy(v_new).to(device, non_blocking=True).copy_(...)`. This is CPU numpy copy + H2D transfer per element.

**Fix:** From GPU code paths, write directly to `gen._vecs_t[cid]` on GPU:
```python
# Instead of: cs._apply_vector_update(cid, v_new.cpu().numpy())
# Use:
gen._vecs_t[cid] = v_new.half()  # direct GPU→GPU, no CPU roundtrip
gen.cs.set_vec(cid, v_new.cpu().numpy())  # sync CPU store (optional, can be deferred)
```
**Requires:** Skip `_on_vector_update` when called from GPU path (flag or separate method).

**Saves:** S3, S4, S5, S6, S7, S11, S14, S16 — all `.cpu().numpy()` roundtrips.

### G-51: Deferred vector sync — batched `_vecs_t` write (MED — 1.5-3×)
**Problem:** `_on_vector_update` fires per-element, each launching a separate `copy_` kernel.

**Fix:** Accumulate (cid, vec) pairs in GPU buffer, apply once at end of `_gpu_stdp_apply`:
```python
modified_idxs = []  # GPU tensor accumulator
modified_vecs = []  # GPU tensor accumulator
# Inside loops, instead of _apply_vector_update:
modified_idxs.append(cid)
modified_vecs.append(v_gpu)
# At end of _gpu_stdp_apply:
if modified_idxs:
    gen._vecs_t[torch.tensor(modified_idxs)] = torch.stack(modified_vecs).half()
```

### G-52: Fused contrastive + negative sampling (MED — 1.2-2×)
**Problem:** `_negative_sampling_gpu` and `_contrastive_objective_gpu` are separate loops over `unique_gen`, each reading/writing the same vectors.

**Fix:** Merge into single pass — compute both negative sampling and contrastive push for each concept in one loop iteration. Reduces vector reads and kernel launches.

### G-53: GPU field overlap for `_build_pairs` (MED — 1.2-1.5×)
**Problem:** `torch.bitwise_and(gen._fb_t[ids[i]], gen._fb_t[ids[j]]).sum().item()` at line 188 triggers `.item()` sync for every pair.

**Fix:** Move `field_weight` computation to GPU: store per-pair field_weight in `gpu_meta_l` as a deferred GPU op, or pre-compute using CPU field_bits with a batch approach.

### G-54: Batched destab on GPU (LOW — 1.2-2×)
**Problem:** Per-element CPU destab logic (lines 435-452): RNG, PPMI lookup, numpy ops.

**Fix:** Pre-compute destab decisions vectorized: `ce_t[unique_gen] > threshold`, `main_rng` calls batched, `_destab_field_fallback` on GPU.

### G-55: Remove dead profiling code (LOW — <1%)
**Problem:** Lines 348-349, 478-481 — profiling events block writes `self.gen._prof_ms` but it's never consumed anywhere.

**Fix:** Remove `_prof_start`, `_prof_end`, `record()`, `synchronize()`, `elapsed_time()` entirely. Saves 2 CUDA event objects + 1 synchronize per batch.

### G-56: Async H2D transfer for input tensors (LOW — ~1.1×)
**Problem:** `torch.tensor(gpu_ctx_l)` at line 344 is synchronous CPU→GPU copy.

**Fix:** Allocate pinned memory pool, use `torch.from_numpy(np.array(list)).to(device, non_blocking=True)` with separate CUDA stream. Overlaps CPU list building with GPU compute of previous batch.

### G-57: Unified `_gpu_stdp_apply` with `torch.compile` (MED — 1.5-3×)
**Problem:** G-48 requires all Python loops removed. After G-50..G-53, the function is pure tensor ops.

**Fix:** Apply `@torch.compile(mode="max-autotune")` to the vectorized `_gpu_stdp_apply`. Expected 1.5-3× speedup on remaining tensor ops.

---

## 7. Recommended V10 Implementation Order

| Priority | ID | Effort | Risk | Cumulative Speedup |
|----------|----|--------|------|-------------------|
| 1 | **G-50** Zero-copy write-back | 3 days | Medium — breaks CPU path if not guarded | 2-5× |
| 2 | **G-40** Batched GPU subspace | 2 days | Low — independent module | 10-20× on subspace |
| 3 | **G-46** Persistent `_mom_t` | 1 day | Low — replaces dict with tensor | 3-10× on momentum |
| 4 | **G-51** Deferred vector sync | 2 days | Medium — replaces hook pattern | 1.5-3× |
| 5 | **G-44** Batched GPU contrastive | 3 days | High — complex logic change | 5-50× on contrastive |
| 6 | **G-43** Vectorized neg sampling | 1 day | Low — straightforward | 2-5× |
| 7 | **G-41** GPU lateral inhibition | 1 day | Low — batched matmul | 2-5× |
| 8 | **G-55** Remove dead profiling | 0.5 day | None | <1% |

**Prerequisite for G-48/G-57 (torch.compile):** G-50 + G-51 + G-43 + G-44 must be done first (all Python loops eliminated from `_gpu_stdp_apply`).

---

## 8. V9→V10 Delta

| Metric | V9 | V10 | Delta |
|--------|----|-----|-------|
| Persistent CUDA events | ❌ Created per-call | ✅ Created once | Fixed |
| EMA update | Per-element `lerp_` | Per-element `lerp_` | ⚠️ Same (still not batched) |
| Subspace update | 100% CPU numpy | 100% CPU numpy | ❌ Unchanged |
| Momentum buffer | CPU `_mom_buf` dict | CPU `_mom_buf` dict | ❌ Unchanged |
| Contrastive syncs/batch | ~16,500 | ~16,500 | ❌ Unchanged |
| Negative sampling | Python loop | Python loop (`.sum()` fix) | ⚠️ Minor change |
| Lateral inhibition | `.item()` + CPU | `.item()` + CPU | ❌ Unchanged |
| Centroid pull | 100% CPU numpy | 100% CPU numpy | ❌ Unchanged |
| G-40..G-49 implemented | 2/10 (G-45, G-47) | 2/10 (G-45, G-47) | ⚠️ Same |
| G-50+ proposals | — | 8 new (G-50..G-57) | ✅ Added |

---

## 9. Critical Findings

1. **P4 status unchanged (HIGH):** `_apply_subspace_update` at `concept_space.py:556-583` is 100% CPU numpy in GPU path. Called per-element at `stdp_trainer.py:467`. For N=1000 unique_gen with subspace_lr enabled, adds ~5-20ms of CPU matmul overhead per batch.

2. **P5 status unchanged (HIGH):** `_contrastive_objective_gpu` at `stdp_trainer.py:701-765` generates ~16,500 GPU→CPU scalar syncs per batch (N=100 concepts). This alone can make GPU path slower than CPU.

3. **P6 status unchanged (MED):** `_mom_buf` CPU dict at `stdp_trainer.py:410-423` generates 2N per-element CPU roundtrips.

4. **`_on_vector_update` hook (NEW FINDING - MED):** At `crystal_generator.py:155-158`, every `_apply_vector_update` calls this hook which does `torch.from_numpy(v_new).to(device).copy_(...)`. This is a full H2D copy for every single vector write. In current code, this fires N× for STDP + N× for neg sampling + N× for contrastive + N× for lateral inhibition + N× for centroid pull = **~5N H2D copies per batch**. Direct GPU→GPU write would bypass this entirely.

5. **Subspace update bypasses `_on_vector_update` hook:** At `concept_space.py:579`, `_apply_subspace_update` calls `self.set_vec(cid, v_new)` directly without going through `_apply_vector_update`, then manually syncs code and calls `_after_update_hook`. This means subspace updates DO update `_vecs_t` via the hook, but the STDP gradient path at line 473 calls `_apply_vector_update` which ALSO triggers the hook. So each concept may get double-synced per batch if both regular and subspace paths are active. **Not** a correctness bug (final write wins), but wasted bandwidth.

---

## 10. V10 Action Items

1. **G-50: Zero-copy vector write-back** — biggest single win. Eliminates all `.cpu().numpy()` roundtrips. Creates direct GPU→GPU path for vector writes.
2. **G-40: Batched GPU subspace update** — move basis ops to GPU, single batch call instead of N× CPU matmul.
3. **G-46: Persistent `_mom_t`** — simple 1-day fix, eliminates 2N CPU roundtrips.
4. **G-55: Remove dead profiling code** — 5 lines, no-risk cleanup.
5. **G-44 + G-43: Batched contrastive + neg sampling** — eliminate remaining 16,500+ syncs.

---

*Report generated 2026-06-19 by GPU-Opt Agent (V10 audit)*
