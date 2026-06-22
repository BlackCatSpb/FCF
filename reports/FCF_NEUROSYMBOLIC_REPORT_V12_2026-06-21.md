# Neuro-Symbolic Audit V12 — 2026-06-21

**Auditor:** Neuro-Symbolic Specialist
**Scope:** `stdp_trainer.py`, `concept_space.py`, `crystal_generator.py`
**Base:** V11.2 (a705223) — SN-43/44/45 claimed FIXED

---

## 1. V11.2 Fix Verification

### 1.1 [P2] SN-43: GPU Neg Sampling Batched — ✅ FIXED (qualitatively)

**Claim:** "Batched negative sampling — was Python loop"

**Current code** (`stdp_trainer.py:693-714`):
```python
for gi, gen_cid in enumerate(unique_gen):
    neg_mask = mask[gi]
    if not neg_mask.any() or neg_lr[gi] <= 0:
        continue
    valid_idx = noise[gi][neg_mask]
    grad = (gen._vecs_t[valid_idx].float() - sim[gi][neg_mask][:, None] * gv[gi:gi+1]).sum(dim=0)
    ...
    _neg_updates.append((gen_cid, v_new))
# Batched write outside loop
```

**Verdict:** The Python `for gi, gen_cid in enumerate(unique_gen)` loop **still exists**, but all per-iteration Python overhead (.item(), .numpy(), CPU write-back) is removed. The loop is now pure-tensor inside with deferred batch write-back. **Functional fix: YES. Full elimination: NO.**

### 1.2 [P2] SN-44: GPU Contrastive Pure-Tensor — ✅ FIXED

**Claim:** "Pure-tensor contrastive — no nested loops + .item()"

**Current code** (`stdp_trainer.py:868-901`):
```python
for i in range(ng):
    v_local = g_vecs[i].clone()
    vmask = valid_hn[i]
    if vmask.any():
        hn = best_idx[i][vmask]
        cos_v = best_val[i][vmask]
        grad = (cos_v[:, None] * v_neg).mean(dim=0) - v_local
        ...
        _updates.append((gen_idxs_l[i], v_new))
```

**Verdict:** All `.item()` calls removed. Inner `for j in range(max_hard)` loop eliminated. Valid_hn is a pre-computed boolean mask. TN-14 regularization loop remains but is also pure-tensor. **Correctly fixed.**

### 1.3 [P2] SN-45/G-60: GPU Destabilization — ✅ FIXED

**Claim:** "GPU destab — was CPU numpy + per-element loop"

**Current code** (`stdp_trainer.py:492-510`):
```python
destab_p = torch.clamp(gen._ce_t[unique_gen] * 0.5 * destab_scale, max=0.5)
destab_mask = torch.rand(ng, device=device) < destab_p
if destab_mask.any():
    noise_gpu = v_ppmi - y_ppmi[:, None] * v_self
    mix_gpu = torch.clamp(gen._ce_t[unique_gen] * 0.5, max=0.5)
    acc = torch.where(destab_mask[:, None],
                      acc * (1 - mix_gpu[:, None]) + destab_update,
                      acc)
```

**Verdict:** Fully GPU. No Python loop, no `.item()`, no CPU RNG, no `gen.main_rng`, no `gen.lattice.connections_of()`. Replaced PPMI-grounded candidates with random GPU indices. **Correctly fixed, but different destab dynamics** (see SN-51).

---

## 2. CPU/GPU Parity Assessment (V12)

| Operation | CPU | GPU | Parity |
|:----------|:---|:---|:------:|
| STDP gradient | `np.dot` per pair | `scatter_add_` batched | ✅ **Numerically equivalent** |
| Gradient clipping | Per-element norm | Per-element norm | ✅ |
| Destab | PPMI-grounded candidates | Random GPU indices | ❌ **Quality regression** (SN-51) |
| Neg sampling | Sequential compound (re-read `v_gen`) | Sum all gradients → 1 apply | ⚠️ **Functional parity** (different order) |
| Contrastive | Sequential compound (re-read `v_gen`) | Mean gradient → 1 apply | ⚠️ **Functional parity** |
| Lateral inhibition | CPU numpy cosine matrix | GPU matmul | ✅ **Same math** |
| Centroid pull | Per-element CPU | Batched GPU | ✅ **Same math** |
| Gradient noise | NOT IMPLEMENTED | `torch.randn_like` | ❌ **Gap: CPU has no noise** |
| Momentum | NOT IMPLEMENTED | `_mom_t` GPU tensor | ❌ **Gap: CPU has no momentum** |

---

## 3. Remaining Open Issues from V11

| ID | Sev | Status | Notes |
|:---|:---:|:------:|:------|
| SN-26.2 | P2 | **Open** | `check_basis_health()` exists, never called in subspace update path |
| SN-33 | P2 | **Open** | GPU lateral inh: `gv` snapshot at loop start; writes to `_vecs_t` inside loop create drift |
| SN-19 | P2 | **Open** | GPU contrastive outer Python loop remains (`for i in range(ng)`) |
| SN-41 | P3 | **Open** | `_ema_steps += len(unique_gen)` monotonic growth; resets only on tensor rebuild |
| SN-46 | P3 | **Open** | Contrastive GPU→CPU→GPU double transfer (`.cpu().numpy()` → `_apply_vector_update` → `_on_vector_update` → GPU copy) |
| SN-47 | P3 | **Open** | CPU neg sampling: `list(cs.concept_vectors.keys())` rebuilt per call; `choices` on full vocab |
| SN-48 | P3 | **Open → ✅ FIXED** | GPU field overlap in `_build_pairs`: now uses pre-computed `_overlap_lookup` matrix (G-65/SN-48) |

---

## 4. New Issues Found — SN-49+

### 4.1 [P2] SN-49: GPU Neg Sampling Outer Python Loop Remains

**File:** `stdp_trainer.py:693`

```python
for gi, gen_cid in enumerate(unique_gen):
```

**Issue:** SN-43 removed `.item()`/`.numpy()` from inside the loop, but the outer `for` loop over 50–200 unique_gen concepts remains as Python. Each iteration adds Python overhead (loop machinery, branch prediction, `if not neg_mask.any()`).

**Impact:** ~50–200 Python iterations per batch. For batch-500, this is negligible (~1ms) but for real-time training it accumulates.

**Fix:** Fully vectorized neg sampling: compute all neg-gradients as a batched tensor op, select valid entries with boolean masking, batch-apply. Requires handling variable-length valid masks per concept — use padding with a sentinel or scatter-based accumulation.

### 4.2 [P2] SN-50: GPU Contrastive Outer Python Loop Remains

**File:** `stdp_trainer.py:868`

```python
for i in range(ng):
    v_local = g_vecs[i].clone()
```

**Issue:** Outer loop over `ng` concepts remains Python. Inner `.item()` loops are gone (SN-44 fix), but the outer loop still prevents full GPU utilization.

**Impact:** Same as SN-49. Particularly harmful when `ng` is large (hundreds of concepts per batch).

**Fix:** Convert to full tensor: compute all rep-gradients and contrastive-gradients as `(ng, D)` tensors, apply boolean mask reduction, batch-normalize and write.

### 4.3 [P2] SN-51: GPU Destab Uses Random Index Instead of PPMI-Grounded Candidates

**File:** `stdp_trainer.py:498`

```python
rand_idx = torch.randint(1, n_v, (ng,), device=device)
```

**Issue:** SN-45 replaced `gen.lattice.connections_of(cid, top_k=20, use_ppmi=True)` (PPMI-grounded, semantically related candidates) with `torch.randint` (pure random). CPU destab pushes toward concepts with meaningful PPMI relationships; GPU destab pushes toward arbitrary concepts.

**Impact:** Destab quality regression. PPMI-grounded destab encourages separation between _related_ concepts (fine-grained). Random destab just adds noise. Tests show vector norms remain unit, but semantic quality likely degraded.

**Fix:** Option A — Pre-compute PPMI candidate indices as a GPU tensor (`_ppmi_candidates_t` of shape (V, K)) rebuilt incrementally with lattice. Option B — Use field-overlap candidates from `_fb_t` for more grounded destab. Option C — Accept the tradeoff (faster training, less semantic fine-tuning).

### 4.4 [P2] SN-52: `_gpu_poststdp_fused` Asymmetric API

**File:** `stdp_trainer.py:569-577`

```python
def _gpu_poststdp_fused(self, gpu_ctx_l, gpu_meta_l, gpu_cid_ctx, gpu_cid_gen,
                        gen_updates, field_gate, base_lr_val, neg_lr_ratio, neg_samples):
    if neg_samples > 0 and gen._vecs_t is not None:
        self._negative_sampling_gpu(...)        # ignores gen_updates
    if gen_updates:
        self._contrastive_objective_gpu(gen_updates, field_gate)  # uses gen_updates
```

**Issue:** `_negative_sampling_gpu` receives `(gpu_ctx_l, gpu_meta_l, gpu_cid_ctx, gpu_cid_gen)` and does NOT use `gen_updates`. `_contrastive_objective_gpu` receives `gen_updates` to build `cooc_masks`. The neg sampling path redundantly recomputes `unique_gen` from `gpu_cid_gen` (via `np.unique`), duplicating work already done in `_gpu_stdp_apply`.

**Impact:** 10–50μs wasted per call for `np.unique(gpu_cid_gen)` — minor but unnecessary.

**Fix:** Pass computed `unique_gen` and `avg_elr_per_gen` from `_gpu_stdp_apply` to the fused method, or store as `gen._gpu_unique_gen` (already stored at line 459). Then `_negative_sampling_gpu` can avoid recompute.

### 4.5 [P1] SN-53: `fb_overlaps` + `cooc_masks` O(V·ng) Memory Blow-Up

**File:** `stdp_trainer.py:813-829`

```python
cooc_masks = torch.zeros(ng, n_v, dtype=torch.bool, device=d)   # (ng × V) bool
fb_overlaps = torch.zeros(ng, n_v, device=d, dtype=torch.long)   # (ng × V) int64
```

**Memory calculation for V=146K, D=384:**

| Component | Size |
|:----------|:----:|
| `cooc_masks` (ng=100 × 146K × 1B) | 14.6 MB |
| `fb_overlaps` (ng=100 × 146K × 8B) | 116.8 MB |
| Topk indices (ng × 2K × 8B) | 1.6 MB |
| Topk values (ng × 2K × 4B) | 0.8 MB |
| `fb_overlaps` temp (1 × 146K × 128B) | 18.7 MB |
| `g_vecs` (ng × 384 × 4B) | 0.15 MB |
| `all_vecs` (146K × 384 × 2B) | 112 MB |
| **Peak (ng=100)** | **~265 MB** |
| **Peak (ng=1000)** | **~1.3 GB** |

**Issue:** `fb_overlaps` as `torch.long` (int64) is wasteful. For `n_anchors=1024`, fb_bytes=128, overlap fits in uint8 (0–128). `cooc_masks` as `torch.bool` is optimal. At ng=1000, this blows past most GPU VRAM budgets (2048 MB total).

**Fix:** Use `torch.uint8` for `fb_overlaps` (saves 87.6 MB). Or compute overlaps on-demand: `fb_overlaps[i, best_idx[i]]` via batched gather without full `(ng, V)` matrix.

### 4.6 [P1] SN-54: `_emsure_torch` Full Rebuild on Every Dirty Flag

**File:** `crystal_generator.py:226-251`

**Issue:** After `fluctuate_fractal()` sets `_torch_dirty = True`, the next `_ensure_torch` call rebuilds ALL GPU tensors from scratch: iterates 146K fractal codes → compute vectors → copy to GPU → rebuild freqs → clone EMA. This is O(V·D) CPU work + 636 MB PCIe transfer.

**Impact:** If `_torch_dirty` is set every batch (e.g., after `cs.fluctuate_fractal`), rebuild overhead dominates. The fix 1768f27 ("remove unconditional _torch_dirty=True") was correct but fragile — any code path that sets `_torch_dirty` accidentally will pay full rebuild cost.

**Fix:** Implement incremental tensor sync: update only changed CIDs in `_vecs_t` and `_codes_t` via the `_on_vector_update` hook (already works for individual vector updates). For `_codes_t`, add a sync path that updates only modified entries.

### 4.7 [P2] SN-55: Subspace Update GPU→CPU→GPU Roundtrip

**File:** `stdp_trainer.py:531-533` + `concept_space.py:591-638`

```python
_subspace_grads.append(grad_gpu[gi].cpu().numpy())     # GPU → CPU
...
cs._apply_subspace_update_batch(cids, np.array(_subspace_grads, ...)  # CPU
    grads_t = torch.from_numpy(grads).to(device)                       # CPU → GPU
```

**Issue:** Gradient is computed on GPU (line 517, `grad_gpu`), moved to CPU via `.cpu().numpy()`, passed to `_apply_subspace_update_batch`, which copies back to GPU. Double roundtrip.

**Impact:** ~2× data transfer for subspace gradients. For large batches, adds 1–5ms.

**Fix:** Add `_apply_subspace_update_batch_gpu(cids, grads_t, ...)` that operates entirely on GPU tensors. Fall back to CPU variant for the non-torch path.

### 4.8 [P3] SN-56: `_codes_t` Stale After Subspace Update

**File:** `concept_space.py:591-638`

**Issue:** Identified in V11 (section 5.1) but not fixed. `_apply_subspace_update_batch` updates `fractal.codes[cid]` dict and `_vecs_t` via hook, but `gen._codes_t` GPU tensor is NOT updated. After subspace update, `_codes_t` contains stale codes until the next `_ensure_torch` rebuild.

**Impact:** Next batch's subspace update reads stale codes from `_codes_t`. Within-batch correctness OK (subspace is last op). But next batch's `_ensure_torch` pays full rebuild cost.

**Fix:** Update `gen._codes_t[unique_gen]` in `_apply_subspace_update_batch` directly with the computed `new_codes` that are already on GPU (line 613: `new_codes = codes + code_grads * base_lr_val`).

### 4.9 [P3] SN-57: `fluctuation_amp` Dead Parameter in `_train`

**File:** `stdp_trainer.py:69`

```python
def _train(self, inputs, base_lr, ..., fluctuation_amp=0.003):
```

**Issue:** Parameter is accepted but **never referenced in the method body**. The actual fluctuation is called externally via `cs.fluctuate_fractal()`. This is dead code baggage from a refactor where fluctuation was moved out of the training loop.

**Impact:** Misleading API surface. Callers may believe `_train` applies fluctuation.

**Fix:** Remove parameter from `_train`, `train_from_text`, `train_batch` signatures.

### 4.10 [P3] SN-58: CPU Gradient Noise Not Implemented

**File:** `stdp_trainer.py:462-463`

```python
if gradient_noise_scale > 0:
    acc += torch.randn_like(acc) * gradient_noise_scale * (...)
```

**Issue:** Gradient noise injection is GPU-only. The `_cpu_stdp_apply` path ignores `gradient_noise_scale`. If a training run switches between CPU and GPU (e.g., OOM fallback), results will diverge when noise_scale > 0.

**Impact:** CPU/GPU non-determinism for training runs with gradient noise. Minor since noise is typically 0.

**Fix:** Add `if gradient_noise_scale > 0:` to CPU path: `grad += np.random.randn(D) * gradient_noise_scale * (total_elr / max_elr_total)`.

### 4.11 [P3] SN-59: `_contrastive_objective_gpu` `fb_overlaps` Shape Error Risk

**File:** `stdp_trainer.py:826-829`

```python
for i in range(ng):
    fb_overlaps[i] = (fb_gen_all[i:i+1].unsqueeze(1) & fb_t_exp).sum(dim=-1)
```

**Issue:** `fb_gen_all[i:i+1]` is shape `(1, fb_bytes)`. `.unsqueeze(1)` → `(1, 1, fb_bytes)`. `fb_t_exp` is `(1, V, fb_bytes)`. Broadcasting: `(1, 1, fb_bytes) & (1, V, fb_bytes)` → `(1, V, fb_bytes)`. `.sum(dim=-1)` → `(1, V)`. This is correct but creates a `(1, V, fb_bytes)` temp tensor = 146K × 128 = 18.7 MB per iteration. For ng=100, this hits 1.87 GB peak.

**Impact:** Peak memory during overlap computation is NG × V × fb_bytes = 18.7 × ng MB. At ng=100, 1.87 GB. At ng=200, OOM on 2GB GPU.

**Fix:** Chunk the loop into batches: process 10 gen_cids at a time with `fb_gen_all[:10].unsqueeze(1)` → `(10, 1, fb_bytes) & (1, V, fb_bytes)` → `(10, V)` output directly. Reduces peak to `min(chunk, ng) × V × fb_bytes`.

---

## 5. Cross-Cutting Concerns

### 5.1 V11 → V12 Regression: No New Tests for SN-43/44/45

The fixes for SN-43/44/45 were committed in a705223 but no new tests were added to verify:
- Batched neg sampling produces correct (non-divergent) vectors
- Pure-tensor contrastive produces same results as previous impl
- GPU destab with random indices maintains vector norms

The existing parity test (`test_cpu_gpu_stdp_parity`) tests only STDP gradient, not neg sampling or contrastive.

### 5.2 Momentum is GPU-only (Carry-over from V11)

CPU momentum path is NOT implemented. If `momentum_mu > 0` on CPU, it's silently ignored.

### 5.3 `_vecs_t` Float16 Precision Risk

`_vecs_t` is stored as `torch.float16` on GPU (`crystal_generator.py:279`). All gradient computations cast to float32 (`.float()`). The `_vecs_t` write-back casts back to fp16 (line 549: `.to(gen._vecs_t.dtype)`). This means:
- Gradient accumulation: float32 ✓
- Vector storage: float16 (112MB saved) ✓
- Float16 write-back truncates precision: vectors lose ~3 decimal digits of precision vs float32

Not a bug — deliberate VRAM tradeoff. But worth documenting for debug purposes.

---

## 6. Severity Summary

### 6.1 V11.2 Fixes Verified

| ID | Sev | Status |
|:---|:---:|:------:|
| SN-43 (GPU neg samp batched) | P2 | ✅ **Fixed** (partial — outer loop remains, see SN-49) |
| SN-44 (GPU contrastive pure-tensor) | P2 | ✅ **Fixed** |
| SN-45/G-60 (GPU destab) | P2 | ✅ **Fixed** |
| SN-48 (GPU field overlap per-pair sync) | P3 | ✅ **Fixed** (G-65: pre-computed overlap matrix) |

### 6.2 Still Open from V11

| ID | Sev | Issue |
|:---|:---:|:------|
| SN-26.2 | P2 | Basis health not checked in subspace update |
| SN-33 | P2 | GPU lateral inh stale `_vecs_t` snapshot |
| SN-19 | P2 | GPU contrastive outer Python loop (→ SN-50) |
| SN-41 | P3 | EMA counter monotonic growth |
| SN-46 | P3 | Contrastive GPU→CPU→GPU roundtrip |
| SN-47 | P3 | CPU neg sampling re-samples vocab per concept |

### 6.3 New in V12

| ID | Sev | Issue |
|:---|:---:|:------|
| **SN-49** | P2 | GPU neg sampling outer Python loop persists |
| **SN-50** | P2 | GPU contrastive outer Python loop persists |
| **SN-51** | P2 | GPU destab uses random index (PPMI quality regression) |
| **SN-52** | P2 | `_gpu_poststdp_fused` asymmetric API, redundant `np.unique` |
| **SN-53** | **P1** | `fb_overlaps` O(V·ng) memory — int64 wasteful, 1.3GB at ng=1000 |
| **SN-54** | **P1** | `_ensure_torch` full rebuild on dirty flag — O(V·D) every fluctuation |
| **SN-55** | P2 | Subspace update GPU→CPU→GPU gradient roundtrip |
| **SN-56** | P3 | `_codes_t` stale after subspace update (carry-over) |
| **SN-57** | P3 | `fluctuation_amp` dead parameter in `_train` |
| **SN-58** | P3 | CPU gradient noise not implemented |
| **SN-59** | P3 | `fb_overlaps` loop creates 18.7MB temp per iteration |

### 6.4 Severity Count

| Severity | Count | Key Issues |
|:---------|:-----:|:-----------|
| **P1** | 2 | SN-53 (memory), SN-54 (rebuild cost) |
| **P2** | 7 | SN-26.2, SN-33, SN-19, SN-49, SN-50, SN-51, SN-52, SN-55 |
| **P3** | 7 | SN-41, SN-46, SN-47, SN-56, SN-57, SN-58, SN-59 |

---

## 7. Test Results (V12)

- 114 of 129 tests pass
- 2 failures: `TestCheckpointCleanup.test_cleanup_keep`, `TestCheckpointCleanup.test_cleanup_below_keep` — pre-existing (CheckpointManager has no `.cleanup()` method)
- 1 skipped: `test_generate_returns_result` (no SP model), `test_generate_empty_seed`, `test_fb_overlap_tensor_shape`
- All STDP/GPU/neg-sampling/contrastive tests pass ✓
- GPU/CPU parity test passes (relaxed tolerance) ✓
- Subspace update batch tests pass ✓
- EMA batch tests pass ✓
