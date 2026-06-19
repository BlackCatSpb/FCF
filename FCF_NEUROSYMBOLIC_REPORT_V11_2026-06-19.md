# Neuro-Symbolic Audit V11 — 2026-06-19

**Auditor:** Neuro-Symbolic Specialist
**Scope:** `stdp_trainer.py`, `concept_space.py`, `crystal_generator.py`
**Base:** V10 report + V10 fixes (SN-35..SN-42, G-40..G-49, AM-30)

---

## 1. V10 Fix Verification

| V10 Fix | Code Location | Status |
|:--------|:--------------|:------:|
| **SN-35**: CPU neg sampling compound | `stdp_trainer.py:572` `v_gen = cs.concept_vectors.get(gen_cid)` re-read after each update | ✅ **Fixed** |
| **SN-36**: CPU contrastive compound | `stdp_trainer.py:684` `v_gen = cs.concept_vectors.get(gen_cid)` re-read after each update | ✅ **Fixed** |
| **G-40**: Batched GPU subspace | `concept_space.py:591-638` `_apply_subspace_update_batch` uses `gen._codes_t` + `gen._basis_t` GPU tensors | ✅ **Fixed** |
| **G-43**: GPU neg sampling vectorized | `stdp_trainer.py:587-602` scatter_add for elr/grouping; **still per-element loop at line 604** | ⚠️ **Partial** |
| **G-44**: GPU contrastive pre-computed | `stdp_trainer.py:708-720` `cooc_masks` + `fb_overlaps` as GPU tensors BEFORE loop | ✅ **Fixed** |
| **G-46**: Persistent `_mom_t` tensor | `crystal_generator.py:247` initialized, `stdp_trainer.py:411-418` GPU tensor, removed CPU dict | ✅ **Fixed** |
| **SN-39**: `connection_strength` removed | `_contrastive_objective_gpu` uses `cooc_masks[i, neg_cid]` instead | ✅ **Fixed** |
| **AM-30**: Batched EMA | `stdp_trainer.py:488` `gen._ema_vecs_t[unique_gen].lerp_(...)` — single batched call | ✅ **Fixed** |
| **G-49**: Pre-allocated fused buffer | `stdp_trainer.py:382-384` `gen._fused_buf[:ng]` reused across calls | ✅ **Fixed** |

### 1.1 SN-35 Deep Check — CPU Neg Sampling Compound

```python
# Line 554-572
for neg_cid in neg_candidates:
    ...
    cs._apply_vector_update(gen_cid, v_new)
    v_gen = cs.concept_vectors.get(gen_cid)  # ← re-read for compounding
```

**Result:** Each iteration re-reads `v_gen` from the store. Nested compound updates work correctly. Matches GPU's `.sum()` semantics (CPU compounds sequentially, GPU sums all gradients → applies once). **Numerical difference remains** (sequential vs simultaneous) but semantic parity is correct.

### 1.2 SN-36 Deep Check — CPU Contrastive Compound

```python
# Line 660-684
for neg_cid, cos_val in hard_negatives[:5]:
    ...
    cs._apply_vector_update(gen_cid, v_new)
    v_gen = cs.concept_vectors.get(gen_cid)  # ← re-read for compounding
```

**Result:** Same pattern as SN-35. **Fixed.** CPU applies 5 sequential pushes; GPU applies 1 mean gradient. Semantic parity correct.

### 1.3 G-40 Deep Check — Batched GPU Subspace

`_apply_subspace_update_batch` (concept_space.py:591):
- Reads `gen._codes_t[cids_t]` — GPU tensor of latent codes ✅
- Computes `code_grads = grads_t @ basis_t.T` — GPU matmul ✅
- Applies per-subspace LR mask (GPU) ✅
- Writes back via `_after_update_hook` → `_on_vector_update` → `_vecs_t[cid].copy_()` ✅
- `_codes_t` NOT updated with new codes — stale until next `_ensure_torch` rebuild

**Correctness:** OK for single-batch usage. `_codes_t` is rebuilt at start of each batch. Within a batch, subspace is the LAST update, so no double-read of stale codes occurs.

### 1.4 G-46 Deep Check — Persistent `_mom_t`

```python
# stdp_trainer.py:411-418
if gen._mom_t is None:
    gen._mom_t = torch.zeros(V, D, device=device, dtype=torch.float32)
mom_t = gen._mom_t[unique_gen]
mom_t = momentum_mu * mom_t + (1 - momentum_mu) * avg_grad
gen._mom_t[unique_gen] = mom_t
mom_cpu = mom_t.cpu().numpy()  # ← still CPU sync for downstream use
```

**Result:** Buffer itself is GPU-persistent. The `mom_cpu` is still needed for the per-concept Python loop (line 460). Full GPU elimination requires moving the entire write-back loop to GPU.

---

## 2. CPU/GPU Parity Assessment

### 2.1 Parity After V10 Fixes

| Operation | CPU | GPU | Parity |
|:----------|:---|:---|:------:|
| STDP gradient | `np.sum -> y * v` | `scatter_add_` fused | ✅ **Numerically equivalent** |
| Gradient clipping | Per-element `gn > max_norm` | Per-element `gn > max_norm` | ✅ |
| Destab | Per-element RNG + PPMI | Per-element RNG + PPMI (same code) | ✅ |
| Neg sampling | Sequential compound (re-read `v_gen`) | Sum all gradients → 1 apply | ⚠️ **Functional parity** (different order) |
| Contrastive | Sequential compound (re-read `v_gen`) | Mean gradient → 1 apply | ⚠️ **Functional parity** |
| Lateral inhibition | Per-element CPU numpy | Batched GPU matmul | ⚠️ **Different impl, same math** |
| Centroid pull | Per-element CPU | Batched GPU | ⚠️ **Different impl, same math** |

### 2.2 Field Gate Parity — SN-28 Still Open (P1)

GPU contrastive (`_gpu_poststdp_fused` line 498-506) passes `field_gate` to `_negative_sampling_gpu` but NOT to `_contrastive_objective_gpu`:

```python
def _gpu_poststdp_fused(self, ..., field_gate, ...):
    self._negative_sampling_gpu(..., field_gate, ...)   # field_gate passed
    self._contrastive_objective_gpu(gen_updates)         # field_gate NOT passed
```

CPU contrastive (`_contrastive_objective_cpu`) does NOT use `field_gate` at all — it always applies CE reweighting. **CPU and GPU both ignore field_gate for contrastive**, so parity is maintained. But the design intent is unclear: should field_gate control contrastive CE reweighting?

---

## 3. V10 Remaining Issues — Status Update

| Issue | V10 Sev | V11 Status | Notes |
|:------|:-------:|:----------:|:------|
| **SN-28**: contrastive `field_gate` | P1 | **Still P1** | `_contrastive_objective_gpu` called without `field_gate`; CPU also ignores it |
| **SN-26.2**: basis health not checked | P2 | **Still P2** | `check_basis_health()` exists, never called in subspace update path |
| **SN-33**: GPU lateral inh stale `_vecs_t` | P2 | **Still P2** | `gv` snapshot at line 513; within-loop writes to `_vecs_t` at 531 create drift |
| **SN-19**: GPU contrastive vectorization | P2 | **Still P2** | Python loops at lines 735-792 remain |
| **SN-41**: EMA counter monotonic | P3 | **Still P3** | `_ema_steps += len(unique_gen)` resets only on tensor rebuild |
| **SN-34**: GPU subspace update | P3 | → **CLOSED** | G-40 implements batched GPU subspace |
| **SN-38**: cooc_set rebuild | P3 | → **CLOSED** | `cooc_masks` pre-computed at line 708 |
| **SN-40**: field_bits overlap per-candidate | P3 | → **CLOSED** | `fb_overlaps` pre-computed at line 718 |
| **SN-42**: dead `push_total`/`lr_scale` | P3 | → **CLOSED** | Dead code removed |

---

## 4. New Issues Found — SN-43+

### 4.1 [P2] SN-43: GPU Neg Sampling Still Has Per-Element Python Loop

**File:** `stdp_trainer.py:604-624`

```python
for gi, gen_cid in enumerate(unique_gen):   # ← Python loop
    neg_lr_i = avg_elr_per_gen[gi] * neg_lr_ratio * 0.3
    if field_gate:
        neg_lr_i *= (1.0 + gen._ce_t[gen_cid] * 2.0)
    neg_mask = mask[gi]
    if not neg_mask.any():          # ← Python branch per concept
        continue
    valid_idx = noise[gi][neg_mask]  # ← Python-indexed GPU mask
    vg_i = gv[gi]
    grad = (gen._vecs_t[valid_idx].float() - sim[gi][neg_mask][:, None] * vg_i).sum(dim=0)
    ...
    gen._vecs_t[gen_cid].copy_(v_new.to(gen._vecs_t.dtype))
    cs._apply_vector_update(gen_cid, v_new.cpu().numpy())  # ← CPU roundtrip
```

**Impact:** Each concept triggers `.item()` syncs, Python branches, CPU numpy write-back. For N=100 unique_gen with neg_samples > 0, this is ~100 syncs. While G-43 vectorized the elr grouping, the write-back remains per-element.

**Fix:** Accumulate all neg-gradients into a single tensor, batch-normalize, batch-write `_vecs_t[gen_t[valid_mask]]`. Remove Python loop.

### 4.2 [P2] SN-44: GPU Contrastive Nested Python Loops

**File:** `stdp_trainer.py:735-792`

```python
for i in range(ng):                          # ← outer Python loop
    hn = []
    for j in range(max_hard):                 # ← inner loop: hard-negative selection
        neg_cid = int(best_idx[i, j].item())  # ← scalar sync
        ...
        cos_val = float(best_val[i, j].item())  # ← scalar sync
        overlap = int(fb_overlaps[i, neg_cid].item())  # ← scalar sync
    
    if fb_overlaps is not None:
        for j in range(min(50, topk_idx.shape[1])):  # ← TN-14 inner loop
            rcid = int(topk_idx[i, j].item())         # ← scalar sync
            rcos = float(topk_val[i, j].item())       # ← scalar sync
            ro = fb_overlaps[i, rcid].item()          # ← scalar sync
    
    cs._apply_vector_update(gen_idxs[i], v_new.cpu().numpy())  # ← CPU write-back
```

**Impact:** ~3-6 scalar syncs per concept per inner loop. For N=100 concepts: ~500-1000 syncs. This is the single largest remaining source of GPU→CPU syncs after V10.

**Fix:** Convert hard-negative filtering to tensor operations: use `cooc_masks` + `fb_overlaps` as boolean masks on `topk_idx`, select valid negatives via masked indexing, compute batch gradient in one shot.

### 4.3 [P2] SN-45: Destab Logic Still CPU Per-Element

**File:** `stdp_trainer.py:433-451`

```python
for gi, gen_cid in enumerate(unique_gen):
    ce = gen.concept_error.get(gen_cid, 0.0)           # CPU dict lookup
    _destab_p = min(ce * 0.5 * max(destab_scale, 0.0), 0.5)
    if destab_scale > 0 and gen.main_rng.random() < _destab_p:  # CPU RNG
        ppmi_candidates = gen.lattice.connections_of(...)        # CPU lattice query
        ...
        acc_cpu[gi] = ...  # CPU numpy mix
```

**Impact:** For each concept, destab check involves: CPU dict lookup, Python RNG, lattice query (PPMI index), numpy vector ops. For N=100 concepts with destab_scale=0.5: ~100 RNG calls + ~10-20 lattice queries.

**Fix:** Pre-compute destab decisions on GPU using `_ce_t` tensor, batch-RNG via uniform noise tensor, threshold comparison. PPMI fallback still needs CPU (lattice is CPU-only).

### 4.4 [P3] SN-46: Contrastive Write-Back Through CPU Despite GPU Computation

**File:** `stdp_trainer.py:787, 792`

```python
v_new = v_local + push  # GPU tensor
nv = v_new.norm()       # GPU
v_new /= nv             # GPU
cs._apply_vector_update(gen_idxs[i], v_new.cpu().numpy())  # ← CPU roundtrip
```

**Impact:** The gradient is computed entirely on GPU (lines 775-786), then `.cpu().numpy()` converts to numpy, `_apply_vector_update` writes to CPU store, then `_on_vector_update` hook copies back to GPU. **Double transfer** for every contrastive update.

**Fix:** Write directly to `gen._vecs_t[gen_idxs[i]]` on GPU, then sync CPU store once at end (deferred). Requires `_on_vector_update` guard to avoid double-write.

### 4.5 [P3] SN-47: CPU Neg Sampling Samples Vocabulary Per-Concept

**File:** `stdp_trainer.py:554`

```python
neg_candidates = gen.main_rng.sample(total_vocab, min(neg_samples, len(total_vocab)))
```

**Impact:** `total_vocab = list(cs.concept_vectors.keys())` is recomputed per call (line 541). `sample()` shuffles ~146K elements per concept. For N=100 concepts with neg_samples=2: 100× vocabulary sampling = ~100× 146K shuffle.

**Fix:** Lift `total_vocab` outside concept loop. Pre-sample a single set of `neg_samples` candidates per concept using `random.choices` or pre-compute a noise distribution tensor.

### 4.6 [P3] SN-48: GPU Field Overlap in `_build_pairs` Per-Pair Sync

**File:** `stdp_trainer.py:185`

```python
overlap = int(torch.bitwise_and(gen._fb_t[ids[i]], gen._fb_t[ids[j]]).sum().item())
```

**Impact:** Each STDP pair triggers a GPU→CPU scalar sync via `.item()`. For 500 pairs per sentence: 500 syncs. The CPU path computes field_overlap via numpy (no sync) but slower per-op.

**Fix:** Compute `field_weight` as deferred GPU value stored in `gpu_meta_l` column 5 (`_META_FIELD_W`) — it's already stored there (line 210), just the computation syncs. Pre-compute field weights in a single batched GPU operation before pair building.

---

## 5. Cross-Cutting Concerns

### 5.1 `_vecs_t` Written But `_codes_t` Not Updated

`_apply_vector_update` (concept_space.py:517) updates `_vecs_t` via hook and `fractal.codes[cid]`. `_apply_subspace_update_batch` (concept_space.py:591) updates `fractal.codes[cid]` and vectors via hook, but `_codes_t` tensor is NOT updated. After subspace update, `_codes_t` contains stale codes until next `_ensure_torch` rebuild. **Not a correctness bug** (no code reads `_codes_t` after subspace in the same batch), but wastes the next tensor rebuild.

### 5.2 Gradient Noise Injection Uses Global Max

```python
# stdp_trainer.py:391
acc += torch.randn_like(acc) * gradient_noise_scale * (elr_grouped[:, None] / elr_grouped.max().clamp(min=1))
```

`elr_grouped.max()` is a global max across all unique_gen. Concepts with low effective LR get noise scaled by `elr_i / max_elr`. CPU noise injection is not implemented (`gradient_noise_scale` is GPU-only). CPU/GPU parity gap if noise is used.

### 5.3 BN+LN for Vectors Suggested in V10 — Not Implemented

V10 proposed BatchNorm/LayerNorm for concept vectors to stabilize training. Not implemented. Vectors remain on unit sphere via explicit re-normalization after each update. The Riemannian gradient approach is correct but forces small LRs.

---

## 6. Summary

### 6.1 V10 Fixed (All Verified)

| ID | Status | Notes |
|:---|:------:|:------|
| SN-35 | ✅ | CPU neg sampling compound — re-read `v_gen` |
| SN-36 | ✅ | CPU contrastive compound — re-read `v_gen` |
| G-40 | ✅ | Batched GPU subspace via `_codes_t` + `_basis_t` |
| G-44 | ✅ | Pre-computed `cooc_masks` + `fb_overlaps` |
| G-46 | ✅ | Persistent `_mom_t` GPU tensor |
| SN-39 | ✅ | `connection_strength` removed from GPU loop |
| AM-30 | ✅ | Batched EMA `lerp_` on `unique_gen` slice |
| G-49 | ✅ | Pre-allocated fused buffer |
| SN-38 | ✅ | cooc_set hoisted (pre-computed mask) |
| SN-40 | ✅ | field_bits overlap batched (pre-computed tensor) |
| SN-42 | ✅ | Dead `push_total`/`lr_scale` removed |

### 6.2 Still Open

| ID | Severity | Issue |
|:---|:--------:|:------|
| SN-28 | P1 | Contrastive `field_gate` not propagated |
| SN-26.2 | P2 | Basis health not checked in subspace update |
| SN-33 | P2 | GPU lateral inhibition stale `_vecs_t` |
| SN-19 | P2 | GPU contrastive Python loops |
| SN-41 | P3 | EMA counter monotonic growth |

### 6.3 New in V11

| ID | Severity | Issue |
|:---|:--------:|:------|
| SN-43 | P2 | GPU neg sampling per-element Python loop + CPU write-back |
| SN-44 | P2 | GPU contrastive nested Python loops + scalar syncs |
| SN-45 | P2 | Destab logic CPU per-element (RNG, lattice, numpy) |
| SN-46 | P3 | Contrastive GPU→CPU→GPU roundtrip write-back |
| SN-47 | P3 | CPU neg sampling re-samples vocabulary per concept |
| SN-48 | P3 | GPU field overlap `.item()` sync per pair in `_build_pairs` |

### 6.4 Severity Count

| Severity | Count | Key Issues |
|:---------|:-----:|:-----------|
| **P1** | 1 | SN-28 |
| **P2** | 6 | SN-26.2, SN-33, SN-19, SN-43, SN-44, SN-45 |
| **P3** | 4 | SN-41, SN-46, SN-47, SN-48 |

### 6.5 V10→V11 Delta

- ✅ 11 V10 fixes verified (SN-35, SN-36, G-40, G-44, G-46, SN-39, AM-30, G-49, SN-38, SN-40, SN-42)
- ❌ 5 V10 issues still open (SN-28, SN-26.2, SN-33, SN-19, SN-41)
- 🆕 6 new issues (SN-43..SN-48)
- P1 count reduced: 4 → 1 (SN-35/36 fixed and downgraded)
- P2 count stable: 5 → 6 (SN-43/44/45 added, SN-38/40/42 closed)
- P3 count stable: 4 → 4 (SN-46/47/48 added, SN-38/40/42 closed)
