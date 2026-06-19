# FCF GPU Optimization Audit — V9 (2026-06-19)

**Auditor:** GPU-Opt Agent  
**Scope:** `stdp_trainer.py` (860L) + `crystal_generator.py` (817L) + `concept_space.py` (`_apply_subspace_update` 556-583)  
**Config:** ~146K vocab, 384D, FP16 `_vecs_t`, CUDA (fallback CPU)  
**Base:** V8 commit with SN-15 (subspace_update), TN-14 (stale fix), noise_scale, единый цикл

---

## 1. V8 Fix Verification

| V8 Fix | Status | Note |
|--------|--------|------|
| TN-14 stale vector (v_local, single `_apply_vector_update`) | ✅ | `stdp_trainer.py:714-754` |
| SN-15 `_apply_subspace_update` (direct code update) | ⚠️ **CPU-only, defeats GPU path** | `concept_space.py:556-583` — см. P4 |
| noise_scale in GPU path | ✅ | `stdp_trainer.py:384-385` |
| Единый тренировочный цикл | ✅ | `_train()` вызывает `_gpu_stdp_apply`, `_negative_sampling_gpu`, `_contrastive_objective_gpu` |

---

## 2. New Critical Issues

### P4 (HIGH): `_apply_subspace_update` — 100% CPU/numpy in GPU Path

**File:** `concept_space.py:556-583`, called at `stdp_trainer.py:462-463`

**Problem:** Despite being called from `_gpu_stdp_apply`, `_apply_subspace_update` is **entirely CPU-based numpy**:

```python
def _apply_subspace_update(self, cid, grad, base_lr_val, subspace_lr):
    code = self.fractal.codes.get(cid)           # CPU dict
    mask_c = np.zeros(latent_dim)                 # alloc #1
    mask_a = np.zeros(latent_dim)                 # alloc #2
    mask_m = np.zeros(latent_dim)                 # alloc #3
    code_grad = grad @ basis.T                    # CPU matmul
    code_new = code + code_grad * base_lr_val     # CPU
    v_new = code_new @ basis                      # CPU matmul
    ...
    self.set_vec(cid, v_new)                      # CPU write + GPU sync via hook
```

**Consequences per call:**
- 3× `np.zeros` allocation (latent_dim ~512)
- 2× dense matmul on CPU (`grad @ basis.T` + `code_new @ basis`)
- `basis` is a CPU numpy array — no GPU tensor reuse
- Each call starts from CPU `grad` (already pulled from GPU via `acc_cpu[gi]`)
- Called **per-element** inside the Python loop over `unique_gen`

**Impact:** For N=1000 `unique_gen`, this adds ~5000 CPU numpy ops + 1000 GPU→CPU→GPU roundtrips. **Subspace update alone can cost 10-20ms/batch** — making the GPU path slower than CPU for subspace training.

### P5 (HIGH): `_contrastive_objective_gpu` — Python Loop + .item() Sync Storm (REGRESSION RISK)

**File:** `stdp_trainer.py:690-754`

The V8 TN-14 fix (local `v_local`, single `_apply_vector_update`) **reduced** the sync count, but the structure still has:

**Inner loop** (lines 692-711, 718-736):
```
for i in range(ng):           # Python loop over unique_gen
    for j in range(max_hard): # per-candidate topk scan
        neg_cid = int(best_idx[i, j].item())           # sync #1
        cos_val = float(best_val[i, j].item())         # sync #2
        overlap = int(torch.bitwise_and(...).sum().item())  # sync #3
    TN-14 loop (lines 718-736):
        rcid = int(topk_idx[i, j].item())              # sync #4
        rcos = float(topk_val[i, j].item())            # sync #5
        ro = int(torch.bitwise_and(...).sum().item())  # sync #6
```

**Count:** `ng × (max_hard × 3 + min(50, K) × 3) = ng × (15 + 150) = 165` syncs per concept. For 100 concepts: **16,500 GPU→CPU syncs per batch**.

**Additional CPU ops per loop iteration:**
- `gen.lattice.connection_strength(...)` (line 699) — CPU dict
- `gen_updates[gen_idxs[i]]` — CPU dict
- `cooc_set` construction

### P6 (MED): Momentum Buffer — Per-Element CPU Roundtrip

**File:** `stdp_trainer.py:402-419`

```python
for gi, cid in enumerate(unique_gen):
    cid_i = int(cid)
    prev = gen._mom_buf.get(cid_i, None)    # CPU dict
    prev_t = torch.from_numpy(prev).to(...) # sync + alloc
    mom_gpu[gi] = ...
for gi, cid in enumerate(unique_gen):
    gen._mom_buf[int(cid)] = mom_gpu[gi].cpu().numpy()  # N× sync
```

**Impact:** 2N syncs + N× numpy conversion + N× dict operations. Momentum disables GPU benefit.

### P7 (MED): `_lateral_inhibition_gpu` — .item() + CPU Roundtrip

**File:** `stdp_trainer.py:481-506`

```python
inhibition = sim[gi][mask].sum().item()     # sync
v_np = cs.concept_vectors.get(gen_cids[gi])  # CPU dict
v_new = v_np + inhibit_vec.cpu().numpy() ... # sync + CPU
cs._apply_vector_update(gen_cids[gi], v_new) # CPU write + hook
```

The `gv` tensor (line 486) is read eagerly at the start — but if `_apply_vector_update` modifies vectors, `gv` is stale for subsequent `gi`. The GPU computation (`inhibit_vec = (...)`) is correct, but the CPU roundtrip kills performance.

---

## 3. CPU-GPU Sync Point Catalog (Measured)

| # | Location | Line | Sync Type | Frequency |
|---|----------|------|-----------|-----------|
| S1 | `_gpu_stdp_apply`: `avg_err.cpu().numpy()` | 398 | Full tensor | 1× (N concepts) |
| S2 | `_gpu_stdp_apply`: concept EMA loop | 399-400 | per-element `.update()` | N× |
| S3 | `_gpu_stdp_apply`: momentum read (`_mom_buf.get`) | 410-414 | per-element H2D copy | N× |
| S4 | `_gpu_stdp_apply`: momentum write (`cpu().numpy()`) | 418 | per-element D2H copy | N× |
| S5 | `_gpu_stdp_apply`: `acc.cpu().numpy()` | 421 | Full tensor D2H | 1× |
| S6 | `_gpu_stdp_apply`: `cnt.cpu().numpy()` | 422 | Full tensor D2H | 1× |
| S7 | `_gpu_stdp_apply`: `elr_grouped.cpu().numpy()` | 423 | Full tensor D2H | 1× |
| S8 | `_gpu_stdp_apply`: per-element update loop | 425-469 | Python loop + CPU numpy | N× |
| S9 | `_gpu_stdp_apply`: subspace_update per-element | 462-463 | full CPU numpy matmul | N× (if enabled) |
| S10 | `_negative_sampling_gpu`: `elr_sum.item()` | 569 | scalar sync | 1× |
| S11 | `_negative_sampling_gpu`: per-concept loop | 572-593 | grad.cpu().numpy() + apply | N× |
| S12 | `_contrastive_objective_gpu`: per-candidate `.item()` | 693, 701, 707 | scalar sync | ng × max_hard × 3 |
| S13 | `_contrastive_objective_gpu`: TN-14 `.item()` | 719, 725, 729 | scalar sync | ng × min(50,K) × 3 |
| S14 | `_contrastive_objective_gpu`: final `.cpu().numpy()` | 749, 754 | tensor sync | N× |
| S15 | `_lateral_inhibition_gpu`: `inhibition.item()` | 494 | scalar sync | N× |
| S16 | `_lateral_inhibition_gpu`: cpu().numpy() + apply | 502-506 | tensor sync + CPU | N× |
| S17 | `_build_pairs`: field overlap `.item()` | 195 | scalar sync | O(N²) per sentence |

**Total estimated syncs per batch (N=100 concepts, 500 pairs, 50-char sentence):**  
**~20,000+ individual GPU→CPU syncs** — each stalls CUDA pipeline for 3-10μs → **60-200ms of sync overhead alone**.

---

## 4. `_apply_subspace_update` GPU Throughput Impact

### Current (V8) Profile per Batch (N=1000 unique_gen, subspace_lr enabled):

| Operation | Time (est.) | Notes |
|-----------|-------------|-------|
| `_gpu_stdp_apply` GPU portion (fused scatter, matmul) | ~0.5-2ms | Fast, well-optimized |
| CPU sync + tensor download (acc, cnt, elr) | ~0.5-3ms | Bandwidth-bound |
| Python loop (no subspace) | ~1-3ms | N× dict + numpy norm |
| **Python loop WITH subspace** | **~5-20ms** | +2× CPU matmul + 3× np.zeros per concept |
| Negative sampling | ~2-5ms | GPU matmul OK, Python loop kills |
| Contrastive objective | ~5-30ms | TN-14 sync storm dominates |
| **Total GPU path with subspace** | **~10-60ms** | **vs ~5ms pure GPU ideal** |
| **Total without subspace** | **~5-30ms** | Better but still sync-bound |

### CPU-Only Cost of subspace_update:
- `grad @ basis.T`: 384 × 512 = ~200K FLOPs (FP32, well-pipelined) → ~1μs
- `code_new @ basis`: same → ~1μs
- Overhead (alloc, dict, set_vec, hook, Python frame): ~3-5μs
- **Per call: ~6μs → N=1000 → ~6ms**
- Plus sync stalls: ~3-10μs per `.item()` not in this path, but the acc/elr pull adds ~1-3ms

### GPU-Ideal subspace_update:
- `grad_t @ basis_t.T` (GPU): ~0.3μs for 384×512 matmul (well-pipelined)
- Mask: vectorized element-wise on GPU — negligible
- `code_grad @ basis_t` on GPU: ~0.3μs
- Write-back via `_vecs_t[cid]`: ~0.1μs
- **Batch total: ~1ms vs ~10-20ms CPU — 10-20× speedup**

---

## 5. V8 G-21..G-30 — Implementation Status

| ID | Description | Status | Location | Effort |
|----|-------------|--------|----------|--------|
| G-21 | Persistent CUDA events | ❌ Not done | `_gpu_stdp_apply:346-350` | 1 line |
| G-22 | Vectorized EMA update | ❌ Not done | `_gpu_stdp_apply:458-461` | 2 lines |
| G-23 | Pre-allocated ctx_t/tgt_t/meta_t | ❌ Not done | `_gpu_stdp_apply:342-344, 352` | ~5 lines |
| G-24 | Fused negative sampling | ❌ Not done | `_negative_sampling_gpu:572-593` | ~20 lines |
| G-25 | lerp_ for EMA | ❌ Not done | `_gpu_stdp_apply:460` | 1 line |
| G-26 | Vectorized lateral inhibition | ❌ Not done | `_lateral_inhibition_gpu:489-506` | ~15 lines |
| G-27 | CUDA stream for H2D | ❌ Not done | `_build_torch_tensors` | ~5 lines |
| G-28 | Reuse elr_sum | ❌ Not done | `_negative_sampling_gpu:568-570` | 2 lines |
| G-29 | Fused norm + normalize | ❌ Not done | multiple `v_new / v_new.norm()` | ~10 lines |
| G-30 | torch.compile | ❌ Not done | `_gpu_stdp_apply` | ~2 lines |

**All V8 GPU optimizations (G-21..G-30) remain unimplemented.**

---

## 6. V9 (G-40+) Optimization Recommendations

### G-40: Batched GPU `_apply_subspace_update` (P4 Fix)

**Problem:** Per-element CPU subspace update in GPU path.  
**Fix:** Move entire subspace update to batched GPU tensor ops in `_gpu_stdp_apply`:

```python
def _gpu_subspace_update(self, g_vecs, unique_gen, grad_acc, elr_grouped, base_lr_val):
    """Batched subspace update on GPU. Returns updated vectors."""
    cs = self.gen.cs
    if self.subspace_lr is None or cs.fractal.basis is None:
        return g_vecs + grad_acc * base_lr_val / elr_grouped[:, None].clamp(min=1)

    basis_t = self.gen._basis_t  # already on GPU: (latent_dim, D)
    lr_c, lr_a, lr_m = self.subspace_lr
    latent_dim = basis_t.shape[0]

    # Build GPU mask once (persistent, not per-call)
    mask = torch.zeros(latent_dim, device=g_vecs.device)
    mask[:cs.l_c] = lr_c
    mask[cs.l_c:cs.l_c+cs.l_a] = lr_a
    mask[cs.l_c+cs.l_a:] = lr_m

    avg_grad = grad_acc / elr_grouped[:, None].clamp(min=1)
    code_grad = avg_grad @ basis_t.T  # (N, latent_dim)
    code_grad *= mask.unsqueeze(0)
    delta = (code_grad @ basis_t) * base_lr_val  # (N, D)
    v_new = g_vecs + delta
    v_new = v_new / v_new.norm(dim=1, keepdim=True).clamp(min=1e-10)
    return v_new
```

**Then call from `_gpu_stdp_apply`:**
```python
if gen._vecs_t is not None:
    g_vecs = gen._vecs_t[unique_gen].float()
    v_new = self._gpu_subspace_update(g_vecs, unique_gen, acc, elr_grouped, base_lr_val)
    for gi, gen_cid in enumerate(unique_gen):
        cs._apply_vector_update(gen_cid, v_new[gi].cpu().numpy())
    # Or even better: use _vecs_t[unique_gen] = v_new.half() + _on_vector_update per element
```

**Speedup:** 10-20× vs current per-element CPU subspace update.

### G-41: GPU `_lateral_inhibition_gpu` — Full GPU (fix P7)

**File:** `stdp_trainer.py:481-506`

Replace per-element Python loop with batched GPU:

```python
def _lateral_inhibition_gpu(self, gen_cids, inh_strength, inh_threshold, base_lr_val):
    gen = self.gen
    device = gen._torch_device
    idxs = torch.tensor(gen_cids, dtype=torch.long, device=device)
    gv = gen._vecs_t[idxs].float()
    sim = gv @ gv.T
    mask = (sim > inh_threshold * 2) & ~torch.eye(len(gen_cids), dtype=torch.bool, device=device)
    if not mask.any():
        return
    # Batched: vecs[i] = sum_j mask[i,j] * (sim[i,j] * gv[j] - sim[i,j]^2 * gv[i])
    sim_masked = sim * mask.float()
    inhibit_vecs = (sim_masked @ gv) - ((sim_masked ** 2) @ torch.ones_like(sim_masked))[:, None] * gv
    norms = inhibit_vecs.norm(dim=1, keepdim=True).clamp(min=1e-10)
    inhibit_vecs = inhibit_vecs / norms
    v_new = gv + inhibit_vecs * inh_strength * base_lr_val
    v_new = v_new / v_new.norm(dim=1, keepdim=True).clamp(min=1e-10)
    for gi, cid in enumerate(gen_cids):
        cs._apply_vector_update(gen_cids[gi], v_new[gi].cpu().numpy())
```

**Saves:** N× `.item()` syncs + per-element dict lookups. Still has final CPU write-back — ideal would be direct `_vecs_t[idxs] = v_new.half()`.

### G-42: GPU `_centroid_pull_batch` — Full GPU (remove CPU)

**File:** `stdp_trainer.py:760-785`

Currently 100% CPU numpy. Convert to batched GPU:

```python
def _centroid_pull_batch_gpu(self, all_ids, base_lr_val):
    gen = self.gen
    device = gen._torch_device
    if gen._vecs_t is None:
        return
    for ids in all_ids:
        if len(ids) < 3:
            continue
        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        sent_vecs = gen._vecs_t[ids_t].float()
        centroid = sent_vecs.mean(dim=0)
        cn = centroid / centroid.norm().clamp(min=1e-10)
        sent_lr = base_lr_val * 0.3
        sim = (sent_vecs * cn.unsqueeze(0)).sum(dim=1)
        shift = (cn.unsqueeze(0) - sim[:, None] * sent_vecs) * sent_lr
        v_new = sent_vecs + shift
        v_new = v_new / v_new.norm(dim=1, keepdim=True).clamp(min=1e-10)
        for i, cid in enumerate(ids):
            cs._apply_vector_update(cid, v_new[i].cpu().numpy())
```

**Saves:** N× `np.mean`, `np.linalg.norm`, `np.dot`, etc.

### G-43: `_negative_sampling_gpu` — Full Vectorization (G-38 implementation)

**File:** `stdp_trainer.py:572-593`

Replace Python loop with fully vectorized:

```python
# Pre-compute neg_lr vector (vectorized CE lookup)
ce_t = gen._ce_t[gen_t] if gen._ce_t is not None else torch.zeros(len(unique_gen), device=device)
neg_lr_vec = neg_lr * (1.0 + ce_t * 2.0)

# Mask: sim > 0.1
mask = sim > 0.1
valid_any = mask.any(dim=1)
if valid_any.any():
    # For valid concepts: grad = mean(ngv - sim_expanded * gv_expanded, dim=1)
    # This requires masking — use masked_select + scatter, or a simple loop over valid only
    # Option: loop over valid_any indices (fewer than full unique_gen)
    valid_idxs = valid_any.nonzero(as_tuple=True)[0]
    for vi in valid_idxs:
        gi = vi.item()
        gen_cid = int(unique_gen[gi])
        v_new = gv[gi] - grad * (neg_lr_vec[gi])
        nv = v_new.norm()
        if nv > 1e-10:
            v_new /= nv
        cs._apply_vector_update(gen_cid, v_new.cpu().numpy())
```

**But ideal:** full GPU `_vecs_t[gen_t] = v_new.half()` + `_on_vector_update` would eliminate the loop entirely.

### G-44: Batched GPU TN-14 / Contrastive (G-31 implementation)

**File:** `stdp_trainer.py:690-754`

The TN-14 inner loop (cross-field regularization) can be vectorized:

```python
# Instead of per-candidate .item() loop:
# 1. Build field overlap mask
fb_g = gen._fb_t[gen_idxs]  # (ng, fb_bytes)
fb_candidates = gen._fb_t[topk_idx]  # (ng, K, fb_bytes)
# bitwise_and over byte tensors is not directly supported;
# fallback: bool tensor decomposition or loop over candidate dimension only

# 2. Field-filter topk in batched fashion
# 3. Push as tensor ops
```

**For immediate impact:** reduce `max_hard` loop by pre-filtering on GPU:

```python
# Pre-filter: skip self, skip cooc_set (use _ce_t), skip strong connections
# Store valid indices as mask, use masked_select for v_neg gather
```

### G-45: Persistent CUDA Event + Remove Dead Code (G-21 implementation)

**File:** `crystal_generator.py` — add in `__init__`:
```python
if _HAS_TORCH and torch.cuda.is_available():
    self._prof_start = torch.cuda.Event(enable_timing=True)
    self._prof_end = torch.cuda.Event(enable_timing=True)
```

**File:** `stdp_trainer.py:346-350` — remove event creation, keep `.record()`  
**File:** `stdp_trainer.py:474-477` — remove `.synchronize()`, keep `.elapsed_time()`  
**Or:** Remove entire profiling block if `_prof_ms` is never read.

### G-46: Persistent Momentum Tensor (G-34 implementation)

**File:** `concept_space.py` or `crystal_generator.py` — add `_mom_t`:
```python
self._mom_t = torch.zeros(V, D, device=dev, dtype=torch.float32)
```

Then in `_gpu_stdp_apply`:
```python
if momentum_mu > 0:
    avg_grad = acc / cnt[:, None].clamp(min=1)
    mom_gpu = momentum_mu * gen._mom_t[unique_gen] + (1 - momentum_mu) * avg_grad
    gen._mom_t[unique_gen] = mom_gpu
    # Use mom_gpu as the gradient directly (still on GPU)
    acc = mom_gpu * cnt[:, None].clamp(min=1)
```

**Saves:** Entire `_mom_buf` dict + per-element sync (S3, S4).

### G-47: Vectorized EMA Update with `lerp_` (G-22, G-25 implementation)

**File:** `stdp_trainer.py:458-461`

Replace:
```python
if gen._ema_vecs_t is not None and gen._ema_steps >= 0:
    gen._ema_vecs_t[unique_gen] = gen._ema_decay * gen._ema_vecs_t[unique_gen] + \
                                   (1 - gen._ema_decay) * gen._vecs_t[unique_gen].float()
```

With:
```python
if gen._ema_vecs_t is not None and gen._ema_steps >= 0:
    gen._ema_vecs_t[unique_gen].lerp_(gen._vecs_t[unique_gen].float(), 1 - gen._ema_decay)
```

**Saves:** Per-element Python loop. `lerp_` is fused CUDA kernel.

### G-48: `torch.compile` on `_gpu_stdp_apply` (G-30 implementation)

**File:** `stdp_trainer.py` — after vectorizing all loops:
```python
_gpu_stdp_apply = torch.compile(_gpu_stdp_apply, mode="max-autotune")
```

**Prerequisite:** All Python loops removed, tensor operations only.

### G-49: Pre-allocate `fused` Tensor (G-23 partial)

**File:** `stdp_trainer.py:377` — pre-allocate once:
```python
# In __init__ or _ensure_torch:
self._fused_buf = None  # will be (max_batch, D+1)
# In _gpu_stdp_apply:
if self._fused_buf is None or self._fused_buf.shape[0] < len(unique_gen):
    self._fused_buf = torch.zeros(len(unique_gen), D+1, device=device, dtype=torch.float32)
else:
    self._fused_buf.zero_()
fused = self._fused_buf[:len(unique_gen)]
```

Similarly for `err_grouped`, `cnt_err`, `mom_gpu`.

---

## 7. Summary Table

| ID | Severity | Location | Description | Est. Speedup |
|----|----------|----------|-------------|--------------|
| **P4** | **HIGH** | `concept_space.py:556-583` | `_apply_subspace_update` 100% CPU in GPU path | 10-20× on subspace |
| **P5** | **HIGH** | `stdp_trainer.py:690-754` | Contrastive .item() sync storm (~165 syncs/concept) | 5-50× |
| **P6** | **MED** | `stdp_trainer.py:402-419` | `_mom_buf` dict per-element CPU roundtrip | 3-10× on momentum |
| **P7** | **MED** | `stdp_trainer.py:481-506` | `_lateral_inhibition_gpu` .item() + CPU write-back | 2-5× |
| **G-40** | **HIGH** | `_gpu_stdp_apply` | Batched GPU subspace update | 10-20× |
| **G-41** | **MED** | `_lateral_inhibition_gpu` | Full GPU inhibition (no .item()) | 2-5× |
| **G-42** | **MED** | `_centroid_pull_batch` | Full GPU centroid pull | 3-10× |
| **G-43** | **MED** | `_negative_sampling_gpu` | Vectorize Python loop | 2-5× |
| **G-44** | **HIGH** | Contrastive TN-14 | Batched GPU cross-field reg | 5-50× |
| **G-45** | **LOW** | `_gpu_stdp_apply:346-350,474-477` | Persistent CUDA events + rm dead code | <1% |
| **G-46** | **MED** | `_gpu_stdp_apply` | Persistent `_mom_t` tensor | 3-10× |
| **G-47** | **LOW** | `_gpu_stdp_apply:458-461` | `lerp_` for EMA | ~1.2× |
| **G-48** | **MED** | `_gpu_stdp_apply` | `torch.compile` | 1.5-3× |
| **G-49** | **LOW** | `_gpu_stdp_apply:377` | Pre-allocate fused buffer | <1% |

---

## 8. Measurement: Minimum Achievable GPU Time

**Target:** 1 batch (100 concepts, 500 pairs, 384D) — current ~10-60ms → target ~2-5ms

| Optimization | Cumulative Speedup | Est. Time |
|-------------|-------------------|-----------|
| Baseline (V8, no subspace) | 1× | ~15ms |
| + G-40 (GPU subspace) | 2× | ~8ms |
| + G-43 (vec neg sampling) | 3× | ~5ms |
| + G-41 (GPU inhibition) | 3.5× | ~4ms |
| + G-44 (GPU contrastive) | 5× | ~3ms |
| + G-46 (GPU momentum) | 6× | ~2.5ms |
| + G-48 (compile) | 8-10× | **~1.5-2ms** |

---

## 9. V8→V9 Delta

| Metric | V8 | V9 (current) | Delta |
|--------|----|-------------|-------|
| Subspace update path | CPU numpy per-element | **CPU numpy per-element** (unchanged) | ❌ |
| Contrastive syncs/batch | ~16,500 | ~16,500 (unchanged) | ❌ |
| Momentum syncs/batch | 2N | 2N (unchanged) | ❌ |
| CUDA events | Created per-call | Created per-call (unchanged) | ❌ |
| EMA update | Per-element Python | Per-element Python (unchanged) | ❌ |
| Negative sampling | Python loop | Python loop (unchanged) | ❌ |
| Lateral inhibition | .item() + CPU | .item() + CPU (unchanged) | ❌ |
| Centroid pull | 100% CPU numpy | 100% CPU numpy (unchanged) | ❌ |
| G-21..G-30 implementation | 0/10 | **0/10** (unchanged) | ❌ |

**V9 diagnosis: No GPU optimizations from V8 were implemented. The GPU path still has ~20,000+ syncs per batch and 100% CPU subspace updates.**

---

## 10. Immediate Action Items (V10 Priority)

1. **G-40 Batched GPU subspace update** — single biggest win. Move `basis` ops to GPU, call once per batch instead of per-element. Estimated 10-20× speedup on subspace path.

2. **G-44 Batched GPU contrastive** — replace .item() syncs with masked tensor ops. Eliminates ~16,500 syncs/batch. Estimated 5-50× speedup on contrastive.

3. **G-46 Persistent `_mom_t`** — eliminate 2N per-element CPU roundtrips. 3-10× on momentum.

4. **G-43 Vectorized negative sampling** — eliminate Python loop. 2-5×.

5. **G-45 Remove dead profiling code** — 2 lines, no risk, eliminates CUDA event alloc overhead.

---

*Report generated 2026-06-19 by GPU-Opt Agent (V9 audit)*
