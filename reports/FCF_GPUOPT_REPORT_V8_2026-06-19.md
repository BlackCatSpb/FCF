# FCF GPU Optimization Audit — V8 (2026-06-19)

**Auditor:** GPU-Opt Agent  
**Scope:** `stdp_trainer.py` (860L) + `crystal_generator.py` (817L)  
**Config:** ~146K vocab, 384D, FP16 `_vecs_t`, CUDA (fallback CPU)

---

## 1. Fixed V7 Issues (Validation)

| Issue | Status | Location |
|-------|--------|----------|
| N-4: concept_error reweighting (0.2→0.3) | ✅ | `_negative_sampling_{cpu,gpu}` — `neg_lr * (1.0 + ce * 2.0)` |
| G-10.1: triple alloc in `_on_vector_update` | ✅ | `gen._vecs_t[gen_cid] = ...` removed; hook in `crystal_generator.py:153-156` |
| G-10.1: `_negative_sampling_gpu` per-concept `neg_lr_i` + ce | ✅ | Loop at line 574-577 |
| GPU Contrastive: topk + batched push (SN-19) | ✅ | `_contrastive_objective_gpu` line 678-754 |

---

## 2. Critical Issues Found

### P1: TN-14 Cross-Field Regularization — CPU-GPU Sync Storm ⚠️

**File:** `stdp_trainer.py:715-741`  
**Problem:** The TN-14 inner loop (up to `min(100, topk_idx.shape[1])` = 100 per concept) executes per-element:

```
for j in range(min(100, topk_idx.shape[1])):
    rcid = int(topk_idx[i, j].item())          # sync #1
    ...
    rcos = float(topk_val[i, j].item())        # sync #2
    ...
    fb_gn = gen._fb_t[gen_idxs[i]]
    fb_rn = gen._fb_t[rcid]
    ro = int(torch.bitwise_and(fb_gn, fb_rn).sum().item())  # sync #3
    ...
    cs._apply_vector_update(gen_idxs[i], v2.cpu().numpy())  # sync #4
```

**Impact:** For N concepts × 100 iterations = **N×400+ GPU→CPU syncs per training batch**. Each `.item()` stalls the CUDA stream. This alone can make the GPU path **slower than CPU**.

### P2: `_build_pairs` — Per-Pair Field Overlap Sync

**File:** `stdp_trainer.py:195`  
```python
overlap = int(torch.bitwise_and(gen._fb_t[ids[i]], gen._fb_t[ids[j]]).sum().item())
```
**Problem:** Called inside the `for i, for j` pair-building loop (O(N²) per sentence). Each call triggers CPU-GPU sync. **Fix:** defer field weight computation to the GPU kernel (meta_l already carries `field_weight`).

### P3: `_subspace_update` — CPU-Only, Not Vectorized

**File:** `stdp_trainer.py:143-155`  
**Problem:** `_subspace_update()` runs on CPU via `numpy`, called per-element even in the GPU path (line 458). The basis projection `grad @ basis.T → code_grad @ basis` is a dense matmul that should live on GPU. For 384D with `l_c + l_a + l_m` splits, the mask/lr logic is trivially vectorized.

---

## 3. FP16 / Memory Analysis

| Buffer | dtype | Allocated | Notes |
|--------|-------|-----------|-------|
| `_vecs_t` | **FP16** | 146K × 384 × 2B = 112 MB | ✅ Good for memory, but... |
| `_vecs_t.float()` casts | FP32 | **ephemeral** 224 MB | Every access `.float()` — line 373-374, 488, 563, 566, 584, 673-674, 733 |
| `_ce_t` | FP32 | 146K × 4 = 0.6 MB | ✅ |
| `_fb_t` | U8 | 146K × 128 = 19 MB | ✅ |
| `_basis_t` | FP32* | latent × 384 × 4 | FP32 from `numpy` conversion |
| `_ema_vecs_t` | **FP32** | 146K × 384 × 4 = 224 MB | Duplicated from `_vecs_t.float()` |

**Total persistent GPU:** ~112 + 0.6 + 19 + 224 = **~356 MB** (within 2GB limit).  
**Fragile point:** `_ema_vecs_t` at 224 MB is pure overhead. Consider FP16 EMA.

**FP16 precision concern:** `_vecs_t` is FP16, but:
- `.float()` is called *always* before arithmetic — the cast happens regardless of whether FP16 ops would suffice
- `_gpu_stdp_apply` line 373-374: `vc = gen._vecs_t[ctx_t].float()` — could keep FP16 and use `torch.matmul(..., dtype=torch.float32)` for the accumulator only

---

## 4. CUDA Events

**File:** `stdp_trainer.py:347-350, 476-479`  
```python
if torch.cuda.is_available():
    gen._prof_start = torch.cuda.Event(enable_timing=True)
    gen._prof_end = torch.cuda.Event(enable_timing=True)
    gen._prof_start.record()
...
gen._prof_end.record()
gen._prof_end.synchronize()
gen._prof_ms = gen._prof_start.elapsed_time(gen._prof_end)
```

**Problem:** CUDA events are **created every call** to `_gpu_stdp_apply`. Events should be created once in `__init__` and reused. Also, `_prof_ms` is never read anywhere — dead code.

---

## 5. Pre-allocated Buffer Gaps

**Good:** `_vecs_t`, `_ce_t`, `_fb_t`, `_basis_t`, `_ema_vecs_t` are persistent.

**Bad (re-allocated every call):**
- `fused` tensor (line 378) — `torch.zeros(len(unique_gen), D+1)`
- `err_grouped`, `cnt_err` (lines 389, 391)  
- `mom_gpu` (line 409)
- Contrastive intermediates: `sim`, `topk_idx`, `push_total`, `lr_scale`
- Negative sampling: `noise`, `ngv`, `sim`
- Lateral inhibition: `idxs`, `gv`, `sim`

**Total re-alloc per batch:** ~5-50 MB depending on batch size. Minor, but adds allocator churn.

---

## 6. G-31+ Optimization Recommendations

### G-31: Vectorize TN-14 with GPU Kernel (P1 Fix)
Replace the Python loop with a pure GPU kernel:
```python
# Instead of for i in range(ng): for j in range(100): ...
# Use masked selection:
sim_mask = torch.zeros(ng, n_v, device=d, dtype=torch.bool)
# Build mask: valid candidates (not self, not cooc, not strong connection)
# Then batch topk over masked + field filter as matrix operation
```
**Estimated speedup:** 10-50× on the TN-14 section.

### G-32: Defer Field Overlap to GPU Kernel (P2 Fix)
Remove `bitwise_and().sum().item()` from `_build_pairs`. Compute `field_weight` as part of `gpu_meta_l` on CPU (or skip it in `_build_pairs` and compute in the GPU kernel using `_fb_t`). 

**If `_fb_t` is on GPU:** compute `field_weight` inside `_gpu_stdp_apply` as a batched bitwise op.

### G-33: GPU `_subspace_update` — Batched
Move to GPU:
```python
def _subspace_update_gpu(self, grad_t, v_gen_t, base_lr_val):
    if self.subspace_lr is None or self.gen.cs.fractal.basis is None:
        return v_gen_t + grad_t * base_lr_val
    basis_t = self.gen._basis_t  # already on GPU
    latent_dim = basis_t.shape[0]
    code_grad = grad_t @ basis_t.T
    # Apply per-subspace LR mask
    mask = torch.zeros(latent_dim, device=grad_t.device)
    mask[:cs.l_c] = lr_c; mask[cs.l_c:cs.l_c+cs.l_a] = lr_a; mask[cs.l_c+cs.l_a:] = lr_m
    code_grad *= mask
    return v_gen_t + (code_grad @ basis_t) * base_lr_val
```
Then call once per batch from `_gpu_stdp_apply` with `g_vecs[unique_gen]` as input.

### G-34: Persistent Momentum Tensor (`_mom_t`)
Replace the CPU dict `gen._mom_buf` with a persistent GPU tensor `gen._mom_t` of shape `(V, D)`, zero-initialized. Eliminates the per-call CPU↔GPU momentum copy (lines 409-419):
```python
if mom_t is None:
    mom_t = torch.zeros(V, D, device=d, dtype=torch.float32)
mom_t[unique_gen] = momentum_mu * mom_t[unique_gen] + (1 - momentum_mu) * avg_grad
```
**Saves:** per-call GPU→CPU copy (line 419-420) + per-element dict lookup.

### G-35: Reuse CUDA Events
Move event creation to `CrystalGenerator.__init__`:
```python
self._prof_start = torch.cuda.Event(enable_timing=True)
self._prof_end = torch.cuda.Event(enable_timing=True)
```
Remove creation from `_gpu_stdp_apply`. Only record/sync.

### G-36: Fuse EMA Update with Gradient Application
In `_gpu_stdp_apply` lines 467-471, the EMA update is per-element in the Python loop. Move to a single tensor operation:
```python
if gen._ema_vecs_t is not None and gen._ema_steps >= 0:
    gen._ema_vecs_t[unique_gen] = gen._ema_decay * gen._ema_vecs_t[unique_gen] + \
                                   (1 - gen._ema_decay) * gen._vecs_t[unique_gen].float()
```
**Saves:** per-element loop (lines 467-470).

### G-37: Remove Redundant `.float()` Casts
`_vecs_t` is FP16 but every read casts to FP32. Options:
1. **Keep FP16, use AMP-style accumulation** — store as FP16, cast to FP32 only for matmul reduction sums
2. **Store as FP32** — simplify (adds 112 MB, within budget)
3. **Use `torch.matmul(..., dtype=torch.float32)`** — keeps FP16 storage, FP32 compute

**Recommendation:** Option 3 (FP16 storage + FP32 matmul) — `_vecs_t[ctx_t].float()` is wasteful when the same result is achievable via `torch.matmul(vg, vc.T, dtype=torch.float32)`.

### G-38: Vectorize `_negative_sampling_gpu` Loop
Replace line 574-595 with:
```python
# neg_lr_i = neg_lr * (1.0 + gen._ce_t[unique_gen] * 2.0)
neg_lr_vec = neg_lr * (1.0 + gen._ce_t[gen_t] * 2.0)
# mask.shape = (ng, n_neg)
valid_any = mask.any(dim=1)
if valid_any.any():
    grad = (gen._vecs_t[noise].float() - sim.unsqueeze(-1) * gv.unsqueeze(1)).mean(dim=1)
    gn = grad.norm(dim=1, keepdim=True).clamp(min=1e-10)
    grad = grad / gn * gn.clamp(max=1.0)
    v_new = gv + grad * (-neg_lr_vec[:, None])
    # normalize + apply
```

### G-39: GPU `_centroid_pull_batch`
Currently fully CPU (line 760-785). Convert:
```python
sent_t = gen._vecs_t[ids].float()
centroid = sent_t.mean(dim=0)
centroid = centroid / centroid.norm().clamp(min=1e-10)
sim = (sent_t * centroid).sum(dim=1)
v_new = sent_t + (centroid - sim[:, None] * sent_t) * sent_lr
v_new = v_new / v_new.norm(dim=1, keepdim=True).clamp(min=1e-10)
```
**Saves:** N× CPU numpy → N× GPU.

---

## 7. Summary Table

| ID | Severity | Location | Metric Impact |
|----|----------|----------|---------------|
| P1 | **HIGH** | `stdp_trainer.py:715-741` | 400+ syncs/batch — TN-14 |
| P2 | **HIGH** | `stdp_trainer.py:195` | O(N²) syncs/sentence — field overlap |
| P3 | **MED** | `stdp_trainer.py:143-155` | subspace_update on CPU, not vectorized |
| G-33 | **HIGH** | `_gpu_stdp_apply:458` | Move subspace to GPU — batch speedup |
| G-34 | **MED** | `_gpu_stdp_apply:406-420` | Replace `_mom_buf` dict → GPU tensor |
| G-35 | **LOW** | `_gpu_stdp_apply:347-350` | Events created per-call |
| G-36 | **MED** | `_gpu_stdp_apply:467-470` | Per-element EMA loop |
| G-37 | **MED** | multiple `.float()` | ~224MB ephemeral FP32 per call |
| G-38 | **MED** | `_negative_sampling_gpu:574-595` | Python loop over unique_gen |
| G-39 | **LOW** | `_centroid_pull_batch:760-785` | Full CPU centroid pull |

---

## 8. Estimated Speedup (G-31 + G-33 + G-38 applied)

- TN-14 Python loop: **10-50×** (GPU kernel vs Python)
- subspace_update: **3-10×** (batch GPU vs per-element CPU)
- negative_sampling loop: **2-5×** (vectorized vs looped)
- Reduced `.float()` casts: **1.2-1.5×** memory bandwidth

**Overall GPU path:** estimated **3-8×** faster than current V8 implementation.

---

## V8 Commit Fixes
- **SN-15 subspace update**: `_apply_subspace_update` — прямой code update, bypass vector roundtrip
- **TN-14 stale vector**: локальная копия `v_local`, единственный `_apply_vector_update`

*Report generated 2026-06-19 by GPU-Opt Agent (V8 audit)*
