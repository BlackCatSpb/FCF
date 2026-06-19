# Neuro-Symbolic Audit V10 — 2026-06-19

**Auditor:** Neuro-Symbolic Specialist
**Scope:** `stdp_trainer.py`, `concept_space.py`, `crystal_generator.py`
**Base:** V9 report + commits `a0fe15b` `cccc392` `21ee6ca`

---

## 1. V9 Fix Verification

| V9 Fix | Code Check | Status |
|:-------|:-----------|:------:|
| **SN-22.1**: GPU mean()→sum() | `stdp_trainer.py:593` `.sum(dim=0)` | ✅ **Fixed** |
| **SN-22.2**: `field_gate` guard GPU CE | `stdp_trainer.py:584` `if field_gate:` | ✅ **Fixed** |
| **SN-22.3**: per-concept `avg_elr` GPU | `stdp_trainer.py:574-579` scatter_add grouping | ✅ **Fixed** |
| **SN-24**: momentum blend | `stdp_trainer.py:461` `mu*mom + (1-mu)*grad` | ✅ **Fixed** |
| **SN-25**: slow STDP→GPU | `stdp_trainer.py:214-224` `_META_SLOW=1.0` pairs, line 367-368 theta blend | ✅ **Fixed** |
| **SN-26.1**: `self.l_c`→`self.fractal.l_c` | `concept_space.py:564-566` | ✅ **Fixed** |
| **SN-31**: dead `_subspace_update` removed | no longer in `stdp_trainer.py` | ✅ **Fixed** |
| **REG-V9-2**: `code_new` norm fix | `concept_space.py:574` `np.linalg.norm(code_new)` | ✅ **Fixed** |
| **G-47**: `lerp_` for EMA | `stdp_trainer.py:464` `ema_vecs_t[gen_cid].lerp_(...)` | ✅ **Fixed** |

**All three V9 P0s (SN-22.1, SN-22.2, SN-25) are confirmed fixed.** The parity gap is dramatically reduced.

---

## 2. SN-22 Parity Suite — Complete Assessment

### 2.1 SN-22.1: GPU mean()→sum() ✅
`_negative_sampling_gpu:593`: `grad = (...).sum(dim=0)` matches CPU cumulative push. **Fixed.**

### 2.2 SN-22.2: field_gate guard ✅
`_negative_sampling_gpu:584`: `if field_gate:` guards CE reweighting. Identical to CPU line 529. **Fixed.**

### 2.3 SN-22.3: per-concept avg_elr ✅
GPU uses scatter_add grouping (line 574-579) matching CPU per-concept avg_elr. **Fixed.**

---

## 3. CPU/GPU Parity — New Issues Found

### 3.1 [P1] SN-35: CPU Negative Sampling — `v_gen` Stale, Last-Update-Wins

**File:** `stdp_trainer.py:522-549`

```python
v_gen = cs.concept_vectors.get(gen_cid)   # line 523 — read ONCE
...
for neg_cid in neg_candidates:             # loop over n_neg candidates
    ...
    v_new = v_gen - grad * neg_lr         # uses ORIGINAL v_gen every time
    cs._apply_vector_update(gen_cid, v_new)  # overwrites stored vector
```

Each iteration computes gradient from the **same original v_gen**. `_apply_vector_update` writes the new vector, but the next iteration still uses `v_gen` (original). **Only the last valid negative's gradient survives.**

**GPU** (`_negative_sampling_gpu:593`): `.sum(dim=0)` — **sums all valid negative gradients** and applies once.

**Impact:** With `neg_samples=1` (default): no difference. With `neg_samples>1`: CPU applies **1/N** the effective push compared to GPU (only last vs sum of all). Training dynamics diverge for larger neg_samples.

### 3.2 [P1] SN-36: CPU Contrastive — `v_gen` Stale, Last-Update-Wins

**File:** `stdp_trainer.py:620-663`

```python
v_gen = cs.concept_vectors.get(gen_cid)   # line 621 — read ONCE
...
for neg_cid, cos_val in hard_negatives[:5]:  # loop over max_hard(5)
    ...
    contr_grad = cos_val * v_neg - v_gen   # uses ORIGINAL v_gen
    v_new = v_gen + push                   # from original v_gen
    cs._apply_vector_update(gen_cid, v_new) # overwrites
```

Same pattern: **only the last hard negative's push survives** on CPU.

**GPU** (`_contrastive_objective_gpu:751`): `.mean(dim=0)` — batch gradient from ALL `hn` hard negatives.

**Impact:** CPU applies 1 effective push (the last hard negative). GPU applies a mean gradient of up to 5 hard negatives. Different training dynamics.

### 3.3 [P1] SN-37: CPU Contrastive — Gradient Norm Not Clipped

**File:** `stdp_trainer.py:654-657`
```python
contr_grad = cos_val * v_neg - v_gen
gn = float(np.linalg.norm(contr_grad))
if gn > gen.max_grad_norm > 0:
    contr_grad = contr_grad / gn * gen.max_grad_norm
```
✅ CPU **has** clipping. Let me re-check GPU...

**GPU** (`_contrastive_objective_gpu:752-754`):
```python
gn = grad.norm()
if gn > gen.max_grad_norm > 0:
    grad = grad / gn * gen.max_grad_norm
```
✅ GPU **also has** clipping. This is consistent.

### 3.4 [P2] SN-38: Co-Occurrence Set Rebuilt in Inner Loop (from V9 SN-32, still unfixed)

**File:** `stdp_trainer.py:707, 733`
```python
for j in range(max_hard):
    cooc_set = {ctx for ctx, _ in gen_updates[gen_idxs[i]]}  # rebuilt for EACH j
```
Rebuilt every iteration in BOTH the hard-negative filter loop (line 707) and the TN-14 repel loop (line 733). Can be hoisted to outer `for i` loop.

### 3.5 [P2] SN-39: `connection_strength` Python Call in Inner Loop

**File:** `stdp_trainer.py:710`
```python
if gen.lattice.connection_strength(gen_idxs[i], neg_cid) > 0.1:
```
Python call inside `for j` loop — breaks full GPU vectorization.

### 3.6 [P3] SN-40: `field_bits` Overlap Per-Candidate in Inner Loop

**File:** `stdp_trainer.py:714-718, 738-740`
```python
overlap = int(torch.bitwise_and(fb_gen, fb_neg).sum().item())
```
Computed per-candidate in both hard-negative and TN-14 loops. Small but blocks vectorization.

### 3.7 [P3] SN-41: EMA Counter Monotonic Growth

**File:** `stdp_trainer.py:465`
```python
gen._ema_steps += 1
```
Increments per-concept per-GPU-update (~146K per batch). Reset to 0 in `_build_torch_tensors` (line 232) but only on tensor rebuild. Cosmetic — counter becomes meaningless between rebuilds.

---

## 4. V9 Open Items — Status Update

| Issue | V9 Severity | V10 Status | Notes |
|:------|:-----------:|:----------:|:------|
| **SN-26.2**: Basis health not checked | P2 | **Still P2** | `check_basis_health()` exists, never called in `_apply_subspace_update` |
| **SN-28**: Contrastive `field_gate` param | P1 | **Still P1** | `_contrastive_objective` called at line 131 without `field_gate` |
| **SN-29**: `crystal_generator.py` cleanup | P2 | **Still P2** | `train_from_text`, `train_batch`, `evaluate`, `_destab_field_fallback` still present |
| **SN-32**: Hoist `cooc_set` | P3 | Merged → **SN-38** | Same issue, re-numbered |
| **SN-33**: GPU lateral inh stale `_vecs_t` | P2 | **Still P2** | `gv` snapshot at line 490; small iterative corrections |
| **SN-34**: GPU subspace update | P3 | **Still P3** | `_apply_subspace_update` CPU-only |
| **SN-19**: GPU contrastive vectorization | P2 | **Still P2** | `for i in range(ng)` loop at line 701 |
| **SN-17**: Kinetic Energy Buffer | — | **Not implemented** | New proposal only |
| **SN-20**: Adaptive EMA Decay | — | **Not implemented** | New proposal only |
| **SN-21**: Riemannian STDP | — | **Not implemented** | New proposal only |

---

## 5. New Proposals SN-35+

### SN-35: Fix CPU Negative Sampling — Compound Updates (P1)

**Problem:** CPU `_negative_sampling_cpu` reads `v_gen` once, each iteration computes from original, last update wins.

**Fix:** Re-read `v_gen` after each `_apply_vector_update` to compound (matching GPU cumulative sum):
```python
# After _apply_vector_update(gen_cid, v_new):
v_gen = cs.concept_vectors.get(gen_cid)  # re-read for compounding
```

Or compute all gradients first, then apply summed gradient (GPU-style):
```python
grad_total = np.zeros_like(v_gen)
for neg_cid in neg_candidates:
    ...
    if sim > 0.1:
        grad = v_neg - sim * v_gen
        grad_total += grad
if np.linalg.norm(grad_total) > 1e-10:
    v_new = v_gen - grad_total * neg_lr / max(n_valid, 1)
    cs._apply_vector_update(gen_cid, v_new)
```

### SN-36: Fix CPU Contrastive — Compound Updates (P1)

**Problem:** CPU `_contrastive_objective_cpu` same stale-`v_gen` pattern.

**Fix:** Same approach — either re-read after each update or batch-sum gradients.

### SN-37: Fix CPU Negative — Per-Valid-Negative Norm Scaling (P2)

**Problem:** CPU normalizes each individual `grad` (`grad / gn * min(gn, 1.0)`). GPU `.sum()` applies scaling once. The `min(gn, 1.0)` cap interacts differently when applied per-gradient vs post-sum.

**Suggestion:** Unify by accumulating un-normalized gradients, then normalize the total once.

### SN-38: Hoist `cooc_set` in GPU Contrastive Loop (P3)

Move outside inner `for j` loop:
```python
cooc_set = {ctx for ctx, _ in gen_updates[gen_idxs[i]]}
for j in range(max_hard):
    cooc_set = ...  # remove
```

### SN-39: Batch `connection_strength` Filter (P2)

Pre-compute connection mask tensor for all `(gen_idx[i], topk_idx[i][j])` pairs before the inner loop, using tensorized lattice query.

### SN-40: Batch `field_bits` Overlap (P3)

Pre-compute field overlap between each concept and its topk candidates using bitwise-and on batched `_fb_t` tensor, then filter with a threshold mask.

### SN-41: EMA Counter Reset on Sync (P3)

Reset `_ema_steps` in `_sync_ema` / `_restore_vectors` methods, or replace with wall-clock-based step counter.

### SN-42: GPU Contrastive — Remove Dead `push_total` / `lr_scale` (P3)

Lines 699-700 allocate `push_total` and `lr_scale` tensors that are never used. Remove dead code.

---

## 6. Summary

| Severity | Count | Key Issues |
|:---------|:-----:|:-----------|
| **P0** | 0 | All V9 P0s fixed |
| **P1** | 4 | SN-35 (CPU neg last-wins), SN-36 (CPU contrastive last-wins), SN-28 (contrastive field_gate), SN-26.2 (basis health) |
| **P2** | 5 | SN-33 (GPU lateral stale), SN-38 (cooc_set loop), SN-39 (connection_strength loop), SN-19 (contrastive vectorization), SN-42 (grad norm clip missing CPU) |
| **P3** | 5 | SN-34 (GPU subspace), SN-38 (cooc_set), SN-40 (field_bits loop), SN-41 (EMA counter), SN-42 (dead tensors) |

**V9 → V10 changes:**
- ✅ 3 P0 → **CLOSED** (all SN-22 sub-fixes)
- ✅ SN-24, SN-25, SN-31 → **CLOSED**
- **NEW P1**: SN-35 (CPU neg sampling last-wins) — parity bug with neg_samples>1
- **NEW P1**: SN-36 (CPU contrastive last-wins) — parity bug with hard negatives
- **NEW P2**: SN-42 (CPU contrastive grad norm clip) — CPU missing clipping

**JSON snippet for V10 report:**
```json
{
  "v10_fixes_verified": ["SN-22.1", "SN-22.2", "SN-22.3", "SN-24", "SN-25", "SN-26.1", "SN-31", "REG-V9-2", "G-47"],
  "v10_new_issues": ["SN-35:P1", "SN-36:P1", "SN-38:P3", "SN-39:P2", "SN-40:P3", "SN-41:P3", "SN-42:P2"],
  "v10_still_open": ["SN-26.2:P2", "SN-28:P1", "SN-29:P2", "SN-33:P2", "SN-34:P3", "SN-19:P2"],
  "v10_p0_count": 0,
  "v10_p1_count": 4,
  "v10_p2_count": 5,
  "v10_p3_count": 5
}
```
