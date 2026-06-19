# Neuro-Symbolic Audit V9 — 2026-06-19

**Auditor:** Neuro-Symbolic Specialist
**Scope:** `stdp_trainer.py`, `concept_space.py`, `crystal_generator.py`
**Base:** V8 report + commit `7ae6d9a` (V8 P0/P1 fixes applied)

---

## 1. V8 Fix Verification

| V8 Claim | Code Check | Status |
|:---------|:-----------|:------:|
| SN-15 NO-OP fixed: `_apply_subspace_update()` updates code directly | ✅ `concept_space.py:556-583` does code→code with subspace LR masks | **Fixed** |
| TN-14 stale vector fixed: `v_local` copy, one `_apply_vector_update` | ✅ `stdp_trainer.py:714` `v_local = g_vecs[i].clone()`, compounding repels | **Fixed** |
| `noise_scale` passed through | ✅ `stdp_trainer.py:385` — TN-6 noise injection uses `noise_scale` | **Fixed** |

**However**, `_apply_subspace_update()` introduced a NEW bug: `self.l_c` on `ConceptSpace` (which has no `.l_c` — it lives on `self.fractal`).

---

## 2. `_apply_subspace_update()` — Regression Analysis

### 2.1 `self.l_c` AttributeError (REGRESSION from V8 fix)

**File:** `concept_space.py:564-566`
```python
mask_c = np.zeros(latent_dim, dtype=np.float32); mask_c[:self.l_c] = 1.0
mask_a = np.zeros(latent_dim, dtype=np.float32); mask_a[self.l_c:self.l_c + self.l_a] = 1.0
mask_m = np.zeros(latent_dim, dtype=np.float32); mask_m[self.l_c + self.l_a:] = 1.0
```

`self` is `ConceptSpace`. `ConceptSpace.__init__` (line 318) sets `self.fractal = FractalField(...)` but NO `self.l_c`. The attributes exist on `self.fractal` (FractalField: lines 95-99). **Must be `self.fractal.l_c`.**

Same bug in `stdp_trainer.py:150-152` (`_subspace_update` dead code): `cs.l_c` → `cs.fractal.l_c`.

**Severity: P1** — crashes with `AttributeError` when `subspace_lr is not None`. Currently dormant because `subspace_lr=None` by default.

### 2.2 Basis health not checked

`_apply_subspace_update:562` uses `self.fractal.basis` without orthogonality check. `FractalField.check_basis_health()` (line 118) exists but is never called in the subspace update path.

**Severity: P2** — drift is slow; re-orthogonalized on save/load only.

### 2.3 CPU/GPU subspace path consistency

Both `_cpu_stdp_apply:287` and `_gpu_stdp_apply:463` call `cs._apply_subspace_update(gen_cid, grad, base_lr_val, self.subspace_lr)`. Same function, shared — consistent.

---

## 3. CPU vs GPU Parity — Still Open P0/P1

All three P0 and three P1 from V8 remain UNFIXED:

### 3.1 [P0] GPU neg sampling `mean` vs CPU cumulative sum

**CPU** (`_negative_sampling_cpu:536-545`): per-negative loop, `v = v - grad * neg_lr` for each valid negative. Total push = `neg_lr × sum(grad_i)`.

**GPU** (`_negative_sampling_gpu:582`): `.mean(dim=0)` → applied once. Effective push = `neg_lr × mean(grad_i) = neg_lr / K × sum(grad_i)`.

**Gap: 1/K × weaker on GPU** when K valid negatives match.

### 3.2 [P0] GPU concept_error reweighting unconditional

**CPU** (`_negative_sampling_cpu:525`): guarded by `if field_gate:`.
**GPU** (`_negative_sampling_gpu:574-575`): always applies `neg_lr_i *= (1.0 + ce * 2.0)`. `field_gate` parameter is accepted (line 548) but **never used**.

### 3.3 [P0] Slow STDP updates silently dropped on GPU

**CPU** (`_cpu_stdp_apply:230`): processes ALL entries from `gen_updates` (both main and slow).
**GPU** (`_gpu_stdp_apply:336`): only processes `gpu_meta_l` entries. Slow updates (lines 207-211 in `_build_pairs`) are added to `gen_updates` but NOT to `gpu_meta_l` — silently dropped.

### 3.4 [P1] GPU global `elr_sum` vs per-concept `avg_elr`

**CPU** (`_negative_sampling_cpu:522`): `avg_elr = sum(elr for _, elr in updates) / max(len(updates), 1)` — per-concept average.
**GPU** (`_negative_sampling_gpu:567-570`): `elr_sum = sum(all pairs' effective_lr) / len(all pairs)` — global average.

Different when concepts have different context counts or PMI weights.

### 3.5 [P1] Momentum replaces gradient (GPU) / missing (CPU)

**GPU** (`_gpu_stdp_apply:455-456`):
```python
if mom_cpu is not None:
    grad = mom_cpu[gi]  # REPLACES grad entirely
```
Standard momentum should blend: `grad = mu × mom + (1-mu) × grad`. The stored `mom` IS correct (line 414: `mom = mu × prev + (1-mu) × avg_grad`), but the final usage replaces rather than blends.

**CPU**: no momentum at all.

---

## 4. P2/P3 Issues — Severity Reassessment After V8

### 4.1 Still P2

| Issue | V8 Severity | V9 Severity | Reason |
|:------|:-----------:|:-----------:|:-------|
| Basis health in subspace (2.1) | P2 | P2 | Unchanged — slow drift |
| CPU contrastive `pass` no-op (3.1) | P2 | P2 | Cosmetic, not a bug |
| GPU contrastive per-concept loop (3.2) | P2 | P2 | Performance issue |
| GPU lateral inh stale `_vecs_t` (7.2) | P2 | P2 | Small iterative corrections |
| `connection_strength` in contrastive loop (4.2) | P2 | P2 | Python call in loop |
| Contrastive missing `field_gate` param (3.3) | P2 | **P1** | Upgraded: diverges from STDP `field_gate` behaviour, affects training consistency |
| AM-24 cleanup not applied (6) | P2 | P2 | Dead forwarding methods still present |

### 4.2 Still P3

| Issue | V8 Severity | V9 Severity |
|:------|:-----------:|:-----------:|
| EMA counter monotonic growth (7.1) | P3 | P3 |
| `_ce_t` redundancy (7.3) | P3 | P3 |
| `_subspace_update` dead code in stdp_trainer.py (NEW) | — | P3 |
| `cooc_set` rebuilt in inner loop (NEW) | — | P3 |

### 4.3 Closed

| Issue | V8 Severity | V9 Status |
|:------|:-----------:|:---------:|
| TN-14 stale vector (4.1) | P1 | ✅ **Fixed** in V8 — `v_local.clone()` |
| SN-15 NO-OP | P0 | ✅ **Fixed** in V8 — code→code update |

---

## 5. New Issues Found in V9

### 5.1 [P3] `_subspace_update` dead code in stdp_trainer.py

`stdp_trainer.py:143-155` — method `_subspace_update()` is never called. All call sites (`_cpu_stdp_apply:287`, `_gpu_stdp_apply:463`) call `cs._apply_subspace_update()` on `ConceptSpace`. Has the same `cs.l_c` bug.

### 5.2 [P3] `cooc_set` rebuilt in inner contrastive loop

`stdp_trainer.py:696`:
```python
cooc_set = {ctx for ctx, _ in gen_updates[gen_idxs[i]]}
```
Rebuilt for every `j` iteration inside `for i in range(ng)` / `for j in range(max_hard)`. Can be hoisted to outer `i` loop.

### 5.3 [P2] Contrastive `field_gate` upgraded to P1

V8 called this P2 (consistency), but it's actually a training behaviour divergence between CPU and GPU contrastive paths when `field_gate=False`. The STDP path respects `field_gate`, but contrastive always uses field bits. This changes training dynamics.

---

## 6. SN-22+ Improvements — Updated

### SN-22: Fix CPU/GPU Negative Sampling Parity
**Status: NOT IMPLEMENTED** (from V8)
1. GPU: `grad.sum(dim=0)` instead of `.mean(dim=0)` — match CPU cumulative
2. GPU: guard concept_error with `if field_gate:` — match CPU
3. GPU: per-concept `avg_elr` grouped by concept — match CPU

### SN-23: Full GPU Vectorization of Contrastive Loop
**Status: NOT IMPLEMENTED** (from V8)
- Batch hard-negative selection with tensor operations
- Remove `for i in range(ng)` loop

### SN-24: Momentum Blend (not replacement)
**Status: NOT IMPLEMENTED** (from V8)
- GPU: `grad = mu * mom_cpu[gi] + (1-mu) * grad`
- CPU: add momentum buffer for parity

### SN-25: Slow STDP Transfer to GPU
**Status: NOT IMPLEMENTED** (from V8)
- Add slow updates to `gpu_meta_l` (extend meta tensor with slow_lr column) or process as separate scatter-add pass

### SN-26: Subspace Update Fixes
**Status: NOT IMPLEMENTED** (from V8)
1. ❗ **P1 regression**: Fix `self.l_c` → `self.fractal.l_c` in `concept_space.py:564-566` and `cs.l_c` → `cs.fractal.l_c` in `stdp_trainer.py:150-152`
2. Add optional `check_basis_health()` call in `_apply_subspace_update`
3. Move subspace update to GPU when `_basis_t` available

### SN-27: TN-14 Cross-Field Repel Cumulative
**Status: ✅ FIXED in V8** — `v_local` accumulates repels. Close.

### SN-28: Contrastive Field-Gate Parameter
**Status: NOT IMPLEMENTED** (from V8)
- **Upgraded to P1** from P2
- Add `field_gate` parameter to `_contrastive_objective`, `_contrastive_objective_cpu`, `_contrastive_objective_gpu`
- Pass through from `_train` — called at line 124, currently no `field_gate` arg

### SN-29: Clean crystal_generator.py (AM-24 completion)
**Status: NOT IMPLEMENTED** (from V8)
- Remove forwarding wrappers: `train_from_text` (line 759), `train_batch` (line 768), `evaluate` (line 779)
- Remove `_destab_field_fallback` (line 135) or inline
- Optional: remove `generate`, `_branch`, `_graph_search`

### SN-30: GPU Vec Write-Back Consolidation
**Status: NOT IMPLEMENTED** (from V8)
- Verify all GPU methods use `cs._apply_vector_update` + `_on_vector_update` hook (no duplicate write-backs) — appears consistent ✅

### SN-31: Remove/Rehabilitate Dead `_subspace_update`
**New, P3 → ✅ FIXED in V9 commit a0fe15b**
- `stdp_trainer.py:143-155` was dead code (never called). **Removed** entirely.

### SN-32: Hoist `cooc_set` in Contrastive GPU Loop
**New, P3**
- Move `cooc_set` construction out of inner `for j in range(max_hard)` loop to outer `for i in range(ng)` loop

### SN-33: GPU Lateral Inhibition — Refresh Vectors
**New, P2**
- `_lateral_inhibition_gpu:486`: `gv = gen._vecs_t[idxs].float()` is a snapshot. After `cs._apply_vector_update` modifies a vector, subsequent iterations read stale `gv` data.
- Fix: either re-read from `_vecs_t` after each update, or accept the staleness (small corrections, sparse).

### SN-34: GPU-Accelerated Subspace Update
**New, P3**
- `_apply_subspace_update` (concept_space.py:556) is CPU-only. When `_basis_t` is available on GPU, move the code_grad computation to GPU for batches.

---

## 7. Updated Summary Table

| Severity | Count | Key Issues |
|:---------|:-----:|:-----------|
| **P0** | 3 | GPU neg sampling mean/sum (3.1), GPU ce reweighting unconditional (3.2), slow STDP dropped on GPU (3.3) |
| **P1** | 4 | `self.l_c` → `self.fractal.l_c` (2.1), GPU global elr (3.4), momentum blend (3.5), contrastive `field_gate` missing (5.3) |
| **P2** | 6 | Basis health (2.2), CPU pass no-op (4.1), GPU contrastive per-concept loop (4.1), GPU lateral inh stale (4.1), connection_strength loop (4.1), AM-24 cleanup (4.1) |
| **P3** | 4 | EMA counter (4.2), `_ce_t` redundancy (4.2), dead `_subspace_update` (5.1), `cooc_set` loop (5.2) |

**V8 → V9 changes:**
- Contrastive `field_gate` escalated P2 → **P1**
- TN-14 stale vector P1 → **CLOSED** (fixed in V8)
- SN-15 NO-OP P0 → **CLOSED** (fixed in V8)
- NEW: `self.l_c` regression P1 (introduced by V8 fix itself)
- NEW: dead code `_subspace_update` P3
- NEW: `cooc_set` loop P3

**V9 fixes applied (commit a0fe15b):**
- ✅ SN-26.1: `self.l_c` → `self.fractal.l_c` (concept_space.py:564-566)
- ✅ SN-25: Slow STDP transfer to GPU (added to gpu_meta_l with _META_SLOW flag)
- ✅ SN-22.2: Guard GPU concept_error with `field_gate`
- ✅ SN-24: Momentum blend (grad = mu*mom + (1-mu)*grad)
- ✅ SN-31: Remove dead `_subspace_update`
- ✅ REG-V9-2: `code_new /= np.linalg.norm(code_new)` norm fix

**Open for next iteration:**
1. SN-22.1: GPU `grad.sum(dim=0)` — `.mean()` → `.sum()` for CPU parity
2. SN-22.3: Per-concept `avg_elr` on GPU
3. SN-28: Add `field_gate` to contrastive (P1)
4. SN-32: Hoist `cooc_set` in Contrastive GPU Loop (P3)
