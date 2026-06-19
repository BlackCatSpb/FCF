# Neuro-Symbolic Audit V8 — 2026-06-19

**Auditor:** Neuro-Symbolic Specialist  
**Scope:** `stdp_trainer.py`, `crystal_generator.py`, `concept_space.py`, `syntax_lattice.py`  
**Base:** V7 report + uncommitted V8 changes in FCF repo (`master`, ahead 5 commits)

---

## 1. SN-9..SN-21 Status from V7

| ID | Title | Status | Location |
|---|---|---|---|
| SN-9 | GPU Layout Optimizations | ✅ V8 | `_gpu_stdp_apply` pass-through |
| SN-10 | Inline FP16 vecs_t | ✅ V6 | `_vecs_t` dtype=float16 |
| SN-11 | Pre-allocated GPU buffers | ✅ V6 | `torch.empty(V, D)` realloc only on shape/device mismatch |
| SN-12 | Fused scatter-add (G-12) | ✅ V8 | `stdp_trainer.py:377-379` |
| SN-13 | Concept Error EMA on GPU (G-16) | ✅ V8 | `stdp_trainer.py:395-401` |
| SN-14 | Gradient Noise Injection (TN-6) | ✅ V8 | `stdp_trainer.py:385-386` |
| SN-15 | Subspace-Kinetic STDP | ✅ V8 | `stdp_trainer.py:143-155` `_subspace_update()` |
| SN-16 | Field-Aware Contrastive Decoupling | ✅ V8 | CPU: `_contrastive_objective_cpu:635-654`; GPU: `_contrastive_objective_gpu:704-741` |
| SN-17 | (z_c, z_a, z_m field-bias metrics) | ❌ Not found | Not implemented |
| SN-18 | EMA sync for stable eval | ✅ V8 | `crystal_generator.py:245-257` `_sync_ema` / `_restore_vectors` |
| SN-19 | Vectorized GPU Contrastive | ✅ V8 | `_contrastive_objective_gpu:656-754` — batched sim, topk |
| SN-20 | (multi-GPU split) | ❌ Not found | Not implemented |
| SN-21 | (online basis adaptation) | ❌ Not found | Not implemented |

**Sub-items from V7:**

| ID | Title | Status | Location |
|---|---|---|---|
| SN-7 | Momentum-Accumulated STDP | ✅ GPU, ❌ CPU | GPU: `stdp_trainer.py:403-420`; CPU: none |
| SN-8 | Concept-Error Adaptive Destab | ✅ | GPU: `stdp_trainer.py:431-448` |
| T-B3/SN-B1 | EMA before `_apply_vector_update` | ✅ FIXED V8 | `stdp_trainer.py:466-471` |
| SN-B2 | concept_error reweighting (field_gate) | ⚠️ Partial | CPU: ✅ line 526-529; GPU: unconditional — **bug** |
| N-4 | Per-concept neg_lr_i + grad fix | ✅ V8 | GPU: `stdp_trainer.py:574-595` |
| TN-14 | Field-Aware Contrastive Regularization | ✅ V8 | `_contrastive_objective_gpu:716-741` |

---

## 2. Subspace-Kinetic STDP (SN-15) — Issues

### 2.1 `_subspace_update` — no basis health check
`stdp_trainer.py:143-155` uses `cs.fractal.basis` directly without validating orthogonality. If the basis has drifted (e.g. after repeated `fluctuate` calls), the `code_grad @ basis.T` / `(code_grad @ basis)` round-trip is inexact.

**Severity:** P2 — drift is slow (re-orthogonalized only on save/load).

### 2.2 Subspace masks use hardcoded latent_dim
```python
mask_c = np.zeros(latent_dim, dtype=np.float32); mask_c[:cs.l_c] = 1.0
mask_a = np.zeros(latent_dim, dtype=np.float32); mask_a[cs.l_c:cs.l_c + cs.l_a] = 1.0
```
Uses `cs.l_c`, `cs.l_a`, `cs.l_m` — but `cs` is `ConceptSpace`, not `FractalField`. `ConceptSpace` does not have `.l_c` / `.l_a` / `.l_m` attributes directly. The correct reference is `cs.fractal.l_c`. **This is a latent runtime bug** — `AttributeError` if `l_c` is accessed on `ConceptSpace`.

**Severity:** P1 — crashes when `subspace_lr` is set (currently default `None`, so not triggered).

### 2.3 CPU vs GPU subspace path consistency
Both CPU (`_cpu_stdp_apply:286`) and GPU (`_gpu_stdp_apply:458`) call `_subspace_update(grad, v, base_lr_val)`. Same function, shared — consistent.

---

## 3. Field-Aware Contrastive (SN-16) — Issues

### 3.1 CPU contrastive: `pass` for cross-field is a no-op
`stdp_trainer.py:641-644`:
```python
if overlap == 0 and cos_val > 0.05:
    pass  # aggressive push for cross-field
elif overlap > 0 and cos_val > 0.3:
    continue
contr_grad = cos_val * v_neg - v_gen
```
The `pass` has no effect. Both cross-field and same-field (cos ≤ 0.3) cases use the same gradient formula. There is no "aggressive push" — only a filter that skips same-field samples above 0.3.

**Severity:** P2 — intended behavior poorly expressed; no actual bug.

### 3.2 GPU contrastive: per-concept Python loop (pseudo-vectorized)
Despite SN-19 claiming "fully vectorized", `_contrastive_objective_gpu:692` still has `for i in range(ng)`. The top-k and masking are vectorized, but the hard-negative selection, TN-14 repel loop, and update application are per-concept.

**Severity:** P2 — performance gap, not correctness.

### 3.3 CPU contrastive: no `field_gate` parameter
`_contrastive_objective` is called unconditionally from `_train` (`stdp_trainer.py:124`) without passing `field_gate`. The CPU path always uses field_bits if available; the GPU path also always uses them. There is no way to disable field-aware filtering in contrastive objective, unlike STDP which has `field_gate` flag.

**Severity:** P2 — consistency with STDP `field_gate` control.

---

## 4. Vectorized GPU Contrastive (SN-19) — Issues

### 4.1 TN-14 stale `g_vecs[i]` after repel updates
`stdp_trainer.py:732-741`:
```python
v2 = g_vecs[i] + rep_grad * contr_lrs[i] * reg_lam
...
cs._apply_vector_update(gen_idxs[i], v2.cpu().numpy())
```
After the first cross-field repel, `g_vecs[i]` still holds the old vector. Subsequent repels for the same `i` use the stale starting point. Each repel within `j` loop is relative to the original `g_vecs[i]`, not compounding.

**Severity:** P1 — repels are independent rather than cumulative; may reduce regularization effectiveness.

### 4.2 GPU contrastive `connection_strength` Python call inside loop
`stdp_trainer.py:701` calls `gen.lattice.connection_strength(gen_idxs[i], neg_cid)` for each candidate — a Python-level call, breaking vectorization for this filter.

**Severity:** P2 — negligible for small batches, O(ng * max_hard).

---

## 5. CPU/GPU Consistency After V8 Fixes

### 5.1 [P0] GPU Neg Sampling: `mean` vs `sum` gradient discrepancy
- **CPU** (`_negative_sampling_cpu:539`): `grad = v_neg - sim * v_gen` — per negative, applied inside loop. Effective total push = `neg_lr * sum(grad_i)` over all valid negatives.
- **GPU** (`_negative_sampling_gpu:584`): `grad = mean(v_neg_i - sim_i * v_gen)` — then applied once. Effective push = `neg_lr * mean(grad) = neg_lr / n * sum(grad_i)`.

The GPU update is **1/n_neg × weaker** than cumulative CPU update when multiple negatives match. If typically 1-2 negatives match, discrepancy is 2×.

**Severity: P0** — different training dynamics between CPU and GPU.

### 5.2 [P0] SN-B2: GPU applies concept_error unconditionally
- **CPU** (`_negative_sampling_cpu:526-529`): concept_error reweighting **only when `field_gate=True`**.
- **GPU** (`_negative_sampling_gpu:575-577`): concept_error reweighting **always applied** (no `field_gate` check).

**Severity: P0** — GPU path diverges from CPU when `field_gate=False`.

### 5.3 [P0] GPU path silently drops slow STDP updates
`_build_pairs` adds **both** main updates (line 205) and slow `theta_slow` updates (lines 207-211) to `gen_updates`. But only the main pairs go into `gpu_meta_l` / `gpu_ctx_l` / `gpu_tgt_l` (lines 213-222). The slow updates are never transferred to GPU lists.

In CPU path, `_cpu_stdp_apply` processes all entries from `gen_updates` (both main and slow). In GPU path, `_gpu_stdp_apply` only processes `gpu_*` lists — slow updates are **silently dropped**.

**Severity: P0** — CPU and GPU produce different STDP updates.

### 5.4 [P1] GPU Neg Sampling: global elr vs per-concept elr
- **CPU**: `avg_elr = sum(elr for _, elr in updates) / len(updates)` — per-concept average.
- **GPU**: `elr_sum = sum(all pairs' effective_lr) / len(all pairs)` — global average from all training pairs.

When different concepts have different context counts or PMI weights, the per-concept LR differs between CPU and GPU.

**Severity: P1** — moderate divergence.

### 5.5 [P1] SN-7 Momentum: GPU replaces gradient, CPU missing
GPU (`_gpu_stdp_apply:456-457`):
```python
if mom_cpu is not None:
    grad = mom_cpu[gi]  # replaces grad entirely, not blended
```

Standard momentum should blend: `grad = mom * prev_momentum + (1-mom) * current_grad`. Current code stores `mom = mu * prev + (1-mu) * avg_grad` (correct), but then **uses `mom` as the final gradient** rather than blending it with `acc / elr`.

CPU path has **no momentum at all**.

**Severity: P1** — momentum behaves as a gradient smoother, not as acceleration; CPU/GPU asymmetry.

---

## 6. crystal_generator.py Cleanup Status (AM-24)

| Item | V8 Description (aspired) | Actual Status |
|---|---|---|
| Remove dead forwarding STDP methods | Deleted | **STILL PRESENT** `train_from_text`, `train_batch`, `evaluate` forwarding wrappers |
| Remove `_destab_field_fallback()` | Deleted | **STILL PRESENT** `crystal_generator.py:135` |
| Remove generation methods | Deleted | **ALL PRESENT** `generate:331`, `_branch:545`, `_graph_search:473` |
| Eager STDPTrainer creation | Applied | ✅ `self._trainer = STDPTrainer(self)` (was `None`) |
| Enable `momentum_mu=0.9` default | Applied | ✅ Signatures now default `momentum_mu=0.9` |
| `momentum_mu`, `noise_scale` passthrough | Applied | ✅ Forwarded through delegation chain |

**Observation:** The crystal_generator.py cleanup described in V8 has NOT been executed. The methods are still alive. If removal is intended, they must be explicitly deleted.

---

## 7. Minor Issues

### 7.1 EMA step counter monotonic growth
`stdp_trainer.py:469`: `gen._ema_steps += 1` increments on every GPU STDP update (per concept). After 146K concepts × many batches, this counter can overflow Python `int` memory (unlikely) or become meaningless. No reset on `_build_torch_tensors`.

**Severity:** P3 — cosmetic.

### 7.2 GPU lateral inhibition stale `_vecs_t`
`stdp_trainer.py:497-508`: `_lateral_inhibition_gpu` reads `cs.concept_vectors.get(gen_cids[gi])` and applies update via `cs._apply_vector_update`. The `_on_vector_update` hook keeps `_vecs_t` in sync. But the inhibition loop reads `gv = gen._vecs_t[idxs]` eagerly at line 488 — if previous iterations modified the vectors, `gv` is stale for subsequent iterations.

**Severity:** P2 — minor; inhibition is sparse and iterative corrections are small.

### 7.3 `concept_error.get` returns float; `ce_t` tensor may desync
CPU path (`_cpu_stdp_apply:258`): `gen.concept_error.update(gen_cid, err)` — updates the Python tracker.  
GPU path (`_gpu_stdp_apply:398-401`): updates both `_ce_t` tensor AND Python tracker.  
But `_build_torch_tensors` re-initializes `_ce_t` from scratch (lines 218-223), discarding GPU-accumulated values from `_ce_t`. The `_ce_t` tensor is rebuilt from `gen.concept_error.items()`. So the Python tracker is source-of-truth — consistent but redundant.

**Severity:** P3 — redundant, no functional issue.

---

## 8. SN-22+ Improvement Proposals

### SN-22: Fix CPU/GPU Negative Sampling Parity
1. Make GPU use cumulative (not mean) gradient: `grad.sum(dim=0)` instead of `.mean(dim=0)`, to match CPU per-negative updates.
2. Guard GPU concept_error reweighting with `field_gate` to match CPU.
3. Use per-concept `avg_elr` on GPU (grouped by concept before neg LR computation).

### SN-23: Full GPU Vectorization of Contrastive Loop
Eliminate `for i in range(ng)` in `_contrastive_objective_gpu`. Batch the hard-negative selection using tensor operations:
- Pre-compute co-occurrence mask tensor
- Pre-compute connection_strength mask tensor  
- Use top-k masking to select hard negatives in one shot
- Single batch `push_total` scatter for all concepts

### SN-24: Momentum Blend (not replacement)
Change GPU momentum from `grad = mom_cpu[gi]` to:
```python
grad = momentum_mu * mom_cpu[gi] + (1 - momentum_mu) * grad
```
And add momentum buffer to CPU path for parity.

### SN-25: Slow STDP Transfer to GPU
Add `slow_lr` entries to `gpu_meta_l` (extend meta with an extra column) or process slow updates as a separate GPU scatter pass. This fixes the P0 CPU/GPU divergence.

### SN-26: Subspace Update Fixes
1. Fix `cs.l_c` → `cs.fractal.l_c` in `_subspace_update`.
2. Add optional basis re-orthogonalization check inside `_subspace_update` (or at least a warning).
3. Consider moving subspace update to GPU when `_basis_t` is available.

### SN-27: TN-14 Cross-Field Repel Cumulative Fix
Accumulate `rep_grad` across all `j` before applying, or update `g_vecs[i]` in-place after each repel by re-reading from `_vecs_t`:
```python
g_vecs_i = gen._vecs_t[gen_idxs[i]].float()  # refreshed
```

### SN-28: Contrastive Field-Gate Parameter
Add `field_gate` parameter to `_contrastive_objective` and both CPU/GPU variants. Pass through from `_train` to match STDP behavior.

### SN-29: Clean crystal_generator.py (AM-24 completion)
1. Remove forwarding methods (`train_from_text`, `train_batch`, `evaluate`) — call `_trainer` directly.
2. Remove `_destab_field_fallback` — inline or remove entirely (only used in GPU destab path as fallback).
3. Optional: remove `generate`, `_branch`, `_graph_search` if generation is fully delegated (this is a bigger refactor).

### SN-30: GPU Vec Write-Back Consolidation
The `_on_vector_update` hook (called from `_apply_vector_update`) already pushes `v_new` to `_vecs_t`. The old pattern of explicit `gen._vecs_t[gen_cid] = torch.from_numpy(v_new)...` was correctly removed from `_gpu_stdp_apply`, `_lateral_inhibition_gpu`, and `_negative_sampling_gpu`. Verify all remaining GPU methods use `cs._apply_vector_update` + `_on_vector_update` (no duplicate write-backs).

---

## 9. Summary Table

| Severity | Count | Key Issues |
|---|---|---|
| **P0** | 3 | GPU neg sampling mean/sum (5.1), GPU ce reweighting unconditional (5.2), slow STDP dropped on GPU (5.3) |
| **P1** | 5 | `cs.l_c` → `cs.fractal.l_c` (2.2), GPU global elr (5.4), momentum blend (5.5), TN-14 stale vec (4.1), contrastive CPU field_gate (3.3) |
| **P2** | 5 | Basis health (2.1), CPU pass no-op (3.1), GPU contrastive per-concept loop (3.2), GPU lateral inh stale (7.2), connection_strength in loop (4.2) |
| **P3** | 3 | EMA counter (7.1), ce_t redundancy (7.3), AM-24 not applied (6) |

## V8 Commit Fixes

| Проблема | Статус |
|:---------|:------:|
| **SN-15 NO-OP** | ✅ Исправлен — `_apply_subspace_update()` обновляет code напрямую |
| **TN-14 stale vector** | ✅ Исправлен — `v_local` копия, один `_apply_vector_update` |
| **noise_scale** | ✅ Исправлен — передаётся в `train_batch()` |

**Overall:** SN-15/16/19 structurally correct. Three P0 CPU/GPU consistency bugs remain from V8 changes. Recommend SN-22/25/27 fixes before further optimization.
