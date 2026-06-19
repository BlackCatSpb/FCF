# FCF Quality & Safety Report V9 — 2026-06-19

**Agent:** Quality-Safety Agent  
**Scope:** `test_stdp.py`, `stdp_trainer.py`, `concept_space.py`, `checkpoint_manager.py`, `rng_registry.py`, `adaptive_error_tracker.py`, `train_full.py`

---

## 1. V8 QN-24..QN-31 Implementation Status

| QN | Recommended Test | Status | Notes |
|---|---|---|---|
| **QN-24** | `test_subspace_update_uniform`, `test_subspace_update_masked`, `test_subspace_update_no_basis` | **❌ NOT IMPLEMENTED** | `_apply_subspace_update()` at `concept_space.py:556` is untested |
| **QN-25** | `test_contrastive_objective_gpu_no_crash`, `test_contrastive_objective_gpu_field_aware`, `test_contrastive_objective_gpu_cross_field_penalty` | **❌ NOT IMPLEMENTED** | `_contrastive_objective_gpu()` at `stdp_trainer.py:654` is untested |
| **QN-26** | `test_gpu_stdp_noise_injection`, `test_gpu_stdp_noise_zero` | **❌ NOT IMPLEMENTED** | `noise_scale` path at `stdp_trainer.py:384-386` untested beyond syntax check |
| **QN-27** | `test_evaluate_missing_file`, `test_evaluate_too_few_lines`, `test_evaluate_basic` | **❌ NOT IMPLEMENTED** | `evaluate()` public method at `stdp_trainer.py:51-53` and `_evaluate()` at `stdp_trainer.py:791-860` untested |
| **QN-28** | `test_rng_deterministic_subseed`, `test_rng_independent_names`, `test_rng_reset_all`, `test_rng_reset_single`, `test_rng_names_property` | **❌ NOT IMPLEMENTED** | `rng_registry.py` has zero tests (QN-18 gap persists) |
| **QN-29** | `test_tracker_ema_math`, `test_tracker_fifo_eviction`, `test_tracker_move_to_end`, `test_tracker_copy`, `test_tracker_dict_interface` | **❌ NOT IMPLEMENTED** | `adaptive_error_tracker.py` has only indirect `update/get` coverage (QN-19 gap persists) |
| **QN-30** | `test_cleanup_removes_npz` (extends QN-17) | **❌ NOT IMPLEMENTED** | `_remove_tag` checks 3 extensions, tests only assert `.json` |
| **QN-31** | `test_sync_ema_backup_restore` | **❌ NOT IMPLEMENTED** | EMA path in `_gpu_stdp_apply` (line 458-461) and `_ema_vecs_t` untested |

**Verdict: 0 / 8 V8-recommended test suites implemented.**

---

## 2. New Untested Code Since V8

### 2.1 `_apply_subspace_update()` — concept_space.py:556-583 (HIGH)

The new V8 fix replaces the old `_subspace_update()` helper. This function:
- Creates 3 masks (`mask_c`, `mask_a`, `mask_m`) for subspace-LR gating
- Projects gradient into code space via `grad @ basis.T`
- Applies per-subspace learning rates
- Normalizes vector and code after update
- Calls `_after_update_hook`

Called from **both** CPU STDP (`stdp_trainer.py:287`) and GPU STDP (`stdp_trainer.py:463`). A regression here silently corrupts subspace learning without visible errors.

### 2.2 `_contrastive_objective_gpu()` — stdp_trainer.py:654-754 (HIGH)

Entire GPU contrastive path untested. Contains:
- SN-16: Field-aware contrastive decoupling (overlap-based push/skip at lines 702-712)
- TN-14: Cross-field regularization penalty (lines 713-736)
- EMA update of `v_local` with field-aware repel
- Hard-negative mining with field gating

### 2.3 `evaluate()` / `_evaluate()` — stdp_trainer.py:51-53 / 791-860 (HIGH)

Public method returning `None` silently on `FileNotFoundError` or `<5` lines. Training scripts and `train_full.py:433` call this — zero metrics could propagate without detection. Core logic: token prediction accuracy, vector accuracy, perplexity calculation.

### 2.4 `TrainingPipeline` — train_full.py:366-471 (HIGH)

New class encapsulating training loop:
- `__init__`: CheckpointManager, early stopping, patience counter
- `_checkpoint()` (line 398): checkpoint naming `{ckpt_k}k`, evaluation scheduling, self-paced learning rescoring, early stopping exit
- Uses `_quiet()` wrappers for error resilience

### 2.5 Unified Batch Training Loop — train_full.py:649-781 (MEDIUM)

V8 unified single loop:
- `_curriculum_p()` / `_curriculum_max_len()` — continuous curriculum
- `get_lr()` — cosine annealing with warm restarts
- Batch scheduling with early flush on fluctuate/decay
- `destab_scale` calculation from global_step
- `noise_scale` injected via `opt.p['noise_scale'].current`

### 2.6 Checkpoint Naming — train_full.py:409-410 (LOW)

`ckpt_k = idx // 1000; ckpt_name = f"{ckpt_k}k"` — naming convention used by `cleanup_old_checkpoints` and resume logic.

---

## 3. STR (Structural Test Reach) Updated

| Module | Total Lines | Covered Lines | STR % | Δ vs V8 |
|---|---|---|---|---|
| `test_stdp.py` | 809 | — | — | — |
| `stdp_trainer.py` | 860 | ~620 | **72%** | ↔ no change |
| `concept_space.py` (new: `_apply_subspace_update`) | 881 | ~0 new | **~0% new** | ⬇️ 5 new untested lines |
| `checkpoint_manager.py` | 117 | ~90 | **77%** | ↔ no change |
| `rng_registry.py` | 47 | 0 | **0%** | ⬜ unchanged |
| `adaptive_error_tracker.py` | 71 | ~30 | **42%** | ⬜ unchanged |
| `train_full.py` (new: TrainingPipeline + loop) | 821 | ~0 | **~0%** | ⬇️ entirely new untested code |

**Overall STR: ~52%** (decreased from ~55% due to new untested code with no compensating tests)

---

## 4. Public & Internal Method Coverage Gap (Updated)

| Method | File:Line | Covered? | Risk |
|---|---|---|---|
| `_apply_subspace_update()` | `concept_space.py:556` | **❌ NO** | HIGH |
| `_contrastive_objective_gpu()` | `stdp_trainer.py:654` | **❌ NO** | HIGH |
| `evaluate()` | `stdp_trainer.py:51` | **❌ NO** | HIGH |
| `_evaluate()` | `stdp_trainer.py:791` | **❌ NO** | HIGH |
| `TrainingPipeline._checkpoint()` | `train_full.py:398` | **❌ NO** | HIGH |
| `TrainingPipeline.__init__()` | `train_full.py:368` | **❌ NO** | HIGH |
| `get_lr()` | `train_full.py:315` | **❌ NO** | MED |
| `_curriculum_p()` / `_curriculum_max_len()` | `train_full.py:478-483` | **❌ NO** | MED |
| `_gpu_stdp_apply()` noise_scale branch | `stdp_trainer.py:384-386` | **❌ NO** (syntax only) | MED |
| `_gpu_stdp_apply()` momentum branch (SN-7) | `stdp_trainer.py:403-420` | **❌ NO** (smoke only) | MED |
| `_gpu_stdp_apply()` destab branch (SN-8) | `stdp_trainer.py:431-448` | **❌ NO** | MED |
| `_gpu_stdp_apply()` EMA branch | `stdp_trainer.py:458-461` | **❌ NO** | MED |
| `rng_registry` all methods | `rng_registry.py:23-47` | **❌ NO** | HIGH |
| `adaptive_error_tracker` dict methods | `adaptive_error_tracker.py:49-71` | **❌ NO** | MED |

---

## 5. Proposed QN-32+ Tests

### QN-32: `_apply_subspace_update` Unit Tests (replaces QN-24 for new function)
- `test_apply_subspace_update_uniform` — `subspace_lr=None` → should use `_apply_vector_update` fallback
- `test_apply_subspace_update_masked` — `subspace_lr=(0.5, 0.3, 0.1)` → verify per-subspace mask application and gradient scaling
- `test_apply_subspace_update_no_basis` — `cs.fractal.basis = None` → no-op
- `test_apply_subspace_update_nan_grad` — NaN gradient → produces valid unit vector
- `test_apply_subspace_update_twice_accumulates_shift` — two calls increase `_total_shift`

### QN-33: GPU Contrastive Objective Tests (replaces QN-25)
- `test_contrastive_objective_gpu_no_crash` — minimum smoke with fake gen_updates
- `test_contrastive_objective_gpu_hard_negative` — verify hard-negative mining selects correct candidates
- `test_contrastive_objective_gpu_cross_field_penalty` — TN-14: verify cross-field repel modifies vector direction
- `test_contrastive_objective_gpu_field_aware_skip` — SN-16: same-field high-sim entries are skipped

### QN-34: Evaluate Tests (replaces QN-27)
- `test_evaluate_missing_file` — `FileNotFoundError` → `None`
- `test_evaluate_too_few_lines` — `<5` lines → `None`
- `test_evaluate_basic` — synthetic 10-line corpus → verify output dict keys (`perplexity`, `vec_perplexity`, `accuracy_top1`, `vec_accuracy_top1`, `total_tokens`, `eval_time_s`)
- `test_evaluate_accuracy_one_perfect` — single token with ctx containing target → acc@1 = 1.0

### QN-35: TrainingPipeline Tests
- `test_pipeline_init_defaults` — verify `ckpt_mgr`, `patience`, `best_score`
- `test_pipeline_checkpoint_no_eval` — `_checkpoint` runs without evaluation if not due
- `test_pipeline_checkpoint_eval_full` — `_checkpoint` triggers full eval when `eval_every_full` condition met
- `test_pipeline_early_stopping` — `patience_counter >= patience` triggers exit

### QN-36: RNGRegistry Tests (replaces QN-28)
- `test_rng_deterministic_subseed` — same name, same master_seed → same RNG state
- `test_rng_independent_names` — different names → different sequences
- `test_rng_reset_all` — `reset_all(new_seed)` clears and changes all RNGs
- `test_rng_reset_single` — `reset(name)` removes and recreates
- `test_rng_names_property` — returns correct list after `get()` calls

### QN-37: AdaptiveErrorTracker Tests (replaces QN-29)
- `test_tracker_ema_math` — `update(0, 0.5)` then `update(0, 0.5)` → `0.9*0.5 + 0.1*0.5 = 0.5`
- `test_tracker_fifo_eviction` — insert `max_size+1` items → oldest removed (FIFO via `popitem(last=False)`)
- `test_tracker_move_to_end` — verify LRU reordering
- `test_tracker_copy` — snapshot independence from original
- `test_tracker_dict_interface` — `__getitem__`, `__setitem__`, `__contains__`, `__bool__`, `__repr__`

### QN-38: CheckpointManager NPZ Cleanup (replaces QN-30)
- `test_cleanup_removes_npz_companion` — `_remove_tag` deletes `.npz` and `.codes.npz` alongside `.json`
- `test_cleanup_removes_opt` — `_remove_tag` deletes `.opt.json` files

### QN-39: GPU STDP Path Coverage
- `test_gpu_stdp_destab` — `destab_scale > 0` triggers `_destab_field_fallback` path
- `test_gpu_stdp_noise_scale_nonzero` — `noise_scale > 0` produces different vectors from `noise_scale=0`
- `test_gpu_stdp_subspace_lr` — `subspace_lr` set triggers `_apply_subspace_update` GPU branch
- `test_gpu_stdp_ema_sync` — verify `_ema_vecs_t` updated after GPU STDP (requires mock `gen._vecs_t`)

### QN-40: Train Full Unit Tests
- `test_curriculum_p_ramp` — `_curriculum_p(0)` = 0, `_curriculum_p(N*fraction)` = 1
- `test_get_lr_warmup` — linear ramp before `lr_warmup_lines`
- `test_get_lr_cosine` — cosine annealing with warm restarts
- `test_cleanup_old_checkpoints_func` — `cleanup_old_checkpoints(keep=2)` with 5 fake files → 3 removed

---

## 6. Critical Safety Findings (Updated)

1. **`_apply_subspace_update()` untested** (concept_space.py:556) — called from both CPU and GPU STDP paths. Incorrect masking causes unbalanced subspace training without visible errors. **NEW in V8, zero coverage.**

2. **`_contrastive_objective_gpu()` untested** (stdp_trainer.py:654) — GPU contrastive contains SN-16 field-aware decoupling and TN-14 cross-field penalty. Regression silently corrupts vector space.

3. **`evaluate()` untested** (stdp_trainer.py:51) — returns `None` on `FileNotFoundError`. Training scripts in `train_full.py` could misinterpret zero metrics.

4. **TrainingPipeline untested** (train_full.py:366-471) — controls checkpointing, early stopping, evaluation scheduling, and self-paced learning rescoring. A bug here affects entire training run.

5. **RNGRegistry still at 0% coverage** — no reproducibility guarantee for named RNGs.

6. **AdaptiveErrorTracker still at 42%** — `move_to_end`, `popitem`, `copy`, `__bool__`, `__repr__`, `__setitem__`, `__getitem__`, `__contains__` untested.

7. **Checkpoint `_remove_tag` NPZ gap persists** — `.npz`/`.codes.npz` orphan files accumulate on disk.

---

## 7. Summary

| Metric | V8 | V9 | Δ |
|---|---|---|---|
| QN-24..QN-31 implemented | — | **0/8** | ⬜ no progress |
| `_apply_subspace_update` coverage | N/A (new) | **0%** | ⬇️ |
| `_contrastive_objective_gpu` coverage | 0% | **0%** | ⬜ |
| `evaluate` coverage | 0% | **0%** | ⬜ |
| TrainingPipeline coverage | N/A (new) | **0%** | ⬇️ |
| RNGRegistry coverage | 0% | **0%** | ⬜ |
| AdaptiveErrorTracker direct coverage | 0% | **0%** | ⬜ |
| STR | ~55% | **~52%** | ⬇️ -3pp |
| Total untested HIGH-risk methods | 5 | **8** | ⬆️ |

- **0 of 8 V8-recommended test suites** (QN-24..QN-31) implemented
- **3 new HIGH-risk untested areas** introduced in V8: `_apply_subspace_update`, TrainingPipeline, unified training loop
- **8 proposed V9 test suites** (QN-32..QN-40, ~30 individual tests)
- **Priority order**: QN-32 (subspace) → QN-33 (GPU contrastive) → QN-34 (evaluate) → QN-36 (RNGRegistry) → QN-35 (TrainingPipeline) → QN-37 (ErrorTracker) → QN-39 (GPU coverage) → QN-38 (NPZ cleanup) → QN-40 (train_full)
