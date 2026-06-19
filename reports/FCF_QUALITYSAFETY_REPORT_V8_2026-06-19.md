# FCF Quality & Safety Report V8 — 2026-06-19

**Agent:** Quality-Safety Agent  
**Scope:** `test_stdp.py`, `stdp_trainer.py`, `checkpoint_manager.py`, `rng_registry.py`, `adaptive_error_tracker.py`, `crystal_generator.py`

---

## 1. STR (Structural Test Reach) Coverage Assessment

| Module | Total Lines | Covered Lines | STR % | Notes |
|---|---|---|---|---|
| `test_stdp.py` | 809 | — | — | test file itself |
| `stdp_trainer.py` | 860 | ~620 | **72%** | 3 public methods uncovered |
| `checkpoint_manager.py` | 114 | ~90 | **79%** | basic error paths covered now |
| `rng_registry.py` | 47 | 0 | **0%** | ❌ no tests at all |
| `adaptive_error_tracker.py` | 71 | ~30 | **42%** | only `update`/`get` covered indirectly |
| `crystal_generator.py` (training parts) | 817 | ~500 | **61%** | `_graph_search`, `_sync_ema` uncovered |

**Overall STR: ~55%** — significant gaps remain.

---

## 2. QN-17 / QN-18 / QN-19 Status

### QN-17: CheckpointManager Error Resilience → **✅ IMPLEMENTED** (test_stdp.py:726–808)
| Test | Status |
|---|---|
| `test_save_roundtrip` | ✅ |
| `test_cleanup_removes_old` | ✅ |
| `test_shutdown_clean` | ✅ |
| `test_save_with_opt` | ✅ |
| `test_save_with_extras` | ✅ |
| `test_failure_cleanup` | ✅ |
| `test_remove_tag` | ✅ |

**Issue:** `_remove_tag` checks 3 extensions (`.json`, `.npz`, `.codes.npz`) but tests only assert `.json` files. No test verifies cleanup of `.npz`/`.codes.npz` companion files.

### QN-18: RNGRegistry Unit Tests → **❌ NOT IMPLEMENTED**
Zero tests exist for `rng_registry.py`. Methods uncovered:
- `get(name)` — deterministic sub-seeding
- `reset_all(master_seed)` — full reset
- `reset(name)` — single RNG reset
- `names` property

### QN-19: AdaptiveErrorTracker Unit Tests → **❌ NOT IMPLEMENTED**
Only `update()` / `get()` tested indirectly via `test_concept_error_fifo`. Methods uncovered:
- `move_to_end(cid)` — LRU touch
- `popitem(last=True/False)` — FIFO/LIFO eviction
- `copy()` — snapshot
- `__bool__` / `__repr__`
- `__setitem__` / `__getitem__` / `__contains__` (dict interface)
- `max_size` eviction boundary (FIFO behavior when > `max_size`)

---

## 3. Untested New Code Areas

| Area | File:Line | Risk | Description |
|---|---|---|---|
| `_subspace_update()` | `stdp_trainer.py:143-155` | **HIGH** | Subspace-LR masking logic (`lr_c`, `lr_a`, `lr_m`); called by both CPU and GPU paths |
| SN-16 Field-Aware Contrastive Decoupling | `stdp_trainer.py:635-644` (CPU), `704-741` (GPU) | **HIGH** | Overlap-based push/skip logic; GPU path completely untested |
| TN-14 Cross-Field Regularization | `stdp_trainer.py:715-741` | **HIGH** | Cross-field penalty in GPU contrastive; no test |
| TN-12 / TN-6 Gradient Noise Injection | `stdp_trainer.py:384-386` | **MEDIUM** | `noise_scale` path in `_gpu_stdp_apply`; only syntax-checked |
| SN-7 Momentum-Accumulated STDP | `stdp_trainer.py:403-420` | **MEDIUM** | `_mom_buf` caching; `test_gpu_stdp_momentum` exists but doesn't verify momentum effect |
| SN-8 Concept-Error Adaptive Destabilization (GPU) | `stdp_trainer.py:431-448` | **MEDIUM** | Destab on GPU; only CPU path tested |
| G-12 Fused scatter-add | `stdp_trainer.py:378-382` | **MEDIUM** | Core GPU optimization; no direct test |
| G-16 GPU Concept Error EMA | `stdp_trainer.py:394-401` | **MEDIUM** | `_ce_t` tensor; no test verifies EMA values |
| `_contrastive_objective_gpu()` | `stdp_trainer.py:656-754` | **HIGH** | Entire GPU contrastive path uncovered |
| `_evaluate()` / `evaluate()` | `stdp_trainer.py:791-860` | **HIGH** | Public `evaluate()` method; no test |
| `_sync_ema()` / `_restore_vectors()` | `crystal_generator.py:245-257` | **MEDIUM** | EMA vector swap; untested |
| `_graph_search()` | `crystal_generator.py:473-541` | **LOW** | BFS semantic search; error paths untested |

---

## 4. Public Method Coverage of STDPTrainer

| Public Method | Covered? | Test |
|---|---|---|
| `train_from_text()` | ✅ Indirect | `test_train_from_text_short_input` |
| `train_batch()` | ✅ Indirect | `test_train_batch_basic` |
| `evaluate()` | **❌ NO** | No test exists |

Internal methods (semi-public via `_` prefix):

| Internal Method | Covered? | Test |
|---|---|---|
| `_train()` | ✅ Indirect | via `train_from_text` / `train_batch` |
| `_subspace_update()` | **❌ NO** | — |
| `_build_pairs()` | ✅ | `test_build_pairs_basic` |
| `_cpu_stdp_apply()` | ✅ | `test_cpu_stdp_vector_update` + others |
| `_lateral_inhibition_cpu()` | ✅ (indirect) | `test_cpu_stdp_lateral_inhibition` |
| `_gpu_stdp_apply()` | ✅ (syntax) | `test_gpu_stdp_apply_no_crash` |
| `_lateral_inhibition_gpu()` | ✅ (smoke) | `test_lateral_inhibition_gpu_smoke` |
| `_negative_sampling_cpu()` | ✅ | `test_negative_sampling_cpu` + `test_negative_sampling_cpu_divergence` |
| `_negative_sampling_gpu()` | ✅ (smoke) | `test_negative_sampling_gpu_no_crash` |
| `_contrastive_objective()` | ✅ (CPU only) | `test_contrastive_objective` + `test_contrastive_objective_cpu_runs` |
| `_contrastive_objective_cpu()` | ✅ | `test_contrastive_objective_cpu_runs` |
| `_contrastive_objective_gpu()` | **❌ NO** | — |
| `_centroid_pull_batch()` | ✅ | `test_centroid_pull_batch` |
| `_evaluate()` | **❌ NO** | — |

---

## 5. Proposed QN-24+ Tests

### QN-24: STDPTrainer Subspace Update
- `test_subspace_update_uniform` — `subspace_lr=None` → simple gradient addition
- `test_subspace_update_masked` — `subspace_lr=(0.5, 0.3, 0.1)` → verify mask dimensions and gradient scaling
- `test_subspace_update_no_basis` — `cs.fractal.basis = None` → falls back to uniform

### QN-25: GPU Contrastive Objective
- `test_contrastive_objective_gpu_no_crash` — minimum smoke
- `test_contrastive_objective_gpu_field_aware` — verify SN-16 overlap logic
- `test_contrastive_objective_gpu_cross_field_penalty` — verify TN-14 regularization (cross-field repel)

### QN-26: Gradient Noise Injection (TN-12/TN-6)
- `test_gpu_stdp_noise_injection` — `noise_scale > 0` produces different vectors
- `test_gpu_stdp_noise_zero` — `noise_scale=0` deterministic

### QN-27: Evaluate Public Method
- `test_evaluate_missing_file` — `FileNotFoundError` → returns `None`
- `test_evaluate_too_few_lines` — `<5` lines → returns `None`
- `test_evaluate_basic` — with synthetic corpus, check output dict keys

### QN-28: RNGRegistry (QN-18 completion)
- `test_rng_deterministic_subseed` — same name, same seed → same RNG
- `test_rng_independent_names` — different names → different sequences
- `test_rng_reset_all` — `reset_all(new_seed)` changes all RNGs
- `test_rng_reset_single` — `reset(name)` removes and recreates
- `test_rng_names_property` — returns correct list

### QN-29: AdaptiveErrorTracker (QN-19 completion)
- `test_tracker_ema_math` — verify decay: `new = decay * old + (1-decay) * error`
- `test_tracker_fifo_eviction` — insert > `max_size` items → oldest removed
- `test_tracker_move_to_end` — LRU reordering
- `test_tracker_copy` — snapshot independence
- `test_tracker_dict_interface` — `__getitem__`, `__setitem__`, `__contains__`, `__bool__`

### QN-30: CheckpointManager Extension Cleanup
- `test_cleanup_removes_npz` — verify `.npz` and `.codes.npz` files are removed

### QN-31: EMA Sync / Restore
- `test_sync_ema_backup_restore` — `_sync_ema()` saves, `_restore_vectors()` restores

---

## 6. Critical Safety Findings

1. **`_contrastive_objective_gpu()` untested** — GPU contrastive contains SN-16 field-aware decoupling and TN-14 cross-field penalty. A regression here would silently corrupt vector space during GPU training.

2. **`_subspace_update()` untested** — called by both CPU and GPU STDP paths. Incorrect masking would cause unbalanced subspace training without visible errors.

3. **`evaluate()` public method untested** — returns `None` silently on `FileNotFoundError`. Training scripts could misinterpret zero metrics.

4. **RNGRegistry untested** — no reproducibility guarantee for named RNGs across versions.

5. **Checkpoint `_remove_tag` extension coverage gap** — `.npz` / `.codes.npz` orphan files may accumulate on disk.

---

## 7. Summary

- **QN-17** fully implemented (7 tests). Minor gap: `.npz`/`.codes.npz` cleanup not verified.
- **QN-18, QN-19** completely missing — **highest priority**.
- **STDPTrainer public method `evaluate()`** uncovered.
## V8 Commit Fixes
- **SN-15**: `_subspace_update` replaced with `_apply_subspace_update` (direct code update)
- **TN-14**: batch-stale vector fixed (local `v_local`, single `_apply_vector_update`)
- **`_subspace_update()`, `_contrastive_objective_gpu()`** are high-risk untested code.
- **3 new QN-series test suites proposed** (QN-24 through QN-31, ~25 tests).
- **Overall STR ≈ 55%** with critical gaps in GPU training paths.
