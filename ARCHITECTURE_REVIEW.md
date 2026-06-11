# Architecture Review: FCF (Fractal Concept Field)

> **Date:** 2026-06-11
> **Scope:** Full codebase audit of FCF `eva/symbolic/`, `train_full.py`, and cross-reference with EVA-Ai knowledge/contradiction modules
> **Lines analyzed:** ~6,500 across 14 source files

---

## 1. Overview

FCF is a neuro-symbolic language learning system that replaces gradient descent with biologically-inspired local plasticity rules (STDP, lateral inhibition, hormonal modulation). The core architecture comprises:

| Component | File | Purpose |
|---|---|---|
| `FractalField` | `concept_space.py:18` | Shared orthogonal basis matrix (512x384) mapping latent codes to concept vectors |
| `ConceptSpace` | `concept_space.py:188` | Vocabulary-level vector container, STDP, PQ compression, homeostasis |
| `SyntaxLattice` | `syntax_lattice.py:52` | N-gram language model + connection strength graph + PPMI |
| `CrystalGenerator` | `crystal_generator.py:27` | Beam search generator using BMSSP-EVA graph BFS + RRF scoring |
| `HormonalSystem` | `hormonal_system.py:17` | DA/5HT/NA/ACh neuromodulation of learning parameters |
| `ParameterOptimizer` | `parameter_optimizer.py:92` | Rule-based hyperparameter auto-tuning with feasibility corridors |
| `VectorHealth` | `vector_health.py:1` | Read-only diagnostics (antonym collapse, clusters, near-duplicates) |
| `train_full.py` | `train_full.py:1` | Main training loop with checkpointing, eval, 3D visualization |

---

## 2. Component-by-Component Analysis

### 2.1 ConceptSpace + FractalField

#### How fractal STDP works (full path)

1. **BPE token → concept vector:** Each BPE token ID (0..vocab_size-1) gets a latent code `coords ∈ R^{512}`. The vector is `v = normalize(coords @ basis)` where `basis ∈ R^{512×384}` is a fixed random orthogonal matrix.
2. **STDP update:** For a context→target pair `(prev_cid, gen_cid)`, the Riemannian gradient is computed:
   - If correct prediction: `shift = (v_ctx - sim * v_gen) * lr` (attract)
   - If wrong: correction toward expected + `shift * -0.05*lr` (repel)
3. **Code projection:** `delta_v` is projected back: `delta_code = delta_v @ basis.T`, then `code += delta_code`, rescale so `|code @ basis| = 1`.
4. **Lateral inhibition:** Up to 200 random similar concepts are pushed away from the winner using the negative Riemannian gradient.

#### Key architectural decisions

- **Batch vs sequential training:** `train_from_text` (crystal_generator.py:443) batches all STDP updates for the same `gen_cid` within a sentence — all context→target pairs are summed into a single gradient before code projection. This is an O(lr²) approximation vs sequential online updates.
- **Generation:** Beam search with BMSSP-EVA multi-source BFS over the PPMI-weighted connection graph, fused with n-gram predictions and vector similarity via RRF scoring.
- **Evaluation:** Full softmax over vocabulary (32K) weighted by: 0.5×PMI + 0.25×ngram_prob + 0.15×vec_sim + 0.02×freq_prior.

---

### 2.2 CrystalGenerator

#### Scoring components in `_branch()`

| Signal | Weight | Source |
|---|---|---|
| Graph candidates (BMSSP-EVA) | 0.70 | `_graph_search` |
| N-gram syntax predictions | 0.15/(K+rank) | `lattice.predict()` |
| Vector cosine similarity | 0.15/(K+1) | `topk_similar_concepts` |
| Frequency prior | 0.02/(K+1) | `concept_freq` |
| Homeostatic boost | up to ±30% | `homeostatic_boost` |
| Intent centroid bonus | up to +30% | query centroid proximity |

---

### 2.3 SyntaxLattice

Stores n-grams (2/3/4), co-occurrence connection graph, skip-2 counts, and PPMI cache. The connection graph uses `min(cid_a, cid_b), max(cid_a, cid_b)` as canonical key, with bidirectional index for O(1) lookup. PPMI is lazily computed on first `use_ppmi=True` call.

### 2.4 HormonalSystem

Four hormones updated per generation step:
- **DA (dopamine):** extrinsic reward + curiosity + mastery + coherence
- **5HT (serotonin):** inversely proportional to match rate
- **NA (noradrenaline):** driven by surprise and (1-confidence)
- **ACh (acetylcholine):** plasticity gate via novelty

All modulate `modulate_stdp_lr`, `modulate_temperature`, `modulate_beam_width`, `modulate_homeostasis`.

### 2.5 ParameterOptimizer

Rule-based optimizer with 11 parameters, each with a `[min, max]` feasibility corridor. Metrics (mean_cos, std_cos, vec_ppl, acc1, vacc1) drive adjustments:
- High mean_cos → repel_strength up
- Low std_cos vs target → noise up
- cos_trend → LR down/up
- vec_ppl plateau → LR up + widen context_window
- vacc1 stuck → enable negative sampling

### 2.6 VectorHealth (new)

Three read-only diagnostics:
- `check_antonym_collapse`: cosine of known antonym pairs
- `detect_concept_clusters`: MiniBatchKMeans + gap detection
- `prune_near_duplicates`: upper-triangular full similarity matrix

### 2.7 Training Loop (train_full.py)

Line-by-line STDP with:
- LR warmup (first 1000 lines)
- Checkpoints every 500 lines (numbered + atomic writes)
- Full evaluation every 1000-2000 lines
- Fluctuation + lattice decay every 2000-3000 lines
- 3D PCA visualization at checkpoints
- KeyboardInterrupt handler for graceful save

---

## 3. Critical Issues

### CRITICAL-1: `FractalField.fluctuate()` creates a new RandomState on every call

**File:** `concept_space.py:117-118`
```python
def fluctuate(self, noise_scale=0.005, decay=0.999):
    rng = np.random.RandomState(42 + self._fluctuation_step)
    self._fluctuation_step += 1
```

Every call to `fluctuate()` creates and discards a `numpy.random.RandomState` instance. For 32K concepts, the loop at line 119 generates 32K random values per call. This is called every `FLUCTUATE_EVERY=2000` lines. Over 145K lines: ~72 fluctuation calls × 32K concepts × 512D = 1.18 billion random samples — each from a freshly seeded RNG.

**Problem:** Creating a new RandomState per call is ~50x slower than using a persistent RNG. More critically, the seed `42 + step` uses a weak linear progression. The correct Ornstein-Uhlenbeck process should use a **persistent** RNG and properly correlated noise.

**Fix:**
```python
def __init__(self, ...):
    ...
    self._fluct_rng = np.random.RandomState(42)

def fluctuate(self, noise_scale=0.005, decay=0.999):
    for cid in list(self.codes.keys()):
        c = self.codes[cid]
        noise = self._fluct_rng.randn(self.latent_dim).astype(np.float32) * noise_scale
        c[:] = c * decay + noise
    self._matrix_dirty = True
```

---

### CRITICAL-2: `_lateral_inhibition_fractal` has sampling bias and state-drift bugs

**File:** `concept_space.py:433-438`
```python
rng = np.random.RandomState(winner_cid + self._inhibition_step)
self._inhibition_step += 1
cids = self._all_cids
n_cids = len(cids)
perm = rng.permutation(n_cids)
sampled_cids = [cids[i] for i in perm[:min(sample_size, n_cids)] if cids[i] != winner_cid][:sample_size]
```

**Bug 1:** `self._inhibition_step` is incremented **before** the early-return check at line 439 (`if not sampled_cids: return`). If the function returns early (no candidates to inhibit), the step counter still advances, desynchronizing the RNG for subsequent calls.

**Bug 2:** The list comprehension filters out `winner_cid` **after** slicing with `[:min(sample_size, n_cids)]`. If the winner_cid appears within the first `sample_size` items of the permutation, the result has at most `sample_size-1` items. The final `[:sample_size]` slice then caps it, but the total filtered set may be systematically smaller than requested.

**Bug 3:** The RNG seed `winner_cid + self._inhibition_step` means the same `winner_cid` called at different inhibition steps gets different samples (correct), but the state of `self._inhibition_step` is global — if another winner_cid call increments it in between, the sampling is non-deterministically interleaved.

**Fix:**
```python
def _lateral_inhibition_fractal(self, winner_cid, ...):
    v_win = self.concept_vectors.get(winner_cid)
    if v_win is None:
        return
    vw_n = v_win / max(np.linalg.norm(v_win), 1e-10)

    n_total = len(self.concept_vectors)
    if sample_size is None:
        sample_size = min(200, n_total)

    cids = self._all_cids
    n_cids = len(cids)
    if n_cids <= 1:
        return

    # Use a persistent RNG with per-winner deterministic offset
    if not hasattr(self, '_inhibit_rng'):
        self._inhibit_rng = np.random.RandomState(42)
    perm = self._inhibit_rng.permutation(n_cids)

    sampled_cids = []
    for i in perm:
        if cids[i] != winner_cid:
            sampled_cids.append(cids[i])
            if len(sampled_cids) >= sample_size:
                break

    if not sampled_cids:
        return

    # ... rest of function ...
```

---

### CRITICAL-3: `normalize_vectors()` modifies `concept_vectors` but NOT `fractal.codes`, causing permanent inconsistency

**File:** `concept_space.py:834-866`
```python
def normalize_vectors(self):
    """Center and normalize all concept vectors onto the unit sphere."""
    ...
    centered = vecs - centroid
    norms = np.linalg.norm(centered, axis=1)
    centered /= norms[:, np.newaxis]
    for i, cid in enumerate(self.concept_vectors):
        self.concept_vectors[cid] = centered[i]
```

This method subtracts the global centroid and renormalizes all vectors, but **never propagates the changes back to `self.fractal.codes`**. After calling `normalize_vectors()`, the invariant `normalize(code @ basis) == concept_vector[cid]` is broken. Subsequent STDP updates via `_apply_vector_update` will start from the stale `code` and produce incorrect results.

This is called from the README-documented workflow but is catastrophically dangerous in the current state.

**Fix:** After renormalizing, recompute all fractal codes:
```python
def normalize_vectors(self):
    ...
    for i, cid in enumerate(self.concept_vectors):
        self.concept_vectors[cid] = centered[i]

    # Rebuild fractal codes to maintain consistency
    for cid, v in self.concept_vectors.items():
        code_new = v @ self.fractal.basis.T
        self.fractal.codes[cid] = code_new
    self.fractal._matrix_dirty = True
    ...
```

---

### CRITICAL-4: `expand_dim()` silently corrupts fractal codes during dimension expansion

**File:** `concept_space.py:769-831`
```python
def expand_dim(self, target_dim):
    ...
    # Update fractal basis instead of creating new field
    old_latent_dim = self.fractal.latent_dim
    rng = np.random.RandomState(42)
    mat = rng.randn(old_latent_dim, new_dim).astype(np.float32)
    Q, _ = np.linalg.qr(mat, mode='reduced')
    self.fractal.basis = Q.astype(np.float32)

    # Project learned vectors back into new codes
    codes = {}
    for cid, v in self.concept_vectors.items():
        code_new = v @ self.fractal.basis.T
        codes[cid] = code_new
    self.fractal.codes = codes
```

The old basis `(latent_dim, old_dim)` is replaced with a completely new random orthogonal matrix `(latent_dim, new_dim)`. The new codes are computed as `v_new @ basis_new.T`. But `v_new` was projected from the old space using `v @ proj` where `proj = q[:old_dim, :new_dim]` from a different QR decomposition. The new codes are then immediately used in `_sync_concept_vectors_from_fractal()` which recomputes `v = normalize(code_new @ basis_new)`. This second projection introduces errors because `basis_new` is a completely different orthogonal frame — the latent representation is destroyed.

**Correct approach:** Use a proper Schur complement / Nyström extension to expand the basis while preserving existing structure:
```python
def expand_dim(self, target_dim):
    old_dim = self.dim
    if target_dim <= old_dim:
        return
    
    # Extend existing basis with orthogonal new columns
    new_cols = target_dim - old_dim
    rng = np.random.RandomState(42)
    # Generate random vectors in the null space of current basis
    extension = rng.randn(self.fractal.latent_dim, new_cols).astype(np.float32)
    # Orthogonalize against existing basis
    extension = extension - self.fractal.basis @ (self.fractal.basis.T @ extension)
    Q_ext, _ = np.linalg.qr(extension, mode='reduced')
    old_basis = self.fractal.basis
    self.fractal.basis = np.concatenate([old_basis, Q_ext], axis=1)
    # Existing codes remain valid — just append zero coefficients for new dims
    ...
```

---

### CRITICAL-5: `_apply_vector_update` has silent failure when fractal code is missing

**File:** `concept_space.py:303-356`
```python
def _apply_vector_update(self, cid, v_new, max_shift=0.5):
    v_old = self.concept_vectors.get(cid)
    self.concept_vectors[cid] = v_new

    code = self.fractal.codes.get(cid)
    if code is None or v_old is None:
        return  # <-- SILENT FAILURE
```

If `code is None` (concept not in fractal field) or `v_old is None` (first update to this concept), the method returns early **after already setting `self.concept_vectors[cid] = v_new`**. The concept vector is updated but the fractal code is not, breaking consistency. Subsequent calls will find `v_old` but still have stale code.

This can happen when `init_concepts()` creates a vector via the fallback path (line 236-238) — the vector exists in `concept_vectors` but the code may not be in `fractal.codes`.

**Fix:**
```python
def _apply_vector_update(self, cid, v_new, max_shift=0.5):
    v_old = self.concept_vectors.get(cid)
    code = self.fractal.codes.get(cid)
    
    if code is None:
        # Initialize code from vector
        if v_old is not None:
            init_code = v_new @ self.fractal.basis.T
            self.fractal.codes[cid] = init_code
        self.concept_vectors[cid] = v_new
        return
    
    if v_old is None:
        self.concept_vectors[cid] = v_new
        return

    self.concept_vectors[cid] = v_new
    delta_v = v_new - v_old
    ...
```

---

## 4. High Severity Issues

### HIGH-1: RNG seed `42 + self._fluctuation_step` overflows Python integer space

**File:** `concept_space.py:117`
```python
rng = np.random.RandomState(42 + self._fluctuation_step)
```

`numpy.random.RandomState` accepts `int` seeds but internally converts to `uint32`. Python integers are unbounded, but numpy truncates via `ctypes.c_uint32(seed).value`. The maximum safe seed is `2^32 - 1 = 4,294,967,295`. At `fluctuation_step ≈ 1M` (reachable after ~2M lines at `FLUCTUATE_EVERY=2000`), the seed is `42 + 1,000,000 = 1,000,042` — still safe but the pattern is concerning. The real issue is that numpy's `RandomState(seed)` with a large seed produces **identical** sequences for seeds that differ by multiples of `2^32`. At step 4,294,967,254, the seed wraps around and repeats the same noise pattern as step 42.

**Fix:** Use a persistent RNG as described in CRITICAL-1.

### HIGH-2: `pq_encode()` uses O(N × n_sub × n_centroids × subdim) Python loop

**File:** `concept_space.py:662-665`
```python
for i in range(N):
    diffs = subvecs[i] - cb  # (n_centroids, subdim)
    dists = np.sum(diffs ** 2, axis=1)
    codes[i, m] = np.argmin(dists).astype(np.uint8)
```

For N=32K, n_sub=32, this is 1,024,000 iterations of the inner Python loop. Can be vectorized using `cdist` or broadcasting:
```python
for m in range(n_sub):
    sub = mat[:, m * subdim:(m + 1) * subdim]  # (N, subdim)
    cb = self.pq_codebooks[m]  # (K, subdim)
    # (N, 1, subdim) - (1, K, subdim) -> (N, K, subdim)
    dists = np.sum((sub[:, None, :] - cb[None, :, :]) ** 2, axis=2)  # (N, K)
    codes[:, m] = np.argmin(dists, axis=1).astype(np.uint8)
```

This gives ~100x speedup for the encoding step.

### HIGH-3: `normalize_vectors()` and `contrastive_spread()` don't invalidate PQ caches

**File:** `concept_space.py:834-866, 868-933`

Both methods set `self._vector_matrix = None` and `self._matrix_dirty = True` but do **not** clear `self.pq_codebooks = None` or `self.pq_codes = None`. After normalization/spread, PQ codes are stale but not invalidated — a subsequent `pq_decode_all()` would silently restore the old (un-normalized) vectors.

### HIGH-4: `pq_adc_search()` interprets L2 distance as cosine in Euclidean space, but vectors are on a sphere

**File:** `concept_space.py:751`
```python
sims = 1.0 - total_dists / 2.0
```

This converts squared Euclidean distance to cosine similarity using the identity `||x - y||² = 2(1 - cos(x,y))` for unit vectors. However, PQ codebook centroids are **not** guaranteed to be unit vectors — only the original vectors are normalized. The `cb - q_sub` distance involves a centroid that may have norm < 1, making `sims = 1 - dist²/2` incorrect. The similarity should be computed as `sim = (1 - dist²/2) * (1 / |cb|)` or centroids should be re-normalized after k-means.

### HIGH-5: `contrastive_spread()` gradient is not the correct Riemannian gradient

**File:** `concept_space.py:903-905`
```python
grad = vj - max_sim * vi
new_vi = vi + lr * grad
```

This uses the Euclidean chord approximation `vj - sim*vi` which is the **negative Riemannian gradient on the sphere** only for `vj` and `vi` on the unit sphere. The fix normalizes afterward (line 907-908), but the normalization step makes the update sub-optimal — the combined push-pull from `vj` and the renormalization don't compose linearly. The correct approach:
```python
# Push vi away from vj along the geodesic
grad = vi - max_sim * vj  # negative Riemannian gradient
new_vi = vi + lr * grad
new_vi /= max(np.linalg.norm(new_vi), 1e-10)
```

This is what `_repel_centroid` does correctly (line 281), but `contrastive_spread` does it differently (line 904). The sign difference (`vj - sim*vi` vs `vi - sim*vj`) matters.

### HIGH-6: `evaluate()` precomputation of `ngram_boost` is O(V × avg_transitions) memory

**File:** `crystal_generator.py:603-623`

The `ngram_boost` dict stores a `{ncid: bval}` map for every `prev_cid` that has any n-gram transitions. For a vocabulary of 32K with an average of ~10 transitions per prefix, this creates a dict with 320K entries — each entry is a Python dict mapping integers to floats. This consumes significant memory (~50MB) and is recomputed on every `evaluate()` call. Should be cached or computed lazily per batch item.

### HIGH-7: `train_from_text()` uses `max(ids) + 100` instead of `cs.vocab_size` for negative sampling

**File:** `crystal_generator.py:463`
```python
vocab_size = max(ids) + 100
```

If the line contains only low token IDs (e.g., max ID = 500), the negative sampling range is [0, 600) — far smaller than the actual vocabulary (32K). This biases negative samples toward low-ID tokens (which are typically BPE control characters and frequent subwords). Should use `cs.vocab_size` directly.

---

## 5. Medium Severity Issues

### MED-1: `homeostatic_boost` caches mean but not std — missing normalization

**File:** `concept_space.py:511-525`
```python
def homeostatic_boost(self, cid):
    usage = self.concept_usage.get(cid, 0.0)
    ...
    vals = list(self.concept_usage.values())
    self._hboost_mean_cache = np.mean(vals) if vals else 1.0
    ...
    boost = (mean_usage - usage) / max(mean_usage, 0.01)
```

The boost is `(mean - usage) / mean` which is not normalized by standard deviation. A concept with `usage = mean * 2` gets boost `-1.0` which clips to `-0.3`. But if the usage distribution has very low variance (all concepts have similar usage), even small deviations get maximum boost. Should divide by `max(std, 0.01*mean)` for statistical z-score normalization.

### MED-2: `_graph_search` hardcodes BFS depth to 5

**File:** `crystal_generator.py:239`
```python
while frontier and step < 5:
```

This is a hardcoded limit that caps semantic search radius to 5 hops. The `B` parameter (max distance = 1.2) already bounds the search — the step limit is redundant and can cut off legitimate paths when edge weights are near 1.0. Either remove the step limit (BFS naturally terminates via `B`) or make it configurable with a sane default.

### MED-3: `param_optimizer.py` `pmi_gate_min` parameter is never used

**File:** `parameter_optimizer.py:109-119`

The parameter `pmi_gate_min` is defined, tuned, and saved but **never referenced** in any training or generation code. The PMI gate in `_pmi_weight()` (crystal_generator.py:438) uses a hardcoded floor of `0.05`, not the tuned parameter.

### MED-4: `HormonalSystem.update()` bootstrap issue with `_prev_avg_match`

**File:** `hormonal_system.py:74-77`
```python
if not hasattr(self, '_prev_avg_match'):
    self._prev_avg_match = avg_match
delta_match = avg_match - self._prev_avg_match
self._prev_avg_match = avg_match
```

On the first call, `delta_match = avg_match - avg_match = 0` which is correct. But `save()` (line 192-203) conditionally saves `_prev_avg_match` and `load()` conditionally restores it. If `load()` doesn't set it (data missing from old serialization), the next `update()` call reinitializes it — but this reinitialization happens **after** `delta_match = 0`, so the first post-load step always has zero mastery signal. Minor, but indicates a fragile bootstrap.

### MED-5: `GrammarChecker` and morphological agreement aren't used in generation

**File:** `pos_tagger.py:75-108`

`check_agreement()` and POS transition probabilities exist but are never integrated into the generator's `_branch()` or scoring. The generator has no morphological agreement checking — it can generate "хороший погода" (wrong gender) without penalty. This is a gap noted in ARCHITECTURE.md that should be closed.

### MED-6: `Lattice.decay_all()` and `decay_connections()` don't trigger PPMI cache invalidation consistently

**File:** `syntax_lattice.py:276-314`

`decay_all()` does NOT invalidate `_ppmi_cache`, but `decay_connections()` does. Since both methods modify the underlying counts, both should invalidate. After `decay_all()`, PPMI values computed from stale cache will be slightly wrong.

### MED-7: `check_antonym_collapse()` prepends `'▁'` unconditionally

**File:** `vector_health.py:36-37`
```python
id_a = sp.PieceToId('▁' + a)
id_b = sp.PieceToId('▁' + b)
```

BPE tokens for mid-word subword units may not have the `▁` prefix. For example, the pair `("ка", "шка")` — `"ка"` is likely a suffix subword that appears without `▁`. The lookup will return `-1` (not found), silently skipping these pairs. Should try both with and without `▁`:
```python
id_a = sp.PieceToId('▁' + a)
if id_a < 0:
    id_a = sp.PieceToId(a)
```

### MED-8: `_semantic_delta()` uses simple mean of recent vectors, not centroid of a proper semantic field

**File:** `crystal_generator.py:65-68**
```python
response_vec = np.mean(recent_vecs, axis=0).astype(np.float32)
response_vec /= max(np.linalg.norm(response_vec), 1e-10)
```

Averaging the last `window` vectors loses temporal ordering information. A sequence `[A, B, C]` produces the same centroid as `[C, B, A]`. For semantic distance to a query, this is reasonable, but for response coherence, temporal attention would be better.

---

## 6. Minor Issues & Improvements

### LOW-1: Duplicate punctuation values in EOS check

**File:** `crystal_generator.py:190`
```python
if token_text in ('.', '!', '?', '…', '!', '?', '.', '...'):
```

`'.'` appears twice, `'!'` appears twice, `'?'` appears twice. This is a minor copy-paste bug. Should be:
```python
if token_text in ('.', '!', '?', '…', '...'):
```

### LOW-2: `_inhibition_step` is incremented even when early-returning

**File:** `concept_space.py:433-434`
```python
rng = np.random.RandomState(winner_cid + self._inhibition_step)
self._inhibition_step += 1
...
if not sampled_cids:
    return  # <-- state already advanced
```

Move the increment after the early-return checks.

### LOW-3: `pq_compression_ratio()` divides int by int in Python 3 → float, but `orig/(codes+cb)` may overflow

**File:** `concept_space.py:764`
```python
return orig / (codes_size + cb_size)
```

`orig = 32768 * 384 * 4 = 50,331,648` fits in int32, but `nbytes` for large arrays could exceed `2^31`. Use `float(orig) / float(codes_size + cb_size)` for safety.

### LOW-4: `check_code_range()` and `validate_vector_norms()` iterate over Python dicts every time

**File:** `concept_space.py:483-506`

Both methods are O(V) Python loops. At 32K vocab, each call iterates through 32K items. These are called at every checkpoint (every 500 lines). Could be vectorized using `np.array(list(self.fractal.codes.values()))` for a 100x speedup.

### LOW-5: `concept_space.load()` uses `__new__` then manually assigns attributes

**File:** `concept_space.py:980-983`
```python
obj = cls.__new__(cls)
obj.dim = data['dim']
obj.vocab_size = data.get('vocab_size', 0)
```

This bypasses `__init__`, so any initialization logic in `__init__` after setting attributes is lost. The manual attribute list at lines 983-1020 is fragile — if new attributes are added to `__init__`, they must be manually replicated in `load()`. Consider using `__init__` with a `from_dict` pattern instead.

### LOW-6: `save()` uses atomic write pattern but `concept_usage` check uses `hasattr`

**File:** `concept_space.py:962-964`
```python
if hasattr(self, 'concept_usage'):
```

`concept_usage` is set in `init_homeostasis()` which is called separately from `__init__`. It's possible to call `save()` before `init_homeostasis()` (e.g., after `load()` which doesn't call `init_homeostasis()` if usage data exists). The `hasattr` check is fragile — use `getattr(..., None) is not None`.

### LOW-7: `cleanup_old_checkpoints()` regex may fail on paths with dots

**File:** `train_full.py:65-66`
```python
files = sorted(glob.glob(...),
    key=lambda p: int(re.search(r'_(\d+)k\.json$', os.path.basename(p)).group(1)),
```

If any filename doesn't match the pattern, the lambda raises `AttributeError`. Either filter non-matching files or use a try/except.

### LOW-8: `save_3d_vis()` computes PCA on the full 32K×384 matrix every checkpoint

**File:** `train_full.py:331-337**
```python
X = np.array([cs.concept_vectors[c] for c in cids], dtype=np.float32)
X_mean = X.mean(axis=0, keepdims=True)
Xc = X - X_mean
pca = PCA(n_components=3, random_state=0)
proj = pca.fit_transform(Xc)
```

Randomized SVD on 32K×384 is ~100ms, done every 500 lines (~290 times over 145K lines = 29 seconds). This is acceptable but could be deferred to a background thread.

### LOW-9: `train_full.py` imports `sklearn.decomposition.PCA` at top but only uses it in `save_3d_vis`

**File:** `train_full.py:15`

This adds scikit-learn as a hard runtime dependency when it's only needed for visualization. If `--resume` without visualization, the import is unnecessary.

### LOW-10: `FractalField.from_dict()` re-orthogonalizes basis unconditionally

**File:** `concept_space.py:180-182`
```python
if err > 1e-5:
    Q, _ = np.linalg.qr(field.basis, mode='reduced')
    field.basis = Q.astype(np.float32)
```

QR decomposition changes the basis, which changes **all** concept vectors since `v = code @ basis`. The threshold `1e-5` is tight enough that float32 roundtrip through JSON may trigger it. When it does trigger, **all existing codes become inconsistent with the new basis**. The basis orthogonality error from JSON serialization should be corrected by re-orthogonalizing, but the codes must also be recomputed afterward. Currently they are not.

---

## 7. Cross-Pollination: EVA-Ai → FCF

### 7.1 Semantic Gap Detection (port from ConceptMiner)

EVA-Ai's `ConceptMiner._detect_semantic_gaps()` (concept_miner.py:508-579) detects clusters whose centroid is far from any existing concept — "phantom candidates" representing missing knowledge.

**Port to FCF:** FCF's `vector_health.py:detect_concept_clusters()` already has a similar mechanism, but it doesn't generate phantom concepts or propose new vector initializations. The gap in FCF is: when STDP converges two unrelated concepts to the same region (antonym collapse), there's no mechanism to **create new latent codes** for the missing anti-concept.

**Concrete recommendation:** After `prune_near_duplicates()` detects near-duplicate pairs, initialize a new latent code at `centroid_of_cluster + noise * spread` for each gap cluster. This gives FCF a rudimentary "concept invention" mechanism.

### 7.2 Duplicate Pruning (port from GraphCurator)

EVA-Ai's `GraphCurator.prune_duplicates()` (graph_curator.py:764-809) identifies and merges near-identical nodes (cosine > 0.95) by marking the lower-confidence one as a contradiction.

**Port to FCF:** FCF's `prune_near_duplicates()` (vector_health.py:105-134) is read-only — it logs duplicates but does **nothing** about them. The next step is merging: when two BPE tokens have cos > 0.95, they likely represent the same concept (e.g., different BPE segmentations of the same word). Merging would: (1) unify their n-gram statistics in SyntaxLattice, (2) interpolate their latent codes, (3) consolidate their connection graph entries. This would significantly reduce concept space redundancy.

### 7.3 Contradiction Mining (port from ContradictionMiner)

EVA-Ai's `ContradictionMiner._detect_candidate_pairs()` (contradiction_miner.py:351-413) finds pairs of semantically similar but logically contradictory nodes through cosine similarity + NLI.

**Port to FCF:** FCF's `check_antonym_collapse()` only handles predefined antonym pairs. A full contradiction detection system would: (1) find all concept pairs with cos > 0.6, (2) check if they share similar n-gram contexts (indicating functional synonymy despite antonym semantics), (3) apply a targeted repulsion gradient to push them apart. This is the natural extension of antonym collapse detection.

### 7.4 Temporal Decay (port from GraphCurator)

EVA-Ai's `GraphCurator.decay_nodes()` (graph_curator.py:811-839) implements temporal decay of node confidence based on access recency.

**Port to FCF:** FCF's `lattice.decay_all()` already decays n-gram counts, but there's no per-concept access tracking in the lattice (only `concept_usage` in ConceptSpace, which decays differently). Adding access-frequency-weighted decay to SyntaxLattice would make the connection graph adapt to distribution shift.

### 7.5 What FCF is Missing (Gap Analysis)

| Feature | EVA-Ai Has | FCF Status | Port Effort |
|---|---|---|---|
| Concept invention (phantoms) | ConceptMiner | Missing | High (requires new latent code creation) |
| Duplicate merging | GraphCurator.prune_duplicates | Read-only diagnostics | Medium |
| Contradiction detection | ContradictionMiner | Only predefined antonyms | Medium |
| Temporal decay with access tracking | GraphCurator.decay_nodes | Partial (lattice.decay_all) | Low |
| Hierarchical index | GraphCurator.build_hierarchical_index | Missing | Medium |
| Asynchronous background curation | GraphCurator + EventBus | Missing | High |
| Self-dialog for resolution | ContradictionMiner → SelfDialogLearning | Missing | High |
| Web verification of concepts | ConceptMiner._verify_web | Missing | Medium |
| NLI-based validation | ConceptMiner/NLI + ContradictionMiner | Missing | Medium |

---

## 8. Recommendations (Prioritized)

### P0 (Fix immediately — correctness bugs)

1. **Fix `_lateral_inhibition_fractal` RNG state drift** — move increment after early-return check (CRITICAL-2)
2. **Fix `normalize_vectors()` to update fractal codes** — prevent permanent inconsistency (CRITICAL-3)
3. **Fix `expand_dim()` to preserve existing structure** — prevent basis corruption (CRITICAL-4)
4. **Fix `_apply_vector_update` silent failure** — handle missing codes (CRITICAL-5)
5. **Fix `fluctuate()` to use persistent RNG** — performance + correctness (CRITICAL-1)

### P1 (Fix before production training)

1. **Fix `negative_sampling` range** — use `cs.vocab_size` instead of `max(ids) + 100` (HIGH-7)
2. **Fix `pq_encode()` vectorization** — 100x speedup for PQ encoding (HIGH-2)
3. **Fix PQ centroid normalization** in `pq_adc_search()` (HIGH-4)
4. **Fix `_graph_search` hardcoded depth limit** — make configurable (MED-2)
5. **Invalidate PQ caches** after `normalize_vectors()` and `contrastive_spread()` (HIGH-3)
6. **Fix `contrastive_spread()` gradient direction** — use correct Riemannian geometry (HIGH-5)

### P2 (Important improvements)

1. **Wire `pmi_gate_min` parameter** into `_pmi_weight()` (MED-3)
2. **Integrate morphological agreement** into generator scoring (MED-5)
3. **Fix `decay_all()` PPMI cache invalidation** (MED-6)
4. **Fix `check_antonym_collapse()` BPE prefix handling** (MED-7)
5. **Implement duplicate merging** in vector_health (Section 7.2)
6. **Add per-concept access tracking** to SyntaxLattice (Section 7.4)

### P3 (Nice to have)

1. **Vectorize `check_code_range()` and `validate_vector_norms()`** (LOW-4)
2. **Remove duplicate EOS punctuation** (LOW-1)
3. **Fix `cleanup_old_checkpoints()` regex safety** (LOW-7)
4. **Defer PCA visualization** to background thread (LOW-8)
5. **Fix `homeostatic_boost` z-score normalization** (MED-1)
6. **Audit all `hasattr` checks** for robustness (LOW-6)

---

## 9. Appendix: Bug Taxonomy Count

| Severity | Count | Key examples |
|---|---|---|
| CRITICAL | 5 | RNG creation, state drift, code-vector inconsistency, basis corruption, silent failures |
| HIGH | 7 | OOB negative sampling, PQ performance, wrong gradient direction, cache inconsistency, memory bloat |
| MEDIUM | 8 | Unused parameters, missing morphological agreement, fragile bootstraps, hardcoded limits |
| LOW | 10 | Duplicate literals, style issues, missing error handling |
| **Total** | **30** | |

---

## 10. Appendix: Fixed Code Snippets

### Fix for `fluctuate()` (CRITICAL-1)
```python
# Replace lines 117-123 in concept_space.py
def fluctuate(self, noise_scale=0.005, decay=0.999):
    if not hasattr(self, '_fluct_rng'):
        self._fluct_rng = np.random.RandomState(42)
    for cid in list(self.codes.keys()):
        c = self.codes[cid]
        noise = self._fluct_rng.randn(self.latent_dim).astype(np.float32) * noise_scale
        c[:] = c * decay + noise
    self._matrix_dirty = True
```

### Fix for `normalize_vectors()` (CRITICAL-3)
```python
# After line 857 in concept_space.py, add:
for cid, v in self.concept_vectors.items():
    code_new = v @ self.fractal.basis.T
    self.fractal.codes[cid] = code_new
self.fractal._matrix_dirty = True
```

### Fix for negative sampling range (HIGH-7)
```python
# Replace line 463 in crystal_generator.py:
# vocab_size = max(ids) + 100
vocab_size = cs.vocab_size
```

### Fix for `_inhibition_step` drift (CRITICAL-2)
```python
# Move increment in _lateral_inhibition_fractal:
rng = np.random.RandomState(winner_cid + self._inhibition_step)
cids = self._all_cids
...
if not sampled_cids:
    return
self._inhibition_step += 1  # <-- moved after early-return check
```

---

## 11. Summary of Top 3 Most Critical Findings

1. **`normalize_vectors()` silently breaks the code↔vector invariant** (CRITICAL-3): The method renormalizes all concept_vectors but never propagates changes back to fractal.codes. After calling this, every subsequent STDP update starts from stale latent codes, corrupting all future training. Any use of this method in its current form will silently destroy the model.

2. **`_lateral_inhibition_fractal` has an RNG state desynchronization bug** (CRITICAL-2): The inhibition step counter is incremented before the early-return check, and the sampling logic systematically undersamples when the winner appears early in the permutation. This means lateral inhibition is applied to a biased, smaller-than-requested subset of concepts, reducing its effectiveness at preventing vector collapse.

3. **`_apply_vector_update` silently drops updates when fractal codes are missing** (CRITICAL-5): When a BPE token has a vector in `concept_vectors` but no corresponding entry in `fractal.codes` (which can happen through the fallback init path at line 236), the method sets `concept_vectors[cid] = v_new` but returns without creating the code. The new vector is permanently detached from the fractal field, and all subsequent STDP updates to this concept skip the critical code projection step.
