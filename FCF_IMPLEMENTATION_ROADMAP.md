# FCF Implementation Roadmap — Comprehensive Analysis

> **Current State**: Epoch 2, cos≈0.024 (up from 0.003), vPPL≈111K (down from 120K), ~2 L/s CPU.  
> **Author**: AI Architect analysis of planned cognitive features against actual codebase.
> **Date**: 2026-06-14

---

## Corrigendum (2026-06-14, after code verification)

The AI Architect analysis below contains several inaccuracies that were verified against the actual codebase. This corrigendum corrects them.

### Verified facts from code

| Claim in roadmap | Actual code | Correction |
|---|---|---|
| `neg_samples=2 typical` | `neg_samples=8` at epoch 1 end (adapted by cos_flat≥5 trigger) | Already optimal, no change needed |
| `inh_strength=0.05` | `inh_strength=0.05` in opt.json — confirmed | Increasing to 0.15 is valid but **may over-push** at current cos=0.024 (std=0.097, so very few pairs exceed threshold 0.10) |
| `inh_threshold=0.10` | `inh_threshold=0.14` in opt.json | Slightly higher than agent assumed |
| `full_lr=0.15` | `full_lr=0.27` (adapted) | Agent assumed FAST-mode default |
| `topk_similar_concepts_by_vec missing` | Confirmed — only `topk_similar_concepts(cid)` exists | ✅ **Valid addition needed** |
| `PQ codes don't exist` | `pq_adc_search()` method DOES exist in ConceptSpace | ❌ Agent missed this. PQ-based retrieval is ready |
| `RAG pipeline = 80% ready` | Centroid + _branch bonus already works at `crystal_generator.py:391-401`. Missing: explicit retrieved-set bonus | **Actually 95% ready** — minimal addition needed |
| `z_a subspace dead` | `_apply_vector_update()` at `concept_space.py:751` bypasses subspace-LR — confirmed | ✅ Correct, but intentional (see docstring: "lr_c=0.01 was freezing 50% of code capacity") |
| `Lateral inhibition = 86% of time` | This was measured in FAST mode (old log). With batched numpy + centroid pull, bottleneck has likely shifted | **Needs re-measurement** |
| `SPA via SVO triples` | PPMI bigrams don't extract SVO relations reliably | ❌ SPA should use **PPMI connections with relation types** from `syntax_lattice.connections_of()` |
| `HDC low priority` | HDC XOR binding gives **exact unbinding** for bipolar vectors — more reliable than circular convolution | Deserves higher priority, especially for memory retrieval |

### Corrected quick-start checklist

Instead of agent's checklist, based on verified current state:

1. ~~Increase neg_samples to 8~~ → **Already 8**, skip
2. ~~Increase inh_strength to 0.15~~ → **Risky at current cos** — monitor first, vectors need separation before stronger inhibition helps
3. **Increase centroid pull LR** (`crystal_generator.py:761`): `sent_lr = base_lr_val * 0.1` → `0.3` — ✅ Valid
4. **Add `topk_similar_concepts_by_vec`** to ConceptSpace — ~30 lines
5. **Add retrieval bonus in `_branch()`** — ~10 lines using existing PQ `pq_adc_search`
6. **Verify training bottleneck** — needs profiling after current optimizations

### Additional observations (lead developer)

#### 1. Training speed bottleneck — needs re-profiling
Agent claims "lateral inhibition = 86% of time" from old FAST-mode log. After batched numpy + centroid pull + field_gate=False, the bottleneck **may have shifted**. Before implementing any features, profile current training:
```python
# In train_full.py, add timer around:
# (a) pair collection + field_overlap (line 552-588)
# (b) STDP gradient (line 614-670)  
# (c) centroid pull (line 753-776)
# (d) lateral inhibition (line 662-667)
# (e) lattice.update (line 778)
```
Without fixing the bottleneck first, every feature adds latency. Target: **10+ L/s** for practical iteration.

#### 2. Circular convolution at 384D — FFT overhead may dominate
At d=384, `np.fft.rfft` + `np.fft.irfft` overhead (~5µs) may exceed O(d²) direct convolution (~384² = 147K ops ≈ 1µs). **Test both approaches** before committing to FFT-based SPA. Also: unbinding via circular correlation is **lossy** at any dimension — recovered vector is approximate, not exact. For reliable memory retrieval, consider **bipolar HDC vectors** (sign(cos) → ±1) where XOR gives exact unbinding.

#### 3. RAG shows results only after cos > 0.05
At cos=0.024, top-K retrieval returns near-random results. Build the RAG pipeline now for **architecture readiness**, but set expectations: real query-specific generation begins when vectors separate. Meanwhile, cosine thresholding in retrieval (`only return if sim > 0.1`) prevents garbage contamination.

#### 4. Missing: evaluation methodology
Roadmap doesn't define **success criteria** for each feature:
- RAG: do different query words → different continuations? (qualitative)
- FIFO: does same query after different contexts → different answers?
- SPA: does `unbind(bind(a,b), a) ≈ b` at 384D? (quantitative: cos > 0.9)
- Predictive Coding: does prediction error decrease over time? (vPPL trend)
- Centroid pull: do within-sentence vectors cluster? (within-cos per sentence)

#### 5. Theta-gate kills learning for late tokens
`theta_gate = exp(-j / theta_tau)` at line 588 modulates LR by token position. With `theta_tau=30`:
- Position 0: gate = 1.0 (full LR)
- Position 30: gate = 0.37 (37% LR)
- Position 60: gate = 0.13 (13% LR)

Last third of long sentences barely learns. This explains why `"В настоящее территории"` dominates generation — these are early-position high-LR tokens. **Consider position-relative theta**, not absolute: `theta_gate = exp(-min(j, 5) / theta_tau)` — only the first 5 tokens get reduced LR, not the entire tail.

#### 6. Centroid pull vs STDP competition
Both modify the same vectors:
- **Centroid pull**: all tokens → sentence mean (isotropic attraction)
- **STDP**: target → context (directional, pairwise)

If centroid `sent_lr` is too high, it washes out STDP structure. The current 0.1× ratio was conservative for this reason. If we increase to 0.3× (agent's suggestion), monitor whether pairwise separation (cos_std) decreases.

#### 7. Lattice moving target problem
Lattice (n-gram frequencies, PPMI, connections) is updated every line during epoch 2 via `lattice.update(ids)`. This means:
- PPMI values shift as the model encounters the same data again
- `connections_of()` returns different results every checkpoint
- n-gram decay (`decay_all()` every 2000 lines) gradually erases epoch 1 statistics

The STDP vectors are chasing a moving target. Consider **freezing lattice after epoch 1** and only updating vectors in subsequent epochs, or use a separate validation lattice for evaluation.

### Corrected dependency graph

```
FIFO (1d) ──┐
             ├──→ RAG (1d, not 2 — 95% code exists) ──→ Predictive Coding (5d)
             │
             └──→ SPA via PPMI connections (2d, not 3) ──→ HDC (1d)
                                                          │
                                      Hierarchical        ┘
                                      Compression (4d)
```

---

## 0. Critical Precondition: The Vector Space Problem

Before any of the 7 features can deliver their full value, a fundamental issue must be acknowledged:

**Current cos=0.024 is essentially random gas** (random init gives cos≈0.0001). All 7 features depend on vector quality:
- **RAG** retrieves by cosine similarity → garbage in, garbage out
- **SPA** binds vectors → garbage composed with garbage
- **Predictive Coding** predicts field state → error signal is noise
- **FIFO memory** stores centroids → centroid of random vectors is random

The code already has the right mechanisms (negative sampling, PMI gating, lateral inhibition) but they're undershooting because:
- `_lateral_injection_fractal` at `threshold=0.10` with `std=0.15` → only ~25% of pairs inhibited (line 845 in `concept_space.py`)
- `neg_samples=2` typical vs 146K vocab → negligible repulsion
- Centroid pull (lines 754-773 in `crystal_generator.py`) has `sent_lr=0.1 * base_lr` → too weak

**Recommendation**: Address vector quality FIRST, or the 7 features will underperform. Specifically:
- Increase `inh_strength` to `0.15` (from 0.05 default)
- Increase `neg_samples` to `8-16` (from 2)
- Increase centroid pull LR to `0.3 * base_lr` (from 0.1)
- Add contrastive pull-push with hard negatives from PPMI

---

## 1. Predictive Coding

### What can be reused

| Component | File:Line | How to reuse |
|-----------|-----------|--------------|
| Perplexity computation | `crystal_generator.py:781-944` | `evaluate()` already computes `-log P(next\|ctx)` — this IS prediction error. Expose per-position log-probs. |
| STDP learning signal | `crystal_generator.py:518-777` | `train_from_text` already processes pairs. Prediction error can modulate the LR per pair via `theta_gate` pattern (line 588). |
| HormonalSystem surprise | `crystal_generator.py:222` | `surprise` is already computed and fed to hormones. This IS a prediction error signal. |
| `lattice.predict()` | `syntax_lattice.py:152-205` | N-gram prediction provides an explicit (non-vector) forward model. |
| `gen_updates` dict grouping | `crystal_generator.py:544-545` | The `defaultdict(list)` pattern groups updates by gen_cid — natural place to inject error modulation. |
| GPU batching of predictions | `crystal_generator.py:591-612` | `_ensure_torch` precomputes `_vecs_t` for all tokens — can compute batched predictions for all positions. |

### What needs to be added

**New module: `eva/symbolic/predictive_coding.py`**
- `FieldPredictor` class: takes current field state (recent N vectors) → predicts next field state
  - Simple baseline: linear predictor over last K vectors (K=3, weights learned by prediction error)
  - Advanced: `v_pred = Σ w_i * v_{t-i}` with w_i learned by running error
- `PredictionError` computation on sphere: `δ = v_pred - sim(v_pred, v_actual) * v_actual` (Riemannian, same pattern as STDP shift at line 836)
- `error_modulated_lr`: `eff_lr = base_lr * (1 + β * ||error||)` — error amplifies learning for surprising transitions

**New data structure:**
- Per-concept prediction error buffer: `concept_space.py` → `self.prediction_errors = np.zeros(V, dtype=float32)`
- Rolling forward model weights: `self.forward_weights = np.zeros((K, D), dtype=float32)` — K=3 context vectors

**Integration points:**

1. **`train_full.py` training loop** (line 490-506): After `train_from_text`, call `predictive_coding.step(context_ids, next_ids)`:
```python
# After line 506:
if prediction_error_weight > 0:
    pred_err = predictor.step(ids, cs, lattice)
    gen.train_lr *= (1.0 + prediction_error_weight * pred_err)
```

2. **`crystal_generator.py:586-588`** (theta_gate computation): Add prediction error as additional gate:
```python
# Current line 588:
theta_gate = math.exp(-j / max(self.theta_tau, 1.0))
# New:
pred_gate = 1.0 + self._prediction_error(ids[i], ids[j])
theta_gate = math.exp(-j / max(self.theta_tau, 1.0)) * pred_gate
```

3. **`_branch()`** (line 338): Use prediction error to downweight candidates that contradict the forward model.

### Implementation plan

```
Day 1: Add `predictive_coding.py` with `FieldPredictor` (linear forward model + error computation)
Day 2: Integrate error modulation into `train_from_text` — modulate LR by prediction surprise
Day 3: Add per-concept error tracking + visualization (which tokens are most surprising?)
Day 4: Predictive gating in `_branch()` — suppress candidates with high prediction error
Day 5: Tune parameters (β, K, decay of error) and validate against vPPL
```

### Expected impact

- **Primary**: Transforms STDP from co-occurrence statistics to prediction-error-driven learning
- **Mechanism**: High error → high LR (learn from surprises), low error → low LR (consolidate)
- **Measurable**: vPPL should drop faster per epoch as the forward model learns
- **Risk**: Without a good forward model, error signal = noise

### Key pitfalls

1. **Circular dependency**: Forward model needs good vectors to predict; vectors need good forward model to learn. Bootstrap by using `lattice.predict()` as initial forward model (non-vector, ngram-based)
2. **Error magnitude**: Sphere prediction error is bounded by 2.0 (max chord distance). Need to calibrate β so error modulates but doesn't dominate LR
3. **Temporal credit assignment**: Current STDP is instantaneous. Predictive coding across multiple tokens needs a proper forward model with state. Start with next-token prediction only.

---

## 2. Semantic Pointer Architecture (SPA / Circular Convolution)

### What can be reused

| Component | File:Line | How to reuse |
|-----------|-----------|--------------|
| 384D unit sphere vectors | `concept_space.py:445-452` | Circular convolution preserves dimensionality on sphere. `concept_vectors` are already normalized. |
| numpy FFT | (stdlib) | `np.fft.rfft` / `np.fft.irfft` — O(d log d) convolution |
| SyntaxLattice connections | `syntax_lattice.py:81-84` | SVO triples from PPMI connections provide training data for binding. `connections_of()` at line 492 |
| RRF scoring framework | `crystal_generator.py:373-385` | Add `bind_score` as another RRF signal alongside graph/ngram/vector |
| `_graph_search()` BFS | `crystal_generator.py:266-334` | Can find compositional paths (agent→action→patient via PPMI graph) |

### What needs to be added

**New function (or method on ConceptSpace): `bind(a, b)` and `unbind(bound, a)`**

```python
def circular_convolution(a, b):
    """a ⊛ b — circular convolution via FFT, preserves 384D."""
    A = np.fft.rfft(a)
    B = np.fft.rfft(b)
    result = np.fft.irfft(A * B, n=len(a))
    return result / np.linalg.norm(result)  # renormalize

def circular_correlation(bound, a):
    """bound ⊘ a — approximate unbinding."""
    A_conj = np.conj(np.fft.rfft(a))
    B = np.fft.rfft(bound)
    result = np.fft.irfft(A_conj * B, n=len(bound))
    return result / np.linalg.norm(result)
```

**New method on ConceptSpace: `compose(subject, verb, object)`**
Returns a single bound vector: `sv ⊛ role_subject + v ⊛ role_verb + ov ⊛ role_object`

**New training signal in `train_from_text`:**
- Extract SVO triples from bigrams with relation types (line 131 in syntax_lattice.py): `add_connection` already stores types
- For each triple, bind → store in a separate `bound_vectors` dict
- Learning: bound vector should cosine-similar to the sentence centroid

### Integration points

1. **`concept_space.py`**: Add `bind()`, `unbind()`, `bound_vectors` dict (key = (subj, verb, obj) → bound vector)
2. **`crystal_generator.py:393-402`** (centroid bonus): Add binding bonus:
```python
# After line 402:
if hasattr(self.cs, 'bound_vectors') and len(seq) >= 3:
    triple = tuple(seq[-3:])
    if triple in self.cs.bound_vectors:
        bv = self.cs.bound_vectors[triple]
        sim = float(np.dot(v, bv))
        combined[cid] *= (1.0 + max(0, sim) * 0.2)
```
3. **`train_from_text()`** (lines 754-773): After centroid pull, also train bound vectors:
```python
# After line 773:
for i in range(len(ids) - 2):
    subj, verb, obj = ids[i], ids[i+1], ids[i+2]
    triple = (subj, verb, obj)
    bound = self.cs.bind(
        self.cs.concept_vectors[subj],
        self.cs.circular_convolution(
            self.cs.concept_vectors[verb],
            self.cs.concept_vectors[obj]
        )
    )
    self.cs.bound_vectors[triple] = bound
```

### Implementation plan

```
Day 1: Add circular_convolution/circular_correlation to concept_space.py; test bind/unbind round-trip
Day 2: Add bound_vectors storage; train bound vectors in train_from_text for SVO triples
Day 3: Integrate binding score into _branch(); test generation with composition
```

### Expected impact

- **Primary**: Enables compositional semantics — "agent catches mouse" ≠ sum of individual vectors
- **Mechanism**: Bound vectors encode role-structured relationships
- **Measurable**: Generation should show more structured predicate-argument relations

### Key pitfalls

1. **Unbinding accuracy**: Circular convolution is lossy — unbinding recovers an approximation, not exact vector. 384D may be too low for reliable unbinding at scale. Test with `a ⊛ b ⊘ b ≈ a`.
2. **Binding explosion**: N concepts → O(N²) bound pairs → O(N³) triples. Need to store sparsely (only observed combinations).
3. **Role vectors**: Need role vectors (subject, verb, object) that are approximately orthogonal. Initialize with deterministic octree paths.

---

## 3. Internal RAG

### What can be reused

| Component | File:Line | Status |
|-----------|-----------|--------|
| Query centroid | `crystal_generator.py:170-177` | **ALREADY WORKS** — `query_words → _centroid` computation exists |
| Centroid bonus in _branch | `crystal_generator.py:393-402` | **ALREADY WORKS** — `intent_bonus = max(0, sim * (1-sim)) * 0.3` |
| topk_similar_concepts | `concept_space.py:973-999` | Direct NN search over concept vectors — at cos=0.024, this returns random, but exists |
| PQ approximate search | `concept_space.py:1122-1167` | ADC search for fast approximate NN — 16× compressed |
| `query_vecs` → centroid | `crystal_generator.py:171-177` | Full pipeline: BPE tokenize → get vectors → average → normalize |
| Beam conditioning | `crystal_generator.py:199` | `_branch(seq, wn, h_temp, expected_cid, self._centroid)` already passes centroid |

### What needs to be added

**Minimal change approach** (leveraging existing code):

The query-conditioning pipeline is already 80% complete. What's missing:
1. **Persistent retrieval storage**: Not just centroid, but explicit list of retrieved concept IDs
2. **Retrieval-weighted prior**: Boost candidates that are in the retrieved set
3. **Multi-query fusion**: When user gives multiple query words, retrieve per-word and fuse

**`CrystalGenerator` modifications:**

```python
# In generate() (line 145), after existing centroid computation (line 177):
# NEW: Retrieve top-K concepts for explicit conditioning
self._retrieved = []
if query_words and centroid is not None:
    k_retrieve = 30
    # Use PQ ADC for fast retrieval if available
    if self.cs.pq_codes is not None:
        retrieved = self.cs.pq_adc_search(centroid, k=k_retrieve)
    else:
        retrieved = self.cs.topk_similar_concepts_by_vec(
            centroid, k=k_retrieve, sample_size=2000
        )
    self._retrieved = [cid for cid, _ in retrieved]
```

**New method `topk_similar_concepts_by_vec`** (add to ConceptSpace, line 973 area):
```python
def topk_similar_concepts_by_vec(self, query_vec, k=10, sample_size=2000):
    """Top-K concepts closest to an arbitrary query vector."""
    if self.pq_codes is not None:
        return self.pq_adc_search(query_vec, k=k)
    # Fast approximate: sample subset for speed
    valid = self.concept_vectors._valid
    mat = self.concept_vectors._data[valid]
    order = np.where(valid)[0]
    if sample_size < len(order):
        idx = self.rng.choice(len(order), sample_size, replace=False)
        mat = mat[idx]
        order = order[idx]
    vn = query_vec / max(np.linalg.norm(query_vec), 1e-10)
    sims = mat @ vn
    top_idx = np.argpartition(-sims, k)[:k]
    return [(int(order[i]), float(sims[i])) for i in top_idx]
```

**In `_branch()`** (line 338): Add explicit retrieval bonus:
```python
# After line 402 (intent centroid bonus), before anti-repetition:
# 6b. Explicit retrieval bonus
if hasattr(self, '_retrieved') and self._retrieved:
    for cid in list(combined.keys()):
        if cid in self._retrieved:
            combined[cid] *= 1.5  # 50% boost for retrieved concepts
        else:
            # Small penalty for non-retrieved (if we have high confidence in retrieval)
            pass
```

### Implementation plan

```
Day 1: Add topk_similar_concepts_by_vec to ConceptSpace; wire into CrystalGenerator.generate()
Day 2: Add explicit retrieval bonus in _branch(); test with different query words producing different outputs
```

### Expected impact

- **Primary**: Query-specific generation — different query → different response
- **Mechanism**: centroid conditions beam search → retrieved concepts get boosted
- **Measurable**: should see "война" generation different from "любовь" even from same seed word

### Key pitfalls

1. **Vector space quality**: At cos=0.024, retrieval returns near-random results. The RAG pipeline is architectural — its value grows with vector quality. Implement NOW for architecture, expect real value after vectors separate.
2. **Retrieval speed**: `topk_similar_concepts` scans all 146K vectors. At current speed (~0.5ms per call? unclear), this is fine. But use PQ if it becomes slow.
3. **Query parroting**: Retrieved concepts include query words themselves. Filter out query CID from retrieved set.

---

## 4. Episodic FIFO Memory

### What can be reused

| Component | File:Line | Status |
|-----------|-----------|--------|
| Sentence centroid computation | `crystal_generator.py:754-762` | **ALREADY WORKS** — centroid = mean of sentence vectors |
| Centroid pull in training | `crystal_generator.py:753-773` | Sentence-level centroid pull already runs |
| `concept_usage` decay | `concept_space.py:911-916` | Exponential decay pattern (0.98) — reuse for memory decay |
| Line-by-line training loop | `train_full.py:490-506` | Natural place to update memory after each sentence |
| `_centroid` in generate | `crystal_generator.py:177` | Existing field, extend to include memory |

### What needs to be added

**New class in `concept_space.py` or new module `episodic_memory.py`:**

```python
class FIFOMemory:
    """Rolling window of sentence-level concept memories."""
    
    def __init__(self, capacity=10, dim=384):
        self.capacity = capacity  # 5-10 recent sentences
        self.dim = dim
        self.buffer = []  # list of (centroid_vector, concept_ids, timestamp)
        self.global_decay = 0.95  # old memories fade
    
    def push(self, centroid, concept_ids):
        """Store a sentence summary."""
        self.buffer.append((centroid.copy(), concept_ids.copy(), time.time()))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    
    def get_context(self, decay=True):
        """Weighted centroid of recent memories (decayed by age)."""
        if not self.buffer:
            return None
        if not decay:
            return np.mean([c for c, _, _ in self.buffer], axis=0)
        now = time.time()
        total_w = 0.0
        acc = np.zeros(self.dim)
        for cent, _, ts in self.buffer:
            age_hours = (now - ts) / 3600
            w = self.global_decay ** age_hours
            acc += cent * w
            total_w += w
        return acc / total_w if total_w > 0 else None
```

### Integration points

1. **`train_full.py` training loop** (after line 506):
```python
# After gen.train_from_text(...):
if idx > 0 and episodic_memory is not None:
    centroid = compute_sentence_centroid(ids, cs)
    episodic_memory.push(centroid, ids)
```

2. **`CrystalGenerator.generate()`** (line 173, centroid computation):
```python
# Current:
self._centroid = np.mean(query_vecs, axis=0).astype(np.float32) if query_vecs else None
# New: Fuse query centroid with episodic memory
if episodic_memory is not None:
    mem_centroid = episodic_memory.get_context(decay=True)
    if mem_centroid is not None and self._centroid is not None:
        # Weighted fusion: 60% from query, 40% from memory
        self._centroid = (0.6 * self._centroid + 0.4 * mem_centroid)
        n = np.linalg.norm(self._centroid)
        if n > 1e-10:
            self._centroid /= n
```

3. **`CrystalGenerator.__init__`** (line 31): Accept optional `episodic_memory` parameter.

### Implementation plan

```
Day 1: Add FIFOMemory class; wire into CrystalGenerator; memory updates in train_full.py
```

### Expected impact

- **Primary**: Cross-sentence coherence — previous sentences influence next generation
- **Mechanism**: Sentence centroids accumulate in rolling buffer, fused with query centroid
- **Measurable**: Should see topic continuity across multiple generate() calls

### Key pitfalls

1. **Memory decay**: Without decay, buffer fills with unrelated topics. Use exponential decay per entry (by timestamp or by insert count).
2. **Memory vs query conflict**: When query topic diverges from recent memory, which dominates? Solution: modulate fusion weight by query-memory similarity (high sim → trust memory, low sim → trust query).
3. **Centroid of centroids**: Averaging centroids loses information. Could store full concept ID sets instead.

---

## 5. Hierarchical Compression (Autoencoder)

### What can be reused

| Component | File:Line | Status |
|-----------|-----------|--------|
| Product Quantization | `concept_space.py:1003-1078` | Existing compression (16×). Training code can be adapted for autoencoder. |
| numpy/PyTorch | - | Standard tools for autoencoder training |
| 384D vectors | Throughout | Input and output dimension of autoencoder |
| `fractal.basis` | `concept_space.py:87` | Could be extended for phrase-level encoding |
| `lattice.ngrams` bigrams | `syntax_lattice.py:68` | PPMI bigrams = candidate multi-word phrases |
| ConceptVectorStore | `concept_space.py:18-61` | Storage for phrase-level vectors (D=384) |

### What needs to be added

**New module: `eva/symbolic/hierarchical_compressor.py`**

```python
class PhraseAutoencoder:
    """384 → 64 → 384 autoencoder for phrase-level compression.
    
    Architecture:
        encoder: Linear(384, 128) → ReLU → Linear(128, 64)
        decoder: Linear(64, 128) → ReLU → Linear(128, 384)
    
    Phrase detection: PPMI bigrams with PMI > threshold (e.g., 3.0)
    Training: reconstruct phrase centroid from bottleneck
    """
    
    def __init__(self, dim=384, bottleneck=64, device='cpu'):
        self.W_enc = np.random.randn(dim, bottleneck).astype(np.float32) * 0.01
        self.b_enc = np.zeros(bottleneck, dtype=np.float32)
        self.W_dec = np.random.randn(bottleneck, dim).astype(np.float32) * 0.01
        self.b_dec = np.zeros(dim, dtype=np.float32)
    
    def encode(self, v):
        """384D → 64D bottleneck."""
        h = v @ self.W_enc + self.b_enc
        h = np.maximum(h, 0)  # ReLU
        return h
    
    def decode(self, z):
        """64D → 384D reconstruction."""
        h = z @ self.W_dec + self.b_dec
        n = np.linalg.norm(h)
        return h / n if n > 1e-10 else h
    
    def train_step(self, phrase_vecs, lr=0.01):
        """One reconstruction training step on batch of phrase centroids."""
        # Simple gradient descent on MSE + sphere projection
        z = self.encode(phrase_vecs)
        recon = z @ self.W_dec + self.b_dec
        # Renormalize recon
        norms = np.linalg.norm(recon, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        recon /= norms
        # Loss: ||recon - phrase_vecs||^2
        error = recon - phrase_vecs
        dW_dec = z.T @ error  # (64, D)
        db_dec = error.sum(axis=0)
        dz = error @ self.W_dec.T
        dh = dz.copy()
        dh[z <= 0] = 0  # ReLU backward
        dW_enc = phrase_vecs.T @ dh
        db_enc = dh.sum(axis=0)
        # SGD update
        self.W_dec -= lr * dW_enc  # wait: these gradients need correction
        ...
```

**New data structures:**
- `phrase_space`: Dict of `phrase_key → (hierarchical_vector, constituent_cids)`
- `phrase_field`: Binary field bits for phrases (same 1024-anchor system)

### Integration points

1. **Phrase extraction**: In `train_from_text` (line 775, after lattice.update), detect PPMI bigrams with high PMI → compute phrase centroid → autoencode → store in phrase_space
2. **Generation integration**: In `_branch()`, check if (prev_cid, candidate) forms a known phrase → use phrase vector for scoring instead of individual vectors
3. **Training**: Add periodic autoencoder training in `train_full.py` checkpoint cycle (around line 552-617)

### Implementation plan

```
Day 1: Add PhraseAutoencoder class; phrase detection from PPMI bigrams
Day 2: Training loop for autoencoder; store phrase vectors
Day 3: Integration into _branch() for phrase-aware candidate scoring
Day 4: Add bottleneck visualization (PCA of bottleneck activations → phrase clusters)
```

### Expected impact

- **Primary**: Phrase-level semantics — "black cat" has its own vector ≠ "black" + "cat"
- **Mechanism**: 64D bottleneck forces compression → captures phrase meaning
- **Measurable**: Phrase reconstruction accuracy; generation should use multi-word units

### Key pitfalls

1. **Phrase boundary detection**: Current system has no phrase parser. Use PPMI bigrams as proxy: if PMI(A, B) > threshold, they form a phrase. This misses longer phrases.
2. **Autoencoder collapse**: 64D is very tight for 384D. Risk of all phrases collapsing to same reconstruction. Use L2 regularization on bottleneck activations.
3. **Training signal**: No explicit phrase labels. Train on reconstruction + contrastive: nearby phrases should have similar bottleneck codes.

---

## 6. Hyperdimensional Computing (HDC) Principles

### What can be reused

HDC is more of a theoretical unification than a concrete feature. Its operations overlap substantially with SPA (circular convolution is binding in HDC). The 384D vectors are in the HDC regime (typically 1000-10000D, but 384D still works).

**Overlap with SPA (Feature 2):**
- HDC bundling = vector averaging (already done as centroid)
- HDC binding = circular convolution (to be added in SPA)
- HDC permutation = circular shift of vector components
- HDC similarity = cosine similarity (already used throughout)

### What needs to be added

Given the overlap with SPA, add HDC operations to the same module:

```python
# In concept_space.py or spa_ops.py:

def hdc_bundle(vectors):
    """HDC bundling = normalized sum."""
    v = np.sum(vectors, axis=0)
    return v / np.linalg.norm(v)

def hdc_bind(a, b):
    """HDC binding = circular convolution (via FFT)."""
    return circular_convolution(a, b)

def hdc_permute(v, shift=1):
    """HDC permutation = roll elements."""
    return np.roll(v, shift)
```

### Implementation plan

```
Day 1: Add HDC operations; verify quasi-orthogonality of random vectors at 384D
```

### Expected impact

- **Primary**: Formal mathematical foundation for existing operations
- **Mechanism**: None — this is a theoretical cleanup
- **Measurable**: None directly

### Key pitfalls

1. **Dimensionality**: 384D may be too low for HDC guarantee of quasi-orthogonality (random vectors at 384D have expected cos ~0.05, vs 10000D at ~0.01). This limits binding capacity.
2. **Learned ≠ random**: HDC theory assumes random i.i.d. vectors. FCF vectors are learned and structured — HDC guarantees may not hold.

---

## 7. Active Inference / Free Energy Principle

### What can be reused

| Component | File:Line | How to map |
|-----------|-----------|------------|
| Perplexity | `crystal_generator.py:932-936` | PPL = exp(-1/N Σ log P) → Free Energy = -log P(o|θ) |
| Predictive Coding | (Feature 1) | Prediction error = variational free energy |
| HormonalSystem surprise | `hormonal_system.py:49-58` | Surprise = -log P(o) = free energy of observation |
| Generation as action | `crystal_generator.py:145-262` | Generation = policy that minimizes expected free energy |
| Fluctuation/drift | `concept_space.py:312-326` | Noise = exploration in parameter space |
| Centroid pull | `crystal_generator.py:753-773` | Prior = prefer vectors near sentence centroid |

### What needs to be added

Active Inference requires a **generative model** with:
1. **Transition model**: P(z_{t+1} | z_t, a_t) — how field state evolves
2. **Observation model**: P(o_t | z_t) — how tokens are generated from latent state
3. **Policy model**: P(a_t | z_t) — generation as action selection
4. **Free Energy computation**: F = E_q[ -log P(o|z) ] + KL[q(z) || p(z)]

**New module: `eva/symbolic/active_inference.py`**

The FCF equivalents are:
- Transition model → STDP (implicitly: if A→B in corpus, P(B|A) > 0)
- Observation model → SentencePiece decoder (deterministic: token ID → text)
- Policy → _branch() with temperature (stochastic next-token selection)
- Free Energy → -log P(next_token | context) from evaluate (line 900)

**Minimal implementation**: Use existing components but reframe them in active inference terms:
- After each sentence: F = -sum(log P(token|context)) — already computed in evaluate
- Generation as action: select next token to minimize expected F
- Learning: minimize F through STDP (already doing this implicitly)

### Integration points

Minimal: Add free energy tracking to training loop:
```python
# In train_full.py, around line 600 (eval):
free_energy = eval_result['total_log_prob']  # -sum(log P)
# Act as if we're minimizing variational free energy
opt.step(free_energy=free_energy, ...)
```

### Implementation plan

```
Day 1: Formalize free energy computation from existing perplexity
Day 2: Add expected free energy to generation (action selection minimizes F)
Day 3: Unify under free energy: learning (STDP) + generation (beam) + memory (prior)
Day 4-5: Active inference loop: perception → action → learning → perception
```

### Expected impact

- **Primary**: Single principle unifying learning, generation, attention, memory
- **Mechanism**: Everything minimizes free energy
- **Measurable**: Not directly, but provides theoretical grounding

### Key pitfalls

1. **Deep theory-practice gap**: Active Inference is a neuroscience theory, not an engineering framework. Implementation requires significant interpretation.
2. **KL on sphere**: Computing KL divergence between von Mises-Fisher distributions on S^384 is non-trivial. Approximate with cosine similarity.
3. **Expected free energy**: Requires averaging over future states (rollouts). Beam search already does this implicitly (multiple branches).

---

## Implementation Roadmap (Ordered)

### Dependency Graph

```
FIFO (1d) ──┐
             ├──→ RAG (2d) ──→ Predictive Coding (5d) ──→ Active Inference (5d)
             │
             └──→ SPA (3d) ──→ HDC (1d)
                              │
             Hierarchical    ─┘
             Compression (4d)
```

### Recommended build order

### Phase 1: Query & Context (Days 1-3)
**Why first**: Immediate user-facing value. Makes generation query-responsive and context-aware.

| Step | Feature | Files to create/modify | Key integration |
|------|---------|------------------------|-----------------|
| 1 | Internal RAG | `concept_space.py:973` (new method) + `crystal_generator.py:145` (retrieve in generate) + `crystal_generator.py:338` (retrieval bonus in _branch) | Lines 393-402 in _branch |
| 2 | Episodic FIFO | `eva/symbolic/episodic_memory.py` (NEW) + `crystal_generator.py:31` (accept memory) + `train_full.py:490` (update memory) | Lines 170-177 in generate, centroid fusion |

### Phase 2: Composition (Days 4-10)
**Why second**: Builds on the query/context foundation. Enables structured reasoning.

| Step | Feature | Files to create/modify | Key integration |
|------|---------|------------------------|-----------------|
| 3 | SPA | `concept_space.py` (circular_convolution + bound_vectors) + `crystal_generator.py:338` (bind score in _branch) + `crystal_generator.py:753` (bind training in centroid pull) | Lines 393-402 bonus, lines 754-773 training |
| 4 | Hierarchical Compression | `eva/symbolic/hierarchical_compressor.py` (NEW) + `crystal_generator.py:338` (phrase score in _branch) + `train_full.py:552` (autoencoder training at checkpoint) | Phrase detection from lattice.ngrams bigrams |

### Phase 3: Prediction & Unification (Days 11-20)
**Why third**: Requires good vectors and compositional representation to be effective.

| Step | Feature | Files to create/modify | Key integration |
|------|---------|------------------------|-----------------|
| 5 | Predictive Coding | `eva/symbolic/predictive_coding.py` (NEW) + `crystal_generator.py:586-588` (error gate) + `train_full.py:490` (post-train error step) | theta_gate at line 588 |
| 6 | HDC | `concept_space.py` (hdc_bundle/bind/permute) | Minimal — theoretical cleanup |
| 7 | Active Inference | `eva/symbolic/active_inference.py` (NEW) + `crystal_generator.py` (expected free energy in _branch) + `train_full.py` (F minimization) | PPL tracking at line 603 |

### Quick-start checklist (what to do right now for maximum impact)

1. **Increase negative sampling** (`train_full.py:line 500` → neg_samples=8)
2. **Increase centroid pull LR** (`crystal_generator.py:762` → sent_lr = base_lr * 0.3)
3. **Increase inh_strength** (`concept_space.py:673-678` → inh_strength=0.15)
4. **Add RAG retrieval** (Phase 1, Step 1 — 2 days, mostly leveraging existing code)
5. **Add FIFO memory** (Phase 1, Step 2 — 1 day, simple buffer)

---

## Major Architectural Gaps (Not Covered by 7 Features)

### Gap 1: No vector separation mechanism (CRITICAL)
**Problem**: cos=0.024 at epoch 2. The STDP + weak inhibition + weak negative sampling recipe does NOT create a clustered semantic space. All vectors are diffuse gas.

**Evidence**: `vacc@1=0.000` — the vector space contributes nothing to token prediction.

**Fix needed**: Not just parameters — the learning rule itself needs a **contrastive objective**. Currently:
- STDP: pull together co-occurring vectors (no push for non-co-occurring)
- Lateral inhibition: push apart similar vectors (only if cos > threshold, which is rare at cos_std=0.05)
- Negative sampling: push apart random pairs (but at 2 samples vs 146K vocab, negligible)

**Recommended addition**: `contrastive_hebbian` update:
```python
# In train_from_text, after positive STDP:
# For each positive pair (i,j), find a hard negative k where:
#   - cos(vi, vk) > 0.1 (similar but not co-occurring)
#   - OR: random sample from a different field (field_overlap == 0)
# Push vi away from vk with Riemannian gradient
```

### Gap 2: z_a subspace is dead code
**Problem**: `z_a` (attention subspace, 128 dimensions) is initialized with noise (`rng.randn * 0.01` at line 148) and never meaningfully updated.

**Evidence**: `shift_attention()` at line 278 is never called from `train_from_text`. The `apply_code_update()` at line 242 applies delta to z_a with lr_a=1.0, but:
- `_apply_vector_update()` at line 751 bypasses subspace-LR entirely (as noted in the comment: "Bypasses the subspace-LR bottleneck")
- Delta is computed from full vector, projected to full code space, not split per subspace

**Fix**: Either remove z_a (simplify to z_c + z_m) or implement attention shift properly:
- After each sentence, compute context-dependent z_a shift
- Store z_a shift in a context buffer
- Use z_a in generation to modulate field mask

### Gap 3: No planning / lookahead in generation
**Problem**: Beam search is greedy (always picks best K next tokens). No lookahead or deliberation.

**Evidence**: `generate()` at line 145 uses beam with `_branch()` at each step, but:
- No evaluation of full-sequence quality vs query
- No backtracking if all beams dead-end
- No "System 2" deliberative mode

**Fix**: Add `rollout_score` — for each beam candidate, simulate 3-5 steps ahead without commitment, score the projected trajectory, use that score in current selection.

### Gap 4: No concept creation
**Problem**: Vocabulary is fixed (146K BPE tokens). New concepts cannot be added for novel patterns.

**Evidence**: `init_concepts()` at line 471 iterates `range(vocab_size)`. No mechanism to extend. `expand_dim()` at line 1178 extends dimension but not vocabulary.

**Fix**: Add `create_concept(v)` — append a new concept with vector v, extend fractal.codes, rebuild field bits. Trigger when a query centroid is far from all existing concept vectors (novelty detection).

### Gap 5: Lateral inhibition is the training bottleneck
**Problem**: Per session log (2026-06-12), lateral inhibition is 86% of train time (~14.5ms/line out of 21.1ms).

**Evidence**: `_lateral_inhibition_fractal()` at line 845 does ~48K calls at 60μs each = 2.88s for 200 lines.

**Fix options (not yet implemented):**
1. Skip inhibition when `total_elr < threshold` (tiny updates don't need inhibition)
2. Batch `delta_code` computation across all affected concepts in one matmul
3. Numba JIT for `_apply_vector_update` + `apply_code_update` hot path
4. Bloom filter: only compute inhibition for gen_cids whose effective LR exceeds threshold

---

## Summary Table

| Feature | Reuse | New Code | Complexity | Impact | Priority |
|---------|-------|----------|------------|--------|----------|
| Internal RAG | **80%** (centroid, topk, _branch) | Retrieval bonus in _branch + topk_by_vec | **2 days** | High (query-specific gen) | **1** |
| Episodic FIFO | **60%** (centroid, decay pattern) | FIFOMemory class, centroid fusion | **1 day** | Medium (cross-sent coherent) | **2** |
| SPA/Convolution | **40%** (vectors, FFT, RRF) | bind/unbind, bound_vectors, SVO triples | **3 days** | Medium-High (composition) | **3** |
| Hierarchical Compression | **30%** (PQ training, vectors) | PhraseAutoencoder, phrase_space | **4 days** | High (phrase semantics) | **4** |
| Predictive Coding | **50%** (PPL, hormones, theta_gate) | FieldPredictor, error gate, per-concept error | **5 days** | High (causal learning) | **5** |
| HDC Principles | **70%** (existing ops) | hdc_bundle/bind/permute wrappers | **1 day** | Low (theoretical) | **6** |
| Active Inference | **40%** (PPL, surprise, STDP) | Free energy formalization, generative model | **5 days** | Very High (unification) | **7** |

### Final recommendation

Build in order: **RAG → FIFO → SPA → Compression → Predictive → HDC → Active Inference**.

But first: **fix the vector space** — increase neg_samples to 8, inh_strength to 0.15, centroid pull LR to 0.3×. Without separated vectors, all 7 features operate on "random gas" and underperform their potential.
