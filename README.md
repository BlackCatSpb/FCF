# FCF — Fractal Cognitive Field

*Neuro-symbolic concept learning on a self-organizing hypersphere.*  
*No transformers. No backpropagation. No gradient descent.*

**FCF** is a fully learnable vector-symbolic architecture (VSA) that builds semantic space through local plasticity rules — STDP, lateral inhibition, contrastive divergence, centroid pull — operating directly on concept vectors embedded in a dynamically expanding fractal field.

Each BPE token IS a concept. Each concept IS a point on the unit hypersphere in **768-dimensional space**. Coordinates arise from a latent fractal code projected through an orthonormal basis. The field autonomously grows, prunes, and reallocates its dimensions based on knowledge density — there is no fixed dimensionality.

---

## Foundations

### Representation

Every concept is represented as a pair `(z, v)`:

```
z ∈ ℝ^{L}     — latent code (sparse, subspace-decomposed)
v ∈ 𝕊^{D-1}  — unit vector on the hypersphere (D = 768)
v = normalize(z · B)  where B ∈ ℝ^{L×D} is a shared orthonormal basis
```

The latent code `z` is split into three functional subspaces:

| Subspace | Ratio | Function | Plasticity |
|----------|-------|----------|------------|
| `z_c` (content) | ~60% | Stable semantic identity | Slow (L1-regularized) |
| `z_a` (attention) | ~25% | Contextual behaviour | Fast |
| `z_m` (meta) | ~15% | Morphological / grammatical form | Medium |

This subspace decomposition prevents catastrophic forgetting: content remains stable while activity adapts rapidly to context.

### Field-in-Field: Learnable Hierarchical Fields

Traditional VSA uses fixed random hypervectors. FCF replaces them with **learned field projections** — binary locality-sensitive hash (LSH) codes that adapt to the geometry of the learned space:

```
field_bits[cid] = packbits(sign(z[cid] · W_proj))   where W_proj ∈ ℝ^{L×512}
```

The field projection matrix `W_proj` is updated via a Hebbian rule:
```
W_proj += lr · mean( z · sign(z · W_proj) )     (column-normalized)
```

This is extended to a **3-level hierarchical sector index**:

| Level | Bits | Purpose |
|-------|------|---------|
| 0 (coarse) | 4 | ~16 broad clusters |
| 1 (medium) | 10 | ~1024 sectors |
| 2 (fine) | 20 | ~1M sub-sectors |

Each level has its own projection matrix `_sector_W[lvl]`. The sector index is an inverted map `{prefix → list[CID]}` enabling **focal search** — only concepts in the same sector are scored, reducing search from O(V) to O(|sector|).

### Dynamic Capacity Growth

The field is not fixed. The model monitors code density every 3 epochs:

```
per_concept_density = mean(|z[cid]| > 1e-4)
```

- If **mean density > 15%** → `grow_capacity()` adds new orthogonal basis vectors (×1.5 factor), extends all codes with zeros, and pads all projection matrices. The field grows.
- If **>30% of dimensions are dead** (<2% active) → `prune_capacity()` removes them, compressing the field.
- Per-concept **L1 regularization** targets 8% active density in `z_c`, adjusted individually via `l1_lambda_per_cid[cid]`.

Result: the model automatically maintains the necessary dimensionality within the bounds of its knowledge — no manual tuning.

---

## How It Learns

### STDP (Spike-Timing-Dependent Plasticity)

If token A precedes token B in text, B's latent code shifts toward A's. The pull magnitude is modulated by:

- **PMI** (Pointwise Mutual Information) — gates pairs below a threshold
- **Distance weight** — nearby tokens exert stronger pull
- **Frequency weight** — rare tokens receive proportionally stronger updates
- **Qwen knowledge factor** — precomputed semantic signal from Qwen modulates LR per pair (boost/repel/neutral)
- **Field gate** — cross-sector pairs are inhibited

All pairs in a micro-batch are processed as a single GPU scatter_add operation.

### Negative Sampling

Random concepts are pushed away from each updated concept. Push strength is weighted by per-concept prediction error — harder concepts receive stronger regularization. Field gates filter invalid negatives.

### Contrastive Objective

Hard negative mining over the top-K most similar concepts. Those that are similar but neither co-occur nor share field proximity are pulled apart. Cross-field pairs are repelled aggressively; within-field pairs are treated gently to preserve cluster structure.

### Lateral Inhibition

All concepts updated in a batch repel each other along the sphere's geodesic. Prevents representational collapse.

### Centroid Pull

All tokens in a sentence are weakly pulled toward their mean centroid. Functions as a sentence-level regularizer.

---

## HDC/VSA Integration

FCF implements the full VSA algebra as a **fallback mechanism**, not the primary representation:

| Operation | Definition | Use |
|-----------|------------|-----|
| **bind** (⊙) | `a * b` (element-wise multiply) | N-gram encoding |
| **permute** (ρ) | `roll(v, 1)` (circular shift) | Position encoding |
| **bundle** | `accum = (1−lr)·accum + lr·v` | N-gram memory accumulation |
| **unbind** | `context ⊙ memory_repr` | Query decoding |

During training, every observed n-gram `(w1..wn)` updates an item memory:
```
hdc_memory[(cid1, cid2)] = bundle(vn)    — for each prefix→next
```

During inference, if the statistical n-gram lattice (SyntaxLattice) returns fewer than 3 candidates, the HDC fallback fires:
```
query = unbind(context_codes, hdc_memory[prefix])
candidates = top-k cos(query, all_codes)
```

This is integrated into the RRF (Reciprocal Rank Fusion) scoring alongside graph-based, syntax-based, and vector-similarity signals.

---

## The Training Pipeline

```
Input text (Russian, SentencePiece BPE 146K)
  →
  Pair building (context window 2–5):
    distance-weighted · frequency-weighted · PMI-gated · field-gated · Qwen-modulated
  →
  GPU micro-batch:
    STDP apply → L1 shrinkage → Negative sampling → Contrastive →
    Centroid pull → Lateral inhibition → HDC n-gram memory update
  →
  Per-concept EMA error tracking → Parameter optimizer → Checkpoint save
```

### Epoch-level adaptations

| Every | Action |
|-------|--------|
| 1 epoch | Hebbian field update (`update_learned_fields`) |
| 2 epochs | Per-concept L1 adjustment (`adjust_l1_lambdas`) |
| 3 epochs | Dynamic capacity check (`auto_adjust_capacity`) |

---

## Comparison with HDC/VSA (Kanerva 2009)

| Aspect | HDC/VSA (2009) | FCF (2026) |
|--------|----------------|------------|
| Vectors | Fixed random i.i.d. hypervectors | Learned via STDP, subspace-decomposed |
| Dimensionality | Fixed (1000–10000) | Dynamic (grows at 15% density, prunes dead dims) |
| Field structure | None | 3-level hierarchical LSH sector index |
| Search | Full O(V) scan | Focal O(|sector|) via inverted sector index |
| N-grams | bind(permute(...)) only | Statistical lattice primary + VSA fallback |
| Item memory | Fixed random for everything | Random only for freq<3; learned STDP for rest |
| Learning | One-pass bundle accumulation | Multi-epoch STDP + contrastive + centroid pull |
| External knowledge | None | Qwen distillation as LR modulator |
| Capacity | Fixed | Adaptive grow/prune |
| Sparsity | Implicit (high-D random) | Explicit L1 regularization per concept |

---

## GPU Implementation

Designed for **2GB VRAM** consumer GPUs (MX550):

- FP16 storage for concept vectors, FP32 for operations
- All STDP pairs in a micro-batch processed as a single kernel
- Persistent GPU tensors — no per-step reallocation
- Deferred synchronization: batched GPU→CPU write-back
- Fused post-STDP: contrastive, negative sampling, centroid pull share one similarity matrix

---

## Quick Start

```bash
# Fresh training with all architectural features
python train_full.py --fresh --learned-fields --field-bits 512 -e 3

# Resume from checkpoint
python train_full.py --resume --learned-fields --field-bits 512 -e 3

# Fast mode (elevated LR, useful for testing)
python train_full.py --fast --learned-fields --field-bits 512

# Or via batch scripts
train.bat          # production launch
train_fast.bat     # fast test launch
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 1 | Number of training epochs |
| `--fresh` | off | Ignore checkpoints, start from scratch |
| `--resume` | auto | Resume from last checkpoint |
| `--fast` | off | Higher LR + aggressive negative sampling |
| `--learned-fields` | off | Use learnable field projection (recommended) |
| `--field-bits` | 512 | Number of field bits for W_proj |
| `--max-lines` | 0 | Limit training to N lines (for testing) |

---

## Project Structure

```
FCF/
├── eva/symbolic/
│   ├── concept_space.py         # 146K concepts, fractal field, L1, sector index, HDC ops
│   ├── crystal_generator.py     # Training + generation engine + RRF scoring
│   ├── stdp_trainer.py          # STDP, negative sampling, contrastive, HDC n-gram
│   ├── fcf_config.py            # Config (dim=768, latent_dim=2048, ...)
│   ├── syntax_lattice.py        # N-gram statistical lattice + connection graph
│   ├── fractal_encoding.py      # Octree paths (legacy, replaced by learned fields)
│   ├── qwen_knowledge.py        # Qwen distillation NPZ loader
│   ├── morph_vocab.py           # Morphological vocabulary
│   ├── hormonal_system.py       # Neuromodulation (ACh, NE, DA, 5HT)
│   └── parameter_optimizer.py   # LR schedule, PMI gate, homeostasis
├── train_full.py                # Training harness
├── inference.py                 # Read-only inference
├── eval_metrics.py              # Validation metrics
├── requirements.txt
├── real_data/                   # Corpus, BPE model, Qwen NPZ, checkpoints
└── tests/                       # 145 automated tests
```

---

## Motivation

FCF asks a question at the intersection of two traditions:

> **Can semantic space be built without gradient descent — using only local plasticity rules, a learnable field projection, and a self-organizing fractal code?**

If yes, it opens a path to fully interpretable language models where:
- Every vector is a **physical fact** about co-occurrence statistics
- Every field bit is a **learned semantic hyperplane**
- Every sector boundary is a **discovered conceptual distinction**
- The model's capacity **grows with its knowledge** and **shrinks when unused**

No latent activations. No uninterpretable deep networks. No fixed dimensionality.

---

## Status

Research prototype. 145 automated tests. All mechanisms operational on 2GB GPU.

- ✅ Adaptive dimensionality (grow/prune latent_dim + basis)
- ✅ Learnable field projection (W_proj) with Hebbian update
- ✅ Hierarchical sector index (4+10+20 bits) for focal search
- ✅ HDC/VSA n-gram fallback with item memory
- ✅ Per-concept L1 sparsity regularization
- ✅ Item memory for rare tokens (freq<3)
- ✅ Qwen distillation as LR modulator (1.53M pairs)
- ✅ 3-level RRF scoring (graph + syntax + vector + HDC)
- ✅ Full GPU pipeline (STDP, contrastive, centroid, negative sampling)
- ✅ Checkpoint/resume, switched evaluation, curriculum learning

---

## Requirements

- Python 3.8+
- PyTorch (optional, CPU fallback)
- SentencePiece
- NumPy, scikit-learn, SciPy

---

## References

- Kanerva, P. (2009). *Hyperdimensional Computing: An Introduction.*
- Gayler, R. W. (2003). *Vector Symbolic Architectures.*
- Plate, T. A. (2003). *Holographic Reduced Representations.*
- Kleyko, D., et al. (2022). *A Survey on Hyperdimensional Computing.*
- Rachkovskij, D. A., & Kussul, E. M. (2001). *Binding and Normalization of Binary Sparse Distributed Representations.*

---

## Русский

FCF — нейро-символическая модель языка, обучаемая без обратного распространения через локальные правила пластичности и саморганизующееся фрактальное поле со динамической размерностью. Подробное описание на русском — в ARCHITECTURE.md.

---

*FCF — research project. Questions, experiments, and contributions welcome.*
