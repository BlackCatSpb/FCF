# FCF — Fractal Concept Framework

Neuro-symbolic concept learning engine. No transformers. No gradient descent.

## Core Facts

| Parameter | Value |
|-----------|-------|
| Concepts | 146,494 × 384D |
| BPE model | `real_data/bpe_ru_146k.model` (SentencePiece) |
| Training corpus | `real_data/full_corpus_ru_clean.txt` (152,946 lines, ~52 MB) |
| GPU | MX550 (2 GB VRAM) |
| Batch size | 32 |
| Epoch time | ~7.3 h |

## Core Components

### `concept_space.py` — ConceptVectorStore, ConceptSpace
- Fractal field encoding: concepts as latent codes projected via shared orthonormal basis
- Subspace decomposition: identity (z_c), attention (z_a), meta (z_m) with per-subspace LR
- Octree field bits via `fractal_encoding.py`: wLCH path, H_weighted, LCP-based prefix grouping
- STDP (GPU: batched scatter_add, CPU: per-pair)
- Lateral inhibition via Riemannian gradient descent
- Centroid pull, sphere normalization, homeostatic boost

### `crystal_generator.py` — Training & inference engine
- `train_from_text()`, `train_batch()`, `evaluate()`, negative sampling
- STDP transitions, PMI-gated connection learning, field fluctuation
- Generation via concept navigation with RRF scoring

### `fractal_encoding.py` — Octree path + H_weighted + LCP
- `path(cid)` → octal digits, `H_weighted(p, q, gamma)` → weighted similarity
- Used by `build_octree_fields()` for anchor → concept field bits

### `fcf_config.py` — `FCFConfig` singleton
- All training hyperparameters in one place

### `syntax_lattice.py` — N-gram lattice + connection graph
- N-gram prefix tree for sequence statistics
- Connection graph with typed edges (related_to, has_quality, etc.)

### `parameter_optimizer.py` — LR schedule, PMI gate, homeostasis

### Other modules
- `morph_vocab.py` — morphological vocabulary (Natasha-based)
- `pos_tagger.py` — POS tagging
- `hormonal_system.py` — neuromodulation (ACh, NE, DA, 5HT)
- `train_full.py` — training harness, batching, checkpointing
- `inference.py` — read-only inference engine
- `eval_metrics.py` — evaluation framework (val vPPL, vector metrics)
- `model/` — HuggingFace wrappers (incomplete)
- `api/main.py` — FastAPI REST API

## Key Algorithms

- **STDP**: GPU batched scatter_add; CPU per-pair with fracture-adjusted LR
- **Lateral inhibition**: Riemannian gradient `-grad_R = sim·v - v_win`, tangent to sphere
- **Centroid pull**: uniform sphere repulsion from global centroid
- **Fractal fluctuation**: autonomous code drift with subspace-specific scaling
- **Contrastive spread**: targeted repulsion of nearest-neighbor pairs

## Issue Tracking

Bug fixes and known issues are tracked in [`AUDIT.md`](AUDIT.md).  
Development roadmap and future work in [`PLAN.md`](PLAN.md).

## Project Structure

```
FCF/
├── eva/symbolic/
│   ├── concept_space.py        # 146K concepts, fractal field, STDP
│   ├── crystal_generator.py    # Training + generation engine
│   ├── fractal_encoding.py     # Octree paths, H_weighted, LCP
│   ├── fcf_config.py           # Config singleton
│   ├── syntax_lattice.py       # N-gram prefix tree + connection graph
│   ├── parameter_optimizer.py  # LR schedule, PMI gate, homeostasis
│   ├── morph_vocab.py          # Morphological vocabulary
│   ├── pos_tagger.py           # POS tagging
│   └── hormonal_system.py      # ACh, NE, DA, 5HT modulation
├── model/                      # HuggingFace wrappers (incomplete)
├── api/main.py                 # FastAPI REST API
├── real_data/                  # Corpus, BPE model, checkpoints
├── train_full.py               # Training harness
├── inference.py                # Inference engine
├── eval_metrics.py             # Evaluation framework
├── eval_checkpoint.py          # Checkpoint text generation test
├── ARCHITECTURE.md             # This file
└── AUDIT.md / PLAN.md          # Bug tracking, roadmap
```
