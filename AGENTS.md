# FCF — Agent Session Log

## Context
Russian BPE concept navigation model. 146K wordpiece tokens, **768D hypersphere** (upgraded from 384D).
Multi-epoch STDP training with centroid pull, **learnable fields** (replaces octree), **HDC/VSA n-gram fallback**,
contrastive objective, adaptive per-concept dimensionality via L1.
Checkpoints at real_data/concept_space_{tag}.json + syntax_lattice_{tag}.\*.

## Progress

### [2026-06-14] Vector space fixes (commit 2f75fb8) … GPU acceleration (commit 4be3d93)
- Centroid pull LR 0.3, theta-gate cap 5, contrastive PPMI-80 sampling
- GPU path: _ensure_torch full-V tensors, no 500-pair threshold
- (See full history in AGENTS.md @ commit 75851c7)

### [2026-06-14] Visualization (commit pending)
- `viz_tsne.py`, PCA `save_3d_vis()` at every 5th checkpoint, Three.js viewer

### [2026-06-22] Qwen distillation, NPZ pruning, eval logging, architecture expansion
- **Architecture: dim 384→768, latent_dim 512→2048** (commit d0598b4)
  - Subspace ratios: `l_c = latent_dim * 3/5` (~60%), `l_a = latent_dim / 4` (25%), remainder `l_m`
  - `_init_identity` sparsity 12.5% → 3% (room for STDP densification)
  - All hardcoded 384/512 replaced with config references
  - Test precision tightened 1e-4→2e-4 for 768D norms
- **L1 regularisation for adaptive sparsity:**
  - `_apply_l1()` / `_apply_l1_batch()` soft-thresholds z_c subspace
  - `strength = l1_lambda * max(0, 1.0 − ce*2.0)` — high CE → weak L1 → denser code
- **Item memory for rare tokens (freq<3):**
  - `reinit_rare(freq_map, threshold=3)` replaces rare concept vectors with random unit vectors
  - Called after `build_octree_fields()` or `build_learned_fields()`
- **Learnable fields (HDC projection):**
  - `W_proj: [latent_dim, n_field_bits]` — random hyperplane projection
  - `field_bits[cid] = packbits(sign(code @ W_proj))` — same uint8 interface as octree
  - Hebbian update: `W += lr * mean(code * sign(code @ W))`, renormalises columns
  - `--learned-fields` and `--field-bits` CLI args in train_full.py
- **HDC/VSA n-gram fallback:**
  - VSA operations: `hdc_bind` (⊙), `hdc_permute` (ρ), `hdc_bundle`, `hdc_ngram_repr`, `hdc_unbind`
  - `hdc_memory: Dict[prefix_cids_tuple, bundled_repr]` — built during training
  - `hdc_predict()` — fallback when lattice has < 3 candidates
  - Integrated into `_branch()` as HDC signal + RRF fusion
- **Dynamic dimension (per-concept adaptive L1):**
  - `l1_lambda_per_cid[cid]` — per-concept L1 strength
  - `adjust_l1_lambdas()` — adjusts after training epoch: too dense → increase L1, too sparse → decrease
  - `l1_target_density = 8%` active in z_c
  - Tracking via `l1_density_window[cid]` (trailing 100 measurements)

### [2026-06-22] Qwen knowledge, minesweeper, antonym, cluster pull
- NPZ pruned & deduplicated (20.6M raw → 1.53M high-confidence pairs, 92 MB → 10 MB)
- Low-sim repulsion in `qwen_knowledge.py:get_factor`: neutral (<0.15), linear repel (0.15–0.20), boost (≥0.20)
- Minesweeper inverted: high CE → boost cluster potential (up to 1.2)
- Antonym repel: 22-pair Ru dictionary, GPU path, inverted pair_delta × 2.0
- Cluster centroid pull: pull_strength=0.05, octree cluster anchors
- Colab notebook updated with `LAYER_OFFSET` parameter for layers 12–24

## Priority Queue
- ✅ Architecture expansion (dim=768, latent_dim=2048)
- ✅ L1 regularisation + per-concept adaptive L1 (dynamic dimension)
- ✅ Item memory for freq<3
- ✅ Learnable fields (HDC projection) + sector index (field-in-field)
- ✅ HDC/VSA n-gram fallback + RRF fusion
- ✅ Dynamic capacity: auto grow/prune latent_dim + basis
- ✅ Focal search: search_in_sector(), focal_refine()
- ✅ Qwen knowledge, minesweeper inversion, antonym repel, cluster centroid pull
- ✅ Low-sim repulsion, eval logging

## Key Decisions
- **Adaptive sparsity via L1** — per-concept density self-regulates to 8% target
- **Dynamic capacity** — `grow_capacity()` adds orthogonal basis rows when mean density > 15%;
  `prune_capacity()` removes near-zero dimensions when >30% dims are sparse;
  `auto_adjust_capacity()` called every 3 epochs
- **Field-in-field (sector index)** — 3-level hierarchical binary partition (4+10+20 bits);
  `_sector_W[depth]` are separate random projections; `_rebuild_sector_index()` inverts
  prefix → CIDs per level; focal search only scores concepts in the same sector
- **Focal search** — `search_in_sector(query_cid, depth, k)` replaces O(V) full scan
  with O(|sector|) lookup; `focal_refine()` progressive coarse→fine refinement
- **Learnable fields** (binary projection via W_proj) replace octree — same uint8 popcount interface
- **HDC n-gram as fallback only** — SyntaxLattice tables remain primary
- **Item memory for freq<3** — rare tokens get random unit vectors
- **--learned-fields flag** — retains octree path for backward compat
- **Old checkpoints incompatible** (384D, 512 latent) — requires full re-training

## Relevant Files
- `eva/symbolic/concept_space.py` — FractalField (L1, field projection, HDC ops), ConceptSpace
- `eva/symbolic/crystal_generator.py` — _branch() with HDC fallback
- `eva/symbolic/stdc_trainer.py` — _update_hdc_ngrams, L1 wiring
- `eva/symbolic/fcf_config.py` — dim=768, latent_dim=2048
- `eva/symbolic/qwen_knowledge.py` — three-regime get_factor
- `train_full.py` — --learned-fields, --field-bits, epoch-level L1/field adaptation
- `eval_metrics.py` — qwen-factor distribution logging
- `precompute_qwen_knowledge.ipynb` — LAYER_OFFSET=12 for layers 12–24
- `real_data/qwen_knowledge.npz` — 1.53M pruned pairs (10 MB)
