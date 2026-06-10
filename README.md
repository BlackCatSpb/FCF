# EVA — Neuro-Symbolic Concept Navigation

No LLM, no transformer, no gradient descent.  
BPE-token concepts on a unit sphere, self-organised via STDP on fractal codes, navigated via PPMI-weighted graph search (BMSSP-EVA).

## Core Architecture

```
corpus → SentencePiece (32K BPE) → STDP training → fractal codes @ basis → unit vectors
                                                      ↓
generation ← CrystalGenerator: BMSSP-EVA graph search + RRF(0.7 graph, 0.15 ngram, 0.15 vector)
                ↑
           SyntaxLattice: n-grams (2/3/4), connections (co-occurrence + PPMI)
```

- **ConceptSpace**: 32K BPE tokens, each = 384D vector on unit sphere from `normalize(code @ basis)`. Basis = random orthogonal 512×384 matrix. Fractal codes (512D) updated via STDP, projected back through `basis.T`. Null space (128D) for Langevin drift.
- **SyntaxLattice**: n-gram counters (up to order 4), connection graph (co-occurrence with typed relations), PPMI cache for all 1.9M connection pairs. Decay-all with floor for bounded growth.
- **CrystalGenerator**: generation as beam search over concept IDs. `_branch()` uses RRF over: BMSSP-EVA graph search (PPMI-weighted multi-source BFS), n-gram predictions, vector cosine similarity. `train_from_text(neg_samples=N)` for negative sampling.
- **Checkpoints**: binary format — fractal codes as `.codes.npz` (np.savez_compressed), lattice as `.lattice.npz` (jagged arrays). Backward compatible with old JSON.

## Generation Example (greedy, beam=1)

| Seed | Output |
|---|---|
| князь | князь литовский Гедимин Канонизирова синхронизации половой |
| война | война показала полную собственность в и митропо Ленинграде |
| дом | дом купца Москворец папы римского автора попадают в |
| человек | человек миллиона долларов 1,386 пробы крупноплотную |

Semantic paths from PPMI graph, not n-gram templates.

## Training

```bash
python train_full.py                          # default: lr=0.03, pmi_gate=True
python train_full.py --fast                   # fast: lr=0.15, neg_samples=3, pmi_gate=False
python train_full.py --resume 2k              # resume from checkpoint
python train_full.py --fast --resume 4k       # resume in fast mode
```

Processes `full_corpus_ru.txt` line-by-line. Checkpoints every 500 lines. Full eval every 1000/2000 lines (fast/default). LR warmup over first 1000 steps.

## Files

| File | Role |
|---|---|
| `train_full.py` | Training loop, diagnostics, 3D vis |
| `eva/symbolic/concept_space.py` | 32K fractal vectors, STDP, inhibition, homeostasis |
| `eva/symbolic/syntax_lattice.py` | N-grams, connections, PPMI cache, decay |
| `eva/symbolic/crystal_generator.py` | Generation, BMSSP-EVA graph search, RRF branching |
| `eva/symbolic/hormonal_system.py` | Homeostatic modulation |
| `serve_vis.py` | HTTP server for 3D visualisation |
| `train.bat` | Desktop launcher |

## Key Metrics (current state)

- vecPPL ~30897 (−4% from random 32158) — vectors still converging
- acc@1 ~0.10 (ngram-based top-1 accuracy)
- vacc@1 = 0.000 (vectors not yet contributing to predictions)
- graph BFS: ~436 nodes explored per source (was 31K before PPMI + semantic filter)
- generation: ~2.6s/8words (greedy)
- checkpoint: 1MB json + 51MB npz (concept), 78MB npz + 0.0MB meta (lattice)
