# FCF — Neuro-Symbolic Concept Field

A neuro-symbolic text generation engine. No LLM, no transformer, no gradient descent — semantic navigation through a learned concept space.

## Core Idea

- **Concepts** are 128D unit vectors on a sphere, each representing a distinct semantic entity (root morphemes, not word forms)
- **Generation** = navigation: a query activates relevant concepts via attractor dynamics; the generator explores learned connections (co-occurrence), n-gram patterns, and modifier fields — never vector similarity between unrelated concepts
- **Training** = structure extraction from raw text: connection graph, role memory, n-gram lattice, STDP-based vector adjustment
- **Storage** uses Product Quantization (14× compression, 128D → 32 bytes per concept)

## Architecture

```
text → tokenizer → morph_parse → gate (attractor field) → core extraction
                                                           ↓
generator ← concept_space (vectors + connections + n-grams)
   ↑                ↓
   └── lattice (syntax patterns, role memory)
```

4 layers: Tokenizer → Gate → ConceptSpace → Generator

## Quick Start

```bash
pip install -r requirements.txt
python train_live.py                     # full training with dashboard
python experiments/test_comprehensive.py  # 51 tests
```

## Training

Processes raw Russian corpus line-by-line. Builds connection graph, n-gram sequences, and role memory from scratch. Cold start ~5–10s/line (morph parse via pymorphy3, cached after first use).

## Design Philosophy

- No token probabilities, no next-word prediction, no transformers
- Output is learned structure, not statistical completion
- Unknown words → neutral CID (no orthographic/BPE fallback — "I don't know" is correct behavior)
- All patterns derive from data via `train_from_text()` — zero hardcoded rules
