"""
Seed rare FCF concept vectors using multilingual-e5-base embeddings.

USAGE:
    python scripts/seed_embeddings.py --cs real_data/concept_space.json --sp real_data/bpe_ru_146k.model

OPTIONS:
    --cs PATH         Path to ConceptSpace checkpoint (.json)
    --sp PATH         Path to SentencePiece model (.model)
    --threshold N     Frequency threshold (default: 3, seed freq<3 tokens)
    --device DEVICE   Torch device (default: cpu)
    --output PATH     Output path (default: same as --cs, overwrites)
    --all             Seed ALL concepts, not just rare ones
"""

import argparse
import os
import sys
import time
import numpy as np

def load_e5_model(device='cpu'):
    from sentence_transformers import SentenceTransformer
    model_name = 'intfloat/multilingual-e5-base'
    print(f"  Loading {model_name} on {device}...", end=' ', flush=True)
    t0 = time.time()
    model = SentenceTransformer(model_name, device=device)
    print(f"{time.time()-t0:.1f}s")
    return model


def seed_embeddings(cs, sp, model, freq_map, threshold=3, seed_all=False):
    from eva.symbolic.concept_space import ConceptVectorStore

    n_ok = 0
    n_skip = 0
    n_byte = 0
    n_empty = 0
    batch_texts = []
    batch_cids = []
    batch_size = 512

    special = {'<s>', '</s>', '<unk>', '<pad>', '<cls>', '<sep>', '<mask>'}

    def flush_batch(): # encode + store
        nonlocal n_ok
        if not batch_cids:
            return
        embs = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
        for cid, emb in zip(batch_cids, embs):
            v = np.asarray(emb, dtype=np.float32)
            v /= np.linalg.norm(v)
            cs.concept_vectors[cid] = v
            cs.fractal.codes.pop(cid, None)
            n_ok += 1
        batch_texts.clear()
        batch_cids.clear()

    for cid in range(cs.vocab_size):
        if not seed_all:
            freq = freq_map.get(cid, 0) if freq_map is not None else 0
            if freq >= threshold:
                n_skip += 1
                continue

        token = sp.IdToPiece(cid)

        if token in special:
            n_skip += 1
            continue

        if token.startswith('<0x') and token.endswith('>'):
            n_byte += 1
            continue

        if not token.strip():
            n_empty += 1
            continue

        batch_texts.append(token)
        batch_cids.append(cid)

        if len(batch_texts) >= batch_size:
            flush_batch()

    flush_batch()

    cs.fractal._matrix_dirty = True
    print(f"  OK: {n_ok} | skip(freq): {n_skip} | byte: {n_byte} | empty: {n_empty}")
    return n_ok


def main():
    parser = argparse.ArgumentParser(description='Seed FCF concept vectors with e5 embeddings')
    parser.add_argument('--cs', required=True, help='ConceptSpace .json path')
    parser.add_argument('--sp', required=True, help='SentencePiece .model path')
    parser.add_argument('--threshold', type=int, default=3, help='Frequency threshold')
    parser.add_argument('--device', default='cpu', help='Torch device (cpu/cuda)')
    parser.add_argument('--output', default=None, help='Output path (default: overwrite --cs)')
    parser.add_argument('--all', action='store_true', help='Seed ALL concepts, not just rare')
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    print("Loading SentencePiece...", end=' ', flush=True)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(args.sp)
    print(f"{sp.vocab_size()} tokens")

    print("Loading ConceptSpace...", end=' ', flush=True)
    from eva.symbolic.concept_space import ConceptSpace
    cs = ConceptSpace.load(args.cs)
    print(f"{cs.vocab_size} concepts, {cs.dim}D")

    freq_map = {}
    if not args.all:
        lattice_path = args.cs.replace('concept_space', 'syntax_lattice').replace('.json', '.lattice.npz')
        if os.path.exists(lattice_path):
            data = np.load(lattice_path, allow_pickle=True)
            freq_data = data.get('concept_freq', data.get('freq', None))
            if freq_data is not None:
                freq_map = {int(cid): int(f) for cid, f in freq_data.items()} \
                    if isinstance(freq_data, dict) else {i: int(f) for i, f in enumerate(freq_data)}
                print(f"  Loaded freq map: {len(freq_map)} entries")
        else:
            print(f"  Warning: syntax lattice not found at {lattice_path}")

    model = load_e5_model(args.device)
    n = seed_embeddings(cs, sp, model, freq_map, args.threshold, args.all)

    output = args.output or args.cs
    cs.save(output)
    print(f"Saved to {output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
