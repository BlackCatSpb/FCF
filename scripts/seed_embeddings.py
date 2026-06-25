"""
Seed FCF concept vectors using multilingual-e5-base embeddings.

Three modes:
  1. Direct: encode each token directly with e5 (default)
  2. Morpheme-bundle: decompose word -> e5(morph_i) -> VSA bundle -> word vec
  3. --all: seed ALL concepts, not just rare

Usage:
    python scripts/seed_embeddings.py --cs real_data/concept_space.json --sp real_data/bpe_ru_146k.model --device cpu
    python scripts/seed_embeddings.py --cs ... --sp ... --morph-bundle
    python scripts/seed_embeddings.py --cs ... --sp ... --all
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

_CONSONANTS = set('\u0431\u0432\u0433\u0434\u0436\u0437\u0439\u043a\u043b'
                  '\u043c\u043d\u043f\u0440\u0441\u0442\u0444\u0445\u0446'
                  '\u0447\u0448\u0449')


def _simple_decompose(word):
    """Rule-based ROOT+ENDING split for VSA bundle seeding."""
    w = word.lower().strip()
    if len(w) < 4:
        return None
    endings = ['\u0430\u043c\u0438','\u044f\u043c\u0438','\u044b\u0445','\u0438\u0445',
               '\u043e\u0433\u043e','\u0435\u0433\u043e','\u043e\u043c\u0443','\u0435\u043c\u0443',
               '\u043e\u0439','\u0435\u0439','\u0438\u0435','\u0438\u044f','\u0443\u044e','\u043e\u044e',
               '\u0435\u043c','\u0438\u043c','\u044b\u043c','\u0430\u043c\u0438',
               '\u0430','\u044b','\u0435','\u0443','\u043e','\u0438','\u044f','\u044e','\u0439']
    for e in sorted(endings, key=len, reverse=True):
        if len(w) > len(e) + 1 and w.endswith(e) and w[-(len(e) + 1)] in _CONSONANTS:
            root = w[:-len(e)]
            if len(root) >= 2:
                return [('ROOT', root), ('ENDING', e)]
    split = max(2, int(len(w) * 0.7))
    return [('ROOT', w[:split]), ('ENDING', w[split:])]


def load_e5_model(device='cpu'):
    from sentence_transformers import SentenceTransformer
    model_name = 'intfloat/multilingual-e5-base'
    print(f"  Loading {model_name} on {device}...", end=' ', flush=True)
    t0 = time.time()
    model = SentenceTransformer(model_name, device=device)
    print(f"{time.time()-t0:.1f}s")
    return model


def seed_embeddings(cs, sp, model, freq_map, threshold=3, seed_all=False, morph_bundle=False):
    n_ok = 0
    n_skip = 0
    n_byte = 0
    n_empty = 0
    batch_texts = []
    batch_cids = []
    batch_size = 512

    special = {'<s>', '</s>', '<unk>', '<pad>', '<cls>', '<sep>', '<mask>'}

    def flush_batch():
        nonlocal n_ok
        if not batch_cids:
            return
        if morph_bundle:
            for cid, token in zip(batch_cids, batch_texts):
                parts = _simple_decompose(token)
                if parts:
                    m_texts = [p for _, p in parts]
                    m_embs = model.encode(m_texts, normalize_embeddings=True,
                                          show_progress_bar=False)
                    bundle = np.mean(m_embs, axis=0).astype(np.float32)
                    bundle /= np.linalg.norm(bundle)
                    cs.concept_vectors[cid] = bundle
                else:
                    emb = model.encode([token], normalize_embeddings=True,
                                       show_progress_bar=False)[0]
                    v = np.asarray(emb, dtype=np.float32)
                    v /= np.linalg.norm(v)
                    cs.concept_vectors[cid] = v
                cs.fractal.codes.pop(cid, None)
                n_ok += 1
        else:
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
    if morph_bundle:
        print("  Mode: morpheme-bundle (VSA compose of root+ending)")
    return n_ok


def main():
    parser = argparse.ArgumentParser(description='Seed FCF concept vectors with e5 embeddings')
    parser.add_argument('--cs', required=True, help='ConceptSpace .json path')
    parser.add_argument('--sp', required=True, help='SentencePiece .model path')
    parser.add_argument('--threshold', type=int, default=3, help='Frequency threshold')
    parser.add_argument('--device', default='cpu', help='Torch device (cpu/cuda)')
    parser.add_argument('--output', default=None, help='Output path (default: overwrite --cs)')
    parser.add_argument('--all', action='store_true', help='Seed ALL concepts, not just rare')
    parser.add_argument('--morph-bundle', action='store_true',
                        help='VSA bundle of morpheme e5 vecs instead of direct')
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
    n = seed_embeddings(cs, sp, model, freq_map, args.threshold, args.all, args.morph_bundle)

    output = args.output or args.cs
    cs.save(output)
    print(f"Saved to {output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
