"""
Test: can VSA composition of morpheme-level e5 embeddings
reconstruct word-level e5 embeddings?

If cos(e5(word), VSA_compose(e5(morph_parts))) > random baseline,
the unified tokenizer-embedder concept is viable.

Usage:
    set HF_HUB_DISABLE_SYMLINKS_WARNING=1
    python scripts/test_vsa_composition.py [--words 200] [--device cpu]
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def encode(model, texts, batch_size=512):
    if not texts:
        return np.empty((0, 768), dtype=np.float32)
    return np.asarray(model.encode(texts, normalize_embeddings=True,
                                   show_progress_bar=False, batch_size=batch_size), dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--words', type=int, default=200)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    corpus_path = os.path.join(os.path.dirname(__file__), '..', 'real_data', 'full_corpus_ru.txt')
    if not os.path.exists(corpus_path):
        print("Corpus not found at", corpus_path)
        return 1

    print("Loading corpus...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        words = sorted(set(f.read().split()))
    words = [w for w in words if 4 <= len(w) <= 15]
    rng = np.random.RandomState(42)
    rng.shuffle(words)
    words = words[:args.words]
    print("Test: %d words, avg len=%.1f" % (len(words), np.mean([len(w) for w in words])))

    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer('intfloat/multilingual-e5-base', device=args.device)
    print("e5 loaded in %.1fs (dim=%d)" % (time.time()-t0, model.get_embedding_dimension()))

    word_embs = encode(model, words)

    # -- Simple heuristic: split each word into ROOT + ENDING --
    # This approximates morpheme-level VSA composition for the test
    def split_word(word):
        word = word.lower().strip()
        if len(word) < 4:
            return None
        # Take last 1-3 chars as potential ending
        for end_len in [3, 2, 1]:
            if len(word) - end_len >= 2:
                root = word[:-end_len]
                ending = word[-end_len:]
                return [('ROOT', root), ('ENDING', ending)]
        return None

    all_parts = []
    keep = []
    for word in words:
        parts = split_word(word)
        if parts:
            keep.append(word)
            all_parts.append(parts)

    print("Decomposed: %d/%d words with >=2 morphemes" % (len(keep), len(words)))

    # Flatten unique morphemes
    unique_morphs = sorted(set(m for parts in all_parts for _, m in parts))
    print("Unique morphemes: %d" % len(unique_morphs))
    morph_embs = encode(model, unique_morphs)
    m_dict = {m: morph_embs[i] for i, m in enumerate(unique_morphs)}

    sub_word_embs = encode(model, keep)

    # -- Test 1: Harmonizer-style compose (bind + role + bundle) --
    sims_bind = []
    for parts, target in zip(all_parts, sub_word_embs):
        composed = np.zeros(768, dtype=np.float32)
        for role, morph in parts:
            if morph in m_dict:
                rng_r = np.random.RandomState(hash('role_' + role) & 0x7FFFFFFF)
                rv = rng_r.randn(768).astype(np.float32)
                rv /= np.linalg.norm(rv)
                composed += m_dict[morph] * rv
        nrm = np.linalg.norm(composed)
        if nrm > 1e-10:
            composed /= nrm
            sims_bind.append(float(np.dot(composed, target)))

    print("\n--- Harmonizer-style composition (bind + bundle) ---")
    if sims_bind:
        arr = np.array(sims_bind)
        print("  cos: mean=%.4f std=%.4f max=%.4f min=%.4f" % (arr.mean(), arr.std(), arr.max(), arr.min()))
        print("  >0.3: %d/%d" % (int((arr > 0.3).sum()), len(arr)))
        print("  >0.5: %d/%d" % (int((arr > 0.5).sum()), len(arr)))

    # -- Test 2: Simple bundle (mean of morpheme vectors, no role bind) --
    sims_bundle = []
    for parts, target in zip(all_parts, sub_word_embs):
        avg = np.zeros(768, dtype=np.float32)
        for _, morph in parts:
            if morph in m_dict:
                avg += m_dict[morph]
        nrm = np.linalg.norm(avg)
        if nrm > 1e-10:
            avg /= nrm
            sims_bundle.append(float(np.dot(avg, target)))

    print("\n--- Simple bundle (mean of morpheme vecs, no roles) ---")
    if sims_bundle:
        arr = np.array(sims_bundle)
        print("  cos: mean=%.4f std=%.4f" % (arr.mean(), arr.std()))

    # -- Test 3: e5 word emb cosine with self (ceiling) --
    print("\n--- Ceiling: word emb cos with self ---")
    print("  cos: 1.0000 (by construction)")

    # -- Baseline: random unit vector --
    sims_rnd = []
    for target in sub_word_embs:
        rnd = rng.randn(768).astype(np.float32)
        rnd /= np.linalg.norm(rnd)
        sims_rnd.append(float(np.dot(rnd, target)))

    print("\n--- Baseline: random unit vector ---")
    arr = np.array(sims_rnd)
    print("  cos: mean=%.4f std=%.4f" % (arr.mean(), arr.std()))

    # -- Verdict --
    h_val = np.mean(sims_bind) if sims_bind else 0.0
    b_val = np.mean(sims_bundle) if sims_bundle else 0.0
    r_val = np.mean(sims_rnd) if sims_rnd else 0.0

    print("\n=== Summary ===")
    print("  Harmonizer-style (bind+role+bundle): %.4f" % h_val)
    print("  Simple bundle (mean, no roles):       %.4f" % b_val)
    print("  Random baseline:                      %.4f" % r_val)

    threshold = r_val + 0.05
    if h_val > threshold or b_val > threshold:
        print("  VERDICT: VSA composition carries semantic signal")
        print("  Unified tokenizer-embedder is viable")
    else:
        print("  VERDICT: VSA composition needs learned role vectors (STDP)")
        print("  Unified tokenizer-embedder requires end-to-end training")

    return 0

if __name__ == '__main__':
    sys.exit(main())
