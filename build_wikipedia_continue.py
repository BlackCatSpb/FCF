"""
build_wikipedia_continue.py — continuation from existing sentences file.
Step 3+: BPE tokenize, build storage, merge heads_meta.

Usage: python build_wikipedia_continue.py
"""
import sys, os, time, json, math, pickle
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix, save_npz, load_npz
from coordinate_packer import CoordinatePacker
from eva.symbolic.bpe_tokenizer import BPEVocab

V = 4101
WORD_OPEN, WORD_CLOSE = 157, 158
SENT_OPEN, SENT_CLOSE = 159, 160
SAVE_V5 = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5'
WP_HIER = os.path.join(SAVE_V5, 'hierarchical')
WIKI_DIR = os.path.join(SAVE_V5, 'wikipedia')
WIKI_TEXT = os.path.join(WIKI_DIR, 'wiki_ru_sentences.txt')

UNIFORM_LP = math.log(1.0 / V)


def tokenize_sentence(text, bpe):
    ids = [SENT_OPEN]
    for word in text.split():
        clean = word.strip('.,;:!?()[]{}«»\u2014\u2013\u2026"\'-')
        if clean:
            ids.append(WORD_OPEN)
            ids.extend(bpe.tokenizer.encode(clean).ids)
            ids.append(WORD_CLOSE)
        for ch in word:
            if ch in '.,;:!?()[]{}«»\u2014\u2013\u2026"\'-':
                try:
                    ids.append(bpe.tokenizer.encode(ch).ids[0])
                except Exception:
                    pass
    ids.append(SENT_CLOSE)
    return ids


def extract_word_spans(ids):
    spans = []
    in_word = False
    start = -1
    for t, tid in enumerate(ids):
        if tid == WORD_OPEN:
            in_word = True
            start = t + 1
        elif tid == WORD_CLOSE and in_word:
            spans.append((start, t - 1))
            in_word = False
    return spans


def main():
    print("=" * 60)
    print("WIKIPEDIA — STEP 3+ : TOKENIZE + STORAGE + MERGE")
    print("=" * 60)

    # Load
    print("\n[1] Loading sentences file, BPE, packer...")
    with open(WIKI_TEXT, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    bpe = BPEVocab()
    packer = CoordinatePacker()
    print(f"  {len(lines):,} sentences loaded")

    # Tokenize
    print("\n[2] BPE tokenization...")
    t0 = time.time()
    all_ids, all_sentences = [], []
    n_tok, n_wrd = 0, 0
    for i, text in enumerate(lines):
        ids = tokenize_sentence(text, bpe)
        spans = extract_word_spans(ids)
        all_ids.append(ids)
        all_sentences.append({
            'tokens': ids, 'word_spans': spans,
            'n_tokens': len(ids), 'n_words': len(spans),
        })
        n_tok += len(ids)
        n_wrd += len(spans)
        if (i + 1) % 100000 == 0:
            print(f"  {i+1:,}/{len(lines):,} ({time.time()-t0:.1f}s)")
    print(f"  Done: {len(lines):,} sent, {n_wrd:,} words, {n_tok:,} tok ({time.time()-t0:.1f}s)")

    # Transitions
    print("\n[3] Building transitions...")
    t0 = time.time()
    trans = defaultdict(int)
    tok_cnt = np.zeros(V, dtype=np.int32)
    for ids in all_ids:
        for t in range(len(ids) - 1):
            trans[(ids[t], ids[t+1])] += 1
            tok_cnt[ids[t]] += 1
        if ids:
            tok_cnt[ids[-1]] += 1

    pairs = list(trans.keys())
    rows = np.array([s for s, _ in pairs], dtype=np.int32)
    cols = np.array([d for _, d in pairs], dtype=np.int32)
    data = np.array([trans[p] for p in pairs], dtype=np.int32)
    si = np.argsort(rows, kind='stable')
    rows, cols, data = rows[si], cols[si], data[si]
    indptr = np.zeros(V + 1, dtype=np.int32)
    for r in rows:
        indptr[r + 1] += 1
    indptr = np.cumsum(indptr, dtype=np.int32)
    for i in range(V):
        s, e = indptr[i], indptr[i+1]
        if s < e:
            o = np.argsort(cols[s:e])
            cols[s:e] = cols[s:e][o]
            data[s:e] = data[s:e][o]
    t_csr = csr_matrix((data, cols, indptr), shape=(V, V), dtype=np.int32)
    rs = np.maximum(np.array(t_csr.sum(axis=1)).flatten(), 1)
    lp_data = np.zeros(len(data), dtype=np.float32)
    for i in range(len(data)):
        lp_data[i] = math.log(data[i] / rs[rows[i]]) if data[i] > 0 else -23.0
    lp_csr = csr_matrix((lp_data, cols.copy(), indptr.copy()), shape=(V, V), dtype=np.float32)
    print(f"  {len(pairs):,} transitions ({time.time()-t0:.1f}s)")

    # Morph/syntax
    print("\n[4] Morph/syntax extraction...")
    t0 = time.time()
    morph = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    syntax = defaultdict(lambda: defaultdict(int))
    for s in all_sentences:
        for wi, (ws, we) in enumerate(s['word_spans']):
            wl = we - ws + 1
            for pi in range(wl):
                tid = s['tokens'][ws + pi]
                morph[wl][pi][tid] += 1
            syntax[wi][s['tokens'][ws]] += 1
    print(f"  morph: {len(morph)} lens, syntax: {len(syntax)} pos ({time.time()-t0:.1f}s)")

    # Save
    print("\n[5] Saving Wikipedia storage...")
    np.savez_compressed(os.path.join(WIKI_DIR, 'sentences.npz'),
        tokens=np.concatenate([s['tokens'] for s in all_sentences]).astype(np.int16),
        token_lens=np.array([s['n_tokens'] for s in all_sentences], dtype=np.uint16),
        word_counts=np.array([s['n_words'] for s in all_sentences], dtype=np.uint16),
        word_spans=np.concatenate([np.array(s['word_spans'], dtype=np.uint16).ravel()
                                   for s in all_sentences]))
    save_npz(os.path.join(WIKI_DIR, 'transitions_csr.npz'), t_csr)
    save_npz(os.path.join(WIKI_DIR, 'log_prob_csr.npz'), lp_csr)
    np.savez_compressed(os.path.join(WIKI_DIR, 'token_counts.npz'), counts=tok_cnt)

    morph_d = {}
    for wl in morph:
        for pos in morph[wl]:
            tids = np.array(list(morph[wl][pos].keys()), dtype=np.int16)
            cnts = np.array(list(morph[wl][pos].values()), dtype=np.int32)
            morph_d[f'{wl}_{pos}'] = {'tids': tids, 'cnts': cnts, 'total': int(cnts.sum())}
    np.savez_compressed(os.path.join(WIKI_DIR, 'morph_cache.npz'), **morph_d)

    syn_d = {}
    for wn in syntax:
        tids = np.array(list(syntax[wn].keys()), dtype=np.int16)
        cnts = np.array(list(syntax[wn].values()), dtype=np.int32)
        syn_d[str(wn)] = {'tids': tids, 'cnts': cnts, 'total': int(cnts.sum())}
    np.savez_compressed(os.path.join(WIKI_DIR, 'syntax_cache.npz'), **syn_d)

    with open(os.path.join(WIKI_DIR, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'n_sentences': len(lines), 'n_tokens': n_tok, 'n_words': n_wrd,
            'vocab_size': V, 'text_id': 2,
            'n_transitions': int(sum(trans.values())),
            'n_unique_transitions': len(pairs),
            'source': 'ruwiki 20231101 via HuggingFace datasets',
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        }, f, ensure_ascii=False, indent=2)

    for fn in sorted(os.listdir(WIKI_DIR)):
        fp = os.path.join(WIKI_DIR, fn)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp) / 1024 / 1024
            print(f"    {fn}: {sz:.1f} MB")

    # Merge heads_meta
    print("\n[6] Merging into heads_meta.pkl...")
    with open(os.path.join(SAVE_V5, 'heads_meta.pkl'), 'rb') as f:
        existing = pickle.load(f)

    # Merge morph: equal weight (1:1 with existing)
    merged_morph = {}
    all_wl = set(existing['morph_logprob'].keys()) | set(morph.keys())
    for wl in sorted(all_wl):
        ex_wl = existing['morph_logprob'].get(wl, {})
        wi_wl = morph.get(wl, {})
        all_pos = set(ex_wl.keys()) | set(wi_wl.keys())
        merged_morph[wl] = {}
        for pos in sorted(all_pos):
            ex_arr = ex_wl.get(pos, np.full(V, UNIFORM_LP, dtype=np.float32))
            wi_counts = wi_wl.get(pos, {})
            wi_total = sum(wi_counts.values())
            if wi_total > 0:
                wi_arr = np.full(V, UNIFORM_LP, dtype=np.float32)
                for tid, cnt in wi_counts.items():
                    wi_arr[int(tid)] = math.log(cnt / wi_total)
            else:
                wi_arr = np.full(V, UNIFORM_LP, dtype=np.float32)
            merged_morph[wl][pos] = (ex_arr + wi_arr) / 2.0

    # Merge syntax
    wi_syn = {}
    for key, val in syn_d.items():
        wn = int(key)
        total = int(val['total'])
        arr = np.full(V, UNIFORM_LP, dtype=np.float32)
        if total > 0:
            for tid, cnt in zip(val['tids'], val['cnts']):
                arr[int(tid)] = math.log(cnt / total)
        wi_syn[wn] = arr

    merged_syn = {}
    all_wn = set(existing['syntax_logprob'].keys()) | set(wi_syn.keys())
    for wn in sorted(all_wn):
        ex_arr = existing['syntax_logprob'].get(wn, np.full(V, UNIFORM_LP, dtype=np.float32))
        wi_arr = wi_syn.get(wn, np.full(V, UNIFORM_LP, dtype=np.float32))
        merged_syn[wn] = (ex_arr + wi_arr) / 2.0

    # Merge token counts and CSR
    ex_tc = existing['token_counts']
    merged_tc = ex_tc + tok_cnt
    ex_csr = load_npz(os.path.join(WP_HIER, 'transitions_csr.npz'))
    cn_csr_path = os.path.join(SAVE_V5, 'conceptnet', 'transitions_csr.npz')
    if os.path.exists(cn_csr_path):
        cn_csr = load_npz(cn_csr_path)
        merged_csr = ex_csr + cn_csr + t_csr
    else:
        merged_csr = ex_csr + t_csr
    merged_csr.eliminate_zeros()

    # Recompute concept scores
    mc = float(max(1, int(merged_tc.max())))
    cs = np.zeros(V, dtype=np.float32)
    for tid in range(V):
        c = int(merged_tc[tid])
        if c > 0:
            nn = int(merged_csr.indptr[tid + 1] - merged_csr.indptr[tid])
            cs[tid] = min(1.0, math.sqrt(float(c) * float(max(1, nn))) / math.sqrt(mc))

    # Update stats
    stats = dict(existing['stats'])
    stats['n_sentences'] = int(stats.get('n_sentences', 0)) + len(lines)
    stats['n_tokens'] = int(merged_tc.sum())
    stats['n_words'] = int(stats.get('n_words', 0)) + n_wrd

    merged = {
        'morph_logprob': dict(merged_morph),
        'syntax_logprob': dict(merged_syn),
        'trans_sim_sparse': existing['trans_sim_sparse'],
        'contra_pairs': existing['contra_pairs'],
        'concept_scores': cs,
        'concept_clusters': existing.get('concept_clusters', {}),
        'ngram_sparse': existing.get('ngram_sparse', {}),
        'token_counts': merged_tc.astype(np.int32),
        'V': int(V),
        'stats': stats,
    }

    with open(os.path.join(SAVE_V5, 'heads_meta.pkl'), 'wb') as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)

    hm_mb = os.path.getsize(os.path.join(SAVE_V5, 'heads_meta.pkl')) / 1024 / 1024
    print(f"\n  heads_meta.pkl: {hm_mb:.1f} MB")
    sent_before = existing['stats'].get('n_sentences', 0)
    print(f"  Sentences before: {sent_before:,}")
    print(f"  Wikipedia added: {len(lines):,}")
    print(f"  Total: {stats['n_sentences']:,}")
    print(f"  Total tokens: {stats['n_tokens']:,}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
