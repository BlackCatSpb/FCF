"""
build_conceptnet_trajectories.py — Tokenize ConceptNet, build storage,
merge heads_meta.pkl with War and Peace for multi-text support.

Usage: python build_conceptnet_trajectories.py
"""
import sys, os, time, json, math, pickle
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix, save_npz, load_npz

from coordinate_packer import CoordinatePacker
from eva.symbolic.bpe_tokenizer import BPEVocab

V = 4101
SAVE_V5 = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5'
WP_HIER = os.path.join(SAVE_V5, 'hierarchical')
CN_DIR = os.path.join(SAVE_V5, 'conceptnet')
CN_TEXT = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\conceptnet\conceptnet_ru.txt'
MERGED_HEADS = os.path.join(SAVE_V5, 'heads_meta_merged.pkl')

WORD_OPEN, WORD_CLOSE = 157, 158
SENT_OPEN, SENT_CLOSE = 159, 160


def tokenize_sentence(text, bpe):
    ids = [SENT_OPEN]
    for word in text.split():
        clean = word.strip('.,;:!?()[]{}«»\u2014\u2013\u2026"\'\\')
        if clean:
            ids.append(WORD_OPEN)
            ids.extend(bpe.tokenizer.encode(clean).ids)
            ids.append(WORD_CLOSE)
        for ch in word:
            if ch in '.,;:!?()[]{}«»\u2014\u2013\u2026"\'\\':
                try:
                    ids.append(bpe.tokenizer.encode(ch).ids[0])
                except:
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


def encode_sentence(ids, packer, text_id):
    L = len(ids)
    word_spans = extract_word_spans(ids)
    n_words = len(word_spans)
    traj = np.zeros((L, packer.DIM), dtype=np.float32)
    for t in range(L):
        tid = ids[t]
        word_idx, pos_in_word, word_len = -1, -1, 0
        for wi, (ws, we) in enumerate(word_spans):
            if ws <= t <= we:
                word_idx, pos_in_word, word_len = wi, t - ws, we - ws + 1
                break
        flags = 0
        if word_idx >= 0:
            if pos_in_word == 0: flags |= (1 << packer.F_WORD_START)
            if pos_in_word == word_len - 1: flags |= (1 << packer.F_WORD_END)
            if word_idx > 0: flags |= (1 << packer.F_HAS_WORD_LEFT)
            if word_idx < n_words - 1: flags |= (1 << packer.F_HAS_WORD_RIGHT)
        else:
            flags |= (1 << packer.F_SPECIAL)
        if t == 0: flags |= (1 << packer.F_SENT_START)
        if t == L - 1: flags |= (1 << packer.F_SENT_END)
        mt = (6 if tid in (WORD_OPEN, WORD_CLOSE, SENT_OPEN, SENT_CLOSE) else
              5 if 48 <= tid <= 57 else
              2 if pos_in_word == 0 and word_idx >= 0 and tid >= 161 else
              1 if pos_in_word == 0 and word_idx >= 0 else
              3 if pos_in_word > 0 else 7)
        if 48 <= tid <= 57: flags |= (1 << packer.F_DIGIT)
        if 4 <= tid <= 155 or tid >= 161: flags |= (1 << packer.F_LETTER)
        ctx = 0
        for di in range(max(0, t - 3), min(L, t + 4)):
            if di != t: ctx ^= (ids[di] & 0xFF) << ((di - t + 3) & 7)
        ctx &= 0xFF
        traj[t] = packer.pack_token(
            token_id=tid, pos_in_word=max(0, pos_in_word),
            word_len=max(1, word_len), word_num=max(0, word_idx),
            pos_in_sent=t, sent_len=L, flags=flags,
            meta_type=mt, context_hash=ctx, text_id=text_id)
    return traj


def main():
    os.makedirs(CN_DIR, exist_ok=True)
    packer = CoordinatePacker()
    bpe = BPEVocab()

    print("=" * 60)
    print("BUILD CONCEPTNET STORAGE")
    print("=" * 60)

    print("\n[1] Reading ConceptNet text...")
    with open(CN_TEXT, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    print(f"  {len(lines):,} sentences")

    print("\n[2] Tokenizing...")
    t0 = time.time()
    all_ids, all_sentences = [], []
    n_tok, n_wrd = 0, 0
    for i, text in enumerate(lines):
        ids = tokenize_sentence(text, bpe)
        spans = extract_word_spans(ids)
        all_ids.append(ids)
        all_sentences.append({
            'tokens': ids, 'word_spans': spans,
            'n_tokens': len(ids), 'n_words': len(spans)})
        n_tok += len(ids); n_wrd += len(spans)
        if (i + 1) % 100000 == 0:
            print(f"  {i+1:,}/{len(lines):,} ({time.time()-t0:.1f}s)")
    print(f"  Done: {len(lines):,} sent, {n_wrd:,} words, {n_tok:,} tok ({time.time()-t0:.1f}s)")

    print("\n[3] Building transitions...")
    t0 = time.time()
    trans = defaultdict(int)
    tok_cnt = np.zeros(V, dtype=np.int32)
    for ids in all_ids:
        for t in range(len(ids) - 1):
            trans[(ids[t], ids[t+1])] += 1
            tok_cnt[ids[t]] += 1
        if ids: tok_cnt[ids[-1]] += 1

    pairs = list(trans.keys())
    rows = np.array([s for s, _ in pairs], dtype=np.int32)
    cols = np.array([d for _, d in pairs], dtype=np.int32)
    data = np.array([trans[p] for p in pairs], dtype=np.int32)

    si = np.argsort(rows, kind='stable')
    rows, cols, data = rows[si], cols[si], data[si]

    indptr = np.zeros(V + 1, dtype=np.int32)
    for r in rows: indptr[r + 1] += 1
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

    print("\n[4] Morph/syntax...")
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

    print("\n[5] Saving ConceptNet storage...")
    np.savez_compressed(os.path.join(CN_DIR, 'sentences.npz'),
        tokens=np.concatenate([s['tokens'] for s in all_sentences]).astype(np.int16),
        token_lens=np.array([s['n_tokens'] for s in all_sentences], dtype=np.uint16),
        word_counts=np.array([s['n_words'] for s in all_sentences], dtype=np.uint16),
        word_spans=np.concatenate([np.array(s['word_spans'], dtype=np.uint16).ravel()
                                   for s in all_sentences]))
    save_npz(os.path.join(CN_DIR, 'transitions_csr.npz'), t_csr)
    save_npz(os.path.join(CN_DIR, 'log_prob_csr.npz'), lp_csr)
    np.savez_compressed(os.path.join(CN_DIR, 'token_counts.npz'), counts=tok_cnt)

    morph_d = {}
    for wl in morph:
        for pos in morph[wl]:
            tids = np.array(list(morph[wl][pos].keys()), dtype=np.int16)
            cnts = np.array(list(morph[wl][pos].values()), dtype=np.int32)
            morph_d[f'{wl}_{pos}'] = {'tids': tids, 'cnts': cnts, 'total': int(cnts.sum())}
    np.savez_compressed(os.path.join(CN_DIR, 'morph_cache.npz'), **morph_d)
    with open(os.path.join(CN_DIR, 'morph_keys.json'), 'w') as f:
        json.dump({'keys': list(morph_d.keys()),
                   'wl_range': [min(morph.keys()), max(morph.keys())]}, f)

    syn_d = {}
    for wn in syntax:
        tids = np.array(list(syntax[wn].keys()), dtype=np.int16)
        cnts = np.array(list(syntax[wn].values()), dtype=np.int32)
        syn_d[str(wn)] = {'tids': tids, 'cnts': cnts, 'total': int(cnts.sum())}
    np.savez_compressed(os.path.join(CN_DIR, 'syntax_cache.npz'), **syn_d)
    with open(os.path.join(CN_DIR, 'syntax_keys.json'), 'w') as f:
        json.dump({'keys': list(syn_d.keys()), 'max_wn': max(syntax.keys())}, f)

    with open(os.path.join(CN_DIR, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'n_sentences': len(lines), 'n_tokens': n_tok, 'n_words': n_wrd,
            'vocab_size': V, 'text_id': 1,
            'n_transitions': int(sum(trans.values())),
            'n_unique_transitions': len(pairs),
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        }, f, ensure_ascii=False, indent=2)

    for fn in sorted(os.listdir(CN_DIR)):
        fp = os.path.join(CN_DIR, fn)
        if os.path.isfile(fp):
            print(f"    {fn}: {os.path.getsize(fp)/1024/1024:.1f} MB")

    # ─── 6. Merge heads_meta ───
    print("\n[6] Merging heads_meta...")
    with open(os.path.join(SAVE_V5, 'heads_meta.pkl'), 'rb') as f:
        wp = pickle.load(f)

    # Merge morph_logprob (arrays of shape V)
    # WP arrays store log(prob) per token; CN data is sparse (dicts).
    # We'll convert CN to V-sized arrays, average weighted by source counts.
    UNIFORM_LP = math.log(1.0 / V)  # -8.32

    def sparse_to_varr(logprob_dict, default=UNIFORM_LP):
        """Convert {tid: logprob} to V-sized array."""
        arr = np.full(V, default, dtype=np.float32)
        for tid, lp in logprob_dict.items():
            if 0 <= tid < V:
                arr[tid] = lp
        return arr

    merged_morph = {}
    all_wl = set(wp['morph_logprob'].keys()) | set(morph.keys())
    for wl in sorted(all_wl):
        wp_wl = wp['morph_logprob'].get(wl, {})
        cn_wl = morph.get(wl, {})
        all_pos = set(wp_wl.keys()) | set(cn_wl.keys())
        merged_morph[wl] = {}
        for pos in sorted(all_pos):
            wp_arr = wp_wl.get(pos, np.full(V, UNIFORM_LP, dtype=np.float32))
            # Build CN logprob array from sparse counts
            cn_counts = cn_wl.get(pos, {})
            cn_total = sum(cn_counts.values())
            if cn_total > 0:
                cn_arr = np.full(V, UNIFORM_LP, dtype=np.float32)
                for tid, cnt in cn_counts.items():
                    cn_arr[int(tid)] = math.log(cnt / cn_total)
            else:
                cn_arr = np.full(V, UNIFORM_LP, dtype=np.float32)
            # Weighted average: WP 2x, CN 1x
            merged_morph[wl][pos] = (2.0 * wp_arr + 1.0 * cn_arr) / 3.0

    cn_syn = {}
    for key, val in syn_d.items():
        wn = int(key)
        total = int(val['total'])
        arr = np.full(V, UNIFORM_LP, dtype=np.float32)
        if total > 0:
            for tid, cnt in zip(val['tids'], val['cnts']):
                arr[int(tid)] = math.log(cnt / total)
        cn_syn[wn] = arr

    merged_syn = {}
    all_wn = set(wp['syntax_logprob'].keys()) | set(cn_syn.keys())
    for wn in sorted(all_wn):
        wp_arr = wp['syntax_logprob'].get(wn, np.full(V, UNIFORM_LP, dtype=np.float32))
        cn_arr = cn_syn.get(wn, np.full(V, UNIFORM_LP, dtype=np.float32))
        merged_syn[wn] = (2.0 * wp_arr + 1.0 * cn_arr) / 3.0

    # Merge token_counts and CSR for concept scores
    wp_tc = np.load(os.path.join(WP_HIER, 'token_counts.npz'))['counts']
    wp_csr = load_npz(os.path.join(WP_HIER, 'transitions_csr.npz'))
    merged_csr = wp_csr + t_csr
    merged_csr.eliminate_zeros()
    merged_tc = wp_tc + tok_cnt

    # Concept scores: sqrt(count * n_next) normalized
    mc = float(max(1, int(merged_tc.max())))
    cs = np.zeros(V, dtype=np.float32)
    for tid in range(V):
        c = int(merged_tc[tid])
        if c > 0:
            nn = int(merged_csr.indptr[tid + 1] - merged_csr.indptr[tid])
            cs[tid] = min(1.0, math.sqrt(float(c) * float(max(1, nn))) / math.sqrt(mc))

    si = np.argsort(cs)[::-1]
    clusters = {}
    cs_size = max(1, V // 10)
    for ci in range(10):
        s = ci * cs_size
        e = min((ci + 1) * cs_size, V)
        clusters[str(ci)] = [int(t) for t in si[s:e].tolist() if merged_tc[t] > 0]

    # Keep existing trans_sim, contra_pairs, and ngram from WP
    ngram = dict(wp.get('ngram_sparse', {}))

    stats = dict(wp['stats'])
    stats['n_sentences'] = int(wp['stats']['n_sentences']) + len(lines)
    stats['n_tokens'] = int(merged_tc.sum())
    stats['n_words'] = int(merged_word_counts.sum()) if 'merged_word_counts' in dir() else 0
    stats['n_ngrams'] = len(ngram)

    merged = {
        'morph_logprob': dict(merged_morph),
        'syntax_logprob': dict(merged_syn),
        'trans_sim_sparse': wp['trans_sim_sparse'],
        'contra_pairs': wp['contra_pairs'],
        'concept_scores': cs,
        'concept_clusters': clusters,
        'ngram_sparse': ngram,
        'token_counts': merged_tc.astype(np.int32),
        'V': int(V),
        'stats': stats,
    }

    with open(MERGED_HEADS, 'wb') as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)

    hm_mb = os.path.getsize(MERGED_HEADS) / 1024 / 1024
    print(f"\n  heads_meta_merged.pkl: {hm_mb:.1f} MB")
    print(f"  morph: {len(merged_morph)} word lengths")
    print(f"  syntax: {len(merged_syn)} positions")
    print(f"  WP sentences: {wp['stats']['n_sentences']:,}")
    print(f"  CN sentences: {len(lines):,}")
    print(f"  Merged total: {stats['n_sentences']:,}")
    print(f"  Token counts summed: {merged_tc.sum():,}")

    # Verify critical structures
    print(f"\n  morph keys: {sorted(merged_morph.keys())[:10]}...")
    print(f"  syntax keys (first 20): {sorted(merged_syn.keys())[:20]}...")
    print(f"  trans_sim tokens: {len(merged['trans_sim_sparse'])}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
