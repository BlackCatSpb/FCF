"""
Modular corpus builder — tokenize any text, build transition CSR + heads_meta.
Supports incremental merge from multiple sources.

Usage:
    python -c "from eva.symbolic.build_corpus import CorpusBuilder; b = CorpusBuilder(); b.build('corpus.txt', 'output_dir')"
"""
import os, math, json, pickle, time
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix, save_npz, load_npz

from eva.symbolic.bpe_tokenizer import HierarchicalVocab

V = 4101
UNIFORM_LP = math.log(1.0 / V)


class CorpusBuilder:
    def __init__(self):
        self.hv = HierarchicalVocab()

    def tokenize_file(self, path, skip_empty=True):
        """Tokenize a text file (one or many sentences). Returns list of token ID lists."""
        print(f'Tokenizing {path}...')
        all_ids = []
        n_lines = 0
        t0 = time.time()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if skip_empty and not line:
                    continue
                ids = self.hv.encode(line)
                if ids:
                    all_ids.append(ids)
                n_lines += 1
                if n_lines % 50000 == 0:
                    print(f'  {n_lines:,} lines, {len(all_ids):,} kept ({time.time()-t0:.1f}s)')
        print(f'  Done: {n_lines:,} lines → {len(all_ids):,} tokenized sequences ({time.time()-t0:.1f}s)')
        return all_ids

    def build_transitions(self, all_ids):
        """Build transition count CSR + token_counts from tokenized sequences."""
        print('Building transitions...')
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

        count_csr = csr_matrix((data, cols, indptr.copy()), shape=(V, V), dtype=np.int32)
        rs = np.maximum(np.array(count_csr.sum(axis=1)).flatten(), 1)
        lp_data = np.zeros(len(data), dtype=np.float32)
        for i in range(len(data)):
            lp_data[i] = math.log(data[i] / rs[rows[i]]) if data[i] > 0 else -23.0
        lp_csr = csr_matrix((lp_data, cols.copy(), indptr.copy()), shape=(V, V), dtype=np.float32)

        print(f'  {len(pairs):,} unique transitions, {int(tok_cnt.sum()):,} total tokens ({time.time()-t0:.1f}s)')
        return count_csr, lp_csr, tok_cnt

    def build_morph_syntax(self, all_ids):
        """Build morph (word_len×pos→tid counts) and syntax (pos_in_sent×tid counts)."""
        print('Building morph/syntax...')
        t0 = time.time()
        morph = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        syntax = defaultdict(lambda: defaultdict(int))

        for ids in all_ids:
            # Compute word structure from token types
            L = len(ids)
            word_starts = []
            cur_word_tokens = []
            for t, tid in enumerate(ids):
                tt = self.hv.token_type[tid]
                if tt == 2:
                    if cur_word_tokens:
                        word_starts.append(cur_word_tokens)
                    cur_word_tokens = [(t, tid)]
                elif tt == 3 and cur_word_tokens:
                    cur_word_tokens.append((t, tid))
                else:
                    if cur_word_tokens:
                        word_starts.append(cur_word_tokens)
                        cur_word_tokens = []
            if cur_word_tokens:
                word_starts.append(cur_word_tokens)

            for wi, tokens in enumerate(word_starts):
                wl = len(tokens)
                for pi, (abs_pos, tid) in enumerate(tokens):
                    morph[wl][pi][tid] += 1
                    syntax[wi][tid] += 1

        print(f'  morph: {sum(len(pos) for pos in morph.values())} positions, '
              f'syntax: {len(syntax)} word positions ({time.time()-t0:.1f}s)')
        return dict(morph), dict(syntax)

    def build_heads_meta(self, count_csr, lp_csr, tok_cnt, morph, syntax, name='corpus'):
        """Build heads_meta dict from raw statistics."""
        print('Building heads_meta...')
        t0 = time.time()

        # Morph logprob
        morph_lp = {}
        for wl in morph:
            morph_lp[wl] = {}
            for pos in morph[wl]:
                total = sum(morph[wl][pos].values())
                if total == 0:
                    continue
                arr = np.full(V, UNIFORM_LP, dtype=np.float32)
                for tid, cnt in morph[wl][pos].items():
                    arr[int(tid)] = math.log(cnt / total)
                morph_lp[wl][pos] = arr

        # Syntax logprob
        syn_lp = {}
        for wn in syntax:
            total = sum(syntax[wn].values())
            if total == 0:
                continue
            arr = np.full(V, UNIFORM_LP, dtype=np.float32)
            for tid, cnt in syntax[wn].items():
                arr[int(tid)] = math.log(cnt / total)
            syn_lp[wn] = arr

        # Concept scores: sqrt(count × n_neighbors)
        mc = float(max(1, int(tok_cnt.max())))
        cs = np.zeros(V, dtype=np.float32)
        for tid in range(V):
            c = int(tok_cnt[tid])
            if c > 0:
                nn = int(count_csr.indptr[tid + 1] - count_csr.indptr[tid])
                cs[tid] = min(1.0, math.sqrt(float(c) * float(max(1, nn))) / math.sqrt(mc))

        meta = {
            'V': int(V),
            'morph_logprob': morph_lp,
            'syntax_logprob': syn_lp,
            'trans_sim_sparse': {},
            'contra_pairs': [],
            'concept_scores': cs,
            'token_counts': tok_cnt.astype(np.int32),
        }
        print(f'  morph LPs: {sum(len(v) for v in morph_lp.values())}, '
              f'syntax LPs: {len(syn_lp)} ({time.time()-t0:.1f}s)')
        return meta

    def build(self, text_path, output_dir, name='corpus'):
        """Full pipeline: tokenize → transitions → morph/syntax → save."""
        os.makedirs(output_dir, exist_ok=True)

        all_ids = self.tokenize_file(text_path)
        count_csr, lp_csr, tok_cnt = self.build_transitions(all_ids)
        morph, syntax = self.build_morph_syntax(all_ids)
        meta = self.build_heads_meta(count_csr, lp_csr, tok_cnt, morph, syntax, name)

        save_npz(os.path.join(output_dir, 'transitions_csr.npz'), count_csr)
        save_npz(os.path.join(output_dir, 'log_prob_csr.npz'), lp_csr)
        np.savez_compressed(os.path.join(output_dir, 'token_counts.npz'), counts=tok_cnt)

        meta_path = os.path.join(output_dir, 'heads_meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

        stats = {
            'name': name, 'source': text_path, 'n_sentences': len(all_ids),
            'n_tokens': int(tok_cnt.sum()), 'n_unique_transitions': count_csr.nnz,
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(os.path.join(output_dir, 'stats.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        mb = sum(os.path.getsize(os.path.join(output_dir, f))
                 for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f)))
        print(f'\nSaved to {output_dir}/ ({mb/1024/1024:.1f} MB)')
        return meta

    @staticmethod
    def merge(metas, output_path, count_csrs=None, weights=None):
        """
        Merge multiple heads_meta dicts with optional weights.
        metas: list of heads_meta dicts
        weights: list of floats (default: equal)
        count_csrs: list of count CSR matrices (for combined concept_scores)
        """
        n = len(metas)
        weights = weights or [1.0] * n
        w = np.array(weights, dtype=np.float32)
        w = w / w.sum()

        print(f'Merging {n} corpora with weights {dict(zip(range(n), weights))}...')

        # Morph merge
        all_wl = set()
        for m in metas:
            all_wl |= set(m['morph_logprob'].keys())
        merged_morph = {}
        for wl in sorted(all_wl):
            merged_morph[wl] = {}
            all_pos = set()
            for m in metas:
                if wl in m['morph_logprob']:
                    all_pos |= set(m['morph_logprob'][wl].keys())
            for pos in sorted(all_pos):
                arr = np.zeros(V, dtype=np.float32)
                for mi, m in enumerate(metas):
                    if wl in m['morph_logprob'] and pos in m['morph_logprob'][wl]:
                        arr += w[mi] * m['morph_logprob'][wl][pos]
                    else:
                        arr += w[mi] * UNIFORM_LP
                merged_morph[wl][pos] = arr

        # Syntax merge
        all_wn = set()
        for m in metas:
            all_wn |= set(m['syntax_logprob'].keys())
        merged_syn = {}
        for wn in sorted(all_wn):
            arr = np.zeros(V, dtype=np.float32)
            for mi, m in enumerate(metas):
                if wn in m['syntax_logprob']:
                    arr += w[mi] * m['syntax_logprob'][wn]
                else:
                    arr += w[mi] * UNIFORM_LP
            merged_syn[wn] = arr

        # Token counts
        merged_tc = sum(w[mi] * m['token_counts'].astype(np.float32)
                        for mi, m in enumerate(metas)).astype(np.int32)

        # Concept scores from combined CSR
        if count_csrs:
            merged_csr = sum(csr * w[mi] for mi, csr in enumerate(count_csrs))
            merged_csr.eliminate_zeros()
            mc = float(max(1, int(merged_tc.max())))
            cs = np.zeros(V, dtype=np.float32)
            for tid in range(V):
                c = int(merged_tc[tid])
                if c > 0:
                    nn = int(merged_csr.indptr[tid + 1] - merged_csr.indptr[tid])
                    cs[tid] = min(1.0, math.sqrt(float(c) * float(max(1, nn))) / math.sqrt(mc))
        else:
            cs = metas[0]['concept_scores']
            merged_csr = None

        merged = {
            'V': int(V),
            'morph_logprob': dict(merged_morph),
            'syntax_logprob': dict(merged_syn),
            'trans_sim_sparse': metas[0].get('trans_sim_sparse', {}),
            'contra_pairs': metas[0].get('contra_pairs', []),
            'concept_scores': cs,
            'token_counts': merged_tc,
        }

        with open(output_path, 'wb') as f:
            pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Merged heads_meta saved to {output_path} ({os.path.getsize(output_path)/1024/1024:.1f} MB)')
        return merged, merged_csr
