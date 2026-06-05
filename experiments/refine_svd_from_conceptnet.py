"""
Refine SVD vectors using ConceptNet semantic relations.
Parse conceptnet_ru.txt, find word→tid mapping, apply vector shifts.
"""
import sys, os, re, math, pickle
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import numpy as np
from collections import defaultdict, Counter
from eva.symbolic.bpe_tokenizer import HierarchicalVocab
from eva.symbolic.heads import HeadsEnsemble
from eva.symbolic.association_graph import AssociationGraph
from eva.symbolic.vector_space import VectorGenerator

CN_PATH = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\conceptnet\conceptnet_ru.txt'

REL_MAP = {
    'это': 'is_a',
    'то же, что': 'synonym',
    'форма слова': 'form_of',
    'противоположно': 'antonym',
    'отличается от': 'distinct_from',
    'похоже на': 'similar_to',
    'часть': 'part_of',
    'способ': 'manner_of',
    'связан с': 'related_to',
    'происходит от': 'derived_from',
    'находится в': 'located_at',
    'используется для': 'used_for',
    'может': 'capable_of',
    'хочет': 'desires',
    'имеет': 'has_a',
    'сделан из': 'made_of',
    'вызывает': 'causes',
    'мотивирован': 'motivated_by_goal',
    'затрудняется': 'obstructed_by',
    'символизирует': 'symbol_of',
    'не хочет': 'not_desires',
    'не': 'not_has_property',
    'относится к': 'related_to',
}
RU_RELS = sorted(REL_MAP.keys(), key=len, reverse=True)  # longest first for prefix match

PULL_RELS = {'is_a', 'synonym', 'similar_to', 'form_of', 'derived_from',
             'related_to', 'part_of', 'manner_of', 'used_for', 'located_at',
             'capable_of', 'desires', 'has_a', 'made_of', 'causes',
             'motivated_by_goal', 'symbol_of'}
PUSH_RELS = {'antonym', 'distinct_from'}


def parse_conceptnet(path):
    triples = []
    stats = Counter()
    rel_pairs = defaultdict(list)
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Формат: Слово — отношение Слово.
            m = re.match(r'^(.+?)\s*[—–-]\s+(.+?)\.\s*$', line)
            if not m:
                stats['no_match'] += 1
                continue
            start = m.group(1).strip().lower()
            rest = m.group(2).strip()
            
            matched = False
            for ru_rel in RU_RELS:
                if rest.startswith(ru_rel + ' '):
                    end = rest[len(ru_rel)+1:].strip().lower()
                    en_rel = REL_MAP[ru_rel]
                    triples.append((start, en_rel, end))
                    rel_pairs[en_rel].append((start, end))
                    stats[en_rel] += 1
                    matched = True
                    break
            if not matched:
                stats['unparsed_rest'] += 1
    
    print(f'Parsed {len(triples)} triples')
    for k, v in stats.most_common(20):
        print(f'  {k}: {v}')
    return triples, rel_pairs


def word_to_tid(hv, word):
    encoded = hv.encode(' ' + word)
    for t in encoded:
        if t < 4096 and hv.token_type[t] == 2:
            if hv.decode([t]).strip().lower() == word:
                return t
    encoded = hv.encode(word)
    for t in encoded:
        if t < 4096 and hv.token_type[t] == 2:
            if hv.decode([t]).strip().lower() == word:
                return t
    return None


def refine_vectors(vs, rel_pairs, hv, lr=0.12, epochs=8):
    vectors = vs.token_vectors
    used_pairs = 0
    
    for epoch in range(epochs):
        epoch_lr = lr * (1.0 - epoch / epochs) * 0.5 ** epoch
        changes = 0
        epoch_pairs = 0
        
        for rel, pairs in rel_pairs.items():
            if rel in PUSH_RELS:
                direction = -1.0
            elif rel in PULL_RELS:
                direction = 1.0
            else:
                continue
            
            for start_word, end_word in pairs:
                tid_a = word_to_tid(hv, start_word)
                tid_b = word_to_tid(hv, end_word)
                if tid_a is None or tid_b is None:
                    continue
                va = vectors.get(tid_a)
                vb = vectors.get(tid_b)
                if va is None or vb is None:
                    continue
                epoch_pairs += 1
                
                diff = vb - va
                d = np.linalg.norm(diff)
                if d < 1e-10:
                    continue
                
                pull = epoch_lr * direction
                vectors[tid_a] = va + pull * diff
                vectors[tid_b] = vb - pull * diff
                changes += 1
        
        used_pairs = epoch_pairs
        print(f'  epoch {epoch+1}: lr={epoch_lr:.5f}, changes={changes}, pairs={epoch_pairs}')
    
    # Renormalize
    for tid in list(vectors.keys()):
        n = np.linalg.norm(vectors[tid])
        if n > 1e-10:
            vectors[tid] = vectors[tid] / n
    
    print(f'Total: {used_pairs} concept pairs applied')
    return used_pairs


def test(vs, hv, label):
    print(f'\n=== {label} ===')
    pairs = [
        ('армия', 'войско', 'synonym'),
        ('армия', 'армии', 'form_of'),
        ('князь', 'дворянин', 'is_a'),
        ('князь', 'сказал', 'cooccurrence'),
        ('человек', 'люди', 'form_of'),
        ('большой', 'маленький', 'antonym'),
        ('сказал', 'говорить', 'synonym'),
        ('сказал', 'говорил', 'form_of'),
        ('брат', 'друг', 'synonym'),
        ('брат', 'сестра', 'cooccurrence'),
        ('бог', 'господь', 'synonym'),
        ('смерть', 'жизнь', 'antonym'),
        ('дом', 'здание', 'is_a'),
        ('дом', 'комната', 'part_of'),
        ('дело', 'работа', 'related_to'),
    ]
    for a, b, label in pairs:
        ta, tb = word_to_tid(hv, a), word_to_tid(hv, b)
        if ta is not None and tb is not None and vs.has_vector(ta) and vs.has_vector(tb):
            sim = vs.similarity(ta, tb)
            print(f'  sim({a:12s}, {b:12s}) [{label:12s}]: {sim:.4f}')
        else:
            print(f'  sim({a:12s}, {b:12s}) [{label:12s}]: MISSING')
    
    for word in ['князь', 'сказал', 'армия', 'бог', 'человек']:
        tid = word_to_tid(hv, word)
        if tid and vs.has_vector(tid):
            top = vs.topk_similar(tid, k=5)
            print(f'\n  Top-5 near "{word}":')
            for t, s in top:
                print(f'    {hv.decode([t]).strip():15s} {s:.4f}')


if __name__ == '__main__':
    hv = HierarchicalVocab()
    heads = HeadsEnsemble(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v8\heads_meta.pkl',
                          r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v8')
    ag = AssociationGraph(n_clusters=48, n_metas=12)
    ag.build(heads.log_prob_csr, hv.token_type, decode_fn=hv.decode)
    
    triples, rel_pairs = parse_conceptnet(CN_PATH)
    
    # Report per-relation overlap with our vocab
    print('\nOverlap with our BPE vocab:')
    for rel, pairs in sorted(rel_pairs.items(), key=lambda x: -len(x[1])):
        n_found = 0
        for s, e in pairs:
            if word_to_tid(hv, s) is not None and word_to_tid(hv, e) is not None:
                n_found += 1
        print(f'  {rel:15s}: {len(pairs):6d} total, {n_found:5d} both in vocab')
    
    vg = VectorGenerator(heads, ag, hv)
    vs = vg.vs
    
    vectors_before = {tid: v.copy() for tid, v in vs.token_vectors.items()}
    test(vs, hv, 'BEFORE')
    
    print('\n--- Refinement ---')
    refine_vectors(vs, rel_pairs, hv, lr=0.12, epochs=6)
    
    test(vs, hv, 'AFTER')
    
    # Save
    save_path = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v8\vectors_refined.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(vs.token_vectors, f)
    print(f'\nSaved to {save_path}')
    
    # Quick generation test
    print('\n--- Generation test after refinement ---')
    for seed in ['сказал', 'был']:
        for run in range(2):
            result = vg.generate(max_tokens=40, seed_word=seed, temperature=0.5)
            print(f'[{seed} #{run}] {result["text"]}')
