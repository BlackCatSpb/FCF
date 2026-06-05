"""
Count first token types in data.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from eva.symbolic.bpe_tokenizer import HierarchicalVocab

hv = HierarchicalVocab()

for src in ['hierarchical', 'wikipedia', 'conceptnet']:
    ids = np.load(f'hierarchical_data/{src}.tokens.npy')
    lens = np.load(f'hierarchical_data/{src}.lengths.npy')
    
    types = {0: 0, 1: 0, 2: 0, 3: 0}
    first_ids = {}
    ptr = 0
    for i in range(len(lens)):
        L = int(lens[i])
        sent = ids[ptr:ptr+L]
        ptr += L
        if len(sent) >= 2:
            first = int(sent[1])
            tt = int(hv.token_type[first])
            types[tt] = types.get(tt, 0) + 1
            first_ids[first] = first_ids.get(first, 0) + 1
    
    total = sum(types.values())
    print(f'{src}:')
    for tt, name in [(0, 'SPECIAL'), (1, 'BYTE'), (2, 'WORD_STARTER'), (3, 'WORD_CONT')]:
        n = types.get(tt, 0)
        print(f'  {name}: {n} ({100*n/total:.1f}%)')
    
    # Most common first IDs for non-WORD_STARTER
    non_ws = {k: v for k, v in first_ids.items() if int(hv.token_type[k]) != 2}
    top = sorted(non_ws.items(), key=lambda x: -x[1])[:5]
    if top:
        print(f'  Most common non-WS first IDs:')
        for tid, cnt in top:
            print(f'    {tid:4d} ({repr(hv.decode([tid])):10s} {hv.type_name(tid):12s}): {cnt}')
    print()
