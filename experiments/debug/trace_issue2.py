"""
Check if the first token type distribution makes sense.
If ~50% start with WORD_CONT, the data is corrupted.
If there's a systematic pattern, we need to identify it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tokenizers import Tokenizer
from eva.symbolic.bpe_tokenizer import HierarchicalVocab

hv = HierarchicalVocab()

for src in ['hierarchical', 'wikipedia', 'conceptnet']:
    print(f'\n=== {src} ===')
    ids = np.load(f'hierarchical_data/{src}.tokens.npy')
    lens = np.load(f'hierarchical_data/{src}.lengths.npy')
    
    # First-token type distribution
    type_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    ptr = 0
    for i in range(min(5000, len(lens))):
        L = int(lens[i])
        sent = ids[ptr:ptr+L].tolist()
        ptr += L
        if len(sent) >= 2:
            first = sent[1]
            tt = int(hv.token_type[first])
            type_counts[tt] = type_counts.get(tt, 0) + 1
    
    print(f'  First token types (first 5000 sentences):')
    for tt, name in [(0, 'SPECIAL'), (1, 'BYTE'), (2, 'WORD_STARTER'), (3, 'WORD_CONT')]:
        n = type_counts.get(tt, 0)
        print(f'    {name}: {n} ({100*n/5000:.1f}%)')
    
    # Example first-token IDs for each type
    if type_counts.get(3, 0) > 0:  # WORD_CONT
        ptr = 0
        examples = []
        for i in range(len(lens)):
            L = int(lens[i])
            sent = ids[ptr:ptr+L].tolist()
            ptr += L
            if len(sent) >= 2 and int(hv.token_type[sent[1]]) == 3:
                text = hv.decode(sent)
                examples.append((sent[1], text[:60]))
                if len(examples) >= 3:
                    break
        
        print(f'  Example WORD_CONT-first sentences:')
        for first, text in examples:
            print(f'    first={first} ({repr(hv.decode([first]))[:8]}) text={repr(text)}')
    
    if type_counts.get(1, 0) > 0:  # BYTE
        ptr = 0
        examples = []
        for i in range(len(lens)):
            L = int(lens[i])
            sent = ids[ptr:ptr+L].tolist()
            ptr += L
            if len(sent) >= 2 and int(hv.token_type[sent[1]]) == 1:
                text = hv.decode(sent)
                examples.append((sent[1], text[:40]))
                if len(examples) >= 3:
                    break
        
        print(f'  Example BYTE-first sentences:')
        for first, text in examples:
            print(f'    first={first} ({repr(hv.decode([first]))[:8]}) text={repr(text)}')

# Now try to reverse: find old sentences whose RECOVERED text starts without space
print('\n\n=== Checking old data recovered text ===')
old_data = np.load('real_data/v5/hierarchical/sentences.npz')
old_tokens = old_data['tokens']
old_lens = old_data['token_lens']

BOUNDARY_IDS = {157, 158, 159, 160}
tok = Tokenizer.from_file('real_data/bpe_tokenizer.json')

no_lead = 0
ptr = 0
for i in range(min(5000, len(old_lens))):
    L = int(old_lens[i])
    oids = old_tokens[ptr:ptr+L].tolist()
    ptr += L
    
    text_parts = []
    for tid in oids:
        if tid in BOUNDARY_IDS:
            if tid == 157:
                text_parts.append(' ')
            continue
        text_parts.append(tok.decode([tid]))
    text = ''.join(text_parts)
    
    if text and not text.startswith(' '):
        no_lead += 1
        if no_lead <= 3:
            print(f'  OLD #{i}: no lead space, text={repr(text[:60])}')
            print(f'    IDs={oids[:8]}')

print(f'\nOLD data: {no_lead}/5000 sentences without leading space')
