"""
Trace back WORD_CONT first tokens to old data to find root cause.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tokenizers import Tokenizer
from eva.symbolic.bpe_tokenizer import HierarchicalVocab

hv = HierarchicalVocab()
tok = Tokenizer.from_file('real_data/bpe_tokenizer.json')

# Load NEW data and find a sentence starting with WORD_CONT
new_ids = np.load('hierarchical_data/hierarchical.tokens.npy')
new_lens = np.load('hierarchical_data/hierarchical.lengths.npy')

ptr = 0
found = []
for i in range(len(new_lens)):
    L = int(new_lens[i])
    sent = new_ids[ptr:ptr+L].tolist()
    ptr += L
    if len(sent) >= 2:
        first = sent[1]
        tt = int(hv.token_type[first])
        if tt == 3:  # WORD_CONT
            found.append((i, sent[:8], hv.decode(sent)[:80]))
            if len(found) >= 3:
                break

print('Sentences starting with WORD_CONT:')
target_text = None
for idx, ids, text in found:
    print('  #{}: first={} text={!r}'.format(idx, ids[1], text))
    if target_text is None:
        target_text = text.strip()[:40]

# Find this text in OLD data
old_data = np.load('real_data/v5/hierarchical/sentences.npz')
old_tokens = old_data['tokens']
old_lens = old_data['token_lens']

BOUNDARY_IDS = {157, 158, 159, 160}

ptr = 0
found_old = None
for i in range(len(old_lens)):
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

    if target_text and target_text in text:
        found_old = (i, oids, text)
        break
    if i >= 200:
        break

if found_old:
    i, oids, text = found_old
    print('\nFound in old data #{}:'.format(i))
    print('  Text: {!r}'.format(text[:80]))
    print('  IDs: {}'.format(oids[:12]))

    # Encode from scratch with new encode()
    new_enc = hv.encode(text)
    print('  Fresh encode: {}'.format(new_enc))
    for tid in new_enc:
        print('    ID {:4d} type={:12s} decode={!r}'.format(tid, hv.type_name(tid), hv.decode([tid])))
else:
    print('\nText not found in first 200 old sentences')
