"""
Compare OLD re-encoding with STORED new data.
Find matching sentences across sources.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tokenizers import Tokenizer
from eva.symbolic.bpe_tokenizer import HierarchicalVocab

hv = HierarchicalVocab()
tok = Tokenizer.from_file('real_data/bpe_tokenizer.json')
BOUNDARY_IDS = {157, 158, 159, 160}

# Load old hierarchical data
old_data = np.load('real_data/v5/hierarchical/sentences.npz')
old_tokens = old_data['tokens']
old_lens = old_data['token_lens']

# Re-encode a few old sentences
print("Re-encoding first 5 old sentences:")
ptr = 0
old_encodings = []
for i in range(5):
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
    
    bpe_ids = tok.encode(text).ids
    new_enc = [2] + bpe_ids + [3]
    old_encodings.append((text, new_enc))
    print(f"  OLD#{i}: {repr(text[:50])} -> enc={new_enc[:6]}...")

# Load NEW data and search for these encodings
print("\nSearching in STORED new data:")
new_ids = np.load('hierarchical_data/hierarchical.tokens.npy')
new_lens = np.load('hierarchical_data/hierarchical.lengths.npy')

ptr = 0
for si in range(min(500, len(new_lens))):
    L = int(new_lens[si])
    sent = new_ids[ptr:ptr+L].tolist()
    ptr += L
    
    for i, (text, enc) in enumerate(old_encodings):
        if len(sent) == len(enc) and sent[:3] == enc[:3]:
            text_new = hv.decode(sent)
            print(f"  FOUND OLD#{i} at NEW#{si}:")
            print(f"    OLD enc: {enc[:8]}...")
            print(f"    NEW enc: {sent[:8]}...")
            print(f"    Match: {sent == enc}")
            print(f"    Recovered: {repr(text)}")
            print(f"    Stored:    {repr(text_new[:60])}")
            if sent != enc:
                print(f"    DIFFERENCE at index {next(j for j in range(len(sent)) if sent[j] != enc[j])}")
            print()
            break
