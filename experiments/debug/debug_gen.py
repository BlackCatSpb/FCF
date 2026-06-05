"""
Step-by-step debug of generation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from eva.symbolic.heads import HeadsEnsemble
from eva.symbolic.bpe_tokenizer import HierarchicalVocab

hv = HierarchicalVocab()
meta_path = r"real_data/v6/heads_meta.pkl"
heads = HeadsEnsemble(meta_path, "real_data/v6")

V = 4101
EOS = 3
BOS = 2

# Step 1: Manually simulate first few steps
tokens = [BOS]
context_tokens = [0] * 30  # fixed-length window

print("=== Step-by-step generation (temperature=0.0) ===\n")

for step in range(10):
    # Build context dict
    ctx = {
        'token_id': tokens[-1],
        'prev_token_id': tokens[-2] if len(tokens) >= 2 else BOS,
        'pos_in_word': 0,
        'word_len': 5,
        'word_num': 0,
        'flags': 0,
        'pos_in_sent': len(tokens) - 1,
        'sent_len': 100,
        'context_tokens': context_tokens,
    }
    
    scores = heads.individual_scores(ctx)
    # Apply masks
    masked = scores.copy()
    for t in (0, 1, 2, 4, 5):
        masked[t] = -np.inf
    for t in range(4096, V):
        masked[t] = -np.inf
    # EOS only allowed after word_num >= 3
    ctx2 = ctx.copy()
    
    best = np.argmax(masked)
    text = hv.decode([best]) if best < 4096 else '?'
    
    print(f"  Step {step}: prev={tokens[-1]} ({hv.decode([tokens[-1]])[:5]!r}) "
          f"→ best={best} ({text[:5]!r})  "
          f"top5 IDs={np.argsort(masked)[-5:][::-1].tolist()}")
    
    # Check transition from prev to best
    prev = tokens[-1]
    if prev < 4096 and best < 4096:
        trans_prob = heads.log_prob_csr[prev, best]
        morph_prob = 0.0
        if 3 in heads.morph_logprob and 0 in heads.morph_logprob[3]:
            morph_prob = heads.morph_logprob[3][0][best]
        print(f"    trans={trans_prob:.2f} morph={morph_prob:.2f}")
    
    # Show top-5 from transition head for prev token
    prev_row = heads.log_prob_csr[prev].toarray()[0]
    trans_top5 = np.argsort(prev_row)[-5:][::-1]
    trans_texts = [repr(hv.decode([int(t)]))[:6] for t in trans_top5]
    print(f"    trans top-5 from {prev}: IDs={trans_top5.tolist()} texts={trans_texts}")
    
    tokens.append(best)
    
    if best == EOS:
        break
