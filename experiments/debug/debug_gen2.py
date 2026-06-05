"""
Debug generation: per-token type scoring and morph distributions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from eva.symbolic.heads import HeadsEnsemble
from eva.symbolic.bpe_tokenizer import HierarchicalVocab

hv = HierarchicalVocab()
heads = HeadsEnsemble("real_data/v6/heads_meta.pkl", "real_data/v6")
V = heads.V

# Check what morph word lengths exist
print("Morph word lengths:", sorted(heads.morph_logprob.keys()))
print()

# For a few word lengths, show unique pos_in_word values
for wl in sorted(heads.morph_logprob.keys())[:5]:
    piw_keys = sorted(heads.morph_logprob[wl].keys())
    print(f"  morph[wl={wl}] piw={piw_keys}")
    piw0 = heads.morph_logprob[wl].get(0)
    if piw0 is not None:
        top5 = np.argsort(piw0)[-5:][::-1]
        top_texts = [repr(hv.decode([int(t)]))[:10] for t in top5]
        print(f"    piw=0 top-5: {list(zip(top5.tolist(), top_texts))}")
    piw_mid = heads.morph_logprob[wl].get(2)
    if piw_mid is not None:
        top5 = np.argsort(piw_mid)[-5:][::-1]
        top_texts = [repr(hv.decode([int(t)]))[:10] for t in top5]
        print(f"    piw=2 top-5: {list(zip(top5.tolist(), top_texts))}")

print()

# Simulate: what should metadata be for NEXT token at position L?
tokens = [2]  # BOS
print(f"Sequence: {tokens}")
print(f"Last token: id={tokens[-1]} text={repr(hv.decode([tokens[-1]]))} type={hv.type_name(tokens[-1])}")

# For position L (next token), we need context
# After BOS: next token is always a WORD_STARTER
# So: pos_in_word=0, word_num=0, word_len=10 (default max)

ctx = {
    'token_id': 0,  # unknown
    'prev_token_id': tokens[-1],
    'pos_in_word': 0,
    'word_len': 10,
    'word_num': 0,
    'flags': 1,  # word_start
    'pos_in_sent': 1,  # position of next token
    'sent_len': 100,
    'context_tokens': tokens[-4:-1] if len(tokens) > 1 else tokens,
}
print(f"Context for next token: piw={ctx['pos_in_word']}, wl={ctx['word_len']}, wn={ctx['word_num']}, flags={ctx['flags']}")

scores = heads.score_all(ctx)
masked = scores.copy()
for t in range(4096, V):
    masked[t] = -np.inf
for t in (0,1,2,4,5):
    masked[t] = -np.inf
top8 = np.argsort(masked)[-8:][::-1]
top_texts = [repr(hv.decode([int(t)]))[:10] for t in top8]
print(f"Top-8 tokens: {list(zip(top8, top_texts))}")
print(f"Top-8 are starter-types: {[hv.type_name(t) == 'WORD_STARTER' for t in top8]}")
print()

# Now simulate next step: what if we pick ID 1602 (WORD_STARTER)?
tokens.append(1602)
print(f"Sequence: {tokens}")
print(f"Last token: id={tokens[-1]} text={repr(hv.decode([tokens[-1]]))} type={hv.type_name(tokens[-1])}")

# For next position, last token is WORD_STARTER (pos_in_word=0)
# If next token is WORD_CONT: pos_in_word=1, wl=10, wn=0
# If next token is WORD_STARTER: pos_in_word=0, wl=10, wn=1

print("\nOption A: continue current word (pos_in_word=1, word_num=0)")
ctx2a = ctx.copy()
ctx2a.update({
    'prev_token_id': tokens[-1],
    'pos_in_word': 1,
    'word_len': 10,
    'word_num': 0,
    'flags': 0,  # not a word start
    'pos_in_sent': 2,
    'context_tokens': tokens[-4:-1] if len(tokens) > 1 else tokens,
})
scores_a = heads.score_all(ctx2a)
masked_a = scores_a.copy()
for t in range(4096, V): masked_a[t] = -np.inf
for t in (0,1,2,4,5): masked_a[t] = -np.inf
top8_a = np.argsort(masked_a)[-8:][::-1]
top_types = [hv.type_name(t) for t in top8_a]
print(f"  Top-8 IDs: {top8_a.tolist()}")
print(f"  Top-8 types: {top_types}")
print()

print("Option B: start new word (pos_in_word=0, word_num=1)")
ctx2b = ctx.copy()
ctx2b.update({
    'prev_token_id': tokens[-1],
    'pos_in_word': 0,
    'word_len': 10,
    'word_num': 1,
    'flags': 1,  # word_start
    'pos_in_sent': 2,
    'context_tokens': tokens[-4:-1] if len(tokens) > 1 else tokens,
})
scores_b = heads.score_all(ctx2b)
masked_b = scores_b.copy()
for t in range(4096, V): masked_b[t] = -np.inf
for t in (0,1,2,4,5): masked_b[t] = -np.inf
top8_b = np.argsort(masked_b)[-8:][::-1]
top_types_b = [hv.type_name(t) for t in top8_b]
print(f"  Top-8 IDs: {top8_b.tolist()}")
print(f"  Top-8 types: {top_types_b}")
print()

# Best token in each option
best_a = int(np.argmax(masked_a))
best_b = int(np.argmax(masked_b))
print(f"Option A best: {best_a} ({repr(hv.decode([best_a]))[:10]}) type={hv.type_name(best_a)} score={float(masked_a[best_a]):.2f}")
print(f"Option B best: {best_b} ({repr(hv.decode([best_b]))[:10]}) type={hv.type_name(best_b)} score={float(masked_b[best_b]):.2f}")
