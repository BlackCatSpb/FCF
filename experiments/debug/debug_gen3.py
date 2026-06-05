"""
Detailed trace of generation context at each step.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from eva.symbolic.heads import HeadsEnsemble
from eva.symbolic.bpe_tokenizer import HierarchicalVocab

hv = HierarchicalVocab()
heads = HeadsEnsemble("real_data/v6/heads_meta.pkl", "real_data/v6")
V = heads.V

BOS, EOS = 2, 3
BLOCKED = {0, 1, 2, 4, 5} | set(range(4096, V))

# Manually simulate generation
tokens = [BOS]
print("Step-by-step generation trace")
print("="*60)

for step in range(10):
    # Exactly what generation_loop does:
    meta = hv.metadata_from_ids(tokens)
    ctx = dict(meta[-1])
    ctx['context_tokens'] = tokens[-4:-1] if len(tokens) > 1 else tokens
    word_num = ctx['word_num']

    # Weights
    w = np.array([1.0, 1.0, 2.0, 0.5, 0.2, 0.5], dtype=np.float32)
    flags = ctx['flags']
    piw = ctx['pos_in_word']
    wl = ctx['word_len']
    is_special = (flags >> 5) & 1
    is_word_start = (flags >> 0) & 1
    is_word_end = (flags >> 1) & 1

    if is_special:
        w = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float32)
    elif is_word_start:
        w = np.array([0.5, 3.0, 1.0, 0.5, 0.2, 0.5], dtype=np.float32)
    elif is_word_end:
        w = np.array([0.5, 1.0, 3.0, 1.0, 0.2, 0.5], dtype=np.float32)
    elif piw > 0 and wl > 2:
        frac = piw / max(wl, 1)
        if 0.2 < frac < 0.8:
            w = np.array([4.0, 1.0, 0.5, 0.5, 0.2, 0.5], dtype=np.float32)

    # Scores
    head_scores = heads.individual_scores(ctx)
    scores = np.dot(w, head_scores)
    for t in BLOCKED:
        scores[t] = -np.inf
    if word_num < 3:
        scores[EOS] = -np.inf

    best = int(np.argmax(scores))
    best_text = repr(hv.decode([best]))[:8]

    # Show top-5
    top5 = np.argsort(scores)[-5:][::-1]
    top_texts = [f"{t}({repr(hv.decode([t]))[:6]})" for t in top5]

    print(f"\nStep {step}: seq_len={len(tokens)}")
    print(f"  Last: id={tokens[-1]}({repr(hv.decode([tokens[-1]]))[:6]})")
    print(f"  Ctx: piw={piw} wl={wl} wn={word_num} flags={flags:04b} (start={is_word_start} end={is_word_end} special={is_special})")
    print(f"  Weights: morph={w[0]:.1f} syntax={w[1]:.1f} trans={w[2]:.1f} sem={w[3]:.1f} concept={w[4]:.1f} contra={w[5]:.1f}")
    print(f"  Best: {best} ({best_text}) score={scores[best]:.3f}")
    print(f"  Top-5: {top_texts}")

    # Show transition from prev
    prev = tokens[-1]
    if prev < 4096 and best < 4096:
        trans = heads.log_prob_csr[prev, best]
        print(f"  trans({prev}→{best}) = {trans:.2f}")
        # All transitions from prev
        prev_row = heads.log_prob_csr[prev].toarray()[0]
        trans_top = np.argsort(prev_row)[-5:][::-1]
        trans_texts = [(t, repr(hv.decode([t]))[:6]) for t in trans_top]
        print(f"  trans from {prev} top-5: {trans_texts}")

    tokens.append(best)
    if best == EOS:
        break
