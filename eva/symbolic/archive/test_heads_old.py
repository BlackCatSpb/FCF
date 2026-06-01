"""
test_heads.py — демонстрация data-driven heads на реальных данных.
"""
import sys, os
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import numpy as np
from coordinate_packer import CoordinatePacker
from eva.symbolic.heads_v5 import HeadsEnsemble
from eva.symbolic.bpe_tokenizer import BPEVocab

packer = CoordinatePacker()
cv = BPEVocab()
V = packer.V

DB = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\heads_db.pkl'
heads = HeadsEnsemble(DB)

print("="*70)
print("TEST: Heads scoring on real trajectories")
print("="*70)

# Load one trajectory
from eva.symbolic.trajectory_store import TrajectoryStore
store = TrajectoryStore()
store.load(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\trajectory_store_v5.pkl')

# Pick a short sentence
for idx in range(min(100, store.total_stored)):
    htraj = store.hierarchical[idx]
    L = len(htraj.ids)
    if 10 <= L <= 30:
        test_idx = idx
        break

htraj = store.hierarchical[test_idx]
traj = htraj.symbol_trajectory
tokens = htraj.ids
text = htraj.text
L = len(tokens)

print(f"\nSentence #{test_idx}:")
print(f"  Length: {L} tokens")
print(f"  Text: {text[:80]}...")
print(f"  Tokens: {tokens[:15]}...")

# ─── Step through the trajectory ───
print(f"\n{'='*70}")
print("STEP-BY-STEP: head scores vs. actual next token")
print("="*70)

errors = []
for t in range(L - 1):
    # Build context from current position
    info = packer.unpack_token(traj[t])
    actual_next = tokens[t + 1]
    
    flags = info['flags']
    is_special = (flags >> packer.F_SPECIAL) & 1
    is_word_start = (flags >> packer.F_WORD_START) & 1
    prev_token = tokens[t - 1] if t > 0 else None
    context_tokens = tokens[:t + 1]
    
    context = {
        'token_id': info['token_id'],
        'pos_in_word': info['pos_in_word'],
        'word_len': info['word_len'],
        'word_num': info['word_num'],
        'pos_in_sent': info['pos_in_sent'],
        'sent_len': info['sent_len'],
        'flags': flags,
        'prev_token_id': prev_token,
        'context_tokens': context_tokens,
    }
    
    # Score all tokens and find best
    scores = heads.score_all(context)
    predicted = int(np.argmax(scores))
    is_correct = predicted == actual_next
    
    if not is_correct:
        errors.append((t, predicted, actual_next, scores[actual_next] - scores[predicted]))
    
    # Get weights and per-head scores for actual next
    weights = heads.compute_weights(context)
    per_head = {}
    for name, h in heads.heads.items():
        per_head[name] = h.score(actual_next, context)
    
    # Show only key positions or errors
    if t < 5 or not is_correct or is_word_start:
        pred_text = cv.decode([predicted])
        actual_text = cv.decode([actual_next])
        tok_text = cv.decode([info['token_id']])
        
        flag_parts = []
        if is_word_start: flag_parts.append('★WORD_START')
        if (flags >> 2) & 1: flag_parts.append('SENT_START')
        if is_special: flag_parts.append('SPECIAL')
        flag_str = ' '.join(flag_parts) if flag_parts else 'inside'
        
        score_str = ' '.join([f"{name}={per_head[name]:+.2f}" for name in ['morph','syntax','transition','semantic','concept','contra']])
        
        mark = '✓' if is_correct else '✗'
        print(f"  [{t:3d}] tok='{tok_text:<8}' ({flag_str:>16}) "
              f"→ pred='{pred_text:<8}' act='{actual_text:<8}' {mark}")
        if not is_correct:
            print(f"        weights: {weights}")
            print(f"        per-head: {score_str}")
            gap = scores[actual_next] - scores[predicted]
            print(f"        gap actual-pred: {gap:.4f}")

print(f"\n  Errors: {len(errors)}/{L - 1} positions")
if errors:
    print(f"  Max gap (actual - pred): {max(-g for _,_,_,g in errors):.4f}")

# ─── Test: what does MorphHead predict at word_pos=0? ───
print(f"\n{'='*70}")
print("MORPH HEAD: top-5 tokens at pos=0 for word_len=3")
print("="*70)
context_pos0 = {
    'token_id': 0, 'pos_in_word': 0, 'word_len': 3,
    'word_num': 0, 'sent_len': 20,
    'flags': 1,  # F_WORD_START
    'prev_token_id': None, 'context_tokens': [],
}
top_morph = heads.heads['morph'].db.get('morph_dist', {}).get(3, {}).get(0, {})
top_tokens = sorted(top_morph.items(), key=lambda x: -x[1])[:10]
for tid, cnt in top_tokens:
    tok = cv.decode([tid])
    print(f"  '{tok}' ({tid:4d}): count={cnt}")

# ─── Test: what does SyntaxHead predict at word_num=0? ───
print(f"\n{'='*70}")
print("SYNTAX HEAD: top-10 word-start tokens at word_num=0")
print("="*70)
top_syntax = heads.heads['syntax'].db.get('syntax_dist', {}).get(0, {})
top_words = sorted(top_syntax.items(), key=lambda x: -x[1])[:10]
for tid, cnt in top_words:
    tok = cv.decode([tid])
    print(f"  '{tok}' ({tid:4d}): count={cnt}")

# ─── Test: transition head ───
print(f"\n{'='*70}")
print("TRANSITION HEAD: P(next | prev=SENT_OPEN(159))")
print("="*70)
probs = heads.db.get('trans_prob', np.zeros((V, V)))
prev = 159  # SENT_OPEN
top_trans = np.argsort(-probs[prev])[:10]
for tid in top_trans:
    if probs[prev, tid] > 0.001:
        tok = cv.decode([tid])
        print(f"  '{tok}' ({tid:4d}): P={probs[prev,tid]:.4f}")

# ─── Test: contradiction examples ───
print(f"\n{'='*70}")
print("CONTRADICTION: high trans_sim, P=0 pairs (first 10)")
print("="*70)
contra = heads.db.get('contra_pairs', [])
for ta, tb, s in contra[:10]:
    a_text = cv.decode([ta])
    b_text = cv.decode([tb])
    print(f"  '{a_text}' <-> '{b_text}': trans_sim={s:.4f}, P=0")

# ─── Test: concept scores ───
print(f"\n{'='*70}")
print("CONCEPT: top-10 highest concept scores (sparsest regions)")
print("="*70)
cs = heads.db.get('concept_scores', None)
if cs is not None:
    top_concept = np.argsort(-cs)[:10]
    for tid in top_concept:
        tok = cv.decode([tid])
        print(f"  '{tok}' ({tid:4d}): concept_score={cs[tid]:.4f}")

print(f"\n{'='*70}")
print("DONE")
