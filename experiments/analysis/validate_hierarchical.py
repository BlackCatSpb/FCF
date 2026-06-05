"""Validate new hierarchical heads_meta: byte, subword, word heads"""
import sys, os, pickle, math
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import numpy as np
from tokenizers import Tokenizer
from coordinate_packer import CoordinatePacker

META = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\heads_meta_hierarchical.pkl'
cp = CoordinatePacker()

with open(META, 'rb') as f:
    meta = pickle.load(f)

print("=== HIERARCHICAL HEADS VALIDATION ===")
print()
print("Available heads:", [k for k in meta.keys() if 'logprob' in k or 'csr' in k or 'vocab' in k])
print()

# 1. Byte head
bt = meta.get('byte_transition_logprob')
print("--- Byte Head ---")
print("  shape:", bt.shape if bt is not None else "NONE")
print("  dtype:", bt.dtype if bt is not None else "N/A")
nz = np.count_nonzero(bt > -14) if bt is not None else 0
print("  non-zero entries (prob > -14): %d / %d (%.4f%%)" % (nz, 256*256, nz/(256*256)*100))

# Check typical byte transitions — Cyrillic UTF-8 
# Cyrillic leading bytes: D0(208) D1(209), followed by 128-191
if bt is not None:
    print("  P(next | D0=208): top-5 bytes")
    top5 = np.argsort(-bt[208])[:5]
    for b in top5:
        print("    byte %d -> %.4f" % (int(b), float(bt[208, b])))
    print("  P(next | D1=209): top-5 bytes")
    top5 = np.argsort(-bt[209])[:5]
    for b in top5:
        print("    byte %d -> %.4f" % (int(b), float(bt[209, b])))

# 2. Subword heads (existing)
print("\n--- Subword Heads ---")
morph = meta.get('morph_logprob', {})
print("  morph: %d (wl,pos) pairs" % sum(len(v) for v in morph.values()))
syntax = meta.get('syntax_logprob', {})
print("  syntax: %d word positions" % len(syntax))

# 3. Word head
wt = meta.get('word_transition_csr')
wv = meta.get('word_vocab')
print("\n--- Word Head ---")
print("  CSR shape:", wt.shape if wt is not None else "NONE")
print("  vocab entries:", len(wv) if wv else 0)
if wt is not None:
    print("  non-zero entries: %d / %d (%.6f%%)" % (wt.nnz, wt.shape[0]*wt.shape[1], wt.nnz/(wt.shape[0]*wt.shape[1])*100))
if wv:
    # Show top words
    from collections import Counter
    rev = {v: k for k, v in wv.items()}
    print("  sample words:", [rev[i] for i in range(min(5, len(rev)))])
    
print("\n=== SELF-TEST: HeadsEnsemble ===")
from eva.symbolic.heads import HeadsEnsemble

# Use hierarchical meta
meta_dir = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5'
csr_path = os.path.join(meta_dir, 'hierarchical')
if not os.path.exists(os.path.join(csr_path, 'log_prob_csr.npz')):
    csr_path = meta_dir

try:
    heads = HeadsEnsemble(META, csr_path=csr_path)
    print("  HeadsEnsemble loaded OK")
    print("  V =", heads.V)
    print("  byte_transition:", heads.byte_transition.shape if heads.byte_transition is not None else None)
    print("  word_transition_csr:", heads.word_transition_csr.shape if heads.word_transition_csr is not None else None)
    print("  Default weights:", heads.default_weights)
    
    # Test scoring with byte context
    byte_ctx = {
        'prev_token_id': cp.byte_id(208),  # D0 (Cyrillic lead)
        'pos_in_word': 0, 'word_len': 1, 'word_num': 0,
        'pos_in_sent': 0, 'sent_len': 5,
        'context_tokens': [], 'flags': 0,
    }
    byte_scores = heads.score_bytes(byte_ctx)
    best_byte = int(np.argmax(byte_scores))
    print("  Byte context [D0]: best next byte=%d (%.3f) [UTF-8 byte]" % (best_byte, float(byte_scores[best_byte])))
    
    # Test scoring with subword context
    sw_ctx = {
        'prev_token_id': cp.subword_id(334),  # 'на'
        'pos_in_word': 0, 'word_len': 1, 'word_num': 0,
        'pos_in_sent': 0, 'sent_len': 5,
        'context_tokens': [], 'flags': 1,  # word_start
    }
    scores2 = heads.score_subword(sw_ctx)
    best2 = int(np.argmax(scores2))
    # Get text for best — using raw BPE tokenizer
    from eva.symbolic.bpe_tokenizer import BPEVocab
    bpev = BPEVocab()
    from eva.symbolic.heads import id_value
    best_text = bpev.tokenizer.decode([id_value(best2)])
    print("  Subword context [на]: best subword=%d (%.3f) [%s]" % (best2, float(scores2[best2]), best_text))
    
    # Test word context
    if wv and wt is not None:
        first_word_id = 0  # most common word 'в'
        word_ctx = {
            'prev_token_id': cp.word_id(first_word_id),
            'pos_in_word': 0, 'word_len': 1, 'word_num': 0,
            'pos_in_sent': 0, 'sent_len': 5,
            'context_tokens': [], 'flags': 0,
        }
        word_scores = heads.score_words(word_ctx)
        best_word = int(np.argmax(word_scores))
        rev = {v: k for k, v in wv.items()}
        print("  Word context [%s]: best next word=%d (%.3f) [%s]" % (rev[0], best_word, float(word_scores[best_word]), rev.get(best_word, '?')))
        
    print("\n  OK: All heads working with hierarchical IDs")
    
except Exception as e:
    print("  ERROR:", e)
    import traceback
    traceback.print_exc()
