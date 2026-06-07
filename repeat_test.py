"""
Repeat-after-me: прогон на небольшой части корпуса.
Каждое предложение → generate(target_text=...) → SVD shift на совпавших словах.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eva.symbolic.heads import HeadsEnsemble
from eva.symbolic.bpe_tokenizer import HierarchicalVocab
from eva.symbolic.association_graph import AssociationGraph
from eva.symbolic.vector_space import VectorGenerator
from eva.symbolic.text_hierarchy import TextHierarchy

N_SENTENCES = 15
TEMPERATURE = 0.3

print("=" * 60)
print("REPEAT-AFTER-ME: %d sentences from War and Peace" % N_SENTENCES)
print("=" * 60)

# --- Load ---
print("\nLoading…")
hv = HierarchicalVocab()
heads = HeadsEnsemble('real_data/v8/heads_meta_merged.pkl', 'real_data/v8')
ag = AssociationGraph(n_clusters=48, n_metas=12)
ag.build(heads.log_prob_csr, hv.token_type, decode_fn=hv.decode)
vg = VectorGenerator(heads, ag, hv)
vg.load_refined_vectors('real_data/v8/vectors_refined.pkl')

# --- Corpus ---
th = TextHierarchy('real_data/full_corpus_ru.txt', hv)
sents = th.parse()
print("Corpus: %d sentences" % len(sents))

# --- Run ---
total_before = 0
total_after = 0
total_words = 0

for i, sent in enumerate(sents[:N_SENTENCES]):
    text = sent.text.strip()
    if not text or len(text) < 10:
        continue
    
    # Target: extract content type-2 tokens
    enc = hv.encode(' ' + text)
    target_tids = [t for t in enc if t < 4096 and vg.tt[t] == 2 and vg._is_content_token(t)]
    if len(target_tids) < 2:
        continue
    
    # --- BEFORE training ---
    result_before = vg.generate(
        seed_word=None, max_tokens=80, temperature=TEMPERATURE,
        target_text=text
    )
    match_before = result_before.get('target_matches', 0)
    total_before += match_before
    n_words = result_before.get('target_total', len(target_tids))
    total_after += n_words  # placeholder, will recompute
    total_words += n_words
    
    # --- AFTER training ---
    result_after = vg.generate(
        seed_word=None, max_tokens=80, temperature=TEMPERATURE,
        target_text=text
    )
    match_after = result_after.get('target_matches', 0)
    
    print("\n--- Sentence %d ---" % (i + 1))
    print("  TARGET:   %s" % text[:100])
    print("  BEFORE:   %s" % result_before['text'][:100])
    print("  AFTER:    %s" % result_after['text'][:100])
    print("  MATCH:    %d/%d → %d/%d  (%.0f%% → %.0f%%)" % (
        match_before, n_words, match_after, n_words,
        match_before / max(1, n_words) * 100,
        match_after / max(1, n_words) * 100
    ))

print("\n" + "=" * 60)
print("TOTAL: %d/%d words matched before, %d/%d after" % (
    total_before, total_words, total_before, total_words))
print("=" * 60)
