"""Clean rebuild of ALL databases from full_corpus_ru.txt.
No ConceptNet, no external NLP. Everything from corpus statistics."""
import os, sys, json, pickle, shutil, math, time
import numpy as np
from scipy.sparse import csr_matrix, save_npz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eva.symbolic.bpe_tokenizer import HierarchicalVocab
from eva.symbolic.auto_config import AutoConfig
from eva.symbolic.build_corpus import CorpusBuilder
from eva.symbolic.heads import HeadsEnsemble
from eva.symbolic.association_graph import AssociationGraph
from eva.symbolic.vector_space import VectorGenerator
from eva.symbolic.text_hierarchy import TextHierarchy

REAL_DATA = os.path.join(os.path.dirname(__file__), '..', 'real_data')
CORPUS_PATH = os.path.join(REAL_DATA, 'full_corpus_ru.txt')

# ─────────────────────────────────────────────
# 1. Clean old artifacts
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Cleaning old databases")
print("=" * 60)
kept = {'full_corpus_ru.txt', 'conceptnet'}
for f in list(os.listdir(REAL_DATA)):
    path = os.path.join(REAL_DATA, f)
    if f in kept:
        continue
    if os.path.isfile(path):
        os.remove(path)
        print(f"  deleted {f}")
    elif os.path.isdir(path):
        shutil.rmtree(path)
        print(f"  deleted {f}/")

NEW_VOCAB_SIZE = 8192

# ─────────────────────────────────────────────
# 2. Build BPE tokenizer
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Building BPE tokenizer (vocab_size=%d)" % NEW_VOCAB_SIZE)
print("=" * 60)
bpe_path = os.path.join(REAL_DATA, 'bpe_tokenizer.json')
if os.path.exists(bpe_path):
    os.remove(bpe_path)
    print("  removed old tokenizer")
from eva.symbolic.bpe_tokenizer import train_bpe
print("  training new BPE tokenizer...")
train_bpe(vocab_size=NEW_VOCAB_SIZE, save_path=bpe_path)
hv = HierarchicalVocab()
print(f"  vocab_size={hv.vocab_size}, word_starter_tokens={int((hv.token_type==2).sum())}, continuation_tokens={int((hv.token_type==3).sum())}")

# ─────────────────────────────────────────────
# 3. Build corpus statistics (transitions + heads_meta)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Building corpus statistics (V=%d)" % hv.vocab_size)
print("=" * 60)
cb = CorpusBuilder(vocab_size=hv.vocab_size)
cb.hv = hv  # use our freshly trained BPE

meta = cb.build(CORPUS_PATH, REAL_DATA, name='corpus')
print(f"  token V={meta['V']}, morph keys={len(meta['morph_logprob'])}, syntax keys={len(meta['syntax_logprob'])}")

# Load back transitions for graph building
from scipy.sparse import load_npz
lp_csr = load_npz(os.path.join(REAL_DATA, 'log_prob_csr.npz'))
count_csr = load_npz(os.path.join(REAL_DATA, 'transitions_csr.npz'))
print(f"  log_prob_csr: {lp_csr.shape}, nnz={lp_csr.nnz}")

# ─────────────────────────────────────────────
# 4. AutoConfig from corpus
# ─────────────────────────────────────────────
# Build TextHierarchy once (used for config + training)
print("\n" + "=" * 60)
print("STEP 4: Parsing corpus hierarchy + AutoConfig")
print("=" * 60)
th = TextHierarchy(CORPUS_PATH, hv)
th.parse()
print(f"  {len(th.sentences)} sentences, {len(th.chapters)} chapters, {len(th.volumes)} volumes")

config = AutoConfig.from_corpus(th, hv)
# Override with actual vocab size
config.vocab_size = hv.vocab_size
config.bpe_limit = hv.vocab_size
config.svd_dim = 256
config.svd_epochs = 3
config.population_mode = True
config.hdbscan_min_cluster_ratio = 0.005  # smaller → more clusters
config.target_boost = 15.0

# Save initial config
config.save(os.path.join(REAL_DATA, 'config.json'))
print(f"  svd_dim={config.svd_dim}, svd_epochs={config.svd_epochs}")

# ─────────────────────────────────────────────
# 5. AssociationGraph (SVD + HDBSCAN + Louvain)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: AssociationGraph")
print("=" * 60)
ag = AssociationGraph(config=config)
ag.build(lp_csr, hv.token_type, decode_fn=hv.decode)
print(f"  starter_embeddings: {ag.starter_embeddings.shape}")
print(f"  concepts: {len(ag.cid_to_tids)}, metas: {len(ag.mid_to_cids)}")

ag_path = os.path.join(REAL_DATA, 'assoc_graph')
ag.save(ag_path)
print(f"  saved assoc_graph ({len(ag.cid_to_tids)} concepts, {len(ag.mid_to_cids)} metas)")

# ─────────────────────────────────────────────
# 6. HeadsEnsemble + VectorGenerator
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: VectorGenerator initialization")
print("=" * 60)
meta_path = os.path.join(REAL_DATA, 'heads_meta.pkl')
heads = HeadsEnsemble(meta_path, REAL_DATA, config=config)
vg = VectorGenerator(heads, ag, hv, config=config)

# Save trained vectors as initial base (no ConceptNet refinement)
vg._trained_vectors = {tid: v.copy() for tid, v in vg.vs.token_vectors.items()}
print(f"  vectors: {len(vg._trained_vectors)} tokens @ {vg.vs.dim}D")

# ─────────────────────────────────────────────
# 7. Stage2 config (needs ag + vg)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Stage2 config calibration")
print("=" * 60)
config = config.stage2(ag, vg, th, hv)
config.save(os.path.join(REAL_DATA, 'config.json'))
print(f"  calibrated: n_clusters={config.n_clusters}, n_metas={config.n_metas}")

# ─────────────────────────────────────────────
# 8. Save rebuild metadata
# ─────────────────────────────────────────────
metadata = {
    'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    'source': 'full_corpus_ru.txt',
    'external_nlp': False,
    'conceptnet': False,
    'vocab_size': hv.vocab_size,
    'svd_dim': config.svd_dim,
    'n_starters': len(ag.starter_list),
    'n_concepts': len(ag.cid_to_tids),
    'n_metas': len(ag.mid_to_cids),
    'transitions_nnz': int(lp_csr.nnz),
    'files': {
        'bpe_tokenizer.json': os.path.getsize(bpe_path),
        'heads_meta.pkl': os.path.getsize(meta_path),
        'assoc_graph.pkl': os.path.getsize(os.path.join(REAL_DATA, 'assoc_graph.pkl')),
        'log_prob_csr.npz': os.path.getsize(os.path.join(REAL_DATA, 'log_prob_csr.npz')),
    }
}
with open(os.path.join(REAL_DATA, 'build_metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(f"\nBuild metadata saved")

# ─────────────────────────────────────────────
# 9. Minimal training test
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Training test (3 sentences, 3 epochs)")
print("=" * 60)

train_sents = []
for sent in th.sentences[:5]:
    text = sent.text.strip()
    if not text or text.isspace():
        continue
    enc = hv.encode(' ' + text)
    tids = [t for t in enc if t < config.bpe_limit and vg.tt[t] == 2 and vg._is_content_token(t)]
    if len(tids) < 2:
        continue
    train_sents.append({'text': text, 'tids': tids, 'fw': hv.decode([tids[0]]).strip()})

print(f"Training on {len(train_sents)} sentences:")
for s in train_sents:
    print(f"  [{s['fw']}] {s['text'][:60]}")

n_epochs = 3
for epoch in range(n_epochs):
    vg.set_epoch(epoch)
    vg.reset_momentum()

    for i, sent in enumerate(train_sents):
        for rep in range(5):
            r = vg.generate(seed_word=sent['fw'], target_text=sent['text'],
                            max_tokens=config.context_window, temperature=0.3,
                            training_mode=True)

    config.target_boost = 0.0
    total_matches = 0
    total_target = 0
    for i, sent in enumerate(train_sents):
        r = vg.generate(seed_word=sent['fw'], max_tokens=config.context_window,
                        temperature=0.0, training_mode=False)
        gen_text = r['text']
        gen_tids = [t for t in hv.encode(' ' + gen_text)
                    if t < config.bpe_limit and vg.tt[t] == 2 and vg._is_content_token(t)]
        matches = sum(1 for t in gen_tids if t in sent['tids'])
        total_matches += matches
        total_target += max(1, len(sent['tids']))
        pct = 100.0 * matches / max(1, len(sent['tids']))
        print(f"  E{epoch} S{i}: {matches}/{len(sent['tids'])} ({pct:.1f}%) gen=\"{gen_text[:50]}\"")
    eval_pct = 100.0 * total_matches / total_target
    print(f">>> Epoch {epoch}: eval {total_matches}/{total_target} ({eval_pct:.1f}%)")
    config.target_boost = 15.0

print("\n=== BUILD COMPLETE ===")
print(f"Metadata: {os.path.join(REAL_DATA, 'build_metadata.json')}")
