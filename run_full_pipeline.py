"""
EVA Symbolic — Full Pipeline Demo.

Loads trained model, discovers words, clusters, grammatical roles,
and generates text at WORD level with back-off to symbols.
Also computes 3D coordinates for visualization.

This is the END-TO-END test of the complete system.
"""

import sys, os, torch, numpy as np, json, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

from eva.symbolic import *
from eva.symbolic.hierarchical_layer import *
from eva.symbolic.word_level import *
from eva.primordial_layer import PrimordialLayer
from eva.config import FCFConfig

OUT = os.path.join(os.path.dirname(__file__), "pipeline_result.json")

print("=" * 60)
print("EVA Symbolic — Full Pipeline Demo")
print("=" * 60)

# === 1. LOAD ===
print("\n[1/6] Loading model...")
config = FCFConfig(); config.d_model = 256; config.vocab_size = 156; config.num_heads = 8
layer = PrimordialLayer(config)
pf = PotentialField(156, 256)
char_vocab = CharacterVocab()

pf.load_state_dict(torch.load('checkpoints/symbolic/final/potential_field.pt', map_location='cpu', weights_only=True))
aff = pf.affinity.cpu().numpy()
count = pf.co_occurrence_count.cpu().numpy()

print(f"  Affinity: mean={aff.mean():.4f}, std={aff.std():.4f}, max={aff.max():.4f}")
print(f"  Counts: total={count.sum():.0f}, max={count.max():.0f}")

# === 2. RE-DISCOVER GRAMMAR FROM AFFINITY ===
print("\n[2/6] Re-discovering grammar from affinity...")
grammar = AssemblyGrammar(pf, 156, 256)
# Discover digrams: use mean affinity as threshold
mean_aff = float(aff.mean())
digrams = grammar.discover_digrams(min_affinity=mean_aff + 0.01)
ngrams = grammar.discover_ngrams(max_n=5, min_coherence=0.4)
total_patterns = sum(len(pats) for pats in grammar.patterns.values())
print(f"  Patterns discovered: {total_patterns} (digrams: {len(grammar.patterns[0])}, ngrams: {len(grammar.patterns[1])})")

# === 3. WORD DISCOVERY ===
print("\n[3/6] Discovering words from patterns...")
disc = WordDiscovery(grammar, pf, char_vocab, min_confidence=0.55)
words = disc.discover_from_grammar()
print(f"  Words discovered: {len(words)}")
if words:
    top_words = sorted(words, key=lambda w: w.confidence * w.occurrence_count, reverse=True)[:20]
    for w in top_words:
        print(f"    '{w.text}' (conf={w.confidence:.3f}, count={w.occurrence_count})")
else:
    print("  No words found from grammar patterns (need more training)")

# === 3. TOPOLOGY ===
print("\n[3/6] Computing manifold...")
topo = TopologicalField(pf, coord_dim=3)  # 3D for visualization
mlm = MultiLayerManifold(topo, pf, disc)

# Compute 3D coordinates for top words
coords_3d = {}
for w in words[:100]:
    c = mlm.compute_word_coordinates(w)
    coords_3d[w.text] = c.tolist() if isinstance(c, np.ndarray) else [0,0,0]

print(f"  3D coordinates computed for {len(coords_3d)} words")

# === 4. GRAMMAR + SEMANTICS ===
print("\n[4/6] Grammatical roles + semantic clusters...")
boundary = WordBoundaryDetector(pf)
roles_disc = GrammaticalRoleDiscovery(pf, disc)
if len(words) > 20:
    roles = roles_disc.discover_roles(n_clusters=6)
    print(f"  Grammatical roles: {len(roles)}")
    for r in roles[:4]:
        sample_words = []
        for wid in r.word_ids[:5]:
            w = disc.words.get(wid)
            if w: sample_words.append(w.text)
        print(f"    {r.name}: {sample_words}")

sem_clust = SemanticClustering(pf, disc, mlm, boundary, n_clusters=15)
if len(words) > 20:
    clusters = sem_clust.cluster_by_context()
    print(f"  Semantic clusters: {len(clusters)}")
    for cid, wids in list(clusters.items())[:5]:
        sample = []
        for wid in wids[:5]:
            w = disc.words.get(wid)
            if w: sample.append(w.text)
        print(f"    cluster_{cid}: {sample}")

# === 5. LOGIC + GENERATOR ===
print("\n[5/6] Word-level generation...")
logic = LogicCompiler(mlm, pf)

# Create symbolic generator for back-off
contra = SymbolicContradictionFilter(pf, topo)
cm = SymbolicConceptMiner(pf, topo, contra, grammar, LogicBridge(pf, 156), GeodesicNavigator(pf, topo, TangentSpace(pf, topo)))
sym_gen = SymbolicGenerator(layer, char_vocab, pf, contra, grammar, cm, topo)

word_gen = WordLevelGenerator(sym_gen, disc, logic, boundary, char_vocab)

# Generate from prompts
generations = []
prompts = ["при", "чело", "зем", "сто", "про"]
for prompt_text in prompts:
    prompt_ids = char_vocab.encode(prompt_text)[1:-1]
    gen_text = word_gen.generate(prompt_ids, temperature=0.6)
    generations.append({"prompt": prompt_text, "generated": gen_text})
    print(f"  '{prompt_text}...' -> '{gen_text[:100]}'")

# === 6. SELF-CHECK + SAVE ===
print("\n[6/6] Final evaluation...")

results = []
for g in generations:
    text = g["generated"]
    symbols = char_vocab.encode(text) if text else []
    # Simple quality: % of known symbols
    if symbols:
        cyr = sum(1 for s in symbols if 80 <= s < 150)
        quality = cyr / max(len(symbols), 1)
    else:
        quality = 0
    results.append({"prompt": g["prompt"], "text": text[:200], "quality": quality})

result_data = {
    "model": {
        "affinity_mean": float(float(aff.mean())),
        "affinity_std": float(float(aff.std())),
        "patterns_discovered": int(sum(len(pats) for pats in grammar.patterns.values())),
        "words_discovered": int(len(words)),
    },
    "generations": [
        {"prompt": r["prompt"], "text": r["text"][:200], "quality": float(r["quality"])}
        for r in results
    ],
    "sample_words": [
        {"text": str(w.text), "confidence": float(w.confidence), "count": int(w.occurrence_count)}
        for w in sorted(words, key=lambda w: w.confidence * w.occurrence_count, reverse=True)[:30]
    ] if words else [],
    "coordinates_3d": {str(k): [float(x) for x in v] for k, v in list(coords_3d.items())[:20]} if coords_3d else {},
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(result_data, f, ensure_ascii=False, indent=2)

print(f"\nResults saved to {OUT}")
print("Done.")
