"""
EVA — ConceptNet Validation: verify our coordinate topology against ConceptNet.

Query ConceptNet for Russian concepts, embed as trajectories, measure
semantic consistency: do related concepts have closer trajectories?
"""

import sys, os, time, torch, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, r"C:\Users\black\OneDrive\Desktop\EVA-Ai")
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")

from eva.symbolic.char_vocab import CharacterVocab
cv = CharacterVocab()
VT = 157

print("=" * 60)
print("EVA — ConceptNet Validation")
print("=" * 60)

# ============================================================
# Load data
# ============================================================
evolved = torch.load(os.path.join(CKPT_DIR, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE)
print(f"Loaded: coordinates {coords.shape}")

# ============================================================
# Connect to ConceptNet
# ============================================================
print("Connecting to ConceptNet...")
try:
    from conceptnet_lite import connect, Label, Concept, Language, edges_for
    
    db_path = r"C:\Users\black\OneDrive\Desktop\EVA-Ai\conceptnet.db"
    connect(db_path)
    print(f"  Connected: {db_path}")
    cn_available = True
except Exception as e:
    print(f"  ConceptNet unavailable: {e}")
    cn_available = False

if not cn_available:
    print("  Skipping validation.")
    exit(0)

# ============================================================
# Query Russian concepts
# ============================================================
print("\n[QUERY] Russian concepts from ConceptNet...")

# Test words — common Russian nouns across semantic categories
test_words = {
    # People
    "человек": "person", "женщина": "woman", "мужчина": "man", "ребенок": "child",
    # Animals
    "кошка": "cat", "собака": "dog", "птица": "bird", "рыба": "fish",
    # Objects
    "стол": "table", "стул": "chair", "дом": "house", "окно": "window",
    # Nature
    "солнце": "sun", "вода": "water", "огонь": "fire", "земля": "earth",
    # Abstract
    "любовь": "love", "счастье": "happiness", "страх": "fear", "мысль": "thought",
    # Actions
    "ходить": "walk", "говорить": "speak", "думать": "think", "делать": "do",
}

# Get ConceptNet edges for each word
cn_relations = {}  # word -> [(relation, other_word, weight)]
lang_ru = Language.get(name='ru')

for word in test_words:
    try:
        label = Label.get_or_create(text=word, language=lang_ru)[0]
        concept = Concept.get(label=label)
        edges = list(edges_for([concept]))
        
        relations = []
        for edge in edges:
            try:
                rel = edge.relation.name if hasattr(edge.relation, 'name') else str(edge.relation)
                weight = edge.etc.get('weight', 1.0) if hasattr(edge, 'etc') else 1.0
                
                start_uri = edge.start.uri
                end_uri = edge.end.uri
                
                # Extract other concept label
                if f'/ru/{word}' in start_uri:
                    other = end_uri.split('/')[-1] if '/' in end_uri else str(end_uri)
                else:
                    other = start_uri.split('/')[-1] if '/' in start_uri else str(start_uri)
                
                relations.append((rel, other, weight))
            except:
                pass
        
        cn_relations[word] = relations
    except Exception as e:
        cn_relations[word] = []

# Print summary
total_edges = sum(len(v) for v in cn_relations.values())
print(f"  Queried {len(test_words)} words, found {total_edges} edges")
for word, rels in list(cn_relations.items())[:5]:
    print(f"    '{word}': {len(rels)} edges, sample: {rels[:3]}")

# ============================================================
# Embed words as trajectories and compare
# ============================================================
print("\n[EMBED] Computing word trajectories in ℝ²⁴...")

word_trajectories = {}  # word -> trajectory [L, 24]
for word in test_words:
    ids = cv.encode(word)[1:-1]  # strip BOS/EOS
    if len(ids) == 0:
        continue
    traj = coords[ids].cpu().numpy()  # [L, 24]
    word_trajectories[word] = traj

# Compute pairwise distances between word trajectories (DTW-like: mean of pairwise)
def trajectory_distance(traj_a, traj_b):
    """Minimum average distance between two trajectories."""
    # Align by resampling to same length
    la, lb = len(traj_a), len(traj_b)
    if la == 0 or lb == 0:
        return 10.0
    # Use shorter length
    n = min(la, lb)
    a = traj_a[:n]
    b = traj_b[:n]
    return np.linalg.norm(a - b, axis=1).mean()

# ============================================================
# Correlation: ConceptNet relation weight vs trajectory distance
# ============================================================
print("\n[CORRELATION] ConceptNet semantic relation vs ℝ²⁴ distance...")

pairs = []
for w1 in test_words:
    for w2 in test_words:
        if w1 >= w2:
            continue
        if w1 not in word_trajectories or w2 not in word_trajectories:
            continue
        
        dist = trajectory_distance(word_trajectories[w1], word_trajectories[w2])
        
        # Count shared ConceptNet relations between these words
        cn_shared = 0
        cn_max_weight = 0
        for rel, other, w in cn_relations.get(w1, []):
            if other == w2 or w2 in other:
                cn_shared += 1
                cn_max_weight = max(cn_max_weight, w)
        
        pairs.append({
            'w1': w1, 'w2': w2,
            'distance': dist,
            'cn_shared': cn_shared,
            'cn_weight': cn_max_weight,
        })

# Sort by distance
pairs_sorted = sorted(pairs, key=lambda p: p['distance'])

print(f"  Total pairs: {len(pairs)}")
print(f"\n  Closest trajectories (lowest ℝ²⁴ distance):")
for p in pairs_sorted[:5]:
    print(f"    '{p['w1']}' ↔ '{p['w2']}': dist={p['distance']:.3f} "
          f"CN_shared={p['cn_shared']} CN_weight={p['cn_weight']:.2f}")

print(f"\n  Farthest trajectories:")
for p in pairs_sorted[-5:]:
    print(f"    '{p['w1']}' ↔ '{p['w2']}': dist={p['distance']:.3f} "
          f"CN_shared={p['cn_shared']} CN_weight={p['cn_weight']:.2f}")

# Check: do ConceptNet-related words have smaller distances?
cn_pairs = [p for p in pairs if p['cn_shared'] > 0]
non_cn_pairs = [p for p in pairs if p['cn_shared'] == 0]

if cn_pairs and non_cn_pairs:
    cn_dist = np.mean([p['distance'] for p in cn_pairs])
    non_cn_dist = np.mean([p['distance'] for p in non_cn_pairs])
    print(f"\n  Mean distance: ConceptNet-related = {cn_dist:.3f}, "
          f"unrelated = {non_cn_dist:.3f}")
    if cn_dist < non_cn_dist:
        print(f"  ✓ ConceptNet-related words are CLOSER in ℝ²⁴ "
              f"(Δ={non_cn_dist-cn_dist:.3f})")
    else:
        print(f"  ✗ ConceptNet-related words are FARTHER in ℝ²⁴ "
              f"(Δ={cn_dist-non_cn_dist:.3f})")

# ============================================================
# Semantic category check
# ============================================================
print("\n[CATEGORIES] Within-category vs cross-category distances...")

categories = [
    ("people", ["человек", "женщина", "мужчина", "ребенок"]),
    ("animals", ["кошка", "собака", "птица", "рыба"]),
    ("objects", ["стол", "стул", "дом", "окно"]),
    ("nature", ["солнце", "вода", "огонь", "земля"]),
]

within_dists = []
cross_dists = []

for ci, (cn, cw) in enumerate(categories):
    cw_valid = [w for w in cw if w in word_trajectories]
    
    # Within-category
    for i in range(len(cw_valid)):
        for j in range(i+1, len(cw_valid)):
            w1, w2 = cw_valid[i], cw_valid[j]
            d = trajectory_distance(word_trajectories[w1], word_trajectories[w2])
            within_dists.append(d)
    
    # Cross-category (to next category)
    if ci + 1 < len(categories):
        other_w = [w for w in categories[ci+1][1] if w in word_trajectories]
        for w1 in cw_valid:
            for w2 in other_w:
                d = trajectory_distance(word_trajectories[w1], word_trajectories[w2])
                cross_dists.append(d)

if within_dists and cross_dists:
    w_mean = np.mean(within_dists)
    c_mean = np.mean(cross_dists)
    print(f"  Within-category mean distance: {w_mean:.3f}")
    print(f"  Cross-category mean distance:  {c_mean:.3f}")
    if w_mean < c_mean:
        print(f"  ✓ Words in same category are CLOSER in ℝ²⁴ (ratio={c_mean/w_mean:.2f})")
    else:
        print(f"  ✗ No categorical clustering in ℝ²⁴ (ratio={c_mean/w_mean:.2f})")

print("\nDone.")
