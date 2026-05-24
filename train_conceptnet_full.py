"""
EVA — ConceptNet Training: enrich affinity matrix with Russian semantic relationships.

Extracts ALL Russian-language edges from ConceptNet, boosts affinity between
semantically related symbols, recomputes MDS, retrains transformer.

Result: coordinates reflect both statistical co-occurrence AND semantic knowledge.
"""

import torch, torch.nn.functional as F, numpy as np, sys, os, time, gc
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, r"C:\Users\black\OneDrive\Desktop\EVA-Ai")
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")

from eva.symbolic.char_vocab import CharacterVocab
cv = CharacterVocab()
VT = 157

print("=" * 60)
print("EVA — ConceptNet Affinity Training")
print("=" * 60)
print(f"Device: {DEVICE}")

# ============================================================
# Connect to ConceptNet
# ============================================================
print("\n[CONNECT] Loading ConceptNet...")
from conceptnet_lite import connect, Label, Concept, Language, edges_for, Edge as CNEdge

db_path = r"C:\Users\black\OneDrive\Desktop\EVA-Ai\conceptnet.db"
connect(db_path)
print(f"  Connected: {db_path}")

# ============================================================
# Phase 1: Get ALL Russian labels from ConceptNet
# ============================================================
print("\n[PHASE 1] Getting Russian labels from ConceptNet...")

lang_ru = Language.get(name='ru')

# Direct query: get labels with Russian language
ru_labels = list(Label.select().where(Label.language == lang_ru))
print(f"  Found {len(ru_labels):,} Russian labels")

# Get concepts for these labels
ru_concepts = []
for label in ru_labels:
    try:
        concepts = Concept.select().where(Concept.label == label)
        for c in concepts:
            ru_concepts.append((label.text, c))
    except:
        pass

print(f"  Found {len(ru_concepts):,} Russian concepts")

# ============================================================
# Phase 2: Get edges for Russian concepts
# ============================================================
print("\n[PHASE 2] Getting edges for Russian concepts...")

all_ru_words = set()
ru_edges = []

for i, (text, concept) in enumerate(ru_concepts):
    try:
        edges = list(edges_for([concept]))
        for edge in edges:
            rel = edge.relation.name
            weight = edge.etc.get('weight', 1.0)
            
            start_uri = edge.start.uri
            end_uri = edge.end.uri
            start_label = start_uri.split('/')[-1] if '/' in start_uri else start_uri
            end_label = end_uri.split('/')[-1] if '/' in end_uri else end_uri
            start_lang = start_uri.split('/')[1] if len(start_uri.split('/')) >= 2 else ''
            end_lang = end_uri.split('/')[1] if len(end_uri.split('/')) >= 2 else ''
            
            ru_edges.append((start_label, end_label, rel, weight, start_lang == 'ru', end_lang == 'ru'))
            
            if start_lang == 'ru':
                all_ru_words.add(start_label.lower().strip())
            if end_lang == 'ru':
                all_ru_words.add(end_label.lower().strip())
    except:
        pass
    
    if (i+1) % 500 == 0:
        print(f"\r  Processed {i+1}/{len(ru_concepts)} concepts, "
              f"{len(ru_edges):,} edges, {len(all_ru_words):,} unique words", end='', flush=True)

print(f"\n  Total: {len(ru_edges):,} edges, {len(all_ru_words):,} unique Russian words")

# ============================================================
# Phase 2: Encode Russian concepts as symbol sequences
# ============================================================
print("\n[PHASE 2] Encoding concepts as symbol sequences...")

# Build frequency map: how often does each symbol pair co-occur via ConceptNet?
cn_cooccur = np.zeros((VT, VT), dtype=np.float64)

# Build a set of all unique Russian words from ConceptNet
ru_words = set()
for s, e, rel, w, s_ru, e_ru in ru_edges:
    if s_ru:
        ru_words.add(s.lower().strip())
    if e_ru:
        ru_words.add(e.lower().strip())

print(f"  Unique Russian words: {len(ru_words):,}")

# Encode each word and collect co-occurrences
encoded_words = {}
encoded_count = 0
for word in all_ru_words:
    ids = cv.encode(word)[1:-1]
    if len(ids) >= 2:
        encoded_words[word] = ids
        encoded_count += 1

print(f"  Encoded words: {encoded_count:,}")

# Count symbol co-occurrences from edges
cn_cooccur = np.zeros((VT, VT), dtype=np.float64)
pair_count = 0

for s, e, rel, weight, s_ru, e_ru in ru_edges:
    s_lower = s.lower().strip()
    e_lower = e.lower().strip()
    
    s_ids = encoded_words.get(s_lower, [])
    e_ids = encoded_words.get(e_lower, [])
    
    if len(s_ids) >= 2 and len(e_ids) >= 2:
        boost = weight
        for si in s_ids:
            for ei in e_ids:
                if 0 < si < VT and 0 < ei < VT:
                    cn_cooccur[si, ei] += boost
                    cn_cooccur[ei, si] += boost
        pair_count += 1

cn_total = cn_cooccur.sum()
print(f"  ConceptNet co-occurrence pairs from {pair_count:,} edges: {cn_total:,.0f}")

# ============================================================
# Phase 3: Merge with existing affinity
# ============================================================
print("\n[PHASE 3] Merging ConceptNet affinity with statistical affinity...")

# Load existing affinity
aff_path = os.path.join(CKPT_DIR, "affinity_word.pt")
aff_data = torch.load(aff_path, map_location='cpu', weights_only=True)
stat_aff = aff_data['affinity'].numpy()
stat_count = aff_data['co_occurrence'].numpy()

print(f"  Statistical affinity: μ={stat_aff.mean():.4f} σ={np.std(stat_aff):.4f}")

# Normalize ConceptNet co-occurrences to [0, 1] range
cn_max = cn_cooccur.max()
if cn_max > 0:
    cn_norm = cn_cooccur / cn_max
else:
    cn_norm = cn_cooccur

# Merge: weighted blend
CN_WEIGHT = 0.3  # 30% ConceptNet, 70% statistical
merged_aff = (1.0 - CN_WEIGHT) * stat_aff + CN_WEIGHT * (0.5 + 0.5 * cn_norm)
np.fill_diagonal(merged_aff, 0.5)

print(f"  ConceptNet affinity: μ={cn_norm.mean():.4f} σ={np.std(cn_norm):.4f}")
print(f"  Merged affinity: μ={merged_aff.mean():.4f} σ={np.std(merged_aff):.4f}")

# ============================================================
# Phase 4: MDS on merged affinity → new coordinates
# ============================================================
print("\n[PHASE 4] MDS on merged affinity...")

from eva.symbolic.topological_field import TopologicalField
from eva.symbolic.potential_field import PotentialField

pf = PotentialField(VT, 256)
pf.affinity = torch.nn.Parameter(torch.tensor(merged_aff, dtype=torch.float32), requires_grad=False)

topo = TopologicalField(pf, coord_dim=24)
topo._compute_coordinates_from_affinity()
cn_coords = topo.coordinates[:VT, :24].clone()

# Diagnostic
cn_np = cn_coords[1:VT].cpu().numpy()
n = 156
aff_156 = merged_aff[1:VT, 1:VT]
D_mds = 1.0 - aff_156; np.fill_diagonal(D_mds, 0.0)
J = np.eye(n) - np.ones((n,n))/n; B = -0.5*J@(D_mds*D_mds)@J
eigvals = np.linalg.eigh(B)[0]; eigvals = np.sort(eigvals)[::-1]
eff_dim = (eigvals[:24].sum() / eigvals[eigvals>0].sum()) if (eigvals>0).sum()>0 else 0
unique = len(np.unique(cn_np.round(decimals=6), axis=0))
print(f"  MDS: eff_dim(24)={eff_dim:.1%}, unique={unique}/{n}")
print(f"  Coordinates: {cn_coords.shape}")

# ============================================================
# Phase 5: Train transformer on ConceptNet-enriched coordinates
# ============================================================
print("\n[PHASE 5] Training transformer with ConceptNet-enriched coordinates...")

from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=24)
if DEVICE == 'cuda':
    ut = ut.cuda()
ut.set_symbol_coordinates(cn_coords.to(DEVICE))
print(f"  {ut.summary()}")

UT_STEPS = 3000; UT_LR = 1e-3; UT_BATCH = 128
opt = torch.optim.AdamW(ut.parameters(), lr=UT_LR, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=UT_STEPS)

# Train on encoded ConceptNet words
encoded_list = [(w, ids) for w, ids in encoded_words.items() if 2 <= len(ids) <= 20]

start_t = time.time()
last_print_t = 0
rng = np.random.RandomState(42)

for step in range(1, UT_STEPS + 1):
    idxs = rng.randint(0, len(encoded_list), UT_BATCH)
    batch_words = [encoded_list[i][1] for i in idxs]
    max_len = max(len(w) for w in batch_words) if batch_words else 5
    
    bt = torch.full((UT_BATCH, max_len), 0, dtype=torch.long, device=DEVICE)
    mask = torch.zeros(UT_BATCH, max_len, device=DEVICE)
    for bi, w in enumerate(batch_words):
        bt[bi, :len(w)] = torch.tensor(w, dtype=torch.long, device=DEVICE)
        mask[bi, :len(w)] = 1.0
    
    ut.train()
    _, scores = ut(bt, return_scores=True)
    target = bt.clamp(1, VT-1)
    loss = F.cross_entropy(scores.view(-1, 157), target.view(-1), reduction='none')
    loss = (loss.view(UT_BATCH, max_len) * mask).sum() / (mask.sum() + 1e-8)
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0)
    opt.step()
    sch.step()
    
    now = time.time()
    if now - last_print_t >= 5 or step == 1 or step == UT_STEPS:
        last_print_t = now
        elapsed = now - start_t
        eta = (elapsed / step) * (UT_STEPS - step)
        with torch.no_grad():
            pred = scores.argmax(dim=-1)
            acc = ((pred == target) & mask.bool()).sum().item() / (mask.sum() + 1e-8)
        print(f"  step {step:>4d}/{UT_STEPS} | loss={loss.item():.4f} |"
              f" acc={acc:.3f} | {elapsed:.0f}s / eta {eta:.0f}s", flush=True)

# ============================================================
# Phase 6: Test — compare before/after ConceptNet
# ============================================================
print("\n[TEST] Before vs after ConceptNet enrichment...")

ut.eval()
test_concepts = [
    ("кошка", "собака", "animals — should be close"),
    ("стол", "стул", "furniture — should be close"),
    ("вода", "огонь", "elements — should be close"),
    ("любовь", "дом", "unrelated — should be far"),
    ("человек", "машина", "agent vs tool"),
    ("солнце", "луна", "celestial — should be close"),
]

for w1, w2, desc in test_concepts:
    ids1 = encoded_words.get(w1, cv.encode(w1)[1:-1])
    ids2 = encoded_words.get(w2, cv.encode(w2)[1:-1])
    
    if len(ids1) >= 2 and len(ids2) >= 2:
        inp1 = torch.tensor([ids1], dtype=torch.long, device=DEVICE)
        inp2 = torch.tensor([ids2], dtype=torch.long, device=DEVICE)
        
        with torch.no_grad():
            emb1 = ut.embed(inp1).mean(dim=1)  # centroid
            emb2 = ut.embed(inp2).mean(dim=1)
            dist = (emb1 - emb2).norm().item()
        
        print(f"  '{w1}' ↔ '{w2}' ({desc}): dist={dist:.3f}")

# Save
cn_weights_path = os.path.join(CKPT_DIR, "conceptnet_weights.pt")
torch.save({
    'model': ut.state_dict(),
    'coords': cn_coords,
    'merged_affinity': torch.tensor(merged_aff),
}, cn_weights_path)
print(f"\nSaved: {cn_weights_path}")
print("Done.")
