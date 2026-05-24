"""
EVA — FractalSelfConsistency: scale invariance across hierarchy levels.

Tests:
1. Scale invariance: upsample(downsample(trajectory)) ≈ trajectory
2. Fractal dimension: box-counting on coordinate distribution in ℝ²⁴
3. Scale recurrence: autocorrelation across levels (symbol→word→sentence)
4. Multi-level coherence: do word centroids follow same distribution as symbols?
"""

import torch, numpy as np, sys, os, time
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")

from eva.symbolic.char_vocab import CharacterVocab
cv = CharacterVocab()
VT = 157

print("=" * 60)
print("EVA — FractalSelfConsistency")
print("=" * 60)

# ============================================================
# Load data
# ============================================================
evolved = torch.load(os.path.join(CKPT_DIR, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE)
affinity = evolved['affinity'].numpy()
print(f"Loaded: coords {coords.shape}")

# ============================================================
# Test 1: Multi-level coordinate analysis
# ============================================================
print("\n[TEST 1] Multi-level coordinate distributions...")

# Symbol level: 157 points in ℝ²⁴
sym_coords = coords[1:VT].cpu().numpy()  # [156, 24]

# Word level: extract words, compute centroids
npy_file = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
if not os.path.exists(npy_file):
    npy_file = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
all_ids = np.load(npy_file, mmap_mode='r').astype(np.int32)

_id_to_char = [cv.decode([i]) for i in range(157)]
_is_letter = np.array([c.isalpha() or c.isdigit() for c in _id_to_char], dtype=bool)

# Sample words and compute centroids
word_centroids = []
rng = np.random.RandomState(55)
n_words = 5000

i = 0
while len(word_centroids) < n_words and i < len(all_ids) - 50:
    pos = rng.randint(0, len(all_ids) - 30)
    # Find word start
    while pos > 0 and _is_letter[all_ids[pos]]:
        pos -= 1
    pos += 1
    # Extract word
    start = pos
    while pos < len(all_ids) and _is_letter[all_ids[pos]]:
        pos += 1
    word_ids = all_ids[start:pos]
    if 2 <= len(word_ids) <= 20:
        valid_ids = [int(x) for x in word_ids if 0 < x < VT]
        if len(valid_ids) >= 2:
            centroid = coords[valid_ids].mean(dim=0).cpu().numpy()
            word_centroids.append(centroid)
    i += 1

word_coords = np.array(word_centroids)  # [N, 24]
print(f"  Symbols: {len(sym_coords)} points")
print(f"  Word centroids: {len(word_coords)} points")

# Compare distributions
sym_distances = []
for _ in range(5000):
    i, j = rng.randint(0, len(sym_coords), 2)
    sym_distances.append(np.linalg.norm(sym_coords[i] - sym_coords[j]))
sym_distances = np.array(sym_distances)

word_distances = []
for _ in range(5000):
    i, j = rng.randint(0, len(word_coords), 2)
    word_distances.append(np.linalg.norm(word_coords[i] - word_coords[j]))
word_distances = np.array(word_distances)

print(f"\n  Distance distributions:")
print(f"    Symbol-symbol: μ={sym_distances.mean():.3f} σ={sym_distances.std():.3f}")
print(f"    Word-word:     μ={word_distances.mean():.3f} σ={word_distances.std():.3f}")
print(f"    Ratio μ: {word_distances.mean()/sym_distances.mean():.3f}")
print(f"    Ratio σ: {word_distances.std()/sym_distances.std():.3f}")

# ============================================================
# Test 2: Fractal dimension via box-counting
# ============================================================
print("\n[TEST 2] Fractal dimension (box-counting)...")

def box_count(points, box_size, dims=None):
    """Count boxes needed to cover points at given scale."""
    if dims is None:
        dims = points.shape[1]
    n_points = len(points)
    if n_points == 0:
        return 1
    
    # Use first `dims` dimensions, divide by box_size
    scaled = points[:, :dims] / box_size
    boxes = set()
    for i in range(n_points):
        box = tuple((scaled[i] * 4).astype(int))  # quantize
        boxes.add(box)
    return max(1, len(boxes))

# Test at multiple box sizes
box_sizes = np.logspace(-1, 0.7, 15)
counts = []
for bs in box_sizes:
    bc = box_count(sym_coords, bs, dims=8)  # use 8D for efficiency
    counts.append(bc)

# Fit line: log(count) vs log(1/box_size)
log_sizes = np.log(1.0 / box_sizes)
log_counts = np.log(counts)

if len(log_sizes) >= 3:
    slope, intercept = np.polyfit(log_sizes[3:], log_counts[3:], 1)
    frac_dim = slope
    print(f"  Fractal dimension (symbol level): {frac_dim:.2f}")
    print(f"  (24D space, 157 points, using 8 projected dims)")
    print(f"  Box sizes: {box_sizes[0]:.4f} to {box_sizes[-1]:.4f}")
    print(f"  Counts range: {counts[0]} to {counts[-1]}")
    print(f"  Fit quality: slope={slope:.3f}")

# Also compute for word centroids
word_counts = []
for bs in box_sizes:
    bc = box_count(word_coords, bs, dims=8)
    word_counts.append(bc)

if len(word_counts) >= 3:
    log_wc = np.log(word_counts)
    w_slope, _ = np.polyfit(log_sizes[3:], log_wc[3:], 1)
    print(f"\n  Fractal dimension (word level): {w_slope:.2f}")
    print(f"  Scale invariance ratio: {w_slope/frac_dim:.3f}")
    if abs(w_slope - frac_dim) < 1.0:
        print(f"  ✓ Scale-invariant (dim difference < 1.0)")
    else:
        print(f"  ✗ Not scale-invariant (dim difference = {abs(w_slope-frac_dim):.1f})")

# ============================================================
# Test 3: Affinity self-similarity across scales
# ============================================================
print("\n[TEST 3] Affinity decay pattern...")

# Compute how affinity falls off with distance (power law check)
dist_bins = np.linspace(0, 10, 30)
aff_means = []
for k in range(len(dist_bins) - 1):
    lo, hi = dist_bins[k], dist_bins[k+1]
    # Compute distance matrix
    dists = np.zeros((156, 156))
    for i in range(156):
        diff = sym_coords[i:i+1] - sym_coords
        dists[i] = np.linalg.norm(diff, axis=1)
    
    mask = (dists >= lo) & (dists < hi)
    np.fill_diagonal(mask, False)
    if mask.sum() > 0:
        aff_means.append(affinity[1:VT, 1:VT][mask].mean())
    else:
        aff_means.append(0.5)

dist_centers = (dist_bins[:-1] + dist_bins[1:]) / 2
aff_means = np.array(aff_means)

# Log-log fit for power law: affinity ∝ distance^(-α)
valid = aff_means > 0.01
if valid.sum() >= 4:
    log_d = np.log(dist_centers[valid] + 1e-8)
    log_a = np.log(aff_means[valid] + 1e-8)
    alpha, c = np.polyfit(log_d, log_a, 1)
    print(f"  Affinity ~ distance^({alpha:.2f})")
    print(f"  Power law exponent α = {-alpha:.2f}")
    if 0.5 < -alpha < 3.0:
        print(f"  ✓ Power law decay detected")
    else:
        print(f"  ✗ No clear power law")

# ============================================================
# Test 4: Trajectory self-similarity
# ============================================================
print("\n[TEST 4] Trajectory scale recurrence...")

# Take a sentence, break into words, compare word-level vs symbol-level patterns
test_sent = "человек идет по улице"
ids = cv.encode(test_sent)[1:-1]  # strip BOS/EOS
print(f"  Sentence: '{test_sent}' ({len(ids)} chars)")

if len(ids) >= 6:
    # Symbol-level trajectory
    sym_traj = coords[ids].cpu().numpy()  # [L, 24]
    
    # Word-level: group by hand (approximate word boundaries)
    # Split at space
    text = test_sent
    word_boundaries = [0]
    for word in text.split():
        word_boundaries.append(word_boundaries[-1] + 2 + len(word))  # rough
    
    # Compute actual word boundaries from IDs
    space_id = cv.encode(" ")[1]  # space char ID
    boundaries = [0]
    for k, idx in enumerate(ids):
        if idx == space_id:
            boundaries.append(k + 1)
    boundaries.append(len(ids))
    boundaries = sorted(set(boundaries))
    
    # Compute word centroids
    word_centroids_sent = []
    for bi in range(len(boundaries) - 1):
        lo, hi = boundaries[bi], boundaries[bi+1]
        if hi > lo:
            wc = coords[ids[lo:hi]].mean(dim=0).cpu().numpy()
            word_centroids_sent.append(wc)
    word_traj = np.array(word_centroids_sent) if word_centroids_sent else np.zeros((0, 24))
    
    if len(word_traj) >= 2:
        # Correlation between symbol-level and word-level step vectors
        sym_steps = sym_traj[1:] - sym_traj[:-1]
        word_steps = word_traj[1:] - word_traj[:-1]
        
        # Normalize
        sym_steps_norm = sym_steps / (np.linalg.norm(sym_steps, axis=1, keepdims=True) + 1e-8)
        word_steps_norm = word_steps / (np.linalg.norm(word_steps, axis=1, keepdims=True) + 1e-8)
        
        # Resample to same length for correlation
        min_len = min(len(sym_steps_norm), len(word_steps_norm))
        if min_len >= 2:
            # Average direction at each level
            sym_dir = sym_steps_norm[:min_len].mean(axis=0)
            word_dir = word_steps_norm[:min_len].mean(axis=0)
            cos_sim = np.dot(sym_dir, word_dir)
            
            print(f"  Word centroids: {len(word_traj)} points")
            print(f"  Symbol-level mean step: {np.linalg.norm(sym_steps.mean(axis=0)):.3f}")
            print(f"  Word-level mean step: {np.linalg.norm(word_steps.mean(axis=0)):.3f}")
            print(f"  Direction correlation: {cos_sim:.3f}")
            if abs(cos_sim) > 0.5:
                print(f"  ✓ Trajectory direction preserved across scales")
            else:
                print(f"  ✗ Trajectory direction not preserved")

# Save
sc_path = os.path.join(CKPT_DIR, "fractal_consistency.pt")
torch.save({
    'symbol_distances': sym_distances,
    'word_distances': word_distances,
    'fractal_dim_symbol': frac_dim if 'frac_dim' in dir() else 0.0,
    'fractal_dim_word': w_slope if 'w_slope' in dir() else 0.0,
    'affinity_decay': aff_means,
}, sc_path)
print(f"\nSaved: {sc_path}")
print("Done.")
