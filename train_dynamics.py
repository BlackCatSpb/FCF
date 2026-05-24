"""
EVA — PotentialDynamics: живая пластика аффинной матрицы.

STDP + LTP/LTD + гомеостаз + метапластичность.
Концепты возникают естественно: частые связи усиливаются → бассейны углубляются.
"""

import torch, torch.nn.functional as F, numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")

from eva.symbolic.char_vocab import CharacterVocab
cv = CharacterVocab()
VT = 157

print("=" * 60)
print("EVA — PotentialDynamics: living affinity")
print("=" * 60)
print(f"Device: {DEVICE}")

# ============================================================
# Load affinity from Phase 1
# ============================================================
aff_path = os.path.join(CKPT_DIR, "affinity_word.pt")
print(f"\nLoading affinity: {aff_path}")
aff_data = torch.load(aff_path, map_location='cpu', weights_only=True)
affinity = aff_data['affinity'].float()  # [157, 157]
print(f"  Shape: {affinity.shape}, mean={affinity.mean():.4f}, std={affinity.std():.4f}")

# ============================================================
# Load corpus
# ============================================================
npy_file = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
if not os.path.exists(npy_file):
    npy_file = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
all_ids = np.load(npy_file, mmap_mode='r').astype(np.int32)
print(f"Corpus: {len(all_ids)/1e6:.1f}M tokens")

# ============================================================
# PotentialDynamics — adapted from legacy
# ============================================================
class PotentialDynamics:
    """Living affinity matrix with STDP, LTP/LTD, homeostasis, metaplasticity."""
    
    def __init__(self, affinity_matrix):
        self.V = affinity_matrix.shape[0]
        self.affinity = affinity_matrix.clone().to(DEVICE)
        self.usage = torch.zeros(self.V, self.V, device=DEVICE)
        self.step_count = 0
        
        # STDP traces
        self.pre_trace = torch.zeros(self.V, device=DEVICE)
        self.post_trace = torch.zeros(self.V, device=DEVICE)
        self.tau = 20.0
        self.A_plus = 0.005
        self.A_minus = 0.003
        
        # Plasticity (how easily each connection changes)
        self.plasticity = torch.ones(self.V, self.V, device=DEVICE)
        
        # History for metaplasticity
        self.history = []
        self.history_size = 50
        
        self.min_aff = 0.001
        self.max_aff = 10.0
        self.target_mean = 0.5
    
    def feed_batch(self, bt, mask):
        """Vectorized: feed BATCH of sequences [B, L]. ~100x faster than per-sequence loop."""
        B, L = bt.shape
        if L < 2:
            return
        
        # Decay traces
        self.pre_trace *= np.exp(-1.0 / self.tau)
        self.post_trace *= np.exp(-1.0 / self.tau)
        
        # Update traces for all active tokens in batch
        active = bt[mask.bool()].unique()
        for idx in active:
            if 0 < idx < self.V:
                self.pre_trace[idx] = 1.0
                self.post_trace[idx] = 1.0
        
        # Extract all adjacent pairs across batch (dist 1..4)
        for dist in range(1, 5):
            if L <= dist:
                continue
            left = bt[:, :L-dist].contiguous()   # [B, L-dist]
            right = bt[:, dist:].contiguous()      # [B, L-dist]
            pair_mask = mask[:, :L-dist] & mask[:, dist:]  # [B, L-dist]
            
            # Flatten valid pairs
            valid = pair_mask & (left > 0) & (left < self.V) & (right > 0) & (right < self.V)
            if valid.sum() == 0:
                continue
            
            li = left[valid].long()   # [N]
            ri = right[valid].long()  # [N]
            
            # STDP: vectorized
            pre_t = self.pre_trace[li]    # [N]
            post_t = self.post_trace[ri]  # [N]
            plast = self.plasticity[li, ri]
            
            dw = torch.zeros(len(li), device=DEVICE)
            
            # Pre-before-post → strengthen
            pre_mask = pre_t > 0.01
            dw[pre_mask] += self.A_plus * pre_t[pre_mask] * plast[pre_mask]
            
            # Post-before-pre → weaken
            post_mask = post_t > 0.01
            dw[post_mask] -= self.A_minus * post_t[post_mask] * plast[post_mask]
            
            # LTP increment (closer = stronger)
            dw += 0.001 * (1.0 / dist)
            
            # Apply updates
            self.affinity[li, ri] += dw
            self.affinity[li, ri] = self.affinity[li, ri].clamp(self.min_aff, self.max_aff)
            
            # Usage counting
            self.usage[li, ri] += 1
            self.usage[ri, li] += 1
        
        self.step_count += 1
        
        if self.step_count % 100 == 0:
            self._ltd()
            self._homeostasis()
        if self.step_count % 500 == 0:
            self._metaplasticity()
    
    def _ltd(self):
        """Depress unused connections."""
        mask = self.usage < 5
        self.affinity[mask] *= 0.999
        self.affinity = self.affinity.clamp(self.min_aff, self.max_aff)
    
    def _homeostasis(self):
        """Keep row means near target_mean."""
        means = self.affinity.mean(dim=1, keepdim=True)
        scale = self.target_mean / (means + 1e-8)
        scale = 1.0 + 0.01 * (scale - 1.0)  # gradual
        self.affinity = self.affinity * scale
        self.affinity = self.affinity.clamp(self.min_aff, self.max_aff)
    
    def _metaplasticity(self):
        """Connections that change a lot become more plastic."""
        self.history.append(self.affinity.clone())
        if len(self.history) > self.history_size:
            self.history.pop(0)
        if len(self.history) > 10:
            recent = torch.stack(self.history[-10:])
            variance = recent.var(dim=0)
            self.plasticity = 0.5 * self.plasticity + 0.5 * (1.0 + variance).clamp(0.1, 2.0)

# ============================================================
# Run dynamics: feed batches through the affinity (vectorized)
# ============================================================
print("\n[DYNAMICS] Vectorized batch feeding...")
dyn = PotentialDynamics(affinity)

TOTAL_BATCHES = 5000
BATCH_SIZE = 256
SEQ_LEN = 64
total_ids = len(all_ids)
rng = np.random.RandomState(77)

start = time.time()
last_print = 0

for step in range(1, TOTAL_BATCHES + 1):
    # Build batch of random subsequences
    max_len = 0
    sequences = []
    for _ in range(BATCH_SIZE):
        pos = rng.randint(0, max(1, total_ids - SEQ_LEN - 1))
        chunk = all_ids[pos:pos + SEQ_LEN]
        valid = (chunk > 0) & (chunk < VT)
        seq = chunk[valid]
        if len(seq) >= 3:
            sequences.append(seq[:SEQ_LEN])
            max_len = max(max_len, len(seq))
    
    if len(sequences) < 32:
        continue
    
    max_len = min(max_len, SEQ_LEN)
    bt = torch.zeros(BATCH_SIZE, max_len, dtype=torch.long, device=DEVICE)
    mask = torch.zeros(BATCH_SIZE, max_len, dtype=torch.bool, device=DEVICE)
    for bi, seq in enumerate(sequences):
        L = min(len(seq), max_len)
        bt[bi, :L] = torch.from_numpy(seq[:L].astype(np.int64)).to(DEVICE)
        mask[bi, :L] = True
    
    dyn.feed_batch(bt, mask)
    
    now = time.time()
    if now - last_print >= 5 or step == 1 or step == TOTAL_BATCHES:
        last_print = now
        elapsed = now - start
        eta = (elapsed / step) * (TOTAL_BATCHES - step)
        aff_mean = dyn.affinity.mean().item()
        aff_std = dyn.affinity.std().item()
        active = (dyn.usage > 10).sum().item()
        seqs_done = step * BATCH_SIZE
        print(f"  batch {step:>5d}/{TOTAL_BATCHES} ({seqs_done:,} seqs)"
              f" | aff μ={aff_mean:.4f} σ={aff_std:.4f}"
              f" | active={active} | {elapsed:.0f}s / eta {eta:.0f}s", flush=True)

# ============================================================
# Compare: old vs new affinity, MDS, concepts
# ============================================================
print("\n[COMPARE] Affinity evolution:")
print(f"  Initial: μ={affinity.mean():.4f} σ={affinity.std():.4f}")
print(f"  Evolved: μ={dyn.affinity.mean().item():.4f} σ={dyn.affinity.std().item():.4f}")

# Compute new MDS coordinates from evolved affinity
print("\n[MDS] New coordinates from evolved affinity...")
from eva.symbolic.topological_field import TopologicalField
from eva.symbolic.potential_field import PotentialField

pf = PotentialField(VT, 256)
pf.affinity = torch.nn.Parameter(dyn.affinity.cpu(), requires_grad=False)

topo = TopologicalField(pf, coord_dim=24)
topo._compute_coordinates_from_affinity()
new_coords = topo.coordinates[:VT, :24].clone()

# Diagnostic
sym_new = new_coords[1:VT].numpy()
aff_ev = dyn.affinity.cpu().numpy()
n = 156
aff_156 = aff_ev[1:VT, 1:VT]
D_mds = 1.0 - aff_156; np.fill_diagonal(D_mds, 0.0)
J = np.eye(n) - np.ones((n,n))/n; B = -0.5*J@(D_mds*D_mds)@J
eigvals = np.linalg.eigh(B)[0]; eigvals = np.sort(eigvals)[::-1]
eff_dim = (eigvals[:24].sum() / eigvals[eigvals>0].sum()) if (eigvals>0).sum()>0 else 0
unique = len(np.unique(sym_new.round(decimals=6), axis=0))
print(f"  Evolved MDS: eff_dim(24)={eff_dim:.1%}, unique={unique}/{n}")

# K-means on new coordinates
from sklearn.cluster import KMeans
N_CONCEPTS = 8
kmeans = KMeans(n_clusters=N_CONCEPTS, random_state=42, n_init=10)
labels = kmeans.fit_predict(sym_new)

concept_groups = [[] for _ in range(N_CONCEPTS)]
for idx in range(156):
    concept_groups[labels[idx]].append(idx)
concept_groups.sort(key=len, reverse=True)

print(f"\n  Concept groups from evolved affinity:")
for ci, group in enumerate(concept_groups):
    chars = ''.join(cv.decode([g+1]) for g in group)
    print(f"  Group {ci}: [{len(group):>3d}] '{chars}'")

# ============================================================
# Update transformer with new coordinates
# ============================================================
print("\n[UPDATE] Retraining transformer with evolved coordinates...")
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=24)
if DEVICE == 'cuda':
    ut = ut.cuda()
# Load existing weights as starting point
word_weights = os.path.join(CKPT_DIR, "word_weights.pt")
if os.path.exists(word_weights):
    ckpt = torch.load(word_weights, map_location='cpu', weights_only=True)
    ut.load_state_dict(ckpt['model'], strict=False)

ut.set_symbol_coordinates(new_coords.to(DEVICE))

# Quick fine-tune (just 2000 steps, low LR)
UT_STEPS = 2000; UT_LR = 1e-4; UT_BATCH = 128
opt = torch.optim.AdamW(ut.parameters(), lr=UT_LR, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=UT_STEPS)

# Extract words for training
print("  Extracting words...")
_id_to_char_arr = [cv.decode([i]) for i in range(157)]
_is_letter_digit = np.array([c.isalpha() or c.isdigit() for c in _id_to_char_arr], dtype=bool)

words = []
i = 0; total = len(all_ids)
chunk_size = 20_000_000
for cs in range(0, total, chunk_size):
    ce = min(cs + chunk_size + 20, total)
    chunk = all_ids[cs:ce]
    valid_mask = _is_letter_digit[chunk]
    in_word = False; start = 0
    for j in range(len(chunk)):
        if (cs + j) >= total: break
        if valid_mask[j]:
            if not in_word: in_word = True; start = j
        elif in_word:
            in_word = False
            wl = j - start
            if 2 <= wl <= 20:
                words.append(chunk[start:j].tolist())
print(f"    {len(words):,} words")

start_t = time.time()
last_print_t = 0
rng = np.random.RandomState(88)

for step in range(1, UT_STEPS + 1):
    idxs = rng.randint(0, len(words), UT_BATCH)
    batch_words = [words[i] for i in idxs]
    max_len = max(len(w) for w in batch_words)
    
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
            correct = ((pred == target) * mask).sum().item()
            tok_acc = correct / (mask.sum() + 1e-8)
        print(f"  step {step:>4d}/{UT_STEPS} | loss={loss.item():.4f} | tok_acc={tok_acc:.3f}"
              f" | {elapsed:.0f}s / eta {eta:.0f}s", flush=True)

# Test
print("\n[TEST] Word reconstruction with evolved coordinates:")
ut.eval()
test_words = ["привет", "человек", "солнце", "трансформер", "метаданные", "фрактал"]
for word in test_words:
    ids = cv.encode(word)[1:-1]
    inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        _, scores = ut(inp, return_scores=True)
        pred = scores[0].argmax(dim=-1).tolist()
    gen = cv.decode(pred)
    ok = "OK" if pred == ids else f"ERR"
    print(f"  '{word}' → '{gen}' [{ok}]")

# Save
evolved_path = os.path.join(CKPT_DIR, "evolved_affinity.pt")
torch.save({'affinity': dyn.affinity.cpu(), 'coords': new_coords, 'model': ut.state_dict()}, evolved_path)
print(f"\nSaved: {evolved_path}")
print("Done.")
