"""
EVA — Trajectory Encoding + RAG Generation.

1. ENCODE: text → trajectory → store
2. GENERATE: seed → retrieve similar trajectories → generate with context
"""

import torch, torch.nn.functional as F, numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')
DEVICE = 'cuda'; CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.trajectory_store import TrajectoryStore
cv = CharacterVocab(); VT = 157

print("=" * 60)
print("EVA — Trajectory Encode + RAG Generate")
print("=" * 60)

# Load model
evolved = torch.load(os.path.join(CKPT, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE); aff = evolved['affinity']

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=24).to(DEVICE)
ut.set_symbol_coordinates(coords)

causal_path = os.path.join(CKPT, "causal_weights.pt")
if os.path.exists(causal_path):
    ut.load_state_dict(torch.load(causal_path, map_location='cpu', weights_only=True)['model'], strict=False)
    print("Loaded: causal weights")
else:
    ut.load_state_dict(torch.load(os.path.join(CKPT, "conceptnet_weights.pt"), map_location='cpu', weights_only=True)['model'], strict=False)
    print("Loaded: ConceptNet weights")

# ============================================================
# ENCODE: scan corpus, store trajectories
# ============================================================
store_path = os.path.join(CKPT, "trajectory_store.pkl")
store = TrajectoryStore(max_trajectories=500000)

if os.path.exists(store_path):
    print("\n[ENCODE] Loading existing trajectory store...")
    store.load(store_path)
    print(f"  {store.stats()}")
else:
    print("\n[ENCODE] Scanning corpus, storing trajectories...")
    
    npy = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
    if not os.path.exists(npy): npy = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
    data = np.load(npy, mmap_mode='r').astype(np.int32); total = len(data)
    
    ut.eval()
    N = 200000  # store 200K trajectories
    BATCH = 256; BLOCK = 64
    pos = 0; t0 = time.time(); stored = 0
    
    while stored < N and pos + BLOCK < total:
        bt = torch.full((BATCH, BLOCK), 0, dtype=torch.long, device=DEVICE)
        mask = torch.zeros(BATCH, BLOCK, device=DEVICE)
        texts_batch = []
        
        for bi in range(BATCH):
            if pos + BLOCK >= total: break
            block = data[pos:pos+BLOCK]
            valid = (block > 0) & (block < VT)
            vb = block[valid]
            vl = min(len(vb), BLOCK)
            if vl >= 4:
                bt[bi, :vl] = torch.from_numpy(vb[:vl].astype(np.int64)).to(DEVICE)
                mask[bi, :vl] = 1.0
                texts_batch.append(cv.decode(vb[:vl].tolist()))
                ids = [int(x) for x in vb[:vl]]
                
                with torch.no_grad():
                    emb = ut.embed(torch.tensor([ids], dtype=torch.long, device=DEVICE))
                    traj = emb[0].cpu().numpy()
                
                store.store(texts_batch[-1], ids, traj)
                stored += 1
            
            pos += max(vl, 16)
        
        if stored % 5000 == 0 and stored > 0:
            elapsed = time.time() - t0
            rate = stored / elapsed
            eta = (N - stored) / rate if rate > 0 else 0
            print(f"\r  {stored}/{N} trajectories ({rate:.0f}/s), eta={eta:.0f}s", end='', flush=True)
    
    print(f"\n  {store.stats()}")
    store.save(store_path)
    print(f"  Saved: {store_path}")

# ============================================================
# RAG Generation: retrieve context → generate
# ============================================================
print("\n[GENERATE] RAG generation (retrieval-augmented)...")
ut.eval()

def rag_generate(seed_ids, max_new=25, temp=0.8, top_k=30):
    """Generate with trajectory retrieval context."""
    ids = list(seed_ids)
    
    with torch.no_grad():
        for _ in range(max_new):
            inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
            
            # Get current trajectory
            emb = ut.embed(inp)
            current_traj = emb[0].cpu().numpy()
            
            # Retrieve similar trajectories
            context = store.get_context_for_generation(ids, current_traj, top_k=5)
            
            _, scores = ut(inp, return_scores=True)
            logits = scores[0, -1] / temp
            
            # Boost logits from context (characters that appear in retrieved contexts)
            if context:
                context_chars = set()
                for ctx in context:
                    if len(ctx['ids']) > len(ids):
                        next_ids = ctx['ids'][len(ids):len(ids)+3]
                        context_chars.update(next_ids)
                
                for cid in context_chars:
                    if 0 < cid < VT:
                        logits[cid] += 1.0  # boost
            
            # Top-p filtering
            sorted_l, sorted_i = logits.sort(descending=True)
            cumprobs = F.softmax(sorted_l, dim=-1).cumsum(dim=-1)
            cutoff = (cumprobs > 0.95).nonzero(as_tuple=True)[0]
            k = cutoff[0].item() + 1 if len(cutoff) > 0 else top_k
            k = min(max(k, 3), top_k)
            
            vals, idx = logits.topk(k)
            probs = F.softmax(vals, dim=-1)
            
            for t in set(ids[-5:]):
                m = (idx == t).nonzero(as_tuple=True)[0]
                if len(m) > 0: probs[m] *= 0.3
            
            probs = probs / probs.sum()
            nt = idx[torch.multinomial(probs, 1)].item()
            if nt <= 0 or nt >= VT: nt = idx[0].item()
            ids.append(nt)
    
    return ids

# Test
tests = [
    ("привет", "привет"),
    ("человек идет", "человек идет"),
    ("солнце светит", "солнце светит"),
    ("сегодня хорошая", "сегодня хорошая"),
    ("я люблю", "я люблю"),
    ("метаданные хранят", "метаданные хранят"),
]

for seed, label in tests:
    ids = cv.encode(seed)[1:-1]
    if len(ids) < 2: continue
    
    result = rag_generate(ids, 25, 0.8)
    gen = cv.decode(result)
    
    # Show retrieved contexts for first word
    if seed == "привет":
        emb = ut.embed(torch.tensor([ids], dtype=torch.long, device=DEVICE))
        ctx = store.get_context_for_generation(ids, emb[0].detach().cpu().numpy(), top_k=3)
        print(f"  Contexts for '{seed}':")
        for c in ctx: print(f"    '{c['text'][:30]}' (d={c['distance']:.2f})")
    
    print(f"  '{label}' -> '{gen}'")

print(f"\n{store.stats()}")
print("Done.")
