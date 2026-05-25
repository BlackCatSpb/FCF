"""
EVA — Continuous Think Loop: consciousness in ℝ⁶⁴.

Модель постоянно в потоке:
- Perception: текст → ℝ⁶⁴ → gradient flow → reflect → store
- Contemplation: свободный дрейф z(t) → поиск новых аттракторов → концепты

Режимы:
1. SUPERVISED: обучение на корпусе (causal LM + GFRE)
2. PERCEPTION: чтение текста, осмысление, сохранение в TrajectoryStore
3. CONTEMPLATION: свободный градиентный поток, открытие концептов
4. DIALECTIC: синтез новых идей из противоречий

Resource estimate:
- GPU: 6 GB VRAM (model + coords + batch)
- RAM: 2 GB (trajectory store + corpus mmap)
- CPU: 1 core (data loading)
- Disk: ~100 MB (checkpoints)
"""

import torch, torch.nn.functional as F, numpy as np, sys, os, time, random
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda'; CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
LOG = os.path.join(os.path.dirname(__file__), "think_loop.txt")

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.self_reflection import SelfReflection
from eva.symbolic.active_causal import ActiveLearner, CausalDiscovery
from eva.symbolic.trajectory_store import TrajectoryStore

cv = CharacterVocab(); VT = 157

print("=" * 60)
print("EVA — Continuous Think Loop (Consciousness in ℝ⁶⁴)")
print("=" * 60)

# ============================================================
# Resource report
# ============================================================
def resources():
    import psutil
    gpu = torch.cuda.get_device_properties(0) if DEVICE == 'cuda' else None
    ram = psutil.virtual_memory()
    vram = torch.cuda.memory_allocated() / 1e9 if DEVICE == 'cuda' else 0
    
    lines = [
        f"RESOURCES:",
        f"  GPU: {gpu.name if gpu else 'N/A'}",
        f"  VRAM used: {vram:.2f} GB / {gpu.total_memory/1e9:.1f} GB" if gpu else "  VRAM: N/A",
        f"  RAM used: {ram.used/1e9:.1f} GB / {ram.total/1e9:.1f} GB",
        f"  CPU cores: {psutil.cpu_count()}",
    ]
    return lines

for line in resources():
    print(line)

# ============================================================
# Load model & data
# ============================================================
evolved = torch.load(os.path.join(CKPT, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE)
coords64 = torch.zeros(157, 64, device=DEVICE); coords64[:, :24] = coords[:, :24]
g = torch.Generator(device=DEVICE).manual_seed(42)
coords64[:, 24:] = torch.randn(157, 40, generator=g, device=DEVICE) * 0.02
coords64 = coords64 / coords64.norm(dim=-1, keepdim=True).clamp(1e-8)

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=64, num_levels=4,
    scales_per_level=4, num_layers=3, d_ff=128).to(DEVICE)
ut.set_symbol_coordinates(coords64)

# Load best weights
for ckpt_name in ["gfre_latest.pt", "v2_latest.pt"]:
    ckpt_path = os.path.join(CKPT, ckpt_name)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        if 'ut' in ckpt: ut.load_state_dict(ckpt['ut'], strict=False)
        else: ut.load_state_dict(ckpt['model'], strict=False)
        print(f"Loaded: {ckpt_name}")
        break
ut.eval()

# Load TrajectoryStore
store_path = os.path.join(CKPT, "trajectory_store.pkl")
store = TrajectoryStore()
if os.path.exists(store_path): store.load(store_path)
print(f"Store: {store.stats()}")

# Self-reflection + Active learner
reflector = SelfReflection()
learner = ActiveLearner()
causal = CausalDiscovery(store)

# Corpus for perception
npy = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
if not os.path.exists(npy): npy = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
data = np.load(npy, mmap_mode='r').astype(np.int32); total = len(data)
rng = np.random.RandomState()

# ============================================================
# Think loop
# ============================================================

def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f: f.write(line + '\n'); f.flush()

def perceive(text_block: str) -> dict:
    """Perception: read text → encode → flow → reflect → store."""
    ids = cv.encode(text_block)[1:-1]
    if len(ids) < 4: return None
    
    # Encode to ℝ⁶⁴ trajectory
    with torch.no_grad():
        inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        z_traj = ut.embed(inp)[0].cpu().numpy()  # [L, 64]
        
        # Gradient flow from centroid
        z0 = z_traj.mean(axis=0)
        z0_t = torch.tensor(z0, device=DEVICE).float().unsqueeze(0)
    
    # Reason about the input
    result = ut.reason(ids, num_hypotheses=2, temperature=0.15, char_vocab=cv)
    
    # Self-reflect
    if result['all_hypotheses']:
        diag = reflector.diagnose(result['all_hypotheses'][0]['trajectory'], ids)
        should_query, urgency, reason = learner.should_query(diagnostic=diag)
    else:
        diag = None; should_query = False; urgency = 0; reason = ""
    
    # Store trajectory
    store.store(text_block, ids, z_traj)
    
    return {
        'text': text_block[:40],
        'answer': result['answer'],
        'confidence': result['confidence'],
        'path_len': result['all_hypotheses'][0]['path_length'] if result['all_hypotheses'] else 0,
        'uncertain': should_query,
        'urgency': urgency,
    }

def contemplate(seed_z=None, steps=10) -> list:
    """Contemplation: free gradient drift → discover new attractors."""
    device = next(ut.parameters()).device
    
    if seed_z is None:
        seed_z = torch.randn(1, 64, device=device)
        seed_z = seed_z / seed_z.norm(dim=-1, keepdim=True)
    
    discoveries = []
    with torch.no_grad():
        for _ in range(steps):
            result = ut.reason([random.randint(1, 156)], num_hypotheses=1,
                              temperature=0.3, char_vocab=cv)
            if result['all_hypotheses']:
                h = result['all_hypotheses'][0]
                discoveries.append({
                    'symbol': result['answer'],
                    'entropy': h['entropy'],
                    'path_len': h['path_length'],
                })
    
    return discoveries

def think_loop(minutes=60, perceive_every=10, contemplate_every=30):
    """Main continuous think loop."""
    log(f"THINK LOOP START: {minutes}min, perceive={perceive_every}s, contemplate={contemplate_every}s")
    
    t_start = time.time()
    iterations = 0; total_tokens = 0
    last_perceive = 0; last_contemplate = 0
    
    while time.time() - t_start < minutes * 60:
        now = time.time()
        
        # Perception: read text from corpus
        if now - last_perceive >= perceive_every:
            pos = rng.randint(0, max(1, total - 256))
            chunk = data[pos:pos + 256]
            valid = chunk[(chunk > 0) & (chunk < VT)]
            
            if len(valid) >= 10:
                text = cv.decode(valid[:80].tolist())
                result = perceive(text)
                if result:
                    total_tokens += len(valid[:80])
                    
                    if iterations % 30 == 0:
                        log(f"  PERCEIVE: '{result['text']}...' → ans='{result['answer']}' "
                            f"conf={result['confidence']:.2f}"
                            f"{' [UNCERTAIN]' if result['uncertain'] else ''}")
            
            last_perceive = now
        
        # Contemplation: free drift
        if now - last_contemplate >= contemplate_every:
            discoveries = contemplate(steps=3)
            
            if iterations % 10 == 0 and discoveries:
                symbols = [d['symbol'] for d in discoveries]
                log(f"  CONTEMPLATE: discovered '{' '.join(symbols)}'")
            
            last_contemplate = now
        
        iterations += 1
    
    elapsed = time.time() - t_start
    vram_used = torch.cuda.memory_allocated() / 1e9 if DEVICE == 'cuda' else 0
    
    log(f"THINK LOOP END: {iterations} iterations, {total_tokens} tokens, "
        f"{elapsed/60:.1f}min, VRAM={vram_used:.2f}GB, "
        f"store={store.total_stored}")
    
    return {
        'iterations': iterations,
        'tokens': total_tokens,
        'trajectories': store.total_stored,
        'vram_gb': vram_used,
    }

# ============================================================
# Run
# ============================================================
print(f"\nStarting think loop: 10 minutes...")
print(f"  Perceive: every 10s (read text, reason, reflect, store)")
print(f"  Contemplate: every 30s (free drift, discover attractors)")
print()

stats = think_loop(minutes=10, perceive_every=10, contemplate_every=30)

print(f"\n{'='*60}")
print(f"Think Loop Complete:")
print(f"  Iterations: {stats['iterations']}")
print(f"  Tokens processed: {stats['tokens']:,}")
print(f"  Trajectories stored: {stats['trajectories']:,}")
print(f"  VRAM peak: {stats['vram_gb']:.2f} GB")
print(f"Done.")
