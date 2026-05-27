"""
EVA — Think loop for War & Peace model. 
Reads wp_latest.pt, does self-reflection every 60s.
"""
import torch, numpy as np, sys, os, time, random
sys.path.insert(0, os.path.dirname(__file__))
DEVICE = 'cuda'; CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.trajectory_store import TrajectoryStore
cv = CharacterVocab(); VT = cv.vocab_size

# Fresh coords like training
c128 = torch.zeros(VT, 128, device=DEVICE)
g = torch.Generator(device=DEVICE).manual_seed(42)
c128[:, :] = torch.randn(157, 128, generator=g, device=DEVICE) * 0.02
c128 = c128 / c128.norm(dim=-1, keepdim=True).clamp(1e-8)

ut = UnifiedMultidimensionalTransformer(vocab_size=VT, coord_dim=128, max_levels=8,
    total_heads=32, num_layers=6, d_ff=128).to(DEVICE)
ut.set_symbol_coordinates(c128)

print("Think Loop — waiting for wp_latest.pt...")
while not os.path.exists(os.path.join(CKPT, "wp_latest.pt")):
    time.sleep(5)

ckpt = torch.load(os.path.join(CKPT, "wp_latest.pt"), map_location='cpu', weights_only=True)
ut.load_state_dict(ckpt['ut'], strict=False)
ut.eval()
print(f"Loaded wp_latest.pt (step {ckpt.get('step','?')})")

# Load trajectory store
store_path = os.path.join(CKPT, "trajectory_store.pkl")
store = TrajectoryStore()
if os.path.exists(store_path):
    store.load(store_path)

# Load War & Peace for perception
npy = os.path.join(os.path.dirname(__file__), "real_data", "war_and_peace.npy")
data = np.load(npy, mmap_mode='r').astype(np.int32)
total = len(data)
rng = np.random.RandomState()

iterations = 0
print("Think loop started. Press Ctrl+C to stop.\n")
while True:
    try:
        # Perception: read sentence from W&P
        pos = rng.randint(0, max(1, total - 64))
        ids = [int(x) for x in data[pos:pos+64] if 0 < x < VT]
        if len(ids) >= 10:
            text = cv.decode(ids[:50])
            # Reason about it
            result = ut.reason(ids[:20], num_hypotheses=2, temperature=0.2, char_vocab=cv)
            # Store trajectory
            with torch.no_grad():
                inp = torch.tensor([ids[:50]], dtype=torch.long, device=DEVICE)
                traj = ut.embed(inp)[0].cpu().numpy()
                store.store(text[:40], ids[:50], traj)
        
        # Contemplation
        if iterations % 3 == 0:
            ut.reason([random.randint(1, 156)], num_hypotheses=1, temperature=0.3, char_vocab=cv)
        
        iterations += 1
        if iterations % 10 == 0:
            # Reload model (training updates it)
            if os.path.exists(os.path.join(CKPT, "wp_latest.pt")):
                try:
                    c = torch.load(os.path.join(CKPT, "wp_latest.pt"), map_location='cpu', weights_only=True)
                    if c.get('step', 0) > ckpt.get('step', 0):
                        ut.load_state_dict(c['ut'], strict=False)
                        ckpt = c
                except: pass
            print(f"  think: {iterations} cycles, store: {store.total_stored}", flush=True)
        
        time.sleep(6)
    except KeyboardInterrupt:
        break
