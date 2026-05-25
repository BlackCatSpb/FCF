"""
EVA — Unified Train+Think: обучение И мышление в одном цикле.

Каждые N шагов обучения — цикл мышления (perceive + contemplate).
Шум мышления = естественная регуляризация.
Uncertainty → curriculum: больше шагов на сложных паттернах.
"""

import torch, torch.nn.functional as F, numpy as np, sys, os, time, random
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda'; CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
LOG = os.path.join(os.path.dirname(__file__), "unified_train_log.txt")

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.self_reflection import SelfReflection
from eva.symbolic.active_causal import ActiveLearner, CausalDiscovery
from eva.symbolic.trajectory_store import TrajectoryStore

cv = CharacterVocab(); VT = 157

print("=" * 60)
print("EVA — Unified Train+Think")
print("=" * 60)

# ============================================================
# Load
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

for ckpt_name in ["gfre_latest.pt", "v2_latest.pt"]:
    ckpt_path = os.path.join(CKPT, ckpt_name)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        if 'ut' in ckpt: ut.load_state_dict(ckpt['ut'], strict=False)
        else: ut.load_state_dict(ckpt['model'], strict=False)
        print(f"Resumed from: {ckpt_name}")
        break

npy = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
if not os.path.exists(npy): npy = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
data = np.load(npy, mmap_mode='r').astype(np.int32); total = len(data)

# Trajectory store
store_path = os.path.join(CKPT, "trajectory_store.pkl")
store = TrajectoryStore()
if os.path.exists(store_path): store.load(store_path)

reflector = SelfReflection(); learner = ActiveLearner(); causal = CausalDiscovery(store)

# Genetics config
from train_genetics import fitness, mutate, crossover
GENETICS_EVERY = 10000
POP_SIZE = 50
GENETIC_GENERATIONS = 10

# ============================================================
# Hyperparams
# ============================================================
STEPS = 100000; LR = 5e-4; B = 32; ML = 128; SAVE_EVERY = 5000
THINK_EVERY = 500  # think every N training steps
THINK_ITERS = 3     # how many perception+contemplation cycles per think session

opt = torch.optim.AdamW(ut.parameters(), lr=LR, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
rng = np.random.RandomState(123)

def log(msg):
    t = time.strftime("%H:%M:%S"); line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f: f.write(line + '\n'); f.flush()

# ============================================================
# Unified train+think loop
# ============================================================
t0 = time.time()
total_uncertain = 0; total_contemplations = 0
think_stats = {'perceived': 0, 'contemplated': 0, 'uncertain': 0, 'discoveries': []}

log(f"START UNIFIED: {STEPS} steps, think every {THINK_EVERY}, batch={B}")

for s in range(1, STEPS + 1):
    # === TRAIN ===
    lens = rng.randint(32, ML + 1, B)
    starts = rng.randint(0, max(1, total - max(lens) - 1), B)
    ml = max(lens)
    bt = torch.full((B, ml), 0, dtype=torch.long, device=DEVICE)
    mask = torch.zeros(B, ml, device=DEVICE)
    
    for bi in range(B):
        vb = data[starts[bi]:starts[bi] + lens[bi]]
        vb = vb[(vb > 0) & (vb < VT)]
        vl = min(len(vb), ml)
        if vl >= 4: bt[bi, :vl] = torch.from_numpy(vb[:vl].astype(np.int64)).to(DEVICE); mask[bi, :vl] = 1.0
    
    if mask.sum() < 50: continue
    
    ut.train(); _, scores = ut(bt, return_scores=True)
    target = bt[:, 1:].clamp(1, VT - 1).contiguous()
    pred = scores[:, :-1, :].contiguous(); t_mask = mask[:, 1:]
    
    loss = F.cross_entropy(pred.view(-1, 157), target.view(-1), reduction='none')
    loss = (loss.view(B, ml - 1) * t_mask).sum() / (t_mask.sum() + 1e-8)
    
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0)
    opt.step(); sch.step()
    
    # === THINK (every N steps) ===
    if s % THINK_EVERY == 0:
        ut.eval()
        think_texts = []
        
        for ti in range(THINK_ITERS):
            # Perception: read random text block
            pos = rng.randint(0, max(1, total - 256))
            chunk = data[pos:pos + 256]
            valid = chunk[(chunk > 0) & (chunk < VT)]
            
            if len(valid) >= 10:
                text = cv.decode(valid[:60].tolist())
                ids = cv.encode(text)[1:-1]
                
                with torch.no_grad():
                    inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
                    traj = ut.embed(inp)[0].cpu().numpy()
                    store.store(text, ids, traj)
                    think_stats['perceived'] += 1
                
                # Reason about it
                result = ut.reason(ids, num_hypotheses=2, temperature=0.15, char_vocab=cv)
                if result['all_hypotheses']:
                    diag = reflector.diagnose(result['all_hypotheses'][0]['trajectory'], ids)
                    should_q, _, _ = learner.should_query(diagnostic=diag)
                    if should_q:
                        think_stats['uncertain'] += 1
                        total_uncertain += 1
                
                think_texts.append(text[:20])
            
            # Contemplation: free drift
            disc = []
            for _ in range(2):
                r = ut.reason([random.randint(1, 156)], num_hypotheses=1,
                             temperature=0.3, char_vocab=cv)
                if r['all_hypotheses']:
                    disc.append(r['answer'])
            
            if disc:
                think_stats['contemplated'] += len(disc)
                total_contemplations += len(disc)
        
        ut.train()
    
    # === LOG ===
    if s % SAVE_EVERY == 0:
        torch.save({'ut': ut.state_dict(), 'step': s}, os.path.join(CKPT, f"unified_{s}.pt"))
        torch.save({'ut': ut.state_dict(), 'step': s}, os.path.join(CKPT, "unified_latest.pt"))
        
        # === Genetic evolution ===
        if s % GENETICS_EVERY == 0 and store.total_stored > POP_SIZE:
            ut.eval()
            pop = []
            indices = np.random.choice(store.total_stored, POP_SIZE * 2, replace=False)
            for idx in indices:
                traj = store.trajectories[idx]
                if traj.shape[1] < 64:
                    padded = np.zeros((traj.shape[0], 64)); padded[:, :traj.shape[1]] = traj
                    padded[:, 24:] = np.random.randn(traj.shape[0], 40) * 0.02
                    traj = padded / (np.linalg.norm(padded, axis=1, keepdims=True) + 1e-8)
                if len(traj) >= 4:
                    pop.append({'trajectory': traj, 'ids': store.ids_list[idx]})
            pop = pop[:POP_SIZE]
            
            for ind in pop: ind['fitness'] = fitness(ind['trajectory'], ind['ids'])
            pop.sort(key=lambda x: x['fitness'], reverse=True)
            
            for gen in range(GENETIC_GENERATIONS):
                elite = pop[:5]; new_pop = list(elite)
                while len(new_pop) < POP_SIZE:
                    if random.random() < 0.3:
                        parent = random.choice(pop[:POP_SIZE//2])
                        child_traj = mutate(parent['trajectory'])
                        child = {'trajectory': child_traj, 'ids': parent['ids']}
                    else:
                        p1 = random.choice(pop[:POP_SIZE//4])
                        p2 = random.choice(pop[:POP_SIZE//2])
                        if id(p1) == id(p2): p2 = random.choice(pop)
                        child_traj = crossover(p1['trajectory'], p2['trajectory'])
                        child = {'trajectory': child_traj, 'ids': p1['ids']}
                    child['fitness'] = fitness(child['trajectory'], child['ids'])
                    new_pop.append(child)
                pop = sorted(new_pop, key=lambda x: x['fitness'], reverse=True)[:POP_SIZE]
            
            gen_best = pop[0]['fitness']
            gen_avg = np.mean([p['fitness'] for p in pop])
            log(f"  ├─ genetics: gen={GENETIC_GENERATIONS} best={gen_best:.4f} avg={gen_avg:.4f}")
            ut.train()
        
        with torch.no_grad():
            acc = (pred.argmax(-1) == target) & t_mask.bool()
            acc = acc.sum().item() / (t_mask.sum() + 1e-8)
        
        elapsed = time.time() - t0; eta = (elapsed / s) * (STEPS - s) if s > 0 else 0
        log(f"  step {s:>6d}/{STEPS} | loss={loss.item():.4f} acc={acc:.3f} | "
            f"│ think: p={think_stats['perceived']} c={think_stats['contemplated']} "
            f"u={think_stats['uncertain']} | {elapsed/60:.0f}min")
        
        think_stats = {'perceived': 0, 'contemplated': 0, 'uncertain': 0, 'discoveries': []}
    
    elif s % 500 == 0:
        with torch.no_grad():
            acc = (pred.argmax(-1) == target) & t_mask.bool()
            acc = acc.sum().item() / (t_mask.sum() + 1e-8)
        elapsed = time.time() - t0
        log(f"  step {s:>6d}/{STEPS} | loss={loss.item():.4f} acc={acc:.3f} | {elapsed/60:.0f}min")

log(f"DONE. Uncertain reports: {total_uncertain}, Contemplations: {total_contemplations}")
torch.save({'ut': ut.state_dict(), 'store': store.total_stored}, os.path.join(CKPT, "unified_final.pt"))
