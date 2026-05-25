"""
EVA — Trajectory Genetics: эволюция траекторий в ℝ⁶⁴.

Генетические операторы над траекториями:
- Mutation: ∇V(z) + Langevin noise → drift toward better regions
- Crossover: linear interpolation between two trajectories → new hybrid
- Selection: top-K by SelfReflection fitness
- Evolution: generations of mutate + crossover + select

Population = best trajectories from TrajectoryStore.
Fitness = confidence * efficiency / (curvature + 1)
"""

import torch, numpy as np, random, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.self_reflection import SelfReflection
from eva.symbolic.trajectory_store import TrajectoryStore

cv = CharacterVocab(); VT = 157

print("=" * 60)
print("EVA — Trajectory Genetics & Evolution")
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

for ckpt_name in ["unified_latest.pt"]:
    ckpt_path = os.path.join(CKPT, ckpt_name)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        ut.load_state_dict(ckpt.get('ut', ckpt.get('model', {})), strict=False)
        print(f"Loaded: {ckpt_name}")
        break
ut.eval()

store_path = os.path.join(CKPT, "trajectory_store.pkl")
store = TrajectoryStore()
if os.path.exists(store_path): store.load(store_path)
print(f"Store: {store.stats()}")

reflector = SelfReflection()

# ============================================================
# Genetic operators
# ============================================================
def fitness(trajectory: np.ndarray, ids=None) -> float:
    """Fitness = confidence * efficiency / (1 + curvature). Higher = better."""
    diag = reflector.diagnose(trajectory, ids)
    return diag.confidence * diag.efficiency / (1.0 + diag.mean_curvature)

def mutate(trajectory: np.ndarray, strength=0.05) -> np.ndarray:
    """
    Gradient flow mutation: add ∇V noise to discover nearby better paths.
    Uses transformer's reason() for gradient guidance.
    """
    if len(trajectory) < 3:
        return trajectory
    
    # Add Gaussian noise scaled by local curvature
    steps = trajectory[1:] - trajectory[:-1]
    step_norms = np.linalg.norm(steps, axis=1, keepdims=True) + 1e-8
    
    # Mutation: perturb each point proportional to step size
    noise = np.random.randn(*trajectory.shape) * strength * step_norms.mean()
    
    # More mutation at high-curvature points
    curvature = np.zeros(len(trajectory))
    for i in range(1, len(trajectory) - 1):
        v1 = trajectory[i] - trajectory[i-1]
        v2 = trajectory[i+1] - trajectory[i]
        n1 = np.linalg.norm(v1) + 1e-8
        n2 = np.linalg.norm(v2) + 1e-8
        curvature[i] = 1.0 - abs(np.dot(v1/n1, v2/n2))
    
    curvature = curvature[:, np.newaxis]
    noise = noise * (1.0 + curvature * 2.0)  # more noise at sharp turns
    
    mutated = trajectory + noise
    # Normalize to unit sphere
    norms = np.linalg.norm(mutated, axis=1, keepdims=True) + 1e-8
    return mutated / norms

def crossover(traj_a: np.ndarray, traj_b: np.ndarray) -> np.ndarray:
    """Crossover: interpolate between two trajectories → hybrid path."""
    # Align lengths
    min_len = min(len(traj_a), len(traj_b))
    a = traj_a[:min_len]
    b = traj_b[:min_len]
    
    # Single-point crossover
    split = random.randint(min_len // 3, 2 * min_len // 3)
    
    # Interpolation blend at crossover point
    alpha = 0.3 + random.random() * 0.4  # 0.3-0.7 blend
    
    child = np.zeros_like(a)
    child[:split] = a[:split]
    child[split:] = b[split:]
    
    # Smooth the crossover point
    if split > 0 and split < min_len - 1:
        for k in range(-2, 3):
            idx = split + k
            if 0 <= idx < min_len:
                t = (k + 2) / 4.0
                child[idx] = (1 - t) * a[idx] + t * b[idx]
    
    return child

def decode_trajectory(traj, nearest_k=1):
    """Decode trajectory to text via nearest-neighbor to symbol coordinates."""
    traj_t = torch.tensor(traj, dtype=torch.float32, device=DEVICE)
    dists = torch.cdist(traj_t, coords64)  # [L, 157]
    ids = dists.argmin(dim=-1).clamp(1, VT-1).tolist()
    return cv.decode(ids)

# ============================================================
# Evolution loop
# ============================================================
POP_SIZE = 100
GENERATIONS = 20
MUTATION_RATE = 0.3
ELITE_SIZE = 10
rng = np.random.RandomState(42)

print(f"\n[EVOLUTION] {GENERATIONS} generations, pop={POP_SIZE}")

# Initialize population from TrajectoryStore
population = []
if store.total_stored > 0:
    indices = rng.choice(min(store.total_stored, 5000), POP_SIZE * 2, replace=False)
    for idx in indices:
        traj = store.trajectories[idx]  # [L, 24] from old store
        # Pad to 64-dim
        if traj.shape[1] < 64:
            padded = np.zeros((traj.shape[0], 64))
            padded[:, :traj.shape[1]] = traj
            padded[:, 24:] = np.random.randn(traj.shape[0], 40) * 0.02
            traj = padded / (np.linalg.norm(padded, axis=1, keepdims=True) + 1e-8)
        ids = store.ids_list[idx]
        if len(traj) >= 4:
            population.append({'trajectory': traj, 'ids': ids, 'text': store.texts[idx]})
else:
    # Fallback: random trajectories
    for _ in range(POP_SIZE):
        L = random.randint(4, 32)
        traj = np.random.randn(L, 64)
        traj = traj / np.linalg.norm(traj, axis=1, keepdims=True)
        population.append({'trajectory': traj, 'ids': None, 'text': ''})

population = population[:POP_SIZE]

# Score initial population
for ind in population:
    ind['fitness'] = fitness(ind['trajectory'], ind['ids'])

population.sort(key=lambda x: x['fitness'], reverse=True)

print(f"  Gen 0: best_fitness={population[0]['fitness']:.4f} "
      f"avg={np.mean([p['fitness'] for p in population]):.4f} "
      f"text='{population[0].get('text','')[:30]}'")

# Evolution
best_history = []
t0 = time.time()

for gen in range(1, GENERATIONS + 1):
    elite = population[:ELITE_SIZE]
    new_pop = list(elite)  # keep elite
    
    # Breed new individuals
    while len(new_pop) < POP_SIZE:
        if random.random() < MUTATION_RATE:
            # Mutation
            parent = random.choice(population[:POP_SIZE//2])  # top half
            child_traj = mutate(parent['trajectory'])
            child_ids = parent['ids']  # approximate
            child = {'trajectory': child_traj, 'ids': child_ids, 'text': ''}
        else:
            # Crossover
            p1 = random.choice(population[:POP_SIZE//4])
            p2 = random.choice(population[:POP_SIZE//2])
            if id(p1) == id(p2): p2 = random.choice(population)
            child_traj = crossover(p1['trajectory'], p2['trajectory'])
            child = {'trajectory': child_traj, 'ids': p1['ids'], 'text': ''}
        
        child['fitness'] = fitness(child['trajectory'], child['ids'])
        new_pop.append(child)
    
    population = sorted(new_pop, key=lambda x: x['fitness'], reverse=True)[:POP_SIZE]
    best = population[0]
    avg_fit = np.mean([p['fitness'] for p in population])
    best_history.append(best['fitness'])
    
    # Decode best every 5 generations
    if gen % 5 == 0 or gen == 1:
        decoded = decode_trajectory(best['trajectory'])
        elite_texts = []
        for e in elite[:3]:
            if e['ids']:
                elite_texts.append(cv.decode(e['ids'][:10]))
        print(f"  Gen {gen:>2d}: best={best['fitness']:.4f} avg={avg_fit:.4f} "
              f"decode='{decoded[:25]}'")

elapsed = time.time() - t0
print(f"\n  Evolution complete: {elapsed:.1f}s")
print(f"  Fitness trend: {best_history[0]:.4f} → {best_history[-1]:.4f}")

# ============================================================
# Show best evolved trajectories
# ============================================================
print(f"\n[TOP TRAJECTORIES]")
for i in range(min(5, POP_SIZE)):
    ind = population[i]
    decoded = decode_trajectory(ind['trajectory'])
    tag = "elite" if i < ELITE_SIZE else "bred"
    print(f"  {i+1}. [{tag}] fit={ind['fitness']:.4f} '{decoded[:40]}'")

# Save population
gen_path = os.path.join(CKPT, "genetics_population.pt")
torch.save({
    'population': [(p['trajectory'], p['fitness'], p.get('text', '')) for p in population[:20]],
    'best_history': best_history,
}, gen_path)
print(f"\nSaved: {gen_path}")
print("Done.")
