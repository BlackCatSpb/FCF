"""
EVA — Autonomous Consolidation. No training. Just thinking.

Загружает чекпоинт + TrajectoryStore. Анализирует существующие знания:
- Perception: читает текст, генерирует продолжение
- Reflection: оценивает качество траекторий  
- Discovery: находит паттерны и концепты в хранилище
"""

import torch, torch.nn.functional as F, numpy as np, sys, os, time, random, json
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda'; CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.trajectory_store import TrajectoryStore, HierarchicalTrajectory
from eva.symbolic.dynamic_vocab import LatentCodeOperator

cv = CharacterVocab(); VT = cv.vocab_size

print("=" * 60)
print("EVA — Autonomous Consolidation")
print("=" * 60)

# Load model
c128 = torch.zeros(VT, 128, device=DEVICE)
g = torch.Generator(device=DEVICE).manual_seed(42)
c128[:, :] = torch.randn(VT, 128, generator=g, device=DEVICE) * 0.02
c128 = c128 / c128.norm(dim=-1, keepdim=True).clamp(1e-8)

ut = UnifiedMultidimensionalTransformer(vocab_size=VT, coord_dim=128, max_levels=8,
    total_heads=32, num_layers=6, d_ff=128).to(DEVICE)
ut.set_symbol_coordinates(c128)

wp = os.path.join(CKPT, "wp_latest.pt")
if os.path.exists(wp):
    ckpt = torch.load(wp, map_location='cpu', weights_only=True)
    ut.load_state_dict(ckpt['ut'], strict=False)
    print(f"Model: step {ckpt['step']}, {sum(p.numel() for p in ut.parameters()):,} params")
else:
    print("No checkpoint found!"); sys.exit(1)
ut.eval()

# Load store
store = TrajectoryStore()
store_path = os.path.join(CKPT, "trajectory_store.pkl")
if os.path.exists(store_path):
    store.load(store_path)
print(f"Store: {store.total_stored} total, {len(store.hierarchical)} hierarchical")

# Load data for perception
npy = os.path.join(os.path.dirname(__file__), "real_data", "war_and_peace_boundary.npy")
if not os.path.exists(npy):
    npy = os.path.join(os.path.dirname(__file__), "real_data", "war_and_peace.npy")
data = np.load(npy, mmap_mode='r').astype(np.int32)
total = len(data)
print(f"Data: {total/1e6:.2f}M tokens")
rng = np.random.RandomState()

# ============================================================
# CONSOLIDATION CYCLE
# ============================================================
print("\n" + "=" * 60)
print("CONSOLIDATION REPORT")
print("=" * 60)

# 1. PERCEPTION: autoencode 10 random blocks, measure reconstruction
print("\n[1] PERCEPTION — Autoencode 10 blocks")
recon_accuracy = []
for i in range(10):
    pos = rng.randint(0, max(1, total - 64))
    ids = [int(x) for x in data[pos:pos+64] if 0 < x < VT]
    if len(ids) < 10: continue
    
    inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        _, scores = ut(inp, return_scores=True)
        pred = scores[0].argmax(dim=-1).tolist()
    correct = sum(1 for p, t in zip(pred, ids) if p == t)
    recon_accuracy.append(correct / len(ids))

print(f"  Mean reconstruction accuracy: {np.mean(recon_accuracy):.1%}")

# 2. GENERATION: try different seeds
print("\n[2] GENERATION — Various seeds")
seeds = ['привет', 'Пьер', 'князь Андрей', 'Наташа', 'война', 'мир']
for seed in seeds:
    ids = [cv.SENT_OPEN_IDX, cv.WORD_OPEN_IDX] + cv.encode(seed)[1:-1] + [cv.WORD_CLOSE_IDX, cv.SENT_CLOSE_IDX]
    ids_out = list(ids)
    with torch.no_grad():
        for _ in range(40):
            _, sc = ut(torch.tensor([ids_out], dtype=torch.long, device=DEVICE), return_scores=True)
            logits = sc[0, -1] / 0.6
            _, idx = torch.topk(logits, 20)
            p = torch.softmax(logits[idx], dim=-1)
            nt = idx[torch.multinomial(p, 1)].item()
            ids_out.append(nt)
    text = cv.decode(ids_out)
    # Extract words between <W> tags
    import re
    w_tags = re.findall(r'<W>(.*?)</W>', text)
    print(f"  '{seed}' -> {text[:90]} ({len(w_tags)} words)")

# 3. STORE ANALYSIS: what patterns exist?
print(f"\n[3] STORE ANALYSIS — {store.total_stored} trajectories")
if store.hierarchical:
    # Average word count per trajectory
    avg_words = np.mean([len(h.word_boundaries) for h in store.hierarchical if h.word_boundaries])
    print(f"  Avg words per trajectory: {avg_words:.1f}")
    
    # Most common words (by centroid clustering)
    all_words = []
    for h in store.hierarchical:
        text = h.text.replace('<S>','').replace('</S>','').replace('<W>','').replace('</W>','')
        all_words.extend(text.split())
    
    from collections import Counter
    common = Counter(all_words).most_common(15)
    print(f"  Most frequent words in store: {common[:10]}")

# 4. SELF-REFLECTION: trajectory quality metrics
print(f"\n[4] SELF-REFLECTION — Trajectory quality")
if store.hierarchical:
    lengths = [h.length for h in store.hierarchical]
    n_words = [len(h.word_boundaries) for h in store.hierarchical]
    weights = [h.word_weights.mean() for h in store.hierarchical if len(h.word_weights) > 0]
    print(f"  Avg trajectory length: {np.mean(lengths):.1f} tokens")
    print(f"  Avg words per entry: {np.mean(n_words):.1f}")
    if weights:
        print(f"  Avg word weight: {np.mean(weights):.3f} (range {np.min(weights):.3f}-{np.max(weights):.3f})")

# 5. CONCEPT DISCOVERY: cluster trajectory centroids
print(f"\n[5] CONCEPT DISCOVERY — Trajectory clustering")
if store.total_stored >= 5:
    centroids = np.array([h.sentence_centroid for h in store.hierarchical])
    # Simple: find pairs with high cosine similarity
    similarities = []
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            sim = np.dot(centroids[i], centroids[j]) / (np.linalg.norm(centroids[i]) * np.linalg.norm(centroids[j]) + 1e-8)
            if sim > 0.5:
                similarities.append((sim, i, j))
    
    similarities.sort(key=lambda x: x[0], reverse=True)
    print(f"  High-similarity pairs (>0.5): {len(similarities)}")
    for sim, i, j in similarities[:5]:
        t1 = store.hierarchical[i].text[:30]
        t2 = store.hierarchical[j].text[:30]
        print(f"    {sim:.3f}: '{t1}' ↔ '{t2}'")

print("\n" + "=" * 60)
print("CONTINUOUS CONSOLIDATION — Ctrl+C to stop")
print("=" * 60)

cycle = 0
total_perceived = 0
total_generated = 0
total_synthesized = 0
total_stored = store.total_stored
t0 = time.time()
last_step = ckpt['step']
rng = np.random.RandomState()

# Latent code operator for trajectory synthesis
latent_op = LatentCodeOperator(store, c128)

while True:
    try:
        cycle += 1
        
        # Reload model if training updated it
        if os.path.exists(wp):
            ckpt = torch.load(wp, map_location='cpu', weights_only=True)
            if ckpt['step'] > last_step:
                ut.load_state_dict(ckpt['ut'], strict=False)
                last_step = ckpt['step']
        
        # PERCEPTION: read random block, autoencode
        pos = rng.randint(0, max(1, total - 128))
        ids = [int(x) for x in data[pos:pos+128] if 0 < x < VT]
        if len(ids) >= 12:
            inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                emb = ut.embed(inp)
                _, scores = ut(inp, return_scores=True)
                pred = scores[0].argmax(dim=-1).tolist()
            correct = sum(1 for p, t in zip(pred, ids) if p == t)
            acc = correct / len(ids)
            
            # Extract hierarchical metadata
            htraj = HierarchicalTrajectory(
                symbol_trajectory=emb[0].cpu().numpy(),
                word_boundaries=[], word_centroids=np.zeros((0,128)),
                word_weights=np.zeros(0), connection_coords=np.zeros((0,128)),
                sentence_centroid=emb[0].mean(dim=0).cpu().numpy(),
                text=cv.decode(ids)[:40], ids=ids,
            )
            store.store_hierarchical(htraj)
            total_perceived += 1
            total_stored = store.total_stored
        
        # GENERATION: try random seed every 5 cycles
        if cycle % 5 == 0:
            seed_words = ['привет','Пьер','князь','Наташа','война','мир','солнце']
            seed = seed_words[cycle % len(seed_words)]
            gids = [cv.SENT_OPEN_IDX, cv.WORD_OPEN_IDX] + cv.encode(seed)[1:-1] + [cv.WORD_CLOSE_IDX, cv.SENT_CLOSE_IDX]
            ids_out = list(gids)
            with torch.no_grad():
                for _ in range(150):
                    _, sc = ut(torch.tensor([ids_out], dtype=torch.long, device=DEVICE), return_scores=True)
                    logits = sc[0, -1] / 0.6
                    
                    # Strong repetition penalty: block if last 3 are same char
                    if len(ids_out) >= 3 and ids_out[-1] == ids_out[-2] == ids_out[-3]:
                        logits[ids_out[-1]] -= 10.0
                    # Penalize token if it appears in last 8 positions
                    for t in set(ids_out[-8:]):
                        if t < VT: logits[t] -= 3.0
                    
                    _, idx = torch.topk(logits, 20)
                    p = torch.softmax(logits[idx], dim=-1)
                    nt = idx[torch.multinomial(p, 1)].item()
                    ids_out.append(nt)
                    
                    # Stop at sentence boundary
                    if nt == cv.SENT_CLOSE_IDX and len(ids_out) > 20:
                        break
            import re
            gen_text = cv.decode(ids_out)
            # Strip boundary tokens, preserve word spacing
            clean = gen_text.replace('</W><W>', ' ').replace('</W></S><S><W>', ' ').replace('</S><S>', ' ')
            clean = clean.replace('<S>','').replace('</S>','').replace('<W>','').replace('</W>','')
            clean = re.sub(r'\s+', ' ', clean).strip()
            total_generated += 1
        
        # PROGRESS every 10 cycles
        if cycle % 5 == 0:
            elapsed = time.time() - t0
            vram = torch.cuda.memory_allocated() / 1e9
            print(f"  [{cycle:>4d}] perceived={total_perceived} gen={total_generated} "
                  f"store={total_stored} | VRAM={vram:.1f}GB | {elapsed:.0f}s", flush=True)
            
            # Show latest generation
            if total_generated > 0:
                print(f"         gen: '{clean[:500]}'")
        
        # FULL REPORT every 50 cycles
        if cycle % 50 == 0:
            print(f"\n  === REPORT @ cycle {cycle} ===")
            print(f"  Store: {store.total_stored} total, {len(store.hierarchical)} hierarchical")
            if store.hierarchical:
                n_words = [len(h.word_boundaries) for h in store.hierarchical if h.word_boundaries]
                if n_words:
                    print(f"  Avg words/entry: {np.mean(n_words):.1f}")
            # Concept discovery
            if len(store.hierarchical) >= 5:
                centroids = np.array([h.sentence_centroid for h in store.hierarchical])
                sims = []
                for i in range(len(centroids)):
                    for j in range(i+1, len(centroids)):
                        s = np.dot(centroids[i], centroids[j]) / (np.linalg.norm(centroids[i])*np.linalg.norm(centroids[j])+1e-8)
                        if s > 0.5: sims.append(s)
                print(f"  Semantic clusters (>0.5 sim): {len(sims)} pairs")
            print(f"  {'='*50}\n")
        
        # Save store periodically + synthesize new trajectories
        if cycle % 100 == 0:
            # Synthesize new trajectories from existing ones
            if store.total_stored >= 10:
                for _ in range(5):
                    idx = rng.randint(0, store.total_stored)
                    query_ids = store.ids_list[idx][:10]
                    if len(query_ids) < 3: continue
                    synth_traj = latent_op.synthesize(query_ids, top_k=5)
                    if synth_traj is not None and len(synth_traj) >= 4:
                        htraj = HierarchicalTrajectory(
                            symbol_trajectory=synth_traj,
                            word_boundaries=[], word_centroids=np.zeros((0,128)),
                            word_weights=np.zeros(0), connection_coords=np.zeros((0,128)),
                            sentence_centroid=synth_traj.mean(axis=0),
                            text=f"SYNTH:{cv.decode(query_ids)[:20]}", ids=query_ids,
                        )
                        store.store_hierarchical(htraj)
                        total_synthesized += 1
            
            store.save(store_path)
            if total_synthesized > 0:
                print(f"         synthesized: {total_synthesized} new trajectories")
    
    except KeyboardInterrupt:
        print(f"\nStopped. Saving store ({total_stored} entries)...")
        store.save(store_path)
        break
