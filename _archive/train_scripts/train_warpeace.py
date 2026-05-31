"""
EVA — Hierarchical Training on War & Peace.
Boundary tokens <W></W> <S></S>, adaptive levels, word weights.
"""

import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, sys, os, time, re
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda'; CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
os.makedirs(CKPT, exist_ok=True)

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.fractal_conv import TrajectoryPredictor
from eva.symbolic.trajectory_store import TrajectoryStore, HierarchicalTrajectory
cv = CharacterVocab(); VT = cv.vocab_size

print("=" * 60)
print("EVA — Hierarchical Training (War & Peace)")
print("=" * 60)
print(f"Vocab: {VT} tokens")

# Coordinates
c128 = torch.zeros(VT, 128, device=DEVICE)
g = torch.Generator(device=DEVICE).manual_seed(42)
c128[:, :] = torch.randn(VT, 128, generator=g, device=DEVICE) * 0.02
c128 = c128 / c128.norm(dim=-1, keepdim=True).clamp(1e-8)

ut = UnifiedMultidimensionalTransformer(vocab_size=VT, coord_dim=128, max_levels=8,
    total_heads=32, num_layers=6, d_ff=128).to(DEVICE)
ut.set_symbol_coordinates(c128)

# ===== TrajLoss: auxiliary головка для предсказания следующей координаты =====
traj_predictor = TrajectoryPredictor(128).to(DEVICE)
traj_loss_weight = 0.1  # коэффициент aux loss

# ===== Dilated KV-Cache: храним прореженные K/V для экономии VRAM =====
dilated_kv_active = True  # можно отключить для сравнения
dilated_rates = [1, 2, 4, 8]  # для каждого уровня attention

def apply_dilated_kv(embeddings, level):
    """Subsample embeddings: для level>0 храним каждый rate-й токен."""
    rate = dilated_rates[min(level, len(dilated_rates)-1)]
    if rate <= 1 or not dilated_kv_active:
        return embeddings
    L = embeddings.shape[1]
    indices = torch.arange(0, L, rate, device=embeddings.device)
    return embeddings[:, indices, :]

# Resume from checkpoint if exists
start_step = 0
wp_path = os.path.join(CKPT, "wp_latest.pt")
if os.path.exists(wp_path):
    ckpt = torch.load(wp_path, map_location='cpu', weights_only=True)
    ut.load_state_dict(ckpt['ut'], strict=False)
    if 'traj_predictor' in ckpt:
        traj_predictor.load_state_dict(ckpt['traj_predictor'])
    start_step = ckpt.get('step', 0)
    print(f"Resumed from step {start_step}")

print(f"Model: {sum(p.numel() for p in ut.parameters()):,} params")
traj_params = sum(p.numel() for p in traj_predictor.parameters())
print(f"TrajPredictor: {traj_params:,} params (aux loss)")

# Load & encode War & Peace with boundaries (skip if already have data)
npy_path = os.path.join(os.path.dirname(__file__), "real_data", "war_and_peace_boundary.npy")
if os.path.exists(npy_path) and start_step > 0:
    data = np.load(npy_path, mmap_mode='r').astype(np.int32)
    print(f"Loaded pre-encoded data: {len(data)/1e6:.2f}M tokens")
else:
    print("Encoding War & Peace with boundary tokens...")
    all_ids = []
    for book in [1, 2]:
        path = rf"C:\Users\black\OneDrive\Desktop\Толстой Лев. Война и мир. Книга {book} - royallib.ru.txt"
        with open(path, 'r', encoding='windows-1251') as f:
            raw = f.read()
        raw = re.sub(r'\r\n|\r', '\n', raw)
        sents = re.split(r'(?<=[.!?…])\s+(?=[А-ЯЁA-Z])', raw)
        for s in sents:
            s = s.strip()
            if len(s) >= 4:
                ids = cv.encode_with_boundaries(s)
                if len(ids) >= 5:
                    all_ids.extend(ids)
        print(f"  Book {book}: {len(sents):,} sentences")
    data = np.array(all_ids, dtype=np.int32)
    np.save(npy_path, data)

total = len(data)
print(f"Total: {total/1e6:.2f}M tokens")

# Extract boundary-delimited blocks for sequential training
blocks = []
i = 0
while i < total - 1:
    start = i
    while i < total and data[i] != cv.SENT_CLOSE_IDX:
        i += 1
    if i < total: 
        blocks.append(data[start:i+1].tolist())
        i += 1
print(f"Blocks: {len(blocks):,}")
sent_ptr = 0

# Trajectory store for hierarchical metadata
store = TrajectoryStore(max_trajectories=100000)
store_pkl_path = os.path.join(CKPT, "trajectory_store.pkl")
if os.path.exists(store_pkl_path):
    store.load(store_pkl_path)
    print(f"Store loaded: {store.total_stored} entries")

def extract_hierarchical(ut, ids_list, text):
    """Extract multi-level metadata from autoencoded text."""
    if len(ids_list) < 5: return None
    
    inp = torch.tensor([ids_list], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        x, scores, weights = ut(inp, return_scores=True, return_weights=True)
    
    traj = x[0].cpu().numpy()  # [L, 128]
    
    # Find word boundaries from <W> and </W> tokens
    w_open = cv.WORD_OPEN_IDX
    w_close = cv.WORD_CLOSE_IDX
    boundaries = []
    in_word = False; start = 0
    for i, tid in enumerate(ids_list):
        if tid == w_open:
            in_word = True; start = i + 1
        elif tid == w_close and in_word:
            boundaries.append((start, i))
            in_word = False
    
    if len(boundaries) < 1: return None
    
    # Word centroids and weights
    w_centroids = np.zeros((len(boundaries), 128))
    w_weights = np.zeros(len(boundaries))
    for wi, (s, e) in enumerate(boundaries):
        if e > s:
            w_centroids[wi] = traj[s:e].mean(axis=0)
            w_weights[wi] = weights[0, s:e].mean().cpu().item()
    
    # Connection coords
    conn_coords = np.zeros((max(0, len(boundaries)-1), 128))
    for wi in range(len(boundaries)-1):
        conn_coords[wi] = w_centroids[wi+1] - w_centroids[wi]
    
    sent_centroid = traj.mean(axis=0)
    
    return HierarchicalTrajectory(
        symbol_trajectory=traj, word_boundaries=boundaries,
        word_centroids=w_centroids, word_weights=w_weights,
        connection_coords=conn_coords, sentence_centroid=sent_centroid,
        text=text, ids=ids_list,
    )

STEPS = 100000; LR = 5e-3; B = 8; ML = 128
opt = torch.optim.AdamW(list(ut.parameters()) + list(traj_predictor.parameters()), lr=LR, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
rng = np.random.RandomState(42)


def traj_loss_fn(hidden_states, token_ids):
    """
    TrajLoss: предсказать следующую 128d координату из hidden states последнего слоя.
    hidden_states: [B, L, 128] — выход трансформера (norm_final)
    token_ids: [B, L]
    """
    with torch.no_grad():
        next_hidden = ut.embed(token_ids[:, 1:])  # [B, L-1, 128]
        next_hidden = next_hidden / (next_hidden.norm(dim=-1, keepdim=True) + 1e-8)
    
    current = hidden_states[:, :-1, :]  # [B, L-1, 128]
    delta_pred = traj_predictor(current)  # [B, L-1, 128]
    
    next_pred = current + delta_pred
    next_pred = next_pred / (next_pred.norm(dim=-1, keepdim=True) + 1e-8)
    
    loss = F.mse_loss(next_pred, next_hidden)
    return loss

def multi_level_generate(seed_text, max_new=40, T=0.6):
    """MultiLevelGenerator: encode → retrieve → fuse → decode."""
    ids = cv.encode(seed_text)[1:-1]
    if len(ids) < 2: return seed_text
    
    # Encode query
    htraj = extract_hierarchical(ut, ids, seed_text)
    if htraj is None or store.total_stored < 5:
        # Fallback: simple autoregressive
        ids_out = list(ids)
        with torch.no_grad():
            for _ in range(max_new):
                inp = torch.tensor([ids_out], dtype=torch.long, device=DEVICE)
                _, sc = ut(inp, return_scores=True)
                logits = sc[0, -1] / T
                _, idx = logits.topk(20); p = F.softmax(logits[idx], dim=-1)
                nt = idx[torch.multinomial(p, 1)].item()
                ids_out.append(nt)
        return cv.decode(ids_out)
    
    # Retrieve similar
    similar = store.find_similar_hierarchical(htraj, top_k=5)
    if not similar:
        # Fallback: simple autoregressive
        ids_out = list(ids)
        with torch.no_grad():
            for _ in range(max_new):
                _, sc = ut(torch.tensor([ids_out], dtype=torch.long, device=DEVICE), return_scores=True)
                logits = sc[0, -1] / T
                _, idx = logits.topk(20); p = F.softmax(logits[idx], dim=-1)
                nt = idx[torch.multinomial(p, 1)].item()
                ids_out.append(nt)
        return cv.decode(ids_out)
    
    # Fuse: average the continuation coords from retrieved trajectories
    fused = htraj.symbol_trajectory.copy()
    for sim in similar:
        sim_traj = sim.symbol_trajectory
        min_len = min(len(fused), len(sim_traj))
        if min_len > 0:
            fused[:min_len] = 0.7 * fused[:min_len] + 0.3 * sim_traj[:min_len]
    
    # Decode fused coordinates
    ids_out = list(ids)
    fused_t = torch.tensor(fused, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        for _ in range(max_new):
            inp = torch.tensor([ids_out], dtype=torch.long, device=DEVICE)
            _, sc = ut(inp, return_scores=True)
            logits = sc[0, -1] / T
            
            # Boost from retrieved next tokens
            for sim in similar:
                if len(sim.ids) > len(ids_out):
                    nxt = sim.ids[len(ids_out)]
                    if 0 < nxt < VT:
                        logits[nxt] += 2.0
            
            sl, si = logits.sort(descending=True)
            cp = F.softmax(sl, dim=-1).cumsum(dim=-1)
            cut = (cp > 0.95).nonzero(as_tuple=True)[0]
            k = cut[0].item() + 1 if len(cut) > 0 else 20; k = min(max(k, 3), 40)
            v, idx = logits.topk(k); p = F.softmax(v, dim=-1)
            for t in set(ids_out[-5:]): m = (idx == t).nonzero(as_tuple=True)[0]; p[m] *= 0.3
            p /= p.sum(); nt = idx[torch.multinomial(p, 1)].item()
            if nt <= 0 or nt >= VT: nt = idx[0].item()
            ids_out.append(nt)
    
    return cv.decode(ids_out)


t0 = time.time()
for s in range(1 + start_step, STEPS + 1):
    bt = torch.zeros(B, ML, dtype=torch.long, device=DEVICE)
    mask = torch.ones(B, ML, device=DEVICE)
    for bi in range(B):
        ids_flat = []
        while len(ids_flat) < ML:
            ids_flat.extend(blocks[sent_ptr % len(blocks)])
            sent_ptr += 1
        ids_flat = ids_flat[:ML]
        bt[bi, :len(ids_flat)] = torch.tensor(ids_flat, dtype=torch.long, device=DEVICE)
    
    ut.train(); hiddens, scores = ut(bt, return_scores=True)
    target = bt[:, 1:].clamp(1, VT-1).contiguous(); pred = scores[:, :-1].contiguous(); tm = mask[:, 1:]
    
    # Основной CE-loss
    loss = F.cross_entropy(pred.view(-1, VT), target.view(-1), reduction='none')
    loss = (loss.view(B, ML-1) * tm).sum() / (tm.sum() + 1e-8)
    
    # SubHSM group auxiliary loss: группа из hidden states
    h_for_group = hiddens[:, :-1].contiguous()
    group_logits = ut.decoder.group_classifier(h_for_group)  # [B, L-1, 4]
    target_groups = ut.decoder.group_ids[target]  # [B, L-1]
    group_loss = F.cross_entropy(group_logits.view(-1, 4), target_groups.view(-1), reduction='none')
    group_loss = (group_loss.view(B, ML-1) * tm).sum() / (tm.sum() + 1e-8)
    loss = loss + 0.05 * group_loss
    
    # TrajLoss: предсказание следующей координаты из hidden states
    t_loss = traj_loss_fn(hiddens, bt)
    loss = loss + traj_loss_weight * t_loss
    
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(traj_predictor.parameters(), 1.0)
    opt.step(); sch.step()
    
    if s % 50 == 0:
        with torch.no_grad(): acc = ((pred.argmax(-1) == target) & tm.bool()).sum().item() / (tm.sum() + 1e-8)
        elapsed = int((time.time()-t0)/60)
        tloss_val = t_loss.item()
        gloss_val = group_loss.item()
        print(f"  {s:>6d} | loss={loss.item():.4f} acc={acc:.3f} traj={tloss_val:.4f} grp={gloss_val:.3f} | {elapsed}min", flush=True)
        
        # Mini-consolidation: perception + generation + store (каждые 50 шагов)
        if s % 50 == 0 and store.total_stored < 5000:
            ut.eval()
            with torch.no_grad():
                pos = rng.randint(0, max(1, total - 32))
                pids = [int(x) for x in data[pos:pos+32] if 0 < x < VT]
                if len(pids) >= 8:
                    inp = torch.tensor([pids], dtype=torch.long, device=DEVICE)
                    emb = ut.embed(inp)
                    htraj = HierarchicalTrajectory(
                        symbol_trajectory=emb[0].cpu().numpy(),
                        word_boundaries=[], word_centroids=np.zeros((0,128)),
                        word_weights=np.zeros(0), connection_coords=np.zeros((0,128)),
                        sentence_centroid=emb[0].mean(dim=0).cpu().numpy(),
                        text=cv.decode(pids)[:30], ids=pids,
                    )
                    # Learnable consolidation
                    htraj = store.consolidate(htraj, device=DEVICE)
                    store.store_hierarchical(htraj)
            ut.train()
    
    if s % 500 == 0:
        torch.save({
            'ut': ut.state_dict(),
            'traj_predictor': traj_predictor.state_dict(),
            'step': s
        }, os.path.join(CKPT, "wp_latest.pt"))
        # Build static topology every 5000 steps
        if s % 5000 == 0 and s > 0:
            ut.eval()
            affinity = torch.eye(VT) * 0.5
            with torch.no_grad():
                aff = torch.zeros(VT, VT, device=DEVICE)
                for _ in range(50):
                    ids = [int(x) for x in data[rng.randint(0,total-ML):rng.randint(0,total-ML)+ML] if 0 < x < VT]
                    for k in range(len(ids)-1):
                        if 0 < ids[k] < VT and 0 < ids[k+1] < VT:
                            aff[ids[k], ids[k+1]] += 1
                aff = aff / aff.max().clamp(min=1)
            ut.topology.topology[:, :, 0] = aff.cpu()
            ut.topology.build_from_store(store)
            if s % 5000 == 0:
                nz = (aff > 0).sum().item()
                print(f"         topology: {nz} connections, fast_paths: {len(ut.topology.fast_path_values)}")
        ut.train()
        # Store hierarchical metadata for a sample block
        ut.eval()
        sample = blocks[(s // 500) % len(blocks)]
        htraj = extract_hierarchical(ut, sample, cv.decode(sample)[:40])
        if htraj:
            htraj = store.consolidate(htraj, device=DEVICE)
            store.store_hierarchical(htraj)
        
        # Generate sample every 500 steps
        seeds = ["Я", "Он", "Мы", "—"]
        gen_texts = []
        with torch.no_grad():
            for sd in seeds:
                gids = list(sd.encode('utf-8'))
                ids_out = gids[:3]
                for _ in range(30):
                    inp = torch.tensor([ids_out], dtype=torch.long, device=DEVICE)
                    _, sc = ut(inp, return_scores=True)
                    logits = sc[0, -1] / 0.8
                    sl, si = logits.sort(descending=True)
                    v = sl[:15]; idx = si[:15]; p = F.softmax(v, dim=-1)
                    for t in set(ids_out[-5:]):
                        m = (idx == t).nonzero(as_tuple=True)[0]
                        if len(m) > 0: p[m[0]] *= 0.3
                    p /= p.sum()
                    nt = idx[torch.multinomial(p, 1)].item()
                    if nt < 4: nt = idx[0].item()
                    ids_out.append(nt)
                gen_texts.append(cv.decode(ids_out)[:60])
        print(f"  Gen500: {gen_texts[0][:40]}|{gen_texts[1][:40]}")
        ut.train()
        
        # Save store periodically
        if s % 5000 == 0 and store.total_stored > 0:
            store.save(os.path.join(CKPT, "trajectory_store.pkl"))
    
    if s % 5000 == 0:
        ut.eval()
        from eva.symbolic.validation_suite import evaluate_model
        results = evaluate_model(ut, cv, store, data, total, CKPT, rng)
        
        print(f"\n  === VALIDATION @ step {s} ===")
        print(f"  Store: {store.total_stored} entries")
        
        if results.get('anna_karenina_acc'):
            print(f"  Anna Karenina accuracy: {results['anna_karenina_acc']:.1%} ({results.get('anna_karenina_samples',0)} sentences)")
        
        print(f"  Coherent generations: {results.get('coherent_ratio',0):.0%}")
        print(f"  Avg confidence: {results.get('avg_confidence',0):.3f}")
        print(f"  Avg curvature: {results.get('avg_curvature',0):.4f}")
        print(f"  TrajLoss: {t_loss.item():.4f}")
        
        print(f"\n  Generation samples:")
        for seed, text in results.get('generation', {}).items():
            print(f"    '{seed}': {text[:90]}")
        
        if results.get('multi_level_gen'):
            print(f"\n  Multi-level generation:")
            for seed, text in results['multi_level_gen'].items():
                print(f"    '{seed}': {text[:90]}")
        
        print()
        ut.train()

print("Done.")
