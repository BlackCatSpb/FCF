"""
EVA — Fresh training on full_corpus_encoded.npy (106M tokens).
RecursiveTensorPotentialField + composition loss.
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, sys, os, time, re
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

DEVICE = 'cuda'
CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
os.makedirs(CKPT, exist_ok=True)

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.fractal_conv import TrajectoryPredictor
from eva.symbolic.trajectory_store import TrajectoryStore, HierarchicalTrajectory
cv = CharacterVocab(); V = cv.vocab_size

print("=" * 60)
print("EVA — Recursive Training on Full Corpus")
print("=" * 60)
print(f"Vocab: {V} tokens")

c128 = torch.zeros(V, 128, device=DEVICE)
g = torch.Generator(device=DEVICE).manual_seed(42)
c128[:, :] = torch.randn(V, 128, generator=g, device=DEVICE) * 0.02
c128 = c128 / c128.norm(dim=-1, keepdim=True).clamp(1e-8)

ut = UnifiedMultidimensionalTransformer(vocab_size=V, coord_dim=128, max_levels=8,
    total_heads=32, num_layers=6, d_ff=128).to(DEVICE)
ut.set_symbol_coordinates(c128)

traj_predictor = TrajectoryPredictor(128).to(DEVICE)
traj_loss_weight = 0.1

# Init base TPF with diagonal prior
with torch.no_grad():
    ut.tensor_potential.base_tpf.P.data.fill_(0.01)
    for i in range(ut.tensor_potential.num_symbols):
        ut.tensor_potential.base_tpf.P.data[i, i, :] = 1.0

# Try resume from full_latest.pt (migrate old TPF.P → base_tpf.P)
start_step = 0
ckpt_path = os.path.join(CKPT, "full_latest.pt")
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    # Remove keys with shape mismatches (depth_scale changed scalar→vector[8])
    old_sd = ckpt['ut']
    for k in list(old_sd.keys()):
        if k in ut.state_dict() and old_sd[k].shape != ut.state_dict()[k].shape:
            print(f"  Skip shape mismatch: {k} {old_sd[k].shape} → {ut.state_dict()[k].shape}")
            del old_sd[k]
    ut.load_state_dict(old_sd, strict=False)
    if 'tensor_potential.P' in old_sd:
        print("Migrating old TPF.P → RecursiveTPF.base_tpf.P")
        with torch.no_grad():
            ut.tensor_potential.base_tpf.P.data.copy_(old_sd['tensor_potential.P'])
    if 'traj_predictor' in ckpt:
        traj_predictor.load_state_dict(ckpt['traj_predictor'])
    start_step = ckpt.get('step', 0)
    print(f"Resumed from step {start_step}")

print(f"RecursiveTPF: {sum(p.numel() for p in ut.tensor_potential.parameters()):,} params")
print(f"WVF: {sum(p.numel() for p in ut.word_valence.parameters()):,} params")
total_model = sum(p.numel() for p in ut.parameters()) + sum(p.numel() for p in traj_predictor.parameters())
print(f"Total: {total_model:,} params")

npy_path = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_encoded.npy")
if not os.path.exists(npy_path):
    print(f"ERROR: {npy_path} not found. Run encode_full_corpus.py first.")
    sys.exit(1)
data = np.load(npy_path, mmap_mode='r').astype(np.int32)
total = len(data)
print(f"Data: {total/1e6:.2f}M tokens")

SENT_CLOSE = cv.SENT_CLOSE_IDX
blocks = []
i = 0
while i < total:
    start = i
    while i < total and data[i] != SENT_CLOSE:
        i += 1
    if i < total:
        blocks.append(data[start:i+1].tolist())
        i += 1
    else:
        break
print(f"Blocks: {len(blocks):,}")
sent_ptr = start_step * 8  # approximate continuation

store = TrajectoryStore(max_trajectories=50000)

STEPS = 100000; LR = 5e-3; B = 12; ML = 192
opt = torch.optim.AdamW(list(ut.parameters()) + list(traj_predictor.parameters()), lr=LR, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
rng = np.random.RandomState(seed=42)
best_curvature = 1e9

def traj_loss_fn(hidden_states, token_ids):
    with torch.no_grad():
        base = ut.embed(token_ids[:, 1:])
        sym = ut.subspace(token_ids[:, 1:])
        next_hidden = base + F.pad(sym, (0, ut.coord_dim - 32))
        next_hidden = next_hidden / (next_hidden.norm(dim=-1, keepdim=True) + 1e-8)
    current = hidden_states[:, :-1, :]
    delta_pred = traj_predictor(current)
    next_pred = current + delta_pred
    next_pred = next_pred / (next_pred.norm(dim=-1, keepdim=True) + 1e-8)
    return F.mse_loss(next_pred, next_hidden)

def extract_hierarchical(ut, ids_list, text):
    if len(ids_list) < 5: return None
    inp = torch.tensor([ids_list], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        x, scores, weights = ut(inp, return_scores=True, return_weights=True)
    traj = x[0].cpu().numpy()
    w_open = cv.WORD_OPEN_IDX; w_close = cv.WORD_CLOSE_IDX
    boundaries = []; in_word = False; start = 0
    for i, tid in enumerate(ids_list):
        if tid == w_open: in_word = True; start = i + 1
        elif tid == w_close and in_word: boundaries.append((start, i)); in_word = False
    if len(boundaries) < 1: return None
    w_centroids = np.zeros((len(boundaries), 128))
    w_weights = np.zeros(len(boundaries))
    for wi, (s, e) in enumerate(boundaries):
        if e > s:
            w_centroids[wi] = traj[s:e].mean(axis=0)
            w_weights[wi] = weights[0, s:e].mean().cpu().item()
    conn_coords = np.zeros((max(0, len(boundaries)-1), 128))
    for wi in range(len(boundaries)-1):
        conn_coords[wi] = w_centroids[wi+1] - w_centroids[wi]
    sent_centroid = traj.mean(axis=0)
    return HierarchicalTrajectory(
        symbol_trajectory=traj, word_boundaries=boundaries,
        word_centroids=w_centroids, word_weights=w_weights,
        connection_coords=conn_coords, sentence_centroid=sent_centroid,
        text=text, ids=ids_list)


def thinking_phase(model, store_obj, data_arr, blocks_arr, step, log_prefix=""):
    model.eval()
    total_blocks = len(blocks_arr)
    with torch.no_grad():
        aff = torch.zeros(V, V, device=DEVICE)
        for _ in range(500):
            pos = rng.randint(0, max(1, total - ML))
            ids = [int(x) for x in data_arr[pos:pos+ML] if 0 < x < V]
            for k in range(len(ids)-1):
                if 0 < ids[k] < V and 0 < ids[k+1] < V:
                    aff[ids[k], ids[k+1]] += 1
        aff = aff / aff.max().clamp(min=1)
        nz = (aff > 0).sum().item()

        model.tensor_potential.init_from_affinity(aff)
        model.topology.topology[:, :, 0] = aff.cpu()

        for _ in range(10):
            pos = rng.randint(0, max(1, total - 64))
            samp = [int(x) for x in data_arr[pos:pos+64] if 0 < x < V]
            if len(samp) >= 16:
                model.update_tensor_potential(
                    torch.tensor([samp], dtype=torch.long, device=DEVICE), lr=0.01)

        for _ in range(5):
            bidx = rng.randint(0, max(1, total_blocks))
            blk = blocks_arr[bidx % total_blocks]
            ht = extract_hierarchical(model, blk, cv.decode(blk)[:40])
            if ht:
                ht = store_obj.consolidate(ht, device=DEVICE)
                store_obj.store_hierarchical(ht)

        model.topology.build_from_store(store_obj)

    model.train()
    return nz


def random_generation(model, blocks_arr, n_samples=3, max_new=30):
    model.eval()
    texts = []
    with torch.no_grad():
        for _ in range(n_samples):
            bidx = rng.randint(0, max(1, len(blocks_arr)))
            blk = blocks_arr[bidx % len(blocks_arr)]
            prompt = [t for t in blk if t >= 4][:4]
            if len(prompt) < 2:
                prompt = [cv._char_to_idx.get('о', 4), cv._char_to_idx.get('н', 5)]
            gen = model.enhanced_generate(prompt, cv, max_new=max_new, temperature=0.8)
            texts.append(gen[:60])
    model.train()
    return texts


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
        sent_cut = 0; word_cut = 0
        for cut in range(len(ids_flat), max(len(ids_flat) - 50, 0), -1):
            if ids_flat[cut - 1] == cv.SENT_CLOSE_IDX and sent_cut == 0:
                sent_cut = cut
            if ids_flat[cut - 1] == cv.WORD_CLOSE_IDX and word_cut == 0:
                word_cut = cut
        if sent_cut > 0:
            ids_flat = ids_flat[:sent_cut]
        elif word_cut > 0:
            ids_flat = ids_flat[:word_cut]
        bt[bi, :len(ids_flat)] = torch.tensor(ids_flat, dtype=torch.long, device=DEVICE)

    ut.train(); hiddens, scores = ut(bt, return_scores=True)
    target = bt[:, 1:].clamp(1, V-1).contiguous(); pred = scores[:, :-1].contiguous(); tm = mask[:, 1:]

    loss = F.cross_entropy(pred.view(-1, V), target.view(-1), reduction='none')
    loss = (loss.view(B, ML-1) * tm).sum() / (tm.sum() + 1e-8)

    h_for_group = hiddens[:, :-1].contiguous()
    group_logits = ut.decoder.group_classifier(h_for_group)
    target_groups = ut.decoder.group_ids[target]
    group_loss = F.cross_entropy(group_logits.view(-1, 4), target_groups.view(-1), reduction='none')
    group_loss = (group_loss.view(B, ML-1) * tm).sum() / (tm.sum() + 1e-8)
    loss = loss + 0.05 * group_loss

    t_loss = traj_loss_fn(hiddens, bt)
    loss = loss + traj_loss_weight * t_loss

    comp_loss = ut.tensor_potential.composition_loss(hiddens)
    loss = loss + 0.01 * comp_loss

    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(traj_predictor.parameters(), 1.0)
    opt.step(); sch.step()

    if s % 50 == 0:
        with torch.no_grad():
            acc = ((pred.argmax(-1) == target) & tm.bool()).sum().item() / (tm.sum() + 1e-8)
        elapsed = int((time.time()-t0)/60)
        print(f"  {s:>6d} | loss={loss.item():.4f} acc={acc:.3f} traj={t_loss.item():.4f} grp={group_loss.item():.3f} comp={comp_loss.item():.4f} | {elapsed}min", flush=True)

    if s % 500 == 0:
        torch.save({'ut': ut.state_dict(), 'traj_predictor': traj_predictor.state_dict(), 'step': s},
            os.path.join(CKPT, "full_latest.pt"))

        nz = thinking_phase(ut, store, data, blocks, s)
        store.save(os.path.join(CKPT, "trajectory_store_full.pkl"))
        print(f"  [think] step={s} topology={nz} store={store.total_stored}", flush=True)

    if s % 2500 == 0:
        gen_texts = random_generation(ut, blocks, n_samples=3, max_new=30)
        print(f"  Gen2500: {gen_texts[0][:40]}|{gen_texts[1][:40]}|{gen_texts[2][:40]}", flush=True)

    if s % 5000 == 0:
        ut.eval()
        from eva.symbolic.validation_suite import evaluate_model
        results = evaluate_model(ut, cv, store, data, total, CKPT, rng)
        print(f"\n  === VALIDATION @ step {s} ===")
        print(f"  Store: {store.total_stored} entries")
        if results.get('coherent_ratio'): print(f"  Coherent: {results['coherent_ratio']:.0%}")
        print(f"  Avg confidence: {results.get('avg_confidence',0):.3f}")
        for seed, text in results.get('generation', {}).items():
            print(f"    '{seed}': {text[:90]}")
        print()
        avg_curv = results.get('avg_curvature', 1e9)
        if avg_curv < best_curvature:
            best_curvature = avg_curv
            torch.save({'ut': ut.state_dict(), 'traj_predictor': traj_predictor.state_dict(), 'step': s, 'curvature': avg_curv},
                os.path.join(CKPT, "full_best.pt"))
            print(f"  [best] curvature={avg_curv:.4f}", flush=True)
        ut.train()

print("Done.")
