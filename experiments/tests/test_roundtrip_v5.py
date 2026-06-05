"""
Round-trip test: sentence → coordinates → reconstruction.
Verifies the coordinate space preserves information.
"""
import torch, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.char_vocab import CharacterVocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cv = CharacterVocab()

# ─── Test 1: Clean model (128-dim, trained) ───
print("=" * 60)
print("TEST 1: Clean model (161 vocab, 128 dim, trained)")
print("=" * 60)

ckpt_path = 'checkpoints/v3/clean_step_40000.pt'
if os.path.exists(ckpt_path):
    model = UnifiedMultidimensionalTransformer().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.eval()
    print(f"Loaded {ckpt_path}")
else:
    print(f"No checkpoint at {ckpt_path}, using fresh model")
    model = UnifiedMultidimensionalTransformer().to(device)
    model.eval()

test_sentences = [
    "привет как дела",
    "это проверка координатного пространства",
    "нейронная сеть учится предсказывать",
]

for sent in test_sentences:
    print(f"\n--- Input: {sent!r} ---")
    ids = cv.encode(sent)  # [BOS, ... , EOS]
    inp = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        h = model.forward(inp, return_scores=True)[0]

    # Round-trip via nearest neighbor in coordinate space:
    # For each h[t], find closest embedding
    sym_coords = model.embed.coordinates  # [V, D]
    h_flat = h[0]  # [L, D]
    dists = torch.cdist(h_flat, sym_coords, p=2)  # [L, V]
    nn_ids = dists.argmin(dim=-1).tolist()

    orig_str = cv.decode(ids, skip_special=True)
    recon_str = cv.decode(nn_ids, skip_special=True)

    specials = {0, 1, 2, 3, 156, 157, 158, 159, 160}
    tok_acc = sum(1 for a, b in zip(ids, nn_ids) if a == b and a not in specials)
    tok_total = sum(1 for a in ids if a not in specials)
    acc = tok_acc / max(tok_total, 1)

    mean_h = h[0].mean(dim=0).norm().item()
    nn_dist_mean = dists.min(dim=-1)[0].mean().item()
    print(f"  Original: {orig_str!r}")
    print(f"  NN-reconstructed: {recon_str!r}")
    print(f"  Token accuracy: {acc:.2%} ({tok_acc}/{tok_total})")
    print(f"  Mean h norm: {mean_h:.4f}, Mean NN dist: {nn_dist_mean:.4f}")

    # Coordinate clustering: same chars closer than different chars
    if len(ids) > 2:
        char = ids[2]
        char_positions = [i for i, tid in enumerate(ids) if tid == char and i < len(ids)]
        if len(char_positions) >= 2:
            h_vals = h[0, char_positions]
            intra_dist = torch.cdist(h_vals, h_vals, p=2).mean().item()
            other_pos = [i for i, tid in enumerate(ids) if tid != char and tid >= 4]
            if other_pos:
                other_h = h[0, other_pos[0]:other_pos[0]+1]
                inter_dist = torch.cdist(h_vals, other_h, p=2).mean().item()
                ratio = intra_dist / max(inter_dist, 1e-8)
                print(f"  Char #{char} intra:{intra_dist:.4f} inter:{inter_dist:.4f} "
                      f"ratio:{ratio:.4f} {'OK' if ratio < 0.8 else 'BAD'}")

# ─── Test 2: Phase 1 model (4101 vocab, 384 dim, untrained) ───
print("\n" + "=" * 60)
print("TEST 2: Phase 1 model (BPE 4101 vocab, 384 dim, untrained)")
print("=" * 60)

from eva.symbolic.phase1_model import UnifiedMultidimensionalTransformerV2
from eva.symbolic.bpe_tokenizer import BPEVocab

model2 = UnifiedMultidimensionalTransformerV2().to(device)
model2.eval()
print(f"Params: {sum(p.numel() for p in model2.parameters()):,}")

bpv = BPEVocab()

for sent in test_sentences:
    ids = bpv.encode(sent)
    inp = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        h = model2.forward(inp, return_scores=True)[0]

    # NN reconstruction
    sym = model2.embed.weight  # [V, D]
    h_flat = h[0]
    dists = torch.cdist(h_flat, sym, p=2)
    nn_ids = dists.argmin(dim=-1).tolist()

    orig_str = bpv.decode(ids, skip_special=True)
    recon_str = bpv.decode(nn_ids, skip_special=True)

    specials = {bpv.PAD_IDX, bpv.UNK_IDX, bpv.BOS_IDX, bpv.EOS_IDX,
                bpv.GAP_FILLER_IDX, bpv.WORD_OPEN_IDX, bpv.WORD_CLOSE_IDX,
                bpv.SENT_OPEN_IDX, bpv.SENT_CLOSE_IDX}
    tok_acc = sum(1 for a, b in zip(ids, nn_ids) if a == b and a not in specials)
    tok_total = sum(1 for a in ids if a not in specials)

    h_mean = h[0].mean(dim=0).norm().item()
    h_pairwise = torch.cdist(h[0], h[0], p=2).mean().item()
    nn_dist_mean = dists.min(dim=-1)[0].mean().item()
    dim_var = h[0].std(dim=0).mean().item()

    print(f"\n--- Input: {sent!r} (BPE/384-dim, untrained) ---")
    print(f"  Original IDs: {ids}")
    print(f"  NN-recon IDs: {nn_ids}")
    print(f"  Match: {sum(1 for a,b in zip(ids,nn_ids) if a==b)}/{len(ids)}")
    print(f"  Mean h norm: {h_mean:.2f}, NN dist: {nn_dist_mean:.4f}")
    print(f"  Pairwise h-dist: {h_pairwise:.2f}, Dim var: {dim_var:.4f}")

# ─── Test 3: Coordinate topology check ───
print("\n" + "=" * 60)
print("TEST 3: Coordinate topology — similar chars cluster?")
print("=" * 60)

# With clean model, check that 'а' and 'я' are closer than 'а' and 'б'
model.eval()
pairs = [
    ('а', 'б', 'similar'),   # both vowels? no, 'а' is vowel 'б' is consonant
    ('а', 'я', 'similar'),   # vowel pair
    ('м', 'н', 'similar'),   # both sonorant consonants
    ('а', '0', 'different'), # char vs digit
    ('в', '.', 'different'), # char vs punctuation
]

for c1, c2, label in pairs:
    ids1 = cv.encode(c1)
    ids2 = cv.encode(c2)
    # skip BOS/EOS, take the actual char ID
    id1 = [i for i in ids1 if i not in (cv.BOS_IDX, cv.EOS_IDX, cv.PAD_IDX)][0]
    id2 = [i for i in ids2 if i not in (cv.BOS_IDX, cv.EOS_IDX, cv.PAD_IDX)][0]

    inp = torch.tensor([[cv.BOS_IDX, id1, cv.EOS_IDX]], device=device)
    with torch.no_grad():
        h1 = model.forward(inp, return_scores=True)[0]
    z1 = h1[0, 1]  # the actual char position

    inp = torch.tensor([[cv.BOS_IDX, id2, cv.EOS_IDX]], device=device)
    with torch.no_grad():
        h2 = model.forward(inp, return_scores=True)[0]
    z2 = h2[0, 1]

    dist = (z1 - z2).norm().item()
    print(f"  Dist({c1!r}, {c2!r}) [{label}]: {dist:.4f}")

# ─── Summary ───
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
# Use Phase 1 model to test dimension activity
h_test = h[0]  # from last forward above
dim_activity = h_test.std(dim=0)  # [384]
active = (dim_activity > dim_activity.mean()).sum().item()
print(f"Phase 1 model: {active}/{h_test.shape[-1]} dims above mean activity")
print(f"  Top-8 dim indices: {dim_activity.argsort(descending=True)[:8].tolist()}")
