"""
EVA — Full Encode→Decode Cycle.

ENCODE: text → tokens → coordinates → metadata (trajectory + affinity)
DECODE: metadata → instruction → transformer → text

The complete loop: text enters, metadata is extracted, instruction formed,
transformer executes instruction, text comes out.
"""

import torch, torch.nn.functional as F, numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer

cv = CharacterVocab()
VT = 157

print("=" * 60)
print("EVA — Full Encode→Decode Cycle")
print("=" * 60)

# ============================================================
# Load model and coordinates
# ============================================================
evolved = torch.load(os.path.join(CKPT_DIR, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE)
affinity = evolved['affinity']

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=24)
if DEVICE == 'cuda':
    ut = ut.cuda()
ut.set_symbol_coordinates(coords.to(DEVICE))

ckpt = torch.load(os.path.join(CKPT_DIR, "word_weights.pt"), map_location='cpu', weights_only=True)
ut.load_state_dict(ckpt['model'], strict=False)
ut.eval()
print(f"Model loaded")

# ============================================================
# ENCODE phase: text → metadata
# ============================================================
def encode_text(text):
    """Encode text into metadata (trajectory + information)."""
    ids = cv.encode(text)[1:-1]  # strip BOS/EOS
    if len(ids) == 0:
        return None
    
    inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        # Get coordinate embeddings
        emb = ut.embed(inp)  # [1, L, 24] — symbol coordinates
        
        # Get transformer output (the processed coordinates)
        output, _ = ut(inp, return_scores=True)
        
        # Extract metadata
        L = emb.shape[1]
        
        # Adjacent affinities from precomputed matrix
        adj_aff = []
        for i in range(L - 1):
            if 0 < ids[i] < VT and 0 < ids[i+1] < VT:
                adj_aff.append(float(affinity[ids[i], ids[i+1]]))
            else:
                adj_aff.append(0.5)
        
        # Trajectory statistics
        trajectory = emb[0].cpu().numpy()  # [L, 24]
        step_vectors = trajectory[1:] - trajectory[:-1]  # [L-1, 24]
        step_norms = np.linalg.norm(step_vectors, axis=1)
        trajectory_length = step_norms.sum()
        
        # Centroid (text-level representation)
        centroid = trajectory.mean(axis=0)
        
        metadata = {
            'text': text,
            'ids': ids,
            'length': L,
            'trajectory': trajectory,
            'centroid': centroid,
            'trajectory_length': trajectory_length,
            'step_norms': step_norms,
            'adj_aff': adj_aff,
            'mean_affinity': np.mean(adj_aff) if adj_aff else 0.5,
            'min_affinity': np.min(adj_aff) if adj_aff else 0.5,
        }
        
        return metadata

# ============================================================
# DECODE phase: metadata → instruction → text
# ============================================================
def decode_from_metadata(metadata):
    """Given metadata, form instruction and execute via transformer."""
    ids = metadata['ids']
    inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        _, scores = ut(inp, return_scores=True)
        pred = scores[0].argmax(dim=-1).tolist()
        text = cv.decode(pred)
        correct = sum(1 for p, t in zip(pred, ids) if p == t)
        accuracy = correct / len(ids)
    
    return text, accuracy

# ============================================================
# Test: complete cycle on example sentences
# ============================================================
print("\n[ENCODE→DECODE] Testing full cycle...")

test_texts = [
    "привет мир",
    "человек идет",
    "солнце светит",
    "мама мыла раму",
    "я люблю программирование",
]

for text in test_texts:
    meta = encode_text(text)
    if meta is None:
        print(f"  '{text}' → SKIP")
        continue
    
    result_text, acc = decode_from_metadata(meta)
    status = "OK" if acc >= 1.0 else f"ERR ({meta['length'] - int(acc*meta['length'])} wrong)"
    
    print(f"  '{text}'")
    print(f"    Encode: {meta['length']} symbols, "
          f"traj_len={meta['trajectory_length']:.2f}, "
          f"affinity μ={meta['mean_affinity']:.3f} min={meta['min_affinity']:.3f}")
    print(f"    Decode: '{result_text}' [{status}]")

# ============================================================
# Advanced: encode → detect critical transitions → decode
# ============================================================
print("\n[CRITICAL] Detecting weak spots in trajectories...")

sentence = "трансформер понимает текст"
meta = encode_text(sentence)

if meta and len(meta['adj_aff']) > 0:
    # Find transition with lowest affinity
    worst_idx = np.argmin(meta['adj_aff'])
    worst_from = meta['ids'][worst_idx]
    worst_to = meta['ids'][worst_idx + 1]
    worst_aff = meta['adj_aff'][worst_idx]
    
    ch_from = cv.decode([worst_from])
    ch_to = cv.decode([worst_to])
    print(f"  Sentence: '{sentence}'")
    print(f"  Weakest transition: '{ch_from}'→'{ch_to}' (affinity={worst_aff:.4f})")
    
    # Show top 3 strongest transitions
    if len(meta['adj_aff']) >= 3:
        top3 = np.argsort(meta['adj_aff'])[-3:][::-1]
        print(f"  Strongest transitions:")
        for idx in top3:
            f = meta['ids'][idx]
            t = meta['ids'][idx + 1]
            a = meta['adj_aff'][idx]
            print(f"    '{cv.decode([f])}'→'{cv.decode([t])}' (affinity={a:.4f})")

# ============================================================
# Round-trip: text → encode → metadata → decode → text
# ============================================================
print("\n[ROUND-TRIP] Multiple sentences round-trip test...")

total_correct = 0
total_chars = 0

round_trip_tests = [
    "привет мир как дела",
    "сегодня хорошая погода",
    "программирование это искусство",
    "знания хранятся в метаданных",
]

for text in round_trip_tests:
    meta = encode_text(text)
    if meta is None: continue
    result, acc = decode_from_metadata(meta)
    total_correct += int(acc * meta['length'])
    total_chars += meta['length']
    status = "✓" if acc >= 1.0 else f"✗ ({acc:.0%})"
    print(f"  {status} '{text}' → '{result}'")

if total_chars > 0:
    print(f"\n  Overall round-trip accuracy: {total_correct}/{total_chars} ({total_correct/total_chars:.1%})")

print("\nDone.")
