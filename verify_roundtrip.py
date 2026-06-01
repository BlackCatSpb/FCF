"""Verify roundtrip encode/decode of trajectories."""
import sys, os
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import pickle, numpy as np
from coordinate_packer import CoordinatePacker
from eva.symbolic.trajectory_store import TrajectoryStore
from eva.symbolic.bpe_tokenizer import BPEVocab

packer = CoordinatePacker()
cv = BPEVocab()

# ─── Load store ───
print("Loading TrajectoryStore...")
store = TrajectoryStore()
store.load(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\trajectory_store_v5.pkl')
print(f"  {store.total_stored} trajectories loaded\n")

# ─── Test: random sample ───
rng = np.random.RandomState(42)
indices = list(range(min(store.total_stored, 27061)))
rng.shuffle(indices)
test_indices = indices[:30]  # 30 random sentences

print("="*70)
print("VERIFICATION: trajectory -> decode -> text")
print("="*70)

errors = 0
for idx in test_indices:
    htraj = store.hierarchical[idx]
    traj = htraj.symbol_trajectory  # [L, 384]
    stored_tokens = htraj.ids
    
    # Decode coordinates -> token IDs
    decoded_ids = [packer.unpack_token(traj[t])['token_id'] for t in range(traj.shape[0])]
    
    # Compare token-level
    token_match = decoded_ids == stored_tokens
    n_err = sum(1 for a, b in zip(decoded_ids, stored_tokens) if a != b)
    
    # Decode to text (skip special tokens for readability)
    original_text = cv.decode(stored_tokens, skip_special=True)
    decoded_text = cv.decode(decoded_ids, skip_special=True)
    
    status = "OK" if token_match else f"ERR({n_err})"
    if not token_match:
        errors += 1
    
    print(f"\n[{idx}] {traj.shape[0]} tokens {status}")
    print(f"  ORIG: {original_text[:120]}")
    print(f"  DEC:  {decoded_text[:120]}")
    
    if not token_match:
        for t in range(min(3, traj.shape[0])):
            if decoded_ids[t] != stored_tokens[t]:
                act = cv.decode([stored_tokens[t]], skip_special=False)[:10]
                got = cv.decode([decoded_ids[t]], skip_special=False)[:10]
                print(f"    pos {t}: expected '{act}' ({stored_tokens[t]}), got '{got}' ({decoded_ids[t]})")

# ─── Full scan ───
print(f"\n{'='*70}")
print(f"FULL SCAN: all {store.total_stored} trajectories...")
print('='*70)

total_errors = 0
total_tokens = 0
error_indices = []

for idx in range(store.total_stored):
    htraj = store.hierarchical[idx]
    traj = htraj.symbol_trajectory
    stored_tokens = htraj.ids
    
    decoded_ids = [packer.unpack_token(traj[t])['token_id'] for t in range(traj.shape[0])]
    total_tokens += len(stored_tokens)
    
    if decoded_ids != stored_tokens:
        n_err = sum(1 for a, b in zip(decoded_ids, stored_tokens) if a != b)
        total_errors += n_err
        error_indices.append((idx, n_err, len(stored_tokens)))
        
        if len(error_indices) <= 5:
            print(f"  ERROR #{len(error_indices)} at traj {idx}: {n_err}/{len(stored_tokens)} mismatches")
            for t in range(min(3, traj.shape[0])):
                if decoded_ids[t] != stored_tokens[t]:
                    act = cv.decode([stored_tokens[t]], skip_special=False)[:10]
                    got = cv.decode([decoded_ids[t]], skip_special=False)[:10]
                    print(f"    pos {t}: exp '{act}' ({stored_tokens[t]}), got '{got}' ({decoded_ids[t]})")
    
    if (idx + 1) % 5000 == 0:
        pct = (idx + 1) / store.total_stored * 100
        print(f"  {idx+1}/{store.total_stored} ({pct:.0f}%) — errors so far: {total_errors}")

print(f"\n{'='*70}")
print(f"РЕЗУЛЬТАТ")
print(f"{'='*70}")
print(f"  Trajectories:   {store.total_stored:,}")
print(f"  Total tokens:   {total_tokens:,}")
print(f"  Token errors:   {total_errors:,}")
print(f"  Trajs w/ errors:{len(error_indices)}")
if total_errors == 0:
    print(f"  >>> 100% PERFECT ROUNDTRIP <<<")
else:
    accuracy = (total_tokens - total_errors) / total_tokens * 100
    print(f"  Accuracy:       {accuracy:.6f}%")
    print(f"  Error trajs:    {[e[0] for e in error_indices[:10]]}")
print(f"{'='*70}")
