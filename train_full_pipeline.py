"""
EVA Symbolic v8 — Full Training Pipeline.

Phase 3: Symbol reproduction from coordinates (no text, no dataset).
Future: Phase 1+2 (affinity + MDS) for topological coordinates.
"""

import sys, os, time, torch, torch.nn.functional as F, math
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 60)
print("EVA v8 — Symbol Reproduction (no dataset)")
print("=" * 60)
print(f"Device: {DEVICE}")

V = 156  # symbols (without PAD)
PAD = 0

cv = CharacterVocab()

# ============================================================
# Fixed coordinates for 156 symbols (deterministic, distinct)
# ============================================================
print("\n[COORDS] Fixed orthonormal coordinates for 156 symbols")
g = torch.Generator()
g.manual_seed(42)
coords = torch.randn(157, 24, generator=g)  # [157, 24] — included PAD=0, unused
coords[0] = 0.0  # PAD at origin
coords = coords / coords.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # unit sphere
print(f"  Shape: {coords.shape}, min_dist={torch.pdist(coords[1:]).min():.4f}")

# ============================================================
# PHASE 3: Symbol reproduction — ALL 156 symbols, GPU only
# ============================================================
print("\n[PHASE 3] Transformer — 156 symbol reproduction (GPU)")
print("  Goal: reproduce ALL 156 symbols from their coordinates")
print("  100% required. No text. No dataset.")

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=24)
if DEVICE == 'cuda':
    ut = ut.cuda()
ut.set_symbol_coordinates(coords.to(DEVICE))
print(f"  {ut.summary()}")

UT_STEPS = 50000; UT_LR = 1e-3
opt = torch.optim.AdamW(ut.parameters(), lr=UT_LR, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=UT_STEPS)

all_symbols = torch.arange(1, 157, dtype=torch.long, device=DEVICE)  # 1..156
symbol_batch = all_symbols.unsqueeze(0)  # [1, 156]

start = time.time()
last_print = 0
first_100pct_saved = False

for step in range(1, UT_STEPS + 1):
    ut.train()
    _, scores = ut(symbol_batch, return_scores=True)

    loss = F.cross_entropy(scores.view(-1, V+1), symbol_batch.view(-1))

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0)
    opt.step()
    sch.step()

    # Save milestone when 100% first achieved
    if not first_100pct_saved:
        with torch.no_grad():
            pred_check = scores[0].argmax(dim=-1)
            if (pred_check == all_symbols).sum().item() == 156:
                first_100pct_saved = True
                CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
                os.makedirs(CKPT_DIR, exist_ok=True)
                milestone_path = os.path.join(CKPT_DIR, "symbol_100pct.pt")
                torch.save({'model': ut.state_dict(), 'coords': coords, 'step': step}, milestone_path)
                print(f"\n  *** 100% reached at step {step}! Saved: {milestone_path}")
                if loss.item() < 0.01:
                    print("  Early exit: loss < 0.01, training complete.")
                    break

    now = time.time()
    if now - last_print >= 3 or step == 1 or step == UT_STEPS:
        last_print = now
        elapsed = now - start
        eta = (elapsed / step) * (UT_STEPS - step)
        with torch.no_grad():
            predicted = scores[0].argmax(dim=-1)  # [156]
            correct = (predicted == all_symbols).sum().item()
        lr = sch.get_last_lr()[0]
        print(f"  step {step:>5d}/{UT_STEPS} | loss={loss.item():.4f} | "
              f"correct={correct:>3d}/156 ({correct/156:.0%}) | lr={lr:.6f} | "
              f"{elapsed:.0f}s / eta {eta:.0f}s", flush=True)

# ============================================================
# PHASE 4: Final symbol test — 100% or fail
# ============================================================
print(f"\n[PHASE 4] FINAL TEST: reproduce ALL 156 symbols")
ut.eval()
with torch.no_grad():
    _, scores = ut(symbol_batch, return_scores=True)
    predicted = scores[0].argmax(dim=-1)
    correct_count = (predicted == all_symbols).sum().item()

print(f"  Result: {correct_count}/156 ({correct_count/156:.0%})")

# Save weights always
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
os.makedirs(CKPT_DIR, exist_ok=True)
symbol_path = os.path.join(CKPT_DIR, "symbol_weights.pt")
torch.save({'model': ut.state_dict(), 'coords': coords}, symbol_path)
print(f"  Weights saved: {symbol_path}")

if correct_count == 156:
    print("  SUCCESS: all 156 symbols reproduced")
else:
    print(f"  FAIL: {156 - correct_count} symbols wrong")
    for i, p in enumerate(predicted):
        if p.item() != (i + 1):
            expected = cv.decode([i + 1])
            got = cv.decode([p.item()])
            print(f"    id={i+1:>3d} expected='{expected}' got='{got}'")
    CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
    os.makedirs(CKPT_DIR, exist_ok=True)
    fail_path = os.path.join(CKPT_DIR, "phase3_fail.pt")
    torch.save({'model': ut.state_dict(), 'coords': coords}, fail_path)
    print(f"  Checkpoint saved: {fail_path}")

print("\nDone.")
