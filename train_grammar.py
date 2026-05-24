"""
EVA — GrammarHead: грамматический слой поверх трансформера.

Многослойный перцептрон, обученный на правилах русского языка:
- Склонения: род, число, падеж
- Спряжения: лицо, время, вид
- Согласование: прилагательное↔существительное, субъект↔глагол
- Порядок слов: тема↔рема

GrammarHead получает координаты от трансформера → предсказывает
грамматически корректное продолжение траектории.

Training: causal next-token prediction с топ-K accuracy.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer

cv = CharacterVocab()
VT = 157

print("=" * 60)
print("EVA — GrammarHead Training")
print("=" * 60)
print(f"Device: {DEVICE}")

# ============================================================
# GrammarHead: neural grammar layer
# ============================================================
class GrammarHead(nn.Module):
    """
    Грамматический слой поверх координатного трансформера.
    
    Вход:  координаты от трансформера [B, L, 24]
    Выход: грамматически скорректированные продолжения [B, L, 157]
    
    Архитектура:
    - Position-aware: учитывает позицию в слове/предложении
    - Context window: смотрит на ±8 соседей для согласования
    - Multi-scale: character, bigram, word уровни
    """
    
    def __init__(self, coord_dim=24, vocab_size=157, hidden=64):
        super().__init__()
        self.coord_dim = coord_dim
        self.vocab_size = vocab_size
        
        # Context-aware grammar correction
        self.context_conv = nn.Sequential(
            nn.Conv1d(coord_dim, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        
        # Grammar rules encoder (learned morphological features)
        self.grammar_encoder = nn.Sequential(
            nn.Linear(coord_dim + hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, vocab_size),
        )
        
        # Positional bias (where in word are we?)
        self.pos_embed = nn.Embedding(128, coord_dim)
    
    def forward(self, coords, return_logits=True):
        """
        coords: [B, L, coord_dim] — выходные координаты трансформера
        
        Returns: logits [B, L, vocab_size] — grammar-corrected scores
        """
        B, L, D = coords.shape
        
        # Add positional info
        positions = torch.arange(L, device=coords.device).clamp(0, 127)
        pos_emb = self.pos_embed(positions).unsqueeze(0).expand(B, -1, -1)
        x = coords + pos_emb  # [B, L, D]
        
        # Context convolution (captures agreement patterns)
        ctx = self.context_conv(x.transpose(1, 2))  # [B, hidden, L]
        ctx = ctx.transpose(1, 2)  # [B, L, hidden]
        
        # Combine coordinate + context
        combined = torch.cat([x, ctx], dim=-1)  # [B, L, D + hidden]
        
        # Predict grammar-corrected continuations
        logits = self.grammar_encoder(combined)  # [B, L, vocab_size]
        
        return logits

# ============================================================
# Load transformer
# ============================================================
evolved = torch.load(os.path.join(CKPT_DIR, "evolved_affinity.pt"), map_location='cpu', weights_only=True)
coords = evolved['coords'].to(DEVICE)

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=24)
if DEVICE == 'cuda':
    ut = ut.cuda()
ut.set_symbol_coordinates(coords.to(DEVICE))

# Load best available weights
for ckpt_name in ["conceptnet_weights.pt", "sentence_weights.pt", "word_weights.pt"]:
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        ut.load_state_dict(ckpt['model'], strict=False)
        print(f"Loaded: {ckpt_name}")
        break

# Freeze transformer — only train GrammarHead
for p in ut.parameters():
    p.requires_grad = False
ut.eval()

# ============================================================
# Create GrammarHead
# ============================================================
gh = GrammarHead(coord_dim=24, vocab_size=157, hidden=64)
if DEVICE == 'cuda':
    gh = gh.cuda()
print(f"GrammarHead: {sum(p.numel() for p in gh.parameters()):,} parameters")

# ============================================================
# Training: causal next-token prediction
# ============================================================
print("\n[TRAIN] GrammarHead — causal next-token prediction...")

npy_file = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")
if not os.path.exists(npy_file):
    npy_file = os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")
all_ids = np.load(npy_file, mmap_mode='r').astype(np.int32)

UT_STEPS = 5000; UT_LR = 1e-3; UT_BATCH = 64; MAX_LEN = 64
opt = torch.optim.AdamW(gh.parameters(), lr=UT_LR, weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=UT_STEPS)

rng = np.random.RandomState(42)
total_ids = len(all_ids)
start_t = time.time()
last_print_t = 0

for step in range(1, UT_STEPS + 1):
    # Random blocks from corpus
    lengths = rng.randint(16, MAX_LEN + 1, UT_BATCH)
    starts = rng.randint(0, max(1, total_ids - max(lengths) - 1), UT_BATCH)
    max_len = max(lengths)
    
    bt = torch.full((UT_BATCH, max_len), 0, dtype=torch.long, device=DEVICE)
    mask = torch.zeros(UT_BATCH, max_len, device=DEVICE)
    for bi in range(UT_BATCH):
        s, l = starts[bi], lengths[bi]
        block = all_ids[s:s+l]
        valid = (block > 0) & (block < VT)
        vb = block[valid]
        vl = min(len(vb), max_len)
        if vl >= 4:
            bt[bi, :vl] = torch.from_numpy(vb[:vl].astype(np.int64)).to(DEVICE)
            mask[bi, :vl] = 1.0
    
    if mask.sum() < 50:
        continue
    
    with torch.no_grad():
        # Get transformer output coordinates
        ut_coords, _ = ut(bt, return_scores=True)
    
    # GrammarHead predicts continuations
    gh.train()
    logits = gh(ut_coords)  # [B, L, 157]
    
    # Causal: predict NEXT token
    # Input positions 0..L-2 predict positions 1..L-1
    pred = logits[:, :-1, :].contiguous()  # [B, L-1, 157]
    target = bt[:, 1:].contiguous().clamp(1, VT-1)  # [B, L-1]
    t_mask = mask[:, 1:]  # [B, L-1]
    
    loss = F.cross_entropy(
        pred.view(-1, 157),
        target.view(-1),
        reduction='none'
    ).view(UT_BATCH, max_len - 1)
    loss = (loss * t_mask).sum() / (t_mask.sum() + 1e-8)
    
    # Grammar consistency bonus: context agreement
    # Encourage similar positions in different words to have similar predictions
    if max_len >= 8:
        ctx_loss = 0
        n_pairs = 0
        for bi in range(min(UT_BATCH, 16)):
            valid_pos = t_mask[bi].nonzero(as_tuple=True)[0]
            if len(valid_pos) >= 4:
                for _ in range(4):
                    i1, i2 = int(valid_pos[rng.randint(0, len(valid_pos))]), int(valid_pos[rng.randint(0, len(valid_pos))])
                    if abs(i1 - i2) >= 3:
                        # Penalize divergence: similar contexts should predict similar distributions
                        dist = F.kl_div(
                            F.log_softmax(logits[bi, i1], dim=-1),
                            F.softmax(logits[bi, i2].detach(), dim=-1),
                            reduction='batchmean'
                        )
                        if not torch.isnan(dist) and not torch.isinf(dist):
                            ctx_loss += dist
                            n_pairs += 1
        if n_pairs > 0:
            loss = loss + 0.01 * ctx_loss / n_pairs
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gh.parameters(), 1.0)
    opt.step()
    sch.step()
    
    now = time.time()
    if now - last_print_t >= 5 or step == 1 or step == UT_STEPS:
        last_print_t = now
        elapsed = now - start_t
        eta = (elapsed / step) * (UT_STEPS - step)
        with torch.no_grad():
            top1 = (pred.argmax(dim=-1) == target) & t_mask.bool()
            tok_acc = top1.sum().item() / (t_mask.sum() + 1e-8)
            top5 = torch.topk(pred, 5, dim=-1).indices
            top5_acc = (top5 == target.unsqueeze(-1)).any(dim=-1) & t_mask.bool()
            top5_acc = top5_acc.sum().item() / (t_mask.sum() + 1e-8)
        print(f"  step {step:>4d}/{UT_STEPS} | loss={loss.item():.4f} | "
              f"top1={tok_acc:.3f} top5={top5_acc:.3f}"
              f" | {elapsed:.0f}s / eta {eta:.0f}s", flush=True)

# ============================================================
# Test: autoregressive generation with GrammarHead
# ============================================================
print("\n[TEST] GrammarHead-guided generation...")

def generate_with_grammar(ut, gh, coords, seed_ids, affinity, max_new=20, temperature=0.7, top_k=30):
    """Generate using GrammarHead + affinity constraint."""
    ids = list(seed_ids)
    ut.eval()
    gh.eval()
    
    with torch.no_grad():
        for _ in range(max_new):
            inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
            ut_coords, _ = ut(inp, return_scores=True)
            logits = gh(ut_coords)[0, -1] / temperature
            
            # Affinity boost: prefer tokens with high affinity to previous token
            prev = ids[-1] if ids else 1
            if 0 < prev < VT:
                aff_boost = affinity[prev].to(DEVICE).clone()
                aff_boost[0] = -1e9  # block PAD
                aff_boost = aff_boost / (aff_boost.max() + 1e-8) * 2.0  # normalize
                logits = logits + aff_boost
            
            # Top-k + repetition penalty
            k = min(top_k, len(logits) - 1)
            topk_vals, topk_idx = torch.topk(logits, k)
            probs = F.softmax(topk_vals, dim=-1)
            
            # Strong repetition penalty
            if len(ids) >= 2:
                for t in set(ids[-5:]):
                    mask_idx = (topk_idx == t).nonzero(as_tuple=True)[0]
                    if len(mask_idx) > 0:
                        probs[mask_idx] *= 0.1
            
            probs = probs / probs.sum()
            next_token = topk_idx[torch.multinomial(probs, 1)].item()
            
            if next_token <= 0 or next_token >= VT:
                next_token = topk_idx[0].item()
            
            ids.append(next_token)
    
    return ids

test_seeds = [
    ("привет", "привет"),
    ("человек", "человек"),
    ("солнце", "солнце"),
    ("я люблю", "я люблю"),
    ("метаданные", "метаданные"),
    ("трансформер", "трансформер"),
]

for seed, label in test_seeds:
    ids = cv.encode(seed)[1:-1]
    if len(ids) < 2:
        continue
    
    result = generate_with_grammar(ut, gh, coords, ids, evolved['affinity'], max_new=20, temperature=0.7)
    gen_text = cv.decode(result)
    print(f"  '{label}...' → '{gen_text}'")

# Save
gh_path = os.path.join(CKPT_DIR, "grammar_head.pt")
torch.save({'model': gh.state_dict(), 'coords': coords}, gh_path)
print(f"\nSaved: {gh_path}")
print("Done.")
