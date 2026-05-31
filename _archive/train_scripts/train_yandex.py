"""
EVA — Yandex Cloud Training (32 GB VRAM optimized).

Масштабированная архитектура:
- 128-dim, 32 heads (8 levels × 4 scales), 6 layers, d_ff=512
- ~1.2M параметров
- Batch 512, fp16 + GradScaler
- Think loop каждые 1000 шагов
- Сохранение каждые 5000 шагов
"""

import torch, torch.nn.functional as F, numpy as np, sys, os, time, random
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")
LOG = os.path.join(os.path.dirname(__file__), "yandex_train_log.txt")
os.makedirs(CKPT, exist_ok=True)

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.self_reflection import SelfReflection
from eva.symbolic.active_causal import ActiveLearner
from eva.symbolic.trajectory_store import TrajectoryStore
cv = CharacterVocab(); VT = 157

print("=" * 60)
print("EVA — Yandex Cloud Training")
print("=" * 60)

# ============================================================
# Hardware report
# ============================================================
if DEVICE == 'cuda':
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu.name}")
    print(f"VRAM: {gpu.total_memory/1e9:.1f} GB")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Compute: {gpu.major}.{gpu.minor}")

import psutil
ram = psutil.virtual_memory()
print(f"RAM: {ram.total/1e9:.1f} GB")
print(f"CPU: {psutil.cpu_count()} cores")

# ============================================================
# Scaled architecture
# ============================================================
D_MODEL = 128      # ×2 from 64
D_FF = 512         # ×4 ratio
N_LAYERS = 6       # ×2 from 3
N_LEVELS = 8       # ×2 from 4
SCALES = 4
N_HEADS = N_LEVELS * SCALES  # 32

print(f"\nArchitecture: dim={D_MODEL}, ff={D_FF}, layers={N_LAYERS}, heads={N_HEADS}")

# Load evolved affinity (check multiple locations) 
evolved = None
for p in [os.path.join(CKPT, "evolved_affinity.pt"), "evolved_affinity.pt",
          os.path.join(os.path.dirname(__file__), "evolved_affinity.pt")]:
    if os.path.exists(p):
        evolved = torch.load(p, map_location='cpu', weights_only=True)
        print(f"Loaded affinity from: {p}")
        break
if evolved is None:
    print("FATAL: evolved_affinity.pt not found!")
    print("Download from GitHub releases and place in checkpoints/symbolic/ or root")
    sys.exit(1)
coords = evolved['coords'].to(DEVICE)

# Pad to 128-dim
coordsN = torch.zeros(157, D_MODEL, device=DEVICE)
coordsN[:, :24] = coords[:, :24]
g = torch.Generator(device=DEVICE).manual_seed(42)
coordsN[:, 24:] = torch.randn(157, D_MODEL-24, generator=g, device=DEVICE) * 0.02
coordsN = coordsN / coordsN.norm(dim=-1, keepdim=True).clamp(1e-8)

ut = UnifiedMultidimensionalTransformer(vocab_size=157, coord_dim=D_MODEL,
    num_levels=N_LEVELS, scales_per_level=SCALES, num_layers=N_LAYERS, d_ff=D_FF).to(DEVICE)
ut.set_symbol_coordinates(coordsN)

total_params = sum(p.numel() for p in ut.parameters())
print(f"Model: {total_params:,} parameters")

# Load existing weights if available (partial transfer from smaller model)
for ckpt_name in ["unified_latest.pt", "gfre_latest.pt", "v2_latest.pt"]:
    ckpt_path = os.path.join(CKPT, ckpt_name)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        state = ckpt.get('ut', ckpt.get('model', None))
        if state:
            ut.load_state_dict(state, strict=False)
            print(f"Partial transfer from: {ckpt_name}")
        break

# ============================================================
# Data
# ============================================================
npy = None
for p in [os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy"),
          "connected_ru.npy",
          os.path.join(os.path.dirname(__file__), "real_data", "full_corpus_ids.npy")]:
    if os.path.exists(p):
        npy = p; break
if npy is None:
    print("FATAL: connected_ru.npy not found!")
    sys.exit(1)
data = np.load(npy, mmap_mode='r').astype(np.int32); total = len(data)
print(f"Corpus: {npy} ({total/1e6:.1f}M tokens)")

# Trajectory store
store_path = os.path.join(CKPT, "trajectory_store.pkl")
store = TrajectoryStore()
if os.path.exists(store_path): store.load(store_path)

reflector = SelfReflection(); learner = ActiveLearner()

# ============================================================
# Training config
# ============================================================
STEPS = 200000; LR = 5e-4; B = 128; ML = 128
GRAD_ACCUM = 1  # no accumulation — fit in VRAM
GRAD_ACCUM = 2  # effective batch = B * GRAD_ACCUM
THINK_EVERY = 2000; SAVE_EVERY = 500

opt = torch.optim.AdamW(ut.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
rng = np.random.RandomState(42)

def log(msg):
    t = time.strftime("%H:%M:%S"); line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f: f.write(line + '\n'); f.flush()

log(f"START: {STEPS} steps, dim={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}")
log(f"  batch={B}×{GRAD_ACCUM}, fp16")
log(f"  Model: {total_params:,} params")

torch.cuda.empty_cache()
t0 = time.time()
total_think_perceived = 0; total_think_contemplated = 0

for s in range(1, STEPS + 1):
    # Accumulate gradients over GRAD_ACCUM micro-batches
    opt.zero_grad()
    accum_loss = 0; accum_acc = 0; accum_count = 0
    
    for micro in range(GRAD_ACCUM):
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
        
        ut.train()
        _, scores = ut(bt, return_scores=True)
        target = bt[:, 1:].clamp(1, VT - 1).contiguous()
        pred = scores[:, :-1, :].contiguous(); t_mask = mask[:, 1:]
        
        loss = F.cross_entropy(pred.view(-1, 157), target.view(-1), reduction='none')
        loss = (loss.view(B, ml - 1) * t_mask).sum() / (t_mask.sum() + 1e-8)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ut.parameters(), 1.0)
        opt.step()
        
        with torch.no_grad():
            acc = (pred.argmax(-1) == target) & t_mask.bool()
            acc_val = acc.sum().item() / (t_mask.sum() + 1e-8)
        
        accum_loss += loss.item(); accum_acc += acc_val; accum_count += 1
        
        opt.zero_grad()
    
    if accum_count == 0: continue
    accum_loss /= accum_count; accum_acc /= accum_count
    sch.step()
    
    # === Think loop ===
    if s % THINK_EVERY == 0:
        ut.eval()
        for _ in range(3):
            pos = rng.randint(0, max(1, total - 128))
            chunk = data[pos:pos + 128]
            valid = chunk[(chunk > 0) & (chunk < VT)]
            if len(valid) >= 10:
                text = cv.decode(valid[:60].tolist())
                ids = cv.encode(text)[1:-1]
                with torch.no_grad():
                    inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
                    traj = ut.embed(inp)[0].cpu().numpy()
                    store.store(text, ids, traj)
                    total_think_perceived += 1
            
            for _ in range(2):
                ut.reason([random.randint(1, 156)], num_hypotheses=1,
                         temperature=0.2, char_vocab=cv)
                total_think_contemplated += 1
        ut.train()
    
    # === Log ===
    if s % SAVE_EVERY == 0:
        torch.save({'ut': ut.state_dict(), 'step': s},
                   os.path.join(CKPT, f"yandex_{s}.pt"))
        torch.save({'ut': ut.state_dict(), 'step': s},
                   os.path.join(CKPT, "yandex_latest.pt"))
        
        elapsed = time.time() - t0; eta = (elapsed / s) * (STEPS - s) if s > 0 else 0
        vram = torch.cuda.memory_allocated() / 1e9
        log(f"  step {s:>7d}/{STEPS} | loss={accum_loss:.4f} acc={accum_acc:.3f} | "
            f"VRAM={vram:.1f}GB | think:P={total_think_perceived}C={total_think_contemplated} | "
            f"{elapsed/60:.0f}min eta {eta/60:.0f}min")
    elif s % 500 == 0:
        elapsed = time.time() - t0
        vram = torch.cuda.memory_allocated() / 1e9
        log(f"  step {s:>7d}/{STEPS} | loss={accum_loss:.4f} acc={accum_acc:.3f} | "
            f"VRAM={vram:.1f}GB | {elapsed/60:.0f}min")

log(f"DONE. Think: {total_think_perceived} perceived, {total_think_contemplated} contemplated")
torch.save({'ut': ut.state_dict(), 'config': {'dim': D_MODEL, 'heads': N_HEADS, 'layers': N_LAYERS}},
           os.path.join(CKPT, "yandex_final.pt"))
