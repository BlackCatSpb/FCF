"""Quick generation test - modern topics."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.char_vocab import CharacterVocab

cv = CharacterVocab()
V = cv.vocab_size

c128 = torch.zeros(V, 128, device="cuda")
g = torch.Generator(device="cuda").manual_seed(42)
c128[:] = torch.randn(V, 128, generator=g, device="cuda") * 0.02
c128 = c128 / c128.norm(dim=-1, keepdim=True).clamp(1e-8)

ut = UnifiedMultidimensionalTransformer(vocab_size=V, coord_dim=128, max_levels=8,
    total_heads=32, num_layers=6, d_ff=128).to("cuda")
ut.set_symbol_coordinates(c128)

ckpt = torch.load("checkpoints/symbolic/full_best.pt", map_location="cuda", weights_only=False)
ut.load_state_dict(ckpt["ut"], strict=False)
print(f"Loaded step {ckpt.get('step','?')}", flush=True)

seeds = ["компьютер", "космос", "робот", "генетика", "искусственный", "Пьер"]
for seed in seeds:
    t0 = time.time()
    ids = cv.encode(seed)
    text = ut.enhanced_generate(ids, cv, max_new=30, temperature=0.8)
    dt = time.time() - t0
    print(f"  [{dt:.1f}s] '{seed}': {text[:100]}", flush=True)
