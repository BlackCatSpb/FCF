"""
EVA — Think loop for War & Peace model. 
Reads wp_latest.pt, does self-reflection every 60s.
"""
import torch, numpy as np, sys, os, time, random
sys.path.insert(0, os.path.dirname(__file__))
DEVICE = 'cuda'; CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "symbolic")

from eva.symbolic.char_vocab import CharacterVocab
from eva.symbolic.unified_transformer import UnifiedMultidimensionalTransformer
from eva.symbolic.trajectory_store import TrajectoryStore
cv = CharacterVocab(); VT = cv.vocab_size

# Fresh coords like training
c128 = torch.zeros(VT, 128, device=DEVICE)
g = torch.Generator(device=DEVICE).manual_seed(42)
c128[:, :] = torch.randn(VT, 128, generator=g, device=DEVICE) * 0.02
c128 = c128 / c128.norm(dim=-1, keepdim=True).clamp(1e-8)

ut = UnifiedMultidimensionalTransformer(vocab_size=VT, coord_dim=128, max_levels=8,
    total_heads=32, num_layers=6, d_ff=128).to(DEVICE)
ut.set_symbol_coordinates(c128)

print("Think Loop — lightweight mode. Reading wp_latest.pt periodically...")
print("Waiting for training to produce checkpoint...")
while not os.path.exists(os.path.join(CKPT, "wp_latest.pt")):
    time.sleep(5)

# Load model briefly for inference, then unload
last_step = 0
iterations = 0

while True:
    try:
        time.sleep(6)
        ckpt_path = os.path.join(CKPT, "wp_latest.pt")
        if not os.path.exists(ckpt_path):
            continue
        
        # Check if checkpoint updated
        mtime = os.path.getmtime(ckpt_path)
        current_step = torch.load(ckpt_path, map_location='cpu', weights_only=True).get('step', 0)
        
        if current_step == last_step and iterations > 0:
            iterations += 1
            if iterations % 10 == 0:
                print(f"  think: {iterations} cycles, step={current_step}, store={store.total_stored}", flush=True)
            continue
        
        last_step = current_step
        
        # Load model, do inference, unload
        ut.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True)['ut'], strict=False)
        ut.eval()
        
        with torch.no_grad():
            # Perception: sample random text, generate continuation
            pos = rng.randint(0, max(1, total - 32))
            ids = [int(x) for x in data[pos:pos+32] if 0 < x < VT]
            if len(ids) >= 8:
                inp = torch.tensor([ids[:16]], dtype=torch.long, device=DEVICE)
                for _ in range(6):
                    _, sc = ut(inp, return_scores=True)
                    _, idx = torch.topk(sc[0, -1], 10)
                    p = torch.softmax(sc[0, -1][idx], dim=-1)
                    nt = idx[torch.multinomial(p, 1)].item()
                    inp = torch.cat([inp, torch.tensor([[nt]], device=DEVICE)], dim=1)
                gen_text = cv.decode(inp[0].tolist())
                if iterations % 10 == 0:
                    print(f"  think: step={current_step} store={store.total_stored} gen='{gen_text[:30]}...'", flush=True)
        
        iterations += 1
        
    except KeyboardInterrupt:
        break
