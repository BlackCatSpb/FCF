"""Generate sample text from the latest EVA model."""
import sys, os, io
sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
import torch
import numpy as np

from eva.symbolic.heads import HeadsEnsemble
from eva.symbolic.weight_transformer import WeightTransformer
from eva.symbolic.generation_loop import GenerationLoop
from eva.symbolic.bpe_tokenizer import BPEVocab

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
META = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\heads_meta.pkl'
CSR = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\hierarchical'
MODEL = r'C:\Users\black\OneDrive\Desktop\FCF\models\weight_transformer_best.pt'
OUT = r'C:\Users\black\OneDrive\Desktop\FCF\generation_samples.txt'

heads = HeadsEnsemble(META, CSR)
transformer = WeightTransformer()
if os.path.exists(MODEL):
    transformer.load_state_dict(torch.load(MODEL, weights_only=True, map_location=DEVICE))
    print(f'Model loaded ({sum(p.numel() for p in transformer.parameters()):,} params) on {DEVICE}')
else:
    transformer = None
transformer.to(DEVICE)
transformer.eval()

gen = GenerationLoop(heads, transformer=transformer, max_tokens=80, device=DEVICE)
vocab = BPEVocab()

lines = []
for temp_label, temp in [('argmax', 0.0), ('low', 0.3), ('medium', 0.8), ('high', 1.5)]:
    lines.append(f'\n--- Temperature {temp_label} ({temp}) ---')
    for s in range(3):
        result = gen.generate(temperature=temp, seed=np.random.randint(0, 99999), return_coords=False)
        text = vocab.decode(result).replace('\ufffd', '?')
        lines.append(f'  [{s}] {text}')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f'Generated samples written to {OUT}')
