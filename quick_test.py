import numpy as np, os, sys, json
sys.path.insert(0, '.')
from eva.symbolic import CharacterVocab, PotentialField
import torch

print("Loading potential field...")
try:
    pf = PotentialField(156, 256)
    state = torch.load('checkpoints/symbolic/final/potential_field.pt', map_location='cpu', weights_only=True)
    pf.load_state_dict(state)
    print(f"Loaded. Avg: {pf.affinity.mean():.4f}, Max: {pf.affinity.max():.4f}")
    
    vocab = CharacterVocab()
    aff = pf.affinity.cpu().numpy()
    
    print("\nTOP 15 DIGRAMS:")
    for sym_i in range(156):
        for sym_j in range(156):
            if sym_i != sym_j and aff[sym_i,sym_j] > 0.6:
                ci = vocab.idx_to_char(sym_i)
                cj = vocab.idx_to_char(sym_j)
                print(f"  {ci}{cj}: {aff[sym_i,sym_j]:.4f}")
except Exception as e:
    print(f"Error: {e}")
    
print("Done")
