"""Quick peek at heads_meta structure"""
import pickle, numpy as np

with open(r'C:\Users\black\OneDrive\Desktop\FCF\real_data\v5\heads_meta.pkl', 'rb') as f:
    m = pickle.load(f)
print('Keys:', list(m.keys()))
for k, v in m.items():
    if isinstance(v, dict):
        print(f'  {k}: dict with {len(v)} keys')
        if len(v) > 0:
            sk = list(v.keys())[0]
            sv = v[sk]
            if isinstance(sv, np.ndarray):
                print(f'    sample key {sk!r}: np.array shape={sv.shape} dtype={sv.dtype}')
            elif isinstance(sv, dict):
                print(f'    sample key {sk!r}: dict with {len(sv)} keys')
            else:
                print(f'    sample key {sk!r}: {type(sv).__name__} = {str(sv)[:60]}')
    elif isinstance(v, np.ndarray):
        print(f'  {k}: np.array shape={v.shape} dtype={v.dtype}')
    elif isinstance(v, (int, float)):
        print(f'  {k}: {v}')
    elif isinstance(v, str):
        print(f'  {k}: {v[:60]}')
    elif v is None:
        print(f'  {k}: None')
    elif isinstance(v, list):
        print(f'  {k}: list len={len(v)}, sample={str(v[:2])[:60]}')
    else:
        print(f'  {k}: {type(v).__name__}')

# Check V=4101 stats
print(f'\nV = {m.get("V", "?")}')
stats = m.get('stats', {})
if stats:
    print(f'Stats: {dict(stats)}')

# Word lengths in morph
morph = m.get('morph_logprob', {})
print(f'\nMorph word lengths: {sorted(morph.keys())}')
print(f'  Total (wl,pos) pairs: {sum(len(v) for v in morph.values())}')
# Show a sample morph logprob array
sample_wl = sorted(morph.keys())[0]
sample_pos = sorted(morph[sample_wl].keys())[0]
arr = morph[sample_wl][sample_pos]
print(f'  Sample: wl={sample_wl}, pos={sample_pos}')
print(f'    shape={arr.shape}, dtype={arr.dtype}')
print(f'    min={arr.min():.3f}, max={arr.max():.3f}, mean={arr.mean():.3f}')
# Non-zero entries
nz = np.count_nonzero(arr > -20)
print(f'    nonzero (prob > -20): {nz}/{len(arr)}')

# Syntax
syntax = m.get('syntax_logprob', {})
print(f'\nSyntax word positions: {sorted(syntax.keys())}')
for k in sorted(syntax.keys())[:5]:
    print(f'  wn={k}: shape={syntax[k].shape}')
