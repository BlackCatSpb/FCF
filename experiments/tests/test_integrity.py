"""Verify all components for Phase 2 training."""
import numpy as np, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eva.symbolic.bpe_tokenizer import BPEVocab

ok = True

# 1. BPE tokenizer
try:
    cv = BPEVocab()
    print(f'1. BPE tokenizer: vocab={cv.vocab_size}, WO={cv.WORD_OPEN_IDX}, WC={cv.WORD_CLOSE_IDX}')
    test = cv.encode('тест')
    print(f'   encode("тест")={test}, decode={cv.decode(test)}')
except Exception as e:
    print(f'1. FAILED: {e}'); ok = False

# 2. Boundary corpus
ids = np.load('real_data/full_corpus_bpe_boundary.npy')
lbl = np.load('real_data/full_corpus_bpe_labels.npy')
print(f'2. Corpus: ids={ids.shape} {ids.dtype} [{ids.min()}..{ids.max()}]')
print(f'   Labels: {lbl.shape} {lbl.dtype} unique={np.unique(lbl)}')
if ids.shape != lbl.shape: print('   MISMATCH!'); ok = False
# Check WO/WC present
wo = (ids == cv.WORD_OPEN_IDX).sum()
wc = (ids == cv.WORD_CLOSE_IDX).sum()
print(f'   WO={wo}, WC={wc} ({wo/len(ids)*100:.1f}%)')
# Check label alignment
sample_start = len(ids)//2
for p in range(sample_start, min(sample_start+20, len(ids))):
    lbl_char = {0:'S',1:'I',2:'E'}.get(lbl[p], '?')
    ids_char = 'WO' if ids[p]==cv.WORD_OPEN_IDX else ('WC' if ids[p]==cv.WORD_CLOSE_IDX else str(ids[p]))
    print(f'   pos {p}: ids={ids_char:>4} label={lbl_char}')
print(f'   ...')
ok_here = True
for p in range(sample_start, min(sample_start+20, len(ids))):
    if ids[p] == cv.WORD_OPEN_IDX and lbl[p] != 0: ok_here = False
    if ids[p] == cv.WORD_CLOSE_IDX and lbl[p] != 2: ok_here = False
print(f'   Alignment check: {"PASS" if ok_here else "FAIL"}')

# 3. Model forward with boundary IDs
from eva.symbolic.phase1_model import UnifiedMultidimensionalTransformerV2
m = UnifiedMultidimensionalTransformerV2(vocab_size=4101)
print(f'3. Model: {sum(p.numel() for p in m.parameters()):,} params')

x = torch.tensor(ids[:16].reshape(1, 16), dtype=torch.long)
h, s, ww, ho = m.forward(x, return_scores=True, return_heads=True,
                          capture_attn=True, update_attractors=True)
print(f'   Forward: h={h.shape}, scores={s.shape}')
print(f'   Heads: {sorted(ho.keys())}')
print(f'   Attractors: {ho["attractor_n_attractors"]}')

# 4. Check GPU
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.memory_allocated(0)/1e6
    total = torch.cuda.get_device_properties(0).total_memory/1e6
    print(f'4. GPU: {gpu}, {mem:.0f}/{total:.0f} MB')
else:
    print('4. GPU: NONE!')
    ok = False

print(f'\nOverall: {"ALL OK" if ok else "ISSUES FOUND"}')
