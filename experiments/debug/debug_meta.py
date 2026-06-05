"""
Debug HierarchicalVocab metadata and head scores.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from eva.symbolic.bpe_tokenizer import HierarchicalVocab
from eva.symbolic.heads import HeadsEnsemble

hv = HierarchicalVocab()
print(f"vocab_size={hv.vocab_size}")
print(f"type_names: SPECIAL={hv.type_name(0)}, BYTE={hv.type_name(6)}, WORD_STARTER={hv.type_name(196)}, WORD_CONT={hv.type_name(197)}")
print()

# ─── 1. Check metadata for a sample sentence ───
test = "Привет, мир!"
ids = hv.encode(test)
full = [2] + ids + [3]
print(f"Test sentence: {repr(test)}")
print(f"Full IDs (incl BOS/EOS): {full}")
meta = hv.metadata_from_ids(full)
for m in meta:
    tid = m["token_id"]
    t = hv.decode([tid])
    print(f"  id={tid:4d} type={m['token_type']} flags={m['flags']:2d} "
          f"piw={m['pos_in_word']:2d} wl={m['word_len']:2d} wn={m['word_num']:2d} "
          f"sent_pos={m['pos_in_sent']:3d} text={repr(t)}")
print()

# ─── 2. Check what morph distributions look like ───
meta_path = r"real_data/v6/heads_meta.pkl"
heads = HeadsEnsemble(meta_path, "real_data/v6")
print(f"Morph keys: {sorted(heads.morph_logprob.keys())}")
wl_sample = 3
if wl_sample in heads.morph_logprob:
    piw_keys = sorted(heads.morph_logprob[wl_sample].keys())
    print(f"  morph[{wl_sample}] piw: {piw_keys}")
    for piw in piw_keys:
        arr = heads.morph_logprob[wl_sample][piw]
        top5 = np.argsort(arr)[-5:][::-1]
        texts = [repr(hv.decode([int(t)])) for t in top5]
        print(f"  morph[3][{piw}] top-5: {texts}")
print()

# ─── 3. Check syntax ───
print(f"Syntax keys: {sorted(heads.syntax_logprob.keys())}")
wn = 2
if wn in heads.syntax_logprob:
    arr = heads.syntax_logprob[wn]
    top8 = np.argsort(arr)[-8:][::-1]
    texts = [repr(hv.decode([int(t)])) for t in top8]
    print(f"  syntax[{wn}] top-8: {texts}")
print()

# ─── 4. Check transitions ───
print("Transition CSR stats:")
print(f"  shape={heads.log_prob_csr.shape}, nnz={heads.log_prob_csr.nnz}")
# From BOS
bos_row = heads.log_prob_csr[2].toarray()[0]
bos_top = np.argsort(bos_row)[-8:][::-1]
bos_texts = [repr(hv.decode([int(t)])) for t in bos_top]
print(f"  From BOS(2) top-8: {bos_texts}")
# From SPACE (0x20 = 32)
sp_row = heads.log_prob_csr[32].toarray()[0]
sp_top = np.argsort(sp_row)[-8:][::-1]
sp_texts = [repr(hv.decode([int(t)])) for t in sp_top]
print(f"  From SPACE(32) top-8: {sp_texts}")
# From most common token after BOS
print(f"  Bos top1 = {bos_top[0]} ({bos_texts[0]})")
if True:
    nxt = bos_top[0]
    nxt_row = heads.log_prob_csr[nxt].toarray()[0]
    nxt_top = np.argsort(nxt_row)[-8:][::-1]
    nxt_texts = [repr(hv.decode([int(t)])) for t in nxt_top]
    print(f"  From {nxt}({bos_texts[0]}) top-8: {nxt_texts}")
