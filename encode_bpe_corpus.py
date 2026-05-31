"""
Encode full_corpus_ru.txt through BPE tokenizer → full_corpus_bpe.npy
Output: flat int64 array of BPE token IDs (0-4095, no boundary tokens)
"""
import numpy as np, os, time, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eva.symbolic.bpe_tokenizer import BPEVocab

cv = BPEVocab()
tok = cv.tokenizer

corpus_path = 'real_data/full_corpus_ru.txt'
out_path = 'real_data/full_corpus_bpe.npy'

total_chars = os.path.getsize(corpus_path)
print(f'Corpus: {total_chars/1e6:.1f} MB')

chunk_lines = 10000
all_ids = [cv.BOS_IDX]
t0 = time.time()
total_raw = 0

with open(corpus_path, 'r', encoding='utf-8') as f:
    buf = []
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        buf.append(line)
        if len(buf) >= chunk_lines:
            text = ' '.join(buf)
            total_raw += len(text)
            ids = tok.encode(text).ids
            all_ids.extend(ids)
            buf = []
            elapsed = time.time() - t0
            pct = total_raw / total_chars * 100
            print(f'  encoded {total_raw/1e6:.1f} MB ({pct:.1f}%) | {all_ids[-1] if len(all_ids)>1 else 0} IDs so far | {elapsed:.0f}s', end='\r')

    if buf:
        text = ' '.join(buf)
        ids = tok.encode(text).ids
        all_ids.extend(ids)

all_ids.append(cv.EOS_IDX)
arr = np.array(all_ids, dtype=np.int64)

elapsed = time.time() - t0
print(f'\nDone: {len(arr):,} BPE tokens ({len(arr)/1e6:.1f}M) | {elapsed:.0f}s ({elapsed/60:.1f}min)')
print(f'Range: {arr.min()}..{arr.max()} (vocab={cv.vocab_size})')

np.save(out_path, arr)
print(f'Saved to {out_path}')
