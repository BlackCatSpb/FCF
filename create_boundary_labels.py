"""
Create boundary labels for BPE-encoded corpus.
Outputs two files:
  - full_corpus_bpe_boundary.npy: BPE tokens with WORD_OPEN/CLOSE markers
  - full_corpus_bpe_labels.npy: boundary labels (0=word_start, 1=inside, 2=word_end, -100=ignore)
For tokens that are single BPE token per word: they get label 2 (both start and end).
"""
import numpy as np, os, sys, time, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eva.symbolic.bpe_tokenizer import BPEVocab

cv = BPEVocab()
tok = cv.tokenizer

corpus_path = 'real_data/full_corpus_ru.txt'
out_ids_path = 'real_data/full_corpus_bpe_boundary.npy'
out_labels_path = 'real_data/full_corpus_bpe_labels.npy'

total_chars = os.path.getsize(corpus_path)
print(f'Corpus: {total_chars/1e6:.1f} MB')

PUNCT = set('.,;:!?()[]{}«»—–-…\"\'')
WO = cv.WORD_OPEN_IDX
WC = cv.WORD_CLOSE_IDX

all_ids = []
all_labels = []
t0 = time.time()
total_raw = 0

with open(corpus_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue

        total_raw += len(line)

        # Split into sentences
        sentences = re.split(r'(?<=[.!?…])\s+(?=[А-ЯЁA-Z0-9])', line)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # Split into words
            words = sent.split()
            for word in words:
                # Split word into alpha part and trailing punctuation
                alpha_chunks = re.findall(r'[А-ЯЁа-яёA-Za-z0-9]+|[.,;:!?()\[\]{}«»—–-…\"\']+', word)
                if not alpha_chunks:
                    continue

                for chunk in alpha_chunks:
                    if chunk[0] in PUNCT:
                        # Punctuation: single token
                        ids = tok.encode(chunk).ids
                        all_ids.extend(ids)
                        for _ in ids:
                            all_labels.append(2)
                    else:
                        # Word: WO + BPE tokens + WC
                        all_ids.append(WO)
                        all_labels.append(0)
                        word_ids = tok.encode(chunk).ids
                        all_ids.extend(word_ids)
                        for j, tid in enumerate(word_ids):
                            if len(word_ids) == 1:
                                all_labels.append(2)
                            elif j == 0:
                                all_labels.append(0)
                            elif j == len(word_ids) - 1:
                                all_labels.append(2)
                            else:
                                all_labels.append(1)
                        all_ids.append(WC)
                        all_labels.append(2)

        if i % 10000 == 0:
            elapsed = time.time() - t0
            pct = total_raw / total_chars * 100
            print(f'  line {i} — {total_raw/1e6:.1f} MB ({pct:.1f}%) | {len(all_ids):,} tokens | {elapsed:.0f}s', end='\r')

arr_ids = np.array(all_ids, dtype=np.int64)
arr_labels = np.array(all_labels, dtype=np.int8)

elapsed = time.time() - t0
print(f'\nDone: {len(arr_ids):,} tokens | {elapsed:.0f}s ({elapsed/60:.1f}min)')
print(f'IDs: min={arr_ids.min()}, max={arr_ids.max()}')
print(f'Labels: 0={int((arr_labels==0).sum())} word_start, '
      f'1={int((arr_labels==1).sum())} inside, '
      f'2={int((arr_labels==2).sum())} word_end')

np.save(out_ids_path, arr_ids)
np.save(out_labels_path, arr_labels)
print(f'Saved {out_ids_path} ({arr_ids.nbytes/1e6:.1f} MB)')
print(f'Saved {out_labels_path} ({arr_labels.nbytes/1e6:.1f} MB)')
