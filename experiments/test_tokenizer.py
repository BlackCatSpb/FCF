"""Test ConceptTokenizer."""
import sys; sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\FCF')
from eva.symbolic.concept_tokenizer import ConceptTokenizer

out = open(r'C:\Users\black\OneDrive\Desktop\FCF\experiments\tokenizer_test.txt', 'w', encoding='utf-8')

tok = ConceptTokenizer()
try:
    tok.initialize()
except Exception as e:
    out.write(f"Error initializing: {e}\n")
    import traceback
    traceback.print_exc(file=out)
    out.close()
    raise

out.write(f"Total vocab: {len(tok)}\n")
out.write(f"BPE vocab: {tok.bpe_vocab_size}\n")
out.write(f"Concepts: {tok.skeleton.n_concepts}\n\n")

# Test: roundtrip
test_text = "Война и мир. Шедевр мировой литературы."
ids = tok.encode(test_text)
decoded = tok.decode(ids)
out.write(f"Original: {test_text}\n")
out.write(f"Encoded:  {ids[:30]}... (len={len(ids)})\n")
out.write(f"Decoded:  {decoded}\n")
out.write(f"Roundtrip OK: {test_text == decoded}\n\n")

# Show full token breakdown
out.write("Token breakdown:\n")
for tid in ids:
    if tid == tok.BOS:
        out.write("  [BOS]\n")
    elif tid == tok.EOS:
        out.write("  [EOS]\n")
    elif tid == tok.WORD_OPEN:
        out.write("  [WO]\n")
    elif tid == tok.WORD_CLOSE:
        out.write("  [WC]\n")
    elif tid == tok.SENT_OPEN:
        out.write("  [SO]\n")
    elif tid == tok.SENT_CLOSE:
        out.write("  [SC]\n")
    elif tid >= tok.N_SPECIAL:
        text = tok.bpe.decode([tid - tok.N_SPECIAL])
        out.write(f"  {tid:5d} '{text}'\n")
    else:
        out.write(f"  {tid:5d} (special)\n")

# Show metadata
meta = tok.metadata_from_ids(ids)
out.write("\nWord-level metadata:\n")
seen_words = set()
for m in meta:
    if m['flags'] & 1:  # word_start
        wn = m['word_num']
        if wn not in seen_words:
            seen_words.add(wn)
            out.write(f"  word[{wn}] concept={m.get('concept_id')} "
                      f"anchor={m.get('anchor_word')!r} is_anchor={m.get('is_anchor')}\n")

# Test with corpus words
out.write("\n\nCorpus word -> concept mapping:\n")
corpus_path = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\full_corpus_ru.txt'
words_found = 0
words_total = 0
corpus_words = set()
with open(corpus_path, 'r', encoding='utf-8') as f:
    for line in f:
        for w in line.strip().split():
            wc = w.strip('.,!?;:()[]«»—–-…\"\'').lower()
            if wc:
                corpus_words.add(wc)
                words_total += 1
                if len(corpus_words) >= 5000:
                    break
        if len(corpus_words) >= 5000:
            break

for w in corpus_words:
    cid, is_anchor, anchor = tok.word_info(w)
    if cid is not None:
        words_found += 1

out.write(f"Corpus words sampled: {len(corpus_words)}\n")
out.write(f"Words with concepts: {words_found} ({100*words_found/len(corpus_words):.1f}%)\n")

out.close()
