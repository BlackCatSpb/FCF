"""Debug ConceptNet form_of hub effect."""
import sys, re
from collections import Counter

CN_PATH = r'C:\Users\black\OneDrive\Desktop\FCF\real_data\conceptnet\conceptnet_ru.txt'

out = open(r'C:\Users\black\OneDrive\Desktop\FCF\experiments\cn_hub.txt', 'w', encoding='utf-8')

# Parse form_of relations
lemma_counter = Counter()
form_pairs = []
with open(CN_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        m = re.match(r'^(.+?)\s*[\u2014\u2013\-]\s+(.+?)\.\s*$', line)
        if not m: continue
        s = m.group(1).strip().lower()
        rest = m.group(2).strip()
        if rest.startswith('форма слова '):
            e = rest[12:].strip().lower()
            form_pairs.append((s, e))
            lemma_counter[e] += 1

out.write(f'Total form_of pairs: {len(form_pairs)}\n')
out.write(f'Unique lemmas: {len(lemma_counter)}\n')
out.write(f'Unique words with form_of: {len(set(s for s,_ in form_pairs))}\n\n')

# Top lemmas
out.write('Top lemmas by number of forms:\n')
for lemma, count in lemma_counter.most_common(30):
    out.write(f'  {lemma:30s} {count:5d} forms\n')

# Check if any word maps to a hub lemma that also maps elsewhere
lemma_map = dict(form_pairs)
hub_words = set()
for s, e in form_pairs:
    if e in lemma_map:
        hub_words.add(s)
        if len(hub_words) <= 10:
            out.write(f'  CHAIN: {s:20s} -> {e:20s} -> {lemma_map[e]}\n')

out.write(f'\nWords whose lemma also has a form_of: {len(hub_words)}\n')

# Biggest component: count how many words resolve to the same ultimate lemma
def resolve(word, seen=None):
    if seen is None:
        seen = set()
    if word in seen:
        return word
    parent = lemma_map.get(word)
    if parent is None or parent == word:
        return word
    seen.add(word)
    return resolve(parent, seen)

out.write('\nUltimate lemma distribution (top 10):\n')
ultimate = Counter()
for s in set(s for s,_ in form_pairs):
    ult = resolve(s)
    ultimate[ult] += 1
for lemma, count in ultimate.most_common(10):
    out.write(f'  {lemma:30s} {count:5d} words\n')

# Also add lemmas themselves
for e in lemma_counter:
    ult = resolve(e)
    ultimate[ult] += 1

out.write('\nAfter adding self-referencing lemmas:\n')
for lemma, count in ultimate.most_common(10):
    out.write(f'  {lemma:30s} {count:5d} words\n')

out.close()
