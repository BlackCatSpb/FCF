"""Build large connected Russian text dataset from Wikipedia."""
import os, re, sys, time
sys.path.insert(0, os.path.dirname(__file__))

SRC = os.path.join(os.path.dirname(__file__), "real_data", "wiki_ru_large.txt")
DST = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.txt")
NPY = os.path.join(os.path.dirname(__file__), "real_data", "connected_ru.npy")

MIN_SENTENCE_LENGTH = 15
MIN_PARAGRAPH_SENTENCES = 2

print(f"Source: {SRC} ({os.path.getsize(SRC)/1024/1024:.0f} MB)")

total_lines = 0
kept_paragraphs = 0
kept_sentences = 0
total_chars = 0

with open(SRC, 'r', encoding='utf-8') as src, open(DST, 'w', encoding='utf-8') as dst:
    current_paragraph = []
    
    for line in src:
        line = line.strip()
        if not line:
            continue
        total_lines += 1
        
        # Keep only Cyrillic + basic punctuation (KEEP periods, commas, etc.)
        cleaned = re.sub(r'[^а-яА-ЯёЁ\s.,;:!?\-—«»()""]', '', line)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        if len(cleaned) < MIN_SENTENCE_LENGTH:
            continue
        
        # Must be mostly Cyrillic
        cyr = sum(1 for c in cleaned if 0x0400 <= ord(c) <= 0x04FF or c in 'ёЁ')
        alpha = sum(1 for c in cleaned if c.isalpha())
        if alpha == 0 or cyr / alpha < 0.85:
            continue
        
        current_paragraph.append(cleaned)
        
        # Paragraph boundary: empty line or sentence ending
        if cleaned.endswith(('.', '!', '?')):
            if len(current_paragraph) >= MIN_PARAGRAPH_SENTENCES:
                paragraph_text = ' '.join(current_paragraph)
                dst.write(paragraph_text + '\n')
                kept_sentences += len(current_paragraph)
                kept_paragraphs += 1
                total_chars += len(paragraph_text)
            current_paragraph = []
        
        if kept_paragraphs % 50000 == 0 and kept_paragraphs > 0:
            print(f"  {kept_paragraphs} paragraphs, {total_chars/1024/1024:.0f} MB")
        
        if total_chars > 500_000_000:  # 500 MB limit
            break

size_mb = os.path.getsize(DST) / 1024 / 1024
print(f"\nConnected text: {kept_paragraphs:,} paragraphs, {kept_sentences:,} sentences, {size_mb:.0f} MB")

# Pre-tokenize
from eva.symbolic import CharacterVocab
import numpy as np

vocab = CharacterVocab()
all_ids = []
count = 0

print(f"\nPre-tokenizing...")
with open(DST, 'r', encoding='utf-8') as f:
    for line in f:
        ids = vocab.encode(line.strip())
        ids.append(0)
        all_ids.extend(ids)
        count += 1
        if count % 100000 == 0:
            print(f"  {count} lines, {len(all_ids)/1e6:.1f}M tokens")

arr = np.array(all_ids, dtype=np.int32)
np.save(NPY, arr)
npy_mb = os.path.getsize(NPY) / 1024 / 1024
print(f"Saved: {npy_mb:.0f} MB npy, {len(all_ids)/1e6:.1f}M tokens from {count} lines")
