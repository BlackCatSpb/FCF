"""
Extract Russian concepts from ConceptNet → training-ready int32 batches.
Each concept = one line. No cut words, no partial sentences.
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, r"C:\Users\black\OneDrive\Desktop\EVA-Ai")

print("=" * 60)
print("EVA — ConceptNet Russian Dataset Builder")
print("=" * 60)

# Connect
from conceptnet_lite import connect, Label, Language
connect(r"C:\Users\black\OneDrive\Desktop\EVA-Ai\conceptnet.db")
lang_ru = Language.get(name='ru')

# Get ALL Russian labels
print("\n[1] Loading Russian labels...")
ru_labels = list(Label.select().where(Label.language == lang_ru))
print(f"  Labels: {len(ru_labels):,}")

# Extract unique Russian words
ru_words = sorted(set(label.text.strip() for label in ru_labels))
print(f"  Unique words: {len(ru_words):,}")

# Filter and encode
from eva.symbolic.char_vocab import CharacterVocab
cv = CharacterVocab()

print("\n[2] Encoding concepts...")
encoded_batches = []
skipped = 0
for word in ru_words:
    # Encode the concept word
    ids = cv.encode(word)
    if len(ids) >= 3:  # BOS + at least 1 char + EOS
        encoded_batches.append(ids)
    else:
        skipped += 1

print(f"  Encoded: {len(encoded_batches):,}")
print(f"  Skipped (too short): {skipped:,}")

# Get edges for richer content
print("\n[3] Getting edges for semantic relationships...")
from conceptnet_lite import Concept, edges_for
import random
random.seed(42)

# Sample 50K words for edges (edges provide sentence-like context)
sample = random.sample(ru_words, min(50000, len(ru_words)))

edge_lines = []
for i, word in enumerate(sample):
    try:
        label = Label.get_or_create(text=word, language=lang_ru)[0]
        concept = Concept.get(label=label)
        edges = list(edges_for([concept]))
        
        for edge in edges:
            rel = edge.relation.name
            weight = edge.etc.get('weight', 1.0)
            
            start_uri = edge.start.uri
            end_uri = edge.end.uri
            
            # Extract labels
            start_label = start_uri.split('/')[-1] if '/' in start_uri else ''
            end_label = end_uri.split('/')[-1] if '/' in end_uri else ''
            
            # Build a sentence-like structure: "start REL end"
            if start_label and end_label and rel in ('RelatedTo', 'IsA', 'PartOf', 'Synonym', 'Antonym', 'DerivedFrom', 'FormOf', 'HasA', 'MadeOf', 'UsedFor', 'CapableOf', 'AtLocation'):
                sentence = f"{start_label} {rel} {end_label}"
                ids = cv.encode(sentence)
                if len(ids) >= 5:
                    edge_lines.append(ids)
    except:
        pass
    
    if (i+1) % 5000 == 0:
        print(f"\r  {i+1}/{len(sample)} words, {len(edge_lines):,} edge-sentences", end='', flush=True)

print(f"\n  Edge sentences: {len(edge_lines):,}")

# Combine: concepts + edge-sentences
all_sequences = encoded_batches + edge_lines
print(f"\n[4] Total sequences: {len(all_sequences):,}")

# Shuffle
random.shuffle(all_sequences)

# Flatten with EOS separators
all_ids = []
for seq in all_sequences:
    all_ids.extend(seq)
    all_ids.append(cv.EOS_IDX)  # sentence separator

total_tokens = len(all_ids)
print(f"  Total tokens: {total_tokens:,}")

# Save
out_path = os.path.join(os.path.dirname(__file__), "real_data", "conceptnet_ru.npy")
arr = np.array(all_ids, dtype=np.int32)
np.save(out_path, arr)
print(f"\nSaved: {out_path} ({total_tokens/1e6:.1f}M tokens, {os.path.getsize(out_path)/1e6:.1f} MB)")
print("Done.")
