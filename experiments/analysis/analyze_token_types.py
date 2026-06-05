"""Analyze BPE token types by decoded content."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizers import Tokenizer

path = 'real_data/bpe_tokenizer.json'
tok = Tokenizer.from_file(path)
vocab_size = tok.get_vocab_size()
print('Vocab size:', vocab_size)

# Special tokens
print('\nSpecial tokens:')
for name in ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<SEP>', '<MASK>']:
    tid = tok.token_to_id(name)
    print(f'  ID {tid:4d}: {name}')

# Sample BPE tokens
print('\nSample BPE tokens:')
for i in [6, 10, 32, 97, 156, 200, 300, 500, 1000, 2000, 3000, 3999, 4095]:
    if i < vocab_size:
        d = tok.decode([i])
        safe = d.encode('ascii', 'replace').decode()
        print('  ID %4d: repr=%-30s start_space=%s' % (i, safe, d.startswith(' ')))

# Test standard encoding (with spaces, not word-by-word)
print('\nStandard encode:')
for text in ['Мама мыла раму', 'Привет, мир!', 'Война и мир']:
    enc = tok.encode(text)
    parts = []
    for tid in enc.ids:
        d = tok.decode([tid])
        parts.append('%s' % d)
    print('  %-20s -> %s' % (text, '|'.join(parts)))
    for tid in enc.ids:
        d = tok.decode([tid])
        print('    ID %4d: start=%s' % (tid, d.startswith(' ')))

# Count tokens by type
n_starter = 0
n_continuer = 0
n_byte_ambig = 0
n_byte_space = 0
for i in range(6, vocab_size):
    d = tok.decode([i])
    if d.startswith(' '):
        n_starter += 1
    elif len(d) == 1:
        n_byte_ambig += 1
    else:
        n_continuer += 1

print(f'\nToken breakdown (IDs 6-{vocab_size-1}):')
print(f'  Word-starters (leading space):   {n_starter:5d}')
print(f'  Word-continuers (multi-char, no space): {n_continuer:5d}')
print(f'  Single-byte (ambiguous):         {n_byte_ambig:5d}')

# Check which single bytes are space vs non-space
space_bytes = []
for i in range(6, 156):
    d = tok.decode([i])
    if d == ' ':
        space_bytes.append(i)
print(f'\nSpace byte tokens: {space_bytes}')
