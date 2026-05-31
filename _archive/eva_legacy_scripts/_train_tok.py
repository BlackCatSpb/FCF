import sys, os
sys.path.insert(0, '.')
from eva.tokenizer_utils import train_tokenizer_on_wikipedia

print('Training BPE tokenizer on Wikipedia (100K articles, 50K words)...')
t = train_tokenizer_on_wikipedia('tokenizer_wiki.json', vocab_size=50257, num_texts=100000)
if t:
    print(f'DONE: {t.get_vocab_size()} tokens')
else:
    print('FAILED')
