from eva.symbolic.char_vocab import CharacterVocab
cv = CharacterVocab()
print(f'vocab_size: {cv.vocab_size}')
print(f'max index: {max(cv._idx_to_char.keys())}')
print(f'last few idx_to_char: {[(k,v) for k,v in sorted(cv._idx_to_char.items())[-10:]]}')
