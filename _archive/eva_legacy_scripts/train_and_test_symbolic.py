"""
EVA Symbolic — обучение + тестирование.

Обучает символьную модель на чистом русском тексте,
затем проверяет: порядок символов, слова, словосочетания.

Тесты:
1. Character order: может ли модель воспроизвести известное слово?
2. Word completion: может ли модель закончить начатое слово?
3. Phrase coherence: может ли связать два слова?
4. Grammar patterns: обнаружила ли модель грамматические паттерны?
"""

import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

from eva.symbolic import *
from eva.primordial_layer import PrimordialLayer
from eva.config import FCFConfig
import torch
import numpy as np

# === ИНИЦИАЛИЗАЦИЯ ===
print("=" * 60)
print("EVA Symbolic — Обучение + Тестирование")
print("=" * 60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Устройство: {device}")

config = FCFConfig()
config.d_model = 256
config.vocab_size = 156
config.num_heads = 8
config.max_seq_len = 256

layer = PrimordialLayer(config)
if device == 'cuda':
    layer = layer.cuda()
print(f"Модель: {layer.summary()}")

char_vocab = CharacterVocab()
trainer = PotentialTrainer(layer=layer, char_vocab=char_vocab, embed_dim=256)

# === ОБУЧЕНИЕ ===
text_file = os.path.join(os.path.dirname(__file__), "real_data", "clean_ru.txt")
max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

print(f"\nДатасет: {text_file}")
print(f"Шагов: {max_steps}")
print()

# Используем пред-токенизированный .npy (355 MB, 93M токенов)
npy_file = os.path.join(os.path.dirname(__file__), "real_data", "clean_ru_ids.npy")
if os.path.exists(npy_file):
    print(f"Pre-tokenized dataset: {os.path.getsize(npy_file)/1024/1024:.0f} MB")
    print(f"Batch size: 128")
    trainer.train_on_npy(npy_file, max_steps=max_steps, batch_size=128, block_size=128, log_interval=500, save_interval=2000)
else:
    print("Pre-tokenized dataset not found, using text file")
    trainer.train_on_file_batched(text_file, max_steps=max_steps, batch_size=64, log_interval=500, save_interval=2000)

# === ТЕСТЫ ===
print("\n" + "=" * 60)
print("ТЕСТЫ")
print("=" * 60)

generator = SymbolicGenerator(
    layer=layer,
    char_vocab=char_vocab,
    potential_field=trainer.potential_field,
    contradiction_filter=trainer.contradiction_filter,
    grammar=trainer.grammar,
    concept_miner=trainer.concept_miner,
    topological_field=trainer.topological_field,
)

# Тест 1: Порядок символов в слове "привет"
print("\n[Тест 1] Порядок символов в слове 'привет':")
word = "привет"
for i in range(1, len(word)):
    prefix = char_vocab.encode(word[:i])
    next_char = char_vocab.encode(word[i])[1]  # [BOS, char, EOS] -> char idx
    dist = trainer.potential_field.get_continuation_potential(prefix[-2]).cpu().numpy()  # last char
    rank = np.sum(dist > dist[next_char])
    print(f"  '{word[:i]}' -> ожидается '{word[i]}': позиция {rank}/{len(dist)} (топ-{np.sum(dist > 0.01)})")

# Тест 2: Завершение слова
print("\n[Тест 2] Завершение слов:")
test_words = ["мам", "пап", "доч", "сын", "стол", "дом", "книг", "рук", "вод", "ног"]
for prefix in test_words:
    syms = char_vocab.encode(prefix)[1:-1]
    cont = trainer.potential_field.get_continuation_potential(syms[-1]).cpu().numpy()
    top5 = np.argsort(cont)[-5:][::-1]
    top5_chars = [char_vocab.idx_to_char(int(i)) for i in top5]
    print(f"  '{prefix}...' -> {', '.join(top5_chars)}")

# Тест 3: Грамматические паттерны
print("\n[Тест 3] Грамматические паттерны (диграммы):")
if 0 in trainer.grammar.patterns:
    digrams = list(trainer.grammar.patterns[0].values())
    digrams.sort(key=lambda p: p.coherence_score, reverse=True)
    for d in digrams[:10]:
        chars = [char_vocab.idx_to_char(s) for s in d.symbol_indices[:2]]
        print(f"  {''.join(chars)} (coherence={d.coherence_score:.3f})")
else:
    print("  Диграммы ещё не обнаружены")

# Тест 4: Противоречия
print(f"\n[Тест 4] Противоречия: обнаружено {len(trainer.contradiction_filter.forbidden)}")
if len(trainer.contradiction_filter.forbidden) > 0:
    for key in list(trainer.contradiction_filter.forbidden.keys())[:5]:
        fc = trainer.contradiction_filter.forbidden[key]
        print(f"  {fc.key} (conf={fc.confidence:.2f}, type={fc.contradiction_type.value})")

# Тест 5: Концепты
print(f"\n[Тест 5] Концепты: обнаружено {len(trainer.concept_miner.concepts)}")
for cid, c in list(trainer.concept_miner.concepts.items())[:5]:
    chars = [char_vocab.idx_to_char(s) for s in c.symbol_indices[:10]]
    print(f"  {''.join(chars)}... (quality={c.quality:.3f}, status={c.status})")

# Тест 6: Генерация по префиксу
print("\n[Тест 6] Генерация (SymbolicGenerator):")
prompts = ["при", "ма", "ст", "чело", "космо", "зем"]
for prompt in prompts:
    syms = char_vocab.encode(prompt)[1:-1]  # без BOS/EOS
    gen = generator.generate(syms, max_new_symbols=20, temperature=0.7)
    text = char_vocab.decode(gen)
    print(f"  '{prompt}...' -> '{text}'")

# === СТАТИСТИКА ===
print("\n" + "=" * 60)
print("СТАТИСТИКА")
print("=" * 60)
print(f"  {trainer.summary()}")
print(f"  {trainer.grammar.summary()}")
print(f"  {trainer.topological_field.summary()}")
print(f"  {trainer.contradiction_filter.summary()}")
print(f"  {trainer.concept_miner.summary()}")
print(f"  {trainer.clusterer.summary()}")
