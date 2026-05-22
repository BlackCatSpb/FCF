"""
TokenizerUtils — обучение и загрузка BPE-токенизатора.

Токенизатор обучается на русском корпусе (Wikipedia) с нуля.
Использует библиотеку HuggingFace `tokenizers` (BPE).
"""

import os
import json
from typing import Optional, Iterator
from loguru import logger


def train_bpe_tokenizer(
    output_path: str,
    text_iterator: Iterator[str],
    vocab_size: int = 50257,
    min_frequency: int = 2,
    max_token_length: int = 256,
    limit_alphabet: int = 1000,
    initial_alphabet: Optional[list] = None,
) -> "Tokenizer":
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
    except ImportError:
        raise ImportError("Установите `tokenizers`: pip install tokenizers")

    if initial_alphabet is None:
        initial_alphabet = list(
            " абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
            "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            ".,!?;:()[]{}'\"-—_/\\@#$%^&*+=<>|~`°№"
            "\n\r\t"
        )

    logger.info(
        f"[Tokenizer] Обучение BPE: vocab_size={vocab_size}, "
        f"min_freq={min_frequency}"
    )

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    special_tokens = [
        "<unk>", "<s>", "</s>", "<pad>",
        "<|im_start|>", "<|im_end|>",
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        max_token_length=max_token_length,
        initial_alphabet=initial_alphabet,
        limit_alphabet=limit_alphabet,
    )

    tokenizer.train_from_iterator(text_iterator, trainer=trainer)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tokenizer.save(output_path)
    logger.info(
        f"[Tokenizer] Токенизатор сохранён: {output_path} "
        f"(vocab_size={tokenizer.get_vocab_size()})"
    )

    _save_vocab_mapping(tokenizer, output_path)

    return tokenizer


def _save_vocab_mapping(tokenizer, tokenizer_path: str):
    vocab_path = tokenizer_path.replace(".json", "_vocab.json")
    try:
        vocab = tokenizer.get_vocab()
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[Tokenizer] Не удалось сохранить словарь: {e}")


def load_tokenizer(path: str):
    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise ImportError("Установите `tokenizers`: pip install tokenizers")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Токенизатор не найден: {path}")

    tokenizer = Tokenizer.from_file(path)
    logger.info(
        f"[Tokenizer] Загружен: {path} (vocab_size={tokenizer.get_vocab_size()})"
    )
    return tokenizer


def train_tokenizer_on_wikipedia(
    output_path: str,
    vocab_size: int = 50257,
    num_texts: int = 100000,
    min_text_length: int = 100,
) -> Optional["Tokenizer"]:
    from eva.data_manager import DataManager

    logger.info(f"[Tokenizer] Загрузка Wikipedia для обучения токенизатора...")

    wiki = DataManager.load_wikipedia(split="train", streaming=True)
    if wiki is None:
        logger.error("[Tokenizer] Wikipedia не загружена")
        return None

    def text_iterator():
        count = 0
        for item in wiki:
            text = item.get("text", "")
            if len(text) >= min_text_length:
                yield text
                count += 1
                if count >= num_texts:
                    break
        logger.info(f"[Tokenizer] Обработано текстов: {count}")

    return train_bpe_tokenizer(
        output_path=output_path,
        text_iterator=text_iterator(),
        vocab_size=vocab_size,
    )


def train_tokenizer_on_files(
    output_path: str,
    file_paths: list,
    vocab_size: int = 50257,
) -> "Tokenizer":
    def file_iterator():
        for file_path in file_paths:
            logger.info(f"[Tokenizer] Чтение: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line

    return train_bpe_tokenizer(
        output_path=output_path,
        text_iterator=file_iterator(),
        vocab_size=vocab_size,
    )


def create_fallback_tokenizer():
    """Создаёт символьный fallback-токенизатор для тестов."""
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    except ImportError:
        raise ImportError("Установите `tokenizers`")

    alphabet = list(
        " абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        ".,!?;:()[]{}'\"-—_/\\@#$%^&*+=<>|~`°№"
        "\n\r\t"
    )

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<unk>", "<s>", "</s>", "<pad>"]
    trainer = trainers.BpeTrainer(
        vocab_size=5000,
        special_tokens=special_tokens,
        show_progress=False,
        initial_alphabet=alphabet,
    )

    dummy_texts = [" ".join(alphabet)] * 10
    tokenizer.train_from_iterator(dummy_texts, trainer=trainer, length=len(dummy_texts))

    return tokenizer
