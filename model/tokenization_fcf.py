"""FCFTokenizer — HuggingFace-compatible tokenizer wrapping ConceptTokenizer."""
import os, json
from typing import List, Optional
from transformers import PreTrainedTokenizer

import sentencepiece as spm


class FCFTokenizer(PreTrainedTokenizer):
    vocab_files_names = {
        "spm_file": "bpe_ru_146k.model",
    }

    def __init__(
        self,
        spm_file: Optional[str] = None,
        model_max_length: int = 512,
        **kwargs,
    ):
        self.spm_file = spm_file
        self._sp = None

        super().__init__(
            model_max_length=model_max_length,
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            **kwargs,
        )

    @property
    def sp(self):
        if self._sp is None and self.spm_file:
            self._sp = spm.SentencePieceProcessor()
            self._sp.load(self.spm_file)
        return self._sp

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size() if self.sp else 0

    def _tokenize(self, text: str) -> List[str]:
        if not self.sp:
            return list(text)
        ids = self.sp.encode(text)
        return [str(i) for i in ids]

    def _tokenize_with_text(self, text: str) -> List[tuple]:
        if not self.sp:
            return [(c, c) for c in text]
        pieces = self.sp.encode_as_pieces(text)
        ids = self.sp.encode(text)
        return list(zip([str(i) for i in ids], pieces))

    def _convert_token_to_id(self, token: str) -> int:
        return int(token)

    def _convert_id_to_token(self, token_id: int) -> str:
        return str(token_id)

    def get_vocab(self):
        if not self.sp:
            return {}
        return {str(i): i for i in range(self.sp.get_piece_size())}

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        return (self.spm_file or "",)

    def encode_text(self, text: str) -> List[int]:
        if not self.sp:
            return []
        return self.sp.encode(text)

    def decode_text(self, token_ids: List[int]) -> str:
        if not self.sp:
            return ""
        return self.sp.decode(token_ids)

    def concept_metadata(self, token_ids: List[int]):
        return []
