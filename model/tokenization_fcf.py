"""FCFTokenizer — HuggingFace-compatible tokenizer wrapping ConceptTokenizer."""
import os, json
from typing import List, Optional
from transformers import PreTrainedTokenizer

from eva.symbolic.concept_tokenizer import ConceptTokenizer as _ConceptTokenizer
from eva.symbolic.concept_net import ConceptSkeleton


class FCFTokenizer(PreTrainedTokenizer):
    vocab_files_names = {
        "bpe_file": "bpe_tokenizer.json",
        "skeleton_file": "concept_skeleton.json",
    }

    def __init__(
        self,
        bpe_file: Optional[str] = None,
        skeleton_file: Optional[str] = None,
        model_max_length: int = 512,
        **kwargs,
    ):
        self.bpe_file = bpe_file
        self.skeleton_file = skeleton_file
        self._tok = None

        super().__init__(
            model_max_length=model_max_length,
            pad_token="<|pad|>",
            bos_token="<|word_open|>",
            eos_token="<|sent_close|>",
            **kwargs,
        )

    @property
    def tok(self):
        if self._tok is None and self.bpe_file and self.skeleton_file:
            self._tok = _ConceptTokenizer(
                bpe_path=self.bpe_file,
                skeleton_path=self.skeleton_file,
            )
            self._tok.initialize()
        return self._tok

    @property
    def vocab_size(self) -> int:
        return len(self.tok) if self.tok else 8200

    def _tokenize(self, text: str) -> List[str]:
        if not self.tok:
            return list(text)
        ids = self.tok.encode(text)
        return [str(i) for i in ids]

    def _convert_token_to_id(self, token: str) -> int:
        return int(token)

    def _convert_id_to_token(self, token_id: int) -> str:
        return str(token_id)

    def get_vocab(self):
        if not self.tok:
            return {}
        return {str(i): i for i in range(len(self.tok))}

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        return (self.bpe_file or "", self.skeleton_file or "")

    def encode_text(self, text: str) -> List[int]:
        if not self.tok:
            return []
        return self.tok.encode(text)

    def decode_text(self, token_ids: List[int]) -> str:
        if not self.tok:
            return ""
        return self.tok.decode(token_ids)

    def concept_metadata(self, token_ids: List[int]):
        if not self.tok:
            return []
        return self.tok.metadata_from_ids(token_ids)
