"""FCFModel — HuggingFace-compatible concept navigation model."""
import os, json, math
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import numpy as np
from transformers import PreTrainedModel
# GenerationMixin mixin for .generate() support
try:
    from transformers import GenerationMixin as HFGenerationMixin
except ImportError:
    HFGenerationMixin = object

from .configuration_fcf import FCFConfig

import sentencepiece as spm
from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.hormonal_system import HormonalSystem
from eva.symbolic.crystal_generator import CrystalGenerator



class _SPTokenizer:

    def __init__(self, model_path=None):
        self.sp = spm.SentencePieceProcessor()
        self._model_path = model_path

    def initialize(self):
        if self._model_path:
            self.sp.load(self._model_path)

    def encode(self, text, add_bos=False, add_eos=False):
        return self.sp.encode(text, add_bos=add_bos, add_eos=add_eos)

    def decode(self, ids):
        return self.sp.decode(ids)

    def IdToPiece(self, cid):
        return self.sp.IdToPiece(cid)

    def PieceToId(self, piece):
        return self.sp.PieceToId(piece)

    def word_to_cid(self, word):
        return self.sp.encode(word)[0]

    def vocab_size(self):
        return self.sp.vocab_size()

    def __len__(self):
        return self.sp.get_piece_size()


@dataclass
class FCFOutput:
    text: str = ""
    concept_path: List[int] = field(default_factory=list)
    hormones: Optional[Dict[str, float]] = None
    confidence: float = 1.0
    intent_anchor: Optional[str] = None
    semantic_delta: float = 0.0


class FCFModel(PreTrainedModel, HFGenerationMixin):
    config_class = FCFConfig
    base_model_prefix = "fcf"
    supports_gradient_checkpointing = False

    def __init__(self, config: FCFConfig):
        super().__init__(config)
        self.config = config

        # Load data files from the standard real_data directory
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "real_data")
        self._data_dir = data_dir

        # Lazy load: components are initialized on first use
        self._space: Optional[ConceptSpace] = None
        self._tok: Optional[_SPTokenizer] = None
        self._lattice: Optional[SyntaxLattice] = None
        self._generator: Optional[CrystalGenerator] = None

    def _load(self):
        if self._space is not None:
            return

        data_dir = self._data_dir
        space_path = os.path.join(data_dir, "concept_space.json")
        lattice_path = os.path.join(data_dir, "syntax_lattice.json")
        bpe_path = os.path.join(data_dir, "bpe_ru_146k.model")

        if not os.path.exists(space_path):
            raise FileNotFoundError(f"ConceptSpace not found at {space_path}. Ensure the model directory contains concept_space.json.")

        # Load tokenizer
        self._tok = _SPTokenizer(model_path=bpe_path)
        self._tok.initialize()

        # Load ConceptSpace
        self._space = ConceptSpace.load(space_path)

        # Load SyntaxLattice
        self._lattice = SyntaxLattice()
        self._lattice.load(lattice_path)

        # Build generator config
        gen_config = {
            "beam_width": self.config.beam_width,
            "max_words": self.config.max_length,
            "min_words": 3,
            "concept_temp": self.config.concept_temp,
            "theta_tau": self.config.theta_tau,
            "learning_rate": self.config.learning_rate,
            "top_p": 0.9,
            "len_norm_alpha": 0.7,
            "block_ngram": 4,
            "mmi_lambda": 0.2,
        }
        self._generator = CrystalGenerator(self._space, self._tok, self._lattice, gen_config)

    @property
    def space(self) -> ConceptSpace:
        self._load()
        return self._space

    @property
    def tok(self) -> _SPTokenizer:
        self._load()
        return self._tok

    @property
    def lattice(self) -> SyntaxLattice:
        self._load()
        return self._lattice

    @property
    def generator(self) -> CrystalGenerator:
        self._load()
        return self._generator

    # ── HuggingFace interface ──

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        self._load()
        if input_ids is None:
            return FCFOutput(text="", concept_path=[], confidence=0.0)
        text = self._tok.decode(input_ids[0].tolist() if hasattr(input_ids, 'tolist') else input_ids)
        return FCFOutput(text=text, concept_path=[], confidence=0.0)

    def generate(
        self,
        prompt: Optional[str] = None,
        seed_word: Optional[str] = None,
        max_words: Optional[int] = None,
        temperature: float = 0.5,
        **kwargs,
    ) -> FCFOutput:
        """Generate text from a prompt or seed word.

        Args:
            prompt: full query text (e.g. "расскажи про войну")
            seed_word: single seed word (overrides prompt for seed)
            max_words: maximum words to generate
            temperature: generation temperature

        Returns:
            FCFOutput with generated text and metadata
        """
        self._load()

        gen_config = dict(self.generator.config)
        if max_words is not None:
            gen_config["max_words"] = max_words
            self.generator.max_words = max_words
        gen_config["temperature"] = temperature
        self.generator.config = gen_config

        if prompt:
            words = prompt.strip().split()
            seed_word = words[0]
            query_words = words[1:] if len(words) > 1 else []
        elif seed_word:
            query_words = [seed_word]
        else:
            query_words = ["человек"]

        result = self.generator.generate(query_words=query_words)

        hm = self.generator.hormones
        return FCFOutput(
            text=result.get("text", ""),
            concept_path=result.get("concept_path", []),
            hormones={
                "da": float(hm.dopamine),
                "5ht": float(hm.serotonin),
                "na": float(hm.noradrenaline),
                "ach": float(hm.acetylcholine),
            },
            confidence=0.0,
            intent_anchor=None,
            semantic_delta=float(result.get("semantic_delta", 0.0)),
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def _reorder_cache(self, past_key_values, beam_idx):
        return past_key_values

    # ── External Training Interface ──

    def train_from_text(self, text: str) -> int:
        """Train model from external text data.

        Decodes text → extracts core concepts → builds connections →
        updates role_memory → organizes semantic space.

        Not gradient descent: pure structure extraction and accumulation.

        Args:
            text: input Russian text (one or more sentences)

        Returns:
            number of sentences processed
        """
        self._load()
        return self._generator.train_from_text(text)

    def train_from_file(self, file_path: str) -> int:
        """Train model from a text file.

        Args:
            file_path: path to .txt file with Russian text

        Returns:
            number of sentences processed
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return self.train_from_text(text)

    # ── Save / Load ──

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save model + config + model state."""
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        self._load()  # ensure loaded
        # Save model state
        self._space.save(os.path.join(save_directory, "concept_space.json"))
        self._lattice.save(os.path.join(save_directory, "syntax_lattice.json"))
        # Save gen config
        gen_config_path = os.path.join(save_directory, "gen_config.json")
        with open(gen_config_path, 'w', encoding='utf-8') as f:
            json.dump(self.generator.config, f, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = FCFConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(config)
        # Load model state from the same directory
        model._data_dir = pretrained_model_name_or_path
        model._load()
        return model
