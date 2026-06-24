"""FCFModel — HuggingFace-compatible concept navigation model."""
import os, json, math, threading
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

        from eva.symbolic.fcf_config import EnvironmentResolver
        src_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "real_data")
        self._env = EnvironmentResolver()
        self._env._data_dir_override = src_data_dir
        self._bpe_fallback: Optional[str] = None

        # Lazy load: components are initialized on first use
        self._space: Optional[ConceptSpace] = None
        self._tok: Optional[_SPTokenizer] = None
        self._lattice: Optional[SyntaxLattice] = None
        self._generator: Optional[CrystalGenerator] = None
        self._load_lock = threading.Lock()

    def _load(self):
        if self._space is not None:
            return
        with self._load_lock:
            if self._space is not None:
                return

        space_path = self._env.cs_path
        lattice_path = self._env.lattice_path
        bpe_path = self._env.bpe_model_path

        if not os.path.exists(space_path):
            raise FileNotFoundError(f"ConceptSpace not found at {space_path}. Ensure the model directory contains concept_space.json.")

        # Use fallback BPE path if model not in custom dir
        if not os.path.exists(bpe_path) and self._bpe_fallback is not None:
            bpe_path = self._bpe_fallback

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

        if max_words is not None:
            self.generator.max_words = max_words
        self.generator.temperature = temperature

        if seed_word:
            query_words = [seed_word] if prompt else []
        elif prompt:
            words = prompt.strip().split()
            seed_word = words[0]
            query_words = words[1:] if len(words) > 1 else []
        else:
            seed_word = None
            query_words = ["человек"]

        result = self.generator.generate(seed_word=seed_word, query_words=query_words)

        hm = self.generator.hormones
        return FCFOutput(
            text=result.text,
            concept_path=result.concept_path,
            hormones={
                "da": float(hm.dopamine),
                "5ht": float(hm.serotonin),
                "na": float(hm.noradrenaline),
                "ach": float(hm.acetylcholine),
            },
            confidence=0.0,
            intent_anchor=None,
            semantic_delta=result.semantic_delta,
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
        from eva.symbolic.fcf_config import EnvironmentResolver
        _save_env = EnvironmentResolver()
        _save_env._data_dir_override = save_directory
        self._space.save(_save_env.cs_path)
        self._lattice.save(_save_env.lattice_path)
        gen_config_path = os.path.join(save_directory, "gen_config.json")
        with open(gen_config_path, 'w', encoding='utf-8') as f:
            json.dump(self.generator.config, f, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = FCFConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(config)
        # Override data_dir to the pretrained model directory
        model._env._data_dir_override = pretrained_model_name_or_path
        # Fallback BPE path if model not in custom dir
        if not os.path.exists(model._env.bpe_model_path):
            model._bpe_fallback = model._env.bpe_model_path
        model._load()
        return model
