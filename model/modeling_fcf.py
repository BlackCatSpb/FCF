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

from eva.symbolic.concept_space import ConceptSpace
from eva.symbolic.concept_tokenizer import ConceptTokenizer
from eva.symbolic.syntax_lattice import SyntaxLattice
from eva.symbolic.hormonal_system import HormonalSystem
from eva.symbolic.concept_inductor import ConceptInductor
from eva.symbolic.crystal_generator import CrystalGenerator


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
        self._tok: Optional[ConceptTokenizer] = None
        self._lattice: Optional[SyntaxLattice] = None
        self._generator: Optional[CrystalGenerator] = None

    def _load(self):
        if self._space is not None:
            return

        data_dir = self._data_dir
        space_path = os.path.join(data_dir, "concept_space.json")
        lattice_path = os.path.join(data_dir, "syntax_lattice.json")
        bpe_path = os.path.join(data_dir, "bpe_tokenizer.json")
        skeleton_path = os.path.join(data_dir, "concept_skeleton.json")

        if not os.path.exists(space_path):
            raise FileNotFoundError(f"ConceptSpace not found at {space_path}. Run rebuild_v2.py first.")

        # Load tokenizer
        self._tok = ConceptTokenizer(bpe_path=bpe_path, skeleton_path=skeleton_path)
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
            "concept_temp": self.config.concept_temp,
            "word_temp": self.config.word_temp,
            "theta_tau": self.config.theta_tau,
            "learning_rate": self.config.learning_rate,
        }
        self._generator = CrystalGenerator(self._space, self._tok, self._lattice, gen_config)

    @property
    def space(self) -> ConceptSpace:
        self._load()
        return self._space

    @property
    def tok(self) -> ConceptTokenizer:
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
        """HF-compatible forward pass.

        If input_ids is provided, we encode them, extract concept intent,
        and return a FCFOutput with the model's internal representation.

        This allows HF pipelines to use the model.
        """
        self._load()
        if input_ids is None:
            return FCFOutput(text="", concept_path=[], confidence=0.0)

        # Decode input IDs to text
        text = self._tok.decode(input_ids[0].tolist() if hasattr(input_ids, 'tolist') else input_ids)
        words = text.split()

        # Extract core concept via semantic gate
        core_cid, modifier_field, centroid, noise = self.generator.gate.extract_core(words)

        anchor = self._space.concept_info.get(core_cid, {}).get("anchor", "?")

        return FCFOutput(
            text=text,
            concept_path=[core_cid],
            intent_anchor=anchor,
            semantic_delta=float(len(noise) / max(len(words), 1)),
            confidence=float(self.generator._query_confidence),
        )

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
            query_words = prompt.strip().split()
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
            confidence=float(self.generator._query_confidence),
            intent_anchor=self._space.concept_info.get(
                result.get("core_cid", 0), {}).get("anchor", None),
            semantic_delta=float(result.get("intent_drift", 0.0)),
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
        """Save model + config."""
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        # config.json written by HF; no model weights to save (stateless)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = FCFConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(config)
