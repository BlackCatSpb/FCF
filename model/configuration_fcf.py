"""FCFConfig — HuggingFace-compatible configuration for FCF concept model."""
from transformers import PretrainedConfig
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eva', 'symbolic'))
from fcf_config import FCFConfig as _RealFCFConfig


class FCFConfig(PretrainedConfig):
    model_type = "fcf"

    def __init__(
        self,
        vocab_size=None,
        hidden_size=None,
        concept_dim=None,
        max_length=512,
        beam_width=None,
        concept_temp=None,
        word_temp=0.3,
        theta_tau=None,
        learning_rate=None,
        use_hormones=True,
        use_induction=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        _real = _RealFCFConfig()
        self.vocab_size = vocab_size if vocab_size is not None else _real.vocab_size
        self.hidden_size = hidden_size if hidden_size is not None else _real.dim
        self.concept_dim = concept_dim if concept_dim is not None else _real.dim
        self.max_length = max_length
        self.beam_width = beam_width if beam_width is not None else _real.beam_width
        self.concept_temp = concept_temp if concept_temp is not None else _real.concept_temp
        self.word_temp = word_temp
        self.theta_tau = theta_tau if theta_tau is not None else _real.theta_tau
        self.learning_rate = learning_rate if learning_rate is not None else _real.fast_lr
        self.use_hormones = use_hormones
        self.use_induction = use_induction
