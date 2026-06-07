"""FCFConfig — HuggingFace-compatible configuration for FCF concept model."""
from transformers import PretrainedConfig


class FCFConfig(PretrainedConfig):
    model_type = "fcf"

    def __init__(
        self,
        vocab_size=8200,
        hidden_size=128,
        concept_dim=128,
        max_length=512,
        beam_width=3,
        concept_temp=0.5,
        word_temp=0.3,
        theta_tau=5.0,
        learning_rate=0.1,
        use_hormones=True,
        use_induction=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.concept_dim = concept_dim
        self.max_length = max_length
        self.beam_width = beam_width
        self.concept_temp = concept_temp
        self.word_temp = word_temp
        self.theta_tau = theta_tau
        self.learning_rate = learning_rate
        self.use_hormones = use_hormones
        self.use_induction = use_induction
