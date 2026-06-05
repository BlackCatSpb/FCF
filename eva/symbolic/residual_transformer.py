"""
ResidualTransformer — small causal transformer above closed-form heads.
Learns residuals: final_logits = head_prior_logits + transformer_logits.

Architecture:
  token_embed: 4101 → 128
  pos_embed: max_len=512, learned
  2-4 causal self-attention layers (d=128, h=4, ff=512)
  output_proj: 128 → 4101

Training: next-token prediction via cross-entropy on head residuals.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, n_heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, n_heads, T, head_dim]

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        mask = torch.triu(torch.ones(1, 1, T, T, device=x.device), diagonal=1).bool()
        att.masked_fill_(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, ff_dim=512, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class ResidualTransformer(nn.Module):
    """
    Small causal transformer for residual correction above closed-form heads.
    Input: subword token IDs [B, T] with hierarchical encoding.
    Output: residual logits [B, T, V] to add to head_prior_logits.
    """

    def __init__(self, vocab_size=4101, d_model=128, n_layers=3,
                 n_heads=4, ff_dim=512, max_len=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=True)

        # Initialize output to small values for stable residual learning
        nn.init.normal_(self.output.weight, mean=0.0, std=0.01)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, head_prior_logits=None):
        """
        Args:
            x: [B, T] token IDs (subword value, 0-4100)
            head_prior_logits: [B, T, V] from heads (optional for inference)
        Returns:
            residual_logits: [B, T, V] if head_prior_logits is None
            final_logits: [B, T, V] if head_prior_logits is provided
        """
        B, T = x.shape
        assert T <= self.max_len

        tok = self.token_embed(x)
        pos = self.pos_embed(torch.arange(T, device=x.device).unsqueeze(0))
        h = self.dropout(tok + pos)

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        residual_logits = self.output(h)  # [B, T, V]

        if head_prior_logits is not None:
            return residual_logits + head_prior_logits
        return residual_logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(self, lr=3e-4, weight_decay=0.01):
        decay = set()
        no_decay = set()
        for name, param in self.named_parameters():
            if 'embed' in name or 'ln' in name or 'bias' in name:
                no_decay.add(name)
            else:
                decay.add(name)
        param_groups = [
            {'params': [p for n, p in self.named_parameters() if n in decay],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.named_parameters() if n in no_decay],
             'weight_decay': 0.0},
        ]
        return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))


if __name__ == '__main__':
    model = ResidualTransformer(vocab_size=4101, d_model=128, n_layers=3)
    print("ResidualTransformer: %.2fM params" % (model.get_num_params() / 1e6))

    # Test forward
    x = torch.randint(0, 4101, (2, 32))
    logits = model(x)
    print("Input: [2, 32] -> Output: %s" % str(logits.shape))

    # Test with head priors
    priors = torch.randn(2, 32, 4101) * 0.5
    final = model(x, head_prior_logits=priors)
    print("With priors: %s" % str(final.shape))
