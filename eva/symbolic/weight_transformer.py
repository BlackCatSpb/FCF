"""
WeightTransformer — lightweight module that learns to weight 6 heads.

Architecture: token embed(8) + 6 scalars → 32 → 6 (Softplus), ~35K params.
"""
import torch
import torch.nn as nn


class WeightTransformer(nn.Module):
    def __init__(self, vocab_size: int = 4101, embed_dim: int = 8, hidden: int = 32):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        in_dim = embed_dim + 6  # embed + 5 scalars + flags_norm
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 6),
            nn.Softplus(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, prev_token_id: torch.Tensor,
                word_len: torch.Tensor, pos_in_word: torch.Tensor,
                word_num: torch.Tensor, pos_in_sent: torch.Tensor,
                sent_len: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
        tok = self.token_embed(prev_token_id)  # (batch, embed_dim)
        # Unsqueeze scalars to (batch, 1) for concatenation
        scalars = torch.stack([word_len, pos_in_word, word_num, pos_in_sent, sent_len, flags], dim=-1)
        x = torch.cat([tok, scalars], dim=-1)  # (batch, embed_dim + 6)
        w = self.net(x)  # (batch, 6), positive via Softplus
        # Нормализация: сумма весов = 6.0 (по числу голов)
        # Предотвращает доминирование одной головы за счёт ограничения бюджета
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8) * 6.0
        return w


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
