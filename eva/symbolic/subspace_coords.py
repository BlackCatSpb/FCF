"""MultiSubspaceEmbedding + WordWeight — координаты с подпространствами."""
import torch, torch.nn as nn, torch.nn.functional as F


class MultiSubspaceEmbedding(nn.Module):
    """
    Структурированные координаты: символ, слово, связь, предложение.
    Каждый уровень — своё подпространство в общем 128-мерном векторе.
    """
    
    def __init__(self, vocab_size=161, total_dim=128, sym_dim=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.total_dim = total_dim
        self.sym_dim = sym_dim
        
        self.symbol_coords = nn.Parameter(torch.randn(vocab_size, sym_dim) * 0.02)
        self.scale = nn.Parameter(torch.ones(1))
        
        # Projectors to other subspaces
        self.to_word = nn.Linear(total_dim, sym_dim)
        self.to_conn = nn.Linear(total_dim, sym_dim)
        self.to_sent = nn.Linear(total_dim, sym_dim)
        
        # Subspace weights (learned importance)
        self.subspace_weights = nn.Parameter(torch.ones(4))  # sym, word, conn, sent
    
    def set_coordinates(self, coords):
        assert coords.shape[0] == self.vocab_size
        self.symbol_coords.copy_(coords[:, :self.sym_dim])
    
    def forward(self, token_ids):
        ids = token_ids.clamp(0, self.vocab_size - 1)
        sym = self.symbol_coords[ids] * self.scale  # [B, L, sym_dim]
        return sym
    
    def full_coords(self, x_transformer):
        """x_transformer: [B, L, total_dim] — output from transformer."""
        return x_transformer


class WordWeightEncoder(nn.Module):
    """Обратное внимание: важные слова получают больший вес."""
    
    def __init__(self, d_model=128):
        super().__init__()
        self.weight_query = nn.Linear(d_model, d_model)
        self.weight_key = nn.Linear(d_model, d_model)
        self.importance_direction = nn.Parameter(torch.randn(d_model))
        nn.init.normal_(self.importance_direction, 0, 0.02)
    
    def forward(self, x, boundary_mask=None):
        """
        x: [B, L, D]
        Returns: weights [B, L], coord_shift [B, L, D]
        """
        q = self.weight_query(x)
        k = self.weight_key(x)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (x.shape[-1] ** 0.5), dim=-1)
        weights = attn.sum(dim=-2)  # [B, L]
        weights = weights / (weights.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        direction = self.importance_direction / (self.importance_direction.norm() + 1e-8)
        coord_shift = weights.unsqueeze(-1) * direction.unsqueeze(0).unsqueeze(0)
        return weights, coord_shift
