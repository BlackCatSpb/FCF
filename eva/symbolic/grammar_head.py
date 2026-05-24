"""GrammarHead: нейрослой грамматических правил поверх координатного трансформера."""
import torch, torch.nn as nn, torch.nn.functional as F

class GrammarHead(nn.Module):
    """
    Грамматический слой: предсказывает грамматически корректные продолжения траектории.
    
    Вход:  координаты от трансформера [B, L, 24]
    Выход: logits [B, L, vocab_size]
    
    Архитектура: Conv1D context window + MLP grammar encoder.
    """
    
    def __init__(self, coord_dim=24, vocab_size=157, hidden=64):
        super().__init__()
        self.coord_dim = coord_dim
        self.vocab_size = vocab_size
        
        self.context_conv = nn.Sequential(
            nn.Conv1d(coord_dim, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        
        self.grammar_encoder = nn.Sequential(
            nn.Linear(coord_dim + hidden, hidden),
            nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, vocab_size),
        )
        
        self.pos_embed = nn.Embedding(128, coord_dim)
    
    def forward(self, coords):
        B, L, D = coords.shape
        positions = torch.arange(L, device=coords.device).clamp(0, 127)
        pos_emb = self.pos_embed(positions).unsqueeze(0).expand(B, -1, -1)
        x = coords + pos_emb
        
        ctx = self.context_conv(x.transpose(1, 2)).transpose(1, 2)
        combined = torch.cat([x, ctx], dim=-1)
        return self.grammar_encoder(combined)
