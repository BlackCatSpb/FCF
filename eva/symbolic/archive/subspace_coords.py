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
    
    def set_coordinates(self, coords):
        assert coords.shape[0] == self.vocab_size
        with torch.no_grad():
            self.symbol_coords.copy_(coords[:, :self.sym_dim])
    
    def forward(self, token_ids):
        ids = token_ids.clamp(0, self.vocab_size - 1)
        sym = self.symbol_coords[ids] * self.scale  # [B, L, sym_dim]
        return sym


class WordWeightEncoder(nn.Module):
    """
    Пулинг токенов в слово: trainable weighted attention + boundary-aware.
    
    Три механизма:
    1. Self-attention across tokens → weights
    2. Boundary-weighted pool: WORD_OPEN/CLOSE сигнализируют границы
    3. Обучаемый word-вектор (weighted sum h с attention weights)
    
    Returns:
        word_vectors: [B, N_words, D] — центроиды слов
        word_weights: [B, N_words] — важность каждого слова
        boundaries: [B, L, 3] — soft boundary scores (start/inside/end)
    """
    
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        head_dim = d_model // n_heads
        
        self.scale = d_model ** -0.5
        
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        
        # Boundary detection: h → (word_start, inside, word_end) logits
        self.boundary_proj = nn.Linear(d_model, d_model // 2)
        self.boundary_out = nn.Linear(d_model // 2, 3)
        
        # Word vector projection: weighted sum → word centroid
        self.word_proj = nn.Linear(d_model, d_model)
        
        # Global word importance
        self.importance = nn.Sequential(
            nn.Linear(d_model, 32), nn.SiLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
    
    def forward(self, x, boundary_logits=None):
        """
        x: [B, L, D] — hidden states
        boundary_logits: [B, L, 3] or None — from BoundaryDetectionHead
        
        Returns:
            word_vecs: [B, N, D] (padded to max_words)
            word_weights: [B, N]
            boundaries: [B, L, 3] — refined boundary scores
        """
        B, L, D = x.shape

        # Token importance via multi-head attention
        q = self.to_q(x).view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
        k = self.to_k(x).view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
        v = self.to_v(x).view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal, float('-inf'))
        attn = torch.softmax(attn, dim=-1)

        token_weights = attn.mean(dim=1).mean(dim=-2)

        # Boundary scores: use shared BoundaryDetectionHead output if available
        if boundary_logits is not None:
            boundaries = boundary_logits
        else:
            h = F.silu(self.boundary_proj(x))
            boundaries = self.boundary_out(h)

        boundary_probs = torch.softmax(boundaries, dim=-1)
        word_start_mask = boundary_probs[..., 0] > 0.5
        word_ids = torch.cumsum(word_start_mask.int(), dim=-1)
        N_words = word_ids.max().item() + 1

        # Vectorized scatter-add
        idx = word_ids.unsqueeze(-1).expand(-1, -1, D)
        w = token_weights.unsqueeze(-1)
        word_vecs = torch.zeros(B, N_words, D, device=x.device)
        word_vecs.scatter_add_(1, idx, x * w)

        count_idx = word_ids.unsqueeze(-1)
        word_counts = torch.zeros(B, N_words, 1, device=x.device)
        word_counts.scatter_add_(1, count_idx, w)

        word_vecs = word_vecs / (word_counts + 1e-8)
        word_weights = self.importance(word_vecs).squeeze(-1)

        return word_vecs, word_weights, boundaries
