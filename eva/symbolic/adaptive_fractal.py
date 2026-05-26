"""
AdaptiveFractalAttention — динамическое число уровней.

LevelController: оценивает сложность входа → выделяет головы по уровням.
ConnectionCoordinateHead: вычисляет edge-вектора между центроидами слов.
"""

import torch, torch.nn as nn, torch.nn.functional as F
from typing import Tuple


class LevelController(nn.Module):
    """Динамический аллокатор уровней: сложность → число уровней → распределение голов."""
    
    def __init__(self, d_model=128, max_levels=8, total_heads=32):
        super().__init__()
        self.max_levels = max_levels
        self.total_heads = total_heads
        self.min_levels = 1
        
        self.complexity_conv = nn.Conv1d(d_model, 1, kernel_size=5, padding=2)
        self.level_gate = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, max_levels), nn.Softmax(dim=-1)
        )
        
        # Scales per level: 1, 2, 4, 8, 16, 32, 64, 128
        self.register_buffer('level_scales', torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, L, D]
        Returns: (num_active_levels [B], head_allocation [B, max_levels], level_probs [B, max_levels])
        """
        B, L, D = x.shape
        
        complexity = self.complexity_conv(x.transpose(1, 2)).squeeze(1)  # [B, L]
        mean_complexity = complexity.mean(dim=-1, keepdim=True)  # [B, 1]
        
        level_probs = self.level_gate(x.mean(dim=1))  # [B, max_levels]
        
        active_mask = level_probs > 0.05
        num_active = active_mask.sum(dim=-1, keepdim=True).float()  # [B, 1]
        num_active = num_active.clamp(self.min_levels, self.max_levels)
        
        head_allocation = (level_probs * self.total_heads).round().long()
        head_allocation = head_allocation.clamp(min=1)
        
        # Normalize allocations to sum to total_heads
        total_alloc = head_allocation.sum(dim=-1, keepdim=True)  # [B, 1]
        head_allocation = (head_allocation.float() * self.total_heads / total_alloc.clamp(min=1)).round().long()
        head_allocation = head_allocation.clamp(min=1)
        
        return num_active, head_allocation, level_probs


class AdaptiveFractalAttention(nn.Module):
    """
    Фрактальное внимание с динамическим числом уровней.
    Заменяет FractalAttention из fractal_v2.py.
    """
    
    def __init__(self, d_model=128, max_levels=8, total_heads=32):
        super().__init__()
        self.d_model = d_model
        self.max_levels = max_levels
        self.total_heads = total_heads
        self.head_dim = max(2, d_model // total_heads)
        self.effective_heads = d_model // self.head_dim
        
        self.level_controller = LevelController(d_model, max_levels, self.effective_heads)
        
        self.W_Q = nn.Linear(d_model, self.effective_heads * self.head_dim, bias=False)
        self.W_K = nn.Linear(d_model, self.effective_heads * self.head_dim, bias=False)
        self.W_V = nn.Linear(d_model, self.effective_heads * self.head_dim, bias=False)
        self.W_O = nn.Linear(self.effective_heads * self.head_dim, d_model, bias=False)
        
        self.level_bias = nn.Parameter(torch.zeros(max_levels, 1, 1))
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Linear(32, max_levels), nn.Sigmoid(),
        )
        self.coord_proj = nn.Linear(d_model, 2, bias=False)
    
    def compute_manifold_bias(self, x, scale):
        B, L, D = x.shape
        proj = self.coord_proj(x)  # [B, L, 2]
        diffs = proj.unsqueeze(2) - proj.unsqueeze(1)
        dists = torch.norm(diffs, dim=-1)
        sigma = scale * 0.5
        return torch.exp(-dists ** 2 / (2 * sigma ** 2 + 1e-8))
    
    def forward(self, x):
        B, L, D = x.shape
        H = self.effective_heads
        Dh = self.head_dim
        
        num_active, head_alloc, level_probs = self.level_controller(x)
        
        q = self.W_Q(x).view(B, L, H, Dh)
        k = self.W_K(x).view(B, L, H, Dh)
        v = self.W_V(x).view(B, L, H, Dh)
        
        outputs = []
        head_offset = 0
        
        for level in range(self.max_levels):
            n_heads = int(head_alloc[0, level].item())
            if n_heads == 0: continue
            
            level_outputs = []
            for h in range(head_offset, head_offset + n_heads):
                if h >= H: break
                
                scores = torch.matmul(q[:, :, h], k[:, :, h].transpose(-2, -1)) / (Dh ** 0.5)
                
                scale = self.level_controller.level_scales[level].item()
                manifold_bias = self.compute_manifold_bias(x, scale)
                scores = scores + manifold_bias + self.level_bias[level]
                
                causal = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
                scores = scores.masked_fill(causal, float('-inf'))
                
                attn = F.softmax(scores, dim=-1)
                out = torch.matmul(attn, v[:, :, h])
                level_outputs.append(out)
            
            if level_outputs:
                actual_n = len(level_outputs)
                level_out = torch.stack(level_outputs, dim=1)  # [B, actual_n, L, Dh]
                level_out = level_out.permute(0, 2, 1, 3).reshape(B, L, actual_n * Dh)
                outputs.append(level_out)
            
            head_offset += n_heads
        
        if not outputs:
            return x
        
        combined = torch.cat(outputs, dim=-1)  # [B, L, H*Dh]
        if combined.shape[-1] < H * Dh:
            pad = torch.zeros(B, L, H * Dh - combined.shape[-1], device=x.device)
            combined = torch.cat([combined, pad], dim=-1)
        
        return self.W_O(combined), level_probs


class ConnectionCoordinateHead(nn.Module):
    """Вычисляет edge-вектора между центроидами соседних слов."""
    
    def __init__(self, d_model=128):
        super().__init__()
        self.edge_net = nn.Sequential(
            nn.Linear(d_model * 2 + d_model, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )
    
    def forward(self, word_centroids, context):
        """
        word_centroids: [B, num_words, D]
        context: [B, D] — sentence-level context
        
        Returns: connection_coords [B, num_words-1, D]
        """
        B, N, D = word_centroids.shape
        if N < 2:
            return torch.zeros(B, 0, D, device=word_centroids.device)
        
        left = word_centroids[:, :-1, :]   # [B, N-1, D]
        right = word_centroids[:, 1:, :]   # [B, N-1, D]
        ctx = context.unsqueeze(1).expand(-1, N-1, -1)  # [B, N-1, D]
        
        edge_input = torch.cat([left, right, ctx], dim=-1)  # [B, N-1, 3D]
        return self.edge_net(edge_input)  # [B, N-1, D]
