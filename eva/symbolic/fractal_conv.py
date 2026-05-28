"""
FractalConv2D + Hybrid Fractal Block — свёртки и внимание вместе.

FractalConv2D: многомерная (L × Dim), фрактальная (4 dilation уровня).
HybridFractalBlock: conv → attention → gate → FFN на всех уровнях.
"""

import torch, torch.nn as nn, torch.nn.functional as F


class FractalConv2D(nn.Module):
    """
    Многомерная фрактальная свёртка.
    4 уровня × dilation (L, Dim) × Conv2D.
    Видит и последовательность, и подпространства одновременно.
    """
    
    def __init__(self, dim=128, out_dim=None):
        super().__init__()
        out_dim = out_dim or dim
        ch_per_level = max(1, out_dim // 4)
        
        self.levels = nn.ModuleList([])
        total_ch = 0
        
        for l in range(4):
            dil_L = 2 ** l          # 1, 2, 4, 8
            dil_D = max(1, 2 ** (l // 2))  # 1, 1, 2, 2
            k_L = 3 if l < 2 else 5
            k_D = min(8 * (l + 1), dim)   # 8, 16, 32, 64
            
            conv = nn.Conv2d(1, ch_per_level, (k_L, k_D),
                           dilation=(dil_L, dil_D),
                           padding=(0, (k_D-1)*dil_D//2))  # manual causal padding for L, symmetric for D
            self.levels.append(conv)
            total_ch += ch_per_level
        
        self.proj = nn.Linear(total_ch, out_dim) if total_ch != out_dim else nn.Identity()
    
    def forward(self, x):
        B, L, D = x.shape
        x2d = x.unsqueeze(1)  # [B, 1, L, D]
        
        level_outs = []
        for l, conv in enumerate(self.levels):
            # Causal: pad only on LEFT side (past context)
            dil_L = 2 ** l
            k_L = 3 if l < 2 else 5
            left_pad = (k_L - 1) * dil_L
            
            # Manual left-only padding
            x_padded = F.pad(x2d, (0, 0, left_pad, 0))  # pad dim=-2 (L) on left
            
            out = conv(x_padded)  # [B, C_l, L, D]
            out = out[:, :, :L, :]  # trim to original length
            out = out.mean(dim=-1)  # [B, C_l, L]
            level_outs.append(out.transpose(1, 2))  # [B, L, C_l]
        
        combined = torch.cat(level_outs, dim=-1)
        
        if isinstance(self.proj, nn.Linear):
            return self.proj(combined)
        return combined


class HybridFractalBlock(nn.Module):
    """
    Гибридный блок: FractalConv2D + SparseAttention + StaticTopology + Gate + SGF-FFN.
    Coordinate Residual Stream + Subspace-Gated FFN.
    """
    
    def __init__(self, dim=128, max_levels=8, total_heads=32, d_ff=128, topology_layer=None):
        super().__init__()
        self.dim = dim
        
        self.fractal_conv = FractalConv2D(dim, dim)
        
        from .adaptive_fractal import AdaptiveFractalAttention
        self.attention = AdaptiveFractalAttention(
            d_model=dim, max_levels=max_levels, total_heads=total_heads
        )
        
        self.topology = topology_layer
        
        self.gate_conv = nn.Linear(dim * 2, dim)
        
        from .unified_transformer import RMSNorm
        self.norm_conv = RMSNorm(dim)
        self.norm_attn = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        
        from .unified_transformer import SwiGLUFFN
        self.ffn = SwiGLUFFN(dim, d_ff)
        
        # --- Coordinate Residual Stream: gate-векторы 128d ---
        self.coord_gate_in = nn.Parameter(torch.zeros(dim))
        self.coord_gate_out = nn.Parameter(torch.zeros(dim))
        
        # --- Subspace-Gated FFN (SGF): 4 subspace gate vectors ---
        self.sgf_router = nn.Linear(dim, 4)
        self.sgf_gates = nn.Parameter(torch.randn(4, dim))
    
    def forward(self, x, token_ids=None, coord_stream=None, capture_attn=False):
        if coord_stream is None:
            coord_stream = torch.zeros_like(x)
        
        conv_out = self.fractal_conv(self.norm_conv(x))
        
        topo_bias = None
        if self.topology is not None and token_ids is not None:
            topo_bias = self.topology.get_topology_bias(token_ids)
        attn_out, _ = self.attention(self.norm_attn(x), topology_bias=topo_bias,
                                      return_attn=capture_attn)
        
        combined = torch.cat([conv_out, attn_out], dim=-1)
        gate = self.gate_conv(combined).sigmoid()
        
        coord_in_g = torch.sigmoid(self.coord_gate_in)
        x = x + gate * conv_out + (1 - gate) * attn_out + coord_stream * coord_in_g
        
        # SGF-FFN
        x_norm = self.norm_ffn(x)
        gate_s = F.silu(self.ffn.W_gate(x_norm))
        up = self.ffn.W_up(x_norm)
        swiglu = gate_s * up
        route = F.softmax(self.sgf_router(x_norm), dim=-1)
        sgf_gate = route @ self.sgf_gates
        swiglu = swiglu * torch.sigmoid(sgf_gate)
        x = x + self.ffn.W_down(swiglu)
        
        coord_out_g = torch.sigmoid(self.coord_gate_out)
        coord_stream = coord_stream + x * coord_out_g
        
        return x, coord_stream


class TrajectoryPredictor(nn.Module):
    """MLP 128→64→128: предсказывает delta-вектор следующей позиции в ℝ¹²⁸ для TrajLoss."""
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64), nn.GELU(),
            nn.Linear(64, dim),
        )
        self.apply(lambda m: m.weight.data.mul_(0.02) if hasattr(m, 'weight') and m.weight.dim() >= 2 else None)
    def forward(self, x):
        return self.net(x)


# ============================================================
# VRAM estimation
# ============================================================
if __name__ == "__main__":
    B, L, D = 8, 128, 128
    blk = HybridFractalBlock(D)
    x = torch.randn(B, L, D)
    
    y = blk(x)
    params = sum(p.numel() for p in blk.parameters())
    
    print(f"HybridFractalBlock: {params:,} params")
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    
    # Compare with attention-only
    from unified_transformer import TransformerBlock
    attn_blk = TransformerBlock(D, total_heads=32, d_ff=128)
    attn_params = sum(p.numel() for p in attn_blk.parameters())
    
    print(f"\nAttention-only block: {attn_params:,} params")
    print(f"Ratio: {params/attn_params:.1%}")
    print(f"VRAM conv: O(L·D/k) ≈ O({L}·{D}) per level")
    print(f"VRAM attn: O(L²) ≈ O({L*L}) per head")
