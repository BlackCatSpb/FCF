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
                           padding=((k_L-1)*dil_L//2, (k_D-1)*dil_D//2))
            self.levels.append(conv)
            total_ch += ch_per_level
        
        self.proj = nn.Linear(total_ch, out_dim) if total_ch != out_dim else nn.Identity()
    
    def forward(self, x):
        # x: [B, L, D] → reshape for Conv2D
        B, L, D = x.shape
        x2d = x.unsqueeze(1)  # [B, 1, L, D]
        
        level_outs = []
        for conv in self.levels:
            out = conv(x2d)  # [B, C_l, L, D]
            # Pool over D dimension: mean
            out = out.mean(dim=-1)  # [B, C_l, L]
            level_outs.append(out.transpose(1, 2))  # [B, L, C_l]
        
        combined = torch.cat(level_outs, dim=-1)  # [B, L, total_ch]
        
        if isinstance(self.proj, nn.Linear):
            return self.proj(combined)
        return combined


class HybridFractalBlock(nn.Module):
    """
    Гибридный блок: FractalConv2D + SparseAttention + StaticTopology + Gate + FFN.
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
    
    def forward(self, x, token_ids=None):
        # 1. FractalConv
        conv_out = self.fractal_conv(self.norm_conv(x))
        
        # 2. Attention + Topology bias
        attn_out, _ = self.attention(self.norm_attn(x))
        
        # 3. Gate merge
        combined = torch.cat([conv_out, attn_out], dim=-1)
        gate = self.gate_conv(combined).sigmoid()
        
        x = x + gate * conv_out + (1 - gate) * attn_out
        
        # 4. FFN
        x = x + self.ffn(self.norm_ffn(x))
        
        return x


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
