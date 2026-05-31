"""
EVA — UnifiedMultidimensionalTransformer v3.

6 слоёв, 128-dim координатное пространство.
Shared encoder + 6 heads: TrajectoryBoundaryPredictor, BoundaryValidator,
ConceptHead, ContradictionHead, UncertaintyHead, MetaWeighter.

Генерация: decoder.linear(z_pred) + 3 источника через MetaWeighter [know, conc, contr].
Обучение: multi-task loss (trajectory_MSE + BCE heads + aux CE + composition).
"""

import torch, torch.nn as nn, torch.nn.functional as F, math
import numpy as np
from typing import List, Tuple, Optional, Dict

from .heads import WeightProjector, DistillationHead, TeacherAdapter, ResidualHead


# ============================================================
# RoPE — Rotary Position Embeddings
# ============================================================

class RoPE(nn.Module):
    """Rotary Position Embedding — кодирует позицию поворотом вектора."""
    
    def __init__(self, dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        positions = torch.arange(L, device=x.device).float()
        freqs = self.freqs.to(x.device)
        angles = torch.outer(positions, freqs)  # [L, D/2]
        
        cos = angles.cos().unsqueeze(0).unsqueeze(2)  # [1, L, 1, D/2]
        sin = angles.sin().unsqueeze(0).unsqueeze(2)
        
        x_reshaped = x.view(B, L, -1, 2)  # [B, L, D/2, 2]
        x0, x1 = x_reshaped[..., 0], x_reshaped[..., 1]
        
        rotated = torch.stack([
            x0 * cos.squeeze(2) - x1 * sin.squeeze(2),
            x1 * cos.squeeze(2) + x0 * sin.squeeze(2),
        ], dim=-1)
        
        return rotated.view(B, L, D)


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA style)."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ============================================================
# 1. CoordinateEmbedding — символ → ℝ⁶⁴ (без lookup)
# ============================================================

class CoordinateEmbedding(nn.Module):
    """Координатный эмбеддинг. Позиции из TopologicalField MDS."""
    
    def __init__(self, vocab_size: int = 157, coord_dim: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.coord_dim = coord_dim
        self.register_buffer("coordinates", torch.randn(vocab_size, coord_dim) * 0.02)
        self.scale = nn.Parameter(torch.ones(1))
    
    def set_coordinates(self, coords: torch.Tensor):
        assert coords.shape[0] == self.vocab_size
        self.coordinates.copy_(coords[:, :self.coord_dim])
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        ids = token_ids.clamp(0, self.vocab_size - 1)
        return self.coordinates[ids] * self.scale


# ============================================================
# 2. CoordinateDecoder — ℝ⁶⁴ → 157 символов
# ============================================================

class CoordinateDecoder(nn.Module):
    """Subspace Hierarchical Softmax (SubHSM) + nearest neighbor."""
    
    def __init__(self, coord_embedding: CoordinateEmbedding):
        super().__init__()
        self.embed = coord_embedding
        self.vocab_size = coord_embedding.vocab_size
        self.coord_dim = coord_embedding.coord_dim
        
        self.linear = nn.Linear(self.coord_dim, self.vocab_size)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.nn_weight = nn.Parameter(torch.tensor(0.1))
        
        # --- SubHSM: 4 группы по ~40 токенов ---
        n_groups = 4
        group_size = (self.vocab_size + n_groups - 1) // n_groups
        self.group_size = group_size
        
        # Register buffer: group_id and local_id for each vocab token
        group_ids = torch.zeros(self.vocab_size, dtype=torch.long)
        local_ids = torch.zeros(self.vocab_size, dtype=torch.long)
        for i in range(self.vocab_size):
            group_ids[i] = i // group_size
            local_ids[i] = i % group_size
        self.register_buffer('group_ids', group_ids)
        self.register_buffer('local_ids', local_ids)
        
        self.group_classifier = nn.Linear(self.coord_dim, n_groups)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Full logits
        logits = self.linear(x) / self.temperature.clamp(min=0.1, max=5.0)
        
        # Nearest neighbor branch
        coords = self.embed.coordinates
        diffs = x.unsqueeze(2) - coords.unsqueeze(0).unsqueeze(0)
        dists = torch.norm(diffs, dim=-1)
        nn_logits = -dists * self.nn_weight
        
        full_logits = logits + nn_logits
        
        # SubHSM bias: each token's logits boosted by its group affinity
        group_probs = F.softmax(self.group_classifier(x), dim=-1)
        gs = self.group_size
        for g in range(4):
            start = g * gs
            end = min(start + gs, self.vocab_size)
            full_logits[:, :, start:end] += group_probs[:, :, g:g+1]
        
        return full_logits
    
    def decode_to_ids(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        group_logits = self.group_classifier(x)
        best_group = group_logits.argmax(dim=-1)
        gs = self.group_size
        start = best_group * gs
        offsets = torch.arange(gs, device=x.device).unsqueeze(0).unsqueeze(0)
        idx = start.unsqueeze(-1) + offsets
        idx = idx.clamp(max=self.vocab_size - 1)
        gathered = logits.gather(-1, idx)
        local_best = gathered.argmax(dim=-1)
        return (start + local_best).clamp(max=self.vocab_size - 1)


# ============================================================
# 3. SwiGLU FFN
# ============================================================

class SwiGLUFFN(nn.Module):
    """Gated FFN: (SiLU(x·W_gate) ⊙ (x·W_up)) · W_down"""
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.W_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.W_up = nn.Linear(dim, hidden_dim, bias=False)
        self.W_down = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.W_gate(x))
        up = self.W_up(x)
        return self.W_down(gate * up)


# ============================================================
# 4. TransformerBlock (Pre-norm, FractalAttention, SwiGLU)
# ============================================================

class TransformerBlock(nn.Module):
    """Один блок: Pre-norm AdaptiveAttention + Pre-norm FFN."""
    
    def __init__(self, dim: int, max_levels: int = 8, total_heads: int = 32, d_ff: int = 128):
        super().__init__()
        self.dim = dim
        
        from .adaptive_fractal import AdaptiveFractalAttention
        self.attention = AdaptiveFractalAttention(
            d_model=dim, max_levels=max_levels, total_heads=total_heads,
        )
        
        self.norm_attn = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, d_ff)
    
    def forward(self, x: torch.Tensor):
        attn_out, _ = self.attention(self.norm_attn(x))
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ============================================================
# 5. UnifiedMultidimensionalTransformer
# ============================================================

class UnifiedMultidimensionalTransformer(nn.Module):
    """Координатный трансформер: 128-dim, адаптивные уровни, ~1.6M params."""
    
    def __init__(self, vocab_size=161, coord_dim=128, max_levels=8, total_heads=32,
                 num_layers=6, d_ff=128, max_seq_len=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.coord_dim = coord_dim
        self.num_layers = num_layers
        
        self.embed = CoordinateEmbedding(vocab_size, coord_dim)
        self.rope = RoPE(coord_dim, max_seq_len)
        self.decoder = CoordinateDecoder(self.embed)
        
        # Multi-subspace projections (agent Phase 3)
        from .subspace_coords import MultiSubspaceEmbedding
        self.subspace = MultiSubspaceEmbedding(vocab_size, coord_dim, sym_dim=32)
        
        from .fractal_conv import HybridFractalBlock
        from .static_topology import StaticTopologyLayer
        self.topology = StaticTopologyLayer(vocab_size, coord_dim)
        self.layers = nn.ModuleList([
            HybridFractalBlock(coord_dim, max_levels, total_heads, d_ff, self.topology)
            for _ in range(num_layers)
        ])
        
        self.norm_final = RMSNorm(coord_dim)
        
        from .subspace_coords import WordWeightEncoder
        self.word_weight = WordWeightEncoder(coord_dim)
        
        from .potential_fields import RecursiveTensorPotentialField, WordValenceField
        self.tensor_potential = RecursiveTensorPotentialField(num_symbols=vocab_size, coord_dim=coord_dim, max_depth=3, K=4)
        self.word_valence = WordValenceField(coord_dim, hidden_dim=128, num_symbols=vocab_size)
        
        from .heads import (
            TrajectoryBoundaryPredictor, BoundaryValidator,
            ConceptHead, ContradictionHead, UncertaintyHead, MetaWeighter,
            WeightProjector, BoundaryDetectionHead,
        )
        self.boundary_predictor = TrajectoryBoundaryPredictor(coord_dim)
        self.boundary_validator = BoundaryValidator(coord_dim)
        self.boundary_detection = BoundaryDetectionHead(coord_dim)
        self.concept_head = ConceptHead(coord_dim)
        self.contra_head = ContradictionHead(coord_dim)
        self.uncertainty_head = UncertaintyHead(coord_dim)
        self.residual_head = ResidualHead(coord_dim)
        self.meta_weighter = MetaWeighter(coord_dim)
        self.weight_projector = WeightProjector(coord_dim)
        
        self.last_attention = None  # для захвата attention weights (среднее по головам)
        self._weight_context = False  # включает/выключает weight token
    
    def set_symbol_coordinates(self, coords: torch.Tensor):
        self.embed.set_coordinates(coords)
        self.subspace.set_coordinates(coords)
        self.tensor_potential.set_symbol_coordinates(coords)
    
    def forward(self, token_ids, return_scores=False, return_weights=False, capture_attn=False,
                return_heads=False, return_latent=False, use_weight=False):
        B, L = token_ids.shape
        
        # ---- Weight token prepending ----
        use_weight = use_weight and self._weight_context
        if use_weight:
            w_token = self.weight_projector().unsqueeze(0).unsqueeze(0)  # [1, 1, D]
            w_token = w_token.expand(B, -1, -1)
            x_weight = w_token
            L_eff = L + 1
        else:
            x_weight = None
            L_eff = L
        
        # Base symbol embedding
        x = self.embed(token_ids)
        
        # Multi-subspace: enrich with word/connection/sentence projections
        sym = self.subspace(token_ids)  # [B, L, 32]
        x = x + F.pad(sym, (0, self.coord_dim - 32))
        
        # Prepend weight token if active
        if use_weight:
            x = torch.cat([x_weight, x], dim=1)
            # Extend token_ids with a dummy (0) for topology bias alignment
            token_ids_ext = torch.cat([torch.zeros(B, 1, dtype=token_ids.dtype, device=token_ids.device), token_ids], dim=1)
        else:
            token_ids_ext = token_ids
        
        x = self.rope(x)
        
        # Coordinate Residual Stream
        coord_stream = torch.zeros_like(x)
        
        self.last_attention = None  # сброс
        for layer in self.layers:
            x, coord_stream = layer(x, token_ids=token_ids_ext, coord_stream=coord_stream,
                                     capture_attn=capture_attn)
            if capture_attn and hasattr(layer.attention, 'last_attn_weights'):
                self.last_attention = layer.attention.last_attn_weights  # [B, H, L, L]
        
        h_all = self.norm_final(x)
        
        # Strip weight token if prepended
        if use_weight:
            h = h_all[:, 1:]
            w_token_out = h_all[:, 0]
        else:
            h = h_all
            w_token_out = None
        
        # Word weights (v2 returns word_vecs, word_weights, boundaries)
        try:
            word_vecs, word_weights, boundaries = self.word_weight(h)
            weights = word_weights
            w_shift = word_vecs.mean(dim=1, keepdim=True).expand(-1, h.shape[1], -1)
        except Exception:
            weights, w_shift = torch.zeros(h.shape[0], h.shape[1], device=h.device), torch.zeros_like(h)
        
        scores = None
        if return_scores:
            scores = self.decoder.forward(h + w_shift * 0.1)
        
        head_out = {}
        if return_heads:
            head_out['concept'] = self.concept_head(h)
            head_out['contradiction'] = self.contra_head(h)
            head_out['boundary_detect'] = self.boundary_detection(h)
            head_out['uncertainty'] = self.uncertainty_head(h)
            end, nxt, conn = self.boundary_predictor(h)
            head_out['boundary_end'] = end
            head_out['boundary_next'] = nxt
            head_out['boundary_conn'] = conn
            head_out['boundary_valid'] = self.boundary_validator(h, h)
            head_out['meta_weights'] = self.meta_weighter(h.mean(dim=1))
            if w_token_out is not None:
                head_out['weight_token'] = w_token_out
            # Teacher hidden states for distillation
            if (hasattr(self, '_teacher') and self._teacher is not None
                    and not token_ids is None):
                h_teacher = self._teacher.get_hidden(token_ids)
                head_out['teacher_hidden'] = h_teacher
            # Residual head: predict delta_z from context
            if token_ids is not None:
                B, L = token_ids.shape
                z_curr = self.embed(token_ids)  # [B, L, D]
                z_pad = torch.zeros(B, 1, self.coord_dim, device=token_ids.device)
                z_prev = torch.cat([z_pad, z_curr[:, :-1]], dim=1)  # [B, L, D]
                delta_pred, res_err = self.residual_head(h, z_prev, z_curr)
                head_out['delta_pred'] = delta_pred
                head_out['residual_error'] = res_err  # [B, L]
        
        if return_latent:
            return h, scores, weights, head_out
        
        if return_heads:
            return h, scores, weights, head_out
        if return_weights:
            return h, scores, weights
        return h, scores
    
    def distill_loss(self, h_eva: torch.Tensor, head_out: dict) -> torch.Tensor:
        """MSE между h_eva и h_teacher, спроецированным в ℝ¹²⁸."""
        if 'teacher_hidden' not in head_out or not hasattr(self, '_distill_head'):
            return torch.tensor(0.0, device=h_eva.device)
        return self._distill_head.loss(h_eva, head_out['teacher_hidden'])

    def residual_loss(self, head_out: dict) -> torch.Tensor:
        """MSE residual delta prediction loss."""
        if 'residual_error' not in head_out:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return head_out['residual_error'].mean()

    def attention_entropy(self) -> torch.Tensor:
        """Энтропия attention weights: низкая = острые паттерны."""
        if self.last_attention is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        attn = self.last_attention  # [B, H, L, L]
        p = attn.clamp(min=1e-8)
        entropy = -(p * p.log()).sum(dim=-1).mean()  # средняя по головам и позициям
        return entropy

    def temporal_smoothness_loss(self, h: torch.Tensor) -> torch.Tensor:
        """Topological smoothness: ||h_t - h_{t-1}||^2."""
        if h.shape[1] < 2:
            return torch.tensor(0.0, device=h.device)
        diffs = h[:, 1:] - h[:, :-1]
        return diffs.pow(2).mean()

    def head_consistency_loss(self, heads_out: dict) -> torch.Tensor:
        """concept ≈ 1 - contradiction: головам выгодно быть согласованными."""
        conc = heads_out.get('concept')
        contra = heads_out.get('contradiction')
        if conc is None or contra is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return (conc - (1.0 - contra)).pow(2).mean()

    def cross_gen_contrastive(self, prompt_ids: List[int], cv, temperature=0.8,
                               max_new=32) -> torch.Tensor:
        """
        Две генерации с одного промпта → align h на совпадающих позициях.
        Использует Gumbel-softmax для дифференцируемой аппроксимации.
        Возвращает MSE(h1, h2) на общем префиксе + diff-регуляризацию.
        """
        device = next(self.parameters()).device
        ids_a = list(prompt_ids)
        ids_b = list(prompt_ids)
        self.train()

        for _ in range(max_new):
            # Forward A
            inp_a = torch.tensor([ids_a], dtype=torch.long, device=device)
            h_a, _, _, _ = self.forward(inp_a, return_heads=True)
            logits_a = self.decoder.forward(h_a[:, -1:])[0, 0] / temperature
            gumbel_a = F.gumbel_softmax(logits_a, tau=0.5, hard=False)
            nt_a = gumbel_a.argmax(dim=-1).item()
            if nt_a < 4:
                nt_a = logits_a.argmax().item()
            ids_a.append(nt_a)
            if nt_a == 160:
                break

            # Forward B
            inp_b = torch.tensor([ids_b], dtype=torch.long, device=device)
            h_b, _, _, _ = self.forward(inp_b, return_heads=True)
            logits_b = self.decoder.forward(h_b[:, -1:])[0, 0] / temperature
            gumbel_b = F.gumbel_softmax(logits_b, tau=0.5, hard=False)
            nt_b = gumbel_b.argmax(dim=-1).item()
            if nt_b < 4:
                nt_b = logits_b.argmax().item()
            ids_b.append(nt_b)
            if nt_b == 160:
                break

        # Align h at common prefix
        prefix_len = min(len(ids_a), len(ids_b), len(prompt_ids) + 8)
        inp_a = torch.tensor([ids_a[:prefix_len]], dtype=torch.long, device=device)
        h_a, _, _, _ = self.forward(inp_a, return_heads=True)
        inp_b = torch.tensor([ids_b[:prefix_len]], dtype=torch.long, device=device)
        h_b, _, _, _ = self.forward(inp_b, return_heads=True)
        return (h_a - h_b).pow(2).mean()

    def self_distill_thought(self, heads_out: dict, h: torch.Tensor) -> torch.Tensor:
        """
        Thought loop self-distillation (differentiable):
        refined_z = differentiable thought_step(h_last, heads)
        loss = MSE(h_last, refined_z) — учимся выдавать refined сразу без цикла.
        """
        z = h[0, -1]
        concept = heads_out['concept'][0, -1]
        contra = heads_out['contradiction'][0, -1]
        uncertainty = heads_out['uncertainty'][0, -1]

        # Differentiable refinement (no .item() calls)
        contra_thresh = 0.3
        refine_lr = 0.1
        contra_refine_strength = 0.3
        refine_amount = (contra - contra_thresh).clamp(min=0) * contra_refine_strength * refine_lr
        z_refined = z - uncertainty * refine_amount

        return (z - z_refined).pow(2).mean()

    def kca_aux_loss(self, h: torch.Tensor, head_out: dict) -> torch.Tensor:
        """
        Differentiable KCA auxiliary loss.
        Maximises SRG(concept, -contra, entropy) на последней позиции h.
        L = -(w_sim * concept + w_ent * (1 - H/ H_max) - w_contra * contra)
        """
        conc = head_out.get('concept', torch.zeros_like(h[..., 0]))
        contra = head_out.get('contradiction', torch.zeros_like(h[..., 0]))
        z = h[0, -1]
        logits = self.decoder(z.unsqueeze(0).unsqueeze(0))[0, 0]
        probs = F.softmax(logits / 0.8, dim=-1)
        entropy = -(probs * torch.log2(probs + 1e-10)).sum(dim=-1)
        max_ent = math.log2(self.vocab_size)
        entropy_score = 1.0 - entropy / max_ent
        srg_like = 0.4 * conc[0, -1] + 0.3 * entropy_score - 0.3 * contra[0, -1]
        return -srg_like  # minimize = maximise SRG

    def srg_loss(self, h: torch.Tensor) -> torch.Tensor:
        """
        Differentiable SRG loss: 1.0 - SRG(query, response).

        Query = first 40% of sequence, Response = last 40% (avoid boundary).
        SRG = w_sim·cos(c_q, c_r) + w_ent·(1 - H/H_max)
        Returns a scalar loss in [0, 1] (minimize = maximise SRG).
        """
        B, L, D = h.shape
        split = int(L * 0.4)
        if split < 1 or L < 3:
            return torch.tensor(0.0, device=h.device)

        c_query = h[:, :split].mean(dim=1)
        c_response = h[:, -split:].mean(dim=1)
        cos_sim = F.cosine_similarity(c_query, c_response, dim=-1).mean()

        # Entropy from decoder logits at last position
        z_last = h[:, -1:]
        logits = self.decoder(z_last)
        probs = F.softmax(logits / 0.8, dim=-1)
        entropy = -(probs * torch.log2(probs + 1e-10)).sum(dim=-1).mean()
        max_ent = math.log2(self.vocab_size)
        entropy_score = 1.0 - entropy / max_ent

        srg = 0.4 * cos_sim + 0.3 * entropy_score
        return 1.0 - srg

    def update_tensor_potential(self, token_ids, lr=0.01, metrics=None):
        self.eval()
        with torch.no_grad():
            inp = token_ids if isinstance(token_ids, torch.Tensor) else torch.tensor([token_ids], device=next(self.parameters()).device)
            _ = self.forward(inp, capture_attn=True)
            if self.last_attention is not None:
                # last_attention: [B, L, L] (mean across heads) → [B, 1, L, L]
                attn_4d = self.last_attention.unsqueeze(1)
                # symbol_idx: [B] tensor of the last token in each sequence
                sym_idx = inp[:, -1] if inp.shape[1] > 1 else inp[:, 0]
                if metrics:
                    self.tensor_potential.update_with_reflection(sym_idx, attn_4d, metrics, base_lr=lr)
                else:
                    self.tensor_potential.update(sym_idx, attn_4d, lr=lr)
        self.train()
    
    def update_weight_token(self):
        """Обновить кэшированный weight token из текущих весов модели."""
        if hasattr(self, '_teacher') and self._teacher is not None:
            self._teacher.get_weight_token(self.weight_projector)
        else:
            self.weight_projector.update(self)
        return self.weight_projector._cached_token

    def set_weight_context(self, enabled: bool = True):
        """Включить/выключить weight token в forward."""
        self._weight_context = enabled

    def set_teacher(self, teacher_model: nn.Module, teacher_hidden_dim: int):
        """
        Установить teacher модель для дистилляции.

        После вызова:
        - weight token берётся из teacher (weights → ℝ¹²⁸)
        - forward возвращает h_teacher в head_out['teacher_hidden']
        - loss дистилляции: MSE(h_eva, proj(h_teacher))

        Args:
            teacher_model: любая PyTorch модель с forward(input_ids) → [B, L, D]
            teacher_hidden_dim: размерность скрытых состояний teacher
        """
        self._teacher = TeacherAdapter(teacher_model, teacher_hidden_dim)
        self._distill_head = DistillationHead(teacher_hidden_dim, self.coord_dim)
        self._teacher.to(next(self.parameters()).device)
        self._distill_head.to(next(self.parameters()).device)
        self.update_weight_token()
        self.set_weight_context(True)
        self._weight_context = True

    def summary(self) -> str:
        params = sum(p.numel() for p in self.parameters())
        moe_on = any(hasattr(l, 'ffn') and hasattr(l.ffn, 'n_experts') for l in self.layers)
        return f"UnifiedTransformer(dim={self.coord_dim}, layers={self.num_layers}, MoE={moe_on}, weight_ctx={self._weight_context}, params={params:,})"
    
    def forward_with_latent(self, token_ids, z_latent=None):
        """
        Forward с additional latent optimization (KCA).

        Если z_latent задан, он добавляется к h перед decoder.
        Возвращает реальные decoder logits от модифицированного h.
        Returns: (h, z_combined, logits, head_out)
        """
        h, _, _, heads_out = self.forward(token_ids, return_heads=True)
        if z_latent is not None:
            B, L, D = h.shape
            if z_latent.dim() == h.ndim:
                if z_latent.shape[1] == 1 and L > 1:
                    z_latent = z_latent.expand(B, L, D)
                z_combined = h + z_latent
            elif z_latent.dim() == 2:
                z_combined = h + z_latent.unsqueeze(1)
            else:
                z_combined = h + z_latent.unsqueeze(0).unsqueeze(0)
        else:
            z_combined = h
        logits = self.decoder(z_combined)
        return h, z_combined, logits, heads_out

    def enhanced_generate(self, prompt_ids, cv, max_new=128, temperature=0.8,
                          thought_loop=False, max_thought=3,
                          knn_retriever=None):
        """
        Генерация через навигацию по координатам (без teacher forcing).
        - z_pred = z_current + nxt (boundary predictor)
        - 3 источника через MetaWeighter [know, conc, contr]
        - Основной: decoder.linear(z_pred) — learned projection
        """
        device = next(self.parameters()).device
        ids = list(prompt_ids)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for _ in range(max_new):
                inp = torch.tensor([ids], dtype=torch.long, device=device)
                h, _, _, heads_out = self.forward(inp, return_heads=True, capture_attn=True)
                
                end, nxt, conn = self.boundary_predictor(h[:, -1:])
                z_curr = h[0, -1]
                z_pred = z_curr + nxt[0, 0]
                
                context = h.mean(dim=1)
                meta_w = self.meta_weighter(context)
                
                sym_coords = self.embed.coordinates
                if len(ids) > 1:
                    last_sym = ids[-1]
                    if last_sym < self.tensor_potential.num_symbols:
                        bias_tpf = self.tensor_potential.recursive_bias(
                            z_pred, torch.tensor(ids, device=device))
                    else:
                        bias_tpf = torch.zeros(self.vocab_size, device=device)
                else:
                    bias_tpf = torch.zeros(self.vocab_size, device=device)
                word_coord = h[0, :].mean(dim=0)
                bias_wvf = self.word_valence.get_valence_bias(
                    word_coord, torch.tensor(ids, device=device)).to(device)
                
                # ---- Source 1: learned decoder (primary) + biases ----
                logits_know = self.decoder(z_pred.unsqueeze(0).unsqueeze(0))[0, 0] + bias_tpf + bias_wvf

                # ---- kNN-LM bias (optional) ----
                if knn_retriever is not None:
                    knn_bias = knn_retriever.retrieve(z_pred, self.vocab_size)
                    logits_know = logits_know + knn_bias.to(logits_know.device)
                
                # ---- Source 2: concept navigation (geometric) ----
                concept_score = heads_out['concept'][0, -1].item()
                logits_concept = -torch.cdist(
                    z_pred.unsqueeze(0), sym_coords, p=2).squeeze(0)
                logits_concept = logits_concept * (1.0 + concept_score)
                
                # ---- Source 3: contra avoidance (geometric) ----
                contra_score = heads_out['contradiction'][0, -1].item()
                logits_contra = -torch.cdist(
                    z_pred.unsqueeze(0), sym_coords, p=2).squeeze(0)
                logits_contra = logits_contra * (1.0 - contra_score * 0.5)
                
                # ---- Weighted sum with uniform temperature ----
                w = meta_w[0]
                logits = (w[0] * logits_know + w[1] * logits_concept + w[2] * logits_contra) / temperature
                
                # ---- Mask special tokens ----
                logits[:4] = -float('inf')
                logits[cv.GAP_FILLER_IDX] = -float('inf')
                logits[cv.WORD_OPEN_IDX] = -float('inf')
                logits[cv.WORD_CLOSE_IDX] = -float('inf')
                logits[cv.SENT_OPEN_IDX] = -float('inf')
                
                # ---- Repetition penalty (logits-level) ----
                freq = set(ids)
                for t in freq:
                    logits[t] -= 1.0
                
                # ---- Sample from top-20 ----
                sl, si = logits.sort(descending=True)
                v = sl[:20]; idx = si[:20]; p = F.softmax(v, dim=-1)
                nt = idx[torch.multinomial(p, 1)].item()
                ids.append(nt)
                if nt == cv.SENT_CLOSE_IDX:
                    break
        self.train(was_training)
        return cv.decode(ids)
    
    def generate_text(self, prompt_ids, cv, max_new=128, temperature=0.8,
                       flow_solver=None, kca_cycle=None,
                       hypothesis_buffer=None, srg_module=None):
        """
        Полный pipeline генерации из Доработки.txt Секция 6.

        Интеграция:
        - BoundaryPredictor
        - GradientFlowSolver (опционально)
        - Potentials (TPF + WVF)
        - Concept/Contra heads
        - MetaWeighter (3 источника: know, conc, contr)
        - SRG + KCA (если low confidence)
        - H2K (сохранение гипотезы)

        Returns: (текст, метрики)
        """
        device = next(self.parameters()).device
        ids = list(prompt_ids)
        was_training = self.training
        self.eval()
        metrics = {'srg_final': 0.0, 'kca_used': False, 'tokens': 0, 'converged': False}
        h_last = None

        with torch.no_grad():
            for step in range(max_new):
                inp = torch.tensor([ids], dtype=torch.long, device=device)
                h, _, _, heads_out = self.forward(inp, return_heads=True, capture_attn=True)
                h_last = h

                context = h.mean(dim=1)
                meta_w = self.meta_weighter(context)[0]
                end, nxt, conn = self.boundary_predictor(h[:, -1:])

                z_current = h[0, -1]
                if flow_solver is not None:
                    with torch.enable_grad():
                        flow_solver.set_potential(lambda z: self.tensor_potential.recursive_bias(z, inp[0]))
                        for flow_t in range(10):
                            z_current = flow_solver.step(z_current, flow_t)
                    _, nxt, _ = self.boundary_predictor(z_current.unsqueeze(0).unsqueeze(0))

                z_pred = (z_current + nxt[0, 0]).unsqueeze(0)
                sym_coords = self.embed.coordinates
                if len(ids) > 1:
                    last_sym = ids[-1]
                    if last_sym < self.tensor_potential.num_symbols:
                        bias_tpf = self.tensor_potential.recursive_bias(z_pred[0], torch.tensor(ids, device=device))
                    else:
                        bias_tpf = torch.zeros(self.vocab_size, device=device)
                else:
                    bias_tpf = torch.zeros(self.vocab_size, device=device)
                bias_wvf = self.word_valence.get_valence_bias(z_pred[0], torch.tensor(ids, device=device)).to(device)

                logits_know = self.decoder(z_pred.unsqueeze(0))[0, 0] + bias_tpf + bias_wvf
                concept_score = heads_out['concept'][0, -1].item()
                contra_score = heads_out['contradiction'][0, -1].item()
                dists = -torch.cdist(z_pred, sym_coords, p=2).squeeze(0)
                logits_conc = dists * (1.0 + concept_score)
                logits_contr = dists * (1.0 - contra_score * 0.5)

                w = meta_w
                final = (w[0] * logits_know + w[1] * logits_conc + w[2] * logits_contr) / temperature

                # ---- Mask special tokens (PAD=0, UNK=1, BOS=2, EOS=3) ----
                final[:4] = -float('inf')
                final[cv.GAP_FILLER_IDX] = -float('inf')
                final[cv.WORD_OPEN_IDX] = -float('inf')
                final[cv.WORD_CLOSE_IDX] = -float('inf')
                final[cv.SENT_OPEN_IDX] = -float('inf')
                
                # ---- Repetition penalty (logits-level) ----
                freq = set(ids)
                for t in freq:
                    final[t] -= 1.0
                
                # ---- Sample from top-20 ----
                sl, si = final.sort(descending=True)
                v, idx = sl[:20], si[:20]
                p = F.softmax(v, dim=-1)
                nt = idx[torch.multinomial(p, 1)].item()
                ids.append(nt)
                if nt == cv.SENT_CLOSE_IDX:
                    metrics['converged'] = True
                    break

        # SRG evaluation (uses last h from generation loop — no double forward)
        if srg_module is not None and h_last is not None:
            c_query = h_last[0].mean(dim=0)
            c_response = h_last[0, -1]
            dummy_logits = torch.zeros(self.vocab_size, device=device)
            metrics['srg_final'] = srg_module.evaluate(
                c_query.unsqueeze(0), c_response.unsqueeze(0),
                dummy_logits.unsqueeze(0))

        # KCA fallback
        if kca_cycle is not None and metrics['srg_final'] < 0.5 and h_last is not None:
            with torch.enable_grad():
                def kca_logits_fn(z):
                    _, zc, lg, _ = self.forward_with_latent(
                        torch.tensor([ids], device=device), z.unsqueeze(0).unsqueeze(0))
                    return lg[0, -1], zc[0, -1]
                z_opt = kca_cycle.optimize(h_last[0, -1], c_query, kca_logits_fn)
            metrics['kca_used'] = True

        # H2K save (use h_last from generation, not a stale re-forward)
        if hypothesis_buffer is not None and h_last is not None:
            hyp = (h_last[0].cpu().numpy(), metrics['srg_final'], metrics['tokens'],
                   self.last_attention, None, ids)
            hypothesis_buffer.add(hyp)

        self.train(was_training)
        return cv.decode(ids), metrics

    # -------- intrinsic label extraction for self-supervised training --------
    
    def _intrinsic_contra_labels(self, h, token_ids):
        """
        Ground-truth for ContradictionHead:
        uncertainty = average pairwise distance in a local neighbourhood.
        High uncertainty = high contradiction.
        Returns: [B, L] values in [0,1].
        """
        B, L, D = h.shape
        with torch.no_grad():
            diffs = h.unsqueeze(2) - h.unsqueeze(1)
            pair_dists = torch.norm(diffs, dim=-1)
            uncertainty = pair_dists.mean(dim=-1)
            uncertainty = (uncertainty - uncertainty.min(dim=-1, keepdim=True)[0])
            uncertainty = uncertainty / (uncertainty.max(dim=-1, keepdim=True)[0] + 1e-8)
        return uncertainty
    
    def _intrinsic_concept_labels(self, h, token_ids):
        """
        Ground-truth for ConceptHead:
        concept_score = cluster density вокруг каждой точки.
        Высокая плотность = много близких соседей = концепт.
        Returns: [B, L] values in [0,1].
        """
        B, L, D = h.shape
        with torch.no_grad():
            diffs = h.unsqueeze(2) - h.unsqueeze(1)
            pair_dists = torch.norm(diffs, dim=-1)
            # density = exp(-dist) summed over neighbours
            density = torch.exp(-pair_dists).sum(dim=-1)
            density = (density - density.min(dim=-1, keepdim=True)[0])
            density = density / (density.max(dim=-1, keepdim=True)[0] + 1e-8)
        return density
