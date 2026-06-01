"""
Phase 1 model: 384-dim, 24 heads (4×6 groups), 12 layers, MultiScaleRoPE.
BPE vocab: 4101 (4096 + 5 boundary).
"""
import torch, torch.nn as nn, torch.nn.functional as F, math
from typing import List, Optional, Tuple, Dict

from .heads import (
    TrajectoryBoundaryPredictor, BoundaryValidator, BoundaryDetectionHead,
    ConceptHead, ContradictionHead, UncertaintyHead, MetaWeighter,
    WeightProjector, ResidualHead,
)
from .subspace_coords import WordWeightEncoder
from .potential_fields import AttractorField, WordValenceField, HierarchicalAdditiveField


# ─── Constants ───
D_MODEL = 384
N_HEADS = 24
HEAD_DIM = 16
N_GROUPS = 6
HEADS_PER_GROUP = 4
GROUP_DIM = 64
N_LAYERS = 12
VOCAB_SIZE = 4101
D_FF = 512
MAX_SEQ_LEN = 2048
THETA_MIN = 500.0
THETA_MAX = 200000.0


# ─── 1. MultiScaleRoPE ───

class MultiScaleRoPE(nn.Module):
    """Standard RoPE with logarithmic θ from 500 to 200000 across 384 dims."""

    def __init__(self, dim=D_MODEL, max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        self.max_seq_len = max_seq_len
        half = dim // 2
        k = torch.arange(half, dtype=torch.float32)
        theta = THETA_MIN * (THETA_MAX / THETA_MIN) ** (k / half)
        self.register_buffer('freqs', 1.0 / theta)

        pos = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(pos, self.freqs)
        self.register_buffer('cos', angles.cos())
        self.register_buffer('sin', angles.sin())

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        if L > self.cos.shape[0]:
            raise ValueError(f'RoPE: seq_len={L} > max_seq_len={self.cos.shape[0]}')
        half = D // 2
        x0, x1 = x[..., :half], x[..., half:]
        cos = self.cos[:L, :half].unsqueeze(0)
        sin = self.sin[:L, :half].unsqueeze(0)
        rotated = torch.cat([x0 * cos - x1 * sin, x1 * cos + x0 * sin], dim=-1)
        return rotated


# ─── 2. GroupedScaleAttention ───

class GroupedScaleAttention(nn.Module):
    """
    Fixed grouping: 6 groups × 4 heads = 24 heads.
    Each group has separate W_O projection (group dim → D_MODEL).
    Soft gating per layer: α_l ∈ ℝ⁶ controls group contributions.
    """

    # Group boundaries: [0..3, 4..7, 8..11, 12..15, 16..19, 20..23]
    # Groups: char(0), morph(1), word(2), phrase(3), sentence(4), discourse(5)
    GROUP_RANGES = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]

    # Per-layer alpha init: layers 0-1 focus on char/morph
    LAYER_GATE_INIT = {
        0:  [0.70, 0.20, 0.10, 0.00, 0.00, 0.00],
        1:  [0.60, 0.25, 0.15, 0.00, 0.00, 0.00],
        2:  [0.40, 0.30, 0.20, 0.10, 0.00, 0.00],
        3:  [0.30, 0.30, 0.25, 0.15, 0.00, 0.00],
        4:  [0.10, 0.30, 0.30, 0.20, 0.10, 0.00],
        5:  [0.05, 0.25, 0.30, 0.25, 0.15, 0.00],
        6:  [0.00, 0.15, 0.25, 0.30, 0.25, 0.05],
        7:  [0.00, 0.10, 0.20, 0.30, 0.30, 0.10],
        8:  [0.00, 0.00, 0.10, 0.25, 0.40, 0.25],
        9:  [0.00, 0.00, 0.05, 0.20, 0.40, 0.35],
        10: [0.00, 0.00, 0.00, 0.10, 0.35, 0.55],
        11: [0.00, 0.00, 0.00, 0.05, 0.30, 0.65],
    }

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, head_dim=HEAD_DIM, layer_idx=0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx

        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)

        # Separate W_O per group: 64 → d_model
        self.w_os = nn.ModuleList([
            nn.Linear(HEADS_PER_GROUP * head_dim, d_model, bias=False)
            for _ in range(N_GROUPS)
        ])

        # Layer-specific soft gating
        init_gate = self.LAYER_GATE_INIT.get(layer_idx, [1.0/6]*N_GROUPS)
        self.gate = nn.Parameter(torch.tensor(init_gate, dtype=torch.float32))

        self.scale = head_dim ** -0.5

    def forward(self, x: torch.Tensor, rope: MultiScaleRoPE,
                group_biases: Optional[List[torch.Tensor]] = None,
                capture_attn=False) -> torch.Tensor:
        B, L, D = x.shape
        Dh = self.head_dim

        # Apply RoPE before splitting into heads
        q = rope.apply(self.to_q(x))
        k = rope.apply(self.to_k(x))
        v = self.to_v(x)

        q = q.view(B, L, self.n_heads, Dh)
        k = k.view(B, L, self.n_heads, Dh)
        v = v.view(B, L, self.n_heads, Dh)

        causal = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
        alphas = torch.softmax(self.gate, dim=-1)

        stored_attn = []
        outputs = []
        for g, (start, end) in enumerate(self.GROUP_RANGES):
            qg = q[:, :, start:end]  # [B, L, 4, 16]
            kg = k[:, :, start:end]
            vg = v[:, :, start:end]

            # Per-head attention within group: each head computes independently
            attn_out = []
            for h in range(start, end):
                qh = q[:, :, h:h+1]  # [B, L, 1, Dh]
                kh = k[:, :, h:h+1]
                vh = v[:, :, h:h+1]

                scores = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale  # [B, L, 1, L]
                scores = scores.squeeze(2) + causal  # [B, L, L]

                if group_biases is not None and g < len(group_biases) and group_biases[g] is not None:
                    scores = scores + group_biases[g]

                attn_w = torch.softmax(scores, dim=-1)  # [B, L, L]
                if capture_attn:
                    stored_attn.append(attn_w)
                head_out = torch.bmm(attn_w, vh.squeeze(2))  # [B, L, Dh]
                attn_out.append(head_out)

            # Concatenate heads in group
            group_out = torch.cat(attn_out, dim=-1)  # [B, L, 64]
            g_out = self.w_os[g](group_out)  # [B, L, 384]
            outputs.append(g_out * alphas[g])

        if capture_attn:
            self._captured_attn = stored_attn
        return sum(outputs)


# ─── 3. RMSNorm ───

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ─── 4. SwiGLU FFN ───

class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


# ─── 5. TransformerBlock V2 ───

class TransformerBlockV2(nn.Module):
    """Pre-norm, GroupedScaleAttention, SwiGLU + 3 residual streams."""

    def __init__(self, d_model=D_MODEL, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = GroupedScaleAttention(d_model, layer_idx=layer_idx)
        self.ffn = SwiGLUFFN(d_model, D_FF)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, rope: MultiScaleRoPE,
                residual1: torch.Tensor, residual2: torch.Tensor,
                residual3: torch.Tensor, capture_attn=False
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, L, D] — main stream
        residual1: [B, L, D] — char/morph stream
        residual2: [B, L, D] — word/phrase stream
        residual3: [B, L, D] — sentence/discourse stream
        Returns: (x_new, res1_new, res2_new, res3_new)
        """
        attn_out = self.attn(self.norm1(x + residual1 + residual2 + residual3), rope,
                             capture_attn=capture_attn)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))

        # Per-layer residual update (α=0.3 for target streams)
        if self.layer_idx < 4:
            residual1 = residual1 + attn_out * 0.3
            residual2 = residual2 + attn_out * 0.1
            residual3 = residual3 + attn_out * 0.05
        elif self.layer_idx < 8:
            residual1 = residual1 + attn_out * 0.1
            residual2 = residual2 + attn_out * 0.3
            residual3 = residual3 + attn_out * 0.15
        else:
            residual1 = residual1 + attn_out * 0.05
            residual2 = residual2 + attn_out * 0.15
            residual3 = residual3 + attn_out * 0.3

        return x, residual1, residual2, residual3


# ─── 6. BPE Embedding ───

class BPEEmbedding(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=D_MODEL):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(token_ids, self.weight)

    def set_coordinates(self, coords: torch.Tensor):
        with torch.no_grad():
            self.weight.copy_(coords)


# ─── 7. BPE Decoder ───

class BPEDecoder(nn.Module):
    def __init__(self, d_model=D_MODEL, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) / self.temperature.clamp(min=0.1, max=10.0)


# ─── 8. UnifiedMultidimensionalTransformerV2 ───

class UnifiedMultidimensionalTransformerV2(nn.Module):
    """
    384-dim, 24 heads, 12 layers, 6 scale groups, BPE vocab.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, d_model=D_MODEL,
                 n_layers=N_LAYERS, d_ff=D_FF):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.embed = BPEEmbedding(vocab_size, d_model)
        self.rope = MultiScaleRoPE(d_model)
        self.decoder = BPEDecoder(d_model, vocab_size)

        self.layers = nn.ModuleList([
            TransformerBlockV2(d_model, layer_idx=i)
            for i in range(n_layers)
        ])
        self.norm_final = RMSNorm(d_model)

        # Heads
        self.word_weight = WordWeightEncoder(d_model)
        self.attractor_field = AttractorField(coord_dim=d_model, max_attractors=vocab_size)
        self.word_valence = WordValenceField(d_model, hidden_dim=128, num_symbols=vocab_size)

        # HierarchicalAdditiveField — иерархическое аддитивное хранение
        self.haf = HierarchicalAdditiveField(coord_dim=d_model)

        self.boundary_predictor = TrajectoryBoundaryPredictor(d_model)
        self.boundary_validator = BoundaryValidator(d_model)
        self.boundary_detection = BoundaryDetectionHead(d_model)
        self.concept_head = ConceptHead(d_model)
        self.contra_head = ContradictionHead(d_model)
        self.uncertainty_head = UncertaintyHead(d_model)
        self.residual_head = ResidualHead(d_model)
        self.meta_weighter = MetaWeighter(d_model)
        self.weight_projector = WeightProjector(d_model)

        self._cached_attention = None

    def set_symbol_coordinates(self, coords: torch.Tensor):
        self.embed.set_coordinates(coords)

    def forward(self, token_ids, return_scores=False, return_weights=False,
                capture_attn=False, return_heads=False, return_latent=False,
                use_weight=False, update_attractors=False):
        B, L = token_ids.shape

        x = self.embed(token_ids)

        residual1 = torch.zeros_like(x)
        residual2 = torch.zeros_like(x)
        residual3 = torch.zeros_like(x)

        for i, layer in enumerate(self.layers):
            x, residual1, residual2, residual3 = layer(
                x, self.rope, residual1, residual2, residual3,
                capture_attn=(capture_attn and i == self.n_layers - 1))

        # Copy captured attention from last layer
        if capture_attn:
            self._cached_attention = self.layers[-1].attn._captured_attn
        else:
            self._cached_attention = None

        h = self.norm_final(x)

        # Attractor field Hebbian update
        if update_attractors and self.training:
            with torch.no_grad():
                for b in range(B):
                    for pos in range(1, L):
                        z_prev = h[b, pos-1]
                        z_curr = h[b, pos]
                        self.attractor_field.hebbian_update(z_prev, z_curr)

        # HAF hierarchical storage
        if update_attractors and self.training and B > 0:
            with torch.no_grad():
                z_pooled = h.mean(dim=1)  # [B, D] — per-batch mean
                for b in range(min(B, 4)):  # max 4 samples per step
                    self.haf.store_hierarchical(z_pooled[b], depth=2)

        scores = None
        if return_scores:
            scores = self.decoder(h)

        head_out = {}
        if return_heads:
            head_out['concept'] = self.concept_head(h)
            head_out['contradiction'] = self.contra_head(h)
            head_out['uncertainty'] = self.uncertainty_head(h)
            head_out['boundary_detect'] = self.boundary_detection(h)

            end, nxt, conn = self.boundary_predictor(h)
            head_out['boundary_end'] = end
            head_out['boundary_next'] = nxt
            head_out['boundary_conn'] = conn
            head_out['boundary_valid'] = self.boundary_validator(h, h)
            head_out['meta_weights'] = self.meta_weighter(h.mean(dim=1))

            # Attractor field diagnostics
            if self.attractor_field.n_attractors > 0:
                head_out['attractor_potential'] = self.attractor_field.potential(
                    h.reshape(-1, self.d_model)).reshape(B, L)
                head_out['attractor_n_attractors'] = self.attractor_field.n_attractors
            else:
                head_out['attractor_potential'] = torch.zeros(B, L, device=h.device)
                head_out['attractor_n_attractors'] = 0

            # HAF diagnostics (no_grad: only for logging, not for gradients)
            head_out['haf_n_attractors'] = self.haf.attractors.n_attractors
            with torch.no_grad():
                if self.training:
                    z_sample = h[0, L // 2]
                    parts, info = self.haf.decompose(z_sample, noise_dropout=0.1)
                    head_out['haf_K'] = info['K']
                    head_out['haf_residual'] = info['final_residual'].norm().item()
                else:
                    head_out['haf_K'] = 0
                    head_out['haf_residual'] = 0.0

            # HAF-based concept/contradiction (replaces MLP heuristics)
            haf_att = self.haf.attractors.n_attractors
            if haf_att > 0:
                z_embed = self.embed(token_ids)
                potential = self.haf.attractors.potential(
                    z_embed.reshape(-1, self.d_model)).reshape(B, L)
                # density ∈ [0, 1]: sigmoid over potential centered at 1.0
                head_out['concept'] = torch.sigmoid(potential - 1.0)

                z_mean = h.mean(dim=1)  # [B, D]
                _, info = self.haf.decompose(z_mean[0], noise_dropout=0.0)
                res_norm = info['final_residual'].norm()
                # residual large → contradiction high
                contra = torch.sigmoid(res_norm * 3.0 - 1.5)
                head_out['contradiction'] = contra.expand(B, L)

            # Residual head
            z_curr = self.embed(token_ids)
            z_pad = torch.zeros(B, 1, self.d_model, device=token_ids.device)
            z_prev = torch.cat([z_pad, z_curr[:, :-1]], dim=1)
            delta_pred, res_err = self.residual_head(h, z_prev, z_curr)
            head_out['delta_pred'] = delta_pred
            head_out['residual_error'] = res_err

        # Word weight — uses boundary_logits from head_out if available
        weights, w_shift, boundaries = None, None, None
        if return_weights or return_heads:
            bd_logits = head_out.get('boundary_detect')
            try:
                weights, w_shift, boundaries = self.word_weight(h, boundary_logits=bd_logits)
            except Exception:
                weights, w_shift = torch.zeros_like(h[..., 0]), torch.zeros_like(h)

        if return_latent:
            return h, scores, weights, head_out
        if return_heads:
            return h, scores, weights, head_out
        if return_weights:
            return h, scores, weights
        return h, scores

    def generate_text(self, prompt_ids, cv, max_new=128, temperature=0.8,
                      use_attractors=False, use_haf=False, **kwargs):
        """Generation compatible with CharacterVocab/BPEVocab interface.
        
        use_attractors=True: AttractorField.nxt_direction()
        use_haf=True: HAF hierarchical nxt_direction (decompose → per-component attractors → combine)
        default: boundary_predictor
        """
        device = next(self.parameters()).device
        ids = list(prompt_ids)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for _ in range(max_new):
                inp = torch.tensor([ids], dtype=torch.long, device=device)
                h, _, _, heads_out = self.forward(inp, return_heads=True, capture_attn=True)

                z_curr = h[0, -1]
                if use_haf and self.haf.attractors.n_attractors > 0:
                    nxt_dir = self.haf.nxt_direction(z_curr.unsqueeze(0))[0]
                    z_pred = z_curr + nxt_dir
                elif use_attractors and self.attractor_field.n_attractors > 0:
                    nxt_dir = self.attractor_field.nxt_direction(z_curr.unsqueeze(0))[0]
                    z_pred = z_curr + nxt_dir
                else:
                    end, nxt, conn = self.boundary_predictor(h[:, -1:])
                    z_pred = z_curr + nxt[0, 0]

                logits_know = self.decoder(z_pred.unsqueeze(0).unsqueeze(0))[0, 0]

                # Concept/contra weighting
                context = h.mean(dim=1)
                meta_w = self.meta_weighter(context)[0]
                concept_score = heads_out['concept'][0, -1].item()
                contra_score = heads_out['contradiction'][0, -1].item()

                sym_coords = self.embed.weight
                dists = -torch.cdist(z_pred.unsqueeze(0), sym_coords, p=2).squeeze(0)
                logits_conc = dists * (1.0 + concept_score)
                logits_contr = dists * (1.0 - contra_score * 0.5)

                w = meta_w
                final = (w[0] * logits_know + w[1] * logits_conc + w[2] * logits_contr) / temperature

                # Mask special tokens
                final[:4] = -float('inf')
                if hasattr(cv, 'GAP_FILLER_IDX') and cv.GAP_FILLER_IDX < len(final):
                    final[cv.GAP_FILLER_IDX] = -float('inf')
                for special_idx in [157, 158, 159, 160]:
                    if special_idx < len(final):
                        final[special_idx] = -float('inf')

                # Repetition penalty
                freq = set(ids)
                for t in freq:
                    if t < len(final):
                        final[t] -= 1.0

                # Top-20 sampling
                sl, si = final.sort(descending=True)
                v, idx = sl[:20], si[:20]
                p = F.softmax(v, dim=-1)
                nt = idx[torch.multinomial(p, 1)].item()
                ids.append(nt)

                eos_idx = cv.EOS_IDX if hasattr(cv, 'EOS_IDX') else 3
                if nt == eos_idx:
                    break

        self.train(was_training)
        return cv.decode(ids), {}

    def summary(self) -> str:
        params = sum(p.numel() for p in self.parameters())
        return (f"UnifiedTransformerV2(dim={self.d_model}, "
                f"layers={self.n_layers}, heads={N_HEADS}, "
                f"groups={N_GROUPS}, params={params:,})")
