"""
EVA — Потенциальные поля: TensorPotentialField, WordValenceField,
SentenceContextField, SemanticRelevanceGate, KCACycle.

Все компоненты для bias-коррекции генерации через потенциалы.
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
import math


# ============================================================
# 1. TensorPotentialField — символьный тензор [V, V, V]
# ============================================================

class TensorPotentialField(nn.Module):
    """
    Тензор потенциалов P_i[j][k] — сила связи j→k, активируемая символом i.
    [V, V, V] — 4M params для V=160.

    Инициализация: P_i[j][k] = A_ij · A_jk (affinity chain).
    Обновление: накопление attention-весов с модуляцией quality.
    """

    def __init__(self, num_symbols=160, dtype=torch.float32):
        super().__init__()
        self.num_symbols = num_symbols
        self.P = nn.Parameter(torch.zeros(num_symbols, num_symbols, num_symbols, dtype=dtype))
        self.register_buffer('count', torch.zeros(num_symbols, dtype=torch.int32))

    def init_from_affinity(self, affinity):
        with torch.no_grad():
            if isinstance(affinity, np.ndarray):
                affinity = torch.from_numpy(affinity).to(self.P.device)
            self.P[:] = affinity.unsqueeze(-1) * affinity.unsqueeze(0)

    def update(self, symbol_idx, attn_weights, lr=0.01):
        """
        symbol_idx: [B] — индекс активирующего символа
        attn_weights: [B, H, S, S] — attention weights
        """
        with torch.no_grad():
            attn = attn_weights.mean(dim=1)  # [B, S, S]
            B, S, _ = attn.shape
            for b in range(B):
                sym = symbol_idx[b].item()
                if sym < self.num_symbols:
                    self.P[sym, :S, :S] += lr * attn[b]
                    self.count[sym] += 1

    def update_with_reflection(self, symbol_idx, attn_weights, metrics, base_lr=0.01):
        """
        metrics: dict с 'confidence', 'curvature', 'contradictions'
        Качество модулирует lr: качественный паттерн → быстрее учится
        """
        conf = metrics.get('confidence', 0.5)
        curv = metrics.get('curvature', 1.0)
        contra = metrics.get('contradictions', 0)
        quality = conf / (1.0 + curv) / (1.0 + contra)
        self.update(symbol_idx, attn_weights, lr=base_lr * quality)

    def get_bias(self, symbol_idx, target_ids, normalize=True):
        """
        symbol_idx: int — контекстный символ
        target_ids: [L] — последовательность ID для усреднения

        Returns: bias_vector [V] для добавления к logits
        """
        P_s = self.P[symbol_idx]  # [V, V]
        if normalize and self.count[symbol_idx] > 0:
            P_s = P_s / (self.count[symbol_idx].float() + 1.0)
        valid = target_ids[target_ids < self.num_symbols]
        if len(valid) > 0:
            return P_s[valid].mean(dim=0)  # [V]
        return P_s.mean(dim=0)

    def forward(self, symbol_idx):
        return self.P[symbol_idx] / (self.count[symbol_idx].float().clamp(min=1).unsqueeze(-1))


# ============================================================
# 2. WordValenceField — словесная валентность
# ============================================================

class WordValenceField(nn.Module):
    """
    Отображение координаты слова → матрица валентности [V, V].
    Использует outer-product декомпозицию для O(V·d) вместо O(V²).

    f_val(x) = left_weights(x) outer right_weights(x)
    left/right: [V] векторы весов для переходов j→k
    """

    def __init__(self, coord_dim=128, hidden_dim=128, num_symbols=160):
        super().__init__()
        self.num_symbols = num_symbols
        self.coord_dim = coord_dim

        # Два MLP предсказывают левый и правый контекстные векторы
        self.left_net = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, num_symbols)
        )
        self.right_net = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, num_symbols)
        )
        self.valence_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, word_coord):
        """
        word_coord: [B, D] — центроид слова
        Returns: valence [B, V, V] — матрица валентности
        """
        left = self.left_net(word_coord)  # [B, V]
        right = self.right_net(word_coord)  # [B, V]
        valence = torch.bmm(left.unsqueeze(2), right.unsqueeze(1))  # [B, V, V]
        return valence * self.valence_scale

    def get_valence_bias(self, word_coord, target_ids):
        """
        word_coord: [D] — центроид слова
        target_ids: [L] — последовательность ID (префикс)

        Returns: bias_vector [V]
        """
        with torch.no_grad():
            val = self.forward(word_coord.unsqueeze(0))  # [1, V, V]
            valid = target_ids[target_ids < self.num_symbols]
            if len(valid) > 0:
                return val[0, valid].mean(dim=0)  # [V]
            return val[0].mean(dim=0)


# ============================================================
# 3. SentenceContextField — поле предложения
# ============================================================

class SentenceContextField(nn.Module):
    """
    RBF-поле предложения: центры (из attention) + рёбра (connection vectors).
    Phi_sentence(z) = sum(γ_k · exp(-||z-c_k||²/2σ²)) + sum(φ_e(z))
    """

    def __init__(self, coord_dim=128, num_centers=3, sigma=0.5):
        super().__init__()
        self.coord_dim = coord_dim
        self.num_centers = num_centers
        self.sigma = sigma
        self.center_gamma = nn.Parameter(torch.ones(num_centers))

    def compute_centers(self, attn_weights, coords):
        """
        attn_weights: [B, H, S, S]
        coords: [B, S, D]
        Returns: centers[B, K, D], weights[B, K]
        """
        B, H, S, _ = attn_weights.shape
        importance = attn_weights.mean(dim=1).sum(dim=-1)  # [B, S]
        _, topk_idx = torch.topk(importance, min(self.num_centers, S), dim=-1)
        centers = torch.gather(coords, 1, topk_idx.unsqueeze(-1).expand(-1, -1, self.coord_dim))
        gamma = self.center_gamma[:topk_idx.shape[1]]
        return centers, gamma

    def compute_edge_field(self, coords, word_centroids):
        """
        coords: [B, S, D]
        word_centroids: [B, N, D]
        Returns: edge_field [B, S]
        """
        B, S, D = coords.shape
        if word_centroids.shape[1] < 2:
            return torch.zeros(B, S, device=coords.device)

        mids = (word_centroids[:, :-1] + word_centroids[:, 1:]) / 2  # [B, N-1, D]
        field = torch.zeros(B, S, device=coords.device)
        for e in range(mids.shape[1]):
            mid = mids[:, e:e+1, :]  # [B, 1, D]
            dist = torch.cdist(coords, mid, p=2).squeeze(-1)  # [B, S]
            field += torch.exp(-dist**2 / (2 * (self.sigma/2)**2))
        return field

    def forward(self, coords, centers, gamma, edge_field=None):
        """
        coords: [B, S, D]
        centers: [B, K, D]
        gamma: [K]
        edge_field: [B, S] или None

        Returns: field_activation [B, S]
        """
        dists = torch.cdist(coords, centers, p=2)  # [B, S, K]
        phi_centers = torch.exp(-dists**2 / (2 * self.sigma**2))  # [B, S, K]
        field = (phi_centers * gamma.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        if edge_field is not None:
            field += edge_field
        return field


# ============================================================
# 4. Semantic Relevance Gate (SRG)
# ============================================================

class SemanticRelevanceGate(nn.Module):
    """
    Оценка качества генерации: сходство + энтропия + этика.
    SRG_conf = w_sim · cos(c_q, c_r) + w_ent · (1 - H(p)/H_max) + w_eth · eth_score
    """

    def __init__(self, w_sim=0.4, w_ent=0.3, w_eth=0.3):
        super().__init__()
        self.w_sim = w_sim
        self.w_ent = w_ent
        self.w_eth = w_eth

    def evaluate(self, c_query, c_response, logits, response_text=None):
        """
        c_query: [B, D] — центроид запроса
        c_response: [B, D] — центроид ответа
        logits: [B, V] — логиты последнего шага
        response_text: str — для этик-фильтра (опционально)

        Returns: float — confidence [0, 1]
        """
        sim = F.cosine_similarity(c_query, c_response, dim=-1).mean().item()
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log2(probs + 1e-10)).sum(dim=-1).mean().item()
        max_ent = math.log2(logits.shape[-1])
        entropy_score = 1.0 - entropy / max_ent
        eth_score = 1.0
        if response_text and hasattr(self, 'ethics_filter') and self.ethics_filter:
            eth_score = self.ethics_filter.evaluate(response_text)
        return max(0.0, min(1.0, self.w_sim * sim + self.w_ent * entropy_score + self.w_eth * eth_score))


# ============================================================
# 5. GradientFlowSolver — стохастический L-BFGS
# ============================================================

class GradientFlowSolver(nn.Module):
    """
    Ланжевеновская динамика: z_{t+1} = z_t - η·∇Φ(z_t) + √(2Dη)·ξ

    С автоматическим детектором осцилляции и адаптивным шагом.
    """

    def __init__(self, potential_fn=None, eta=0.05, D=0.01, max_steps=50, tol=1e-3):
        super().__init__()
        self.potential_fn = potential_fn
        self.eta = eta
        self.D = D
        self.max_steps = max_steps
        self.tol = tol
        self.prev_prev_z = None
        self.prev_z = None

    def set_potential(self, potential_fn):
        self.potential_fn = potential_fn

    def step(self, z, t):
        """Один шаг Эйлер-Маруяма."""
        z_in = z.detach() if z.requires_grad else z
        z_in.requires_grad_(True)
        if self.potential_fn is None:
            phi = z_in.norm(p=2, dim=-1)
        else:
            phi = self.potential_fn(z_in)
        grad = torch.autograd.grad(phi.sum(), z_in, create_graph=False)[0]

        # Обновление prev_z / prev_prev_z для детектора осцилляции
        if t >= 2 and self.prev_z is not None and self.prev_prev_z is not None:
            cos_grad = F.cosine_similarity(
                (z_in - self.prev_z).view(1, -1),
                (self.prev_z - self.prev_prev_z).view(1, -1)
            ).item()
            if cos_grad < -0.5:
                z_in = (z_in + self.prev_z) / 2

        noise = torch.randn_like(z_in) * math.sqrt(2 * self.D * self.eta)
        z_new = z_in - self.eta * grad + noise

        self.prev_prev_z = (self.prev_z.detach().clone() if self.prev_z is not None
                            else z_in.detach().clone())
        self.prev_z = z_in.detach().clone()
        return z_new.detach()

    def solve(self, z0, temperature=0.1):
        """Полный цикл из z0 к равновесию. Возвращает путь."""
        self.prev_z = None
        self.prev_prev_z = None
        z = z0.clone()
        trajectory = [z.cpu().numpy()]
        for t in range(self.max_steps):
            z = self.step(z, t)
            trajectory.append(z.detach().cpu().numpy())
            if t >= 2:
                grad_norm = torch.norm(self.prev_z - z).item()
                if grad_norm < self.tol:
                    break
        return trajectory, z

    def _curvature_penalty(self, z, prev, prev_prev):
        d1 = (z - prev).view(1, -1)
        d2 = (prev - prev_prev).view(1, -1)
        cos = F.cosine_similarity(d1, d2)
        return (1.0 - cos) * d1.squeeze(0)


# ============================================================
# 6. KCA-цикл — итеративная коррекция
# ============================================================

class KCACycle(nn.Module):
    """
    Коррекция латентного кода через:
    L_KCA = -λ₁·SRG + λ₂·KL(p||p_target) + λ₃·||c_out - c_target||²

    Градиентный спуск по z с экспоненциальным затуханием lr.
    """

    def __init__(self, srg, lambda_conf=1.0, lambda_kl=0.1, lambda_dist=0.5,
                 eta0=0.01, rho=0.85, max_iter=5):
        super().__init__()
        self.srg = srg
        self.lambda_conf = lambda_conf
        self.lambda_kl = lambda_kl
        self.lambda_dist = lambda_dist
        self.eta0 = eta0
        self.rho = rho
        self.max_iter = max_iter

    def optimize(self, z_init, c_query, logits_fn, c_target=None, p_target=None):
        """
        z_init: [D] — начальный латентный код
        c_query: [D] — центроид запроса
        logits_fn: callable(z) → (logits [V], c_out [D])
        c_target: [D] — целевой центроид (опционально)
        p_target: [V] — целевое распределение (опционально)

        Returns: z_optimized [D]
        """
        z = z_init.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=self.eta0)

        for t in range(self.max_iter):
            opt.param_groups[0]['lr'] = self.eta0 * (self.rho ** t)
            logits, c_out = logits_fn(z)

            # Differentiable SRG components
            sim = F.cosine_similarity(c_query.unsqueeze(0), c_out.unsqueeze(0), dim=-1)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log2(probs + 1e-10)).sum(dim=-1)
            max_ent = math.log2(logits.shape[-1])
            entropy_score = 1.0 - entropy / max_ent

            srg_score = self.srg.w_sim * sim + self.srg.w_ent * entropy_score

            loss = -self.lambda_conf * srg_score
            if p_target is not None:
                p = F.softmax(logits, dim=-1)
                loss += self.lambda_kl * F.kl_div(p.log(), p_target, reduction='sum')
            if c_target is not None:
                loss += self.lambda_dist * F.mse_loss(c_out, c_target)

            opt.zero_grad()
            loss.backward()
            opt.step()

        return z.detach()


# ============================================================
# 7. RecursiveTensorPotentialField — рекурсивная декомпозиция
# ============================================================

class RecursiveTensorPotentialField(nn.Module):
    """
    V(x) = TPF[quantize(x)] + Σ 0.5^depth × TPF[quantize(decomp_depth(x))]

    Каждая координата — точка в ℝ¹²⁸. Декомпозиция: x → [v₁..v_K].
    Каждый vₖ — снова точка → рекурсия. Цикл (vₖ ≈ x) → unfold в tensor product.
    Все quantize — batched cdist, ни одного Python item().
    """

    def __init__(self, num_symbols=160, coord_dim=128, max_depth=8, K=6):
        super().__init__()
        self.num_symbols = num_symbols
        self.coord_dim = coord_dim
        self.max_depth = max_depth
        self.K = K

        # Base TPF — тот же [V,V,V]
        self.base_tpf = TensorPotentialField(num_symbols)

        # Декомпозиция: x → K субвекторов
        self.decomp_proj = nn.Linear(coord_dim, K * coord_dim, bias=False)

        # Композиция (aux loss): K субвекторов → x'
        self.compose_proj = nn.Linear(K * coord_dim, coord_dim, bias=False)

        # Гейтинг: softmax по K, дифференцируем
        self.gate_net = nn.Sequential(
            nn.Linear(coord_dim, K), nn.Softmax(dim=-1),
        )

        # Symbol coordinates for quantization (set externally)
        self.register_buffer('sym_coords', torch.zeros(num_symbols, coord_dim))

        # Depth scale per level (learnable вектор)
        self.depth_scale = nn.Parameter(torch.ones(max_depth) * 0.5)
        self.max_cap = 4096  # макс общее число путей

    def set_symbol_coordinates(self, coords):
        self.sym_coords.copy_(coords[:, :self.coord_dim])

    def quantize(self, vectors):
        """vectors: [N, D] → ids: [N] — nearest symbol. Один cdist, никаких item()."""
        dists = torch.cdist(vectors, self.sym_coords, p=2)
        return dists.argmin(dim=-1)

    def decompose(self, vectors):
        """vectors: [N, D] → (sub: [N, K, D], gates: [N, K])"""
        N = vectors.shape[0]
        sub = self.decomp_proj(vectors).view(N, self.K, self.coord_dim)
        gates = self.gate_net(vectors)  # [N, K] — softmax
        return sub, gates

    def compose(self, sub):
        """sub: [N, K, D] → recon: [N, D] — for aux reconstruction loss"""
        N, K, D = sub.shape
        return self.compose_proj(sub.view(N, K * D))

    def init_from_affinity(self, affinity):
        self.base_tpf.init_from_affinity(affinity)

    def update(self, symbol_idx, attn_weights, lr=0.01):
        self.base_tpf.update(symbol_idx, attn_weights, lr)

    def update_with_reflection(self, symbol_idx, attn_weights, metrics, base_lr=0.01):
        self.base_tpf.update_with_reflection(symbol_idx, attn_weights, metrics, base_lr)

    def forward(self, symbol_idx):
        return self.base_tpf(symbol_idx)

    def get_bias(self, symbol_idx, target_ids, normalize=True):
        return self.base_tpf.get_bias(symbol_idx, target_ids, normalize)

    def recursive_bias(self, x, context_ids):
        """
        x: [D] — вектор-запрос
        context_ids: [L] — ID контекста для valid-фильтра
        Returns: bias [V]

        BFS по дереву декомпозиции, все quantize — batched.
        Cycle detection: ||sub - parent|| < 0.01 → geometric series unfold.
        Chunked P[syms] gather to cap memory at ~1 MB per level.
        """
        device = x.device
        valid = context_ids[context_ids < self.num_symbols]

        # ========= Level 0 =========
        sym0 = self.quantize(x.unsqueeze(0))  # [1]
        P0 = self.base_tpf.P[sym0]  # [1, V, V]
        bias = P0[:, valid].mean(dim=1).squeeze(0) if len(valid) > 0 else P0.mean(dim=1).squeeze(0)

        if self.max_depth == 0:
            return bias

        # Chunked gather: process P[syms] in batches to cap intermediate memory
        def _gather_biases(syms):
            batch = 256
            outs = []
            for i in range(0, len(syms), batch):
                c = syms[i:i+batch]
                g = self.base_tpf.P[c]  # [≤batch, V, V]
                if len(valid) > 0:
                    outs.append(g[:, valid].mean(dim=1))
                else:
                    outs.append(g.mean(dim=1))
            return torch.cat(outs, dim=0)  # [N, V]

        # ========= Level 1+ : batched BFS with cap + cycle detection =========
        all_vecs = []
        all_scales = []
        total_paths = 0

        frontier = x.unsqueeze(0)  # [1, D]
        frontier_scale = torch.ones(1, device=device)

        for depth in range(1, self.max_depth + 1):
            N = frontier.shape[0]
            if N == 0:
                break

            # Budget cap: stop if adding this level exceeds max_cap
            if total_paths + N > self.max_cap:
                remaining = self.max_cap - total_paths
                if remaining <= 0:
                    break
                idx = torch.randperm(N, device=device)[:remaining]
                frontier = frontier[idx]
                frontier_scale = frontier_scale[idx]
                N = remaining
            total_paths += N

            # Quantize frontier → bias
            syms = self.quantize(frontier)  # [N]
            level_biases = _gather_biases(syms)  # [N, V]
            all_vecs.append(level_biases)
            all_scales.append(frontier_scale)

            # Decompose to next level
            if depth < self.max_depth:
                subs, gates = self.decompose(frontier)  # [N, K, D], [N, K]
                Nk = N * self.K
                subs_flat = subs.reshape(Nk, self.coord_dim)  # [N*K, D]
                gates_flat = gates.reshape(Nk)  # [N*K]

                # Scale = parent_scale * depth_scale[depth-1] * gate
                parent_scales = frontier_scale.unsqueeze(-1).expand(-1, self.K).reshape(Nk)
                new_scales = parent_scales * self.depth_scale[depth-1] * gates_flat

                # ── Cycle detection ──
                parent_flat = frontier.unsqueeze(1).expand(-1, self.K, -1).reshape(Nk, self.coord_dim)
                diff_norm = (subs_flat - parent_flat).norm(dim=-1)  # [N*K]
                cycle_mask = diff_norm < 0.01

                if cycle_mask.any():
                    r = self.depth_scale[depth-1] * gates_flat[cycle_mask]
                    series_sum = (1.0 / (1.0 - r + 1e-10)).clamp(max=10.0)
                    cycle_idx = torch.where(cycle_mask)[0]
                    parent_idx = cycle_idx // self.K
                    cycle_bias = level_biases[parent_idx]
                    cycle_contrib = cycle_bias * (frontier_scale[parent_idx] * series_sum).unsqueeze(-1)
                    bias = bias + cycle_contrib.sum(dim=0)

                # Gate filter + exclude cycles
                keep = (gates_flat > 0.05) & (~cycle_mask)
                frontier = subs_flat[keep]
                frontier_scale = new_scales[keep]

        # Combine all BFS levels
        if all_vecs:
            all_bias = torch.stack(all_vecs)
            all_scale = torch.stack(all_scales).unsqueeze(-1)
            bias = bias + (all_bias * all_scale).sum(dim=0)

        return bias

    def composition_loss(self, vectors):
        """
        vectors: [B, L, D] — hidden states from transformer
        Returns: scalar MSE loss = ||compose(decompose(x)) - x||²
        + diversity loss = mean(cosine(v_i, v_j)) for i≠j
        + gate entropy bonus: -0.01 × H(gates)
        """
        B, L, D = vectors.shape
        flat = vectors.reshape(B * L, D)
        subs, gates = self.decompose(flat)  # [BL, K, D], [BL, K]
        recon = self.compose(subs)  # [BL, D]
        loss = F.mse_loss(recon, flat)

        # Diversity: mean cosine between subvectors for i≠j
        subs_norm = F.normalize(subs, dim=-1)  # [BL, K, D]
        cos_mat = torch.bmm(subs_norm, subs_norm.transpose(1, 2))  # [BL, K, K]
        K = cos_mat.shape[1]
        mask = 1.0 - torch.eye(K, device=cos_mat.device).unsqueeze(0)  # [1, K, K]
        diversity = (cos_mat * mask).sum(dim=(1, 2)) / (K * (K - 1))
        diversity = diversity.mean()

        # Gate entropy bonus: encourage sparsity
        entropy = -(gates * torch.log(gates + 1e-10)).sum(dim=-1).mean()
        return loss + 0.1 * diversity - 0.01 * entropy



