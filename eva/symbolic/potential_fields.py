"""
EVA — Потенциальные поля: TensorPotentialField, WordValenceField,
SentenceContextField, SemanticRelevanceGate, KCACycle.

Все компоненты для bias-коррекции генерации через потенциалы.
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
import math
from typing import Optional


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

class AttractorField(nn.Module):
    """
    Hebbian attractor field: область знаний = плотность пересечения траекторий.

    Аттрактор a = (μ_a, w_a, r_a):
      μ_a ∈ ℝ^D — центр (среднее всех точек, прошедших через аттрактор)
      w_a ∈ ℝ — счётчик (сколько треков прошло)
      r_a ∈ ℝ^D — рефрактерный вектор (направление выхода из аттрактора)

    Поле: P(z) = Σ_a w_a · exp(-||z - μ_a||² / 2σ²)
    Генерация: nxt(z) = η · (μ* - z) + (1-η) · r*
    """

    def __init__(self, coord_dim=128, sigma=0.5, eta=0.7,
                 max_attractors=10000, lr_center=0.01, lr_refract=0.05,
                 creation_threshold=0.1, decay=0.999):
        super().__init__()
        self.coord_dim = coord_dim
        self.sigma = sigma
        self.eta = eta
        self.max_attractors = max_attractors
        self.lr_center = lr_center
        self.lr_refract = lr_refract
        self.creation_threshold = creation_threshold
        self.decay = decay

        # Buffers — all growable
        self.register_buffer('centers', torch.zeros(max_attractors, coord_dim))
        self.register_buffer('counts', torch.zeros(max_attractors))
        self.register_buffer('refractory', torch.zeros(max_attractors, coord_dim))
        self.register_buffer('valid_mask', torch.zeros(max_attractors, dtype=torch.bool))

        self._n_attractors = 0

    @property
    def n_attractors(self):
        return self._n_attractors

    def add_attractor(self, z: torch.Tensor, r: Optional[torch.Tensor] = None):
        """Add a new attractor at position z."""
        idx = self._n_attractors
        if idx >= self.max_attractors:
            # Find least-used valid attractor and replace
            scores = self.counts + 1e-8
            idx = scores.argmin().item()
        self.centers[idx] = z.detach()
        self.counts[idx] = 1.0
        if r is not None:
            self.refractory[idx] = r.detach()
        else:
            self.refractory[idx] = 0.0
        self.valid_mask[idx] = True
        self._n_attractors = min(idx + 1, self._n_attractors + 1)

    def hebbian_update(self, z: torch.Tensor, z_next: Optional[torch.Tensor] = None):
        """
        Hebbian update: найти ближайший аттрактор, инкрементировать счётчик,
        сдвинуть центр (count-normalized), обновить рефрактерный вектор.

        z: [D] or [B, D] — точка на траектории
        z_next: [D] or [B, D] — следующая точка (для r)

        Returns: (dist_to_closest, closest_idx)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        B, D = z.shape

        valid = self.valid_mask[:self._n_attractors]
        if not valid.any():
            self.add_attractor(z[0])
            return torch.tensor(0.0, device=z.device), 0

        centers = self.centers[:self._n_attractors][valid]
        counts = self.counts[:self._n_attractors][valid]

        dists = torch.cdist(z, centers, p=2)
        min_dists, closest = dists.min(dim=-1)

        for b in range(B):
            z_i = z[b]
            z_n = z_next[b] if z_next is not None else None

            # Create new attractor if too far from any existing one
            if min_dists[b].item() > self.creation_threshold:
                self.add_attractor(z_i, r=(z_n - z_i) if z_n is not None else None)
                continue

            c = closest[b].item()
            idx = torch.where(valid)[0][c].item()
            n = self.counts[idx].item()

            # Hebbian: increment counter
            self.counts[idx] += 1.0

            # Count-normalized center update: lr decays with frequency
            lr_eff = self.lr_center / max(n, 1)
            self.centers[idx] += lr_eff * (z_i - self.centers[idx])

            # Update refractory vector (EMA of exit direction)
            if z_n is not None:
                direction = z_n - z_i
                self.refractory[idx] += self.lr_refract * (direction - self.refractory[idx])

        # Global decay once per forward, not per batch element
        n_valid = self._n_attractors
        if n_valid > 0:
            self.counts[:n_valid] *= self.decay
            # Prune dead attractors periodically
            if self._n_attractors > 100 and np.random.rand() < 0.01:
                dead = self.counts[:n_valid] < 0.01
                if dead.any() and dead.sum() < n_valid // 2:
                    self.valid_mask[:n_valid][dead] = False
                    self._n_attractors = self.valid_mask[:n_valid].sum().item()

        return min_dists, closest

    def potential(self, z: torch.Tensor) -> torch.Tensor:
        """P(z) = Σ w_a · exp(-||z - μ_a||² / 2σ²)"""
        valid = self.valid_mask[:self._n_attractors]
        if not valid.any():
            return torch.zeros(z.shape[0], device=z.device)

        centers = self.centers[:self._n_attractors][valid]
        counts = self.counts[:self._n_attractors][valid]

        dists = torch.cdist(z, centers, p=2)  # [B, N]
        return (counts.unsqueeze(0) * torch.exp(-dists ** 2 / (2 * self.sigma ** 2))).sum(dim=-1)

    def gradient(self, z: torch.Tensor) -> torch.Tensor:
        """∇P(z) — градиент потенциала для генерации."""
        valid = self.valid_mask[:self._n_attractors]
        if not valid.any():
            return torch.zeros_like(z)

        centers = self.centers[:self._n_attractors][valid]
        counts = self.counts[:self._n_attractors][valid]

        z_exp = z.unsqueeze(1)  # [B, 1, D]
        diffs = z_exp - centers.unsqueeze(0)  # [B, N, D]
        dists_sq = diffs.pow(2).sum(dim=-1)  # [B, N]
        weights = counts.unsqueeze(0) * torch.exp(-dists_sq / (2 * self.sigma ** 2))  # [B, N]

        grad = (weights.unsqueeze(-1) * diffs).sum(dim=1) / (self.sigma ** 2)  # [B, D]
        return -grad  # P gradient = direction to higher density

    def nxt_direction(self, z: torch.Tensor) -> torch.Tensor:
        """
        Генерация: nxt(z) = η · (μ* - z) / ||μ* - z|| + (1-η) · r*
        """
        valid = self.valid_mask[:self._n_attractors]
        if not valid.any():
            return torch.randn(z.shape[0], self.coord_dim, device=z.device) * 0.1

        centers = self.centers[:self._n_attractors][valid]
        refract = self.refractory[:self._n_attractors][valid]

        dists = torch.cdist(z, centers, p=2)
        closest = dists.argmin(dim=-1)

        B = z.shape[0]
        nxt = torch.zeros_like(z)
        for b in range(B):
            c = closest[b].item()
            idx = torch.where(valid)[0][c].item()

            to_center = centers[c] - z[b]
            to_center = to_center / (to_center.norm() + 1e-8)
            r_vec = refract[c]
            r_vec = r_vec / (r_vec.norm() + 1e-8) if r_vec.norm() > 0 else to_center

            nxt[b] = self.eta * to_center + (1 - self.eta) * r_vec

        return nxt

    def decay_old(self, threshold: float = 0.1):
        """Удалить аттракторы с низким счётчиком."""
        old = self.counts[:self._n_attractors] < threshold
        self.valid_mask[:self._n_attractors][old] = False
        self._n_attractors = self.valid_mask[:self._n_attractors].sum().item()


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

    def set_ethics_filter(self, ethics_filter):
        self.ethics_filter = ethics_filter

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
            eth_result = self.ethics_filter.evaluate(response_text)
            eth_score = eth_result[0] if isinstance(eth_result, (tuple, list)) else eth_result
        return max(0.0, min(1.0, self.w_sim * sim + self.w_ent * entropy_score + self.w_eth * eth_score))


# ============================================================
# 5. GradientFlowSolver — стохастический L-BFGS
# ============================================================

class GradientFlowSolver(nn.Module):
    """
    Ланжевеновская динамика: z_{t+1} = z_t - η·∇Φ(z_t) + √(2Dη)·ξ

    Из Доработки.txt Секция 2:
    - curvature_weight: штраф за резкие повороты траектории
    - Адаптивная остановка: сходимость + осцилляция + таймаут
    - Шум Ланжевена для исследования пространства
    """

    def __init__(self, potential_fn=None, contradiction_field=None,
                 curvature_weight=0.1, eta=0.05, D=0.01,
                 max_steps=50, tol=1e-5, timeout=10.0):
        super().__init__()
        self.potential_fn = potential_fn
        self.contradiction_field = contradiction_field
        self.curvature_weight = curvature_weight
        self.eta = eta
        self.D = D
        self.max_steps = max_steps
        self.tol = tol
        self.timeout = timeout
        self.prev_prev_z = None
        self.prev_z = None

    def set_potential(self, potential_fn):
        self.potential_fn = potential_fn

    def step(self, z, t):
        """Один шаг Эйлер-Маруяма с curvature penalty + oscillation detection."""
        import time
        z_in = z.detach() if z.requires_grad else z
        z_in.requires_grad_(True)

        if self.potential_fn is None:
            phi = z_in.norm(p=2, dim=-1)
        else:
            phi = self.potential_fn(z_in)
        grad = torch.autograd.grad(phi.sum(), z_in, create_graph=False)[0]

        if self.contradiction_field is not None:
            grad = grad + self.contradiction_field.gradient(z_in)

        # Curvature penalty (из Доработки.txt)
        if t >= 2 and self.prev_z is not None and self.prev_prev_z is not None:
            d1 = z_in - self.prev_z
            d2 = self.prev_z - self.prev_prev_z
            cos_curve = F.cosine_similarity(d1.view(1, -1), d2.view(1, -1))
            curvature = (1.0 - cos_curve) * d1
            grad = grad + self.curvature_weight * curvature

        # Oscillation detection
        if t >= 2 and self.prev_z is not None and self.prev_prev_z is not None:
            cos_osc = F.cosine_similarity(
                (z_in - self.prev_z).view(1, -1),
                (self.prev_z - self.prev_prev_z).view(1, -1)
            ).item()
            if cos_osc < -0.5:
                z_in = (z_in + self.prev_z) * 0.5

        # Langevin noise
        noise = torch.randn_like(z_in) * math.sqrt(2 * self.D * self.eta)
        z_new = z_in - self.eta * grad + noise

        self.prev_prev_z = (self.prev_z.detach().clone() if self.prev_z is not None
                            else z_in.detach().clone())
        self.prev_z = z_in.detach().clone()
        return z_new.detach()

    def solve(self, z0):
        """Полный цикл из z0 к равновесию. Возвращает (путь, финальный z)."""
        import time
        self.prev_z = None
        self.prev_prev_z = None
        z = z0.clone()
        trajectory = [z.cpu().numpy()]
        start_time = time.time()
        for t in range(self.max_steps):
            z_new = self.step(z, t)
            trajectory.append(z_new.detach().cpu().numpy())

            # Convergence check
            if torch.norm(z_new - z).item() < self.tol:
                return trajectory, z_new

            # Timeout
            if time.time() - start_time > self.timeout:
                return trajectory, z_new

            z = z_new
        return trajectory, z


# ============================================================
# 6. KCA-цикл — итеративная коррекция
# ============================================================

class KCACycle(nn.Module):
    """
    Коррекция латентного кода через:
    L_KCA = -λ₁·SRG + λ₂·KL(p||p_target) + λ₃·||c_out - c_target||²

    Адаптивная остановка (из Доработки.txt Секция 1.2):
    - ‖z_{t+1} − z_t‖ < ε — сходимость по норме
    - |SRG(t) − SRG(t-1)| < δ × 3 шага — стагнация
    - cos(grad_t, grad_{t-1}) < −0.5 — осцилляция
    - wall_time < timeout — аппаратная защита
    """

    def __init__(self, srg, lambda_conf=1.0, lambda_kl=0.1, lambda_dist=0.5,
                 eta0=0.01, rho=0.85, epsilon=1e-5, delta_srg=0.001, timeout=10.0):
        super().__init__()
        self.srg = srg
        self.lambda_conf = lambda_conf
        self.lambda_kl = lambda_kl
        self.lambda_dist = lambda_dist
        self.eta0 = eta0
        self.rho = rho
        self.epsilon = epsilon
        self.delta_srg = delta_srg
        self.timeout = timeout

    def optimize(self, z_init, c_query, logits_fn, c_target=None, p_target=None):
        """
        z_init: [D] — начальный латентный код
        c_query: [D] — центроид запроса
        logits_fn: callable(z) → (logits [V], c_out [D])
        c_target: [D] — целевой центроид (опционально)
        p_target: [V] — целевое распределение (опционально)

        Returns: z_optimized [D]
        """
        import time
        z = z_init.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=self.eta0)
        prev_z = z.clone()
        srg_history = [0.0, 0.0, 0.0]
        prev_grad = None
        start_time = time.time()
        t = 0

        while True:
            opt.param_groups[0]['lr'] = self.eta0 * (self.rho ** t)

            logits, c_out = logits_fn(z)

            # Differentiable SRG
            sim = F.cosine_similarity(c_query.unsqueeze(0), c_out.unsqueeze(0), dim=-1)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log2(probs + 1e-10)).sum(dim=-1)
            max_ent = math.log2(logits.shape[-1])
            entropy_score = 1.0 - entropy / max_ent
            srg_score = self.srg.w_sim * sim + self.srg.w_ent * entropy_score

            # Loss
            loss = -self.lambda_conf * srg_score
            if p_target is not None:
                p = F.softmax(logits, dim=-1)
                loss += self.lambda_kl * F.kl_div(p.log(), p_target, reduction='sum')
            if c_target is not None:
                loss += self.lambda_dist * F.mse_loss(c_out, c_target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # ---- Adaptive stopping criteria ----

            # 1. Convergence norm
            if torch.norm(z - prev_z).item() < self.epsilon:
                break

            # 2. SRG stagnation (3 consecutive steps)
            srg_history.pop(0)
            srg_history.append(srg_score.item())
            if max(srg_history) - min(srg_history) < self.delta_srg and t >= 3:
                break

            # 3. Oscillation detection
            if prev_grad is not None and z.grad is not None:
                cos = F.cosine_similarity(
                    z.grad.view(1, -1), prev_grad.view(1, -1)
                ).item()
                if cos < -0.5:
                    z.data = (z.data + prev_z.data) * 0.5
                    break

            prev_z = z.clone()
            if z.grad is not None:
                prev_grad = z.grad.clone()

            # 4. Wall-clock timeout
            if time.time() - start_time > self.timeout:
                break

            t += 1

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
        self.max_cap = 256  # макс общее число путей (MX550: 2.1GB)

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

        # Combine all BFS levels (different N per level -> can't stack)
        if all_vecs:
            for lvl_bias, lvl_scale in zip(all_vecs, all_scales):
                bias = bias + (lvl_bias * lvl_scale.unsqueeze(-1)).sum(dim=0)

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


# ============================================================
# 8. HierarchicalAdditiveField — иерархическое аддитивное поле
# ============================================================

class HierarchicalAdditiveField(nn.Module):
    """
    Variable-arity additive decomposition с иерархическими аттракторами.

    Принцип (из обсуждения 01.06.2026):
      z ∈ ℝᴰ → [z₁, ..., z_K], K ∈ [0, max_arity], z = Σ zᵢ + ε
      Zeros (null vectors) — разделители: [0, z₁, z₂, 0] = две компоненты
      Множественные пути: z = z₁+z₂ = w₁+w₂+w₃ (разные разложения)
      Каждая zᵢ рекурсивно раскладывается

    Архитектура — Sequential Decomposition (гарантирует реконструкцию):
      r₀ = z
      for k in 1..max_arity:
        stop_k = σ(MLP_stop(r_{k-1})) — вероятность остановки
        v_k = MLP_slot(r_{k-1})       — следующий компонент
        r_k = r_{k-1} - v_k           — residual
      z_hat = Σ v_k

      Multi-path: dropout noise в MLP_slot даёт разные разложения
    """

    def __init__(self, coord_dim=384, max_arity=8, max_depth=5,
                 attractor_sigma=0.5, creation_threshold=0.1):
        super().__init__()
        self.coord_dim = coord_dim
        self.max_arity = max_arity
        self.max_depth = max_depth

        # Slot MLP: predicts component from residual with skip-connection
        # v_k = MLP_slot(r) + r  → guarantees v_k ≈ r on init
        self.slot_net = nn.Sequential(
            nn.Linear(coord_dim, coord_dim), nn.SiLU(),
            nn.Linear(coord_dim, coord_dim),
        )
        # Init last layer near-zero so initial v_k ≈ r
        with torch.no_grad():
            self.slot_net[-1].weight.zero_()
            self.slot_net[-1].bias.zero_()

        # STOP head — one sigmoid per step
        self.stop_head = nn.Linear(coord_dim, 1)
        # Init STOP bias so initial P(stop) ≈ 0.3 (keeps ~2-3 steps)
        with torch.no_grad():
            self.stop_head.bias.data.fill_(-0.7)

        # Slot embedding — learnable positional bias per slot
        self.slot_pos = nn.Parameter(torch.randn(max_arity, coord_dim) * 0.02)

        # Depth-dependent scaling
        self.depth_scale = nn.Parameter(torch.ones(max_depth) * 0.5)

        # Attractor field
        self.attractors = AttractorField(
            coord_dim=coord_dim, sigma=attractor_sigma,
            creation_threshold=creation_threshold,
        )

        # Gumbel-Softmax temperature (learned, for STOP)
        self.gs_temp = nn.Parameter(torch.tensor(1.0))

    # ─── Sequential decomposition ───

    def decompose(self, z, noise_dropout=0.0, max_steps=None):
        """
        Sequential decomposition: each step explains residual.

        z: [D]
        noise_dropout: если > 0, применяется dropout в slot_net (multi-path)

        Returns:
          parts: [K, D] — активные компоненты
          info: dict с residuals, stop_probs, K для loss
        """
        D = self.coord_dim
        K_max = max_steps or self.max_arity

        parts_list = []
        residuals = []
        stop_probs = []
        r = z.clone()

        for k in range(K_max):
            residuals.append(r.clone())

            # Stop probability from residual
            stop_logit = self.stop_head(r)  # [1]
            stop_prob = torch.sigmoid(stop_logit)  # [0, 1]
            stop_probs.append(stop_prob)

            # Differentiable stopping via Gumbel-Sigmoid during training
            if self.training:
                # Straight-through Gumbel-Sigmoid
                tau = self.gs_temp.clamp(min=0.1, max=10.0)
                gumbel_noise = -torch.log(-torch.log(
                    torch.rand_like(stop_logit) + 1e-10) + 1e-10)
                stop_logit_noisy = (stop_logit + gumbel_noise) / tau
                stop_hard = (torch.sigmoid(stop_logit_noisy) > 0.5).float()
                stop_soft = torch.sigmoid(stop_logit_noisy)
                stop_mask = stop_hard + stop_soft - stop_soft.detach()
                # stop_mask ≈ 0 if we should continue (k < K), ≈ 1 if stop
            else:
                # Inference: hard stop
                stop_mask = (stop_prob > 0.5).float()

            # If this is the first step (k=0), we never stop at step 0
            # (need at least one component)
            if k == 0:
                stop_mask = torch.zeros_like(stop_mask)

            # Generate slot vector: v_k = MLP(r + pos) + r (skip)
            # skip-connection guarantees v_k ≈ r on init → 1-step near-perfect AE
            r_input = r + self.slot_pos[k]
            if noise_dropout > 0 and self.training:
                r_input = F.dropout(r_input, p=noise_dropout)

            v_k = self.slot_net(r_input) + r  # residual prediction
            parts_list.append(v_k)

            # Update residual
            r = r - v_k

            # If stop, freeze residual for remaining slots (keep = zeros)
            if stop_mask.item() > 0.5:
                break

        K = len(parts_list)
        if K == 0:
            parts = torch.zeros(0, D, device=z.device)
        else:
            parts = torch.stack(parts_list)

        info = {
            'residuals': residuals,
            'stop_probs': stop_probs,
            'K': K,
            'final_residual': r,
        }

        return parts, info

    def compose(self, parts):
        """parts: [K, D] → z_hat = Σ parts"""
        if parts.shape[0] == 0:
            return torch.zeros(self.coord_dim, device=parts.device)
        return parts.sum(dim=0)

    def reconstruct(self, z, noise_dropout=0.0):
        """z → decompose → compose → z_hat"""
        parts, info = self.decompose(z, noise_dropout=noise_dropout)
        z_hat = self.compose(parts)
        return z_hat, info

    def forward(self, z, noise_dropout=0.0):
        return self.reconstruct(z, noise_dropout=noise_dropout)

    # ─── Hierarchical decomposition (tree) ───

    def hierarchical_decompose(self, z, depth=None, noise_dropout=0.0):
        """Рекурсивное дерево декомпозиции."""
        if depth is None:
            depth = self.max_depth
        parts, info = self.decompose(z, noise_dropout=noise_dropout)
        node = {'z': z, 'parts': parts, 'info': info, 'children': []}
        if depth > 0 and parts.shape[0] > 0:
            for p in parts:
                child = self.hierarchical_decompose(p, depth - 1, noise_dropout)
                node['children'].append(child)
        return node

    # ─── Attractor storage ───

    def store_hierarchical(self, z, depth=None, noise_dropout=0.0):
        """Рекурсивное сохранение всей иерархии в аттракторы."""
        if depth is None:
            depth = self.max_depth
        parts, info = self.decompose(z, noise_dropout=noise_dropout)
        self.attractors.hebbian_update(z)
        for p in parts:
            self.attractors.hebbian_update(p)
            if depth > 0:
                self.store_hierarchical(p, depth - 1, noise_dropout)

    # ─── Multi-path training loss ───

    def multi_path_loss(self, z, n_paths=2,
                        w_recon=1.0, w_cross=0.1, w_diversity=0.01,
                        w_sparsity=0.001, w_residual=0.01):
        """
        Multi-path loss: consistency + diversity + sparsity.

        n_paths: each path uses different dropout → different decomposition
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        paths = []
        for i in range(n_paths):
            dropout = 0.1 + 0.3 * (i / max(n_paths - 1, 1))
            parts, info = self.decompose(z[0], noise_dropout=dropout)
            paths.append((parts, info))

        loss_recon = 0.0
        loss_cross = 0.0
        loss_diversity = 0.0
        loss_sparsity = 0.0
        loss_residual = 0.0

        z_hats = []
        for parts, info in paths:
            z_hat = self.compose(parts)
            z_hats.append(z_hat)
            loss_recon = loss_recon + F.mse_loss(z_hat, z[0])

            # Sparsity: few components
            loss_sparsity = loss_sparsity + info['K'] / self.max_arity

            # Residual: ||r_K|| should be small (reconstruction quality)
            loss_residual = loss_residual + info['final_residual'].norm()

        # Cross-consistency
        if len(z_hats) >= 2:
            for i in range(len(z_hats)):
                for j in range(i + 1, len(z_hats)):
                    loss_cross = loss_cross + F.mse_loss(z_hats[i], z_hats[j])

        # Diversity: different paths → different slot patterns
        if len(paths) >= 2 and all(p[0].shape[0] > 0 for p in paths):
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    # Compare component-averaged vectors
                    a_i = paths[i][0].mean(dim=0)
                    a_j = paths[j][0].mean(dim=0)
                    sim = F.cosine_similarity(a_i.unsqueeze(0), a_j.unsqueeze(0))
                    loss_diversity = loss_diversity + sim

        total = (w_recon * loss_recon + w_cross * loss_cross
                 + w_diversity * loss_diversity + w_sparsity * loss_sparsity
                 + w_residual * loss_residual)

        return {
            'total': total,
            'recon': loss_recon,
            'cross': loss_cross,
            'diversity': loss_diversity,
            'sparsity': loss_sparsity,
            'residual': loss_residual,
        }

    # ─── Generation ───

    def nxt_direction(self, z):
        """
        z: [B, D] → decompose → attractor direction per component → combine.
        """
        B, D = z.shape
        results = []
        for b in range(B):
            parts, info = self.decompose(z[b])
            if parts.shape[0] == 0:
                results.append(torch.zeros(D, device=z.device))
                continue
            dirs = []
            for p in parts:
                if self.attractors.n_attractors > 0:
                    dirs.append(
                        self.attractors.nxt_direction(p.unsqueeze(0))[0])
                else:
                    dirs.append(torch.randn(D, device=z.device) * 0.1)
            results.append(torch.stack(dirs).mean(dim=0))
        return torch.stack(results)

    def summary(self):
        n_att = self.attractors.n_attractors if hasattr(self, 'attractors') else 0
        params = sum(p.numel() for p in self.parameters())
        return (f"HierarchicalAdditiveField(D={self.coord_dim}, "
                f"max_arity={self.max_arity}, max_depth={self.max_depth}, "
                f"attractors={n_att}, params={params})")


# ============================================================
# 9. Demo: иерархическое аддитивное разложение чисел
# ============================================================

def demo_hierarchical_addition():
    """
    Демонстрация HierarchicalAdditiveField на задаче аддитивного разложения.

    Структурированные данные: z генерируется как сумма 2-4 случайных
    базисных векторов. Модель учится раскладывать z обратно.
    """
    D = 64
    depth = 2
    n_basis = 20  # 20 "концептов" в базе

    haf = HierarchicalAdditiveField(
        coord_dim=D, max_arity=6, max_depth=depth,
        creation_threshold=0.05,
    )

    # Генератор структурированных данных
    # z = c₁·b₁ + c₂·b₂ + ... (sum of weighted random basis vectors)
    basis = torch.randn(n_basis, D)
    basis = F.normalize(basis, dim=-1)

    def make_structured_z():
        n_active = torch.randint(2, 5, (1,)).item()  # 2-4 components
        idx = torch.randperm(n_basis)[:n_active]
        coeffs = torch.randn(n_active) * 1.5
        z = (basis[idx] * coeffs.unsqueeze(-1)).sum(dim=0)
        return z, idx, coeffs

    opt = torch.optim.AdamW(haf.parameters(), lr=1e-3, weight_decay=1e-5)
    n_steps = 500

    print(f"=== HierarchicalAdditiveField Demo ===")
    print(f"Dim={D}, max_arity=6, depth={depth}")
    print(f"Structured synthetic: z = Sum(c_k * basis[idx_k]), 2-4 components")
    print(f"Training {n_steps} steps...\n")

    for step in range(n_steps + 1):
        z, _, _ = make_structured_z()
        loss_dict = haf.multi_path_loss(z, n_paths=2, w_cross=0.05, w_sparsity=0.005)

        opt.zero_grad()
        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(haf.parameters(), 1.0)
        opt.step()

        if step % 100 == 0:
            parts, info = haf.decompose(z)
            z_hat = haf.compose(parts)
            err = (z_hat - z).norm().item()
            print(f"  step {step:4d}: total={loss_dict['total'].item():.4f} "
                  f"recon={loss_dict['recon'].item():.4f} "
                  f"cross={loss_dict['cross'].item():.4f} "
                  f"K={info['K']} "
                  f"|z_hat - z|={err:.4f} "
                  f"residual={info['final_residual'].norm().item():.4f}")

    # Final evaluation on held-out test vector
    z_test, true_idx, true_coeffs = make_structured_z()

    # Path A: low dropout
    parts_a, info_a = haf.decompose(z_test, noise_dropout=0.05)
    z_hat_a = haf.compose(parts_a)

    # Path B: high dropout (different decomposition)
    parts_b, info_b = haf.decompose(z_test, noise_dropout=0.5)
    z_hat_b = haf.compose(parts_b)

    err_a = (z_hat_a - z_test).norm().item()
    err_b = (z_hat_b - z_test).norm().item()
    cross_err = (z_hat_a - z_hat_b).norm().item()

    print(f"\n=== Final Evaluation ===")
    print(f"  True components: K={len(true_idx)}")
    print(f"  Path A (dropout=0.05): K={info_a['K']}, err={err_a:.4f}")
    print(f"  Path B (dropout=0.50): K={info_b['K']}, err={err_b:.4f}")
    print(f"  Cross-consistency: {cross_err:.4f}")
    print(f"  Stop probs A: {[f'{p.item():.3f}' for p in info_a['stop_probs']]}")
    print(f"  Stop probs B: {[f'{p.item():.3f}' for p in info_b['stop_probs']]}")

    # Hierarchical decomposition tree
    tree = haf.hierarchical_decompose(z_test, noise_dropout=0.1)
    print(f"\n=== Tree (depth={depth}) ===")
    def _print_tree(node, indent=0):
        prefix = '  ' * indent
        print(f"{prefix}K={node['info']['K']} "
              f"|res|={node['info']['final_residual'].norm().item():.3f}")
        for child in node['children']:
            _print_tree(child, indent + 1)
    _print_tree(tree)

    # Attractor storage
    for _ in range(100):
        z, _, _ = make_structured_z()
        haf.store_hierarchical(z, depth=1)

    print(f"\n  Attractors stored: {haf.attractors.n_attractors}")

    # Nearest attractor to test vector
    if haf.attractors.n_attractors > 0:
        valid = haf.attractors.valid_mask[:haf.attractors.n_attractors]
        centers = haf.attractors.centers[:haf.attractors.n_attractors][valid]
        dists = torch.cdist(z_test.unsqueeze(0), centers, p=2)
        min_dist = dists.min().item()
        print(f"  Nearest attractor to test: {min_dist:.4f}")

    print("=== Demo Complete ===")
    return haf

