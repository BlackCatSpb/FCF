"""
Heads — все heads для shared encoder EVA Symbolic v3.

TrajectoryBoundaryPredictor — 3×ℝ¹²⁸: конец слова, начало следующего, вектор связи
BoundaryValidator      — softmax(word, sentence): разрешение неоднозначности
ConceptHead            — sigmoid: важность концепта из intrinsic density
ContradictionHead      — sigmoid: уровень противоречия из intrinsic uncertainty
UncertaintyHead        — log-variance per coordinate: intrinsic uncertainty signal
MetaWeighter           — softmax(3): веса know/conc/contr с bias на знания
WeightProjector        — проекция весов модели → ℝ¹²⁸ как токен контекста
MoEFFN                 — Mixture-of-Experts FFN: per-token expert routing
"""
import torch, torch.nn as nn, torch.nn.functional as F
import math


class MoEFFN(nn.Module):
    """
    Mixture-of-Experts FFN: per-token routing между N экспертами.

    Каждый эксперт — SwiGLUFFN с hidden_dim/n_experts.
    Router: h → softmax(N).
    Output: weighted sum.
    """
    def __init__(self, dim: int, hidden_dim: int, n_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        expert_hidden = max(hidden_dim // n_experts, 16)

        self.router = nn.Linear(dim, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, expert_hidden, bias=False),
                nn.SiLU(),
                nn.Linear(expert_hidden, dim, bias=False),
            ) for _ in range(n_experts)
        ])

    def forward(self, x):
        route_logits = self.router(x)
        if self.training and self.top_k < self.n_experts:
            noise = torch.randn_like(route_logits) * 0.1
            route_logits = route_logits + noise
        route_weights = F.softmax(route_logits, dim=-1)

        if self.top_k < self.n_experts:
            top_w, top_idx = route_weights.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(route_weights)
            mask.scatter_(-1, top_idx, 1.0)
            route_weights = route_weights * mask
            route_weights = route_weights / (route_weights.sum(dim=-1, keepdim=True) + 1e-8)

        out = 0.0
        for i, expert in enumerate(self.experts):
            w = route_weights[..., i:i+1]
            out = out + w * expert(x)
        return out


class TrajectoryBoundaryPredictor(nn.Module):
    """
    h → (end_coord, next_coord, conn_vector) — все в ℝ^d_model.
    """
    def __init__(self, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 256), nn.SiLU(),
            nn.Linear(256, d_model * 3),
        )

    def forward(self, h):
        out = self.mlp(h)
        d = self.d_model
        end = out[..., :d]
        nxt = out[..., d:2*d]
        conn = out[..., 2*d:]
        return end, nxt, conn


class BoundaryValidator(nn.Module):
    """
    h + z_current → softmax(word_boundary, sentence_boundary).
    """
    def __init__(self, d_model=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, 64), nn.SiLU(),
            nn.Linear(64, 2), nn.Softmax(dim=-1),
        )

    def forward(self, h, z_current):
        inp = torch.cat([h, z_current], dim=-1)
        return self.mlp(inp)


class ConceptHead(nn.Module):
    """
    h → concept_probability [0,1].
    Учится предсказывать intrinsic cluster density траектории:
    concept = слово, вокруг которого плотный кластер в trajectory space.
    """
    def __init__(self, d_model=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, h):
        return self.mlp(h).squeeze(-1)


class ContradictionHead(nn.Module):
    """
    h → contradiction_probability [0,1].
    Учится предсказывать intrinsic uncertainty траектории:
    contradiction = позиция с высокой variance предсказания.
    """
    def __init__(self, d_model=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, h):
        return self.mlp(h).squeeze(-1)


class UncertaintyHead(nn.Module):
    """
    h → log_variance [128].
    Предсказывает per-dimension variance trajectory prediction.
    Обучается на реальной ошибке: MSE(z_pred, z_true).
    
    Это intrinsic contradiction signal:
    - низкая variance = модель уверена = нет противоречия
    - высокая variance = модель не уверена = противоречие
    """
    def __init__(self, d_model=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 64), nn.SiLU(),
            nn.Linear(64, d_model),
        )

    def forward(self, h):
        log_var = self.mlp(h)
        return log_var.exp()  # [B, L, 128] — per-dim variance


class ResidualHead(nn.Module):
    """
    Предсказывает delta_z = z_t - z_{t-1} из контекста h_t.

    Концепция:
    - Если модель может предсказать, куда в embedding space перейдёт следующий токен,
      значит она понимает семантический сдвиг
    - Если residual error ||delta_pred - delta_true||² велика → высокая неопределённость
    - Используется как сигнал для thought loop (contra/uncertainty) и как auxiliary loss

    Forward:
        h: [B, L, D] — скрытые состояния
        z_prev: [B, L, D] — координаты предыдущих токенов (z_{t-1})
        z_curr: [B, L, D] — координаты текущих токенов (z_t)

    Returns:
        delta_pred: [B, L, D] — предсказанные дельты
        residual_error: [B, L] — ||delta_pred - delta_true||² per position
    """
    def __init__(self, d_model=128):
        super().__init__()
        self.proj = nn.Linear(d_model * 3, d_model)  # [h_t, z_prev, z_curr] → D
        self.res_mlp = nn.Sequential(
            nn.Linear(d_model, 64), nn.SiLU(),
            nn.Linear(64, d_model),  # output: delta_z
        )

    def forward(self, h, z_prev, z_curr):
        delta_true = z_curr - z_prev  # [B, L, D]
        inp = torch.cat([h, z_prev, z_curr], dim=-1)  # [B, L, 3D]
        shortcut = self.proj(inp)
        delta_pred = self.res_mlp(shortcut)  # [B, L, D]
        residual_error = (delta_pred - delta_true).pow(2).sum(dim=-1)  # [B, L]
        return delta_pred, residual_error

    def residual_loss(self, delta_pred, delta_true):
        """MSE дельты: per-position → mean."""
        return (delta_pred - delta_true).pow(2).mean()


class MetaWeighter(nn.Module):
    """
    context_hidden → [w_know, w_conc, w_contr] — softmax, 3 sources.
    
    Head-only режим: только knowledge, concept, contra (без decoder).
    Bias по умолчанию: знания (индекс 0) в приоритете.
    Тренировочный прогрев: temperature растёт от 0.1 до 1.0 за warmup_steps.
    """
    def __init__(self, d_model=128, warmup_steps=1000):
        super().__init__()
        n_out = 3
        self.proj = nn.Linear(d_model, 64)
        self.weight_net = nn.Linear(64, n_out)
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        self.warmup_steps = warmup_steps
        self.register_buffer('_bias', torch.tensor([1.0, 0.0, 0.0]))
        self.current_step = 0

    def forward(self, context_hidden):
        if self.training and self.current_step < self.warmup_steps:
            progress = self.current_step / max(self.warmup_steps, 1)
            self.temperature.data = torch.tensor([0.1 + 0.9 * progress],
                device=self.temperature.device)
            self.current_step += 1

        h = F.silu(self.proj(context_hidden))
        raw = self.weight_net(h)
        bias = self._bias.to(raw.device) * self.temperature
        return torch.softmax(raw + bias, dim=-1)

    def kl_loss(self, context_hidden, prior=None):
        w = self.forward(context_hidden)
        if prior is None:
            prior = torch.full_like(w, 1.0 / w.shape[-1])
        return F.kl_div(w.log(), prior, reduction='batchmean')


class WeightProjector(nn.Module):
    """
    Проекция текущего состояния весов модели в ℝ¹²⁸.

    Берёт mean+std каждого parameter tensor → конкатенация → MLP → ℝ¹²⁸.
    Кэширует результат; пересчёт только при вызове update().

    Используется как токен контекста: [W_token, t1, t2, ...].
    """
    def __init__(self, coord_dim=128, max_stats=512):
        super().__init__()
        self.max_stats = max_stats
        self.mlp = nn.Sequential(
            nn.Linear(max_stats, 256), nn.SiLU(),
            nn.Linear(256, coord_dim),
        )
        self.register_buffer('_cached_token', torch.zeros(coord_dim))

    def _extract_stats(self, model):
        stats = []
        for name, param in model.named_parameters():
            if param.numel() > 0:
                stats.append(param.data.mean().view(1))
                if param.numel() > 1:
                    stats.append(param.data.std().view(1))
        flat = torch.cat(stats)
        n = flat.numel()
        if n >= self.max_stats:
            return flat[:self.max_stats]
        return F.pad(flat, (0, self.max_stats - n))

    def update(self, model):
        with torch.no_grad():
            s = self._extract_stats(model)
            s = s.to(self._cached_token.device)
            tok = self.mlp(s.unsqueeze(0)).squeeze(0)
            self._cached_token.copy_(tok)
        return self._cached_token

    def forward(self):
        return self._cached_token


class DistillationHead(nn.Module):
    """
    Проекция hidden states teacher модели в ℝ¹²⁸ + MSE loss.

    Teacher может быть любой PyTorch моделью (BERT, GPT, SmallTransformer, etc).
    DistillationHead учится проецировать h_teacher → ℝ¹²⁸ так, чтобы
    MSE(h_eva, h_teacher_proj) → 0.

    WeightProjector при этом учится извлекать из весов teacher модель сигнал,
    достаточный для предсказания его hidden states.
    """
    def __init__(self, teacher_hidden_dim: int, coord_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(teacher_hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, coord_dim),
        )
        self.mse = nn.MSELoss()

    def project(self, h_teacher: torch.Tensor) -> torch.Tensor:
        return self.proj(h_teacher)

    def loss(self, h_eva: torch.Tensor, h_teacher: torch.Tensor) -> torch.Tensor:
        """
        h_eva: [B, L, coord_dim] — скрытые состояния EVA
        h_teacher: [B, L, teacher_hidden_dim] — скрытые состояния teacher
        Returns: scalar MSE loss
        """
        h_teacher_proj = self.project(h_teacher)
        return self.mse(h_eva, h_teacher_proj)


class TeacherAdapter(nn.Module):
    """
    Адаптер для любой teacher модели.

    Оборачивает teacher, извлекает:
    - weight_token: проекция весов teacher → ℝ¹²⁸ (через переданный WeightProjector)
    - h_teacher: скрытые состояния teacher на том же input

    Используется в EVA forward:
    >>> teacher = SmallTransformer()
    >>> adapter = TeacherAdapter(teacher, teacher_hidden_dim=64)
    >>> adapter.to(device)
    >>> w_token = adapter.get_weight_token(weight_projector)  # ℝ¹²⁸
    >>> h_teacher = adapter.get_hidden(input_ids)             # [B, L, 64]
    """
    def __init__(self, teacher_model: nn.Module, teacher_hidden_dim: int):
        super().__init__()
        self.teacher = teacher_model
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        self.teacher_hidden_dim = teacher_hidden_dim

    def get_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Извлекает hidden states teacher модели на input_ids."""
        with torch.no_grad():
            if hasattr(self.teacher, 'get_hidden'):
                h_teacher = self.teacher.get_hidden(input_ids)
            elif hasattr(self.teacher, 'forward'):
                out = self.teacher(input_ids)
                if isinstance(out, tuple):
                    h_teacher = out[0]
                elif hasattr(out, 'last_hidden_state'):
                    h_teacher = out.last_hidden_state
                else:
                    h_teacher = out
            else:
                raise AttributeError('Teacher model must support forward() or get_hidden()')
        return h_teacher

    def get_weight_token(self, projector: WeightProjector) -> torch.Tensor:
        """Обновляет и возвращает weight token из весов teacher."""
        return projector.update(self.teacher)


class BoundaryDetectionHead(nn.Module):
    """
    h → [word_start, word_inside, word_end] logits [B, L, 3].

    Supervised from full_corpus_encoded.npy:
    - Position before WORD_OPEN(157) → word_end
    - Position at WORD_CLOSE(158) → word_end
    - Position at WORD_OPEN(157) → word_start
    - Regular chars → word_inside
    - SENT_OPEN/SENT_CLOSE → ignore

    Labels: 0=start, 1=inside, 2=end, -100=ignore.
    """
    def __init__(self, d_model=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 64), nn.SiLU(),
            nn.Linear(64, 3),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)  # [B, L, 3]

    @staticmethod
    def make_labels(token_ids: torch.Tensor, WORD_OPEN: int = 157,
                    WORD_CLOSE: int = 158, SENT_OPEN: int = 159,
                    SENT_CLOSE: int = 160) -> torch.Tensor:
        B, L = token_ids.shape
        labels = torch.full((B, L), -100, dtype=torch.long, device=token_ids.device)

        # Position before WORD_OPEN → word_end
        mask_end = torch.zeros_like(token_ids, dtype=torch.bool)
        if L > 1:
            mask_end[:, :-1] = token_ids[:, 1:] == WORD_OPEN
        labels[mask_end] = 2

        # Position at WORD_CLOSE → word_end (token is boundary, not char)
        labels[token_ids == WORD_CLOSE] = 2

        # Position at WORD_OPEN → word_start
        labels[token_ids == WORD_OPEN] = 0

        # Regular chars → word_inside (4..155, exclude special tokens)
        is_char = (token_ids >= 4) & (token_ids <= 155) & (token_ids != 156)
        labels[is_char] = 1

        return labels

    @staticmethod
    def boundary_loss(logits: torch.Tensor, token_ids: torch.Tensor,
                      **kwargs) -> torch.Tensor:
        labels = BoundaryDetectionHead.make_labels(token_ids, **kwargs)
        return F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1),
                               ignore_index=-100)


def potential_guided_logits(z_pred, sym_coords, bias_tpf, bias_wvf, temperature=0.8):
    """
    Формирование логитов через расстояния до символов (НЕ обучаемый слой).
    
    z_pred:  [D] — предсказанная координата
    sym_coords: [V, D] — таблица координат символов
    bias_tpf: [V] — bias от TensorPotentialField
    bias_wvf: [V] — bias от WordValenceField
    
    Returns: logits [V]
    """
    dists = torch.cdist(z_pred.unsqueeze(0), sym_coords, p=2).squeeze(0)
    return -dists / temperature + bias_tpf * 0.1 + bias_wvf * 0.05
