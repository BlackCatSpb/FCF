"""
StateAlgebra — композиция знаний в латентном пространстве (Пункт 7).

Операторы:
- sum: z_{A+B} = Proj(z_A + z_B) — аддитивная композиция
- scale: z_{αA} = Proj(α * z_A) — усиление/ослабление
- subtract: z_{A\\B} = Proj(clamp(z_A - β*z_B, -τ, τ)) — фильтрация
- cross_attend: z_comp = Proj(CrossAttn([z_A; z_B])) — нелинейная композиция

Proj_ℳ — проектор на многообразие валидных состояний (лёгкий автоэнкодер).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
from loguru import logger


class Projector(nn.Module):
    """Proj_ℳ: автоэнкодер для проекции на многообразие валидных состояний."""

    def __init__(self, dim: int = 2560, bottleneck: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(z)
        decoded = self.decoder(encoded)
        return decoded

    def project(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(z)


class CrossAttendBlock(nn.Module):
    """TransformerBlock для кросс-аттеншн композиции латентных кодов (§8.2)."""

    def __init__(self, dim: int = 2560, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.W_Q = nn.Linear(dim, dim, bias=False)
        self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False)
        self.W_O = nn.Linear(dim, dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q = self.W_Q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_K(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_V(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim), dim=-1
        )
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        out = self.W_O(out)
        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x.mean(dim=1)


class StateAlgebra:

    def __init__(self, dim: int = 2560, bottleneck: int = 256):
        self.dim = dim
        self.projector = Projector(dim, bottleneck)
        self.cross_attn_block = CrossAttendBlock(dim)
        self.translator = nn.Linear(dim * 2, dim, bias=False)
        nn.init.xavier_uniform_(self.translator.weight)

    def sum(self, z_A: np.ndarray, z_B: np.ndarray) -> np.ndarray:
        z_a = torch.from_numpy(z_A).float().unsqueeze(0)
        z_b = torch.from_numpy(z_B).float().unsqueeze(0)
        result = self.projector.project(z_a + z_b)
        return result.squeeze(0).numpy()

    def scale(self, z: np.ndarray, alpha: float) -> np.ndarray:
        z_t = torch.from_numpy(z).float().unsqueeze(0)
        result = self.projector.project(alpha * z_t)
        return result.squeeze(0).numpy()

    def subtract(
        self,
        z_A: np.ndarray,
        z_B: np.ndarray,
        beta: float = 1.0,
        tau: float = 1.0,
    ) -> np.ndarray:
        z_a = torch.from_numpy(z_A).float().unsqueeze(0)
        z_b = torch.from_numpy(z_B).float().unsqueeze(0)
        diff = torch.clamp(z_a - beta * z_b, -tau, tau)
        result = self.projector.project(diff)
        return result.squeeze(0).numpy()

    def cross_attend(self, z_A: np.ndarray, z_B: np.ndarray) -> np.ndarray:
        """
        Cross-attention через полноценный TransformerBlock (§8.2).

        z_comp = Proj_M(TransformerBlock([z_A; z_B]))
        """
        z_a = torch.from_numpy(z_A).float().unsqueeze(0).unsqueeze(1)
        z_b = torch.from_numpy(z_B).float().unsqueeze(0).unsqueeze(1)
        combined = torch.cat([z_a, z_b], dim=1)
        attn_out = self.cross_attn_block(combined)
        result = self.projector.project(attn_out)
        return result.squeeze(0).numpy()

    def translate(self, z_A: np.ndarray, c_domain_B: np.ndarray) -> np.ndarray:
        """
        Cross-Domain Translation: z_{A→B} = Proj(Translator(z_A, c_domain_B)) (§8.2).
        """
        za = torch.from_numpy(z_A).float().unsqueeze(0)
        cb = torch.from_numpy(c_domain_B).float().unsqueeze(0)
        x = torch.cat([za, cb], dim=-1)
        translated = self.translator(x)
        result = self.projector.project(translated)
        return result.squeeze(0).numpy()

    def compose(
        self,
        components: List[Tuple[str, np.ndarray]],
        operation: str = "sum",
    ) -> np.ndarray:
        if not components:
            return np.zeros(self.dim, dtype=np.float32)

        if operation == "sum":
            z = np.sum([c[1] for c in components], axis=0)
            return self.projector.project(
                torch.from_numpy(z).float().unsqueeze(0)
            ).squeeze(0).numpy()

        if operation == "mean":
            z = np.mean([c[1] for c in components], axis=0)
            return self.projector.project(
                torch.from_numpy(z).float().unsqueeze(0)
            ).squeeze(0).numpy()

        if operation == "weighted":
            if len(components) >= 2:
                return self.cross_attend(components[0][1], components[1][1])

        return components[0][1]

    def save(self, path: str):
        torch.save(self.projector.state_dict(), path)

    def load(self, path: str):
        self.projector.load_state_dict(torch.load(path, map_location="cpu"))

    def train_projector(
        self,
        examples: List[np.ndarray],
        epochs: int = 100,
        lr: float = 1e-3,
    ):
        optimizer = torch.optim.Adam(self.projector.parameters(), lr=lr)
        data = torch.from_numpy(np.stack(examples)).float()

        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed = self.projector(data)
            loss = F.mse_loss(reconstructed, data)
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                logger.debug(
                    f"[StateAlgebra] epoch={epoch}, loss={loss.item():.6f}"
                )

        logger.info(f"[StateAlgebra] Проектор обучен: loss={loss.item():.6f}")
