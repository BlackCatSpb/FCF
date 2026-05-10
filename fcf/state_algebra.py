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


class StateAlgebra:

    def __init__(self, dim: int = 2560, bottleneck: int = 256):
        self.dim = dim
        self.projector = Projector(dim, bottleneck)

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
        z_a = torch.from_numpy(z_A).float().unsqueeze(0)
        z_b = torch.from_numpy(z_B).float().unsqueeze(0)
        combined = torch.cat([z_a, z_b], dim=-1)

        attn_out = self._simple_cross_attention(z_a, z_b)
        result = self.projector.project(attn_out)
        return result.squeeze(0).numpy()

    def _simple_cross_attention(
        self, A: torch.Tensor, B: torch.Tensor
    ) -> torch.Tensor:
        sim = torch.mm(A, B.T)
        attn = torch.softmax(sim, dim=-1)
        return torch.mm(attn, B)

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
