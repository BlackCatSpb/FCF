"""
Cross-Domain Translation + Attention over Domains.

Cross-Domain Translation: z_{A→B} = Proj(Translator(z_A, c_domain_B))
Переводит код из домена A в домен B, используя центроид целевого домена.

Attention over Domains: при композиции кодов из нескольких доменов
запрос взаимодействует с центроидами ближайших доменов,
итоговый код = взвешенная сумма кодов доменов.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from loguru import logger


class CrossDomainTranslator(nn.Module):
    """
    Переводит латентный код из домена A в домен B.
    z_{A→B} = Proj(W·[z_A; c_B] + b)
    """

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.SiLU(),
            nn.Linear(dim, dim // 2), nn.SiLU(),
            nn.Linear(dim // 2, dim),
        )

    def forward(self, z: torch.Tensor, c_target: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, c_target], dim=-1)
        translated = self.net(x)
        return translated / (torch.norm(translated, dim=-1, keepdim=True) + 1e-8)


class CrossDomainAttention(nn.Module):
    """
    Междоменное внимание: запрос взаимодействует с центроидами
    нескольких доменов, возвращает взвешенную сумму кодов.
    """

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)

    def forward(self, z_query: torch.Tensor,
                centroids: torch.Tensor,
                domain_codes: torch.Tensor) -> torch.Tensor:
        q = self.W_q(z_query)
        k = self.W_k(centroids)
        scores = torch.mm(q, k.T) / np.sqrt(z_query.shape[-1])
        weights = torch.softmax(scores, dim=-1)
        return torch.mm(weights, domain_codes)


class CrossDomainModule:
    """Обёртка для кросс-доменных операций. Делегирует StateAlgebra для translate."""

    def __init__(self, dim: int = 2560):
        self.dim = dim
        self.translator = CrossDomainTranslator(dim)
        self.attention = CrossDomainAttention(dim)

    def translate(self, z_A: np.ndarray, c_domain_B: np.ndarray) -> np.ndarray:
        z = torch.from_numpy(z_A).float().unsqueeze(0)
        c = torch.from_numpy(c_domain_B).float().unsqueeze(0)
        with torch.no_grad():
            result = self.translator(z, c)
        return result.squeeze(0).numpy()

    def translate_from_algebra(self, z_A: np.ndarray, c_domain_B: np.ndarray,
                               state_algebra) -> np.ndarray:
        if state_algebra is not None:
            return state_algebra.translate(z_A, c_domain_B)
        return self.translate(z_A, c_domain_B)

    def attend(self, z_query: np.ndarray,
               centroids: List[np.ndarray],
               codes: List[np.ndarray]) -> np.ndarray:
        z = torch.from_numpy(z_query).float().unsqueeze(0)
        c = torch.from_numpy(np.stack(centroids)).float()
        d = torch.from_numpy(np.stack(codes)).float()
        with torch.no_grad():
            result = self.attention(z, c, d)
        return result.squeeze(0).numpy()
