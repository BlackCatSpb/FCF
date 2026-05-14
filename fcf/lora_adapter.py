"""
LoRA (Low-Rank Adaptation) — низкоранговые адаптеры для доменных правил.

Каждый адаптер представляет пару матриц A (d_model × rank) и B (rank × d_model),
которые добавляются к весам слоя: W' = W + alpha * B @ A

Применяется к матрицам внимания (W_Q, W_K, W_V, W_O) и FFN (gate_proj, up_proj, down_proj).

Иерархия рангов (из FCF-дизайна):
- rank=4: базовые слои (грамматика, стиль)
- rank=8: доменно-специфичная информация
- rank=16: сложные рассуждения
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List
from loguru import logger


TARGET_MODULES = ["W_Q", "W_K", "W_V", "W_O", "gate_proj", "up_proj", "down_proj"]


class LoRAAdapter(nn.Module):

    def __init__(
        self,
        d_model: int = 2560,
        rank: int = 8,
        alpha: float = 1.0,
        target_modules: Optional[List[str]] = None,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.alpha = alpha
        self.ff_mult = ff_mult
        self.target_modules = target_modules or TARGET_MODULES

        self.A: Dict[str, nn.Parameter] = {}
        self.B: Dict[str, nn.Parameter] = {}

        for module_name in self.target_modules:
            in_dim, out_dim = self._get_module_dims(module_name)
            self.A[module_name] = nn.Parameter(
                torch.randn(rank, in_dim) * 0.02
            )
            self.B[module_name] = nn.Parameter(
                torch.zeros(out_dim, rank)
            )

    def _get_module_dims(self, module_name: str) -> tuple:
        if module_name in ("gate_proj", "up_proj"):
            return (self.d_model, self.d_model * self.ff_mult)
        elif module_name == "down_proj":
            return (self.d_model * self.ff_mult, self.d_model)
        else:
            return (self.d_model, self.d_model)

    def forward(
        self, x: torch.Tensor, W: torch.Tensor, module_name: str
    ) -> torch.Tensor:
        A, B = self.A[module_name], self.B[module_name]
        delta = (B @ A) * (self.alpha / self.rank)
        return nn.functional.linear(x, W + delta.to(W.device))

    def get_delta(self, module_name: str) -> torch.Tensor:
        A, B = self.A[module_name], self.B[module_name]
        return (B @ A) * (self.alpha / self.rank)

    def apply_to_layer(self, layer: nn.Module) -> Dict[str, torch.Tensor]:
        saved = {}
        for module_name in self.target_modules:
            if hasattr(layer, module_name):
                original = getattr(layer, module_name)
                saved[module_name] = original.weight.data.clone()
                delta = self.get_delta(module_name)
                original.weight.data = original.weight.data + delta.to(
                    original.weight.device
                )
        return saved

    def remove_from_layer(
        self, layer: nn.Module, saved: Dict[str, torch.Tensor]
    ):
        for module_name, original_weight in saved.items():
            if hasattr(layer, module_name):
                getattr(layer, module_name).weight.data = original_weight.to(
                    getattr(layer, module_name).weight.device
                )

    def get_trainable_parameters(self):
        params = []
        for A in self.A.values():
            params.append(A)
        for B in self.B.values():
            params.append(B)
        return params

    def to_numpy(self) -> Dict[str, Dict[str, np.ndarray]]:
        data = {}
        for name in self.target_modules:
            data[name] = {
                "A": self.A[name].detach().cpu().numpy(),
                "B": self.B[name].detach().cpu().numpy(),
            }
        return data

    @classmethod
    def from_numpy(
        cls,
        data: Dict[str, Dict[str, np.ndarray]],
        alpha: float = 1.0,
        ff_mult: int = 4,
        d_model: int = 2560,
    ) -> "LoRAAdapter":
        first_name = list(data.keys())[0]
        rank = data[first_name]["A"].shape[0]
        target_modules = list(data.keys())

        adapter = cls(
            d_model=d_model,
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
            ff_mult=ff_mult,
        )

        for name in target_modules:
            adapter.A[name] = nn.Parameter(
                torch.from_numpy(data[name]["A"]).float()
            )
            adapter.B[name] = nn.Parameter(
                torch.from_numpy(data[name]["B"]).float()
            )

        return adapter

    def save(self, path: str):
        import pickle
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "d_model": self.d_model,
            "rank": self.rank,
            "alpha": self.alpha,
            "ff_mult": self.ff_mult,
            "target_modules": self.target_modules,
            "weights": self.to_numpy(),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"[LoRA] Адаптер сохранён: {path}")

    @classmethod
    def load(cls, path: str) -> "LoRAAdapter":
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        adapter = cls.from_numpy(
            data["weights"],
            alpha=data.get("alpha", 1.0),
            ff_mult=data.get("ff_mult", 4),
        )
        adapter.d_model = data["d_model"]
        adapter.rank = data["rank"]
        adapter.target_modules = data["target_modules"]
        logger.info(f"[LoRA] Адаптер загружен: {path}")
        return adapter

    @property
    def param_count(self) -> int:
        total = 0
        for name in self.target_modules:
            total += self.A[name].numel() + self.B[name].numel()
        return total

    def summary(self) -> str:
        return (
            f"LoRAAdapter(d_model={self.d_model}, rank={self.rank}, "
            f"alpha={self.alpha}, modules={self.target_modules}, "
            f"params={self.param_count:,})"
        )

    def __repr__(self) -> str:
        return self.summary()
