"""
EVA — Gradient Flow Reasoning Engine (GFRE).

Заменяет дискретную авторегрессию на непрерывную динамическую систему:
  dz/dt = -∇V_real(z) - λ_c·∇V_contr(z) + λ_m·F_manifold(z) + η(t)

Траектория z(t) = процесс рассуждения.
Точки равновесия = ответы.
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class FlowHypothesis:
    """Одна гипотеза: траектория + равновесие + метрики."""
    trajectory: np.ndarray           # [T, D] — путь в координатном пространстве
    equilibrium_z: np.ndarray        # [D] — точка равновесия
    equilibrium_text: str            # декодированный текст
    basin_depth: float               # V(z) в равновесии (глубже = реальнее)
    path_length: int                 # число шагов до сходимости
    curvature_profile: np.ndarray    # [T] — кривизна вдоль пути


class CompositePotentialField(nn.Module):
    """
    Составной потенциал: V_real + V_contradiction + V_curvature.
    
    - V_real(z):    низкий в "реальных" регионах, высокий в случайных
    - V_contr(z):   отталкивание от запрещённых зон (ContradictionFilter)
    - V_curvature(z): штраф за высокую кривизну (нестабильные переходы)
    """
    
    def __init__(self, V_real, contradiction_filter, coords, sigma=0.3):
        super().__init__()
        self.V_real = V_real
        self.contradiction = contradiction_filter
        self.register_buffer('coords', coords)  # [V, D]
        self.sigma = sigma
        
        # Learnable weights for each force component
        self.lambda_contr = nn.Parameter(torch.tensor(0.1))
        self.lambda_curv = nn.Parameter(torch.tensor(0.05))
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """V_effective(z): скалярный потенциал в точке z."""
        V_r = self.V_real(z)
        
        # Contradiction repulsion: high where forbidden regions are near
        V_c = self._contradiction_repulsion(z)
        
        # Curvature penalty: high curvature = unstable
        V_k = self._curvature_potential(z)
        
        return V_r + self.lambda_contr * V_c + self.lambda_curv * V_k
    
    def _contradiction_repulsion(self, z: torch.Tensor) -> torch.Tensor:
        """Отталкивание от запрещённых соединений."""
        if not hasattr(self.contradiction, 'forbidden') or not self.contradiction.forbidden:
            return torch.zeros_like(z[..., 0])
        
        # Use forbidden mask if available (faster)
        if hasattr(self.contradiction, 'forbidden_mask') and self.contradiction.forbidden_mask is not None:
            # Find nearest symbol
            dists = torch.cdist(z, self.coords)  # [B, D] vs [V, D]
            nearest = dists.argmin(dim=-1)  # [B]
            
            # Count forbidden transitions from nearest symbols
            repulsion = torch.zeros(z.shape[0], device=z.device)
            for i in range(z.shape[0]):
                n = nearest[i].item()
                if n < len(self.contradiction.forbidden_mask):
                    n_forbidden = self.contradiction.forbidden_mask[n].sum().item()
                    repulsion[i] = n_forbidden * 0.001
            return repulsion
        
        return torch.zeros_like(z[..., 0])
    
    def _curvature_potential(self, z: torch.Tensor) -> torch.Tensor:
        """Penalize points far from any symbol (high curvature proxy)."""
        dists = torch.cdist(z, self.coords)  # [B, D] vs [V, D]
        min_dists, _ = dists.min(dim=-1)  # [B]
        return min_dists ** 2  # квадрат расстояния до ближайшего символа
    
    def gradient(self, z: torch.Tensor) -> torch.Tensor:
        """∇V(z) через autograd."""
        z = z.detach().requires_grad_(True)
        V = self.forward(z)
        grad = torch.autograd.grad(V.sum(), z, create_graph=True)[0]
        return grad


class GradientFlowSolver:
    """
    Решатель: эволюция z₀ → z_equilibrium через градиентный поток.
    
    Метод: Euler-Maruyama (стохастическое дифференциальное уравнение).
    """
    
    def __init__(self, V_composite, decoder, coords,
                 dt=0.05, max_steps=200, tolerance=1e-3):
        self.V = V_composite
        self.decoder = decoder  # CoordinateDecoder
        self.coords = coords
        self.dt = dt
        self.max_steps = max_steps
        self.tolerance = tolerance
    
    def solve(self, z0: torch.Tensor, temperature=0.1,
              num_hypotheses=5, char_vocab=None) -> List[FlowHypothesis]:
        """
        Эволюция из z0 к равновесию.
        Разные seeds → разные гипотезы.
        """
        hypotheses = []
        device = z0.device
        
        for seed in range(num_hypotheses):
            torch.manual_seed(seed)
            z = z0.clone().detach()
            trajectory = [z.cpu().numpy()]
            
            with torch.enable_grad():
                for step in range(self.max_steps):
                    # Градиент составного потенциала
                    grad = self.V.gradient(z)
                    
                    # Langevin noise
                    noise = torch.randn_like(z) * np.sqrt(2 * temperature * self.dt)
                    
                    # Euler-Maruyama step
                    z = z - self.dt * grad + noise
                    
                    # Project to manifold: normalize to unit sphere
                    z_norm = torch.norm(z, dim=-1, keepdim=True)
                    z = z / z_norm.clamp(min=1e-8)
                    
                    trajectory.append(z.detach().cpu().numpy())
                    
                    # Convergence check
                    if torch.norm(grad) < self.tolerance:
                        break
            
            # Decode equilibrium
            with torch.no_grad():
                scores = self.decoder.forward(z)
                eq_id = scores.argmax(dim=-1).item()
                eq_text = char_vocab.decode([eq_id]) if char_vocab else str(eq_id)
                basin_depth = self.V.V_real(z).item()
            
            # Curvature profile
            traj_np = np.array(trajectory).squeeze(1)  # [T, D]
            curvature = np.zeros(len(traj_np) - 2)
            for i in range(1, len(traj_np) - 1):
                v1 = traj_np[i] - traj_np[i-1]
                v2 = traj_np[i+1] - traj_np[i]
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                curvature[i-1] = 1.0 - cos_angle  # 0 = straight, 2 = reversed
            
            hypotheses.append(FlowHypothesis(
                trajectory=traj_np,
                equilibrium_z=z.cpu().numpy().squeeze(0),
                equilibrium_text=eq_text,
                basin_depth=basin_depth,
                path_length=step + 1,
                curvature_profile=curvature,
            ))
        
        # Rank by basin depth (lower V = more "real")
        hypotheses.sort(key=lambda h: h.basin_depth)
        return hypotheses
    
    def find_attractors(self, num_starts=20, char_vocab=None) -> List[FlowHypothesis]:
        """Найти ВСЕ аттракторы: запустить из случайных точек, собрать уникальные равновесия."""
        device = next(self.V.parameters()).device
        all_hypotheses = []
        
        for _ in range(num_starts):
            z0 = torch.randn(1, self.coords.shape[1], device=device)
            z0 = z0 / z0.norm(dim=-1, keepdim=True)
            
            hyps = self.solve(z0, temperature=0.2, num_hypotheses=1, char_vocab=char_vocab)
            all_hypotheses.extend(hyps)
        
        # Deduplicate by equilibrium proximity
        unique = []
        for h in all_hypotheses:
            is_new = True
            for u in unique:
                dist = np.linalg.norm(h.equilibrium_z - u.equilibrium_z)
                if dist < 0.1:
                    is_new = False
                    break
            if is_new:
                unique.append(h)
        
        unique.sort(key=lambda h: h.basin_depth)
        return unique
