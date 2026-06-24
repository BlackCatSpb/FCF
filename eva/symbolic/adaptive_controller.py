"""AdaptiveArchitectureController — динамическая подстройка subspace ratios.

    l_c/l_a/l_m адаптируются на основе плотности кодов (density).
    Пороги роста/прунинга — перцентильные (авто-адаптивные).

    Все дефолты из FCFConfig — SubspaceConfig удалён.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from eva.symbolic.fcf_config import FCFConfig


class AdaptiveArchitectureController:
    """Adjusts subspace ratios and thresholds based on code statistics.

    All defaults from FCFConfig.subspace_* fields.
    """

    def __init__(self, config: Optional[FCFConfig] = None,
                 latent_dim: int = 2048):
        cfg = config if config is not None else FCFConfig()
        self._cfg = cfg
        self.latent_dim = latent_dim
        self._n_updates = 0
        self._density_history: list[float] = []

    # ── Ratios from config ──────────────────────────────────
    @property
    def l_c_ratio(self) -> float:
        return self._cfg.subspace_l_c_ratio

    @l_c_ratio.setter
    def l_c_ratio(self, v: float):
        self._cfg.subspace_l_c_ratio = v

    @property
    def l_a_ratio(self) -> float:
        return self._cfg.subspace_l_a_ratio

    @l_a_ratio.setter
    def l_a_ratio(self, v: float):
        self._cfg.subspace_l_a_ratio = v

    @property
    def l_m_ratio(self) -> float:
        return self._cfg.subspace_l_m_ratio

    @l_m_ratio.setter
    def l_m_ratio(self, v: float):
        self._cfg.subspace_l_m_ratio = v

    @property
    def l_c(self) -> int:
        return max(8, int(self.latent_dim * self._cfg.subspace_l_c_ratio))

    @property
    def l_a(self) -> int:
        return max(8, int(self.latent_dim * self._cfg.subspace_l_a_ratio))

    @property
    def l_m(self) -> int:
        return self.latent_dim - self.l_c - self.l_a

    @property
    def l1_target_density(self) -> float:
        return self._cfg.subspace_l1_target_density

    @property
    def growth_factor(self) -> float:
        return self._cfg.subspace_growth_factor

    @property
    def sector_depths(self) -> list:
        return self._cfg.subspace_sector_depths

    @property
    def density_threshold_grow(self) -> float:
        """Auto-threshold: 90th percentile of all concept densities."""
        if len(self._density_history) < 100:
            return self._cfg.subspace_density_threshold_grow
        return float(np.quantile(self._density_history, 0.9))

    @property
    def density_threshold_prune(self) -> float:
        """Auto-threshold: 10th percentile."""
        if len(self._density_history) < 100:
            return self._cfg.subspace_density_threshold_prune
        return float(np.quantile(self._density_history, 0.1))

    def update(self, codes: dict) -> dict:
        """Update ratios based on code density statistics."""
        self._n_updates += 1
        if not codes:
            return self._snapshot()

        densities = []
        for c in codes.values():
            z_c = np.array(c[:self.l_c]) if hasattr(c, '__len__') else c
            if hasattr(z_c, 'size') and z_c.size > 0:
                active = np.mean(np.abs(z_c) > self._cfg.subspace_density_epsilon)
                densities.append(float(active))

        if not densities:
            return self._snapshot()

        mean_density = float(np.mean(densities))
        self._density_history.append(mean_density)
        if len(self._density_history) > self._cfg.subspace_density_history_maxlen:
            self._density_history.pop(0)

        target = self._cfg.subspace_l1_target_density

        if self._n_updates > self._cfg.subspace_warmup_updates:
            ratio = self._cfg.subspace_l_c_ratio
            if mean_density < target * 0.5:
                ratio = min(ratio * self._cfg.subspace_adjust_up_rate,
                            self._cfg.subspace_adjust_up_max)
            elif mean_density > target * 2.0:
                ratio = max(ratio * self._cfg.subspace_adjust_down_rate,
                            self._cfg.subspace_adjust_down_min)
            self._cfg.subspace_l_c_ratio = ratio

            remaining = 1.0 - self._cfg.subspace_l_c_ratio
            self._cfg.subspace_l_a_ratio = remaining * self._cfg.subspace_redistribute_a_ratio
            self._cfg.subspace_l_m_ratio = remaining * self._cfg.subspace_redistribute_m_ratio

        return self._snapshot()

    def _snapshot(self) -> dict:
        cfg = self._cfg
        return {
            'l_c': self.l_c,
            'l_a': self.l_a,
            'l_m': self.l_m,
            'l_c_ratio': cfg.subspace_l_c_ratio,
            'l_a_ratio': cfg.subspace_l_a_ratio,
            'l_m_ratio': cfg.subspace_l_m_ratio,
            'density_threshold_grow': self.density_threshold_grow,
            'density_threshold_prune': self.density_threshold_prune,
            'l1_target_density': cfg.subspace_l1_target_density,
            'growth_factor': cfg.subspace_growth_factor,
            'sector_depths': cfg.subspace_sector_depths,
        }
