"""
EVA — Self-Reflection: анализ собственных траекторий.

Анализирует качество траекторий: кривизну, длину, близость к запретам.
Помогает модели понимать КАК она думает.
"""

import numpy as np, torch
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class TrajectoryDiagnostic:
    """Диагностика одной траектории."""
    length: int                  # число шагов
    total_distance: float         # суммарное расстояние
    mean_curvature: float         # средняя кривизна (0=прямая)
    max_curvature: float          # макс кривизна (острые повороты)
    n_contradictions: int         # число пересечений запрещённых зон
    centroid_shift: float         # смещение центроида (насколько далеко ушла мысль)
    efficiency: float             # distance / length (эффективность)
    confidence: float             # 1 / (1 + curvature + contradictions)


class SelfReflection:
    """Анализатор качества траекторий."""
    
    def __init__(self, contradiction_filter=None, coords=None):
        self.contradiction = contradiction_filter
        self.coords = coords
    
    def diagnose(self, trajectory: np.ndarray, ids: List[int] = None) -> TrajectoryDiagnostic:
        """
        Проанализировать траекторию.
        
        Args:
            trajectory: [T, D] — путь в координатном пространстве
            ids: [T] — ID символов вдоль пути
        """
        T, D = trajectory.shape
        if T < 3:
            return TrajectoryDiagnostic(
                length=T, total_distance=0, mean_curvature=0, max_curvature=0,
                n_contradictions=0, centroid_shift=0, efficiency=0, confidence=1.0
            )
        
        # Step vectors
        steps = trajectory[1:] - trajectory[:-1]  # [T-1, D]
        step_norms = np.linalg.norm(steps, axis=1)
        total_dist = step_norms.sum()
        
        # Curvature
        curvature = np.zeros(T - 2)
        for i in range(1, T - 1):
            v1 = trajectory[i] - trajectory[i-1]
            v2 = trajectory[i+1] - trajectory[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                curvature[i-1] = 1.0 - np.dot(v1, v2) / (n1 * n2)
        
        mean_curv = curvature.mean() if len(curvature) > 0 else 0
        max_curv = curvature.max() if len(curvature) > 0 else 0
        
        # Contradictions
        n_contra = 0
        if ids and self.contradiction and hasattr(self.contradiction, 'forbidden_mask'):
            mask = self.contradiction.forbidden_mask
            for i in range(len(ids) - 1):
                if ids[i] < len(mask) and ids[i+1] < len(mask):
                    if mask[ids[i], ids[i+1]]:
                        n_contra += 1
        
        # Centroid shift
        centroid_shift = np.linalg.norm(trajectory[-1] - trajectory[0])
        
        # Efficiency
        efficiency = centroid_shift / (total_dist + 1e-8)
        
        # Confidence
        confidence = 1.0 / (1.0 + mean_curv * 2 + n_contra * 0.5)
        
        return TrajectoryDiagnostic(
            length=T, total_distance=total_dist,
            mean_curvature=float(mean_curv), max_curvature=float(max_curv),
            n_contradictions=n_contra, centroid_shift=float(centroid_shift),
            efficiency=float(efficiency), confidence=float(confidence),
        )
    
    def compare(self, trajectories: List[Dict]) -> List[Dict]:
        """Сравнить несколько траекторий, выбрать лучшие."""
        for t in trajectories:
            diag = self.diagnose(
                t.get('trajectory', np.zeros((1, 64))),
                t.get('ids', None)
            )
            t['diagnostic'] = diag
        
        trajectories.sort(key=lambda t: t['diagnostic'].confidence, reverse=True)
        return trajectories
