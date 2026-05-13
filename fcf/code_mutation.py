"""
CodeMutation — эволюционный поиск + дистилляция кодов.

Мутация: с вероятностью 5% создаётся изменённая копия кода.
Если мутант показывает лучший SRG — заменяет оригинал.

Дистилляция: новый код проверяется на выразимость через существующие.
Если z_new ≈ Σ α_i · z_existing_i через State Algebra — сохраняется
только ссылка на композицию, а не сам код.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger


class CodeMutation:
    """
    Эволюционный поиск в окрестности успешных кодов.

    При сохранении кода с вероятностью mutation_rate создаётся мутант.
    Если SRG мутанта выше оригинала — замена.
    """

    def __init__(self,
                 mutation_rate: float = 0.05,
                 noise_scale: float = 0.01,
                 num_mutants: int = 3):
        self.mutation_rate = mutation_rate
        self.noise_scale = noise_scale
        self.num_mutants = num_mutants
        self.history: List[Dict] = []

    def should_mutate(self) -> bool:
        return random.random() < self.mutation_rate

    def mutate(self, code: np.ndarray, srg_fn,
               context: np.ndarray = None) -> Tuple[np.ndarray, float, bool]:
        origin_score = srg_fn(code, context) if context is not None else 1.0
        best_code = code.copy()
        best_score = origin_score
        improved = False

        for _ in range(self.num_mutants):
            noise = np.random.randn(*code.shape) * self.noise_scale
            mutant = code + noise
            mutant = mutant / (np.linalg.norm(mutant) + 1e-8)

            score = srg_fn(mutant, context) if context is not None else 1.0

            self.history.append({
                "origin_score": float(origin_score),
                "mutant_score": float(score),
                "improved": score > origin_score,
            })

            if score > best_score:
                best_score = score
                best_code = mutant
                improved = True

        if improved:
            logger.info(
                f"[Mutation] Улучшение: {origin_score:.3f} → {best_score:.3f}"
            )

        return best_code, best_score, improved


class CodeDistillation:
    """
    Проверяет, можно ли новый код выразить как композицию существующих.

    Если z_new ≈ Σ α_i · z_i, сохраняется ссылка на композицию,
    а не сам код → экономия памяти.
    """

    def __init__(self,
                 similarity_threshold: float = 0.95,
                 max_components: int = 5):
        self.similarity_threshold = similarity_threshold
        self.max_components = max_components
        self.distilled: List[Dict] = []

    def try_distill(self, z_new: np.ndarray,
                    existing_codes: List[np.ndarray],
                    max_attempts: int = 50) -> Optional[Dict]:
        if len(existing_codes) < 2:
            return None

        z = z_new.flatten()
        z = z / (np.linalg.norm(z) + 1e-8)

        best_sim = -1.0
        best_combo = None

        for _ in range(max_attempts):
            n = random.randint(2, min(self.max_components, len(existing_codes)))
            indices = random.sample(range(len(existing_codes)), n)

            combined = np.zeros_like(z)
            for idx in indices:
                c = existing_codes[idx].flatten()
                c = c / (np.linalg.norm(c) + 1e-8)
                combined += c
            combined = combined / (np.linalg.norm(combined) + 1e-8)

            sim = float(np.dot(z, combined))
            if sim > best_sim:
                best_sim = sim
                best_combo = {
                    "indices": indices,
                    "similarity": sim,
                    "components": n,
                }

        if best_combo and best_combo["similarity"] >= self.similarity_threshold:
            logger.info(
                f"[Distillation] Код выразим через {best_combo['components']} "
                f"существующих (sim={best_combo['similarity']:.3f})"
            )
            self.distilled.append(best_combo)
            return best_combo

        return None
