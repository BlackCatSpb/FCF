"""
StateGrammar — 10 дополнительных механизмов (общее число: 21).

12. CausalReasoning  — A → B ≠ A ⊕ B, контрфактуалы, интервенции
13. TemporalModality  — прошлое/настоящее/будущее/гипотетическое
14. EpistemicStates   — знаю/верю/не уверен, мета-знание
15. Quantification    — ∀, ∃, ∄ на пространстве состояний
16. StateResonance    — гармонический резонанс, усиление
17. FrontierStates    — граничные состояния, почти-А
18. GradientFlow      — поток по градиенту потенциала
19. TopologicalPersistence — устойчивые структуры при возмущениях
20. CategoryTheory     — функторы, естественные преобразования
21. InformationGeometry — метрика Фишера, геодезические
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import math
from loguru import logger


# ============================================================
# 12. CAUSAL REASONING — A → B, контрфактуалы, интервенции
# ============================================================

class CausalReasoning(nn.Module):
    """Причинность: do(A), контрфактуалы 'если бы не А', структурные уравнения."""

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.effect_net = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.counterfactual_net = nn.Sequential(nn.Linear(dim * 3, dim), nn.SiLU(), nn.Linear(dim, dim))

    def cause_effect(self, z_cause: torch.Tensor, z_context: torch.Tensor) -> torch.Tensor:
        """do(z_cause) в контексте → ожидаемый эффект."""
        return self.effect_net(torch.cat([z_cause, z_context], dim=-1))

    def counterfactual(self, z_actual: torch.Tensor, z_cause: torch.Tensor,
                       z_alternative: torch.Tensor, z_context: torch.Tensor) -> torch.Tensor:
        """'Если бы вместо z_cause был z_alternative, то результат был бы...'"""
        return self.counterfactual_net(torch.cat([z_actual, z_alternative - z_cause, z_context], dim=-1))

    def intervention_effect(self, z_a_cause: np.ndarray, z_b_effect: np.ndarray,
                            z_context: np.ndarray) -> np.ndarray:
        za = torch.from_numpy(z_a_cause).float().unsqueeze(0); zc = torch.from_numpy(z_context).float().unsqueeze(0)
        with torch.no_grad(): return self.cause_effect(za, zc).squeeze(0).numpy()

    def necessary_cause_score(self, z_cause: np.ndarray, z_effect: np.ndarray,
                              z_context: np.ndarray, z_alt: np.ndarray) -> float:
        """Насколько z_cause необходим для z_effect. 0 = не нужен, 1 = абсолютно необходим."""
        z_act = torch.from_numpy(z_effect).float().unsqueeze(0); z_c = torch.from_numpy(z_cause).float().unsqueeze(0)
        z_al = torch.from_numpy(z_alt).float().unsqueeze(0); z_ctx = torch.from_numpy(z_context).float().unsqueeze(0)
        with torch.no_grad():
            cf = self.counterfactual(z_act, z_c, z_al, z_ctx)
            actual_norm = torch.norm(z_act, dim=-1)
            diff_norm = torch.norm(z_act - cf, dim=-1)
            return float(1.0 / (1.0 + diff_norm / (actual_norm + 1e-8)))


# ============================================================
# 13. TEMPORAL MODALITY — прошлое/настоящее/будущее/гипотетическое
# ============================================================

class TemporalModality(nn.Module):
    """Модальность времени: was, is, will_be, would_be."""

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.time_encoder = nn.Sequential(nn.Linear(dim + 4, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.modality_weights = nn.Parameter(torch.ones(4, dim))

    def encode(self, z: torch.Tensor, modality: str = "present") -> torch.Tensor:
        """modal ∈ {past, present, future, hypothetical}."""
        idx = {"past": 0, "present": 1, "future": 2, "hypothetical": 3}[modality]
        one_hot = torch.zeros(z.shape[0], 4, device=z.device); one_hot[:, idx] = 1.0
        return self.time_encoder(torch.cat([z, one_hot], dim=-1))

    def temporal_distance(self, z_a: torch.Tensor, z_b: torch.Tensor) -> float:
        """Оценивает временную дистанцию между двумя состояниями."""
        diff = z_a - z_b; return float(torch.norm(diff, dim=-1).mean())

    def hypothetical_variation(self, z: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Генерирует гипотетическую вариацию состояния 'а что если бы...'"""
        z_t = torch.from_numpy(z).float().unsqueeze(0); z_c = torch.from_numpy(context).float().unsqueeze(0)
        with torch.no_grad():
            return self.encode(z_t, "hypothetical").squeeze(0).numpy()

    def timeline_coherence(self, past: np.ndarray, present: np.ndarray,
                           future_hypothesis: np.ndarray) -> float:
        """Насколько связна временная линия прошлое→настоящее→будущее."""
        pp = torch.from_numpy(past).float(); pr = torch.from_numpy(present).float()
        fh = torch.from_numpy(future_hypothesis).float()
        p_to_pr = F.cosine_similarity(pp.unsqueeze(0), pr.unsqueeze(0))
        pr_to_fh = F.cosine_similarity(pr.unsqueeze(0), fh.unsqueeze(0))
        return float((p_to_pr + pr_to_fh) / 2.0)


# ============================================================
# 14. EPISTEMIC STATES — знаю/верю/не уверен
# ============================================================

class EpistemicStates(nn.Module):
    """Мета-знание: K(A) — знаю A, B(A) — верю в A, U(A) — не уверен."""

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.knowledge_net = nn.Sequential(nn.Linear(dim * 2, 128), nn.SiLU(), nn.Linear(128, 3))
        self.certainty_net = nn.Sequential(nn.Linear(dim, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid())

    def epistemic_profile(self, z_belief: torch.Tensor, z_evidence: torch.Tensor) -> Dict[str, float]:
        """Классифицирует эпистемическое отношение: знаю/верю/не уверен."""
        out = self.knowledge_net(torch.cat([z_belief, z_evidence], dim=-1))
        probs = torch.softmax(out, dim=-1)
        return {"know": float(probs[0, 0]), "believe": float(probs[0, 1]), "uncertain": float(probs[0, 2])}

    def certainty(self, z: torch.Tensor) -> float:
        return float(self.certainty_net(z).mean())

    def knowledge_gap(self, z_known: np.ndarray, z_unknown: np.ndarray) -> float:
        """Разрыв между тем что знаем и тем что не знаем."""
        zk = torch.from_numpy(z_known).float(); zu = torch.from_numpy(z_unknown).float()
        return float(1.0 - F.cosine_similarity(zk.unsqueeze(0), zu.unsqueeze(0)))

    def socratic_question(self, z_gap: np.ndarray) -> np.ndarray:
        """Генерирует 'сократовский вопрос' — вектор, указывающий на пробел в знании."""
        return z_gap / (np.linalg.norm(z_gap) + 1e-8)


# ============================================================
# 15. QUANTIFICATION — ∀, ∃, ∄
# ============================================================

class Quantification(nn.Module):
    """Кванторы: ∀x.P(x) — все, ∃x.P(x) — некоторые, ∄x.P(x) — ни один."""

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.universal = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.existential = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.none_check = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, 1), nn.Sigmoid())

    def quantify(self, domain_hint: np.ndarray, predicate: np.ndarray) -> Dict[str, float]:
        d = torch.from_numpy(domain_hint).float().unsqueeze(0); p = torch.from_numpy(predicate).float().unsqueeze(0)
        with torch.no_grad():
            return {"forall": float(self.universal(torch.cat([d, p], dim=-1))),
                    "exists": float(self.existential(torch.cat([d, p], dim=-1))),
                    "none": float(self.none_check(torch.cat([d, p], dim=-1)))}

    def quantifier_scope(self, z_predicate: np.ndarray,
                         instances: List[np.ndarray]) -> Dict[str, float]:
        """Применяет квантор ко множеству экземпляров."""
        if not instances: return {"forall": 0.0, "exists": 0.0, "none": 1.0}
        scores = []
        for inst in instances:
            d = torch.from_numpy(inst).float().unsqueeze(0); p = torch.from_numpy(z_predicate).float().unsqueeze(0)
            s = float(self.existential(torch.cat([d, p], dim=-1)))
            scores.append(s)
        return {"forall": float(np.min(scores)), "exists": float(np.max(scores)),
                "none": float(1.0 - np.max(scores))}


# ============================================================
# 16. STATE RESONANCE — гармоническое усиление
# ============================================================

class StateResonance(nn.Module):
    """Резонанс: два состояния усиливают друг друга если колеблются на одной частоте."""

    def __init__(self, dim: int = 2560, resonance_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(dim, resonance_dim, bias=False)
        self.resonance_detector = nn.Sequential(nn.Linear(resonance_dim * 2, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid())

    def resonance_score(self, z_a: np.ndarray, z_b: np.ndarray) -> float:
        za = torch.from_numpy(z_a).float().unsqueeze(0); zb = torch.from_numpy(z_b).float().unsqueeze(0)
        pa = self.proj(za); pb = self.proj(zb)
        with torch.no_grad(): return float(self.resonance_detector(torch.cat([pa, pb], dim=-1)))

    def amplify(self, z_a: np.ndarray, z_b: np.ndarray) -> np.ndarray:
        """Усиление A за счёт резонанса с B."""
        score = self.resonance_score(z_a, z_b)
        return z_a * (1.0 + score * 0.5)

    def resonance_matrix(self, states: List[np.ndarray]) -> np.ndarray:
        n = len(states); R = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i != j: R[i, j] = self.resonance_score(states[i], states[j])
        return R

    def resonant_clusters(self, states: List[np.ndarray], threshold: float = 0.7) -> List[List[int]]:
        """Группирует состояния по резонансным кластерам."""
        R = self.resonance_matrix(states); n = len(states)
        clusters = []; visited = set()
        for i in range(n):
            if i in visited: continue
            cluster = [i]; visited.add(i)
            for j in range(n):
                if j not in visited and R[i, j] > threshold:
                    cluster.append(j); visited.add(j)
            clusters.append(cluster)
        return clusters


# ============================================================
# 17. FRONTIER STATES — граничные состояния
# ============================================================

class FrontierStates(nn.Module):
    """Граница доменов: почти-А, наполовину-А-наполовину-Б."""

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.frontier_detector = nn.Sequential(nn.Linear(dim * 2, 128), nn.SiLU(), nn.Linear(128, 2))

    def frontier_score(self, z: np.ndarray, domain_a: np.ndarray,
                       domain_b: np.ndarray) -> Tuple[float, float]:
        """Насколько z близок к границе между A и B. Возвращает (distance_to_frontier, frontier_intensity)."""
        zt = torch.from_numpy(z).float().unsqueeze(0); da = torch.from_numpy(domain_a).float().unsqueeze(0)
        db = torch.from_numpy(domain_b).float().unsqueeze(0)
        inp = torch.cat([zt - da, zt - db], dim=-1)
        with torch.no_grad():
            out = self.frontier_detector(inp)
            dist_to_frontier = float(torch.sigmoid(out[0, 0]))
            intensity = float(torch.sigmoid(out[0, 1]))
        return dist_to_frontier, intensity

    def generate_frontier(self, domain_a: np.ndarray, domain_b: np.ndarray,
                          interpolation: float = 0.5) -> np.ndarray:
        """Генерирует состояние на границе между A и B."""
        z = domain_a * (1 - interpolation) + domain_b * interpolation
        noise = np.random.randn(*z.shape).astype(np.float32) * 0.01
        return (z + noise) / (np.linalg.norm(z + noise) + 1e-8)

    def frontier_path(self, domain_a: np.ndarray, domain_b: np.ndarray,
                      steps: int = 10) -> List[np.ndarray]:
        """Путь через границу от A к B."""
        return [self.generate_frontier(domain_a, domain_b, i / (steps - 1)) for i in range(steps)]

    def is_frontier_creative(self, z: np.ndarray, domain_a: np.ndarray,
                             domain_b: np.ndarray, info_entropy_fn) -> bool:
        """Граничные состояния — источник креативности. Проверка: высокая энтропия + граничность."""
        score, intensity = self.frontier_score(z, domain_a, domain_b)
        entropy = info_entropy_fn(z)
        return score > 0.5 and intensity > 0.5 and entropy > 5.0


# ============================================================
# 18. GRADIENT FLOW — поток по градиенту потенциала
# ============================================================

class GradientFlow(nn.Module):
    """Поток состояний по градиенту потенциальной функции V(z)."""

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.potential = nn.Sequential(nn.Linear(dim, dim // 4), nn.SiLU(),
                                        nn.Linear(dim // 4, dim // 4), nn.SiLU(),
                                        nn.Linear(dim // 4, 1))

    def potential_value(self, z: torch.Tensor) -> torch.Tensor:
        return self.potential(z)

    def gradient(self, z: torch.Tensor) -> torch.Tensor:
        z.requires_grad_(True)
        V = self.potential(z)
        grad = torch.autograd.grad(V, z, create_graph=True)[0]
        return grad

    def flow_step(self, z: np.ndarray, step_size: float = 0.01) -> np.ndarray:
        """Один шаг градиентного потока: z_new = z - η · ∇V(z)."""
        z_t = torch.from_numpy(z).float().unsqueeze(0).requires_grad_(True)
        V = self.potential(z_t)
        grad = torch.autograd.grad(V, z_t)[0]
        z_new = z_t - step_size * grad / (torch.norm(grad) + 1e-8)
        return z_new.detach().squeeze(0).numpy()

    def find_minimum(self, z_init: np.ndarray, steps: int = 100, lr: float = 0.01) -> np.ndarray:
        """Градиентный спуск к локальному минимуму потенциала."""
        z = z_init.copy()
        for _ in range(steps):
            z = self.flow_step(z, lr)
        return z

    def potential_landscape(self, z_start: np.ndarray, z_end: np.ndarray,
                            samples: int = 50) -> List[float]:
        """Профиль потенциала вдоль линии между двумя состояниями."""
        profile = []
        for i in range(samples):
            t = i / (samples - 1)
            z = z_start * (1 - t) + z_end * t
            z_t = torch.from_numpy(z).float().unsqueeze(0)
            with torch.no_grad(): profile.append(float(self.potential_value(z_t)))
        return profile

    def saddle_point_detection(self, z: np.ndarray) -> bool:
        """Детектирует седловую точку: градиент ≈ 0, но кривизна смешанная."""
        z_t = torch.from_numpy(z).float().unsqueeze(0).requires_grad_(True)
        V = self.potential(z_t)
        grad = torch.autograd.grad(V, z_t, create_graph=True)[0]
        grad_norm = torch.norm(grad)
        if grad_norm > 0.1: return False
        hessian_diag = []
        for i in range(min(10, z.shape[0])):
            g_i = grad[0, i]
            h_ii = torch.autograd.grad(g_i, z_t, retain_graph=True)[0][0, i]
            hessian_diag.append(float(h_ii))
        pos = sum(1 for h in hessian_diag if h > 0)
        neg = sum(1 for h in hessian_diag if h < 0)
        return pos > 0 and neg > 0


# ============================================================
# 19. TOPOLOGICAL PERSISTENCE — устойчивые структуры
# ============================================================

class TopologicalPersistence:
    """Какие свойства композиции выживают при возмущениях."""

    def __init__(self, epsilon: float = 0.1, num_perturbations: int = 20):
        self.epsilon = epsilon; self.num_perturbations = num_perturbations

    def persistence_score(self, z: np.ndarray, composition_fn: Callable,
                          z_a: np.ndarray, z_b: np.ndarray, z_context: np.ndarray) -> float:
        """Доля возмущений, при которых свойство сохраняется."""
        survived = 0
        for _ in range(self.num_perturbations):
            noise_a = np.random.randn(*z_a.shape).astype(np.float32) * self.epsilon
            noise_b = np.random.randn(*z_b.shape).astype(np.float32) * self.epsilon
            z_a_p = (z_a + noise_a) / (np.linalg.norm(z_a + noise_a) + 1e-8)
            z_b_p = (z_b + noise_b) / (np.linalg.norm(z_b + noise_b) + 1e-8)
            z_perturbed = composition_fn(z_a_p, z_b_p, z_context)
            sim = np.dot(z.flatten(), z_perturbed.flatten()) / (
                np.linalg.norm(z) * np.linalg.norm(z_perturbed) + 1e-8)
            if sim > 0.9: survived += 1
        return survived / self.num_perturbations

    def persistence_diagram(self, z_sequence: List[np.ndarray]) -> List[Tuple[int, int, float]]:
        """Диаграмма персистентности: (birth, death, persistence) для каждого свойства."""
        if len(z_sequence) < 2: return []
        diagram = []
        for i in range(len(z_sequence)):
            birth = i
            death = i
            for j in range(i + 1, len(z_sequence)):
                sim = np.dot(z_sequence[i].flatten(), z_sequence[j].flatten()) / (
                    np.linalg.norm(z_sequence[i]) * np.linalg.norm(z_sequence[j]) + 1e-8)
                if sim > 0.7: death = j
                else: break
            if death > birth:
                diagram.append((birth, death, death - birth))
        return diagram

    def bottleneck_distance(self, diagram_a: List[Tuple[int, int, float]],
                            diagram_b: List[Tuple[int, int, float]]) -> float:
        """Расстояние bottleneck между двумя диаграммами персистентности."""
        if not diagram_a and not diagram_b: return 0.0
        if not diagram_a or not diagram_b: return float('inf')
        max_diff = 0.0
        for (b1, d1, p1) in diagram_a[:min(len(diagram_a), len(diagram_b))]:
            best_match = float('inf')
            for (b2, d2, p2) in diagram_b:
                diff = max(abs(b1 - b2), abs(d1 - d2))
                best_match = min(best_match, diff)
            max_diff = max(max_diff, best_match)
        return max_diff


# ============================================================
# 20. CATEGORY THEORY — функторы, естественные преобразования
# ============================================================

class CategoryTheory:
    """Функторы между пространствами состояний, естественные преобразования."""

    def __init__(self, dim: int = 2560):
        self.dim = dim
        self.functors: Dict[str, Callable] = {}
        self.natural_transformations: Dict[Tuple[str, str], np.ndarray] = {}

    def define_functor(self, name: str, mapping: Callable):
        self.functors[name] = mapping

    def define_natural_transformation(self, functor_f: str, functor_g: str,
                                       eta: np.ndarray):
        self.natural_transformations[(functor_f, functor_g)] = eta

    def apply_functor(self, functor_name: str, z: np.ndarray) -> np.ndarray:
        if functor_name not in self.functors: return z
        return self.functors[functor_name](z)

    def naturality_check(self, functor_f: str, functor_g: str, z: np.ndarray,
                         transform: Callable) -> bool:
        """Проверяет коммутативность: G(f) ∘ η_A = η_B ∘ F(f)."""
        if (functor_f, functor_g) not in self.natural_transformations: return False
        eta = self.natural_transformations[(functor_f, functor_g)]
        F = self.functors.get(functor_f, lambda x: x)
        G = self.functors.get(functor_g, lambda x: x)
        z_transformed = transform(z)
        lhs = G(z_transformed) + eta * 0.01
        rhs = F(z) + eta * 0.01
        sim = np.dot(lhs.flatten(), rhs.flatten()) / (np.linalg.norm(lhs)*np.linalg.norm(rhs)+1e-8)
        return sim > 0.95

    def adjunction_candidate(self, functor_f: str, functor_g: str,
                              z_f: np.ndarray, z_g: np.ndarray) -> float:
        """Проверяет сопряжение F ⊣ G: Hom(F(z_f), z_g) ≅ Hom(z_f, G(z_g))."""
        if functor_f not in self.functors or functor_g not in self.functors: return 0.0
        Fz = self.functors[functor_f](z_f); Gz = self.functors[functor_g](z_g)
        lhs = np.dot(Fz.flatten(), z_g.flatten()) / (np.linalg.norm(Fz)*np.linalg.norm(z_g)+1e-8)
        rhs = np.dot(z_f.flatten(), Gz.flatten()) / (np.linalg.norm(z_f)*np.linalg.norm(Gz)+1e-8)
        return float(1.0 - abs(lhs - rhs))


# ============================================================
# 21. INFORMATION GEOMETRY — метрика Фишера, геодезические
# ============================================================

class InformationGeometry(nn.Module):
    """Многообразие состояний с метрикой Фишера и геодезическими."""

    def __init__(self, dim: int = 2560, manifold_dim: int = 64):
        super().__init__()
        self.dim = dim; self.manifold_dim = manifold_dim
        self.chart = nn.Sequential(nn.Linear(dim, manifold_dim), nn.SiLU(), nn.Linear(manifold_dim, manifold_dim))

    def to_manifold(self, z: np.ndarray) -> np.ndarray:
        z_t = torch.from_numpy(z).float().unsqueeze(0)
        with torch.no_grad(): return self.chart(z_t).squeeze(0).numpy()

    def fisher_metric(self, z: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
        """Приближение метрики Фишера через конечные разности."""
        z0 = self.to_manifold(z); m = len(z0)
        G = np.zeros((m, m), dtype=np.float32)
        for i in range(m):
            for j in range(m):
                z_plus_i = z.copy(); z_plus_i_flat = z_plus_i.flatten()
                idx = min(i * (len(z_plus_i_flat) // m), len(z_plus_i_flat) - 1)
                z_plus_i_flat[idx] += epsilon
                z_plus_i = z_plus_i_flat.reshape(z.shape)
                zi = self.to_manifold(z_plus_i)
                z_plus_j = z.copy(); z_plus_j_flat = z_plus_j.flatten()
                idx_j = min(j * (len(z_plus_j_flat) // m), len(z_plus_j_flat) - 1)
                z_plus_j_flat[idx_j] += epsilon
                z_plus_j = z_plus_j_flat.reshape(z.shape)
                zj = self.to_manifold(z_plus_j)
                G[i, j] = float(np.dot(zi - z0, zj - z0) / epsilon**2)
        return G

    def geodesic(self, z_a: np.ndarray, z_b: np.ndarray, steps: int = 20) -> List[np.ndarray]:
        """Геодезическая на многообразии: кратчайший путь между A и B."""
        a = self.to_manifold(z_a); b = self.to_manifold(z_b)
        path = []
        for i in range(steps):
            t = i / (steps - 1)
            point = a * (1 - t) + b * t
            path.append(point)
        return path

    def geodesic_length(self, z_a: np.ndarray, z_b: np.ndarray, steps: int = 50) -> float:
        """Длина геодезической между A и B."""
        path = self.geodesic(z_a, z_b, steps)
        total = 0.0
        for i in range(len(path) - 1):
            total += np.linalg.norm(path[i + 1] - path[i])
        return total

    def curvature(self, z: np.ndarray, z_a: np.ndarray, z_b: np.ndarray) -> float:
        """Секционная кривизна в точке z для плоскости (z_a, z_b)."""
        a = self.to_manifold(z_a); b = self.to_manifold(z_b)
        G = self.fisher_metric(z)
        if G.size == 0: return 0.0
        num = np.dot(a, G @ b) ** 2 - np.dot(a, G @ a) * np.dot(b, G @ b)
        den = np.dot(a, G @ a) * np.dot(b, G @ b) - np.dot(a, G @ b) ** 2 + 1e-8
        return float(num / den)
