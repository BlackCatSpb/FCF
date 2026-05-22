"""
StateGrammar — финальные 10 механизмов (общее число: 31).

22. RecursiveSelfModification — система модифицирует свои правила (meta-learning)
23. DialecticalSynthesis    — thesis ⊕ antithesis → synthesis (Hegel engine)
24. Abduction               — effect + rules → most probable cause (Peirce)
25. AnalogicalMapping       — A:B :: C:D structure-mapping theory
26. ZeroShotComposition     — композиция невиданных состояний из правил
27. FractalSelfConsistency  — масштабная инвариантность правил
28. TeleologicalReasoning   — purpose-driven transformation (зачем)
29. NarrativeCoherence      — сюжетная арка, tension→climax→resolution
30. EmotionalValence        — эмоциональный заряд состояний
31. CounterfactualImagination — рекурсивные альтернативные миры
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import math, random
from loguru import logger


# ============================================================
# 22. RECURSIVE SELF-MODIFICATION — meta-learning правил
# ============================================================

class RecursiveSelfModification(nn.Module):
    """Система учится модифицировать СВОИ правила композиции. Meta-learning."""

    def __init__(self, dim: int = 2560, num_rules: int = 9):
        super().__init__()
        self.dim = dim; self.num_rules = num_rules
        self.rule_encoder = nn.Sequential(nn.Linear(dim * 3, 256), nn.SiLU(), nn.Linear(256, 128))
        self.modification_head = nn.Sequential(nn.Linear(128, num_rules * 3), nn.SiLU(), nn.Linear(num_rules * 3, num_rules * 2))
        self.meta_confidence = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.modification_history: List[Dict] = []

    def propose_modification(self, z_a: np.ndarray, z_b: np.ndarray,
                             z_result: np.ndarray, expected: np.ndarray) -> Dict:
        """Предлагает модификацию правил на основе ошибки композиции."""
        za = torch.from_numpy(z_a).float().unsqueeze(0); zb = torch.from_numpy(z_b).float().unsqueeze(0)
        zr = torch.from_numpy(z_result).float().unsqueeze(0); ze = torch.from_numpy(expected).float().unsqueeze(0)
        error = zr - ze
        encoded = self.rule_encoder(torch.cat([za, zb, error], dim=-1))
        mods = self.modification_head(encoded).view(-1, self.num_rules, 2)
        confidence = float(self.meta_confidence(encoded))

        proposals = []
        for i in range(self.num_rules):
            proposals.append({"rule_idx": i, "scale": float(mods[0, i, 0]), "bias": float(mods[0, i, 1])})

        self.modification_history.append({"confidence": confidence, "error_norm": float(torch.norm(error))})
        return {"confidence": confidence, "proposals": proposals, "error_norm": float(torch.norm(error))}

    def should_modify(self) -> bool:
        """Решение: нужно ли модифицировать правила."""
        if len(self.modification_history) < 10: return False
        recent_errors = [h["error_norm"] for h in self.modification_history[-10:]]
        return np.mean(recent_errors) > 0.5 and np.std(recent_errors) < 0.1

    def modification_impact(self, before_error: float, after_error: float) -> str:
        improvement = (before_error - after_error) / (before_error + 1e-8)
        if improvement > 0.5: return "BREAKTHROUGH"
        if improvement > 0.1: return "improved"
        if improvement > -0.1: return "neutral"
        return "degraded"


# ============================================================
# 23. DIALECTICAL SYNTHESIS — thesis ⊕ antithesis → synthesis
# ============================================================

class DialecticalSynthesis(nn.Module):
    """Hegel engine: разрешение противоречия в новом качестве."""

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.synthesis_net = nn.Sequential(nn.Linear(dim * 3, dim * 2), nn.SiLU(),
                                           nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.sublation_net = nn.Sequential(nn.Linear(dim * 2, 128), nn.SiLU(), nn.Linear(128, 3))
        self.contradiction_detector = nn.Sequential(nn.Linear(dim * 2, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid())

    def synthesize(self, thesis: np.ndarray, antithesis: np.ndarray,
                   context: np.ndarray) -> np.ndarray:
        """Synthesis = Aufheben(thesis, antithesis) — снятие противоречия."""
        th = torch.from_numpy(thesis).float().unsqueeze(0); an = torch.from_numpy(antithesis).float().unsqueeze(0)
        ctx = torch.from_numpy(context).float().unsqueeze(0)
        with torch.no_grad():
            return self.synthesis_net(torch.cat([th, an, ctx], dim=-1)).squeeze(0).numpy()

    def sublation_analysis(self, z_synthesis: np.ndarray, z_thesis: np.ndarray,
                           z_antithesis: np.ndarray) -> Dict[str, float]:
        zs = torch.from_numpy(z_synthesis).float().unsqueeze(0)
        zt = torch.from_numpy(z_thesis).float().unsqueeze(0)
        za = torch.from_numpy(z_antithesis).float().unsqueeze(0)
        diff = torch.cat([zs - zt, zs - za], dim=-1)
        with torch.no_grad():
            out = self.sublation_net(diff)
            probs = torch.softmax(out, dim=-1)
        return {"preserved": float(probs[0, 0]), "negated": float(probs[0, 1]), "transcended": float(probs[0, 2])}

    def contradiction_intensity(self, thesis: np.ndarray, antithesis: np.ndarray) -> float:
        th = torch.from_numpy(thesis).float().unsqueeze(0); an = torch.from_numpy(antithesis).float().unsqueeze(0)
        with torch.no_grad(): return float(self.contradiction_detector(torch.cat([th, an], dim=-1)))

    def dialectical_progress(self, thesis: np.ndarray, antithesis: np.ndarray,
                             synthesis: np.ndarray) -> bool:
        """Оценивает: является ли synthesis прогрессом относительно thesis/antithesis."""
        ci = self.contradiction_intensity(thesis, antithesis)
        syn_ci = self.contradiction_intensity(synthesis, synthesis)
        return syn_ci < ci * 0.5


# ============================================================
# 24. ABDUCTION — effect + rules → most probable cause
# ============================================================

class Abduction(nn.Module):
    """Peirce: deduction + induction + abduction = полное мышление."""

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.cause_generator = nn.Sequential(nn.Linear(dim * 2, dim * 2), nn.SiLU(),
                                             nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.plausibility = nn.Sequential(nn.Linear(dim * 3, 128), nn.SiLU(), nn.Linear(128, 1), nn.Sigmoid())

    def abduce(self, effect: np.ndarray, rule_context: np.ndarray,
               num_candidates: int = 5) -> List[Tuple[np.ndarray, float]]:
        """Генерирует наиболее вероятные причины для данного эффекта."""
        ef = torch.from_numpy(effect).float().unsqueeze(0).expand(num_candidates, -1)
        rc = torch.from_numpy(rule_context).float().unsqueeze(0).expand(num_candidates, -1)
        noise = torch.randn(num_candidates, self.cause_generator[-1].out_features) * 0.1
        with torch.no_grad():
            causes = self.cause_generator(torch.cat([ef, rc], dim=-1)) + noise

        candidates = []
        for i in range(num_candidates):
            cause = causes[i:i+1]
            plaus = float(self.plausibility(torch.cat([cause, ef[:1], rc[:1]], dim=-1)))
            candidates.append((cause.squeeze(0).numpy(), plaus))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def best_explanation(self, effect: np.ndarray, rule_context: np.ndarray,
                         candidate_causes: List[np.ndarray]) -> int:
        """Выбирает лучшее объяснение среди кандидатов."""
        scores = []
        for cause in candidate_causes:
            c = torch.from_numpy(cause).float().unsqueeze(0); ef = torch.from_numpy(effect).float().unsqueeze(0)
            rc = torch.from_numpy(rule_context).float().unsqueeze(0)
            with torch.no_grad(): scores.append(float(self.plausibility(torch.cat([c, ef, rc], dim=-1))))
        return int(np.argmax(scores))

    def explanatory_power(self, cause: np.ndarray, effect: np.ndarray,
                          alternative_effects: List[np.ndarray]) -> float:
        """Насколько причина объясняет именно этот эффект, а не альтернативные."""
        c = torch.from_numpy(cause).float().unsqueeze(0); ef = torch.from_numpy(effect).float().unsqueeze(0)
        with torch.no_grad():
            p_effect = float(self.plausibility(torch.cat([c, ef, ef], dim=-1)))
            p_alts = []
            for alt in alternative_effects[:5]:
                a = torch.from_numpy(alt).float().unsqueeze(0)
                p_alts.append(float(self.plausibility(torch.cat([c, ef, a], dim=-1))))
            return p_effect / (np.mean(p_alts) + 1e-8) if p_alts else p_effect


# ============================================================
# 25. ANALOGICAL MAPPING — A:B :: C:D
# ============================================================

class AnalogicalMapping(nn.Module):
    """Structure-mapping theory: analog(A, B, C, D)."""

    def __init__(self, dim: int = 2560):
        super().__init__()
        rel_dim = dim // 2
        self.relation_encoder = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, rel_dim))
        self.analogy_scorer = nn.Sequential(nn.Linear(rel_dim, 128), nn.SiLU(), nn.Linear(128, 1), nn.Sigmoid())
        self.mapping_net = nn.Sequential(nn.Linear(dim * 3, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def score_analogy(self, z_a: np.ndarray, z_b: np.ndarray,
                      z_c: np.ndarray, z_d: np.ndarray) -> float:
        """Оценивает: насколько отношение A:B аналогично C:D."""
        za = torch.from_numpy(z_a).float().unsqueeze(0); zb = torch.from_numpy(z_b).float().unsqueeze(0)
        zc = torch.from_numpy(z_c).float().unsqueeze(0); zd = torch.from_numpy(z_d).float().unsqueeze(0)
        rel_ab = self.relation_encoder(torch.cat([za, zb], dim=-1))
        rel_cd = self.relation_encoder(torch.cat([zc, zd], dim=-1))
        with torch.no_grad(): return float(self.analogy_scorer(rel_ab - rel_cd))

    def complete_analogy(self, z_a: np.ndarray, z_b: np.ndarray,
                         z_c: np.ndarray) -> np.ndarray:
        """A:B :: C:? — найти D."""
        za = torch.from_numpy(z_a).float().unsqueeze(0); zb = torch.from_numpy(z_b).float().unsqueeze(0)
        zc = torch.from_numpy(z_c).float().unsqueeze(0)
        with torch.no_grad():
            return self.mapping_net(torch.cat([za, zb - za, zc], dim=-1)).squeeze(0).numpy()

    def analogy_matrix(self, pairs: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Матрица аналогий: для каждой пары пар — насколько аналогичны отношения."""
        n = len(pairs); M = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                M[i, j] = self.score_analogy(pairs[i][0], pairs[i][1], pairs[j][0], pairs[j][1])
        return M

    def systematicity_score(self, mappings: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> float:
        """Systematicity principle: хорошая аналогия сохраняет систему отношений."""
        if len(mappings) < 2: return 1.0
        scores = [self.score_analogy(*m) for m in mappings]
        return float(np.mean(scores))


# ============================================================
# 26. ZERO-SHOT COMPOSITION — невиданные комбинации из правил
# ============================================================

class ZeroShotComposition(nn.Module):
    """Композиция состояний, не встречавшихся вместе — только из правил."""

    def __init__(self, dim: int = 2560, num_rules: int = 9):
        super().__init__()
        self.dim = dim; self.num_rules = num_rules
        self.rule_selector = nn.Sequential(nn.Linear(dim * 2, 256), nn.SiLU(), nn.Linear(256, num_rules))
        self.composer = nn.Sequential(nn.Linear(dim * 3, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def compose_unseen(self, z_a: np.ndarray, z_b: np.ndarray,
                       context: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Композиция полностью новых состояний — выбор правил по типам."""
        za = torch.from_numpy(z_a).float().unsqueeze(0); zb = torch.from_numpy(z_b).float().unsqueeze(0)
        zc = torch.from_numpy(context).float().unsqueeze(0)
        with torch.no_grad():
            rule_weights = torch.softmax(self.rule_selector(torch.cat([za, zb], dim=-1)), dim=-1)
            selected = torch.topk(rule_weights, 3, dim=-1).indices[0].tolist()
            result = self.composer(torch.cat([za, zb, zc], dim=-1))
        return result.squeeze(0).numpy(), selected

    def generalization_gap(self, z_result: np.ndarray, z_expected: np.ndarray) -> float:
        """Мера 'неожиданности' zero-shot композиции."""
        return float(np.linalg.norm(z_result - z_expected) / (np.linalg.norm(z_expected) + 1e-8))

    def composition_surprise(self, z_a: np.ndarray, z_b: np.ndarray,
                             z_result: np.ndarray) -> float:
        """Насколько результат отличается от простой суммы компонентов."""
        simple = z_a + z_b
        return float(np.linalg.norm(z_result - simple) / (np.linalg.norm(simple) + 1e-8))


# ============================================================
# 27. FRACTAL SELF-CONSISTENCY — масштабная инвариантность
# ============================================================

class FractalSelfConsistency:
    """Правила композиции на уровне слов должны иметь аналог на уровне текстов."""

    def __init__(self):
        self.scale_invariants: Dict[str, List[float]] = defaultdict(list)

    def check_scale_invariance(self, rule_name: str,
                               low_level_result: np.ndarray,
                               high_level_result: np.ndarray,
                               upscale_fn: Callable,
                               downscale_fn: Callable) -> float:
        """Проверяет: upscale(rule(small)) ≈ rule(upscale(small))."""
        commutator = np.linalg.norm(upscale_fn(low_level_result) - high_level_result)
        score = float(1.0 / (1.0 + commutator))
        self.scale_invariants[rule_name].append(score)
        return score

    def fractal_dimension(self, z: np.ndarray) -> float:
        """Оценивает фрактальную размерность состояния через box-counting."""
        z_flat = z.flatten()
        scales = [2, 4, 8, 16, 32]
        counts = []
        for s in scales:
            chunks = np.array_split(z_flat, max(1, len(z_flat) // s))
            count = sum(1 for c in chunks if np.std(c) > 0.01)
            counts.append(count)
        if len(counts) < 2: return 1.0
        log_scales = np.log(np.array(scales[:len(counts)], dtype=np.float32) + 1e-8)
        log_counts = np.log(np.array(counts, dtype=np.float32) + 1e-8)
        slope = np.polyfit(log_scales, log_counts, 1)[0]
        return float(abs(slope))

    def scale_recurrence(self, z: np.ndarray, scale_factor: float = 0.5) -> float:
        """Насколько состояние самоподобно при масштабировании."""
        try:
            z_flat = z.flatten(); n = len(z_flat)
            if n < 4: return 0.0
            small_n = max(2, int(n * scale_factor))
            indices_small = np.linspace(0, n - 1, small_n, dtype=int)
            small = z_flat[indices_small]
            indices_big = np.linspace(0, small_n - 1, n, dtype=int)
            scaled_up = small[indices_big]
            return float(np.corrcoef(z_flat, scaled_up)[0, 1])
        except Exception:
            return 0.0


# ============================================================
# 28. TELEOLOGICAL REASONING — purpose-driven
# ============================================================

class TeleologicalReasoning(nn.Module):
    """Не 'как', а 'ЗАЧЕМ'. Целеполагание состояний."""

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.purpose_enc = nn.Sequential(nn.Linear(dim, dim // 2), nn.SiLU(), nn.Linear(dim // 2, dim // 4))
        self.purpose_dec = nn.Sequential(nn.Linear(dim + dim // 4, dim // 2), nn.SiLU(), nn.Linear(dim // 2, dim))
        self.purposefulness = nn.Sequential(nn.Linear(dim * 2, 128), nn.SiLU(), nn.Linear(128, 1), nn.Sigmoid())

    def extract_purpose(self, z_state: np.ndarray) -> np.ndarray:
        zt = torch.from_numpy(z_state).float().unsqueeze(0)
        with torch.no_grad(): return self.purpose_enc(zt).squeeze(0).numpy()

    def purpose_driven_transform(self, z_current: np.ndarray,
                                  z_purpose: np.ndarray) -> np.ndarray:
        zc = torch.from_numpy(z_current).float().unsqueeze(0)
        zp = torch.from_numpy(z_purpose).float().unsqueeze(0)
        with torch.no_grad():
            return self.purpose_dec(torch.cat([zc, zp], dim=-1)).squeeze(0).numpy()

    def purpose_alignment(self, z_state: np.ndarray, z_goal: np.ndarray) -> float:
        zs = torch.from_numpy(z_state).float().unsqueeze(0); zg = torch.from_numpy(z_goal).float().unsqueeze(0)
        with torch.no_grad(): return float(self.purposefulness(torch.cat([zs, zg], dim=-1)))

    def teleological_distance(self, z_current: np.ndarray, z_goal: np.ndarray) -> float:
        pc = self.extract_purpose(z_current); pg = self.extract_purpose(z_goal)
        return float(np.linalg.norm(pc - pg))


# ============================================================
# 29. NARRATIVE COHERENCE — сюжетная арка
# ============================================================

class NarrativeCoherence(nn.Module):
    """Состояния образуют истории: tension→climax→resolution."""

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.narrative_encoder = nn.GRU(dim, 128, 1, batch_first=True, bidirectional=True)
        self.arc_classifier = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 5))
        self.coherence_scorer = nn.Sequential(nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid())

    def analyze_arc(self, state_sequence: List[np.ndarray]) -> Dict[str, float]:
        """Определяет тип сюжетной арки."""
        if len(state_sequence) < 2: return {"coherence": 0.0, "arc_type": "none"}
        x = torch.from_numpy(np.stack(state_sequence)).float().unsqueeze(0)
        with torch.no_grad():
            _, h = self.narrative_encoder(x)
            h_combined = torch.cat([h[0], h[1]], dim=-1)
            arc_logits = self.arc_classifier(h_combined)
            arc_probs = torch.softmax(arc_logits, dim=-1)
            coherence = float(self.coherence_scorer(h_combined))
        arc_types = ["rags_to_riches", "tragedy", "man_in_hole", "icarus", "oedipus"]
        best_arc = arc_types[int(torch.argmax(arc_probs))]
        return {"coherence": coherence, "arc_type": best_arc,
                "arc_probs": {arc_types[i]: float(arc_probs[0, i]) for i in range(5)}}

    def tension_curve(self, state_sequence: List[np.ndarray]) -> List[float]:
        """Кривая напряжения вдоль последовательности."""
        if len(state_sequence) < 2: return []
        curve = [0.0]
        for i in range(1, len(state_sequence)):
            diff = np.linalg.norm(state_sequence[i] - state_sequence[i - 1])
            cum = np.linalg.norm(state_sequence[i] - state_sequence[0])
            curve.append(float(diff / (cum + 1e-8) * (1 + i / len(state_sequence))))
        return curve

    def climax_detection(self, state_sequence: List[np.ndarray]) -> int:
        """Находит точку кульминации."""
        curve = self.tension_curve(state_sequence)
        if not curve: return 0
        return int(np.argmax(curve))

    def resolution_quality(self, before_climax: List[np.ndarray],
                           after_climax: List[np.ndarray]) -> float:
        """Качество разрешения: tension снизился?"""
        if not before_climax or not after_climax: return 0.5
        tension_before = np.linalg.norm(before_climax[-1] - before_climax[0])
        tension_after = np.linalg.norm(after_climax[-1] - after_climax[0])
        return float(1.0 / (1.0 + abs(tension_before - tension_after)))


# ============================================================
# 30. EMOTIONAL VALENCE — эмоциональный заряд
# ============================================================

class EmotionalValence(nn.Module):
    """Состояния несут эмоциональный заряд: красота, тревога, радость."""

    def __init__(self, dim: int = 2560):
        super().__init__()
        self.emotion_encoder = nn.Sequential(nn.Linear(dim, 128), nn.SiLU(), nn.Linear(128, 8))
        self.valence_scorer = nn.Sequential(nn.Linear(dim, 64), nn.SiLU(), nn.Linear(64, 1), nn.Tanh())
        self.arousal_scorer = nn.Sequential(nn.Linear(dim, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid())

    EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]

    def emotion_profile(self, z: np.ndarray) -> Dict[str, float]:
        """Восьмимерный эмоциональный профиль состояния."""
        zt = torch.from_numpy(z).float().unsqueeze(0)
        with torch.no_grad():
            logits = self.emotion_encoder(zt)
            probs = torch.softmax(logits, dim=-1)
        return {self.EMOTIONS[i]: float(probs[0, i]) for i in range(8)}

    def valence_arousal(self, z: np.ndarray) -> Tuple[float, float]:
        """Двумерная модель: валентность (±) × возбуждение (0–1)."""
        zt = torch.from_numpy(z).float().unsqueeze(0)
        with torch.no_grad():
            v = float(self.valence_scorer(zt)); a = float(self.arousal_scorer(zt))
        return v, a

    def emotional_resonance(self, z_a: np.ndarray, z_b: np.ndarray) -> float:
        """Насколько эмоции двух состояний совпадают."""
        prof_a = self.emotion_profile(z_a); prof_b = self.emotion_profile(z_b)
        a_vec = np.array([prof_a[e] for e in self.EMOTIONS])
        b_vec = np.array([prof_b[e] for e in self.EMOTIONS])
        return float(np.dot(a_vec, b_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(b_vec) + 1e-8))

    def emotional_arc(self, state_sequence: List[np.ndarray]) -> List[float]:
        """Эмоциональная дуга последовательности."""
        return [self.valence_arousal(s)[0] for s in state_sequence]


# ============================================================
# 31. COUNTERFACTUAL IMAGINATION — рекурсивные альтернативные миры
# ============================================================

class CounterfactualImagination(nn.Module):
    """Что если X, тогда Y, тогда Z... — рекурсивное построение миров."""

    def __init__(self, dim: int = 2560, max_depth: int = 5):
        super().__init__()
        self.dim = dim; self.max_depth = max_depth
        self.world_generator = nn.Sequential(nn.Linear(dim * 2, dim * 2), nn.SiLU(),
                                             nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.world_evaluator = nn.Sequential(nn.Linear(dim, 128), nn.SiLU(), nn.Linear(128, 2))
        self.imagination_tree: Dict[int, List[Dict]] = {}

    def imagine(self, z_start: np.ndarray, intervention: np.ndarray,
                depth: int = 0, world_id: int = 0) -> Dict:
        """Рекурсивно строит альтернативный мир."""
        if depth >= self.max_depth: return {"state": z_start, "children": [], "depth": depth}

        zs = torch.from_numpy(z_start).float().unsqueeze(0); zi = torch.from_numpy(intervention).float().unsqueeze(0)
        with torch.no_grad():
            z_next_t = self.world_generator(torch.cat([zs, zi], dim=-1))
            eval_out = self.world_evaluator(z_next_t)
            plausibility = float(torch.sigmoid(eval_out[0, 0]))
            interestingness = float(torch.sigmoid(eval_out[0, 1]))
            z_next = z_next_t.squeeze(0).numpy()

        children = []
        if interestingness > 0.5 and depth < self.max_depth - 1:
            for branch in range(2):
                mutated_intervention = intervention + np.random.randn(*intervention.shape).astype(np.float32) * 0.1
                child = self.imagine(z_next, mutated_intervention, depth + 1, world_id * 10 + branch)
                children.append(child)

        return {"state": z_next, "children": children, "depth": depth,
                "plausibility": plausibility, "interestingness": interestingness}

    def alternative_worlds(self, z_start: np.ndarray, interventions: List[np.ndarray]) -> List[Dict]:
        """Строит параллельные миры для разных интервенций."""
        worlds = []
        for i, interv in enumerate(interventions):
            world = self.imagine(z_start, interv, 0, i)
            worlds.append(world)
        return worlds

    def best_world(self, worlds: List[Dict]) -> int:
        """Выбирает лучший альтернативный мир по interestingness."""
        def score(w): return w.get("interestingness", 0.0) - w.get("depth", 0) * 0.1
        scores = [score(w) for w in worlds]
        return int(np.argmax(scores))

    def creativity_index(self, world: Dict) -> float:
        """Мера креативности: высокая interestingness × низкая plausibility."""
        p = world.get("plausibility", 0.5); i = world.get("interestingness", 0.5)
        return i * (1.0 - p)
