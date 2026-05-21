"""
UnifiedStateGrammar — единый класс, объединяющий все 41 механизм.

41 механизм в одном API:
  compose()  — композиция состояний с полной цепочкой правил
  analyze()  — анализ последовательности состояний
  discover() — обнаружение правил из данных
  validate() — валидация правил на отложенных данных
  visualize() — минимальная текстовая визуализация
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import math, random, json, os
from loguru import logger


@dataclass
class CompositionResult:
    result: np.ndarray
    validity: float = 0.5; delta_I: float = 0.0
    tversky_similarity: float = 0.5; mutual_information: float = 0.0
    contradictory: bool = False; excluded_middle: float = 0.0
    self_ref_degree: float = 0.0; russell_paradox: bool = False
    causal_necessity: float = 0.0; epistemic: Dict = field(default_factory=dict)
    temporal_coherence: float = 0.0; resonance: float = 0.0
    frontier_intensity: float = 0.0; creativity: float = 0.0
    cultural_variants: List[np.ndarray] = field(default_factory=list)
    metaphor_strength: float = 0.0; ethical_profile: Dict = field(default_factory=dict)
    narrative_arc: str = ""; emotional_valence: float = 0.0
    counterfactual_creativity: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"CompositionResult:"]
        for k, v in self.__dict__.items():
            if k == 'result' or k == 'cultural_variants' or k == 'metadata':
                continue
            if isinstance(v, float): lines.append(f"  {k}: {v:.3f}")
            elif isinstance(v, bool): lines.append(f"  {k}: {v}")
            elif isinstance(v, dict): lines.append(f"  {k}: {len(v)} keys")
        return "\n".join(lines)


class UnifiedStateGrammar:
    """Единая грамматика состояний — все 41 механизм."""

    def __init__(self, dim: int = 2560):
        self.dim = dim
        self._init_core(); self._init_extended(); self._init_final(); self._init_deep()
        self.composition_counter: Dict[Tuple[str, str], int] = defaultdict(int)
        self.rule_registry: Dict[str, float] = {}
        self.emerged_concepts: Dict[str, np.ndarray] = {}
        logger.info(f"[UnifiedGrammar] {41} механизмов инициализировано (dim={dim})")

    def _init_core(self):
        from eva.state_grammar import (
            StateValenceV2, TemporalChainV2, NegationAlgebraV2,
            SuperpositionCollapseV2, CompositionalValidatorV2,
            StateInheritanceGraphV2, EmergentGenesisV2,
            TransformDistanceV2, ConservationLawsV2,
            SelfReferenceV2, InformationEntropyV2, ContextualComposerV2
        )
        self.valence = StateValenceV2(self.dim)
        self.temporal = TemporalChainV2(self.dim)
        self.negation = NegationAlgebraV2(self.dim)
        self.superposition = SuperpositionCollapseV2(self.dim)
        self.validator = CompositionalValidatorV2(self.dim)
        self.inheritance = StateInheritanceGraphV2()
        self.emergence = EmergentGenesisV2()
        self.distance = TransformDistanceV2()
        self.conservation = ConservationLawsV2()
        self.self_ref = SelfReferenceV2()
        self.info_entropy = InformationEntropyV2(self.dim)
        self.ctx_composer = ContextualComposerV2(self.dim)

    def _init_extended(self):
        from eva.state_grammar_ext import (
            CausalReasoning, TemporalModality, EpistemicStates,
            Quantification, StateResonance, FrontierStates,
            GradientFlow, TopologicalPersistence,
            CategoryTheory, InformationGeometry
        )
        self.causal = CausalReasoning(self.dim)
        self.temporal_mod = TemporalModality(self.dim)
        self.epistemic = EpistemicStates(self.dim)
        self.quantification = Quantification(self.dim)
        self.resonance = StateResonance(self.dim)
        self.frontier = FrontierStates(self.dim)
        self.gradient_flow = GradientFlow(self.dim)
        self.persistence = TopologicalPersistence()
        self.category = CategoryTheory(self.dim)
        self.info_geom = InformationGeometry(self.dim)

    def _init_final(self):
        from eva.state_grammar_final import (
            RecursiveSelfModification, DialecticalSynthesis, Abduction,
            AnalogicalMapping, ZeroShotComposition, FractalSelfConsistency,
            TeleologicalReasoning, NarrativeCoherence, EmotionalValence,
            CounterfactualImagination
        )
        self.meta_mod = RecursiveSelfModification(self.dim)
        self.dialectic = DialecticalSynthesis(self.dim)
        self.abduction = Abduction(self.dim)
        self.analogy = AnalogicalMapping(self.dim)
        self.zero_shot = ZeroShotComposition(self.dim)
        self.fractal_cons = FractalSelfConsistency()
        self.teleo = TeleologicalReasoning(self.dim)
        self.narrative = NarrativeCoherence(self.dim)
        self.emotion = EmotionalValence(self.dim)
        self.counterfactual = CounterfactualImagination(self.dim)

    def _init_deep(self):
        from eva.state_grammar_deep import (
            CulturalRelativity, DreamRecombination, EthicalCalculus,
            StateEconomy, EvolutionaryPressure, GameTheoretic,
            AttentionEconomy, MetaphorGeneration, RecursiveIntrospection,
            EntropySeeking
        )
        self.culture = CulturalRelativity(self.dim)
        self.dream = DreamRecombination(self.dim)
        self.ethics = EthicalCalculus(self.dim)
        self.economy = StateEconomy()
        self.evolution = EvolutionaryPressure(population_size=20)
        self.game_theory = GameTheoretic()
        self.attention = AttentionEconomy(self.dim)
        self.metaphor = MetaphorGeneration(self.dim)
        self.introspect = RecursiveIntrospection(self.dim)
        self.curiosity = EntropySeeking(self.dim)

    def compose(self, z_a: np.ndarray, z_b: np.ndarray,
                z_context: np.ndarray,
                label_a: str = "", label_b: str = "",
                alpha_a: float = 1.0, alpha_b: float = 1.0) -> CompositionResult:
        """Полная композиция через все механизмы."""

        za = torch.from_numpy(z_a).float().unsqueeze(0)
        zb = torch.from_numpy(z_b).float().unsqueeze(0)
        zc = torch.from_numpy(z_context).float().unsqueeze(0)

        za_mod, _ = self.valence(za, alpha_a)
        zb_mod, _ = self.valence(zb, alpha_b)

        valid_score, _ = self.validator.validate(z_a, z_b, z_context)
        validity = valid_score

        z_result_t = self.ctx_composer(za_mod, zb_mod, zc)
        result = z_result_t.squeeze(0).detach().numpy()

        result = CompositionResult(result=result, validity=validity)

        try:
            result.delta_I = self.info_entropy.composition_surprise(z_a, z_b, result.result)
            result.tversky_similarity = self.valence.tversky_similarity(za, zb)
            result.mutual_information = self.info_entropy.mutual_information(z_a, z_b)
            result.contradictory = self.negation.is_contradictory(za, zb)
            result.excluded_middle = self.negation.excluded_middle_satisfaction(za)
            result.self_ref_degree = self.self_ref.self_reference_degree(z_a, lambda x: x * 0.99)
            result.russell_paradox = self.self_ref.detect_russell_paradox(z_a,
                lambda a, b: float(F.cosine_similarity(torch.from_numpy(a).float(), torch.from_numpy(b).float(), dim=-1)))
            result.causal_necessity = self.causal.necessary_cause_score(z_a, z_b, z_context, z_a * 0.5)
            result.epistemic = self.epistemic.epistemic_profile(za, zb)
            result.temporal_coherence = self.temporal_mod.timeline_coherence(z_a * 0.8, z_a, z_a * 1.2)
            result.resonance = self.resonance.resonance_score(z_a, z_b)
            result.frontier_intensity, _ = self.frontier.frontier_score(result.result, z_a, z_b)
            result.metaphor_strength = self.metaphor.metaphor_strength(z_a, z_b)
            result.ethical_profile = self.ethics.deontological_profile(result.result)
            result.emotional_valence, _ = self.emotion.valence_arousal(result.result)
            result.creativity = self.dream.creative_potential(
                self.dream.dream_sequence(result.result, z_context, 3, 0.5))

            narratives = [z_a, z_b, result.result]
            result.narrative_arc = self.narrative.analyze_arc(narratives).get("arc_type", "none")

            cf_world = self.counterfactual.imagine(result.result, z_context)
            result.counterfactual_creativity = self.counterfactual.creativity_index(cf_world)

        except Exception as e:
            logger.debug(f"[Grammar] partial metrics error: {e}")

        if label_a and label_b:
            self.composition_counter[(label_a, label_b)] += 1
            self.emergence.record_composition(label_a, label_b, result.result)

        state_id = len(self.economy.costs)
        self.economy.register(state_id, result.result)
        self.curiosity.mark_known(result.result, 0.5)

        return result

    def analyze(self, state_sequence: List[np.ndarray],
                context: np.ndarray) -> Dict:
        """Полный анализ последовательности."""
        if len(state_sequence) < 2:
            return {"error": "need at least 2 states"}

        result = {}

        try:
            stacked = torch.from_numpy(np.stack(state_sequence)).float().unsqueeze(0)
            _ = self.temporal(stacked)
            result["coherence"] = float(1.0 / (1.0 + torch.norm(
                _[:, 1:] - stacked[:, 1:], dim=-1).mean()))
            result["entropy_rate"] = self.info_entropy.entropy_rate(state_sequence)
            result["transition_entropy"] = self.temporal.transition_entropy(stacked)
        except Exception:
            result["coherence"] = 0.0

        try:
            result["narrative"] = self.narrative.analyze_arc(state_sequence)
            result["emotional_arc"] = self.emotion.emotional_arc(state_sequence)
        except Exception:
            pass

        try:
            result["persistence"] = self.persistence.persistence_diagram(state_sequence)
        except Exception:
            pass

        try:
            result["fractal_dim"] = self.fractal_cons.fractal_dimension(
                np.mean(state_sequence, axis=0))
        except Exception:
            pass

        return result

    def discover(self, training_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                  epochs: int = 100, lr: float = 1e-4) -> Dict[str, float]:
        """Обнаруживает закономерности композиции из данных."""
        params = list(self.ctx_composer.parameters()) + list(self.validator.parameters())

        valid_pairs = [data for data in training_data[:100] if data]
        invalid_pairs = []

        for i in range(min(len(valid_pairs), 50)):
            a, b, _ = valid_pairs[i % len(valid_pairs)]
            noise_a = np.random.randn(*a.shape).astype(np.float32) * 0.05
            noise_b = np.random.randn(*b.shape).astype(np.float32) * 0.05
            invalid_pairs.append((noise_a, noise_b, np.zeros_like(a)))

        optimizer = torch.optim.Adam(params, lr=lr)
        history = []

        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(min(len(valid_pairs), len(invalid_pairs), 32)):
                optimizer.zero_grad()

                a_v, b_v, c_v = valid_pairs[i]
                za = torch.from_numpy(a_v).float().unsqueeze(0)
                zb = torch.from_numpy(b_v).float().unsqueeze(0)
                zc = torch.from_numpy(c_v).float().unsqueeze(0)

                predicted = self.ctx_composer(za, zb, zc)
                loss_mse = F.mse_loss(predicted, zc)

                a_i, b_i, _ = invalid_pairs[i]
                zai = torch.from_numpy(a_i).float().unsqueeze(0)
                zbi = torch.from_numpy(b_i).float().unsqueeze(0)
                zci = torch.from_numpy(c_v).float().unsqueeze(0)

                raw_logits = self.validator(
                    torch.cat([zai, zbi, zci], dim=-1)
                )
                loss_val = F.binary_cross_entropy_with_logits(
                    raw_logits,
                    torch.zeros_like(raw_logits))

                loss = loss_mse + 0.1 * loss_val
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            history.append(total_loss)
            if epoch % 20 == 0:
                logger.info(f"[Discover] epoch={epoch}, loss={total_loss:.4f}")

        self.rule_registry["discovery_loss"] = history[-1] if history else 0.0
        self.rule_registry["discovery_epochs"] = epochs
        return self.rule_registry

    def validate_rules(self, test_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Dict:
        """Валидация правил на отложенных данных."""
        if not test_data: return {"error": "no test data"}

        mse_grammar = 0.0; mse_baseline = 0.0; n = 0

        for a, b, c in test_data[:50]:
            za = torch.from_numpy(a).float().unsqueeze(0)
            zb = torch.from_numpy(b).float().unsqueeze(0)
            zc = torch.from_numpy(c).float().unsqueeze(0)

            with torch.no_grad():
                predicted = self.ctx_composer(za, zb, zc)
                mse_grammar += F.mse_loss(predicted, zc).item()

            baseline = za + zb
            mse_baseline += F.mse_loss(baseline, zc).item()
            n += 1

        mse_grammar /= max(n, 1); mse_baseline /= max(n, 1)
        improvement = (mse_baseline - mse_grammar) / (mse_baseline + 1e-8)

        logger.info(f"[Validate] MSE grammar={mse_grammar:.4f}, "
                    f"baseline={mse_baseline:.4f}, improvement={improvement:.2%}")

        return {"mse_grammar": mse_grammar, "mse_baseline": mse_baseline,
                "improvement": improvement, "samples": n,
                "better_than_baseline": mse_grammar < mse_baseline}

    def visualize(self) -> str:
        """Минимальная текстовая визуализация системы."""
        lines = ["=" * 60, "UnifiedStateGrammar — 41 механизмов", "=" * 60]
        lines.append(f"  Dimension: {self.dim}")
        lines.append(f"  Compositions recorded: {sum(self.composition_counter.values())}")
        lines.append(f"  Emerged concepts: {len(self.emergence.emergent)}")
        lines.append(f"  Economy states: {len(self.economy.costs)}")
        lines.append(f"  Rule registry: {len(self.rule_registry)} entries")

        if self.emergence.emergent:
            lines.append(f"  Discovered concepts:")
            for cid, c in list(self.emergence.emergent.items())[:5]:
                lines.append(f"    {cid}: count={c.composition_count}, stability={c.stability:.3f}")

        if self.rule_registry:
            lines.append(f"  Rule metrics:")
            for k, v in list(self.rule_registry.items())[:5]:
                lines.append(f"    {k}: {v}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def save(self, path: str):
        """Сохранить состояние грамматики."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "dim": self.dim,
            "composition_counter": dict(self.composition_counter),
            "rule_registry": self.rule_registry,
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"[Grammar] Saved: {path}")
        except Exception as e:
            logger.warning(f"[Grammar] Save error: {e}")

    @classmethod
    def load(cls, dim: int = 2560, path: str = None) -> "UnifiedStateGrammar":
        grammar = cls(dim)
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                grammar.composition_counter = defaultdict(int, data.get("composition_counter", {}))
                grammar.rule_registry = data.get("rule_registry", {})
            except Exception:
                pass
        return grammar
