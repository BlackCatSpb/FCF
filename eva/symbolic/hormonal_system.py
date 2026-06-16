"""Hormonal (neuro-modulatory) system for intrinsic motivation.

Each hormone modulates a different aspect of learning and generation:
  - Dopamine (DA): reward prediction error -> STDP strength, confidence
  - Serotonin (5HT): punishment / aversion -> risk (temperature), caution
  - Noradrenaline (NA): uncertainty / novelty -> attention (beam focus)
  - Acetylcholine (ACh): plasticity gate -> learning rate, new pattern formation

The system drives self-improvement: the model intrinsically seeks to
maximize prediction accuracy, novelty, and coherence via hormonal feedback.
"""

import math
import numpy as np


class HormonalSystem:
    """Neuro-modulatory system for concept generation.

    Hormone levels (0..1) are updated each generation step based on:
    - Prediction confidence (how certain was the model?)
    - Target match (did we predict correctly?)
    - Novelty (is this a new pattern?)
    - Surprise (how unexpected was this outcome?)
    - Coherence (how smooth was the concept transition?)
    """

    def __init__(self):
        # Baselines (tonic levels)
        self.dopamine = 0.5      # reward (DA)
        self.serotonin = 0.5     # aversion (5HT)
        self.noradrenaline = 0.3 # arousal (NA)
        self.acetylcholine = 0.5 # plasticity (ACh)

        # Phasic signals (spikes)
        self.da_phasic = 0.0
        self.ach_phasic = 0.0

        # Running stats
        self.step = 0
        self.recent_confidences = []
        self.recent_matches = []
        self.reward_history = []

        # Decay rates
        self.tonic_decay = 0.95    # hormone drift toward baseline
        self.phasic_decay = 0.7    # phasic signal decay per step

        # Dynamic state (initialized here instead of via setattr)
        self._prev_avg_match = 0.0
        self._repetition_counter = 0
        self._last_few_cids = []

    def update(self, confidence=0.5, is_match=False, novelty=0.0,
               surprise=0.0, expected_cid=None, gen_cid=None):
        """Update hormone levels based on generation event.

        Args:
            confidence: prediction confidence (0..1)
            is_match: whether generated concept matched target
            novelty: how novel is this transition (0..1)
            surprise: how surprising was the outcome (0..1)
            expected_cid: target concept ID (or None)
            gen_cid: generated concept ID
        """
        self.step += 1

        # Track recent stats
        self.recent_confidences.append(confidence)
        self.recent_matches.append(1.0 if is_match else 0.0)
        if len(self.recent_confidences) > 50:
            self.recent_confidences.pop(0)
            self.recent_matches.pop(0)

        avg_confidence = np.mean(self.recent_confidences) if self.recent_confidences else 0.5
        avg_match = np.mean(self.recent_matches) if self.recent_matches else 0.0

        # Track match rate change (for mastery drive)
        delta_match = avg_match - self._prev_avg_match
        self._prev_avg_match = avg_match

        # ---- Dopamine: reward signal (phasic) ----
        # Three sources of reward:
        # 1. Extrinsic: target match (supervised)
        # 2. Intrinsic: curiosity (novelty)
        # 3. Intrinsic: mastery (improving match rate)
        # 4. Baseline: coherence (smooth generation)

        da_extrinsic = 0.0
        da_curiosity = 0.0
        da_mastery = 0.0
        da_coherence = 0.05  # small baseline for trying

        if expected_cid is not None:
            # Extrinsic: reward prediction error
            if is_match:
                da_extrinsic = max(0.5, 1.0 - confidence)  # more reward for hard-fought matches
            else:
                da_extrinsic = -0.3 * (1.0 + confidence)  # punishment for misses
        else:
            # Curiosity: novel patterns explored
            da_curiosity = novelty * 0.4

        # Mastery: improving match rate
        da_mastery = max(0, delta_match) * 0.5

        # Boredom penalty: repeating same cid multiple times
        if gen_cid is not None:
            self._last_few_cids.append(gen_cid)
            if len(self._last_few_cids) > 5:
                self._last_few_cids.pop(0)
            if len(self._last_few_cids) >= 3 and len(set(self._last_few_cids)) == 1:
                da_coherence -= 0.1  # boredom from repetition

        intrinsic = da_curiosity + da_mastery + da_coherence
        self.da_phasic = da_extrinsic + intrinsic

        # ---- Acetylcholine: phasic surprise/novelty signal ----
        # Phasic ACh responds to prediction errors and novel stimuli,
        # signaling 'this is important, learn from it'
        ach_surprise = 0.0
        ach_novelty = 0.0
        ach_pe = 0.0

        if expected_cid is not None:
            # Supervised mode: prediction error drives ACh
            if not is_match:
                ach_surprise = surprise * 0.6       # unexpected outcome
                ach_pe = (1.0 - confidence) * 0.5   # low confidence → high uncertainty
            else:
                ach_surprise = surprise * 0.15       # even matched outcomes carry surprise
        else:
            # Free generation: novelty-driven ACh
            ach_novelty = novelty * 0.5              # novel transitions → learn

        self.ach_phasic = ach_surprise + ach_novelty + ach_pe
        self.ach_phasic = max(0.0, min(1.0, self.ach_phasic))  # clamp to [0,1]

        # ---- Serotonin: aversion / risk ----
        # Low match rate -> serotonin rises (aversion, caution)
        # High match rate -> serotonin drops (safety, exploration)
        target_5ht = 0.3 + 0.4 * (1.0 - avg_match)
        self.serotonin += (target_5ht - self.serotonin) * 0.1

        # ---- Noradrenaline: uncertainty / novelty ----
        # High surprise or low confidence -> NA rises (focus)
        # High confidence + low novelty -> NA drops (relaxed)
        target_na = 0.2 + 0.5 * surprise + 0.3 * (1.0 - confidence)
        self.noradrenaline += (target_na - self.noradrenaline) * 0.3
        self.noradrenaline = min(max(self.noradrenaline, 0.0), 1.0)

        # ---- Acetylcholine: plasticity gate ----
        novelty_target = 0.3 + 0.5 * novelty
        if is_match and confidence > 0.8:
            novelty_target = 0.2  # well-known pattern → low plasticity

        # Drift tonic toward target
        self.acetylcholine += (novelty_target - self.acetylcholine) * 0.15
        # Integrate phasic ACh into tonic (mirrors DA phasic integration)
        self.acetylcholine += self.ach_phasic * 0.1
        self.acetylcholine = max(0.1, min(1.0, self.acetylcholine))

        # ---- Integrate phasic into tonic BEFORE decay ----
        # Floor at 0.1 so model doesn't get stuck in anhedonia
        new_da = self.dopamine * self.tonic_decay + self.da_phasic * 0.1
        self.dopamine = max(0.1, min(1.0, new_da))

        # ---- Decay phasic signals (after integration) ----
        self.da_phasic *= self.phasic_decay
        self.ach_phasic *= self.phasic_decay

        # Track reward
        self.reward_history.append(self.dopamine)

    # ---- Modulation functions ----

    def modulate_stdp_lr(self, base_lr):
        """Acetylcholine gates plasticity. Dopamine modulates STDP magnitude.
        High ACh + High DA -> strong learning (novel correct pattern).
        Low ACh -> weak learning (consolidation, familiar)."""
        plasticity = self.acetylcholine * (0.5 + 0.5 * self.dopamine)
        # Phasic ACh provides immediate LR boost for surprising events
        phasic_boost = 1.0 + self.ach_phasic * 1.5
        return base_lr * max(plasticity * phasic_boost, 0.05)

    def modulate_temperature(self, base_temp):
        """Serotonin modulates risk-taking.
        Low 5HT (safe) -> higher temperature (explore freely).
        High 5HT (aversion) -> lower temperature (cautious, exploit)."""
        risk = 1.0 - self.serotonin  # inverse of aversion
        return base_temp * max(0.1 + 0.9 * risk, 0.05)

    def modulate_beam_width(self, base_width):
        """Noradrenaline modulates attention breadth.
        High NA (uncertainty) -> narrow beam (focused search).
        Low NA (relaxed) -> broad beam (parallel exploration)."""
        focus = 1.0 - self.noradrenaline * 0.5
        return max(1, int(base_width * focus))

    def modulate_homeostasis(self, base_boost):
        """Dopamine modulates homeostatic plasticity.
        High DA -> homeostasis is relaxed (rewarding state).
        Low DA -> homeostasis is strong (need for novelty)."""
        drive = 1.0 - self.dopamine * 0.6
        return base_boost * max(drive, 0.2)

    def confidence_score(self):
        """Current confidence based on recent match rate and dopamine."""
        avg_m = np.mean(self.recent_matches) if self.recent_matches else 0.0
        return 0.3 + 0.4 * avg_m + 0.3 * self.dopamine

    def summary(self):
        return {
            "da": self.dopamine,
            "5ht": self.serotonin,
            "na": self.noradrenaline,
            "ach": self.acetylcholine,
            "da_phasic": round(self.da_phasic, 3),
            "ach_phasic": round(self.ach_phasic, 3),
        }

    def save(self):
        return {
            "dopamine": self.dopamine,
            "serotonin": self.serotonin,
            "noradrenaline": self.noradrenaline,
            "acetylcholine": self.acetylcholine,
            "da_phasic": self.da_phasic,
            "ach_phasic": self.ach_phasic,
            "step": self.step,
            "recent_confidences": self.recent_confidences[-20:],
            "recent_matches": self.recent_matches[-20:],
            "_prev_avg_match": self._prev_avg_match,
            "_last_few_cids": self._last_few_cids,
        }

    def load(self, data):
        self.dopamine = data.get('dopamine', 0.5)
        self.serotonin = data.get('serotonin', 0.5)
        self.noradrenaline = data.get('noradrenaline', 0.3)
        self.acetylcholine = data.get('acetylcholine', 0.5)
        self.da_phasic = data.get('da_phasic', 0.0)
        self.ach_phasic = data.get('ach_phasic', 0.0)
        self.step = data.get('step', 0)
        self.recent_confidences = data.get('recent_confidences', [])
        self.recent_matches = data.get('recent_matches', [])
        self._prev_avg_match = data.get('_prev_avg_match', 0.0)
        self._last_few_cids = data.get('_last_few_cids', [])


if __name__ == '__main__':
    # Quick test
    hs = HormonalSystem()
    print("Initial:", hs.summary())

    # Simulate: 15 correct (improving), 5 wrong (surprising), then 10 free-gen
    for i in range(30):
        conf = min(0.3 + i * 0.025, 0.85)
        match = i < 15
        nov = max(0.5 - i * 0.015, 0.1)
        surp = 0.1 if match else 0.5
        expected = 1 if i < 15 else None
        gen = 1 if match else 2
        hs.update(confidence=conf, is_match=match,
                   novelty=nov, surprise=surp,
                   expected_cid=expected, gen_cid=gen)
        if i % 5 == 0 or i == 19:
            print(f"  step {i:2d}: {hs.summary()} "
                  f"temp={hs.modulate_temperature(0.5):.3f} "
                  f"lr={hs.modulate_stdp_lr(0.1):.3f} "
                  f"beam={hs.modulate_beam_width(4)}")
