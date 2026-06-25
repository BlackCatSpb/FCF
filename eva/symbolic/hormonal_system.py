"""Hormonal (neuro-modulatory) system for intrinsic motivation.

Each hormone modulates a different aspect of learning and generation:
  - Dopamine (DA): reward prediction error -> STDP strength, confidence
  - Serotonin (5HT): punishment / aversion -> risk (temperature), caution
  - Noradrenaline (NA): uncertainty / novelty -> attention (beam focus)
  - Acetylcholine (ACh): plasticity gate -> learning rate, new pattern formation

The system drives self-improvement: the model intrinsically seeks to
maximize prediction accuracy, novelty, and coherence via hormonal feedback.

ALL numerical coefficients come from FormulaCoefficients — zero hardcode.
"""

import math
import numpy as np
from eva.symbolic.fcf_config import FormulaCoefficients, FCFConfig


class HormonalSystem:
    """Neuro-modulatory system for concept generation.

    Hormone levels (0..1) are updated each generation step based on:
    - Prediction confidence (how certain was the model?)
    - Target match (did we predict correctly?)
    - Novelty (is this a new pattern?)
    - Surprise (how unexpected was this outcome?)
    - Coherence (how smooth was the concept transition?)

    All coefficients from FormulaCoefficients — pass via constructor or
    default factory.
    """

    def __init__(self, formula=None):
        _fc = formula if formula is not None else FCFConfig().formula

        # Baselines (tonic levels)
        self.dopamine = _fc.da_baseline
        self.serotonin = _fc.ht_baseline
        self.noradrenaline = _fc.na_baseline
        self.acetylcholine = _fc.ach_baseline

        # Phasic signals (spikes)
        self.da_phasic = 0.0
        self.ach_phasic = 0.0

        # Running stats
        self.step = 0
        self.recent_confidences = []
        self.recent_matches = []

        # Decay rates
        self.tonic_decay = _fc.tonic_decay
        self.phasic_decay = _fc.phasic_decay

        # Dynamic state
        self._prev_avg_match = 0.0
        self._repetition_counter = 0
        self._last_few_cids = []

        # Store formula coefficients for update()
        self._fc = _fc

        # Reward history (bounded FIFO)
        self.reward_history = []

    def update(self, confidence=0.5, is_match=False, novelty=0.0,
               surprise=0.0, expected_cid=None, gen_cid=None):
        """Update hormone levels based on generation event."""
        _fc = self._fc
        self.step += 1

        self.recent_confidences.append(confidence)
        self.recent_matches.append(1.0 if is_match else 0.0)
        if len(self.recent_confidences) > _fc.hormone_recent_window:
            self.recent_confidences.pop(0)
            self.recent_matches.pop(0)

        avg_confidence = np.mean(self.recent_confidences) if self.recent_confidences else 0.5
        avg_match = np.mean(self.recent_matches) if self.recent_matches else 0.0

        delta_match = avg_match - self._prev_avg_match
        self._prev_avg_match = avg_match

        # ---- Dopamine: reward signal (phasic) ----
        da_extrinsic = 0.0
        da_curiosity = 0.0
        da_mastery = 0.0
        da_coherence = _fc.da_coherence_strength

        if expected_cid is not None:
            if is_match:
                da_extrinsic = max(_fc.da_match_hard_threshold, 1.0 - confidence)
            else:
                da_extrinsic = _fc.da_mismatch_penalty * (1.0 + confidence)
        else:
            da_curiosity = novelty * _fc.da_curiosity_strength

        da_mastery = max(0, delta_match) * _fc.da_mastery_strength

        if gen_cid is not None:
            self._last_few_cids.append(gen_cid)
            if len(self._last_few_cids) > _fc.hormone_boredom_window:
                self._last_few_cids.pop(0)
            if (len(self._last_few_cids) >= _fc.hormone_boredom_repeat
                    and len(set(self._last_few_cids)) == 1):
                da_coherence -= _fc.da_boredom_penalty

        intrinsic = da_curiosity + da_mastery + da_coherence
        self.da_phasic = da_extrinsic + intrinsic

        # ---- Acetylcholine: phasic surprise/novelty signal ----
        ach_surprise = 0.0
        ach_novelty = 0.0
        ach_pe = 0.0

        if expected_cid is not None:
            if not is_match:
                ach_surprise = surprise * _fc.ach_surprise_strength
                ach_pe = (1.0 - confidence) * _fc.ach_uncertainty_strength
            else:
                ach_surprise = surprise * _fc.ach_match_strength
        else:
            ach_novelty = novelty * _fc.ach_novelty_scale

        self.ach_phasic = ach_surprise + ach_novelty + ach_pe
        self.ach_phasic = max(0.0, min(1.0, self.ach_phasic))

        # ---- Serotonin: aversion / risk ----
        target_5ht = _fc.ht_baseline_part + _fc.ht_match_scale * (1.0 - avg_match)
        self.serotonin += (target_5ht - self.serotonin) * _fc.ht_adapt_rate

        # ---- Noradrenaline: uncertainty / novelty ----
        target_na = (_fc.na_baseline_part
                     + _fc.na_surprise_scale * surprise
                     + _fc.na_confidence_scale * (1.0 - confidence))
        self.noradrenaline += (target_na - self.noradrenaline) * _fc.na_adapt_rate
        self.noradrenaline = min(max(self.noradrenaline, 0.0), 1.0)

        # ---- Acetylcholine: plasticity gate ----
        novelty_target = _fc.ach_novelty_baseline + _fc.ach_novelty_scale_tonic * novelty
        if is_match and confidence > 0.8:
            novelty_target = _fc.ach_well_known_floor

        self.acetylcholine += (novelty_target - self.acetylcholine) * _fc.ach_tonic_drift
        self.acetylcholine += self.ach_phasic * _fc.ach_phasic_integration
        self.acetylcholine = max(_fc.da_floor, min(1.0, self.acetylcholine))

        # ---- Integrate phasic into tonic ----
        new_da = self.dopamine * self.tonic_decay + self.da_phasic * _fc.da_phasic_to_tonic
        self.dopamine = max(_fc.da_floor, min(1.0, new_da))

        # ---- Decay phasic signals ----
        self.da_phasic *= self.phasic_decay
        self.ach_phasic *= self.phasic_decay

        # Track reward
        self.reward_history.append(self.dopamine)
        if len(self.reward_history) > _fc.hormone_reward_history_maxlen:
            self.reward_history = self.reward_history[-_fc.hormone_reward_history_maxlen:]

    # ---- Modulation functions ----

    def modulate_temperature(self, base_temp):
        _fc = self._fc
        risk = 1.0 - self.serotonin
        return base_temp * max(_fc.da_temperature_baseline + _fc.da_temperature_scale * risk,
                               _fc.da_temperature_min)

    def modulate_beam_width(self, base_width):
        focus = 1.0 - self.noradrenaline * self._fc.na_beam_scale
        return max(1, int(base_width * focus))

    def reset(self):
        self.__init__(formula=self._fc)

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
        self.dopamine = data.get('dopamine', self._fc.da_baseline)
        self.serotonin = data.get('serotonin', self._fc.ht_baseline)
        self.noradrenaline = data.get('noradrenaline', self._fc.na_baseline)
        self.acetylcholine = data.get('acetylcholine', self._fc.ach_baseline)
        self.da_phasic = data.get('da_phasic', 0.0)
        self.ach_phasic = data.get('ach_phasic', 0.0)
        self.step = data.get('step', 0)
        self.recent_confidences = data.get('recent_confidences', [])
        self.recent_matches = data.get('recent_matches', [])
        self._prev_avg_match = data.get('_prev_avg_match', 0.0)
        self._last_few_cids = data.get('_last_few_cids', [])


if __name__ == '__main__':
    hs = HormonalSystem()
    print("Initial:", hs.summary())
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
                  f"beam={hs.modulate_beam_width(4)}")
