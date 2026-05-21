"""
ContemplationLoop — постоянный мыслительный процесс модели.

Во время простоя модель ищет КОНЦЕПТЫ и ПРОТИВОРЕЧИЯ.
LogicGuard — защита от «безумия» (логические ограничения).
"""

import torch, numpy as np, time, threading, random as py_random
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from collections import deque
from loguru import logger


class LogicGuard:
    def __init__(self, contradiction_filter, grammar, max_depth=5, min_coherence=0.4, max_concept_len=20):
        self.contra_filter = contradiction_filter
        self.grammar = grammar
        self.max_depth = max_depth
        self.min_coherence = min_coherence
        self.max_concept_len = max_concept_len
        self.total_checks = 0
        self.passed_checks = 0
        self.blocked = 0

    def validate(self, sequence, source=None, target=None):
        self.total_checks += 1
        details = {}
        if len(sequence) > self.max_concept_len * 3:
            self.blocked += 1; return False, "too_long", details
        coherence = self._coherence(sequence)
        details["coherence"] = coherence
        if coherence < self.min_coherence:
            self.blocked += 1; return False, f"low_coherence", details
        contra = 0
        for i in range(len(sequence) - 1):
            f, _, _ = self.contra_filter.is_forbidden(sequence[:i+1], sequence[i+1])
            if f: contra += 1
        details["contra"] = contra / max(len(sequence)-1, 1)
        self.passed_checks += 1
        return True, "valid", details

    def _coherence(self, seq):
        if len(seq) < 2: return 1.0
        scores = []
        for ph, p in self.grammar.patterns.get(0, {}).items():
            if len(p.symbol_indices) >= 2:
                for k in range(len(seq)-1):
                    if seq[k] == p.symbol_indices[0] and seq[k+1] == p.symbol_indices[1]:
                        scores.append(p.coherence_score)
        return float(np.mean(scores)) if scores else 0.3

    def summary(self):
        r = self.passed_checks / max(self.total_checks, 1)
        return f"LogicGuard({self.passed_checks}/{self.total_checks}, {r:.0%})"


class ContemplationLoop:
    def __init__(self, potential_field, topological_field, contradiction_filter,
                 concept_miner, grammar, geodesic_navigator, logic_guard, knowledge_base=None,
                 interval=10.0, max_per_cycle=5):
        self.pf = potential_field
        self.topo = topological_field
        self.contra = contradiction_filter
        self.concept_miner = concept_miner
        self.grammar = grammar
        self.geodesic = geodesic_navigator
        self.guard = logic_guard
        self.kb = knowledge_base
        self.interval = interval
        self.max_per_cycle = max_per_cycle
        self._running = False
        self._thread = None
        self._last_active = time.time()
        self.total = 0
        self.concepts = 0
        self.contradictions = 0
        self.discarded = 0

    def mark_active(self): self._last_active = time.time()

    @property
    def is_idle(self): return (time.time() - self._last_active) > 5.0

    def start(self):
        if self._running: return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("[Think] Background thinking started")

    def stop(self): self._running = False

    def _loop(self):
        while self._running:
            if self.is_idle:
                self._contemplate()
            time.sleep(self.interval)

    def _contemplate(self):
        for _ in range(self.max_per_cycle):
            self.total += 1
            seed = self._seed()
            if not seed: continue
            target = self._target(seed)
            if not target: continue
            path = self.geodesic.find_geodesic(seed, target)
            if path is None or not path.steps: self.discarded += 1; continue
            full = seed + target
            valid, reason, details = self.guard.validate(full, seed, target)
            if not valid:
                if "contra" in reason:
                    for i in range(len(seed)):
                        self.contra.forbid(seed[:i+1], [target[0]] if target else [],
                                          None, 0.5)
                    self.contradictions += 1
                else: self.discarded += 1
            else:
                self.concept_miner.search_free_space(from_known=seed, max_tries=3)
                self.concepts += 1

    def _seed(self):
        cand = []
        for level, pats in self.grammar.patterns.items():
            for ph, p in pats.items():
                if p.length >= 2: cand.append(p)
        if not cand: return [py_random.randint(0, 155) for _ in range(3)]
        return py_random.choice(cand).symbol_indices[:py_random.choice(cand).length]

    def _target(self, seed):
        if not seed: return None
        cont = self.pf.get_continuation_potential(seed[-1]).cpu().numpy()
        low = np.where((cont > 0.48) & (cont < 0.55))[0]
        if len(low) == 0: low = np.where(cont < 0.6)[0]
        if len(low) == 0: return None
        s = int(py_random.choice(low))
        return [s] + [py_random.randint(0, 155) for _ in range(2)]

    def summary(self):
        return f"Contemplation(thoughts={self.total}, concepts={self.concepts}, contradictions={self.contradictions})"
