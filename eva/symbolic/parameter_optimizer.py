"""ParameterOptimizer — auto-tuning of training hyperparameters with feasibility corridors.

Each parameter has a [min, max] corridor and drifts toward default when
the metric that pushed it away normalises.  Rule-based adaptation uses
metric trends and plateau detection, not gradients.
"""

import math
import numpy as np
from collections import deque


class Param:
    """A single parameter with feasibility corridor."""

    __slots__ = ('name', 'min', 'max', 'default', 'current', 'step_scale')

    def __init__(self, name, min_val, max_val, default, step_scale=0.1):
        self.name = name
        self.min = min_val
        self.max = max_val
        self.default = default
        self.current = default
        self.step_scale = step_scale

    def set(self, value):
        self.current = max(self.min, min(self.max, value))
        return self.current

    def scale(self, factor):
        return self.set(self.current * factor)

    def shift(self, delta):
        return self.set(self.current + delta)

    def clamp(self):
        self.current = max(self.min, min(self.max, self.current))
        return self.current

    def toward_default(self, rate=0.03):
        old = self.current
        diff = self.default - self.current
        if abs(diff) < 1e-10:
            return self.current
        step = diff * rate
        if abs(step) < 1e-10:
            step = math.copysign(1e-10, diff)
        return self.set(self.current + step)

    @property
    def rng(self):
        return self.max - self.min

    @property
    def pct(self):
        return (self.current - self.min) / max(self.rng, 1e-10)

    def __repr__(self):
        return f"{self.name}={self.current:.4f} [{self.min:.4f}, {self.max:.4f}]"


class MetricBuffer:
    """Ring buffer with trend / plateau helpers."""

    def __init__(self, maxlen=10):
        self.data = deque(maxlen=maxlen)

    def push(self, value):
        self.data.append(value)

    def last(self):
        return self.data[-1] if self.data else None

    def trend(self, window=3):
        if len(self.data) < window + 1:
            return None
        recent = list(self.data)[-window:]
        return recent[-1] - recent[0]

    def plateau(self, patience=3, rel_thresh=0.005):
        if len(self.data) < patience:
            return False
        recent = list(self.data)[-patience:]
        r = max(recent) - min(recent)
        base = max(abs(max(recent)), abs(min(recent)), 1e-10)
        return r / base < rel_thresh

    def __len__(self):
        return len(self.data)


class ParameterOptimizer:
    """Auto-optimiser with feasibility corridors.

    Usage in training loop::

        opt = ParameterOptimizer()
        ...
        # at each checkpoint:
        changes = opt.step(mean_cos=m, std_cos=s, vec_ppl=vp, ...)
        for name, val in changes.items():
            setattr(locals(), name, val)  # or apply explicitly
    """

    TARGET_STD = 1.0 / math.sqrt(384)  # ~0.051 — uniform random d-sphere

    def __init__(self):
        self.p = {
            'full_lr':        Param('full_lr',        0.003,  0.15,   0.03,   0.10),
            'repel_strength': Param('repel_strength', 0.01,   0.20,   0.08,   0.05),
            'noise_scale':    Param('noise_scale',    0.0002, 0.01,   0.001,  0.05),
            'inh_threshold':  Param('inh_threshold',  0.05,   0.30,   0.10,   0.05),
            'inh_strength':   Param('inh_strength',   0.01,   0.15,   0.05,   0.05),
            'inh_sample':     Param('inh_sample',     100,    600,    200,    100),
            'context_window': Param('context_window', 1,      4,      2,      0.5),
            'theta_tau':      Param('theta_tau',      5,      30,     15,     2.0),
            'neg_samples':    Param('neg_samples',    0,      5,      2,      0.5),
            'pmi_gate_min':   Param('pmi_gate_min',   0.05,   0.5,    0.20,   0.02),
            'decay_rate':     Param('decay_rate',      0.998,  0.9999, 0.9998, 0.00005),
        }

        self.m = {
            'mean_cos': MetricBuffer(10),
            'std_cos':  MetricBuffer(10),
            'vec_ppl':  MetricBuffer(8),
            'acc1':     MetricBuffer(8),
            'vacc1':    MetricBuffer(8),
            'delta':    MetricBuffer(8),
            'ppl':      MetricBuffer(8),
            'ng_new':   MetricBuffer(6),
        }

        self._prev_mean_cos = 0.0
        self._vacc1_stuck = 0
        self._step = 0

    def ingest(self, **kw):
        for k, v in kw.items():
            if v is not None and k in self.m:
                self.m[k].push(v)

    def step(self, **kw):
        """Run one param-adjustment step.  Returns dict {name: value} of changes."""
        self.ingest(**kw)
        self._step += 1
        changes = {}

        mean_cos = kw.get('mean_cos', self._prev_mean_cos)
        std_cos = kw.get('std_cos')
        vec_ppl = kw.get('vec_ppl')
        vacc1 = kw.get('vacc1')
        acc1 = kw.get('acc1')
        delta = kw.get('delta')
        ng_new = kw.get('ng_new')

        # ── 1. Repel ← mean_cos ─────────────────────────────────
        p = self.p['repel_strength']
        if mean_cos > 0.01:
            p.scale(1.10)
            changes['repel_strength'] = p.current
        elif mean_cos < -0.005:
            p.scale(0.90)
            changes['repel_strength'] = p.current
        else:
            old = p.current
            p.toward_default(0.02)
            if abs(p.current - old) > 1e-6:
                changes['repel_strength'] = p.current

        # ── 2. Noise ← std_cos vs TARGET_STD ────────────────────
        p = self.p['noise_scale']
        if std_cos is not None:
            if std_cos < self.TARGET_STD * 0.80:
                p.scale(1.15)
                changes['noise_scale'] = p.current
            elif std_cos > self.TARGET_STD * 1.30:
                p.scale(0.90)
                changes['noise_scale'] = p.current
            else:
                old = p.current
                p.toward_default(0.02)
                if abs(p.current - old) > 1e-6:
                    changes['noise_scale'] = p.current

        # ── 3. LR ← cos_trend + vecPPL plateau ──────────────────
        cos_trend = mean_cos - self._prev_mean_cos
        p_lr = self.p['full_lr']
        lr_changed = False

        if cos_trend > 0.001 and mean_cos > 0.005:
            p_lr.scale(0.95)
            lr_changed = True
        elif cos_trend < -0.001 and mean_cos < -0.005:
            p_lr.scale(1.05)
            lr_changed = True

        if vec_ppl is not None and self.m['vec_ppl'].plateau(patience=3, rel_thresh=0.002):
            if p_lr.current < 0.10:
                p_lr.scale(1.08)
                lr_changed = True
            # Widen context window on plateau
            p_ctx = self.p['context_window']
            old_ctx = p_ctx.current
            p_ctx.shift(0.5)
            if p_ctx.current > old_ctx:
                changes['context_window'] = int(round(p_ctx.current))

        # Taper LR toward default if all quiet
        if not lr_changed:
            old = p_lr.current
            p_lr.toward_default(0.01)
            if abs(p_lr.current - old) > 1e-6:
                lr_changed = True
        if lr_changed:
            changes['full_lr'] = p_lr.current

        self._prev_mean_cos = mean_cos

        # ── 4. decay_rate ← ng_new ──────────────────────────────
        p = self.p['decay_rate']
        if ng_new is not None:
            if ng_new < 100:
                p.shift(-0.0001)
                changes['decay_rate'] = p.current
            elif ng_new > 10000:
                p.shift(0.00005)
                changes['decay_rate'] = p.current

        # ── 5. neg_samples ← vacc1 stuck ───────────────────────
        p = self.p['neg_samples']
        if vacc1 is not None:
            if vacc1 == 0.0:
                self._vacc1_stuck += 1
            else:
                self._vacc1_stuck = 0
            if self._vacc1_stuck >= 3 and p.current < 1:
                p.shift(1)
                changes['neg_samples'] = int(round(p.current))
            elif vacc1 > 0.01 and p.current > 0:
                p.shift(-1)
                if p.current < 0.5:
                    p.set(0)
                changes['neg_samples'] = int(round(p.current))

        # ── 6. inh_threshold ← estimated repel fraction ────────
        p = self.p['inh_threshold']
        if std_cos is not None and std_cos > 0:
            t = p.current
            est_frac = math.erfc(t / (std_cos * math.sqrt(2)))
            if est_frac > 0.15:
                p.shift(0.02)
                changes['inh_threshold'] = p.current
            elif est_frac < 0.01 and p.current > 0.06:
                p.shift(-0.02)
                changes['inh_threshold'] = p.current

        # ── 7. pmi_gate_min ← δ ────────────────────────────────
        p = self.p['pmi_gate_min']
        if delta is not None:
            if delta < 2.0 and p.current > 0.05:
                p.shift(-0.01)
                changes['pmi_gate_min'] = p.current
            elif delta > 20.0 and p.current < 0.3:
                p.shift(0.01)
                changes['pmi_gate_min'] = p.current

        # ── 8. theta_tau ← acc1 plateau ─────────────────────────
        p = self.p['theta_tau']
        if acc1 is not None and self.m['acc1'].plateau(patience=3, rel_thresh=0.02):
            if p.current < 25:
                p.shift(2)
                changes['theta_tau'] = int(round(p.current))

        return changes

    def save_state(self):
        """Return serialisable dict of all param values, metric buffers, and internal state."""
        return {
            'params': {name: {'current': p.current, 'min': p.min, 'max': p.max,
                              'default': p.default, 'step_scale': p.step_scale}
                       for name, p in self.p.items()},
            'metrics': {name: list(buf.data) for name, buf in self.m.items()},
            '_prev_mean_cos': self._prev_mean_cos,
            '_vacc1_stuck': self._vacc1_stuck,
            '_step': self._step,
        }

    def load_state(self, state):
        """Restore state from a dict produced by save_state()."""
        for name, pd in state.get('params', {}).items():
            if name in self.p:
                p = self.p[name]
                p.min = pd['min']
                p.max = pd['max']
                p.default = pd['default']
                p.step_scale = pd['step_scale']
                p.current = pd['current']
        for name, data in state.get('metrics', {}).items():
            if name in self.m:
                self.m[name].data.clear()
                for v in data:
                    self.m[name].data.append(v)
        self._prev_mean_cos = state.get('_prev_mean_cos', 0.0)
        self._vacc1_stuck = state.get('_vacc1_stuck', 0)
        self._step = state.get('_step', 0)

    def summary(self):
        return ' | '.join(f"{p.name}={p.current:.4g}" for p in self.p.values())
