"""ParameterOptimizer — auto-tuning of training hyperparameters with feasibility corridors.

Each parameter has a [min, max] corridor and drifts toward default when
the metric that pushed it away normalises.  Rule-based adaptation uses
metric trends and plateau detection, not gradients.

Rules are loaded from FCFConfig — no hardcoded if/elif chains.
"""

import math, time
import numpy as np
from collections import deque
from typing import Optional
from eva.symbolic.fcf_config import FCFConfig, AdaptRule, ParamDef


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

    @classmethod
    def from_def(cls, d: ParamDef) -> 'Param':
        return cls(d.name, d.min_val, d.max_val, d.default, d.step_scale)

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
    """Auto-optimiser with config-driven rules.

    Usage::

        opt = ParameterOptimizer(config)
        ...
        opt.step(mean_cos=m, std_cos=s, vec_ppl=vp, ...)
    """

    def __init__(self, config: Optional[FCFConfig] = None):
        if config is None:
            from eva.symbolic.fcf_config import FCFConfig
            config = FCFConfig()

        self.config = config
        self.TARGET_STD = 1.0 / math.sqrt(self.config.dim)  # uniform random d-sphere
        self._rules = []  # (trigger_fn, param_name, action, value, rate)

        # Build params from config
        self.p = {}
        for pd in config.params:
            self.p[pd.name] = Param.from_def(pd)

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
        self._flat_thresh = 0.002    # |cos| below this = flat (symmetric plateau)
        self._flat_steps = 0         # consecutive steps with |cos| below thresh
        self._cos_trend_buffer = deque(maxlen=5)  # abs(cos) history for plateau detection

    def _eval_trigger(self, trigger: str, ctx: dict) -> bool:
        """Evaluate a trigger string against context dict of metrics."""
        try:
            # Plateau triggers
            if trigger == 'vec_ppl_plateau':
                return self.m['vec_ppl'].plateau(patience=3, rel_thresh=0.002)
            if trigger == 'acc1_plateau':
                return self.m['acc1'].plateau(patience=3, rel_thresh=0.02)

            # vacc1 stuck
            if trigger.startswith('vacc1_stuck >= '):
                n = int(trigger.split('>=')[1].strip())
                return self._vacc1_stuck >= n

            # cos_flat >= N — cos stuck in symmetric plateau
            if trigger.startswith('cos_flat >= '):
                n = int(trigger.split('>=')[1].strip())
                return self._flat_steps >= n

            # est_frac > X
            if trigger.startswith('est_frac > '):
                thresh = float(trigger.split('>')[1].strip())
                std_cos = ctx.get('std_cos')
                if std_cos is None or std_cos <= 0:
                    return False
                t = ctx.get('inh_threshold', 0.1)
                est_frac = math.erfc(t / (std_cos * math.sqrt(2)))
                ctx['_est_frac'] = est_frac
                return est_frac > thresh

            if trigger.startswith('est_frac < '):
                thresh = float(trigger.split('<')[1].strip())
                std_cos = ctx.get('std_cos')
                if std_cos is None or std_cos <= 0:
                    return False
                t = ctx.get('inh_threshold', 0.1)
                est_frac = math.erfc(t / (std_cos * math.sqrt(2)))
                ctx['_est_frac'] = est_frac
                return est_frac < thresh

            # cos_trend condition
            if trigger.startswith('cos_trend > '):
                parts = trigger.split(' and ')
                trend_thresh = float(parts[0].split('>')[1].strip())
                mean_thresh = float(parts[1].split('>')[1].strip()) if '>' in parts[1] else -999
                cos_trend = ctx.get('mean_cos', 0) - self._prev_mean_cos
                return cos_trend > trend_thresh and ctx.get('mean_cos', 0) > mean_thresh

            if trigger.startswith('cos_trend < '):
                parts = trigger.split(' and ')
                trend_thresh = float(parts[0].split('<')[1].strip())
                mean_thresh = float(parts[1].split('<')[1].strip()) if '<' in parts[1] else 999
                cos_trend = ctx.get('mean_cos', 0) - self._prev_mean_cos
                return cos_trend < trend_thresh and ctx.get('mean_cos', 0) < mean_thresh

            # Simple comparisons
            for op in ['>=', '<=', '>', '<']:
                if op in trigger:
                    parts = trigger.split(op)
                    lhs = parts[0].strip()
                    rhs = parts[1].strip()

                    # TARGET reference
                    if rhs == '0.80*TARGET':
                        rhs_val = self.TARGET_STD * 0.80
                    elif rhs == '1.30*TARGET':
                        rhs_val = self.TARGET_STD * 1.30
                    else:
                        rhs_val = float(rhs)

                    lhs_val = ctx.get(lhs)
                    if lhs_val is None:
                        return False

                    if op == '>':
                        return lhs_val > rhs_val
                    elif op == '<':
                        return lhs_val < rhs_val
                    elif op == '>=':
                        return lhs_val >= rhs_val
                    elif op == '<=':
                        return lhs_val <= rhs_val

            return False
        except Exception:
            return False

    def ingest(self, **kw):
        for k, v in kw.items():
            if v is not None and k in self.m:
                self.m[k].push(v)

    def step(self, **kw):
        """Run one param-adjustment step. Returns dict {name: value} of changes."""
        self.ingest(**kw)
        self._step += 1
        changes = {}

        # Build context for rule evaluation
        ctx = dict(kw)
        for name, p in self.p.items():
            ctx[name] = p.current

        cos_trend = kw.get('mean_cos', self._prev_mean_cos) - self._prev_mean_cos
        ctx['cos_trend'] = cos_trend

        # Apply rules from config
        for pd in self.config.params:
            p = self.p.get(pd.name)
            if p is None or not pd.rules:
                continue

            rule_applied = False
            for rule in pd.rules:
                if self._eval_trigger(rule.trigger, ctx):
                    old = p.current
                    if rule.action == 'scale':
                        p.scale(rule.value)
                    elif rule.action == 'shift':
                        p.shift(rule.value)
                    elif rule.action == 'set':
                        p.set(rule.value)
                    elif rule.action == 'toward_default':
                        p.toward_default(rule.rate)

                    if abs(p.current - old) > 1e-8:
                        changes[pd.name] = p.current
                    rule_applied = True

            # Drift to default if no rule fired
            has_drift = any(r.action == 'toward_default' for r in pd.rules)
            if not rule_applied and has_drift:
                old = p.current
                p.toward_default(0.02)
                if abs(p.current - old) > 1e-8:
                    changes[pd.name] = p.current

        # Special: clamp neg_samples to int
        if 'neg_samples' in changes:
            p = self.p['neg_samples']
            p.current = int(round(p.current))

        # Special: clamp context_window to int
        if 'context_window' in changes:
            p = self.p['context_window']
            p.current = int(round(p.current))

        # Special: theta_tau to int
        if 'theta_tau' in changes:
            p = self.p['theta_tau']
            p.current = int(round(p.current))

        # Track vacc1 stuck counter
        vacc1 = kw.get('vacc1')
        if vacc1 is not None:
            if vacc1 == 0.0:
                self._vacc1_stuck += 1
            else:
                self._vacc1_stuck = 0

        # Track symmetric-plateau (cos_flat) detection
        mean_cos = kw.get('mean_cos')
        if mean_cos is not None:
            self._cos_trend_buffer.append(abs(mean_cos))
            if abs(mean_cos) < self._flat_thresh:
                self._flat_steps += 1
            else:
                self._flat_steps = 0

        self._prev_mean_cos = kw.get('mean_cos', self._prev_mean_cos)
        return changes

    def save_state(self):
        return {
            'params': {name: {'current': p.current, 'min': p.min, 'max': p.max,
                              'default': p.default, 'step_scale': p.step_scale}
                       for name, p in self.p.items()},
            'metrics': {name: list(buf.data) for name, buf in self.m.items()},
            '_prev_mean_cos': self._prev_mean_cos,
            '_vacc1_stuck': self._vacc1_stuck,
            '_step': self._step,
            '_flat_steps': self._flat_steps,
            '_cos_trend_buffer': list(self._cos_trend_buffer),
        }

    def load_state(self, state):
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
        self._flat_steps = state.get('_flat_steps', 0)
        buf = state.get('_cos_trend_buffer', [])
        self._cos_trend_buffer = deque(buf, maxlen=5)

    def summary(self):
        return ' | '.join(f"{p.name}={p.current:.4g}" for p in self.p.values())
