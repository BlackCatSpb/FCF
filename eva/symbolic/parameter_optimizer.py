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
    """A single parameter with feasibility corridor.

    Stores a reference to its ParamDef so that ``toward_default()``
    re-reads the current default from FCFConfig — enabling runtime
    config cascade without reconstruction.
    """

    __slots__ = ('name', 'min', 'max', 'default', 'current', 'step_scale', '_def')

    def __init__(self, name, min_val, max_val, default, step_scale=0.1, param_def=None):
        self.name = name
        self.min = min_val
        self.max = max_val
        self.default = default
        self.current = default
        self.step_scale = step_scale
        self._def = param_def

    @classmethod
    def from_def(cls, d: ParamDef) -> 'Param':
        return cls(d.name, d.min_val, d.max_val, d.default, d.step_scale, param_def=d)

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
        # re-read default from ParamDef for config cascade
        if self._def is not None:
            self.default = self._def.default
        old = self.current
        diff = self.default - self.current
        if abs(diff) < 1e-10:
            return self.current
        step = diff * rate
        if abs(step) < 1e-10:
            step = math.copysign(1e-10, diff)
        return self.set(self.current + step)

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
        self.TARGET_STD = 1.0 / math.sqrt(max(self.config.dim, 1))  # uniform random d-sphere
        self._rules = []  # (trigger_fn, param_name, action, value, rate)

        # Build params from config
        self.p = {}
        for param in config.params:
            self.p[param.name] = Param.from_def(param)

        _c = self.config
        self.m = {
            'mean_cos': MetricBuffer(_c.metric_maxlen_primary),
            'std_cos':  MetricBuffer(_c.metric_maxlen_primary),
            'vec_ppl':  MetricBuffer(_c.metric_maxlen_secondary),
            'acc1':     MetricBuffer(_c.metric_maxlen_secondary),
            'vacc1':    MetricBuffer(_c.metric_maxlen_secondary),
            'delta':    MetricBuffer(_c.metric_maxlen_secondary),
            'ppl':      MetricBuffer(_c.metric_maxlen_secondary),
            'ng_new':   MetricBuffer(_c.metric_maxlen_tiny),
        }

        self._prev_mean_cos = 0.0
        self._vacc1_stuck = 0
        self._step = 0
        self._flat_thresh = _c.opt_flat_threshold
        self._flat_steps = 0
        self._cos_trend_buffer = deque(maxlen=_c.opt_cos_trend_window)
        self._full_stuck_counter = 0

    def _eval_trigger(self, trigger: str, ctx: dict) -> bool:
        """Evaluate a trigger string against context dict of metrics."""
        try:
            # Full stuck: all metrics in plateau simultaneously
            if trigger == 'full_stuck':
                return self._full_stuck_counter >= self.config.opt_full_stuck_threshold
            if trigger == 'vec_ppl_plateau':
                return self.m['vec_ppl'].plateau(patience=self.config.plateau_patience, rel_thresh=self.config.plateau_rel_thresh_ppl)
            if trigger == 'acc1_plateau':
                return self.m['acc1'].plateau(patience=self.config.plateau_patience, rel_thresh=self.config.plateau_rel_thresh_acc1)

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
                t = ctx.get('inh_threshold', self.config.opt_inh_threshold_fallback)
                est_frac = math.erfc(t / (std_cos * math.sqrt(2)))
                ctx['_est_frac'] = est_frac
                return est_frac > thresh

            if trigger.startswith('est_frac < '):
                thresh = float(trigger.split('<')[1].strip())
                std_cos = ctx.get('std_cos')
                if std_cos is None or std_cos <= 0:
                    return False
                t = ctx.get('inh_threshold', self.config.opt_inh_threshold_fallback)
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
        for param in self.config.params:
            p = self.p.get(param.name)
            if p is None or not param.rules:
                continue

            rule_applied = False
            for rule in param.rules:
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
                        changes[param.name] = p.current
                    rule_applied = True

            # Drift to default if no rule fired
            has_drift = any(r.action == 'toward_default' for r in param.rules)
            if not rule_applied and has_drift:
                old = p.current
                p.toward_default(self.config.opt_toward_default_rate)
                if abs(p.current - old) > 1e-8:
                    changes[param.name] = p.current

        # Special: clamp neg_samples to int
        if 'neg_samples' in changes:
            p = self.p['neg_samples']
            p.current = round(p.current)

        # Special: clamp context_window to int
        if 'context_window' in changes:
            p = self.p['context_window']
            p.current = round(p.current)

        # Special: theta_tau to int
        if 'theta_tau' in changes:
            p = self.p['theta_tau']
            p.current = round(p.current)

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

        # Full-stuck detection: all key metrics in plateau simultaneously
        vacc1 = kw.get('vacc1')
        mean_cos = kw.get('mean_cos')
        vec_ppl = kw.get('vec_ppl')
        cos_plateau = mean_cos is not None and abs(mean_cos) < self._flat_thresh
        ppl_plateau = vec_ppl is not None and self.m['vec_ppl'].plateau(patience=self.config.plateau_patience, rel_thresh=self.config.plateau_rel_thresh_default)
        v1_stuck = vacc1 is not None and vacc1 == 0.0
        if cos_plateau and ppl_plateau and v1_stuck:
            self._full_stuck_counter += 1
        else:
            self._full_stuck_counter = 0
        if self._full_stuck_counter >= self.config.opt_full_stuck_threshold:
            changes['full_stuck'] = True

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
            '_full_stuck_counter': self._full_stuck_counter,
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
        self._full_stuck_counter = state.get('_full_stuck_counter', 0)
        buf = state.get('_cos_trend_buffer', [])
        self._cos_trend_buffer = deque(buf, maxlen=5)

    def summary(self):
        return ' | '.join(f"{p.name}={p.current:.4g}" for p in self.p.values())


class PlateauDetector:
    """Мягкий детектор плато с EMA loss + std threshold (§4 Training Dynamics V18).

    All defaults from FCFConfig.detector_* fields.
    """

    def __init__(self, config=None, window=None, patience=None, threshold_std=None,
                 min_decay=None, recovery_factor=None):
        from eva.symbolic.fcf_config import FCFConfig
        _c = config if config is not None else FCFConfig()
        self.window = window if window is not None else _c.detector_window
        self.patience = patience if patience is not None else _c.detector_patience
        self.threshold_std = threshold_std if threshold_std is not None else _c.detector_threshold_std
        self.min_decay = min_decay if min_decay is not None else _c.detector_min_decay
        self.recovery_factor = recovery_factor if recovery_factor is not None else _c.detector_recovery_factor
        self.losses = []
        self.ema_loss = None
        self.ema_alpha = _c.detector_ema_alpha
        self._plateau_steps = 0
        self._decay_factor = 1.0
        self._last_reduction_step = 0

    def update(self, loss: float, step: int) -> float:
        self.losses.append(loss)
        if len(self.losses) > self.window * 2:
            self.losses.pop(0)
        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = self.ema_alpha * loss + (1 - self.ema_alpha) * self.ema_loss
        if len(self.losses) >= self.window:
            recent = self.losses[-self.window:]
            mean = float(np.mean(recent))
            std = float(np.std(recent))
            if std < self.threshold_std * abs(mean) and std > 0:
                self._plateau_steps += 1
            else:
                if self._plateau_steps > 0:
                    self._plateau_steps = max(0, self._plateau_steps - 1)
        if self._plateau_steps >= self.patience:
            from eva.symbolic.fcf_config import FCFConfig
            steps_in_plateau = self._plateau_steps - self.patience
            decay = 1.0 - (steps_in_plateau * FCFConfig().detector_decay_per_step)
            self._decay_factor = max(self.min_decay, decay)
            self._last_reduction_step = step
        else:
            if self._decay_factor < 1.0:
                recovery = self.recovery_factor * (1.0 - self._decay_factor)
                self._decay_factor = min(1.0, self._decay_factor + recovery)
        return self._decay_factor

    def is_plateau(self) -> bool:
        return self._plateau_steps >= self.patience

    def get_metrics(self) -> dict:
        return {'decay_factor': self._decay_factor, 'plateau_steps': self._plateau_steps, 'ema_loss': self.ema_loss}
