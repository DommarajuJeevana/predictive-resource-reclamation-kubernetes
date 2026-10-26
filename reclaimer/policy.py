"""Reclamation policies: the proposed RF-predictive policy and the comparison baselines.

Every policy implements ``decide(obs, current, tick) -> Decision`` where

* ``obs``      - usage *as measured* so far (1-D array, one value per 15 s reading).
                 Measured usage is capped by the allocation in force, so the
                 predictive policy first repairs saturated readings (see ``impute``).
* ``current``  - allocation currently in force for this resource.
* ``tick``     - index of the latest reading (used for cooldown bookkeeping).
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import math
import numpy as np

from .config import PolicyConfig, ResourceSpec
from .model import PeakForecaster


@dataclass
class Decision:
    target: float                 # allocation to apply from now on
    reason: str                   # short human-readable justification
    predicted_peak: float | None = None
    safe_requirement: float | None = None

    def changed(self, current: float) -> bool:
        return abs(self.target - current) > 1e-9


# ----------------------------------------------------------------- helpers
def round_up(x: float, step: float) -> float:
    return math.ceil(x / step - 1e-9) * step


def impute(obs: np.ndarray, alloc: np.ndarray, spec: ResourceSpec, cfg: PolicyConfig) -> np.ndarray:
    """Repair censored readings.

    When usage touches the allocation the true demand is *at least* that value.  We
    replace such readings by a pessimistic estimate (alloc * escalation, capped at the
    original allocation) so that the forecaster never learns that "demand stops at the
    new, smaller limit" - the feedback loop that would otherwise make shrinking
    self-reinforcing.
    """
    obs = np.asarray(obs, float).copy()
    alloc = np.asarray(alloc, float)
    sat = obs >= cfg.saturation * alloc
    obs[sat] = np.minimum(spec.maximum, np.maximum(obs[sat], alloc[sat] * cfg.escalation))
    return obs


class Guardrails:
    """Cooldown, hysteresis, max-step and floor/ceiling enforcement (shared)."""

    def __init__(self, spec: ResourceSpec, cfg: PolicyConfig):
        self.spec, self.cfg = spec, cfg
        self.last_resize = -10 ** 9
        self.last_was_up = False

    def reset(self):
        self.last_resize = -10 ** 9
        self.last_was_up = False

    def gate(self, current: float, desired: float, tick: int, saturated: bool = False) -> tuple[float, str]:
        s, c = self.spec, self.cfg
        desired = min(max(round_up(desired, s.step), s.minimum), s.maximum)
        if saturated:                                         # protect first
            desired = max(desired, min(s.maximum, round_up(current * c.escalation, s.step)))
        if desired > current + 1e-9:                          # scale-UP: never delayed
            self.last_resize, self.last_was_up = tick, True
            return desired, "scale-up: safe requirement above allocation" if not saturated else "scale-up: saturation guard"
        if desired < current - 1e-9:                          # scale-DOWN: gated
            wait = c.cooldown * (c.up_cooldown_mult if self.last_was_up else 1.0)
            if tick - self.last_resize < wait:
                return current, "hold: cooldown"
            if (current - desired) / current < c.hysteresis or (current - desired) < c.min_abs_change:
                return current, "hold: reduction below hysteresis"
            floor = round_up(current * (1 - c.max_down_frac), s.step)
            desired = max(desired, floor, s.minimum)
            if desired < current - 1e-9:
                self.last_resize, self.last_was_up = tick, False
                return desired, "scale-down: reclaim"
        return current, "hold: allocation adequate"


# --------------------------------------------------------- proposed policy
class PredictiveReclaimer:
    """Random-Forest peak forecast + conformal safety margin + guardrails."""
    name = "RF-Predictive"

    def __init__(self, spec: ResourceSpec, cfg: PolicyConfig):
        self.spec, self.cfg = spec, cfg
        self.guard = Guardrails(spec, cfg)
        self.forecaster = PeakForecaster(cfg)
        self.pending: deque = deque()            # (tick, prediction) awaiting ground truth
        self.residuals: deque = deque(maxlen=60)
        self.misses: deque = deque(maxlen=max(cfg.miss_lookback, 1))
        self.fallback_until = -1
        self._last_fit = -10 ** 9

    def reset(self):
        self.guard.reset(); self.forecaster = PeakForecaster(self.cfg)
        self.pending.clear(); self.residuals.clear(); self.misses.clear(); self.fallback_until = -1; self._last_fit = -10 ** 9

    def decide(self, obs, current, tick, demand=None) -> Decision:
        """``demand`` = imputed (uncensored) history; defaults to ``obs``."""
        cfg, spec = self.cfg, self.spec
        d = np.asarray(obs if demand is None else demand, float)
        if len(d) < cfg.warmup:
            return Decision(current, "warm-up: collecting history")

        # 1. (re)fit the forest on windows whose targets are already known
        if tick - self._last_fit >= cfg.retrain_every or not self.forecaster.fitted:
            if self.forecaster.fit(d):
                self._last_fit = tick

        # 2. forecast the peak requirement for the next `horizon` readings
        degenerate = np.ptp(d) <= 0.05 * max(d.max(), 1e-9)       # e.g. flat memory series
        if self.forecaster.fitted and not degenerate:
            pred = self.forecaster.predict(d)
            track = True
        else:
            pred = d[-cfg.warmup:].max()                          # peak-hold fallback
            track = False
        # 3. turn matured predictions into residuals -> conformal upper margin
        while self.pending and self.pending[0][0] + cfg.horizon <= tick:
            t0, p0, u0 = self.pending.popleft()
            actual = d[t0 + 1: t0 + 1 + cfg.horizon].max()
            self.residuals.append(actual - p0)
            self.misses.append(actual > u0 * (1 + 1e-9))   # upper bound failed to cover the peak
        if len(self.residuals) >= cfg.min_residuals:
            q = max(float(np.quantile(self.residuals, cfg.coverage)), 0.0)
            upper = pred + q
        else:
            upper = pred * (1 + cfg.fallback_margin)

        if track:
            self.pending.append((tick, pred, upper))
        # 3b. uncertainty-aware fallback: bound has been failing -> hold the recent peak for a while
        fb = False
        if cfg.fallback_misses > 0:
            if sum(self.misses) >= cfg.fallback_misses:
                self.fallback_until = tick + cfg.fallback_hold; self.misses.clear()
            if tick < self.fallback_until:
                upper = max(upper, d[-cfg.fallback_window:].max()); fb = True
        # 4. safe requirement = max(forecast upper bound, recent peak) * (1 + margin)
        recent_peak = d[-cfg.peak_window:].max()
        safe = max(upper, recent_peak) * (1 + spec.margin)

        saturated = bool(np.asarray(obs, float)[-1] >= cfg.saturation * current)
        target, why = self.guard.gate(current, safe, tick, saturated)
        return Decision(target, ("[fallback] " if fb else "") + why, predicted_peak=pred, safe_requirement=safe)


# ---------------------------------------------------------------- baselines
class StaticPolicy:
    name = "Static"
    def __init__(self, spec, cfg): pass
    def reset(self): pass
    def decide(self, obs, current, tick, demand=None) -> Decision:
        return Decision(current, "static")


class ThresholdPolicy:
    """Reactive rule: shrink after `patience` consecutive low-utilisation readings."""
    name = "Threshold"

    def __init__(self, spec, cfg, low=0.30, high=0.85, patience=3, down=0.75, up=1.5):
        self.spec, self.cfg = spec, cfg
        self.low, self.high, self.patience, self.down, self.up = low, high, patience, down, up
        self.last_resize = -10 ** 9

    def reset(self): self.last_resize = -10 ** 9

    def decide(self, obs, current, tick, demand=None) -> Decision:
        s, c = self.spec, self.cfg
        util = np.asarray(obs, float)[-self.patience:] / current
        if util[-1] >= self.high:
            tgt = min(s.maximum, round_up(current * self.up, s.step))
            if tgt > current:
                self.last_resize = tick
                return Decision(tgt, "util above high threshold")
        elif len(util) == self.patience and (util < self.low).all() and tick - self.last_resize >= c.cooldown:
            tgt = max(s.minimum, round_up(current * self.down, s.step))
            if tgt < current:
                self.last_resize = tick
                return Decision(tgt, "util below low threshold")
        return Decision(current, "hold")


class VPAStylePolicy:
    """Simplified emulation of the VPA recommender (NOT the real VPA).

    target = percentile(usage history) * (1 + margin); resize when the current
    allocation falls outside [lower, upper] = [P50, P95] * (1 + margin) - the same
    structure the real updater uses.  Percentiles come from the (exponentially
    decayed) history of *measured* usage; there is no forecast and no cooldown.
    """
    name = "VPA-style"

    def __init__(self, spec, cfg, pct=0.90, margin=0.15, half_life=240, lower_pct=0.50, upper_pct=0.95):
        self.spec, self.cfg = spec, cfg
        self.pct, self.margin, self.half_life = pct, margin, half_life
        self.lower_pct, self.upper_pct = lower_pct, upper_pct

    def reset(self): pass

    @staticmethod
    def _wpct(x, w, q):
        o = np.argsort(x); x, w = x[o], w[o]
        cw = np.cumsum(w) / w.sum()
        return float(x[np.searchsorted(cw, q)])

    def decide(self, obs, current, tick, demand=None) -> Decision:
        s = self.spec
        x = np.asarray(obs, float)
        if len(x) < self.cfg.warmup:
            return Decision(current, "warm-up")
        w = 0.5 ** ((len(x) - 1 - np.arange(len(x))) / self.half_life)
        target = self._wpct(x, w, self.pct) * (1 + self.margin)
        lower = self._wpct(x, w, self.lower_pct) * (1 + self.margin)
        upper = self._wpct(x, w, self.upper_pct) * (1 + self.margin)
        target = min(max(round_up(target, s.step), s.minimum), s.maximum)
        if current < lower or current > upper:
            if abs(target - current) / current >= 0.10:
                return Decision(target, "outside recommendation bounds", safe_requirement=target)
        return Decision(current, "within bounds", safe_requirement=target)
