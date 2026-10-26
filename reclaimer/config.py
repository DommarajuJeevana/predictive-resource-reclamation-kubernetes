"""Configuration objects shared by the simulator and the live controller."""
from __future__ import annotations
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class ResourceSpec:
    """Allocation envelope for one resource of one container."""
    name: str            # "cpu" | "memory"
    unit: str            # "m" (millicores) | "Mi"
    initial: float       # original (static) allocation = ceiling for reclamation
    minimum: float       # hard floor - never reclaim below this
    step: float          # resize granularity (allocation is rounded UP to this)
    margin: float        # multiplicative safety margin on the predicted need

    @property
    def maximum(self) -> float:
        return self.initial


def default_cpu(initial: float = 1000.0) -> ResourceSpec:
    return ResourceSpec("cpu", "m", initial, minimum=100.0, step=50.0, margin=0.10)


def default_memory(initial: float = 128.0) -> ResourceSpec:
    # Memory is less forgiving than CPU (OOM-kill vs. throttling): larger margin.
    return ResourceSpec("memory", "Mi", initial, minimum=16.0, step=8.0, margin=0.50)


@dataclass(frozen=True)
class PolicyConfig:
    # --- sampling ---------------------------------------------------------
    interval_s: int = 15          # one "reading" = one metrics-server window
    # --- forecaster -------------------------------------------------------
    horizon: int = 2              # predict the peak of the next `horizon` readings
    n_lags: int = 8
    warmup: int = 40              # readings observed before any reclamation
    min_train: int = 24           # minimum training windows before RF is trusted
    retrain_every: int = 4
    n_estimators: int = 200
    min_samples_leaf: int = 2
    max_features: float = 0.5
    seed: int = 0
    # --- safety -----------------------------------------------------------
    coverage: float = 0.80        # conformal coverage of the peak forecast
    min_residuals: int = 8        # residuals needed before conformal margin is used
    fallback_margin: float = 0.25 # relative headroom used until then
    peak_window: int = 2          # recent readings whose peak the target must cover
    saturation: float = 0.95      # usage/alloc >= this => throttled/censored reading
    escalation: float = 1.5       # scale-up factor when saturated
    # --- stability --------------------------------------------------------
    cooldown: int = 2             # readings between consecutive scale-DOWNs
    hysteresis: float = 0.10      # ignore reductions smaller than this fraction
    max_down_frac: float = 0.50   # largest single reduction (fraction of current)
    min_abs_change: float = 0.0   # ignore reductions smaller than this many resource units (m / Mi)
    up_cooldown_mult: float = 1.0 # after a scale-UP the next scale-down waits cooldown * this
    # --- burst protection (uncertainty-aware fallback); 0 = off ----------
    fallback_misses: int = 0      # forecast upper bound exceeded this often in the lookback -> fallback
    miss_lookback: int = 12
    fallback_hold: int = 12       # readings to stay in fallback after it triggers
    fallback_window: int = 12     # fallback = peak of this many recent readings
    osc_window: int = 4           # reversal window (readings) used by the metric

    def with_(self, **kw) -> "PolicyConfig":
        return replace(self, **kw)


def selected_config(**override) -> PolicyConfig:
    """The policy settings chosen on development traces in results_v3/ (RF-v3).

    This is the ONE place the live controller and the experiments take their settings from.
    tests/test_all.py checks it against results_v3/chosen_settings.json so they cannot drift.
    Note: `min_abs_change` is in CPU millicores; the memory policy gets its own copy (see
    `memory_config`) because the same number would mean 100 Mi there.
    """
    base = PolicyConfig(n_estimators=50, retrain_every=8,          # same forest size/refit cadence as the study
                        coverage=0.95, cooldown=8, min_abs_change=100.0, up_cooldown_mult=3.0)
    return replace(base, **override)


def memory_config(cfg: PolicyConfig) -> PolicyConfig:
    return replace(cfg, min_abs_change=0.0)


def describe(cfg: PolicyConfig, cpu: ResourceSpec) -> str:
    s = cfg.interval_s
    return (f"coverage={cfg.coverage} margin={cpu.margin} cooldown={cfg.cooldown} readings ({cfg.cooldown * s}s) "
            f"after-scale-up cooldown={cfg.cooldown * cfg.up_cooldown_mult:g} readings ({cfg.cooldown * cfg.up_cooldown_mult * s:g}s) "
            f"hysteresis={cfg.hysteresis} min_abs_change={cfg.min_abs_change:g}m max_down_frac={cfg.max_down_frac} "
            f"warmup={cfg.warmup} horizon={cfg.horizon} trees={cfg.n_estimators}")
