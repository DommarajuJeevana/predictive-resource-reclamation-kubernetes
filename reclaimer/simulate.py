"""Closed-loop replay of the recorded workload under each allocation policy.

Demand is replayed from the dataset (assumed independent of the allocation).  What a
policy *sees* is the measured usage min(demand, allocation) - i.e. throttling hides
demand exactly as it does on a real cluster.  Allocation chosen at the end of reading t
is in force during reading t+1 (an in-place resize completes well inside one 15 s window).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import PolicyConfig, ResourceSpec
from .policy import impute


def run_policy(policy, demand: np.ndarray, spec: ResourceSpec, cfg: PolicyConfig, use_imputed: bool = False) -> pd.DataFrame:
    demand = np.asarray(demand, float)
    n = len(demand)
    alloc = np.full(n, spec.initial)
    obs = np.zeros(n)
    policy.reset()
    rows = []
    for t in range(n):
        obs[t] = min(demand[t], alloc[t])
        if t >= n - 1:
            break
        hist, a_hist = obs[: t + 1], alloc[: t + 1]
        dem = impute(hist, a_hist, spec, cfg) if use_imputed else None
        dec = policy.decide(hist, alloc[t], t, demand=dem)
        alloc[t + 1] = dec.target
        rows.append(dict(t=t + 1, alloc=dec.target, demand=demand[t + 1],
                         pred=dec.predicted_peak, safe=dec.safe_requirement, reason=dec.reason))
    return pd.DataFrame(rows)


def evaluate(trace: pd.DataFrame, spec: ResourceSpec, cfg: PolicyConfig) -> dict:
    """Metrics over the evaluation window (everything after warm-up)."""
    tr = trace[trace.t > cfg.warmup].reset_index(drop=True)
    a, d = tr.alloc.values, tr.demand.values
    served = np.minimum(d, a)
    prev = np.concatenate([[spec.initial], a[:-1]])
    delta = a - prev
    resize_idx = np.flatnonzero(np.abs(delta) > 1e-9)
    sign = np.sign(delta[resize_idx])
    osc = sum(1 for k in range(1, len(resize_idx))
              if sign[k] != sign[k - 1] and resize_idx[k] - resize_idx[k - 1] <= cfg.osc_window)
    hours = len(tr) * cfg.interval_s / 3600
    return {
        "mean_alloc": a.mean(),
        "utilization_%": 100 * served.sum() / a.sum(),
        "waste": (a - served).mean(),
        "reclaimed": (spec.initial - a).mean(),
        "reclaimed_%": 100 * (spec.initial - a).mean() / spec.initial,
        "resizes": len(resize_idx),
        "resizes_per_hour": len(resize_idx) / hours,
        "oscillations": osc,
        "throttled_readings_%": 100 * (d > a + 1e-9).mean(),
        "throttled_demand_%": 100 * np.maximum(d - a, 0).sum() / d.sum() if d.sum() else 0.0,
    }


def oracle_trace(demand: np.ndarray, spec: ResourceSpec, cfg: PolicyConfig) -> pd.DataFrame:
    """Non-causal reference: allocation = true next-`horizon` peak * (1 + margin)."""
    from .policy import round_up
    n = len(demand); rows = []
    for t in range(n - 1):
        fut = demand[t + 1: t + 1 + cfg.horizon].max()
        tgt = min(max(round_up(fut * (1 + spec.margin), spec.step), spec.minimum), spec.maximum)
        rows.append(dict(t=t + 1, alloc=tgt, demand=demand[t + 1], pred=fut, safe=tgt, reason="oracle"))
    return pd.DataFrame(rows)
