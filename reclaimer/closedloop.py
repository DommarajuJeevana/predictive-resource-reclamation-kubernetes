"""Closed-loop simulator in which demand DEPENDS on the allocation.

Model (an assumption, stated explicitly - NOT a measurement of real application behaviour):
  effective demand_t = exogenous demand_t + backlog_t
  served_t           = min(effective demand_t, allocation_t)         <- what metrics-server shows
  unserved_t         = effective demand_t - served_t
  backlog_{t+1}      = carry * unserved_t      (carry=0: work lost, 1: all retried later)
  slowdown_t         = max(1, effective demand_t / allocation_t)     (latency proxy; 1.0 = unaffected)
A resize decided after reading t takes effect at reading t+1+delay; while one is in flight no new
decision is made (mirrors the controller skipping pods with a pending resize).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import PolicyConfig, ResourceSpec
from .policy import impute, round_up


class OraclePolicy:
    """Non-causal reference: knows the exogenous demand of the next `horizon` readings."""
    name = "Oracle"
    def __init__(self, spec, cfg, demand): self.spec, self.cfg, self.demand = spec, cfg, np.asarray(demand, float)
    def reset(self): pass
    def decide(self, obs, current, tick, demand=None):
        from .policy import Decision
        fut = self.demand[tick + 1: tick + 1 + self.cfg.horizon]
        need = (fut.max() if len(fut) else 0.0) * (1 + self.spec.margin)
        return Decision(min(max(round_up(need, self.spec.step), self.spec.minimum), self.spec.maximum), "oracle")


def run_closed_loop(policy, demand, spec: ResourceSpec, cfg: PolicyConfig, use_imputed=False,
                    carry=0.5, delay=0) -> pd.DataFrame:
    demand = np.asarray(demand, float); n = len(demand)
    alloc, obs, eff, uns = (np.zeros(n) for _ in range(4))
    cur, backlog, pending = spec.initial, 0.0, None
    pred = np.full(n, np.nan)
    policy.reset()
    for t in range(n):
        if pending is not None and pending[0] <= t:
            cur, pending = pending[1], None
        alloc[t] = cur
        eff[t] = demand[t] + backlog
        obs[t] = min(eff[t], cur)
        uns[t] = eff[t] - obs[t]
        backlog = carry * uns[t]
        if t == n - 1 or pending is not None:
            continue
        dem = impute(obs[: t + 1], alloc[: t + 1], spec, cfg) if use_imputed else None
        d = policy.decide(obs[: t + 1], cur, t, demand=dem)
        pred[t] = d.predicted_peak if d.predicted_peak is not None else np.nan
        if abs(d.target - cur) > 1e-9:
            pending = (t + 1 + delay, d.target)
    return pd.DataFrame(dict(t=np.arange(n), alloc=alloc, demand=demand, eff=eff, served=obs, unserved=uns, pred=pred))


def metrics(tr: pd.DataFrame, spec: ResourceSpec, cfg: PolicyConfig) -> dict:
    e = tr[tr.t > cfg.warmup]
    a, d, eff, served, uns = (e[c].values for c in ("alloc", "demand", "eff", "served", "unserved"))
    prev = np.concatenate([[tr.alloc.values[cfg.warmup]], a[:-1]])
    delta = a - prev; idx = np.flatnonzero(np.abs(delta) > 1e-9); sg = np.sign(delta[idx])
    osc = sum(1 for k in range(1, len(idx)) if sg[k] != sg[k - 1] and idx[k] - idx[k - 1] <= cfg.osc_window)
    slow = np.maximum(1.0, eff / a)
    hours = len(e) * cfg.interval_s / 3600
    return {"reclaimed_%": 100 * (spec.initial - a).mean() / spec.initial,
            "utilization_%": 100 * served.sum() / a.sum(),
            "waste_m": (a - served).mean(),
            "throttled_demand_%": 100 * uns.sum() / max(d.sum(), 1e-9),
            "throttled_readings_%": 100 * (uns > 1e-9).mean(),
            "slowdown_p95": float(np.percentile(slow, 95)), "slowdown_p99": float(np.percentile(slow, 99)),
            "slowdown_max": float(slow.max()),
            "resizes": len(idx), "resizes_per_hour": len(idx) / hours, "oscillations": osc}
