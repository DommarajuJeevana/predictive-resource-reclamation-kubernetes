from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import experiments_v2 as E
from reclaimer.policy import PredictiveReclaimer, Decision, impute
from reclaimer.closedloop import metrics


OUT = "results_quantile_audit_identified"
os.makedirs(OUT, exist_ok=True)

CASES = [
    ("bursty", 202),
    ("bursty", 205),
    ("bursty", 206),
    ("periodic", 203),
    ("periodic", 207),
]

METRIC_COLUMNS = [
    "reclaimed_%",
    "throttled_demand_%",
    "resizes_per_hour",
    "slowdown_p99",
    "resizes",
]


class TracedReclaimer(PredictiveReclaimer):
    """Replay the existing policy and log its internal decisions."""

    def __init__(self, spec, cfg, quantile_method, family, seed):
    	super().__init__(spec, cfg)
    	self.quantile_method = quantile_method
    	self.family = family
    	self.seed = seed
    	self.events = []
    	self.matured = []

    def reset(self):
        super().reset()
        self.events.clear()
        self.matured.clear()

    def decide(self, obs, current, tick, demand=None):
        cfg, spec = self.cfg, self.spec
        d = np.asarray(obs if demand is None else demand, float)

        if len(d) < cfg.warmup:
            decision = Decision(current, "warm-up: collecting history")
            self.events.append({
                "t": tick,
                "predicted_peak": np.nan,
                "residual_quantile": np.nan,
                "upper_bound": np.nan,
                "recent_peak": np.nan,
                "safe_requirement": np.nan,
                "target": decision.target,
                "reason": decision.reason,
                "residual_count": len(self.residuals),
                "forecast_tracked": False,
                "saturated": False,
            })
            return decision

        if tick - self._last_fit >= cfg.retrain_every or not self.forecaster.fitted:
            if self.forecaster.fit(d):
                self._last_fit = tick

        degenerate = np.ptp(d) <= 0.05 * max(d.max(), 1e-9)

        if self.forecaster.fitted and not degenerate:
            pred = self.forecaster.predict(d)
            track = True
        else:
            pred = d[-cfg.warmup:].max()
            track = False

        while self.pending and self.pending[0][0] + cfg.horizon <= tick:
            t0, p0, u0 = self.pending.popleft()
            actual = d[t0 + 1:t0 + 1 + cfg.horizon].max()
            residual = actual - p0
            covered = actual <= u0 * (1 + 1e-9)

            self.residuals.append(residual)
            self.misses.append(not covered)

            self.matured.append({
                "t_issued": t0,
                "t_matured": tick,
                "prediction": p0,
                "upper_at_issue": u0,
                "actual_peak": actual,
                "residual": residual,
                "covered_by_upper": covered,
                "quantile_method": self.quantile_method,
		"family": self.family,
		"seed": self.seed,
            })

        q = np.nan
        if len(self.residuals) >= cfg.min_residuals:
            q = max(
                float(np.quantile(
                    self.residuals,
                    cfg.coverage,
                    method=self.quantile_method,
                )),
                0.0,
            )
            upper = pred + q
        else:
            upper = pred * (1 + cfg.fallback_margin)

        if track:
            self.pending.append((tick, pred, upper))

        fb = False
        if cfg.fallback_misses > 0:
            if sum(self.misses) >= cfg.fallback_misses:
                self.fallback_until = tick + cfg.fallback_hold
                self.misses.clear()
            if tick < self.fallback_until:
                upper = max(upper, d[-cfg.fallback_window:].max())
                fb = True

        recent_peak = d[-cfg.peak_window:].max()
        safe = max(upper, recent_peak) * (1 + spec.margin)

        saturated = bool(
            np.asarray(obs, float)[-1] >= cfg.saturation * current
        )

        target, why = self.guard.gate(current, safe, tick, saturated)
        reason = ("[fallback] " if fb else "") + why

        decision = Decision(
            target,
            reason,
            predicted_peak=pred,
            safe_requirement=safe,
        )

        self.events.append({
            "t": tick,
            "input_last": d[-1],
            "predicted_peak": pred,
            "residual_quantile": q,
            "upper_bound": upper,
            "recent_peak": recent_peak,
            "safe_requirement": safe,
            "target": target,
            "reason": reason,
            "residual_count": len(self.residuals),
            "forecast_tracked": track,
            "saturated": saturated,
        })

        return decision


def replay(family, seed, method, params):
    demand = E.make(family, seed, E.REAL)

    # Match the existing comparison experiment's configuration and seed.
    _, spec, cfg, use_imputed = E.build(
        "RF-Predictive", params, demand, seed=0
    )

    policy = TracedReclaimer(spec, cfg, method, family, seed)

    n = len(demand)
    alloc = np.zeros(n)
    obs = np.zeros(n)
    eff = np.zeros(n)
    uns = np.zeros(n)

    cur = spec.initial
    backlog = 0.0
    pending = None
    rows = []

    policy.reset()

    for t in range(n):
        if pending is not None and pending[0] <= t:
            cur, pending = pending[1], None

        alloc[t] = cur
        backlog_before = backlog
        eff[t] = demand[t] + backlog
        obs[t] = min(eff[t], cur)
        uns[t] = eff[t] - obs[t]
        backlog = 0.5 * uns[t]

        row = {
            "t": t,
            "family": family,
            "seed": seed,
            "quantile_method": method,
            "exogenous_demand": demand[t],
            "backlog_before": backlog_before,
            "effective_demand": eff[t],
            "allocation": alloc[t],
            "served": obs[t],
            "unserved": uns[t],
            "backlog_after": backlog,
            "resize_in_flight": pending is not None,
            "decision_called": False,
        }

        # Match run_closed_loop: no decision on the final tick or
        # while a resize is in flight.
        if t != n - 1 and pending is None:
            dem = (
                impute(obs[:t + 1], alloc[:t + 1], spec, cfg)
                if use_imputed else None
            )

            decision = policy.decide(
                obs[:t + 1], cur, t, demand=dem
            )
            event = policy.events[-1]

            row.update(event)
            row["decision_called"] = True

            if abs(decision.target - cur) > 1e-9:
                pending = (t + 1, decision.target)
                row["resize_scheduled"] = True
                row["scheduled_target"] = decision.target
            else:
                row["resize_scheduled"] = False
                row["scheduled_target"] = np.nan
        else:
            row["resize_scheduled"] = False
            row["scheduled_target"] = np.nan

        rows.append(row)

    trace = pd.DataFrame(rows)

    # Same metric function as the original evaluation.
    trace_for_metrics = trace.rename(columns={
        "allocation": "alloc",
        "exogenous_demand": "demand",
        "effective_demand": "eff",
    })
    m = metrics(trace_for_metrics, spec, cfg)

    summary = {
        "family": family,
        "seed": seed,
        "quantile": method,
        **m,
    }

    return trace, pd.DataFrame(policy.matured), summary


def main():
    with open(
        os.path.join("results_v3", "chosen_settings.json"),
        encoding="utf-8",
    ) as f:
        chosen = json.load(f)

    setting = chosen["RF-Predictive"]["setting"]
    params = chosen["RF-Predictive"]["params"]

    print("V3 setting:", setting)
    print("Parameters:", params)

    traces = []
    residuals = []
    summaries = []

    for family, seed in CASES:
        for method in ("linear", "higher"):
            print(f"Replaying {family}, seed={seed}, method={method} ...")

            trace, matured, summary = replay(
                family, seed, method, params
            )

            traces.append(trace)
            residuals.append(matured)
            summaries.append(summary)

    detail = pd.concat(traces, ignore_index=True)
    residual_detail = pd.concat(residuals, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    detail_path = os.path.join(OUT, "trace_diagnostics.csv")
    residual_path = os.path.join(OUT, "trace_diagnostics_residuals.csv")
    summary_path = os.path.join(OUT, "trace_diagnostics_metrics.csv")

    detail.to_csv(detail_path, index=False)
    residual_detail.to_csv(residual_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # Verify that the tracing harness reproduces the existing
    # quantile comparison before interpreting the logs.
    original_path = os.path.join("results_quantile_seed0", "comparison.csv")
    original = pd.read_csv(original_path)
    failures = []

    for _, actual in summary_df.iterrows():
        expected_rows = original[
            (original["family"] == actual["family"])
            & (original["seed"] == actual["seed"])
            & (original["quantile"] == actual["quantile"])
        ]

        if len(expected_rows) != 1:
            failures.append(
                f"Missing/duplicate original row: "
                f"{actual['family']} seed={actual['seed']} "
                f"{actual['quantile']}"
            )
            continue

        expected = expected_rows.iloc[0]

        for metric in METRIC_COLUMNS:
            difference = abs(
                float(actual[metric]) - float(expected[metric])
            )
            if difference > 1e-8:
                failures.append(
                    f"{actual['family']} seed={actual['seed']} "
                    f"{actual['quantile']} {metric}: "
                    f"new={actual[metric]} original={expected[metric]} "
                    f"abs_diff={difference}"
                )

    print("\n=== REPLAY VALIDATION ===")
    if failures:
        print("FAIL: traced replay does not exactly reproduce the CSV.")
        for failure in failures:
            print(" -", failure)
        print("Do not interpret the traces until this mismatch is resolved.")
    else:
        print("PASS: all 8 replays match the original metrics.")

    print("\n=== METRICS FOR SELECTED TRACES ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== RESIZE EVENTS ===")
    changes = detail[
        detail["resize_scheduled"] == True
    ]
    print(
        changes[
            [
                "family", "seed", "quantile_method", "t",
                "exogenous_demand", "effective_demand", "allocation",
                "predicted_peak", "residual_quantile", "upper_bound",
                "recent_peak", "safe_requirement", "scheduled_target",
                "reason", "saturated",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )

    print("\nSaved:")
    print(detail_path)
    print(residual_path)
    print(summary_path)


if __name__ == "__main__":
    main()


