from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

import experiments_v2 as E
from reclaimer.policy import PredictiveReclaimer
from reclaimer.closedloop import run_closed_loop, metrics


OUT = "results_quantile_seed0"
os.makedirs(OUT, exist_ok=True)

CACHE = os.path.join(OUT, "runs.jsonl")
RESULTS = os.path.join(OUT, "comparison.csv")


class HigherQuantileReclaimer(PredictiveReclaimer):
    """Same RF policy as V3, changing only quantile interpolation."""

    def decide(self, obs, current, tick, demand=None):
        cfg, spec = self.cfg, self.spec
        d = np.asarray(obs if demand is None else demand, float)

        if len(d) < cfg.warmup:
            from reclaimer.policy import Decision
            return Decision(current, "warm-up: collecting history")

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
            self.residuals.append(actual - p0)
            self.misses.append(actual > u0 * (1 + 1e-9))

        if len(self.residuals) >= cfg.min_residuals:
            q = max(
                float(np.quantile(
                    self.residuals, cfg.coverage, method="higher"
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

        from reclaimer.policy import Decision
        return Decision(
            target,
            ("[fallback] " if fb else "") + why,
            predicted_peak=pred,
            safe_requirement=safe,
        )


def load_selected():
    path = os.path.join("results_v3", "chosen_settings.json")
    with open(path, encoding="utf-8") as f:
        chosen = json.load(f)

    item = chosen["RF-Predictive"]
    return item["setting"], item["params"]


def cached_run(key, fn):
    cache = {}

    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                cache[row["key"]] = row["metrics"]

    if key in cache:
        return cache[key]

    result = fn()

    with open(CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "key": key,
            "metrics": result,
        }) + "\n")

    return result


def run_one(method, params, family, seed):
    demand = E.make(family, seed, E.REAL)

    # Reuse the V2 builder so RF parameters and resource settings match V3.
    policy, spec, cfg, use_imputed = E.build(
        "RF-Predictive", params, demand, seed=0
    )

    if method == "higher":
        policy = HigherQuantileReclaimer(spec, cfg)

    trace = run_closed_loop(
        policy,
        demand,
        spec,
        cfg,
        use_imputed=use_imputed,
        carry=0.5,
        delay=0,
    )
    return metrics(trace, spec, cfg)


def run_static(family, seed):
    demand = E.make(family, seed, E.REAL)
    policy, spec, cfg, use_imputed = E.build(
        "Static", {}, demand, seed=seed
    )
    trace = run_closed_loop(
        policy, demand, spec, cfg,
        use_imputed=use_imputed,
        carry=0.5,
        delay=0,
    )
    return metrics(trace, spec, cfg)


def evaluate(smoke=False):
    setting, params = load_selected()

    families = ["bursty"] if smoke else E.FAMS
    seeds = [200] if smoke else list(range(200, 210))

    rows = []
    total = len(families) * len(seeds)

    print("Selected V3 setting:", setting)
    print("Parameters:", params)
    print("Quantile comparison:", "SMOKE TEST" if smoke else "FULL EVALUATION")
    print("Traces per method:", total, flush=True)

    for family in families:
        for seed in seeds:
            static_key = f"Static||{family}|{seed}|0.5|0"
            static_metrics = cached_run(
                static_key, lambda: run_static(family, seed)
            )

            for method in ("linear", "higher"):
                label = f"{setting}|quantile={method}"
                key = f"RF-Quantile-{method}|{family}|{seed}|0.5|0"

                m = cached_run(
                    key,
                    lambda method=method, family=family, seed=seed:
                        run_one(method, params, family, seed),
                )

                rows.append({
                    "policy": f"RF-Quantile-{method}",
                    "setting": label,
                    "quantile": method,
                    "family": family,
                    "seed": seed,
                    **m,
                    "static_throttled_%":
                        static_metrics["throttled_demand_%"],
                    "extra_throttled_%":
                        m["throttled_demand_%"]
                        - static_metrics["throttled_demand_%"],
                })

        print(f"Completed family: {family}", flush=True)

    df = pd.DataFrame(rows)
    out_path = (
        os.path.join(OUT, "smoke_test.csv") if smoke else RESULTS
    )
    df.to_csv(out_path, index=False)

    print("\nSaved:", out_path)
    print("\nOverall comparison:")
    print(
        df.groupby("quantile").agg(
            traces=("seed", "size"),
            reclaimed_pct=("reclaimed_%", "mean"),
            added_throttling_pp=("extra_throttled_%", "mean"),
            resizes_per_hour=("resizes_per_hour", "mean"),
            p99_slowdown=("slowdown_p99", "mean"),
            worst_trace_throttling=("extra_throttled_%", "max"),
            worst_trace_p99=("slowdown_p99", "max"),
        ).round(4).to_string()
    )

    if not smoke:
        print("\nPer-family comparison:")
        print(
            df.groupby(["family", "quantile"]).agg(
                reclaimed_pct=("reclaimed_%", "mean"),
                added_throttling_pp=("extra_throttled_%", "mean"),
                resizes_per_hour=("resizes_per_hour", "mean"),
                p99_slowdown=("slowdown_p99", "mean"),
                worst_trace_throttling=("extra_throttled_%", "max"),
            ).round(4).to_string()
        )


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "smoke"

    if mode == "smoke":
        evaluate(smoke=True)
    elif mode == "full":
        evaluate(smoke=False)
    else:
        raise SystemExit("Usage: python compare_quantiles.py [smoke|full]")
