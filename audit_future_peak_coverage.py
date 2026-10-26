
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_quantile_closedloop_v1"
INPUT = RESULTS / "closedloop_traces.csv"

FAMILIES = [
    "constant", "low", "high", "ramp",
    "periodic", "bursty", "irregular",
]
HORIZON = 2
BOOTSTRAP_REPS = 10_000
RANDOM_SEED = 20261025


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)
    required = {
        "t", "family", "seed", "quantile_method",
        "exogenous_demand", "effective_demand",
        "decision_called", "predicted_peak", "upper_bound",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[df["family"].isin(FAMILIES)].copy()
    df = df.sort_values(["family", "seed", "quantile_method", "t"])

    run_rows = []
    tick_rows = []

    for (family, seed, method), group in df.groupby(
        ["family", "seed", "quantile_method"], sort=True
    ):
        g = group.sort_values("t").reset_index(drop=True)
        times = g["t"].to_numpy(dtype=int)

        if len(times) < HORIZON + 1:
            raise ValueError(f"Trace too short: {family}/{seed}/{method}")
        if not np.all(np.diff(times) == 1):
            raise ValueError(f"Non-contiguous ticks: {family}/{seed}/{method}")

        exogenous = g["exogenous_demand"].to_numpy(dtype=float)
        effective = g["effective_demand"].to_numpy(dtype=float)
        upper = pd.to_numeric(g["upper_bound"], errors="coerce").to_numpy(float)
        pred = pd.to_numeric(g["predicted_peak"], errors="coerce").to_numpy(float)
        decision = g["decision_called"].astype(bool).to_numpy()

        valid_count = 0
        covered_exogenous = 0
        covered_effective = 0
        exogenous_underprediction_sum = 0.0
        exogenous_underprediction_max = 0.0

        for i, t in enumerate(times):
            # A complete two-reading future window is required.
            if i + HORIZON >= len(g):
                continue
            if not decision[i] or not np.isfinite(upper[i]):
                continue

            actual_exogenous_peak = float(
                exogenous[i + 1:i + 1 + HORIZON].max()
            )
            actual_effective_peak = float(
                effective[i + 1:i + 1 + HORIZON].max()
            )
            covered_x = upper[i] + 1e-9 >= actual_exogenous_peak
            covered_e = upper[i] + 1e-9 >= actual_effective_peak
            miss_amount = max(actual_exogenous_peak - upper[i], 0.0)

            valid_count += 1
            covered_exogenous += int(covered_x)
            covered_effective += int(covered_e)
            exogenous_underprediction_sum += miss_amount
            exogenous_underprediction_max = max(
                exogenous_underprediction_max, miss_amount
            )

            tick_rows.append({
                "family": family,
                "seed": int(seed),
                "method": method,
                "t": int(t),
                "predicted_peak": pred[i],
                "upper_bound": upper[i],
                "actual_exogenous_peak_next_2": actual_exogenous_peak,
                "actual_effective_peak_next_2": actual_effective_peak,
                "covered_exogenous_peak": covered_x,
                "covered_effective_peak": covered_e,
                "exogenous_underprediction_amount": miss_amount,
            })

        run_rows.append({
            "family": family,
            "seed": int(seed),
            "method": method,
            "eligible_forecasts": valid_count,
            "exogenous_coverage_pct": (
                100.0 * covered_exogenous / valid_count
                if valid_count else np.nan
            ),
            "effective_coverage_pct": (
                100.0 * covered_effective / valid_count
                if valid_count else np.nan
            ),
            "exogenous_misses": valid_count - covered_exogenous,
            "mean_exogenous_underprediction": (
                exogenous_underprediction_sum / valid_count
                if valid_count else np.nan
            ),
            "max_exogenous_underprediction": exogenous_underprediction_max,
        })

    run_df = pd.DataFrame(run_rows)
    tick_df = pd.DataFrame(tick_rows)

    # Require complete linear/higher pairs.
    counts = run_df.groupby(["family", "seed"])["method"].nunique()
    if len(counts) != len(FAMILIES) * 10 or not (counts == 2).all():
        raise ValueError("Coverage runs are not complete paired runs.")

    # Pair method differences within each synthetic workload and seed.
    metrics = [
        "exogenous_coverage_pct",
        "effective_coverage_pct",
        "exogenous_misses",
        "mean_exogenous_underprediction",
        "max_exogenous_underprediction",
    ]
    wide = run_df.pivot(
        index=["family", "seed"],
        columns="method",
        values=metrics,
    )
    wide.columns = [f"{m}_{method}" for m, method in wide.columns]
    wide = wide.reset_index()

    for metric in metrics:
        wide[f"delta_{metric}"] = (
            wide[f"{metric}_higher"] - wide[f"{metric}_linear"]
        )

    # Equal weight per family; paired seed bootstrap within each family.
    rng = np.random.default_rng(RANDOM_SEED)
    overall_rows = []
    for metric in metrics:
        family_values = {
            family: group[f"delta_{metric}"].to_numpy(dtype=float)
            for family, group in wide.groupby("family")
        }
        observed = float(np.mean([
            values.mean() for values in family_values.values()
        ]))

        boot = np.empty(BOOTSTRAP_REPS, dtype=float)
        for b in range(BOOTSTRAP_REPS):
            sampled_means = []
            for values in family_values.values():
                sampled = rng.choice(values, size=len(values), replace=True)
                sampled_means.append(sampled.mean())
            boot[b] = np.mean(sampled_means)

        low, high = np.quantile(boot, [0.025, 0.975])
        overall_rows.append({
            "metric": metric,
            "mean_delta_higher_minus_linear": observed,
            "ci95_low": float(low),
            "ci95_high": float(high),
        })

    overall = pd.DataFrame(overall_rows)

    run_path = RESULTS / "future_peak_coverage_run_metrics.csv"
    tick_path = RESULTS / "future_peak_coverage_forecasts.csv"
    paired_path = RESULTS / "future_peak_coverage_paired.csv"
    overall_path = RESULTS / "future_peak_coverage_bootstrap.csv"

    run_df.to_csv(run_path, index=False)
    tick_df.to_csv(tick_path, index=False)
    wide.to_csv(paired_path, index=False)
    overall.to_csv(overall_path, index=False)

    print("FUTURE-PEAK COVERAGE AUDIT: PASS")
    print("Horizon:", HORIZON, "future readings")
    print("Eligible forecasts:", len(tick_df))
    print("Coverage denominator: decision ticks with finite upper bounds")
    print("Ground-truth measures: exogenous demand and effective demand")
    print("Delta convention: higher minus linear")
    print("\n=== RUN-LEVEL COVERAGE ===")
    print(run_df.groupby("method")[
        ["eligible_forecasts", "exogenous_coverage_pct",
         "effective_coverage_pct", "exogenous_misses",
         "mean_exogenous_underprediction",
         "max_exogenous_underprediction"]
    ].mean().to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== PAIRED BOOTSTRAP DIFFERENCES ===")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nSaved:")
    for p in [run_path, tick_path, paired_path, overall_path]:
        print(p)


if __name__ == "__main__":
    main()
