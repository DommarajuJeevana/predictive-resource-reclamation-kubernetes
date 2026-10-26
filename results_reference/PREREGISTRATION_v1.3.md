# Phase 5 Preregistration — Robustness and Generalization

Version: 1.3
Status: FROZEN before evaluation
Output directory: results_phase5_v1

## 1. Research question

Does the empirical choice of residual-quantile interpolation method
(linear versus higher) remain consequential when the predictive CPU
reclamation policy encounters unseen random seeds and shifted synthetic
workload distributions?

This experiment evaluates robustness within a simulator. It does not
establish real-cluster performance or a formal conformal coverage guarantee.

## 2. Frozen policy

Load RF-Predictive parameters from results_v3/chosen_settings.json.
Do not retune on Phase 5 traces.

Expected parameters:
- coverage = 0.95
- margin = 0.10
- cooldown = 8 readings
- min_abs_change = 100 m
- up_cooldown_mult = 3.0
- random forest seed = 0
- 50 trees
- retrain every 8 readings

Abort if the loaded chosen settings disagree with these expected values.
Do not silently substitute PolicyConfig defaults.

## 3. Simulation protocol

- 160 readings per trace.
- 15 seconds per reading (40 minutes per trace).
- Initial CPU allocation: 1000 m.
- Minimum allocation: 100 m.
- Resize granularity: 50 m.
- Demand observation: allocation-censored served demand, using the
  existing quantile replay logic.
- Backlog carry: 0.5.
- Resize delay: 0 readings.
- Policy random forest seed: 0 for every run.
- Workload RNG: NumPy default_rng(10000 + workload_seed).
- Paired methods receive exactly the same exogenous demand trace.
- Each method starts from the same initial allocation.
- Existing V3 and quantile outputs are read-only.

The simulator's backlog and slowdown are modelling assumptions, not
measurements of real application performance.

## 4. Part A: unseen-seed generalization

Workload families:
constant, low, high, ramp, periodic, bursty, irregular.

Workload seeds: integers 200 through 249 inclusive.

Methods:
- linear: baseline interpolation
- higher: conservative interpolation ablation

This produces 350 workload cases (7 families x 50 seeds),
or 700 policy runs.

No parameter selection or threshold adjustment is permitted using
these evaluation traces.

## 5. Part B: synthetic distribution-shift stress tests

Use a separate module under the Phase 5 directory. Do not modify
reclaimer/workloads.py.

Use seeds 300 through 349 inclusive for each shift case. Generate
one deterministic trace per case and seed; replay that exact trace
under both methods.

Shift case B1 — intensified bursts:
- Base level: uniform(100, 200) m.
- Background noise: Normal(0, 15) m.
- At each reading, initiate a burst with probability 0.16.
- Burst duration: integer from 2 through 5 readings inclusive.
- Burst peak: uniform(800, 1300) m.
- A burst raises each affected reading to at least its sampled peak.
- Clip negative demand to zero and round to integer millicores.

Shift case B2 — faster periodic workload:
- Period: integer from 8 through 16 readings inclusive.
- Random phase uniformly sampled from [0, period).
- High-demand fraction of each period: 0.50.
- High level: uniform(700, 950) m.
- Low level: uniform(80, 150) m.
- Add Normal(0, 25) m noise; clip negative demand to zero and round.

Shift case B3 — faster regime changes:
- Start with consecutive regimes.
- Regime duration: geometric(p = 1/12) + 3 readings.
- Regime level chosen from [100, 300, 700, 1000] m.
- Add Normal(0, 50) m noise.
- Continue until 160 readings are generated; truncate the final regime.
- Clip negative demand to zero and round.

Use an independent RNG initialized from 10000 + seed for each
case/seed pair. Implement and test these rules before generating
evaluation results. Record the exact generated traces or their hashes.

These shifts are designed stress cases, not claims that the distributions
match real production workloads.

## 6. Outcomes

Primary:
1. Reclaimed CPU percentage.
2. Throttled demand percentage.

Secondary:
3. p99 slowdown proxy.
4. Resizes per hour.
5. Oscillation reversals.
6. Total unserved demand, in mCPU-readings, summed over the evaluation window.
7. Maximum unserved demand, in mCPU, over the evaluation window.
8. Maximum backlog, in mCPU, over the evaluation window.
9. Cumulative backlog, in mCPU-readings, over the evaluation window.

Calculate all outcomes over the evaluation window, defined as readings with
t > cfg.warmup. Use the existing metric implementation wherever applicable.
Preserve its metric definitions and warm-up exclusion. Document any additional
metrics rather than silently changing existing ones.

## 7. Statistical analysis

For every case and seed, calculate higher minus linear.

Report:
- paired mean difference;
- 95% paired bootstrap confidence interval, using the percentile method:
  the 2.5th and 97.5th percentiles of the bootstrap estimates;
- per-family or per-shift descriptive summaries;
- number of pairs with different allocation trajectories.

For an overall synthetic summary, weight workload families equally rather
than allowing families with more observations to dominate.

Use 10,000 bootstrap replicates and bootstrap RNG seed 20261023.
Part A: resample paired seeds within each workload family, calculate a
mean difference per family, and average family means with equal weights.
Part B: resample paired seeds within each shift case, calculate a mean
difference per shift case, and average the three case means with equal
weights. Use the same bootstrap procedure and fixed seed for reproducibility.

Report Part A and Part B separately. Do not combine them into one headline
estimate. Do not treat individual time ticks as independent replicates.

Do not interpret a confidence interval crossing zero as proof of equivalence.
Do not call the empirical coverage a formal conformal guarantee.

## 8. Validation and integrity

Before evaluation:
- confirm the existing output directory contains no evaluation artifacts;
  preserve its existing files and record that its pre-preregistration
  contents cannot be established from the current inspection;
- load and validate frozen V3 settings;
- test deterministic trace generation;
- verify paired methods receive identical demand;
- verify initial allocations match;
- verify all runs have 160 readings;
- verify no existing results directory is written.

Save this v1.3 preregistration, configuration, source hashes, environment
details, raw traces, metrics, paired comparisons, and analysis outputs.
Expected evaluation size: Part A 700 runs; Part B 300 runs; total 1,000.
Do not overwrite V3 or quantile experiment artifacts.

## 9. Interpretation limits

Synthetic seed generalization is not distribution-shift validation.
Distribution-shift tests are not independent real-cluster evidence.
The closed-loop model's backlog and slowdown are assumed proxies.
A result favoring higher does not by itself justify replacing linear
without evaluating the trade-off between safety, reclamation, and stability.

Any changes after inspecting evaluation results must be recorded as
exploratory follow-up work, not silently folded into this preregistered run.

## 10. Preregistration history and integrity note

The results_phase5_v1 directory existed before its contents were inspected.
At inspection, it contained PREREGISTRATION.md and no other files. The
earlier contents of the directory cannot be independently established.
This corrected version is a new file; the existing v1.0 file is retained
for the record. No policy evaluation results have been examined for Phase 5.

The preregistration is not frozen until its contents have been reviewed,
approved, and its hash recorded before evaluation.

## 11. Additional safety metric definitions

Use the same evaluation window as the existing metrics: readings with
t > cfg.warmup.

For each trace:
- unserved_total_m_readings = sum of the existing unserved column;
- unserved_max_m = maximum of the existing unserved column;
- backlog_t = max(effective_demand_t - exogenous_demand_t, 0);
- backlog_max_m = maximum backlog over the evaluation window;
- backlog_sum_m_readings = sum of backlog over the evaluation window.

The units mCPU-readings are accumulated simulator quantities, not
physical CPU-time measurements. These are derived simulation metrics;
they do not represent measured application-level harm. Do not change
the simulator dynamics or the existing metric definitions.




