# Real-cluster validation protocol

The simulator (`closedloop.py`) *assumes* how throttling feeds back into demand. The following must be
measured on the cluster before any safety claim is made. Items marked **[not done]** have no data yet.

## 1. Design
* Policies: Static, Threshold, real VPA (`k8s/vpa-inplace.yaml`), RF-Predictive (`--apply`). One policy per run, same node, same pod spec, fresh pod each run.
* Workloads: constant, low, high, ramp, periodic, bursty, irregular. Use a *load generator with a request rate*
  (e.g. k6 / hey / wrk2 against an HTTP service whose CPU cost per request is fixed) - NOT a pure CPU burner -
  so that latency and throughput exist and demand reacts to throttling (closed-loop). Replay the same
  request-rate schedule for every policy (fixed seed).
* Repetitions: >= 5 runs per policy x workload, >= 30 min each (after a >= 10 min warm-up), randomised order.
  Keep tuning runs (dev schedules) separate from reporting runs (held-out schedules, new seeds).
* Hold constant: node, CPU manager policy (must be `none`: in-place resize is not supported with the static CPU manager),
  metrics-server resolution, kubelet version, no co-located noisy neighbours (or record them).

## 2. Measurements  **[not done]**
| Outcome | Source |
|---|---|
| CPU throttling | cgroup `cpu.stat` (`nr_throttled`, `throttled_usec`) or cAdvisor `container_cpu_cfs_throttled_periods_total / container_cpu_cfs_periods_total` |
| Latency p50/p95/p99, errors | load generator output |
| Throughput | load generator (completed req/s) |
| Memory pressure / OOM | `container_memory_working_set_bytes`, pod `lastState.terminated.reason=OOMKilled`, restart count |
| Resize completion time | timestamp of the `/resize` PATCH -> time `status.containerStatuses[].resources` equals the request (poll every 1 s) |
| Resize failures / deferrals | pod conditions `PodResizePending` (reason `Deferred`/`Infeasible`) and `PodResizeInProgress`, controller log |
| Disruption | pod UID changes, container restart counts, evictions |
| Resources saved | integral of (original limit - applied limit) over time, per run |

Per-run decision log: `decisions.csv` written by the controller (`--decision-log`).

## 3. Reporting
Mean and 95% CI over repeated runs, worst run, and paired differences versus Static. State explicitly which
settings were tuned on dev schedules. Financial savings require a pricing model and node bin-packing evidence
(freed CPU only saves money if nodes can be removed or more pods scheduled) - do not infer them from the
allocation reduction alone.

## 4. Novelty (needs your literature work, not code)
The simulation results do not by themselves show novelty. A defensible, narrower claim to test is e.g.:
"*safety-aware reclamation through in-place resize under uncertainty: the conformal margin gives a controllable
risk knob (coverage -> added throttling) that threshold rules and VPA percentile targets do not expose*".
That is testable with the coverage sweep in `results_v2/tuning_summary.csv` and, if it holds, on the cluster.
It is a hypothesis; check it against the papers in your survey (esp. 2.14 Kozhinov & Mazzara, 2.13 Kang et al.).
