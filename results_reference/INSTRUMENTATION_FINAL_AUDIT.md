# Instrumented Phase 5 v2 — Final Integrity Audit

**Overall status: PASS**

This audit reads existing outputs and does not rerun the simulation.
It writes only this report and its JSON log in the v2 output directory.

## Checks

| Check | Status | Detail |
|---|---|---|
| Required artifacts exist | PASS | results_phase5_instrumented_v1\phase5_traces.csv: present; results_phase5_instrumented_v1\phase5_metrics.csv: present; results_phase5_instrumented_v2\phase5_traces.csv: present; results_phase5_instrumented_v2\phase5_metrics.csv: present; results_phase5_instrumented_v2\PREREGISTRATION_v1.3.md: present; results_phase5_instrumented_v2\PREREGISTRATION_SHA256.txt: present |
| Preregistration hash matches | PASS | expected=72897BA28397C741B1DC535F9261E7285C3F6608EEBAEB2A9F21B456F5691499; actual=72897BA28397C741B1DC535F9261E7285C3F6608EEBAEB2A9F21B456F5691499 |
| Required trace columns exist | PASS | missing=none |
| Expected 1000 runs and 160 readings per run | PASS | runs=1000; rows=160000; run-length-values=[160] |
| Legacy alias preserved | PASS | resize_in_flight equals resize_in_flight_before_decision |
| Resize lifecycle is consistent | PASS | applied=7317; previous_tick_scheduled=7317; unmatched_applications=0; unapplied_schedules=0; nonconsecutive_transitions=0 |
| v1/v2 shared trace data unchanged | PASS | all 28 shared columns match exactly |
| At least one non-trace CSV compared | PASS | compared=4 |
| Original Phase 5 directory exists | PASS | results_phase5_v1 |
| Original Phase 5 hash manifest exists | PASS | results_phase5_v1\PHASE5_POSTRUN_SHA256.txt |
| CSV comparison: phase5_bootstrap_summary.csv | PASS | exact data match |
| CSV comparison: phase5_group_summary.csv | PASS | exact data match |
| CSV comparison: phase5_metrics.csv | PASS | exact data match |
| CSV comparison: phase5_paired_comparison.csv | PASS | exact data match |

## Interpretation

- A PASS supports the specific integrity check described; it is not by itself evidence that one policy is superior in real-world systems.
- The v1/v2 comparison checks whether adding instrumentation changed recorded simulation outputs.
- The original Phase 5 manifest is identified but is not verified by this Python script; use the existing PowerShell manifest verifier separately.
- Any mismatch must be investigated before treating the implementation as complete.
