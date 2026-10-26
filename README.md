# Predictive Resource Reclamation for Kubernetes Pods

Research code and reference outputs for the manuscript **“Predictive Resource Reclamation for Kubernetes Using In-Place Pod Resizing for Cloud-Native Applications”**.

## What this repository contains

- `reclaimer/`: policy, workload, forecasting, simulation, and controller implementation.
- `run_phase5_instrumented_v2.py`: entry point for the main evaluation experiment. `phase5` is retained only as the internal code/artifact identifier.
- `experiments_v2.py`, `trace_quantile_cases.py`, and audit/analysis scripts required by the experiment and diagnostic workflows.
- `results_reference/`: configuration, environment details, preregistration, audit reports, hashes, and compact result tables.
- `results_v3/chosen_settings.json`: frozen policy settings consumed by the separate closed-loop diagnostic runner.
- `tests/`: project tests.
- `docs/CLUSTER_EXPERIMENT_PROTOCOL.md`: protocol for real-cluster validation. The manuscript's reported evaluation is simulated; do not interpret this protocol as evidence of completed production-cluster testing.

The full CSV datasets, including the large trace-level tables, are distributed separately through the linked dataset repository/record. Add the final public dataset URL to this README after depositing the dataset.

## Environment

Python 3.10+ is recommended. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Reproduce the main evaluation

Run from the repository root:

```bash
python run_phase5_instrumented_v2.py
```

The script writes output to `results_phase5_instrumented_v2/`. This experiment is computationally intensive because it includes 1,000 policy runs and bootstrap analysis. Do not overwrite or mix the archived reference outputs unless intentionally conducting a new run. The supplied CSVs in the dataset release are the archived results reported by the manuscript.

## Run checks

```bash
python tests/test_all.py
```

Some checks may require optional Kubernetes dependencies or environment-specific configuration. A passing unit test is not equivalent to validation on a live Kubernetes cluster.

## Data and provenance

The dataset release contains the archived main-evaluation CSVs and the separate quantile/closed-loop diagnostic CSVs. The main evaluation and the diagnostic use different evaluation designs and must not be combined as though they were one sample. The preregistration, run configuration, environment record, source hashes, and audit report are included in `results_reference/`.

**Dataset URL:** https://doi.org/10.6084/m9.figshare.34785774

## Citation

Please cite the associated manuscript after its bibliographic details are finalized. A DOI can be added here after publication or repository deposition.

## Before public release

1. Review the code and files for secrets, personal data, credentials, and unintended files.
2. Add a project-approved software license and a data license/terms of use.
3. Replace the dataset URL placeholder with the persistent public URL/DOI.
4. Ensure the repository contents correspond to the version used for the manuscript.
