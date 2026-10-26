from pathlib import Path
import hashlib
import json
import sys
import pandas as pd

ROOT = Path(".")
V1 = ROOT / "results_phase5_instrumented_v1"
V2 = ROOT / "results_phase5_instrumented_v2"
ORIGINAL = ROOT / "results_phase5_v1"
REPORT = V2 / "INSTRUMENTATION_FINAL_AUDIT.md"
LOG = V2 / "INSTRUMENTATION_FINAL_AUDIT.json"

errors = []
checks = []
details = {}

def check(name, passed, detail):
    checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})
    print(f"{'PASS' if passed else 'FAIL'}: {name} — {detail}")
    if not passed:
        errors.append(name)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()

# 1. Required artifacts and preregistration integrity.
required = [
    V1 / "phase5_traces.csv",
    V1 / "phase5_metrics.csv",
    V2 / "phase5_traces.csv",
    V2 / "phase5_metrics.csv",
    V2 / "PREREGISTRATION_v1.3.md",
    V2 / "PREREGISTRATION_SHA256.txt",
]
check("Required artifacts exist", all(p.is_file() for p in required),
      "; ".join(f"{p}: {'present' if p.is_file() else 'MISSING'}" for p in required))

if all(p.is_file() for p in required[-2:]):
    manifest_text = (V2 / "PREREGISTRATION_SHA256.txt").read_text(
        encoding="utf-8-sig"
    )
    actual_hash = sha256(V2 / "PREREGISTRATION_v1.3.md")
    # Extract the first 64-character hex digest from the manifest.
    import re
    match = re.search(r"\b([A-Fa-f0-9]{64})\b", manifest_text)
    expected_hash = match.group(1).upper() if match else None
    check("Preregistration hash matches",
          expected_hash is not None and actual_hash == expected_hash,
          f"expected={expected_hash}; actual={actual_hash}")
    details["preregistration_sha256"] = actual_hash

# 2. Trace schema, counts, and lifecycle.
trace_v1 = pd.read_csv(V1 / "phase5_traces.csv") if (V1 / "phase5_traces.csv").is_file() else None
trace_v2 = pd.read_csv(V2 / "phase5_traces.csv") if (V2 / "phase5_traces.csv").is_file() else None

if trace_v2 is not None:
    keys = ["part", "case", "seed", "method"]
    instrumentation = [
        "resize_in_flight",
        "resize_in_flight_before_decision",
        "resize_in_flight_after_decision",
        "resize_in_flight_before_application",
        "resize_applied_this_tick",
        "resize_scheduled",
    ]
    missing = [c for c in keys + ["t"] + instrumentation if c not in trace_v2.columns]
    check("Required trace columns exist", not missing,
          f"missing={missing or 'none'}")

    if not missing:
        runs = trace_v2[keys].drop_duplicates()
        run_counts = trace_v2.groupby(keys, dropna=False).size()
        check("Expected 1000 runs and 160 readings per run",
              len(runs) == 1000 and (run_counts == 160).all()
              and len(trace_v2) == 160000,
              f"runs={len(runs)}; rows={len(trace_v2)}; "
              f"run-length-values={sorted(run_counts.unique().tolist())}")

        legacy_ok = (
            trace_v2["resize_in_flight"].astype(bool)
            == trace_v2["resize_in_flight_before_decision"].astype(bool)
        ).all()
        check("Legacy alias preserved", legacy_ok,
              "resize_in_flight equals resize_in_flight_before_decision")

        df = trace_v2.sort_values(keys + ["t"]).copy()
        grouped = df.groupby(keys, sort=False)
        df["scheduled_previous_tick"] = (
            grouped["resize_scheduled"].shift(1).fillna(False).astype(bool)
        )
        df["tick_previous"] = grouped["t"].shift(1)
        consecutive = (
            df["tick_previous"].isna()
            | (df["t"] == df["tick_previous"] + 1)
        )
        applied = df["resize_applied_this_tick"].astype(bool)
        scheduled_prev = df["scheduled_previous_tick"]
        before_apply = df["resize_in_flight_before_application"].astype(bool)

        lifecycle_ok = (
            (applied == before_apply).all()
            and (applied[consecutive] == scheduled_prev[consecutive]).all()
            and int((~consecutive).sum()) == 0
        )
        check("Resize lifecycle is consistent", lifecycle_ok,
              f"applied={int(applied.sum())}; "
              f"previous_tick_scheduled={int(scheduled_prev.sum())}; "
              f"unmatched_applications={int((applied & ~scheduled_prev & consecutive).sum())}; "
              f"unapplied_schedules={int((scheduled_prev & ~applied & consecutive).sum())}; "
              f"nonconsecutive_transitions={int((~consecutive).sum())}")

        details["trace_rows_v2"] = len(trace_v2)
        details["unique_runs_v2"] = len(runs)
        details["resize_applications_v2"] = int(applied.sum())

# 3. Compare v1 and v2 outputs. Instrumentation-only columns are excluded.
if trace_v1 is not None and trace_v2 is not None:
    extra_v2 = [
        "resize_in_flight_before_application",
        "resize_applied_this_tick",
    ]
    common_cols = [c for c in trace_v1.columns if c in trace_v2.columns]
    t1 = trace_v1[common_cols].sort_values(
        ["part", "case", "seed", "method", "t"]
    ).reset_index(drop=True)
    t2 = trace_v2[common_cols].sort_values(
        ["part", "case", "seed", "method", "t"]
    ).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(t1, t2, check_dtype=False, check_exact=True)
        trace_match = True
        trace_detail = f"all {len(common_cols)} shared columns match exactly"
    except AssertionError as e:
        trace_match = False
        trace_detail = str(e).splitlines()[0]
    check("v1/v2 shared trace data unchanged", trace_match, trace_detail)

# Compare every same-named CSV except traces, whose schema intentionally changed.
compared = []
for p1 in sorted(V1.glob("*.csv")):
    if p1.name == "phase5_traces.csv":
        continue
    p2 = V2 / p1.name
    if not p2.is_file():
        continue
    try:
        a = pd.read_csv(p1)
        b = pd.read_csv(p2)
        pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=True)
        compared.append({"file": p1.name, "match": True, "detail": "exact data match"})
        print(f"PASS: v1/v2 CSV unchanged — {p1.name}")
    except Exception as e:
        compared.append({"file": p1.name, "match": False,
                         "detail": str(e).splitlines()[0]})
        print(f"FAIL: v1/v2 CSV differs — {p1.name}: {str(e).splitlines()[0]}")
        errors.append(f"v1/v2 CSV match: {p1.name}")

check("At least one non-trace CSV compared", len(compared) > 0,
      f"compared={len(compared)}")
details["csv_comparisons"] = compared

# 4. Confirm original result directory still exists; do not modify it.
check("Original Phase 5 directory exists", ORIGINAL.is_dir(), str(ORIGINAL))
original_manifest = ORIGINAL / "PHASE5_POSTRUN_SHA256.txt"
check("Original Phase 5 hash manifest exists", original_manifest.is_file(),
      str(original_manifest))

# Report the audit outcome.
status = "PASS" if not errors else "FAIL"
details["status"] = status
details["checks"] = checks
details["errors"] = errors
details["original_manifest_path"] = str(original_manifest)

LOG.write_text(json.dumps(details, indent=2), encoding="utf-8")

lines = [
    "# Instrumented Phase 5 v2 — Final Integrity Audit",
    "",
    f"**Overall status: {status}**",
    "",
    "This audit reads existing outputs and does not rerun the simulation.",
    "It writes only this report and its JSON log in the v2 output directory.",
    "",
    "## Checks",
    "",
    "| Check | Status | Detail |",
    "|---|---|---|",
]
for item in checks:
    detail = item["detail"].replace("|", "\\|").replace("\n", " ")
    lines.append(f"| {item['check']} | {'PASS' if item['passed'] else 'FAIL'} | {detail} |")
for item in compared:
    detail = item["detail"].replace("|", "\\|").replace("\n", " ")
    lines.append(f"| CSV comparison: {item['file']} | "
                 f"{'PASS' if item['match'] else 'FAIL'} | {detail} |")
lines += [
    "",
    "## Interpretation",
    "",
    "- A PASS supports the specific integrity check described; it is not by itself "
    "evidence that one policy is superior in real-world systems.",
    "- The v1/v2 comparison checks whether adding instrumentation changed recorded "
    "simulation outputs.",
    "- The original Phase 5 manifest is identified but is not verified by this "
    "Python script; use the existing PowerShell manifest verifier separately.",
    "- Any mismatch must be investigated before treating the implementation as complete.",
    "",
]
REPORT.write_text("\n".join(lines), encoding="utf-8")

print(f"\nOverall audit status: {status}")
print("Report:", REPORT.resolve())
print("JSON log:", LOG.resolve())
if errors:
    print("Failures:", "; ".join(errors))
    sys.exit(1)
