"""Policy x workload study with a dev/held-out split.   python experiments_v2.py {tune,test,stress,forecast,all}

* tune     : choose ONE setting per policy using development traces only (seeds 0-3, synthetic families).
* test     : evaluate the chosen settings on held-out traces (seeds 100-109) + the real recorded trace.
* stress   : repeat on bursty/periodic/irregular with resize delay and different retry (carry) assumptions.
* forecast : walk-forward forecast quality per family (MAE/RMSE/under-prediction, incl. during spikes).
Raw per-run records are cached in results_v2/*.jsonl so an interrupted run can resume.
"""
from __future__ import annotations
import itertools, json, os, sys, time, warnings
from dataclasses import replace
import numpy as np, pandas as pd

from reclaimer.config import PolicyConfig, default_cpu
from reclaimer.data import load_usage_csv, collapse_to_readings
from reclaimer.model import walk_forward
from reclaimer.policy import PredictiveReclaimer, StaticPolicy, ThresholdPolicy, VPAStylePolicy
from reclaimer.closedloop import run_closed_loop, metrics, OraclePolicy
from reclaimer.workloads import FAMILIES, make

warnings.filterwarnings("ignore")
OUT = "results_v2"; os.makedirs(OUT, exist_ok=True)
CFG = PolicyConfig(n_estimators=50, retrain_every=8)
DEV_SEEDS, TEST_SEEDS = range(0, 4), range(100, 110)
FAMS = list(FAMILIES)
REAL = collapse_to_readings(load_usage_csv("data/resize-usage-history.csv")).cpu.values.astype(float)
EXTRA_THR_LIMIT = 1.0      # selection rule: mean throttled demand added vs. static must be <= 1 percentage point


# ------------------------------------------------------------ policy zoo
def grid():
    g = {"RF-Predictive": [], "Threshold": [], "VPA-style": []}
    for cov, mg in itertools.product([0.5, 0.8, 0.95, 0.99], [0.1, 0.25]):
        g["RF-Predictive"].append((f"cov={cov},margin={mg}", dict(coverage=cov, margin=mg)))
    for lo, pat, dn in itertools.product([0.1, 0.15, 0.2, 0.3, 0.5], [2, 3, 4, 6, 8], [0.5, 0.75, 0.9]):
        g["Threshold"].append((f"low={lo},patience={pat},down={dn}", dict(low=lo, patience=pat, down=dn)))
    for pc, mg, hl, up in itertools.product([0.5, 0.75, 0.9, 0.95, 0.99], [0.05, 0.15, 0.3], [20, 240], [0.9, 0.95, 0.99]):
        g["VPA-style"].append((f"pct={pc},margin={mg},half_life={hl},upper={up}", dict(pct=pc, margin=mg, half_life=hl, upper_pct=up)))
    return g


def build(policy, params, demand=None, seed=0):
    spec, cfg = default_cpu(1000), CFG.with_(seed=seed)
    if policy == "Static":   return StaticPolicy(spec, cfg), spec, cfg, False
    if policy == "Oracle":   return OraclePolicy(spec, cfg, demand), spec, cfg, False
    if policy == "Threshold":
        return ThresholdPolicy(spec, cfg, **params), spec, cfg, False
    if policy == "VPA-style": return VPAStylePolicy(spec, cfg, **params), spec, cfg, False
    p = dict(params); mg = p.pop("margin"); spec = replace(spec, margin=mg); cfg = cfg.with_(**p)
    return PredictiveReclaimer(spec, cfg), spec, cfg, True


def one_run(policy, params, family, seed, carry=0.5, delay=0, trace=None):
    x = trace if trace is not None else make(family, seed, REAL)
    pol, spec, cfg, imp = build(policy, params, x)
    tr = run_closed_loop(pol, x, spec, cfg, imp, carry=carry, delay=delay)
    return metrics(tr, spec, cfg)


def cached(path, key, fn):
    cache = {}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); cache[r["key"]] = r
    if key in cache: return cache[key]["m"]
    m = fn()
    with open(path, "a") as f: f.write(json.dumps({"key": key, "m": m}) + "\n")
    return m


def paired(path, policy, label, params, fams, seeds, **kw):
    """Run policy and Static on the same traces; return rows incl. throttling *added* by the policy."""
    rows = []
    for fam in fams:
        for s in seeds:
            tag = f"{kw.get('carry',0.5)}|{kw.get('delay',0)}"
            m = cached(path, f"{policy}|{label}|{fam}|{s}|{tag}", lambda: one_run(policy, params, fam, s, **kw))
            st = cached(path, f"Static||{fam}|{s}|{tag}", lambda: one_run("Static", {}, fam, s, **kw))
            rows.append(dict(policy=policy, setting=label, family=fam, seed=s, **m,
                             **{"extra_throttled_%": m["throttled_demand_%"] - st["throttled_demand_%"],
                                "static_throttled_%": st["throttled_demand_%"]}))
    return rows


# ---------------------------------------------------------------- stages
def tune():
    path = f"{OUT}/dev_runs.jsonl"; rows = []
    for pol, settings in grid().items():
        for label, params in settings:
            rows += paired(path, pol, label, params, FAMS, DEV_SEEDS)
        print("tuned", pol, time.strftime("%T"), flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/dev_all_runs.csv", index=False)
    agg = df.groupby(["policy", "setting"]).agg(reclaimed=("reclaimed_%", "mean"), extra_thr=("extra_throttled_%", "mean"),
                                                 worst_extra_thr=("extra_throttled_%", "max"), p99=("slowdown_p99", "mean"),
                                                 resizes_ph=("resizes_per_hour", "mean")).reset_index()
    agg.round(3).to_csv(f"{OUT}/tuning_summary.csv", index=False)
    chosen = {}
    for pol, g in agg.groupby("policy"):
        ok = g[g.extra_thr <= EXTRA_THR_LIMIT]
        best = ok.sort_values("reclaimed", ascending=False).iloc[0] if len(ok) else g.sort_values("extra_thr").iloc[0]
        chosen[pol] = dict(setting=best.setting, feasible=bool(len(ok)), dev_reclaimed=float(best.reclaimed), dev_extra_thr=float(best.extra_thr))
        chosen[pol]["params"] = dict(grid()[pol])[best.setting]
    json.dump(chosen, open(f"{OUT}/chosen_settings.json", "w"), indent=2); print(json.dumps(chosen, indent=1))


def chosen_settings(): return json.load(open(f"{OUT}/chosen_settings.json"))


def test():
    ch = chosen_settings(); path = f"{OUT}/test_runs.jsonl"; rows = []
    for pol, c in ch.items():
        rows += paired(path, pol, c["setting"], c["params"], FAMS + ["recorded-like"], TEST_SEEDS)
        print("test", pol, time.strftime("%T"), flush=True)
    rows += paired(path, "Oracle", "non-causal", {}, FAMS + ["recorded-like"], TEST_SEEDS)
    # Static rows themselves
    for fam in FAMS + ["recorded-like"]:
        for s in TEST_SEEDS:
            m = cached(path, f"Static||{fam}|{s}|0.5|0", lambda: one_run("Static", {}, fam, s))
            rows.append(dict(policy="Static", setting="-", family=fam, seed=s, **m, **{"extra_throttled_%": 0.0, "static_throttled_%": m["throttled_demand_%"]}))
    pd.DataFrame(rows).to_csv(f"{OUT}/test_all_runs.csv", index=False)
    # the real recorded trace (single trace - cannot give a CI)
    real = []
    for pol, c in list(ch.items()) + [("Static", dict(params={}, setting="-")), ("Oracle", dict(params={}, setting="-"))]:
        seeds = range(5) if pol == "RF-Predictive" else [0]
        ms = [one_run(pol, c["params"], "real", s, trace=REAL) for s in seeds]
        real.append(dict(policy=pol, setting=c["setting"], **pd.DataFrame(ms).mean().to_dict()))
    pd.DataFrame(real).round(3).to_csv(f"{OUT}/real_trace_result.csv", index=False)


def stress():
    ch = chosen_settings(); path = f"{OUT}/stress_runs.jsonl"; rows = []
    for name, kw in {"delay=1 (resize takes +15s)": dict(delay=1), "carry=0 (throttled work lost)": dict(carry=0.0),
                     "carry=1 (all throttled work retried)": dict(carry=1.0)}.items():
        for pol, c in ch.items():
            for r in paired(path, pol, c["setting"], c["params"], ["bursty", "periodic", "irregular"], TEST_SEEDS, **kw):
                rows.append(dict(variant=name, **r))
        print("stress", name, time.strftime("%T"), flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/stress_all_runs.csv", index=False)


def forecast():
    rows = []
    for fam in FAMS + ["recorded-like"]:
        for s in TEST_SEEDS:
            x = make(fam, s, REAL); wf = walk_forward(x, CFG)
            hi = wf.actual >= wf.actual.quantile(0.75)
            for m in ("rf", "max4", "p90_hist"):
                e = wf[m] - wf.actual
                rows.append(dict(family=fam, seed=s, predictor=m, MAE=e.abs().mean(), RMSE=float(np.sqrt((e ** 2).mean())),
                                 under_rate=(e < 0).mean(), under_rate_spikes=(e[hi] < 0).mean(),
                                 shortfall_spikes=(-e[hi & (e < 0)]).mean() if (hi & (e < 0)).any() else 0.0))
    pd.DataFrame(rows).to_csv(f"{OUT}/forecast_all_runs.csv", index=False); print("forecast done")


if __name__ == "__main__":
    for st in (sys.argv[1:] or ["all"]):
        for f in ({"tune": [tune], "test": [test], "stress": [stress], "forecast": [forecast], "all": [tune, test, stress, forecast]}[st]):
            f()
