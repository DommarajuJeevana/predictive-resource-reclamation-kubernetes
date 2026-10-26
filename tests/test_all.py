import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import json
import numpy as np
from reclaimer.config import PolicyConfig, default_cpu, selected_config
from reclaimer.data import load_usage_csv, collapse_to_readings, build_dataset
from reclaimer.policy import PredictiveReclaimer, Guardrails
from reclaimer.simulate import run_policy, evaluate
from reclaimer.closedloop import run_closed_loop
from reclaimer.controller import parse_cpu_m, parse_mem_mi, resize_patch, Controller, ReplayCluster

CSV = os.path.join(os.path.dirname(__file__), "..", "data", "resize-usage-history.csv")


def test_collapse_removes_staircase():
    raw = load_usage_csv(CSV); rd = collapse_to_readings(raw)
    assert len(raw) == 315 and 100 <= len(rd) <= 115


def test_dataset_has_no_lookahead():
    s = np.arange(30, dtype=float)
    X, y, idx = build_dataset(s, 8, 2)
    assert all(y[k] == s[idx[k] + 1: idx[k] + 3].max() for k in range(len(y)))
    assert idx.max() + 2 < len(s)


def test_quantities():
    assert parse_cpu_m("250m") == 250 and parse_cpu_m("1") == 1000 and abs(parse_cpu_m("774363n") - 0.774363) < 1e-9
    assert parse_mem_mi("4096Ki") == 4 and parse_mem_mi("1Gi") == 1024


def test_guardrails():
    cfg, spec = PolicyConfig(), default_cpu(1000)
    g = Guardrails(spec, cfg)
    assert g.gate(1000, 100, 10)[0] == 500                 # max single reduction 50 %
    assert g.gate(500, 100, 11)[1] == "hold: cooldown"      # cooldown blocks 2nd shrink
    assert g.gate(500, 480, 20)[0] == 500                   # below hysteresis
    assert g.gate(500, 700, 21)[0] == 700                   # scale-up never delayed
    assert g.gate(500, 10, 40)[0] >= spec.minimum           # floor respected
    assert g.gate(300, 100, 60, saturated=True)[0] >= 450   # saturation escalation


def test_resize_patch_preserves_qos_ratio():
    p = resize_patch("app", 400, 64, {"cpu": (1000, 1000), "memory": (128, 128)})
    r = p["spec"]["containers"][0]["resources"]
    assert r["requests"]["cpu"] == r["limits"]["cpu"] == "400m" and r["limits"]["memory"] == "64Mi"
    p = resize_patch("app", 400, 64, {"cpu": (500, 1000), "memory": (128, 128)})
    assert p["spec"]["containers"][0]["resources"]["requests"]["cpu"] == "200m"


def _replay(manage_memory=False, apply=True, trace=None):
    rd = collapse_to_readings(load_usage_csv(CSV)); cpu, mem = rd.cpu.values.astype(float), rd.mem.values.astype(float)
    cl = ReplayCluster(cpu if trace is None else trace, mem[: len(cpu if trace is None else trace)] if trace is None else np.full(len(trace), 4.0), 1000, 128)
    ctl = Controller(cl, "app", None, 1000, 128, apply=apply, manage_memory=manage_memory)   # cfg=None -> selected_config()
    cpu_allocs, mem_allocs = [], []
    n = len(cpu) if trace is None else len(trace)
    for t in range(n - 1):
        ctl.step("replay-pod"); cl.advance(); cpu_allocs.append(cl.cpu); mem_allocs.append(cl.mem)
    return ctl, cl, np.array(cpu_allocs), mem_allocs, (cpu if trace is None else trace)


def test_selected_config_matches_experiment_choice():
    ch = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results_v3", "chosen_settings.json")))["RF-Predictive"]["params"]
    cfg = selected_config()
    for k, v in ch.items():
        if k == "margin": assert default_cpu(1000).margin == v
        else: assert getattr(cfg, k) == v, (k, getattr(cfg, k), v)
    assert cfg.cooldown == 8 and cfg.min_abs_change == 100.0 and cfg.up_cooldown_mult == 3.0 and cfg.coverage == 0.95


def test_controller_effective_config_is_selected_config():
    cl = ReplayCluster([100.0] * 5, [4.0] * 5, 1000, 128)
    ctl = Controller(cl, "app", None, 1000, 128)                       # library default
    assert ctl.cfg == selected_config() and ctl.cfg_mem.min_abs_change == 0.0 and ctl.manage_memory is False
    assert ctl.spec_c.margin == 0.10


def test_cli_main_builds_selected_config(monkeypatch=None):
    import reclaimer.controller as C
    seen = {}
    orig_k8s, orig_run = C.K8s, C.Controller.run
    C.K8s = lambda ns, container: ReplayCluster([100.0], [4.0], 1000, 128)
    C.Controller.run = lambda self, *a, **k: seen.setdefault("ctl", self)
    try:
        test_log = os.path.join(os.path.dirname(__file__), "_test_controller.csv")
        C.main([
            "--selector", "app=x",
            "--container", "app",
            "--cpu-alloc", "1000",
            "--mem-alloc", "128",
            "--decision-log", test_log,
        ])
    finally:
        C.K8s, C.Controller.run = orig_k8s, orig_run
    ctl = seen["ctl"]
    assert ctl.cfg == selected_config(interval_s=15) and not ctl.apply and not ctl.manage_memory   # dry-run, CPU only, v3 settings
    assert ctl.cfg.cooldown * ctl.cfg.interval_s == 120                                           # 120 s scale-down cooldown


def test_controller_replay_matches_simulator():
    ctl, cl, allocs, _, cpu = _replay()
    cfg, spec = selected_config(), default_cpu(1000)
    sim = run_closed_loop(PredictiveReclaimer(spec, cfg), cpu, spec, cfg, True, carry=0.0, delay=0)
    assert np.allclose(allocs, sim.alloc.values[1:])            # same decisions as the validated simulator
    assert allocs.min() >= spec.minimum and allocs.max() <= 1000


def test_live_guardrails_hold_in_replay():
    """Every scale-down is >= 8 readings after the previous resize (>=24 after a scale-up) and >= 100 m."""
    from reclaimer.workloads import make
    for fam in ("periodic", "bursty", "irregular"):
        _, _, allocs, _, _ = _replay(trace=make(fam, 7))
        prev, last_t, last_up = 1000.0, -10 ** 9, False
        for t, a in enumerate(allocs):
            if a < prev:
                assert t - last_t >= (24 if last_up else 8), (fam, t, last_t)
                assert prev - a >= 100
            if a != prev: last_t, last_up = t, a > prev
            prev = a


def test_memory_untouched_by_default_and_patch_omits_it():
    ctl, cl, _, mem_allocs, _ = _replay()
    assert set(mem_allocs) == {128.0}
    for r in cl.applied: assert "memory" in r and r["memory"] == "128Mi" or "memory" not in r   # never changed
    p = resize_patch("app", 400, None, {"cpu": (1000, 1000), "memory": (128, 128)})
    assert "memory" not in p["spec"]["containers"][0]["resources"]["limits"]


def test_dry_run_never_calls_resize():
    _, cl, allocs, _, _ = _replay(apply=False)
    assert cl.applied == [] and set(allocs) == {1000.0}


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)


def test_phase5_resize_in_flight_instrumentation_semantics():
    """Legacy field remains pre-decision; new field records post-decision."""
    from run_phase5 import resize_in_flight_fields

    # A resize is scheduled during this tick.
    fields = resize_in_flight_fields(None, (11, 500.0))

    assert fields == {
        "resize_in_flight": False,
        "resize_in_flight_before_decision": False,
        "resize_in_flight_after_decision": True,
    }

    # A resize is already pending before the decision.
    pending = (11, 500.0)
    fields = resize_in_flight_fields(pending, pending)

    assert fields["resize_in_flight"] is True
    assert fields["resize_in_flight_before_decision"] is True
    assert fields["resize_in_flight_after_decision"] is True

    # No resize is pending on either side of the decision.
    fields = resize_in_flight_fields(None, None)

    assert fields["resize_in_flight"] is False
    assert fields["resize_in_flight_after_decision"] is False
