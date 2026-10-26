"""Live controller: Metrics API -> RF forecast -> policy -> in-place resize (/resize subresource).

Safe by default: runs in --dry-run mode (logs decisions, changes nothing) unless --apply is given.

    python -m reclaimer.controller --namespace default --selector app=cpu-burst --container app \
        --cpu-alloc 1000 --mem-alloc 128 --bootstrap-csv data/resize-usage-history.csv --apply
"""
from __future__ import annotations
import argparse, csv, json, logging, os, time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .config import PolicyConfig, default_cpu, default_memory, selected_config, memory_config, describe
from .data import load_usage_csv, collapse_to_readings
from .policy import PredictiveReclaimer, impute

log = logging.getLogger("reclaimer")


# --------------------------------------------------------- quantity parsing
def parse_cpu_m(q: str) -> float:
    q = str(q)
    for suf, k in (("n", 1e-6), ("u", 1e-3), ("m", 1.0)):
        if q.endswith(suf):
            return float(q[:-1]) * k
    return float(q) * 1000.0


def parse_mem_mi(q: str) -> float:
    q = str(q)
    units = {"Ki": 1 / 1024, "Mi": 1, "Gi": 1024, "Ti": 1024 ** 2, "k": 1000 / 2 ** 20, "M": 1e6 / 2 ** 20,
             "G": 1e9 / 2 ** 20, "m": 1e-3 / 2 ** 20}
    for suf, k in units.items():
        if q.endswith(suf):
            return float(q[: -len(suf)]) * k
    return float(q) / 2 ** 20


def fmt_cpu(m: float) -> str: return f"{int(round(m))}m"
def fmt_mem(mi: float) -> str: return f"{int(round(mi))}Mi"


def resize_patch(container: str, cpu_m: float, mem_mi: float | None, cur: dict) -> dict:
    """Strategic-merge patch for the /resize subresource.

    `cur` = {"cpu": (request_m, limit_m), "memory": (request_mi, limit_mi)}.  The new value is
    applied to the *limit* and the request keeps its current request/limit ratio, so the QoS
    class (immutable) is preserved, e.g. Guaranteed stays Guaranteed.
    """
    def pair(new, fmt, key):
        req, lim = cur[key]
        ratio = (req / lim) if lim else 1.0
        return {"requests": fmt(new * ratio), "limits": fmt(new)}
    c = pair(cpu_m, fmt_cpu, "cpu")
    req, lim = {"cpu": c["requests"]}, {"cpu": c["limits"]}
    if mem_mi is not None:                       # memory is only touched when explicitly managed
        m = pair(mem_mi, fmt_mem, "memory"); req["memory"], lim["memory"] = m["requests"], m["limits"]
    return {"spec": {"containers": [{"name": container, "resources": {"requests": req, "limits": lim}}]}}


# ------------------------------------------------------------------ sources
@dataclass
class PodState:
    cpu_alloc: float          # millicores (limit)
    mem_alloc: float          # Mi (limit)
    cur: dict                 # request/limit pairs, for resize_patch
    resizing: bool = False    # a previous resize is still pending / in progress


class K8s:
    """Thin wrapper around the kubernetes client (imported lazily so tests need no cluster)."""

    def __init__(self, namespace: str, container: str):
        from kubernetes import client, config
        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()
        self.core, self.custom, self.api = client.CoreV1Api(), client.CustomObjectsApi(), client.ApiClient()
        self.ns, self.container = namespace, container

    def pods(self, selector: str) -> list[str]:
        ps = self.core.list_namespaced_pod(self.ns, label_selector=selector).items
        return [p.metadata.name for p in ps if p.status.phase == "Running"]

    def usage(self, pod: str) -> tuple[float, float]:
        d = self.custom.get_namespaced_custom_object("metrics.k8s.io", "v1beta1", self.ns, "pods", pod)
        c = next(c for c in d["containers"] if c["name"] == self.container)
        return parse_cpu_m(c["usage"]["cpu"]), parse_mem_mi(c["usage"]["memory"])

    def state(self, pod: str) -> PodState:
        p = self.core.read_namespaced_pod(pod, self.ns)
        spec = next(c for c in p.spec.containers if c.name == self.container).resources
        req, lim = spec.requests or {}, spec.limits or {}
        cl = parse_cpu_m(lim.get("cpu", req.get("cpu", "0")))
        ml = parse_mem_mi(lim.get("memory", req.get("memory", "0")))
        cur = {"cpu": (parse_cpu_m(req.get("cpu", lim.get("cpu", "0"))), cl),
               "memory": (parse_mem_mi(req.get("memory", lim.get("memory", "0"))), ml)}
        busy = any(c.type in ("PodResizePending", "PodResizeInProgress") and c.status == "True"
                   for c in (p.status.conditions or []))
        return PodState(cl, ml, cur, busy)

    def resize(self, pod: str, patch: dict):
        # PATCH /api/v1/namespaces/{ns}/pods/{name}/resize  (works on any client version)
        return self.api.call_api(
            f"/api/v1/namespaces/{self.ns}/pods/{pod}/resize", "PATCH", body=patch,
            header_params={"Content-Type": "application/strategic-merge-patch+json"},
            auth_settings=["BearerToken"], response_type="object", _return_http_data_only=True)


class ReplayCluster:
    """Offline stand-in for the cluster: replays demand from a CSV and applies resizes locally."""

    def __init__(self, demand_cpu, demand_mem, cpu0, mem0):
        self.d_cpu, self.d_mem, self.i = list(demand_cpu), list(demand_mem), 0
        self.cpu, self.mem = cpu0, mem0
        self.applied: list = []

    def pods(self, selector): return ["replay-pod"]
    def usage(self, pod): return min(self.d_cpu[self.i], self.cpu), min(self.d_mem[self.i], self.mem)
    def state(self, pod): return PodState(self.cpu, self.mem, {"cpu": (self.cpu, self.cpu), "memory": (self.mem, self.mem)})

    def resize(self, pod, patch):
        r = patch["spec"]["containers"][0]["resources"]["limits"]
        self.cpu = parse_cpu_m(r["cpu"])
        if "memory" in r: self.mem = parse_mem_mi(r["memory"])
        self.applied.append(r)

    def advance(self): self.i += 1


# --------------------------------------------------------------- controller
class Controller:
    def __init__(self, cluster, container, cfg: PolicyConfig | None, cpu0, mem0, apply=False,
                 bootstrap: np.ndarray | None = None, log_path: str | None = None, manage_memory: bool = False):
        cfg = cfg if cfg is not None else selected_config()     # default = the validated v3 settings
        self.k, self.container, self.cfg, self.apply = cluster, container, cfg, apply
        self.cfg_mem = memory_config(cfg)                        # min_abs_change is CPU millicores only
        self.manage_memory = manage_memory                       # OFF by default: memory is unvalidated
        self.spec_c, self.spec_m = default_cpu(cpu0), default_memory(mem0)
        self.boot = bootstrap
        self.pol = {}                                   # pod -> (cpu policy, mem policy)
        self.obs = defaultdict(lambda: {"cpu": [], "mem": [], "ca": [], "ma": []})
        self.tick = defaultdict(int)
        self.log_path = log_path
        if log_path and not os.path.exists(log_path):
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "ts", "pod", "cpu_use", "cpu_alloc", "cpu_pred",
                    "cpu_safe", "cpu_target", "cpu_reason",
                    "mem_use", "mem_alloc", "mem_target", "mem_reason", "applied"
                ])

    def _init_pod(self, pod, st: PodState):
        pc, pm = PredictiveReclaimer(self.spec_c, self.cfg), PredictiveReclaimer(self.spec_m, self.cfg_mem)
        o = self.obs[pod]
        if self.boot is not None:                       # warm start from the recorded dataset
            o["cpu"] = list(self.boot["cpu"]); o["mem"] = list(self.boot["mem"])
            o["ca"] = [self.spec_c.initial] * len(o["cpu"]); o["ma"] = [self.spec_m.initial] * len(o["mem"])
        self.pol[pod] = (pc, pm)

    def step(self, pod: str):
        st = self.k.state(pod)
        if pod not in self.pol:
            self._init_pod(pod, st)
        cpu_u, mem_u = self.k.usage(pod)
        o = self.obs[pod]
        for key, val in (("cpu", cpu_u), ("mem", mem_u), ("ca", st.cpu_alloc), ("ma", st.mem_alloc)):
            o[key].append(val)
        t = self.tick[pod] = len(o["cpu"]) - 1     # tick == index of the latest reading
        pc, pm = self.pol[pod]
        dc = pc.decide(np.array(o["cpu"]), st.cpu_alloc, t,
                       demand=impute(o["cpu"], o["ca"], self.spec_c, self.cfg))
        if self.manage_memory:
            dm = pm.decide(np.array(o["mem"]), st.mem_alloc, t,
                           demand=impute(o["mem"], o["ma"], self.spec_m, self.cfg_mem))
        else:                                            # memory left exactly as configured
            from .policy import Decision
            dm = Decision(st.mem_alloc, "memory management disabled")
        changed = dc.changed(st.cpu_alloc) or dm.changed(st.mem_alloc)
        applied = False
        if changed and st.resizing:
            log.info("%s: previous resize still in progress - skipping", pod)
        elif changed:
            patch = resize_patch(self.container, dc.target, dm.target if self.manage_memory else None, st.cur)
            if self.apply:
                self.k.resize(pod, patch); applied = True
            mem_txt = (f" | mem {fmt_mem(st.mem_alloc)}->{fmt_mem(dm.target)} ({dm.reason})") if self.manage_memory else ""
            log.info("%s: %s cpu %s->%s (%s)%s", pod, "RESIZE" if self.apply else "DRY-RUN",
                     fmt_cpu(st.cpu_alloc), fmt_cpu(dc.target), dc.reason, mem_txt)
            if not self.apply:           # keep the policy's bookkeeping consistent with "what would be"
                pass
        else:
            log.debug("%s: hold cpu=%s use=%.0fm", pod, fmt_cpu(st.cpu_alloc), cpu_u)
        if self.log_path:
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    time.strftime("%FT%T"),
                    pod,
                    round(cpu_u),
                    round(st.cpu_alloc),
                    None if dc.predicted_peak is None else round(dc.predicted_peak),
                    None if dc.safe_requirement is None else round(dc.safe_requirement),
                    round(dc.target),
                    dc.reason,
                    round(mem_u, 1),
                    round(st.mem_alloc),
                    round(dm.target),
                    dm.reason,
                    applied,
                ])
        return dc, dm

    def run(self, selector, interval, iterations=None):
        n = 0
        while iterations is None or n < iterations:
            for pod in self.k.pods(selector):
                try:
                    self.step(pod)
                except Exception:
                    log.exception("%s: step failed", pod)
            n += 1
            time.sleep(interval)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--namespace", default="default"); ap.add_argument("--selector", required=True)
    ap.add_argument("--container", required=True)
    ap.add_argument("--cpu-alloc", type=float, required=True, help="original CPU limit, millicores (reclamation ceiling)")
    ap.add_argument("--mem-alloc", type=float, required=True, help="original memory limit, Mi (reclamation ceiling)")
    ap.add_argument("--interval", type=int, default=15, help="seconds; keep >= metrics-server resolution")
    ap.add_argument("--bootstrap-csv", help="recorded usage history of this workload for a warm start")
    ap.add_argument("--decision-log", default="decisions.csv")
    ap.add_argument("--apply", action="store_true", help="actually resize pods (default: dry-run)")
    ap.add_argument("--manage-memory", action="store_true",
                    help="ALSO resize memory (default: CPU only). The recorded memory is constant 4 Mi and the 128 Mi limit "
                         "is an assumption - memory reclamation is NOT validated")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    boot = None
    if a.bootstrap_csv:
        r = collapse_to_readings(load_usage_csv(a.bootstrap_csv)); boot = {"cpu": r.cpu.tolist(), "mem": r.mem.tolist()}
    if a.interval < 15:
        ap.error("--interval must be >= 15 s (metrics-server resolution): cooldowns are counted in 15 s readings")
    ctl = Controller(K8s(a.namespace, a.container), a.container, selected_config(interval_s=a.interval),
                     a.cpu_alloc, a.mem_alloc, apply=a.apply, bootstrap=boot, log_path=a.decision_log,
                     manage_memory=a.manage_memory)
    log.info("mode=%s memory=%s", "APPLY" if a.apply else "DRY-RUN", "MANAGED" if a.manage_memory else "untouched")
    log.info("effective policy: %s", describe(ctl.cfg, ctl.spec_c))
    ctl.run(a.selector, a.interval)


if __name__ == "__main__":
    main()
