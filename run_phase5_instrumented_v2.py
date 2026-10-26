from __future__ import annotations
import hashlib, json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import experiments_v2 as E
from reclaimer.policy import PredictiveReclaimer, Decision, impute
from reclaimer.closedloop import metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_phase5_instrumented_v2"
PRE = OUT / "PREREGISTRATION_v1.3.md"
HASHFILE = OUT / "PREREGISTRATION_SHA256.txt"
METHODS = ("linear", "higher")
FAMILIES = ("constant", "low", "high", "ramp", "periodic", "bursty", "irregular")
A_SEEDS, B_SEEDS = range(200, 250), range(300, 350)
N, BOOT, BOOTSEED, CARRY = 160, 10_000, 20261023, 0.5
OUTFILES = ["phase5_run_config.json","phase5_source_hashes.json","phase5_environment.txt",
"phase5_traces.csv","phase5_metrics.csv","phase5_paired_comparison.csv",
"phase5_group_summary.csv","phase5_bootstrap_summary.csv","phase5_validation.json"]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resize_in_flight_fields(pending_before, pending_after):
    """Return explicitly named pre/post-decision resize-state fields.

    Keep the original resize_in_flight field as a backward-compatible
    alias for the pre-decision state.
    """
    before = pending_before is not None
    after = pending_after is not None

    return {
        "resize_in_flight": before,
        "resize_in_flight_before_decision": before,
        "resize_in_flight_after_decision": after,
    }


class Variant(PredictiveReclaimer):
    def __init__(self, spec, cfg, method):
        super().__init__(spec, cfg)
        if method not in METHODS: raise ValueError(method)
        self.method, self.events = method, []
    def reset(self):
        super().reset(); self.events.clear()
    def decide(self, obs, current, tick, demand=None):
        cfg, spec = self.cfg, self.spec
        d = np.asarray(obs if demand is None else demand, float)
        if len(d) < cfg.warmup:
            z = Decision(current, "warm-up: collecting history")
            self.events.append({"predicted_peak":np.nan,"residual_quantile":np.nan,
                "upper_bound":np.nan,"recent_peak":np.nan,"safe_requirement":np.nan,
                "target":z.target,"reason":z.reason,"residual_count":len(self.residuals),
                "forecast_tracked":False,"saturated":False})
            return z
        if tick-self._last_fit >= cfg.retrain_every or not self.forecaster.fitted:
            if self.forecaster.fit(d): self._last_fit=tick
        degenerate = np.ptp(d) <= 0.05*max(d.max(),1e-9)
        if self.forecaster.fitted and not degenerate:
            pred, track = self.forecaster.predict(d), True
        else:
            pred, track = d[-cfg.warmup:].max(), False
        while self.pending and self.pending[0][0]+cfg.horizon <= tick:
            t0,p0,u0=self.pending.popleft()
            actual=d[t0+1:t0+1+cfg.horizon].max()
            self.residuals.append(actual-p0)
            self.misses.append(actual > u0*(1+1e-9))
        q=np.nan
        if len(self.residuals)>=cfg.min_residuals:
            q=max(float(np.quantile(self.residuals,cfg.coverage,method=self.method)),0.0)
            upper=pred+q
        else: upper=pred*(1+cfg.fallback_margin)
        if track: self.pending.append((tick,pred,upper))
        fb=False
        if cfg.fallback_misses>0:
            if sum(self.misses)>=cfg.fallback_misses:
                self.fallback_until=tick+cfg.fallback_hold; self.misses.clear()
            if tick<self.fallback_until:
                upper=max(upper,d[-cfg.fallback_window:].max()); fb=True
        peak=d[-cfg.peak_window:].max()
        safe=max(upper,peak)*(1+spec.margin)
        saturated=bool(np.asarray(obs,float)[-1]>=cfg.saturation*current)
        target,why=self.guard.gate(current,safe,tick,saturated)
        reason=("[fallback] " if fb else "")+why
        z=Decision(target,reason,predicted_peak=pred,safe_requirement=safe)
        self.events.append({"predicted_peak":pred,"residual_quantile":q,"upper_bound":upper,
            "recent_peak":peak,"safe_requirement":safe,"target":target,"reason":reason,
            "residual_count":len(self.residuals),"forecast_tracked":track,"saturated":saturated})
        return z

def shifted(case, seed):
    r=np.random.default_rng(10000+seed); n=N
    if case=="B1_intensified_bursts":
        x=r.uniform(100,200)+r.normal(0,15,n)
        for t in range(n):
            if r.random()<.16:
                end=min(t+int(r.integers(2,6)),n); peak=r.uniform(800,1300)
                x[t:end]=np.maximum(x[t:end],peak)
    elif case=="B2_faster_periodic":
        p=int(r.integers(8,17)); ph=r.uniform(0,p)
        high=(((np.arange(n)+ph)%p)/p)<.5
        x=np.where(high,r.uniform(700,950),r.uniform(80,150))+r.normal(0,25,n)
    elif case=="B3_faster_regime_changes":
        x=np.zeros(n); t=0
        while t<n:
            d=int(r.geometric(1/12))+3; end=min(t+d,n)
            x[t:end]=r.choice([100,300,700,1000]); t=end
        x+=r.normal(0,50,n)
    else: raise ValueError(case)
    return np.clip(np.round(x),0,None)

def build_cases():
    out=[]
    for fam in FAMILIES:
        for s in A_SEEDS: out.append(("A_unseen_seed",fam,s,np.asarray(E.make(fam,s,E.REAL),float)))
    for case in ("B1_intensified_bursts","B2_faster_periodic","B3_faster_regime_changes"):
        for s in B_SEEDS: out.append(("B_distribution_shift",case,s,shifted(case,s)))
    return out

def replay(part,case,seed,demand,method,params):
    demand=np.asarray(demand,float)
    if len(demand)!=N: raise ValueError(f"Bad trace length: {case}/{seed}")
    _,spec,cfg,use_imputed=E.build("RF-Predictive",params,demand,seed=0)
    if not use_imputed or cfg.seed!=0 or cfg.n_estimators!=50 or cfg.retrain_every!=8:
        raise RuntimeError(f"Unexpected policy configuration: {cfg}, imputed={use_imputed}")
    if (spec.initial,spec.minimum,spec.step)!=(1000,100,50) or abs(spec.margin-.1)>1e-12:
        raise RuntimeError(f"Unexpected resource config: {spec}")
    pol=Variant(spec,cfg,method); pol.reset()
    alloc=np.zeros(N); obs=np.zeros(N); eff=np.zeros(N); uns=np.zeros(N)
    backlog=0.; cur=spec.initial; pending=None; rows=[]
    for t in range(N):
        pending_before_application = pending
        resize_applied_this_tick = pending is not None and pending[0] <= t
        if resize_applied_this_tick:
            cur, pending = pending[1], None
        alloc[t]=cur; before=backlog; eff[t]=demand[t]+backlog
        obs[t]=min(eff[t],cur); uns[t]=eff[t]-obs[t]; backlog=CARRY*uns[t]
        
        # Preserve the pending-resize state at the beginning
        # of the decision stage.
        
        pending_before_decision = pending

        row = {
            "part": part,
            "case": case,
            "seed": seed,
            "method": method,
            "t": t,
            "resize_in_flight_before_application": pending_before_application is not None,
            "resize_applied_this_tick": bool(resize_applied_this_tick),
            "exogenous_demand": demand[t],
            "backlog_before": before,
            "effective_demand": eff[t],
            "allocation": alloc[t],
            "served": obs[t],
            "unserved": uns[t],
            "backlog_after": backlog,
            "decision_called": False,
        }

        if t!=N-1 and pending is None:
            dem=impute(obs[:t+1],alloc[:t+1],spec,cfg) if use_imputed else None
            dec=pol.decide(obs[:t+1],cur,t,demand=dem); row.update(pol.events[-1])
            row["decision_called"]=True
            if abs(dec.target-cur)>1e-9:
                pending=(t+1,dec.target); row["resize_scheduled"]=True; row["scheduled_target"]=dec.target
            else: row["resize_scheduled"]=False; row["scheduled_target"]=np.nan
        else: row["resize_scheduled"]=False; row["scheduled_target"]=np.nan
        
        # Record both states after the decision/scheduling branch.
        # The legacy field still represents the pre-decision state.
        row.update(
            resize_in_flight_fields(
                pending_before_decision,
                pending,
            )
        )
        rows.append(row)

    tr=pd.DataFrame(rows)
    m=metrics(tr.rename(columns={"allocation":"alloc","exogenous_demand":"demand","effective_demand":"eff"}),spec,cfg)
    ev=tr[tr.t>cfg.warmup]
    backlog_eval=np.maximum(ev.effective_demand.to_numpy()-ev.exogenous_demand.to_numpy(),0)
    summary={"part":part,"case":case,"seed":seed,"method":method,**m,
      "unserved_total_m_readings":float(ev.unserved.sum()),"unserved_max_m":float(ev.unserved.max()),
      "backlog_max_m":float(backlog_eval.max()),"backlog_sum_m_readings":float(backlog_eval.sum()),
      "evaluation_readings":int(len(ev))}
    return tr,summary

def bootstrap(pairs):
    outcomes=[c[:-len("_higher_minus_linear")] for c in pairs if c.endswith("_higher_minus_linear")]
    group_rows=[]
    for (part,case),g in pairs.groupby(["part","case"],sort=True):
        row={"part":part,"case":case,"n_pairs":len(g),
             "different_trajectories":int(g.different_allocation_trajectory.sum())}
        for m in outcomes: row[m+"_mean_diff"]=float(g[m+"_higher_minus_linear"].mean())
        group_rows.append(row)
    rng=np.random.default_rng(BOOTSEED); rows=[]
    for part in ("A_unseen_seed","B_distribution_shift"):
        sub=pairs[pairs.part==part]; names=sorted(sub.case.unique())
        for m in outcomes:
            vals={c:sub[sub.case==c][m+"_higher_minus_linear"].to_numpy(float) for c in names}
            observed=float(np.mean([v.mean() for v in vals.values()])); reps=np.empty(BOOT)
            for b in range(BOOT):
                reps[b]=np.mean([rng.choice(v,size=len(v),replace=True).mean() for v in vals.values()])
            rows.append({"part":part,"metric":m,"n_groups":len(names),
                "mean_diff_higher_minus_linear":observed,"ci95_low":float(np.quantile(reps,.025)),
                "ci95_high":float(np.quantile(reps,.975)),"bootstrap_replicates":BOOT,"bootstrap_seed":BOOTSEED})
    return pd.DataFrame(group_rows),pd.DataFrame(rows)

def main():
    if not PRE.exists() or not HASHFILE.exists(): raise FileNotFoundError("Frozen preregistration/hash missing")
    expected=HASHFILE.read_text(encoding="utf-8").split()[0]
    if sha(PRE).lower()!=expected.lower(): raise RuntimeError("Preregistration hash mismatch; stopping")
    if "Status: FROZEN before evaluation" not in PRE.read_text(encoding="utf-8"):
        raise RuntimeError("Preregistration is not marked frozen")
    exists=[x for x in OUTFILES if (OUT/x).exists()]
    if exists: raise FileExistsError("Refusing to overwrite: "+", ".join(exists))
    settings=ROOT/"results_v3"/"chosen_settings.json"
    rf=json.loads(settings.read_text(encoding="utf-8"))["RF-Predictive"]
    params=rf["params"]
    expected_params={"coverage":.95,"margin":.1,"cooldown":8,"min_abs_change":100.0,"up_cooldown_mult":3.0}
    if params!=expected_params: raise RuntimeError(f"Frozen settings mismatch: {params}")
    cases=build_cases()
    if len(cases)!=500: raise RuntimeError(f"Expected 500 cases, got {len(cases)}")
    for part,case,seed,d in cases:
        if len(d)!=N or not np.isfinite(d).all() or (d<0).any(): raise RuntimeError(f"Invalid trace {case}/{seed}")
        again=np.asarray(E.make(case,seed,E.REAL),float) if part=="A_unseen_seed" else shifted(case,seed)
        if not np.array_equal(d,again): raise RuntimeError(f"Non-deterministic trace {case}/{seed}")
    src=[ROOT/"trace_quantile_cases.py",ROOT/"experiments_v2.py",ROOT/"reclaimer"/"policy.py",
      ROOT/"reclaimer"/"model.py",ROOT/"reclaimer"/"data.py",ROOT/"reclaimer"/"config.py",
      ROOT/"reclaimer"/"closedloop.py",ROOT/"reclaimer"/"workloads.py",settings,PRE]
    if any(not p.exists() for p in src): raise FileNotFoundError("Missing source file for hash manifest")
    hashes={str(p.relative_to(ROOT)):sha(p) for p in src}
    config={"preregistration":"PREREGISTRATION_v1.3.md","preregistration_sha256":expected,
      "frozen_v3_setting":rf["setting"],"frozen_v3_params":params,"methods":METHODS,
      "part_a_families":FAMILIES,"part_a_seeds":[200,249],
      "part_b_cases":["B1_intensified_bursts","B2_faster_periodic","B3_faster_regime_changes"],
      "part_b_seeds":[300,349],"readings":N,"seconds_per_reading":15,"initial_allocation_m":1000,
      "minimum_allocation_m":100,"step_m":50,"backlog_carry":CARRY,"bootstrap_replicates":BOOT,
      "bootstrap_seed":BOOTSEED,"expected_cases":500,"expected_policy_runs":1000}
    traces=[]; summaries=[]
    for i,(part,case,seed,demand) in enumerate(cases,1):
        for method in METHODS:
            print(f"[{i}/500] {part} {case} seed={seed} method={method}",flush=True)
            tr,s=replay(part,case,seed,demand.copy(),method,params); traces.append(tr); summaries.append(s)
    trace_df=pd.concat(traces,ignore_index=True); met=pd.DataFrame(summaries)
    keys=["part","case","seed"]; metric_names=[c for c in met if c not in keys+["method"]]
    wide=met.pivot(index=keys,columns="method",values=metric_names)
    wide.columns=[f"{m}_{method}" for m,method in wide.columns]; wide=wide.reset_index()
    wide["different_allocation_trajectory"]=False
    for (part,case,seed),g in trace_df.groupby(keys,sort=False):
        a=g[g.method=="linear"].sort_values("t").allocation.to_numpy()
        b=g[g.method=="higher"].sort_values("t").allocation.to_numpy()
        mask=(wide.part==part)&(wide.case==case)&(wide.seed==seed)
        wide.loc[mask,"different_allocation_trajectory"]=not np.array_equal(a,b)
    for m in metric_names: wide[m+"_higher_minus_linear"]=wide[m+"_higher"]-wide[m+"_linear"]
    group_df,boot_df=bootstrap(wide)
    validation={"preregistration_hash_matches":True,"frozen_settings_match":True,
      "deterministic_generation_checks_passed":True,"workload_cases":len(cases),
      "policy_runs":len(met),"trace_rows":len(trace_df),
      "all_runs_160_readings":bool(trace_df.groupby(keys+["method"]).size().eq(N).all()),
      "different_trajectory_pairs":int(wide.different_allocation_trajectory.sum()),
      "existing_v3_and_quantile_outputs_written":False}
    if len(met)!=1000 or len(trace_df)!=160000 or not validation["all_runs_160_readings"]:
        raise RuntimeError("Count validation failed; refusing to save")
    (OUT/"phase5_run_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    (OUT/"phase5_source_hashes.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
    (OUT/"phase5_environment.txt").write_text(
      f"Python: {sys.version}\nExecutable: {sys.executable}\nPlatform: {platform.platform()}\n"
      f"NumPy: {np.__version__}\nPandas: {pd.__version__}\nscikit-learn: {sklearn.__version__}\n",encoding="utf-8")
    trace_df.to_csv(OUT/"phase5_traces.csv",index=False)
    met.to_csv(OUT/"phase5_metrics.csv",index=False)
    wide.to_csv(OUT/"phase5_paired_comparison.csv",index=False)
    group_df.to_csv(OUT/"phase5_group_summary.csv",index=False)
    boot_df.to_csv(OUT/"phase5_bootstrap_summary.csv",index=False)
    (OUT/"phase5_validation.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
    print("\nPHASE 5 COMPLETE\n"+json.dumps(validation,indent=2))
    print(boot_df.to_string(index=False,float_format=lambda x:f"{x:.6f}"))
    print(f"\nOutputs: {OUT}")

if __name__=="__main__": main()
