import json, numpy as np
from models import *
from sched_sim import simulate, summarize, ORDER

OUT = "results"
import os; os.makedirs(OUT, exist_ok=True)
R = {}
fit = fit_latency_model()
R["fit"] = {p: {k: (float(v) if not isinstance(v, np.ndarray) else v.tolist()) for k, v in fit[p].items()} for p in ["fp32", "fp16", "int8"]}
R["gamma_T"] = float(fit["gamma_T"]); R["gamma_T_range"] = [float(x) for x in fit["gamma_T_range"]]
R["dav2_cal_flops"] = float(fit["dav2_cal_flops_per_img"])
cat = build_catalog()
R["catalog"] = {k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv) for kk, vv in v.items()} for k, v in cat.items()}

# ---------------- Table: predicted isolated latency per precision ----------------
PREC = ["fp32", "fp16", "int8"]
lat = {k: {p: float(predict_latency(cat[k], p, fit)) for p in PREC} for k in cat}
lat_lo = {k: {p: float(predict_latency(cat[k], p, fit, gamma_T=fit["gamma_T_range"][0])) for p in PREC} for k in cat}
lat_hi = {k: {p: float(predict_latency(cat[k], p, fit, gamma_T=fit["gamma_T_range"][1])) for p in PREC} for k in cat}
R["latency"] = lat; R["latency_lo"] = lat_lo; R["latency_hi"] = lat_hi

# ---------------- Policy assignment ----------------
KAPPA = 0.004
def make_tasks(prec_map, kappa=KAPPA, eff_mult=1.0, gamma_T=None, cat=cat, occ_scale=1.0):
    T = {}
    for k in ORDER + ["asr"]:
        p = prec_map[k]
        ph = task_phases(cat[k], p, fit, kappa=kappa, gamma_T=gamma_T, eff_mult=eff_mult)
        ph = [(d, min(1.0, o * occ_scale) if i == 1 else o, b) for i, (d, o, b) in enumerate(ph)]
        T[k] = dict(e=float(sum(x[0] for x in ph)), phases=ph,
                    occ=cat[k]["occupancy"], bytes=memory_bytes_per_inference(cat[k], p, kappa),
                    prio=-1 if k == "obstacle" else 0)
    return T
MIXED = dict(obstacle="int8", product="fp16", ocr="fp16", tracking="fp16", asr="fp16")
ALLFP16 = dict(obstacle="fp16", product="fp16", ocr="fp16", tracking="fp16", asr="fp16")
ALLFP32 = {k: "fp32" for k in MIXED}
ALLINT8 = {k: "int8" for k in MIXED}

# bandwidth demand table
bwd = {}
for k in ORDER + ["asr"]:
    T = make_tasks(MIXED)
    d, o, b = T[k]["phases"][1]
    bwd[k] = b / d / 1e6  # GB/s during throughput phase
R["bw_demand_mixed"] = bwd
R["bw_demand_sum_oncycle"] = float(sum(bwd[k] for k in ORDER))

# ---------------- Experiment 1: policy comparison (nominal) ----------------
POLS = ["sequential", "equal", "priority", "priority_throttled_asr"]
LABEL = {"sequential": "Sequential (1 stream)", "equal": "Parallel, equal priority",
         "priority": "Parallel, priority-aware", "priority_throttled_asr": "Priority-aware + ASR throttled"}
def run_policy(pol, tasks, **kw):
    thr = 0.5 if pol == "priority_throttled_asr" else 1.0
    p = "priority" if pol == "priority_throttled_asr" else pol
    return simulate(tasks, p, asr_throttle=thr, **kw)

E1 = {}
for cfg_name, pm in [("mixed", MIXED), ("allfp16", ALLFP16), ("allfp32", ALLFP32), ("allint8", ALLINT8)]:
    tasks = make_tasks(pm)
    E1[cfg_name] = {"e": {k: tasks[k]["e"] for k in tasks}}
    for pol in POLS:
        res = run_policy(pol, tasks, n_cycles=500, asr_rate_per_s=2 / 60.0, seed=11)
        s = summarize(res)
        E1[cfg_name][pol] = s
        if cfg_name == "mixed":
            E1[cfg_name][pol + "_crit"] = res["crit"].tolist()
            E1[cfg_name][pol + "_obs"] = res["finish"]["obstacle"].tolist()
R["E1"] = E1

# paired comparison priority vs equal, same seed (common random numbers)
tasks = make_tasks(MIXED)
ra = run_policy("equal", tasks, n_cycles=500, asr_rate_per_s=2 / 60.0, seed=11)
rb = run_policy("priority", tasks, n_cycles=500, asr_rate_per_s=2 / 60.0, seed=11)
d = ra["finish"]["obstacle"] - rb["finish"]["obstacle"]
from scipy import stats
R["paired_obs"] = dict(mean_diff=float(d.mean()), sd=float(d.std(ddof=1)),
                       ci95=[float(x) for x in stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))],
                       t=float(stats.ttest_rel(ra["finish"]["obstacle"], rb["finish"]["obstacle"]).statistic),
                       p=float(stats.ttest_rel(ra["finish"]["obstacle"], rb["finish"]["obstacle"]).pvalue),
                       wilcoxon_p=float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
dc = ra["crit"] - rb["crit"]
R["paired_crit"] = dict(mean_diff=float(dc.mean()), ci95=[float(x) for x in stats.t.interval(0.95, len(dc) - 1, loc=dc.mean(), scale=stats.sem(dc))],
                        p=float(stats.ttest_rel(ra["crit"], rb["crit"]).pvalue))

# ---------------- Experiment 2: sensitivity — latency multiplier x kernel occupancy ----------------
effs = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
kappas = [0.3, 0.5, 0.7, 1.0]   # (variable name kept for the plotting code) = occupancy scale
grid = {pol: np.zeros((len(kappas), len(effs))) for pol in ["sequential", "equal", "priority"]}
grid_obs = {pol: np.zeros((len(kappas), len(effs))) for pol in ["equal", "priority"]}
for i, oc in enumerate(kappas):
    for j, ef in enumerate(effs):
        tasks = make_tasks(MIXED, eff_mult=ef, occ_scale=oc)
        for pol in grid:
            res = run_policy(pol, tasks, n_cycles=200, asr_rate_per_s=2 / 60.0, seed=5)
            s = summarize(res)
            grid[pol][i, j] = s["crit_p99"]
            if pol in grid_obs:
                grid_obs[pol][i, j] = s["obs_p99"]
R["E2"] = dict(effs=effs, occs=kappas, **{k: v.tolist() for k, v in grid.items()},
               **{"obs_" + k: v.tolist() for k, v in grid_obs.items()})

# ---------------- Experiment 3: ASR arrival rate sweep ----------------
rates = [0, 1, 2, 5, 10, 20, 40]  # per minute
E3 = {pol: [] for pol in POLS}
for r in rates:
    tasks = make_tasks(MIXED)
    for pol in POLS:
        res = run_policy(pol, tasks, n_cycles=400, asr_rate_per_s=r / 60.0, seed=21)
        E3[pol].append(summarize(res))
R["E3"] = dict(rates=rates, **E3)

# ---------------- Experiment 4: jitter CV sweep ----------------
cvs = [0.02, 0.05, 0.08, 0.12, 0.2, 0.3]
E4 = {pol: [] for pol in ["sequential", "equal", "priority"]}
for cv in cvs:
    tasks = make_tasks(MIXED)
    for pol in E4:
        res = run_policy(pol, tasks, n_cycles=1000, cv=cv, asr_rate_per_s=0.0, seed=31)
        E4[pol].append(summarize(res))
R["E4"] = dict(cvs=cvs, **E4)

# ---------------- Experiment 5: load headroom envelope (scale OCR/ReID crops, depth resolution, cameras) ----------------
E5 = []
for depth_res in [364, 518, 728]:
    for crops in [5, 10, 20, 40]:
        c2 = build_catalog(ocr_crops=crops, reid_crops=max(4, crops // 2), depth_res=depth_res)
        tasks = make_tasks(MIXED, cat=c2)
        row = dict(depth_res=depth_res, crops=crops, e=tasks["obstacle"]["e"], e_ocr=tasks["ocr"]["e"], e_trk=tasks["tracking"]["e"],
                   seq_sum=sum(tasks[k]["e"] for k in ORDER))
        for pol in ["sequential", "equal", "priority"]:
            res = run_policy(pol, tasks, n_cycles=200, asr_rate_per_s=5 / 60.0, seed=41)
            s = summarize(res)
            row[pol] = dict(p99=s["crit_p99"], miss=s["miss_T"], obs_p99=s["obs_p99"], barrier_miss=s["miss_barrier"])
        E5.append(row)
R["E5"] = E5

# ---------------- Experiment 6: kernel occupancy (concurrency potential) x memory intensity ----------------
occs = [0.3, 0.5, 0.7, 1.0]
E6 = []
for oc in occs:
    for kap in [0.002, 0.004, 0.008]:
        tasks = make_tasks(MIXED, kappa=kap, occ_scale=oc)
        row = dict(occ_scale=oc, kappa=kap)
        for pol in ["sequential", "equal", "priority"]:
            res = run_policy(pol, tasks, n_cycles=200, asr_rate_per_s=2 / 60.0, seed=7)
            sm = summarize(res)
            row[pol] = dict(crit_mean=sm["crit_mean"], crit_p99=sm["crit_p99"], obs_mean=sm["obs_mean"], obs_p99=sm["obs_p99"])
        E6.append(row)
R["E6"] = E6

# ---------------- Quantization theory numbers ----------------
Q = {}
Q["opt_clip_laplace"], Q["mse_laplace"] = [float(x) for x in optimal_clip("laplace")]
Q["opt_clip_gauss"], Q["mse_gauss"] = [float(x) for x in optimal_clip("gauss")]
Q["rel_mse_laplace"] = Q["mse_laplace"] / 2.0   # variance of Laplace(0,1)=2
Q["rel_mse_gauss"] = Q["mse_gauss"] / 1.0
Q["laplace_over_gauss"] = Q["rel_mse_laplace"] / Q["rel_mse_gauss"]
Q["ratio_k"] = {k: float(int8_mse_uniform(k) / fp16_mse(1.0)) for k in [1, 2, 3, 4]}
Q["sqnr_int8_gauss_db"] = 10 * np.log10(1 / Q["rel_mse_gauss"])
Q["sqnr_int8_laplace_db"] = 10 * np.log10(1 / Q["rel_mse_laplace"])
Q["sqnr_fp16_db"] = 10 * np.log10(1 / (fp16_mse(1.0)))
R["Q"] = Q

# ---------------- Imaging ----------------
IM = []
for E in [200, 500, 700, 1000]:
    for iso in [800, 1600, 3200]:
        t = exposure_time_s(E, iso, 1.8)
        IM.append(dict(lux=E, iso=iso, t_ms=t * 1e3, blur_px=blur_pixels(0.8, t, 0.016, 0.4, 1.55e-6)))
R["imaging"] = IM
R["blur_limit_exposure_ms"] = 3 * 0.4 * 1.55e-6 / (0.8 * 0.016) * 1e3

# ---------------- Safety ----------------
SF = []
T_cycle = 0.1; V = 0.8; D_FLAG = 0.6
for a in [0.8, 1.2, 2.0]:
    tau_max = (D_FLAG - V**2 / (2 * a)) / V
    for pol in ["sequential", "equal", "priority"]:
        sm = E1["mixed"][pol]
        # perception-to-brake latency: exposure+readout (10 ms) + half-cycle sampling delay + obstacle p99 finish + planner/actuation (30 ms)
        tau = 0.010 + T_cycle / 2 + sm["obs_p99"] / 1000 + 0.030
        ds = stopping_distance(V, tau, a)
        slack_s = tau_max - tau
        k = int(np.floor(slack_s / T_cycle))  # extra frames in which a miss can be recovered
        pu = (1 - 0.986) ** (k + 1) if k >= 0 else 1.0
        SF.append(dict(a=a, pol=pol, tau_ms=tau * 1e3, tau_max_ms=tau_max * 1e3, d_stop=ds, slack_ms=slack_s * 1e3, frames=k + 1, p_undetected=pu))
R["safety"] = SF

json.dump(R, open(OUT + "/results.json", "w"), indent=1, default=float)


print("cart experiments E1-E6 complete -> results/results.json")
