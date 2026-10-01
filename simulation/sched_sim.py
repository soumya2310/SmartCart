"""Two-phase fluid simulator of CUDA stream co-scheduling (see paper Section 5)."""
import numpy as np

ORDER = ["obstacle", "product", "ocr", "tracking"]

def simulate(tasks, policy, n_cycles=500, T=100.0, dt=0.05, bw=204.8, cv=0.08,
             asr_rate_per_s=0.0, e_slam=9.0, seed=0, asr_throttle=1.0, bw_eff=0.85,
             max_overrun=4.0, p_local=1.0, fallback_delay=0.0):
    """
    tasks: dict name -> dict(phases=[(dur_ms, occ, bytes), ...], prio=int)
    Each phase runs at rate rho in [0,1]; compute draw = occ*rho; bandwidth draw = (bytes/dur)*rho.
    Per-cycle jitter: all phase durations of a task scaled by one LogNormal(0,cv) draw.
    """
    rng = np.random.default_rng(seed)
    BW = bw * bw_eff * 1e9 / 1e3  # bytes per ms usable
    names = ORDER
    n = len(names)
    jit = {k: rng.lognormal(0.0, cv, n_cycles) for k in names + ["asr"]}
    slam = e_slam * rng.lognormal(0.0, cv, n_cycles)
    horizon = n_cycles * T
    n_arr = rng.poisson(asr_rate_per_s * horizon / 1000.0)
    all_arr = np.sort(rng.uniform(0, horizon, n_arr))
    keep = rng.uniform(size=n_arr) < p_local          # queries served on the cart GPU (all, or fallbacks)
    asr_queue = list(all_arr[keep] + fallback_delay)   # fallbacks start after the offload timeout
    asr_arrival_times = list(all_arr[keep])
    asr_resp = []
    asr_state = None
    asr_busy_ms = 0.0
    finish = {k: np.zeros(n_cycles) for k in names}
    iso = {k: np.zeros(n_cycles) for k in names}
    crit = np.zeros(n_cycles)
    def new_state(k, j):
        ph = [(d * j, o, b) for (d, o, b) in tasks[k]["phases"]]
        return dict(phases=ph, idx=0, work=1.0, prio=tasks[k]["prio"])
    for c in range(n_cycles):
        t0 = c * T
        st = {k: new_state(k, jit[k][c]) for k in names}
        for k in names:
            iso[k][c] = sum(p[0] for p in st[k]["phases"])
        done = {k: None for k in names}
        seq_idx = 0
        t = 0.0
        while (any(v is None for v in done.values()) or t < T) and t < max_overrun * T:
            tnow = t0 + t
            if asr_state is None and asr_queue and asr_queue[0] <= tnow:
                asr_queue.pop(0)
                asr_state = new_state("asr", jit["asr"][c]); asr_state["cap"] = asr_throttle
                asr_state["t_arr"] = asr_arrival_times.pop(0)
            # fast path: no on-cycle task active -> advance ASR alone (or idle) analytically
            if all(v is not None for v in done.values()):
                next_arr = (asr_queue[0] - t0) if asr_queue else float("inf")
                if asr_state is None:
                    t = min(T, max(t, next_arr)) if next_arr < T else T
                    continue
                d, o, b = asr_state["phases"][asr_state["idx"]]
                rate = asr_state["cap"]
                t_fin = asr_state["work"] * d / rate
                step = min(t_fin, T - t)
                if step <= 0:
                    break
                asr_state["work"] -= rate * step / d
                asr_busy_ms += step
                t += step
                if asr_state["work"] <= 1e-9:
                    asr_state["idx"] += 1; asr_state["work"] = 1.0
                    if asr_state["idx"] >= len(asr_state["phases"]):
                        asr_resp.append(t0 + t - asr_state["t_arr"])
                        asr_state = None
                continue
            if policy == "sequential":
                while seq_idx < n and done[names[seq_idx]] is not None:
                    seq_idx += 1
                active = [names[seq_idx]] if seq_idx < n else []
            else:
                active = [k for k in names if done[k] is None]
            entries = []
            for k in active:
                s = st[k]; d, o, b = s["phases"][s["idx"]]
                entries.append(dict(name=k, occ=o, beta=b / d, prio=(s["prio"] if policy != "equal" else 0), cap=1.0, dur=d))
            if asr_state is not None:
                s = asr_state; d, o, b = s["phases"][s["idx"]]
                entries.append(dict(name="asr", occ=o, beta=b / d, prio=s["prio"], cap=s["cap"], dur=d))
            rho = {}
            remaining = 1.0
            for p in sorted(set(en["prio"] for en in entries)):
                grp = [en for en in entries if en["prio"] == p]
                demand = sum(en["occ"] * en["cap"] for en in grp)
                share = 1.0 if demand <= remaining + 1e-12 else remaining / demand
                for en in grp:
                    rho[en["name"]] = en["cap"] * share
                remaining -= min(demand, remaining)
            bw_demand = sum(en["beta"] * rho[en["name"]] for en in entries)
            if bw_demand > BW:
                scale = BW / bw_demand
                for k in rho:
                    rho[k] *= scale
            for en in entries:
                k = en["name"]
                s = asr_state if k == "asr" else st[k]
                s["work"] -= rho[k] * dt / en["dur"]
                if k == "asr":
                    asr_busy_ms += dt
                if s["work"] <= 0:
                    s["idx"] += 1; s["work"] = 1.0
                    if s["idx"] >= len(s["phases"]):
                        if k == "asr":
                            asr_resp.append(tnow + dt - asr_state["t_arr"])
                            asr_state = None
                        else:
                            done[k] = t + dt
            t += dt
        for k in names:
            finish[k][c] = done[k] if done[k] is not None else max_overrun * T
        crit[c] = max(max(finish[k][c] for k in names), slam[c])
    return dict(finish=finish, iso=iso, crit=crit, slam=slam, asr_busy_frac=asr_busy_ms / horizon,
                asr_resp=np.array(asr_resp), n_asr=int(n_arr), n_local=int(keep.sum()))

def summarize(res, T=100.0, barrier=80.0):
    crit = res["crit"]
    obs = res["finish"]["obstacle"]
    s = dict(
        crit_mean=crit.mean(), crit_sd=crit.std(ddof=1), crit_p50=np.percentile(crit, 50),
        crit_p99=np.percentile(crit, 99), crit_max=crit.max(),
        obs_mean=obs.mean(), obs_sd=obs.std(ddof=1), obs_p99=np.percentile(obs, 99),
        miss_T=float(np.mean(crit > T)), miss_barrier=float(np.mean(obs > barrier)),
        asr_busy=res["asr_busy_frac"],
    )
    for k in ORDER:
        s[k + "_mean"] = res["finish"][k].mean()
    return s
