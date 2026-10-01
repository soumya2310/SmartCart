import json, numpy as np, os
from models import *
from sched_sim import simulate, summarize, ORDER

OUT = "results"
fit = fit_latency_model(); cat = build_catalog()
MIXED = dict(obstacle="int8", product="fp16", ocr="fp16", tracking="fp16", asr="fp16")
def make_tasks(prec_map=MIXED, kappa=0.004, eff_mult=1.0):
    T = {}
    for k in ORDER + ["asr"]:
        ph = task_phases(cat[k], prec_map[k], fit, kappa=kappa, eff_mult=eff_mult)
        T[k] = dict(e=float(sum(x[0] for x in ph)), phases=ph, prio=-1 if k == "obstacle" else 0)
    return T
R = {}
S_EDGE = edge_service_time_ms(fit); R["s_edge_ms"] = S_EDGE
R["edge_gpu"] = EDGE_GPU
NET = dict(edge_rtt_ms=5.0, cloud_rtt_ms=40.0, uplink_mbps=50.0, payload_kb=12.0, encode_ms=5.0, timeout_ms=300.0)
R["net"] = NET
# offloadability table inputs: per-task uplink demand at 10 Hz
R["uplink_demand"] = dict(product_frame_kb=120, product_mbps_per_cart=120 * 8 * 10 / 1000, ocr_crops_kb=15, ocr_mbps_per_cart=15 * 8 * 10 / 1000,
                          depth_frame_kb=100, depth_mbps_per_cart=100 * 8 * 10 / 1000, voice_kb_per_query=12)

# ---------------- E7: on-device vs edge vs cloud at nominal and at high query rate ----------------
E7 = {}
for lam_min in [2, 20]:
    tasks = make_tasks()
    row = {}
    # on-device
    res = simulate(tasks, "priority", n_cycles=1000, asr_rate_per_s=lam_min / 60, seed=11)
    s = summarize(res); rr = res["asr_resp"]
    row["on_device"] = dict(crit_p99=s["crit_p99"], crit_mean=s["crit_mean"], obs_p99=s["obs_p99"],
                            resp_mean=float(rr.mean()) if len(rr) else None, resp_p99=float(np.percentile(rr, 99)) if len(rr) > 5 else float(rr.max()) if len(rr) else None, n=int(len(rr)))
    for mode, rtt, c in [("edge", NET["edge_rtt_ms"], 1), ("cloud", NET["cloud_rtt_ms"], 64)]:
        # fleet of 50 carts sharing the server
        off = offload_response_ms(rtt, NET["uplink_mbps"], NET["payload_kb"], S_EDGE, 50 * lam_min / 60, c, NET["encode_ms"])
        res = simulate(tasks, "priority", n_cycles=1000, asr_rate_per_s=lam_min / 60, seed=11, p_local=0.0)
        s = summarize(res)
        row[mode] = dict(crit_p99=s["crit_p99"], crit_mean=s["crit_mean"], obs_p99=s["obs_p99"], resp_mean=off["mean"], resp_p99=off["p99"], util=off["util"])
    E7[str(lam_min)] = row
R["E7"] = E7

# ---------------- E8: fleet scalability — response vs number of carts, edge c=1,2,4; cloud elastic ----------------
Ns = [1, 5, 10, 20, 50, 100, 150, 200, 300]
E8 = dict(N=Ns, lam_min=2)
for c in [1, 2, 4]:
    E8["edge_c%d" % c] = [offload_response_ms(NET["edge_rtt_ms"], NET["uplink_mbps"], NET["payload_kb"], S_EDGE, N * 2 / 60, c, NET["encode_ms"]) for N in Ns]
E8["cloud"] = [offload_response_ms(NET["cloud_rtt_ms"], NET["uplink_mbps"], NET["payload_kb"], S_EDGE, N * 2 / 60, max(4, int(np.ceil(N * 2 / 60 * S_EDGE / 1000 / 0.3))), NET["encode_ms"]) for N in Ns]
# capacity: max carts per L4 at 2/min such that p99 wait < 50 ms
def max_carts(c, lam_min=2, lim=50):
    N = 1
    while offload_response_ms(NET["edge_rtt_ms"], NET["uplink_mbps"], NET["payload_kb"], S_EDGE, N * lam_min / 60, c)["wq_p99"] < lim and N < 5000:
        N += 1
    return N - 1
E8["max_carts_p99wait50"] = {("c%d" % c): max_carts(c) for c in [1, 2, 4]}
E8["max_carts_20permin"] = {("c%d" % c): max_carts(c, 20) for c in [1, 2, 4]}
R["E8"] = E8

# ---------------- E9: RTT crossover ----------------
rtts = [2, 5, 10, 20, 40, 60, 80, 100, 150, 200]
E9 = dict(rtt=rtts)
for lam_min in [2, 20]:
    E9["off_p99_%d" % lam_min] = [offload_response_ms(r, NET["uplink_mbps"], NET["payload_kb"], S_EDGE, 50 * lam_min / 60, 2, NET["encode_ms"])["p99"] for r in rtts]
    E9["off_mean_%d" % lam_min] = [offload_response_ms(r, NET["uplink_mbps"], NET["payload_kb"], S_EDGE, 50 * lam_min / 60, 2, NET["encode_ms"])["mean"] for r in rtts]
    E9["dev_mean_%d" % lam_min] = E7[str(lam_min)]["on_device"]["resp_mean"]
    E9["dev_p99_%d" % lam_min] = E7[str(lam_min)]["on_device"]["resp_p99"]
# crossover: RTT at which offload mean == on-device mean
for lam_min in [2, 20]:
    dev = E9["dev_mean_%d" % lam_min]
    xs = np.array(rtts); ys = np.array(E9["off_mean_%d" % lam_min])
    E9["crossover_rtt_%d" % lam_min] = float(np.interp(dev, ys, xs)) if ys[-1] > dev > ys[0] else None
R["E9"] = E9

# ---------------- E10: connectivity outage / fallback ----------------
pouts = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
E10 = dict(p_out=pouts)
for lam_min in [2, 20]:
    rows = []
    tasks = make_tasks()
    for po in pouts:
        res = simulate(tasks, "priority", n_cycles=1000, asr_rate_per_s=lam_min / 60, seed=11, p_local=po, fallback_delay=NET["timeout_ms"])
        s = summarize(res); rr = res["asr_resp"]
        off = offload_response_ms(NET["edge_rtt_ms"], NET["uplink_mbps"], NET["payload_kb"], S_EDGE, 50 * lam_min / 60, 2, NET["encode_ms"])
        # blended response: (1-po) offloaded, po fallback (timeout + local)
        fb_mean = float(rr.mean()) if len(rr) else 0.0
        blended_mean = (1 - po) * off["mean"] + po * fb_mean
        rows.append(dict(p_out=po, crit_p99=s["crit_p99"], crit_mean=s["crit_mean"], obs_p99=s["obs_p99"], n_local=res["n_local"], fb_resp_mean=fb_mean, blended_resp_mean=blended_mean))
    E10["lam%d" % lam_min] = rows
R["E10"] = E10

json.dump(R, open(OUT + "/results_cloud.json", "w"), indent=1, default=float)


print("continuum experiments E7-E10 complete -> results/results_cloud.json")
