"""Print the numbers behind the manuscript's tables from results/results.json and
results/results_cloud.json. Run after the two experiment scripts:
    python3 print_tables.py
Table numbers refer to the manuscript. Values are rounded as in the paper."""
import json, sys
R = json.load(open("results/results.json")); C = json.load(open("results/results_cloud.json"))
F, L, Llo, Lhi, E1, M = R["fit"], R["latency"], R["latency_lo"], R["latency_hi"], R["E1"], R["E1"]["mixed"]
pct = lambda x: "%.1f%%" % (100 * x)
LAB = {"sequential": "Sequential", "equal": "Equal priority", "priority": "Priority-aware", "priority_throttled_asr": "Priority-aware, throttled"}

def hdr(t): print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)

hdr("Table 4 - Latency-model calibration (Section 4.1)")
for p in ["fp32", "fp16", "int8"]:
    f = F[p]; print("%-6s a=%.2f ms  1/T=%.4f ms/GFLOP  T_eff=%.1f  eta=%.2f  MAPE=%.1f%%  maxAPE=%.1f%%" % (p.upper(), f["a_ms"], f["ms_per_gflop"], f["eff_tflops"], f["eta"], f["mape"], f["maxape"]))
print("gamma_T = %.2f  (range %.2f-%.2f)" % (R["gamma_T"], *R["gamma_T_range"]))

hdr("Table 6 - Predicted isolated latencies (ms)")
for k in ["obstacle", "product", "ocr", "tracking", "asr"]:
    print("%-9s FP32 %6.1f [%5.1f-%5.1f]   FP16 %5.1f [%5.1f-%5.1f]   INT8 %5.1f [%5.1f-%5.1f]" % (k, L[k]["fp32"], Llo[k]["fp32"], Lhi[k]["fp32"], L[k]["fp16"], Llo[k]["fp16"], Lhi[k]["fp16"], L[k]["int8"], Llo[k]["int8"], Lhi[k]["int8"]))
for cfg in ["allfp32", "allfp16", "allint8", "mixed"]:
    print("sum of on-cycle tasks (%s): %.1f ms" % (cfg, sum(E1[cfg]["e"][k] for k in ["obstacle", "product", "ocr", "tracking"])))
print("Section 4.2 bandwidth demand (GB/s):", {k: round(v, 1) for k, v in R["bw_demand_mixed"].items()}, "on-cycle sum %.0f" % R["bw_demand_sum_oncycle"])

hdr("Table 7 - Policy comparison at the mixed-precision design point (ms)")
print("%-28s %-14s %6s %6s %6s %-14s %8s %9s %9s" % ("policy", "crit mean+-sd", "p50", "p99", "max", "obst mean+-sd", "obst p99", "barrier", "deadline"))
for p in ["sequential", "equal", "priority", "priority_throttled_asr"]:
    s = M[p]; print("%-28s %5.1f +- %-5.1f %6.1f %6.1f %6.1f %5.1f +- %-5.1f %8.1f %9s %9s" % (LAB[p], s["crit_mean"], s["crit_sd"], s["crit_p50"], s["crit_p99"], s["crit_max"], s["obs_mean"], s["obs_sd"], s["obs_p99"], pct(s["miss_barrier"]), pct(s["miss_T"])))
print("paired obstacle difference equal-priority: %.2f ms, 95%% CI [%.2f, %.2f]" % (R["paired_obs"]["mean_diff"], *R["paired_obs"]["ci95"]))
print("paired critical-path difference: %.2f ms, 95%% CI [%.2f, %.2f]" % (R["paired_crit"]["mean_diff"], *R["paired_crit"]["ci95"]))
for cfg in ["allfp32", "allfp16", "allint8"]:
    for p in ["sequential", "equal", "priority"]:
        s = E1[cfg][p]; print("  %-8s %-16s crit %.1f p99 %.1f obst %.1f p99 %.1f barrier %s deadline %s" % (cfg, LAB[p], s["crit_mean"], s["crit_p99"], s["obs_mean"], s["obs_p99"], pct(s["miss_barrier"]), pct(s["miss_T"])))

hdr("Table 8 - Contention regimes E6 (ms)")
for r in R["E6"]:
    print("occ %.1f kappa %.3f  seq %.1f  equal %.1f (obst %.1f)  priority %.1f (obst %.1f)  benefit %.1f" % (r["occ_scale"], r["kappa"], r["sequential"]["crit_mean"], r["equal"]["crit_mean"], r["equal"]["obs_mean"], r["priority"]["crit_mean"], r["priority"]["obs_mean"], r["equal"]["obs_mean"] - r["priority"]["obs_mean"]))

hdr("Figure 5 grid E2 - p99 critical path (rows: occupancy scale %s; cols: latency multiplier %s)" % (R["E2"]["occs"], R["E2"]["effs"]))
for key in ["sequential", "priority", "obs_equal", "obs_priority"]:
    print(key); [print("   ", [round(v) for v in row]) for row in R["E2"][key]]

hdr("Figure 6 / E3 - obstacle p99 and critical-path p99 vs voice-query rate (per minute) %s" % R["E3"]["rates"])
for p in ["sequential", "equal", "priority", "priority_throttled_asr"]:
    print("%-28s obst p99 %s   crit p99 %s" % (LAB[p], [round(s["obs_p99"], 1) for s in R["E3"][p]], [round(s["crit_p99"], 1) for s in R["E3"][p]]))
print("E4 jitter sweep cv %s" % R["E4"]["cvs"])
for p in ["sequential", "equal", "priority"]:
    print("%-28s crit p99 %s" % (LAB[p], [round(s["crit_p99"], 1) for s in R["E4"][p]]))

hdr("Table 9 - Load headroom E5 (ms)")
for r in R["E5"]:
    print("res %d crops %2d  e_obst %.1f sum %.1f  seq p99 %.1f  equal p99 %.1f (obst %.1f)  priority p99 %.1f (obst %.1f)" % (r["depth_res"], r["crops"], r["e"], r["seq_sum"], r["sequential"]["p99"], r["equal"]["p99"], r["equal"]["obs_p99"], r["priority"]["p99"], r["priority"]["obs_p99"]))

hdr("Section 4.4 / Supplement S1 - quantization theory")
Q = R["Q"]; print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in Q.items()})

hdr("Table 11 - Safety budget")
for x in R["safety"]:
    print("a %.1f %-16s tau %.0f ms  tau_max %.0f ms  d_stop %.2f m  slack %.0f ms  frames %d" % (x["a"], LAB[x["pol"]], x["tau_ms"], x["tau_max_ms"], x["d_stop"], x["slack_ms"], x["frames"]))

hdr("Section 4.8 / Table 12 - Offloading (E7)")
print("edge service time s_edge = %.1f ms; network: %s" % (C["s_edge_ms"], C["net"]))
for lam in ["2", "20"]:
    for m in ["on_device", "edge", "cloud"]:
        s = C["E7"][lam][m]; print("%2s q/min %-10s crit mean %.1f p99 %.1f obst p99 %.1f  resp mean %.0f p99 %.0f  util %s" % (lam, m, s["crit_mean"], s["crit_p99"], s["obs_p99"], s["resp_mean"], s["resp_p99"], s.get("util", "-")))

hdr("Figure 7 / E8 - fleet capacity")
print("max carts with p99 queueing < 50 ms at 2 q/min:", C["E8"]["max_carts_p99wait50"], " at 20 q/min:", C["E8"]["max_carts_20permin"])
for key in ["edge_c1", "edge_c2", "edge_c4", "cloud"]:
    print("%-8s carts %s -> mean %s / p99 %s" % (key, C["E8"]["N"], [round(r["mean"]) for r in C["E8"][key]], [round(r["p99"]) for r in C["E8"][key]]))

hdr("Figure 8 / E9-E10 - RTT crossover and fallback")
print("crossover RTT (ms): 2 q/min %.1f, 20 q/min %.1f" % (C["E9"]["crossover_rtt_2"], C["E9"]["crossover_rtt_20"]))
for lam in ["lam2", "lam20"]:
    print(lam, [(r["p_out"], round(r["crit_p99"], 1), round(r["blended_resp_mean"])) for r in C["E10"][lam]])
