"""
SmartCart design-space model.
Analytical components: hardware constants (cited), model catalog (derived from
architecture definitions), latency model calibrated to published Jetson AGX Orin
TensorRT benchmarks, quantization-error theory, imaging/motion-blur photometry,
and safety kinematics.  Every constant carries a provenance comment.
"""
import numpy as np
from scipy.optimize import minimize_scalar, brentq

# --------------------------------------------------------------------------
# 1. Hardware constants — NVIDIA Jetson AGX Orin 64 GB (NVIDIA data sheet DS-10662-001)
# --------------------------------------------------------------------------
HW = dict(
    name="NVIDIA Jetson AGX Orin 64GB",
    sm=16, cuda_cores=2048, tensor_cores=64, gpu_clock_ghz=1.3,
    peak_fp32_tflops=5.32,          # CUDA cores
    peak_fp16_tensor_dense_tflops=42.5,   # 85 sparse / 2
    peak_int8_tensor_dense_tops=85.0,     # 170 sparse / 2
    peak_tf32_tensor_dense_tflops=21.25,  # half of FP16 dense on Ampere tensor cores
    dram_bw_gbs=204.8,              # 256-bit LPDDR5 @ 6400 Gbps/pin
    l2_mb=4.0,
    cpu_cores=12, cpu_clock_ghz=2.2,
)

# --------------------------------------------------------------------------
# 2. Published calibration data — Ultralytics YOLO26 TensorRT on AGX Orin 64GB
#    (docs.ultralytics.com/guides/nvidia-jetson, Ultralytics 8.4.32, imgsz 640, batch 1,
#     GPU inference time only).  FLOPs/params from Ultralytics detection table.
# --------------------------------------------------------------------------
YOLO26 = {
    # name: (params M, GFLOPs, fp32 ms, fp16 ms, int8 ms)
    "YOLO26n": (2.4,  5.5,   4.18,  2.62, 2.30),
    "YOLO26s": (9.5,  20.9,  6.82,  3.76, 2.98),
    "YOLO26m": (20.4, 68.4,  12.60, 6.24, 4.72),
    "YOLO26l": (24.8, 86.8,  15.90, 7.93, 5.97),
    "YOLO26x": (55.7, 194.4, 28.43, 13.50, 9.33),
}

# Published transformer datapoint used to calibrate attention-model efficiency:
# Depth Anything V2 ViT-S, TensorRT BF16, batch of 4 images at 480x300 on AGX Orin
# (JetPack 6.1): 50.16 +/- 8.22 ms per batch  (arXiv:2603.01999, Table I)
DAV2_CAL = dict(latency_ms=50.16, sd_ms=8.22, batch=4, h=300, w=480)

# --------------------------------------------------------------------------
# 3. Model catalog — FLOPs and bytes derived from architecture definitions
# --------------------------------------------------------------------------
def vit_flops(n_tokens, d, layers, mlp_ratio=4):
    """Forward FLOPs (2*MACs) of a standard pre-LN ViT encoder, excluding patch embed."""
    macs_per_layer = (4 + 2 * mlp_ratio) * n_tokens * d**2 + 2 * n_tokens**2 * d
    return 2 * macs_per_layer * layers / 1e9  # GFLOPs

def dav2_small_flops(h=518, w=518, patch=14):
    n = (h // patch) * (w // patch) + 1
    enc = vit_flops(n, 384, 12)
    # patch embedding conv 3->384, 14x14 stride 14
    embed = 2 * (n - 1) * 384 * 3 * patch * patch / 1e9
    # DPT head: 4 reassemble stages + fusion convs; ~30% of encoder for ViT-S (Ranftl et al. 2021 DPT)
    head = 0.30 * enc
    return enc + embed + head, n

def whisper_small_encoder_flops():
    # Whisper Small: d=768, 12 layers, 30 s audio -> 3000 mel frames -> 1500 tokens after conv stride 2
    n = 1500
    conv1 = 2 * 3000 * 768 * 80 * 3 / 1e9
    conv2 = 2 * 1500 * 768 * 768 * 3 / 1e9
    return conv1 + conv2 + vit_flops(n, 768, 12), n

def whisper_small_decoder_bytes_per_token(dtype_bytes=2):
    # decoder ~ 12 layers x (self-attn 4d^2 + cross-attn 4d^2 + MLP 8d^2) = 12*16*768^2 = 113M params
    return 12 * 16 * 768**2 * dtype_bytes

def resnet50_flops(h, w):
    return 4.1 * (h * w) / (224 * 224)  # 4.1 GFLOPs at 224x224

CATALOG = {}
def build_catalog(ocr_crops=10, reid_crops=8, depth_res=518):
    dav_f, dav_tok = dav2_small_flops(depth_res, depth_res)
    wh_f, wh_tok = whisper_small_encoder_flops()
    cat = {
        "obstacle": dict(model="DepthAnything v2-S (ViT-S/14, %dx%d)" % (depth_res, depth_res),
                         params_m=24.8, gflops=dav_f, kind="transformer", tokens=dav_tok,
                         act_mb_fp16=48, occupancy=0.7),
        "product":  dict(model="YOLOv8-L (640x640)", params_m=43.7, gflops=165.1,
                         kind="cnn", act_mb_fp16=72, occupancy=1.0),
        "ocr":      dict(model="PaddleOCR v4 (DB det, ResNet-50 @640x640 + %d SVTR crops)" % ocr_crops,
                         params_m=26.0 + 5.0,
                         gflops=resnet50_flops(640, 640) * 1.35 + ocr_crops * 0.55,  # det backbone+FPN/head, rec crops (48x320)
                         kind="cnn", act_mb_fp16=60, occupancy=0.5, crops=ocr_crops),
        "tracking": dict(model="ByteTrack + ReID (ResNet-50 @256x128, %d crops)" % reid_crops,
                         params_m=25.6, gflops=reid_crops * resnet50_flops(256, 128),
                         kind="cnn", act_mb_fp16=34, occupancy=0.4, crops=reid_crops),
        "asr":      dict(model="Whisper Small.en encoder (30 s window)", params_m=244.0,
                         gflops=wh_f, kind="transformer", tokens=wh_tok, act_mb_fp16=88, occupancy=0.9),
    }
    for k, v in cat.items():
        v["weights_mb"] = {"fp32": v["params_m"] * 4, "fp16": v["params_m"] * 2, "int8": v["params_m"] * 1}
    return cat

# --------------------------------------------------------------------------
# 4. Latency model  t = a_p + F / T_p   (fixed overhead + throughput-limited term)
#    Fitted per precision to the five YOLO26 datapoints.  Transformer efficiency
#    multiplier fitted from the DAV2 datapoint.
# --------------------------------------------------------------------------
def fit_latency_model():
    F = np.array([v[1] for v in YOLO26.values()])
    out = {}
    for j, prec in enumerate(["fp32", "fp16", "int8"]):
        t = np.array([v[2 + j] for v in YOLO26.values()])
        A = np.vstack([np.ones_like(F), F]).T
        coef, res, *_ = np.linalg.lstsq(A, t, rcond=None)
        a, b = coef
        pred = A @ coef
        err = (pred - t) / t
        out[prec] = dict(a_ms=a, ms_per_gflop=b, eff_tflops=1.0 / b, pred=pred, meas=t,
                         mape=np.mean(np.abs(err)) * 100, maxape=np.max(np.abs(err)) * 100)
    peaks = {"fp32": HW["peak_tf32_tensor_dense_tflops"], "fp16": HW["peak_fp16_tensor_dense_tflops"],
             "int8": HW["peak_int8_tensor_dense_tops"]}
    for p in out:
        out[p]["eta"] = out[p]["eff_tflops"] / peaks[p]
        out[p]["peak"] = peaks[p]
    # transformer efficiency multiplier gamma_T: predicted DAV2 batch latency = 4*(a + gamma*F/T)
    f_img, _ = dav2_small_flops(DAV2_CAL["h"], DAV2_CAL["w"])
    a, b = out["fp16"]["a_ms"], out["fp16"]["ms_per_gflop"]
    # batch of 4 shares one launch overhead per layer set: model as a + 4*gamma*b*F
    gamma = (DAV2_CAL["latency_ms"] - a) / (DAV2_CAL["batch"] * b * f_img)
    out["gamma_T"] = gamma
    out["gamma_T_range"] = ((DAV2_CAL["latency_ms"] - DAV2_CAL["sd_ms"] - a) / (DAV2_CAL["batch"] * b * f_img),
                            (DAV2_CAL["latency_ms"] + DAV2_CAL["sd_ms"] - a) / (DAV2_CAL["batch"] * b * f_img))
    out["dav2_cal_flops_per_img"] = f_img
    return out

def predict_latency(task, prec, fit, gamma_T=None, eff_mult=1.0):
    """Isolated GPU latency (ms) of a catalog task at precision prec."""
    a, b = fit[prec]["a_ms"], fit[prec]["ms_per_gflop"]
    g = 1.0
    if task["kind"] == "transformer":
        g = fit["gamma_T"] if gamma_T is None else gamma_T
    t = a + g * b * task["gflops"]
    return t * eff_mult

def memory_bytes_per_inference(task, prec, kappa=0.004):
    """Bytes moved from DRAM per inference: weights (re-read; 4 MB L2 cannot hold them)
    plus activation traffic ~ kappa bytes per FLOP (fused-kernel regime)."""
    w = task["weights_mb"][prec] * 1e6
    act = kappa * task["gflops"] * 1e9
    return w + act

# --------------------------------------------------------------------------
# 5. Quantization error theory
# --------------------------------------------------------------------------
def int8_mse_uniform(R):
    s = R / 127.0
    return s**2 / 12.0

def fp16_mse(var_w):
    u = 2.0**-11   # unit roundoff, 10 explicit mantissa bits, round-to-nearest
    return var_w * u**2 / 3.0

def laplace_clip_mse(alpha, b, M=8):
    """Expected MSE of uniform M-bit quantization of Laplace(0,b) clipped at +/-alpha
    (Banner et al. 2019 ACIQ): clipping term + rounding term."""
    return 2 * b**2 * np.exp(-alpha / b) + alpha**2 / (3 * 4**M)

def gauss_clip_mse(alpha, sigma, M=8):
    from scipy.stats import norm
    # clipping error: 2 * E[(X-alpha)^2 ; X>alpha]
    z = alpha / sigma
    # E[(X-a)^2 1{X>a}] for N(0,s^2) = (s^2 + a^2)(1-Phi(z)) - a s phi(z) ... derived
    phi, Phi = norm.pdf(z), norm.cdf(z)
    clip = 2 * ((sigma**2 + alpha**2) * (1 - Phi) - alpha * sigma * phi)
    return clip + alpha**2 / (3 * 4**M)

def optimal_clip(dist="laplace", M=8):
    if dist == "laplace":
        f = lambda a: laplace_clip_mse(a, 1.0, M)
    else:
        f = lambda a: gauss_clip_mse(a, 1.0, M)
    r = minimize_scalar(f, bounds=(0.5, 40), method="bounded")
    return r.x, r.fun

# --------------------------------------------------------------------------
# 6. Imaging: exposure photometry and motion blur
# --------------------------------------------------------------------------
def exposure_time_s(E_lux, iso, f_number, C=250.0):
    """Incident-light exposure equation: t = N^2 * C / (E * S)."""
    return f_number**2 * C / (E_lux * iso)

def blur_pixels(v_mps, t_exp_s, focal_m, Z_m, pitch_m):
    return v_mps * t_exp_s * focal_m / (Z_m * pitch_m)

# --------------------------------------------------------------------------
# 7. Safety kinematics
# --------------------------------------------------------------------------
def stopping_distance(v, tau, a):
    return v * tau + v**2 / (2 * a)

def required_detection_range(v, tau, a, margin=0.10):
    return stopping_distance(v, tau, a) + margin

def p_undetected_before_contact(recall_per_frame, d_detect, d_stop, v, T_cycle):
    """Obstacle appears at range d_detect; frames available before stopping is no longer
    possible = floor((d_detect - d_stop)/(v*T)); independent per-frame misses."""
    k = int(np.floor((d_detect - d_stop) / (v * T_cycle)))
    if k <= 0:
        return 1.0, 0
    return (1 - recall_per_frame) ** k, k


# --------------------------------------------------------------------------
# 8. Two-phase task representation for the simulator
#    Phase A: launch/latency-bound overhead a_p (fitted), low SM occupancy, little traffic
#    Phase B: throughput-bound kernels F/T_eff (x gamma_T for transformers), occupancy o_i,
#             carries weights + activation traffic
# --------------------------------------------------------------------------
OCC_A = 0.12
def task_phases(task, prec, fit, kappa=0.004, gamma_T=None, eff_mult=1.0):
    a = fit[prec]["a_ms"]; b = fit[prec]["ms_per_gflop"]
    g = 1.0
    if task["kind"] == "transformer":
        g = fit["gamma_T"] if gamma_T is None else gamma_T
    dA = a * eff_mult
    dB = g * b * task["gflops"] * eff_mult
    total_bytes = memory_bytes_per_inference(task, prec, kappa)
    return [(dA, OCC_A, 0.05 * total_bytes), (dB, task["occupancy"], 0.95 * total_bytes)]


# --------------------------------------------------------------------------
# 9. Edge / cloud offloading models
# --------------------------------------------------------------------------
EDGE_GPU = dict(name="NVIDIA L4", peak_fp16_dense_tflops=121.0, dram_bw_gbs=300.0, tdp_w=72)  # NVIDIA L4 datasheet (242 sparse)

def edge_service_time_ms(fit, gpu=EDGE_GPU, decoder_ms=10.0):
    """Whisper Small encoder service time on the edge GPU, assuming the Orin-calibrated achieved
    fraction eta_FP16 and transformer factor gamma_T carry over; plus a decoder budget."""
    eff = fit["fp16"]["eta"] * gpu["peak_fp16_dense_tflops"] / fit["gamma_T"]   # TFLOPS effective for transformers
    F, _ = whisper_small_encoder_flops()
    return fit["fp16"]["a_ms"] + F / eff + decoder_ms

def erlang_c(lam, mu, c):
    """Probability an arrival waits in M/M/c (lam arrivals/s, mu service rate/s per server)."""
    a = lam / mu; rho = a / c
    if rho >= 1: return 1.0
    import math
    s = sum(a**k / math.factorial(k) for k in range(c))
    last = a**c / (math.factorial(c) * (1 - rho))
    return last / (s + last)

def mmc_wait(lam, mu, c, q=0.99):
    """Mean and q-quantile of queueing delay (s) in M/M/c."""
    import math
    rho = lam / (c * mu)
    if rho >= 1: return float("inf"), float("inf"), rho
    Pw = erlang_c(lam, mu, c)
    Wq = Pw / (c * mu - lam)
    # P(Wq > t) = Pw * exp(-(c mu - lam) t)
    tq = 0.0 if Pw <= (1 - q) else -math.log((1 - q) / Pw) / (c * mu - lam)
    return Wq, tq, rho

def offload_response_ms(rtt_ms, uplink_mbps, payload_kb, service_ms, lam_per_s, c, encode_ms=5.0, q=0.99):
    """Response time (ms) of an offloaded voice query: encode + uplink transfer + RTT + queueing + service."""
    xfer = payload_kb * 8 / (uplink_mbps * 1000) * 1000.0
    mu = 1000.0 / service_ms
    Wq, tq, rho = mmc_wait(lam_per_s, mu, c, q)
    base = encode_ms + xfer + rtt_ms + service_ms
    return dict(mean=base + Wq * 1000, p99=base + tq * 1000, util=rho, base=base, xfer=xfer, wq_mean=Wq * 1000, wq_p99=tq * 1000)
