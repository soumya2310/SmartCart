# Simulation package for "Edge–cloud co-design of a priority-aware multitask inference pipeline for retail navigation carts"

This package reproduces every number reported in the manuscript's tables and figures. It
contains the analytical models, the cart-level scheduling simulator, the two experiment scripts,
a script that prints the manuscript's tables, and the results files behind the submitted paper.

Everything runs on an ordinary laptop. No GPU, no internet connection after installation, no
compilation, and no configuration are required.

---

## 1. Contents

| File | Purpose |
|---|---|
| `models.py` | Hardware constants, model catalog, calibrated latency model (Eq. 1), bandwidth model (Eq. 2), quantization theory, safety kinematics, edge/cloud offloading model (Eqs. 13–16) |
| `sched_sim.py` | Fluid discrete-time simulator of CUDA stream co-scheduling on the cart (Eqs. 3–4) |
| `run_cart_experiments.py` | Experiments E1–E6 (Sections 6.1–6.6) → `results/results.json` |
| `run_continuum_experiments.py` | Experiments E7–E10 (Section 6.8) → `results/results_cloud.json` |
| `print_tables.py` | Prints the manuscript's tables and the data behind its figures from the two results files |
| `expected_results/` | The two results files used for the submitted manuscript, for comparison |
| `requirements.txt` | Python dependencies (NumPy, SciPy) |

---

## 2. Requirements

* Python 3.9 or later (3.10–3.12 tested).
* NumPy 1.24 or later and SciPy 1.10 or later. Both are installed in step 3.
* About 200 MB of free disk space for the Python environment, and about 1 MB for the results.
* Runtime: about 5 minutes in total on a laptop CPU (see step 4).

---

## 3. Installation (step by step)

### 3.1 Unzip the package
Extract the zip file. It creates a folder named `smartcart_simulation`. All commands below are
run from inside that folder.

**Windows (PowerShell)**
```
cd path\to\smartcart_simulation
```
**macOS / Linux (Terminal)**
```
cd path/to/smartcart_simulation
```

### 3.2 Check that Python is available
```
python3 --version
```
On Windows the command is usually `python` rather than `python3`; if `python3` is not found,
use `python` in every command below. The printed version must be 3.9 or higher. If Python is
not installed, download it from https://www.python.org/downloads/ and, on Windows, tick
"Add python.exe to PATH" during installation.

### 3.3 Create and activate an isolated environment (recommended)
This keeps the package's dependencies separate from anything else on the machine.

**Windows (PowerShell)**
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
If PowerShell refuses to run the activation script, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once and retry.

**macOS / Linux**
```
python3 -m venv .venv
source .venv/bin/activate
```
The prompt now starts with `(.venv)`.

### 3.4 Install the dependencies
```
pip install -r requirements.txt
```
This downloads NumPy and SciPy (about 60 MB). An internet connection is needed only for this
step.

---

## 4. Running the simulation

Run the three commands below, in order, from inside the `smartcart_simulation` folder with the
environment activated.

### 4.1 Cart experiments E1–E6
```
python3 run_cart_experiments.py
```
Runtime: about 4 minutes on a laptop CPU (the simulator is a pure-Python time-stepped loop;
older machines may take up to 10 minutes). The script prints nothing until it finishes, then
prints

```
cart experiments E1-E6 complete -> results/results.json
```

### 4.2 Continuum experiments E7–E10
```
python3 run_continuum_experiments.py
```
Runtime: about 1 minute. On completion it prints

```
continuum experiments E7-E10 complete -> results/results_cloud.json
```

### 4.3 Print the manuscript's tables
```
python3 print_tables.py
```
This reads the two results files and prints, in order: Table 4 (latency calibration), Table 6
(predicted latencies), the Section 4.2 bandwidth demands, Table 7 with the paired statistics,
Table 8 (contention regimes), the Figure 5 grid, the Figure 6 and E4 sweeps, Table 9 (load
headroom), the quantization quantities of Section 4.4 and Supplement S1, Table 11 (safety
budget), Table 12 (voice modes), the Figure 7 fleet-capacity data, and the Figure 8 crossover
and fallback data. The first lines of output are

```
====================================================================================================
Table 4 - Latency-model calibration (Section 4.1)
====================================================================================================
FP32   a=4.01 ms  1/T=0.1273 ms/GFLOP  T_eff=7.9  eta=0.37  MAPE=4.5%  maxAPE=12.7%
FP16   a=2.50 ms  1/T=0.0573 ms/GFLOP  T_eff=17.5  eta=0.41  MAPE=3.8%  maxAPE=7.5%
INT8   a=2.26 ms  1/T=0.0373 ms/GFLOP  T_eff=26.8  eta=0.32  MAPE=4.1%  maxAPE=8.0%
gamma_T = 3.99  (range 3.31-4.68)
```

To keep a copy of the printed tables:
```
python3 print_tables.py > tables.txt
```

---

## 5. Confirming that the results match the manuscript

All random seeds are fixed inside the scripts, so the results files are identical to those in
`expected_results/` on any platform with NumPy 1.17 or later. To confirm:

```
python3 -c "import json; a=json.load(open('results/results.json')); b=json.load(open('expected_results/results.json')); print('cart results identical:', a==b)"
python3 -c "import json; a=json.load(open('results/results_cloud.json')); b=json.load(open('expected_results/results_cloud.json')); print('continuum results identical:', a==b)"
```
Both commands print `True`. If a future NumPy release changes its random-number implementation,
the files may differ in the last decimal places; in that case `print_tables.py` can be pointed at
the expected files by copying them into `results/`.

---

## 6. Where each manuscript result comes from

| Manuscript item | Script | Key in results file |
|---|---|---|
| Table 4 (latency calibration), γ_T | `run_cart_experiments.py` | `fit`, `gamma_T`, `gamma_T_range` |
| Table 6 (predicted latencies) | `run_cart_experiments.py` | `latency`, `latency_lo`, `latency_hi` |
| Section 4.2 bandwidth demand | `run_cart_experiments.py` | `bw_demand_mixed`, `bw_demand_sum_oncycle` |
| Table 7, Fig. 4 (E1) and paired statistics | `run_cart_experiments.py` | `E1`, `paired_obs`, `paired_crit` |
| Fig. 5 (E2 sensitivity grid) | `run_cart_experiments.py` | `E2` |
| Fig. 6 (E3) and the E4 jitter sentence | `run_cart_experiments.py` | `E3`, `E4` |
| Table 9 (E5 load headroom) | `run_cart_experiments.py` | `E5` |
| Table 8 (E6 contention regimes) | `run_cart_experiments.py` | `E6` |
| Section 4.4 and Supplement S1 (quantization) | `run_cart_experiments.py` | `Q` |
| Table 11 (safety budget) | `run_cart_experiments.py` | `safety` |
| Supplement S2 (exposure and blur) | `run_cart_experiments.py` | `imaging`, `blur_limit_exposure_ms` |
| Eq. 13 edge service time, Table 12 (E7) | `run_continuum_experiments.py` | `s_edge_ms`, `E7` |
| Fig. 7 (E8 fleet capacity) | `run_continuum_experiments.py` | `E8` |
| Fig. 8a (E9 crossover), Fig. 8b (E10 fallback) | `run_continuum_experiments.py` | `E9`, `E10` |

The results files are plain JSON and can be opened in any text editor or loaded with
`json.load` for further analysis.

---

## 7. Running a single experiment or changing a parameter

Nominal parameters are set at the top of each experiment script (`KAPPA`, the precision maps
`MIXED`, `ALLFP16`, `ALLFP32`, `ALLINT8`, and the sweep lists) and in `models.py` (`HW`,
`EDGE_GPU`, `build_catalog()`, `fit_latency_model()`). Each experiment in the manuscript is one
block of the corresponding script, delimited by a comment line such as
`# ---------------- Experiment 3: ...`, so a different parameter range is a one-line change to
the sweep list in that block.

The simulator can also be called directly from a Python session:

```python
from models import *
from sched_sim import simulate, summarize, ORDER
fit = fit_latency_model(); cat = build_catalog()
prec = dict(obstacle="int8", product="fp16", ocr="fp16", tracking="fp16", asr="fp16")
tasks = {}
for k in ORDER + ["asr"]:
    ph = task_phases(cat[k], prec[k], fit)
    tasks[k] = dict(e=sum(p[0] for p in ph), phases=ph, prio=-1 if k == "obstacle" else 0)
res = simulate(tasks, "priority", n_cycles=500, asr_rate_per_s=2/60, seed=11)
print(summarize(res))
```

`simulate()` accepts the policy (`"sequential"`, `"equal"`, or `"priority"`), the number of cycles,
the jitter coefficient of variation `cv`, the voice-query rate `asr_rate_per_s`, the local
fraction `p_local` and `fallback_delay` for offloaded modes, and the random `seed`; its docstring
lists every argument.

---

## 8. Troubleshooting

| Symptom | Cause and remedy |
|---|---|
| `python3: command not found` (Windows) | Use `python` instead of `python3`. |
| `No module named numpy` or `scipy` | The environment is not activated or step 3.4 was skipped. Activate it (step 3.3) and run `pip install -r requirements.txt`. |
| `FileNotFoundError: results/results.json` when running `print_tables.py` | Run steps 4.1 and 4.2 first; `print_tables.py` only reads their output. |
| The cart script seems to hang | It prints nothing until it finishes; allow up to 10 minutes on a slow machine. |
| `MemoryError` | Not expected; the scripts use well under 500 MB. Close other applications and retry. |

### Deactivating the environment when finished
```
deactivate
```
