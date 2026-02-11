# DC-Ada: Reward-Only Decentralized Sensor Adaptation for Heterogeneous Multi-Robot Teams

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2026.XXXXX-b31b1b.svg)](https://arxiv.org/)

This repository contains the **complete experimental pipeline** for the paper:

**DC-Ada: Reward-Only Decentralized Observation-Interface Adaptation for Heterogeneous Multi-Robot Teams**

## Overview

DC-Ada targets a practical deployment regime: a **shared policy is pretrained under nominal sensing and kept frozen**, while each robot **adapts only a compact observation transform** to mitigate performance degradation under heterogeneous sensing (missing modalities, reduced range, altered resolution). Adaptation is **reward-only**, **gradient-free**, and **communication-minimal** (scalar team return per rollout).
---

## Highlights

- **Reward-only adaptation:** no policy gradients, no privileged state, no centralized belief.
- **Decentralized and communication-light:** no exchange of raw observations, maps, or gradients; only scalar returns (logged/broadcast).
- **Fixed observation interface across heterogeneity levels:** prevents checkpoint incompatibility and confounded comparisons.
- **Reproducible sweeps:** deterministic seeding and Common Random Numbers (CRN) for low-variance candidate evaluation.
- **Portable simulator:** lightweight multi-robot 2D environments implemented in pure NumPy (no PyBullet).

---

## Repository layout

```
.
├── configs/                 # YAML configs (default + strong/extended)
├── scripts/
│   ├── smoke_test.py        # Quick sanity checks (recommended first)
│   ├── pretrain_policy.py   # Shared-policy pretraining (per environment)
│   ├── run_experiment.py    # Main experiment runner (writes a single JSON)
│   ├── generate_figures.py  # Plotting + LaTeX table generation from JSON
│   ├── validate_results.py  # Validates completeness/errors in a results JSON
│   └── run_sanity_checks.py # Additional checks (optional)
├── src/
│   ├── envs/                # Warehouse / Search&Rescue / Mapping environments
│   ├── agents/              # Shared policy, transforms, methods/baselines
│   └── utils/               # Seeding, logging, utilities
├── run_all.sh               # End-to-end pipeline: pretrain → sweep → figures
├── requirements.txt
└── LICENSE
```

Outputs produced by the pipeline:
- `checkpoints/` — pretrained shared policies (`*.pth`)
- `results/` — consolidated results JSON (`results_YYYYMMDD_HHMMSS.json`)
- `figures/` — generated plots + LaTeX table (`results_table.tex`)

---

## Installation

### Requirements
- Python **3.8+**
- CPU-only is sufficient (PyTorch is used for the policy/transform networks).

### Setup

```bash
git clone https://github.com/alqithami/DC-ADA.git
cd DC-ADA

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Quick start (recommended)

### 1) Verify the installation

```bash
python scripts/smoke_test.py
```

### 2) Run the full pipeline (default budget)

```bash
chmod +x run_all.sh
./run_all.sh
```

This performs:
1. shared-policy pretraining (one policy per environment)
2. the full sweep (3 envs × 5 methods × 4 H-levels × 5 seeds = 300 runs) under a matched step budget
3. figure and table generation from the saved JSON

### 3) Run the **extended/strong** sweep (paper-grade budget)

The repository includes an extended configuration:

```bash
CONFIG=configs/strong.yaml ./run_all.sh
```

`configs/strong.yaml` uses a larger environment-step budget per run (e.g., 200k) and stronger adaptation settings while keeping comparisons budget-matched.

---

## Running a smaller targeted experiment

To run a smaller sweep without editing YAML, use `run_experiment.py` directly:

```bash
# Example: 2 methods on mapping, H0–H3, with 3 seeds and a 50k step budget
python scripts/run_experiment.py \
  --env mapping \
  --methods shared_policy dc_ada \
  --heterogeneity 0 1 2 3 \
  --seeds 3 \
  --budget 50000 \
  --output results
```

Notes:
- `--seeds K` means seeds `{0, 1, ..., K-1}` (for reproducibility).
- Budget is counted in **environment steps** (one step advances all robots jointly).

---

## Environments, success conditions, and progress metrics

The simulator is a lightweight 2D multi-robot environment suite. Each episode runs for up to `max_steps` with early termination on success.

Success conditions are **explicit** and configurable in YAML (e.g., `configs/default.yaml`):

- **Warehouse** (`WarehouseEnv`)
  - progress: `delivered_count`, `delivery_ratio`
  - success: `delivered_count >= target_deliveries`

- **Search & Rescue** (`SearchRescueEnv`)
  - progress: `rescued_count`, `rescue_ratio` (and `found_*` for discovery)
  - success: `rescued_count >= target_rescues`

- **Collaborative Mapping** (`CollaborativeMappingEnv`)
  - progress: `coverage`
  - success: `coverage >= target_coverage`

All episode-level counters (including targets) are saved into the results JSON under `episode_results`, enabling **threshold sensitivity** analyses without re-running experiments.

---

## Methods (DC-Ada + baselines)

The pipeline compares five methods:

| Method | Name in code/config | Summary |
|---|---|---|
| Shared Policy | `shared_policy` | Frozen shared policy, no adaptation |
| DC-Ada (ours) | `dc_ada` | Reward-only, per-robot interface adaptation (accept/reject random search with CRN) |
| Random Perturbation | `random_perturbation` | Same transform module, random updates without selection |
| Local Fine-Tuning | `local_finetuning` | Gradient-based updates of transform parameters (shared policy frozen) |
| Observation Normalization | `obs_normalization` | Running mean/variance normalization on the fixed-layout observation |

### DC-Ada update rule (accept/reject, best-of-**M**)
This repository implements a conservative accept/reject variant:

1. Every `K` episodes, for each robot:
2. Run a **baseline** truncated rollout (`T_c`) under CRN.
3. Sample `M` perturbations and evaluate each candidate under the **same seed** (CRN).
4. Apply the best perturbation only if it improves baseline return by a margin `τ`.

See `src/agents/methods.py` (`DCADAMethod`) for the exact implementation.

---

## Reproducing paper figures and tables

### Generate plots + LaTeX table from a results JSON
After running experiments, you will have a consolidated results file such as:
`results/results_YYYYMMDD_HHMMSS.json`

Generate the standard paper plots:

```bash
python scripts/generate_figures.py \
  --results results/results_YYYYMMDD_HHMMSS.json \
  --output figures/
```

This writes:
- `figures/performance_{warehouse,search_rescue,mapping}.pdf`
- `figures/scaling_{warehouse,search_rescue,mapping}.pdf`
- `figures/success_rate_{warehouse,search_rescue,mapping}.pdf`
- `figures/heatmap_reward_{warehouse,search_rescue,mapping}.pdf`
- `figures/heatmap_success_{warehouse,search_rescue,mapping}.pdf`
- `figures/results_table.tex` (LaTeX table for reward)

### Validate results completeness (recommended for long sweeps)
```bash
python scripts/validate_results.py --results results/results_YYYYMMDD_HHMMSS.json
```

### Optional: threshold-sensitivity plots (Fig.-8 style)
The default `generate_figures.py` script produces the main plots, but threshold-sensitivity curves can be generated directly from the saved `episode_results`. Below is a minimal example that outputs three PDFs under `figures/` using Matplotlib only:

```bash
python - << 'PY'
import json, os
import numpy as np
import matplotlib.pyplot as plt

RESULTS = "results/results_YYYYMMDD_HHMMSS.json"  # <-- set this
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

with open(RESULTS, "r") as f:
    data = json.load(f)

def curve(env, key, thresholds, h=3):
    # Aggregates across seeds per method at a fixed heterogeneity level.
    curves = {}
    for exp in data["experiments"]:
        if exp["env_name"] != env or exp["heterogeneity_level"] != h:
            continue
        m = exp["method_name"]
        vals = [ep.get(key, None) for ep in exp["episode_results"]]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        vals = np.asarray(vals, dtype=float)
        sr = [(vals >= t).mean() for t in thresholds]
        curves.setdefault(m, []).append(sr)
    # mean over seeds
    out = {}
    for m, arr in curves.items():
        A = np.asarray(arr, dtype=float)
        out[m] = A.mean(axis=0)
    return out

# Warehouse: delivered_count >= k
wh_thresh = np.arange(0, 4, 1)  # k = 0..3
wh = curve("warehouse", "delivered_count", wh_thresh, h=3)

# Search&Rescue: rescued_count >= k
sr_thresh = np.arange(0, 4, 1)  # k = 0..3
sr = curve("search_rescue", "rescued_count", sr_thresh, h=3)

# Mapping: coverage >= tau
mp_thresh = np.linspace(0.4, 0.9, 26)
mp = curve("mapping", "coverage", mp_thresh, h=3)

def plot_curves(curves, x, xlabel, outname, title):
    plt.figure()
    for m, y in sorted(curves.items()):
        plt.plot(x, y, label=m)
    plt.xlabel(xlabel)
    plt.ylabel("Success rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, outname))
    plt.close()

plot_curves(wh, wh_thresh, "k (deliveries)", "threshold_sensitivity_warehouse_H3.pdf",
            "Warehouse H3: delivered ≥ k")
plot_curves(sr, sr_thresh, "k (rescues)", "threshold_sensitivity_search_rescue_H3.pdf",
            "Search&Rescue H3: rescued ≥ k")
plot_curves(mp, mp_thresh, "τ (coverage)", "threshold_sensitivity_mapping_H3.pdf",
            "Mapping H3: coverage ≥ τ")

print("Saved threshold sensitivity plots to", OUTDIR)
PY
```

---

## Configuration

Primary knobs (see `configs/default.yaml` and `configs/strong.yaml`):

- `total_budget`: total environment-step budget per run (budget-matched across methods)
- `max_steps`: episode horizon
- `num_seeds`, `heterogeneity_levels`, `methods`, `environments`
- Environment thresholds: `warehouse.target_deliveries`, `search_rescue.target_rescues`, `mapping.target_coverage`
- DC-Ada hyperparameters: `num_candidates`, `noise_scale`, `step_size`, `acceptance_margin`, `adaptation_interval`, `candidate_rollout_fraction`

---

## Troubleshooting

- **Success rate appears low in Warehouse:** warehouse success is thresholded (e.g., 2 deliveries). Under severe heterogeneity, success can be rare; use reward/progress metrics and threshold sensitivity rather than relying on a single strict threshold.
- **`ModuleNotFoundError: cfdefect`**: this module is not part of DC-Ada; if you see it, it came from running an unrelated command after the pipeline finished.
- **Mac/Apple Silicon:** use a fresh venv and install from `requirements.txt`. CPU-only runs are supported.

---

## Citation

If you use this codebase, please cite the paper:

```bibtex
@article{alqithami_dcada_2026,
  title   = {DC-Ada: Reward-Only Decentralized Observation-Interface Adaptation for Heterogeneous Multi-Robot Teams},
  author  = {Saad Alqithami},
  journal = {IEEE Access},
  note    = {Under review},
  year    = {2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This work recieved no external supported or funding.
- We thank the reviewers for their constructive feedback.

## Contact

For questions or issues, please open a GitHub issue or contact [salqithami@bu.edu.sa].
