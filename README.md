# PFMU: Fog-Based Unified Mobility Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-IAENG-green.svg)](https://www.iaeng.org/)

> **A Fog-Based Unified Mobility Framework for Coordinated Traffic Signal Control and Smart Parking Allocation**  
> V. Venkataramanan, Pankaj Mishra, Aarit Mehta, Jaychand Upadhyay, Supriya Dicholkar, Vats Shah, Aditya Ravi  
> *IAENG International Journal of Computer Science*

---

## Overview

PFMU (**P**redictive **F**og-Based **U**nified **M**obility Framework) is a decentralised fog-computing architecture that jointly optimises **traffic signal control** and **smart parking allocation** using:

- **Coordination-Aware Multi-Agent Reinforcement Learning (MARL)** for traffic signal phases
- **Congestion-Coupled Slot Allocation (EASA+)** for parking assignment
- A **Shared Congestion Index** linking both subsystems under a unified objective

### Key Results (vs. Centralised Cloud Baseline)

| Metric | Improvement |
|--------|-------------|
| Signal Response Latency | **↓ 74.4%** (1.21s → 0.31s) |
| Intersection Throughput | **↑ 26.6%** (682 → 864 veh/h) |
| Parking Search Time | **↓ 31.4%** (13.5s → 8.2s) |
| Uplink Communication Volume | **↓ 93.0%** (180 → 12 MB/h/node) |
| Decision Cycle Runtime | **↓ 55.6%** (142ms → 63ms) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    IoT Device Layer                      │
│  Traffic Cameras │ Parking Sensors │ Environmental IoT  │
└────────────────────────┬────────────────────────────────┘
                         │ Real-Time Data
┌────────────────────────▼────────────────────────────────┐
│                      Fog Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │Data Acq. &   │  │ Intelligent  │  │    Unified    │  │
│  │Preprocessing │→ │  Decision &  │→ │ Coordination  │  │
│  │              │  │Optimization  │  │ & Info Share  │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
│      MARL Agents + EASA+ + Shared Congestion Index       │
└────────────────────────┬────────────────────────────────┘
                         │ Summary Data (Periodic)
┌────────────────────────▼────────────────────────────────┐
│                     Cloud Layer                          │
│        Long-Term Storage │ Global Analytics              │
└─────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
pfmu/
├── README.md                        # This file
├── LICENSE                          # MIT License
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package installation
├── .gitignore
│
├── src/                             # Core source code
│   ├── __init__.py
│   ├── agents/                      # RL agents
│   │   ├── __init__.py
│   │   ├── marl_agent.py            # Coordination-aware MARL agent
│   │   ├── q_learning.py            # Q-learning implementation
│   │   └── reward_shaping.py        # Reward formulation (Eq. 19)
│   │
│   ├── environment/                 # Traffic simulation environment
│   │   ├── __init__.py
│   │   ├── traffic_network.py       # Graph model G=(I,E), queue dynamics
│   │   ├── intersection.py          # MDP state/action/reward per node
│   │   ├── fog_node.py              # Fog node orchestrator
│   │   └── grid_topology.py         # 5×5 grid setup
│   │
│   ├── parking/                     # Smart parking subsystem
│   │   ├── __init__.py
│   │   ├── parking_allocator.py     # EASA+ allocation algorithm
│   │   ├── slot_manager.py          # Occupancy tracking
│   │   └── congestion_index.py      # Shared congestion index φ_t
│   │
│   ├── utils/                       # Utilities
│   │   ├── __init__.py
│   │   ├── config.py                # Configuration loader
│   │   ├── logger.py                # Structured logging
│   │   ├── metrics.py               # Performance metrics
│   │   └── statistics.py            # Statistical validation (t-tests)
│   │
│   └── visualization/               # Plotting and dashboards
│       ├── __init__.py
│       ├── plots.py                 # All paper figures (Fig 2–23)
│       └── dashboard.py             # Real-time monitoring
│
├── configs/                         # Experiment configurations
│   ├── default.yaml                 # Default simulation parameters
│   ├── cloud_baseline.yaml          # CLD baseline config
│   ├── fog_rl.yaml                  # F-RL baseline config
│   ├── hybrid.yaml                  # HCF baseline config
│   └── pfmu.yaml                    # Full PFMU config
│
├── experiments/                     # Experiment runners
│   ├── __init__.py
│   ├── run_pfmu.py                  # Main PFMU experiment
│   ├── run_baselines.py             # All 6 baseline comparisons
│   ├── ablation_study.py            # Component ablation (Fig 15)
│   ├── scalability_test.py          # Network size scaling (Fig 9)
│   ├── robustness_test.py           # Packet loss / RTT variation (Fig 10)
│   └── sensitivity_analysis.py     # λ4 and ξ sweeps (Fig 11–12)
│
├── tests/                           # Unit & integration tests
│   ├── __init__.py
│   ├── test_traffic_network.py
│   ├── test_marl_agent.py
│   ├── test_parking_allocator.py
│   ├── test_congestion_index.py
│   └── test_metrics.py
│
├── notebooks/                       # Jupyter notebooks
│   ├── 01_quick_start.ipynb         # 5-minute demo
│   ├── 02_results_reproduction.ipynb # Reproduce all paper tables/figures
│   └── 03_parameter_tuning.ipynb    # Hyperparameter exploration
│
├── scripts/                         # Shell scripts
│   ├── install.sh                   # Environment setup
│   ├── run_all_experiments.sh       # Full paper reproduction
│   └── generate_plots.sh            # Regenerate all figures
│
├── results/                         # Output directory (auto-generated)
│   ├── figures/                     # Generated plots
│   ├── logs/                        # Experiment logs
│   └── data/                        # Raw results (CSV/JSON)
│
└── docs/                            # Extended documentation
    ├── methodology.md               # Detailed math derivations
    ├── simulation_setup.md          # iFogSim configuration guide
    └── api_reference.md             # Module API reference
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Quick Install

```bash
# Clone the repository
git clone https://github.com/<your-username>/pfmu.git
cd pfmu

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Verify Installation

```bash
python -c "import src; print('PFMU installed successfully')"
python -m pytest tests/ -v
```

---

## Quick Start

### Run PFMU (Full System)

```bash
python experiments/run_pfmu.py --config configs/pfmu.yaml --episodes 500 --seed 42
```

### Run All Baselines + PFMU (Paper Reproduction)

```bash
bash scripts/run_all_experiments.sh
```

### Run a Single Comparison

```bash
python experiments/run_baselines.py --methods CLD F-RL A-MORL HCF PFMU --episodes 500
```

### Generate All Paper Figures

```bash
bash scripts/generate_plots.sh
# Figures saved to results/figures/
```

---

## Reproducing Paper Results

All tables and figures from the paper can be reproduced in one command:

```bash
bash scripts/run_all_experiments.sh
```

This runs 10 independent seeds per method and produces:

| Output | Paper Reference |
|--------|----------------|
| `results/figures/fig02_latency.png` | Figure 2 — Signal latency |
| `results/figures/fig03_throughput.png` | Figure 3 — Intersection throughput |
| `results/figures/fig04_fairness.png` | Figure 4 — Jain's fairness index |
| `results/figures/fig05_switching.png` | Figure 5 — Switching frequency |
| `results/figures/fig06_convergence.png` | Figure 6 — Learning convergence |
| `results/figures/fig07_parking_util.png` | Figure 7 — Parking utilisation |
| `results/figures/fig08_search_time.png` | Figure 8 — Parking search time |
| `results/figures/fig09_scalability.png` | Figure 9 — Latency vs network size |
| `results/figures/fig10_robustness.png` | Figure 10 — Packet loss robustness |
| `results/figures/fig11_lambda4.png` | Figure 11 — Sensitivity λ4 |
| `results/figures/fig12_xi.png` | Figure 12 — Sensitivity ξ |
| `results/figures/fig13_runtime.png` | Figure 13 — Runtime comparison |
| `results/figures/fig14_latency_cdf.png` | Figure 14 — Latency CDF |
| `results/figures/fig15_ablation.png` | Figure 15 — Ablation study |
| `results/tables/table5_improvement.csv` | Table V — % improvements |
| `results/tables/table6_absolute.csv` | Table VI — Absolute values |
| `results/tables/table7_runtime.csv` | Table VII — Runtime |
| `results/tables/table9_stats.csv` | Table IX — Statistical tests |

---

## Core Modules

### Traffic Network (`src/environment/traffic_network.py`)

Models the road network as a directed graph `G = (I, E)` with queue dynamics:

```
q_{t+1,i} = q_{t,i} + a_{t,i} - d_{t,i}
```

### MARL Agent (`src/agents/marl_agent.py`)

Coordination-aware Q-learning with neighbour penalty (Eq. 19):

```
R_t^i = -(α·Σ_l q_{t,l} + β·D_t + η·L_t + ξ·Σ_{j∈N(i)} |q_{t,i} - q_{t,j}|)
```

### EASA+ Parking Allocator (`src/parking/parking_allocator.py`)

Congestion-coupled cost minimisation (Eq. 21):

```
c*_{vp} = λ1·d_{vp} + λ2·T_{vp} + λ3·θ_p + λ4·φ_t
```

### Shared Congestion Index (`src/parking/congestion_index.py`)

Coupling variable between traffic and parking (Eq. 22):

```
φ_t = κ1·q̄_t + κ2·w̄_t
```

---

## Configuration

All parameters are defined in `configs/default.yaml`. Key parameters:

```yaml
simulation:
  topology: "5x5_grid"
  num_intersections: 25
  num_parking_lots: 8
  duration_seconds: 3600
  num_runs: 10
  time_resolution: 1

traffic:
  arrival_model: "nonhomogeneous_poisson"
  demand_range: [400, 1200]   # veh/h
  phase_duration_range: [5, 60]  # seconds
  saturation_flow: 1800       # veh/h

marl:
  learning_rate: 0.01
  discount_factor: 0.95
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 0.995
  coordination_weight: 0.3    # ξ
  alpha: 1.0                  # queue weight
  beta: 0.5                   # delay weight
  eta: 0.2                    # switching penalty

parking:
  slot_capacities: [50, 120]
  parking_duration: [20, 90]  # minutes (truncated normal)
  lambda1: 0.4                # distance weight
  lambda2: 0.3                # wait time weight
  lambda3: 0.2                # occupancy weight
  lambda4: 0.1                # congestion weight (φ_t)

fog:
  rtt_range: [5, 15]          # ms (fog-device)
  sync_interval: 10           # steps between neighbour sync
  kappa1: 0.6                 # queue weight in φ_t
  kappa2: 0.4                 # wait-time weight in φ_t

cloud:
  rtt_range: [60, 120]        # ms (fog-cloud)
```

---

## Baselines

| ID | Description | Traffic | Parking |
|----|-------------|---------|---------|
| `CLD` | Centralised cloud control | ✓ | ✓ |
| `F-RL` | Fog-based independent RL | ✓ | — |
| `A-MORL` | Multi-objective RL | ✓ | — |
| `HCF` | Hybrid cloud-fog control | ✓ | ✓ |
| `F-PARK` | Fog-assisted parking only | — | ✓ |
| `F-IPS` | Distributed IoT parking | — | ✓ |
| **`PFMU`** | **Proposed unified framework** | **✓** | **✓** |

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific module
python -m pytest tests/test_marl_agent.py -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=html
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{venkataramanan2025pfmu,
  title   = {A Fog-Based Unified Mobility Framework for Coordinated Traffic Signal Control and Smart Parking Allocation},
  author  = {Venkataramanan, V. and Mishra, Pankaj and Mehta, Aarit and Upadhyay, Jaychand and Dicholkar, Supriya and Shah, Vats and Ravi, Aditya},
  journal = {IAENG International Journal of Computer Science},
  year    = {2025}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

The authors thank K J Somaiya School of Engineering and Dwarkadas J. Sanghvi College of Engineering for providing computing resources.
