# Hybrid Memetic Algorithm for DG Queue Optimization

Source code for the paper:

> **Path-Dependent Hosting Capacity and Sequential Queue Optimization for Distributed Generation Grid Access**
>
> Williams F. Fontinele, Pablo T. Caballero, Eduardo C. M. da Costa

## Overview

This repository contains the implementation of a hybrid memetic algorithm (MA) for optimizing the connection order of distributed generation (DG) projects in utility access queues. The algorithm uses sequential evaluation with dynamic hosting capacity (HC) update through Newton-Raphson power flow.

## Structure

```
execution/
├── memetic.py            # Hybrid MA (GA + AOS + local search)
├── evaluator.py          # Sequential evaluation with dynamic HC
├── network.py            # IEEE 33-bus and 69-bus network models
├── scenarios.py          # Penetration scenarios (I1-I4)
├── baselines.py          # FIFO, Greedy-Largest, Greedy-Smallest
├── queue_generator.py    # Random project queue generation
├── run_experiment.py     # Main experiment (33-bus)
├── run_69bus_experiment.py  # 69-bus experiment
├── run_part2_ablation.py # Ablation study (MA vs GA-only)
├── run_part3_fairness.py # Fairness analysis (D_max)
├── analyze_results.py    # Statistical analysis
└── generate_figures*.py  # Figure generation
```

## Requirements

- Python 3.12+
- pandapower 3.4+
- NumPy, SciPy, Matplotlib

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run main experiment (IEEE 33-bus, all scenarios)
python -m execution.run_experiment

# Run 69-bus experiment
python -m execution.run_69bus_experiment

# Run ablation study (MA vs GA-only)
python -m execution.run_part2_ablation

# Run fairness analysis (D_max variants)
python -m execution.run_part3_fairness
```

## License

MIT
