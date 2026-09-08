# -*- coding: utf-8 -*-
"""
run_part2_ablation.py — Ablation study comparing MA (full) vs GA-only across networks.

Part of: Hybrid Memetic Algorithm for DG Queue Optimization
Paper: "Connection Order Matters: Path-Dependent Hosting Capacity in
        Distributed Generation Interconnection Queues"
Authors: Williams F. Fontinele, Pablo T. Caballero, Eduardo C. M. da Costa
License: MIT
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .baselines import fifo_baseline, greedy_baseline, greedy_smallest_first
from .memetic import MAConfig, run_memetic_algorithm
from .network import TechnicalConstraints
from .scenarios import SCENARIOS, create_scenario, create_scenario_69bus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MA_POP = 20
MA_GENS = 30
MA_CROSSOVER = 0.9
MA_MUTATION = 0.15
MA_LS_PROB = 0.3
MA_PATIENCE = 10
N_ALGO_RUNS = 10
INSTANCE_SEEDS = [42, 7, 13]
OUTPUT_DIR = Path("analysis-output")


def run_ma_runs(net, projects, constraints, n_runs, base_seed,
                ls_prob=MA_LS_PROB):
    results = []
    for i in range(n_runs):
        run_seed = base_seed * 1000 + i
        cfg = MAConfig(
            pop_size=MA_POP, n_generations=MA_GENS,
            crossover_rate=MA_CROSSOVER, mutation_rate=MA_MUTATION,
            local_search_prob=ls_prob, convergence_patience=MA_PATIENCE,
            seed=run_seed,
            d_max=float("inf"),
        )
        t0 = time.time()
        result = run_memetic_algorithm(net, projects, constraints, cfg)
        elapsed = time.time() - t0
        results.append({
            "fitness": result.best_fitness,
            "n_connected": result.best_eval.n_connected,
            "time_s": elapsed,
            "generations": result.generation_converged,
            "seed": run_seed,
        })
        logger.info(
            f"  Run {i+1}/{n_runs}: fitness={result.best_fitness:.0f} kW, "
            f"n={result.best_eval.n_connected}, t={elapsed:.1f}s"
        )
    return results


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    constraints = TechnicalConstraints()
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("PART 2: Ablation MA vs GA-only (3 seeds)")
    logger.info("=" * 60)

    ablation_results = {}
    ablation_combos = [
        ("33bus", create_scenario, ["I3", "I4"]),
        ("69bus", create_scenario_69bus, ["I4"]),
    ]

    for network_name, create_fn, scenarios in ablation_combos:
        for scenario_name in scenarios:
            for seed in INSTANCE_SEEDS:
                key = f"{network_name}_{scenario_name}_s{seed}"
                logger.info(f"\n--- Ablation {key} ---")
                net, projects, config = create_fn(scenario_name, seed=seed)

                ma_full = run_ma_runs(net, projects, constraints, N_ALGO_RUNS, seed, ls_prob=MA_LS_PROB)
                ga_only = run_ma_runs(net, projects, constraints, N_ALGO_RUNS, seed, ls_prob=0.0)

                ma_fits = [r["fitness"] for r in ma_full]
                ga_fits = [r["fitness"] for r in ga_only]

                ablation_results[key] = {
                    "network": network_name, "scenario": scenario_name, "seed": seed,
                    "ma_full": {"mean": float(np.mean(ma_fits)), "std": float(np.std(ma_fits)), "runs": ma_full},
                    "ga_only": {"mean": float(np.mean(ga_fits)), "std": float(np.std(ga_fits)), "runs": ga_only},
                }
                logger.info(
                    f"  MA: {np.mean(ma_fits):.0f}+/-{np.std(ma_fits):.0f} | "
                    f"GA: {np.mean(ga_fits):.0f}+/-{np.std(ga_fits):.0f}"
                )

    # Save immediately
    p2_file = OUTPUT_DIR / f"ablation_multiseed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(p2_file, "w") as f:
        json.dump({
            "ablation_multiseed": ablation_results,
            "time_h": (time.time() - t_start) / 3600,
        }, f, indent=2, default=str)
    logger.info(f"\nPart 2 DONE. Saved to {p2_file}")
    logger.info(f"Total time: {(time.time() - t_start)/3600:.2f} hours")


if __name__ == "__main__":
    main()
