# -*- coding: utf-8 -*-
"""
run_69bus_experiment.py — IEEE 69-bus experiment runner with ablation and fairness analysis.

Part of: Hybrid Memetic Algorithm for DG Queue Optimization
Paper: "Path-Dependent Hosting Capacity and Sequential Queue Optimization
        for Distributed Generation Grid Access"
Authors: Williams F. Fontinele, Pablo T. Caballero, Eduardo C. M. da Costa
License: MIT
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .baselines import (
    fifo_baseline,
    greedy_baseline,
    greedy_smallest_first,
)
from .evaluator import evaluate_permutation
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


def run_baselines(net, projects, constraints):
    """Run deterministic baselines."""
    results = {}
    for name, fn in [
        ("FIFO", fifo_baseline),
        ("Greedy_Largest", greedy_baseline),
        ("Greedy_Smallest", greedy_smallest_first),
    ]:
        res = fn(net, projects, constraints)
        results[name] = {
            "fitness": res.fitness,
            "n_connected": res.n_connected,
            "n_rejected": res.n_rejected,
            "time_s": res.eval_time_s,
        }
    return results


def run_ma_runs(net, projects, constraints, n_runs, base_seed,
                pop=MA_POP, gens=MA_GENS, ls_prob=MA_LS_PROB, d_max=None):
    """Run MA multiple times."""
    results = []
    for i in range(n_runs):
        run_seed = base_seed * 1000 + i
        cfg = MAConfig(
            pop_size=pop,
            n_generations=gens,
            crossover_rate=MA_CROSSOVER,
            mutation_rate=MA_MUTATION,
            local_search_prob=ls_prob,
            convergence_patience=MA_PATIENCE,
            seed=run_seed,
            d_max=float("inf") if d_max is None else d_max,
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
            f"n_connected={result.best_eval.n_connected}, time={elapsed:.1f}s"
        )
    return results


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_results = {
        "experiment": "69bus_ablation_fairness",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "pop": MA_POP, "gens": MA_GENS,
            "crossover": MA_CROSSOVER, "mutation": MA_MUTATION,
            "ls_prob": MA_LS_PROB, "patience": MA_PATIENCE,
            "n_runs": N_ALGO_RUNS, "seeds": INSTANCE_SEEDS,
        },
        "results_69bus": {},
        "ablation_multiseed": {},
        "fairness_multiseed": {},
    }

    t_start = time.time()
    constraints = TechnicalConstraints()

    # =========================================================
    # PART 1: Main results on IEEE 69-bus (I3, I4 x 3 seeds)
    # =========================================================
    logger.info("=" * 60)
    logger.info("PART 1: IEEE 69-bus main results")
    logger.info("=" * 60)

    for scenario_name in ["I3", "I4"]:
        for seed in INSTANCE_SEEDS:
            key = f"{scenario_name}_s{seed}"
            logger.info(f"\n--- 69-bus {key} ---")

            net, projects, config = create_scenario_69bus(scenario_name, seed=seed)

            baselines = run_baselines(net, projects, constraints)
            logger.info(f"  FIFO: {baselines['FIFO']['fitness']:.0f} kW")
            logger.info(f"  Greedy-L: {baselines['Greedy_Largest']['fitness']:.0f} kW")
            logger.info(f"  Greedy-S: {baselines['Greedy_Smallest']['fitness']:.0f} kW")

            ma_results = run_ma_runs(net, projects, constraints, N_ALGO_RUNS, seed)
            fitnesses = [r["fitness"] for r in ma_results]

            all_results["results_69bus"][key] = {
                "scenario": scenario_name,
                "seed": seed,
                "network": "IEEE 69-bus",
                "penetration_pct": float(sum(p.p_kw for p in projects) / (net.load.p_mw.sum() * 1000) * 100),
                "baselines": baselines,
                "ma_runs": ma_results,
                "ma_mean": float(np.mean(fitnesses)),
                "ma_std": float(np.std(fitnesses)),
            }
            logger.info(
                f"  MA: {np.mean(fitnesses):.0f} +/- {np.std(fitnesses):.0f} kW"
            )

    # =========================================================
    # PART 2: Ablation MA vs GA-only (3 seeds, BOTH networks)
    # Only I4 on 69-bus (I3 is trivial — all connect)
    # =========================================================
    logger.info("\n" + "=" * 60)
    logger.info("PART 2: Ablation MA vs GA-only (3 seeds)")
    logger.info("=" * 60)

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

                # MA full
                ma_full = run_ma_runs(net, projects, constraints, N_ALGO_RUNS, seed,
                                     ls_prob=MA_LS_PROB)
                # GA only (ls_prob=0)
                ga_only = run_ma_runs(net, projects, constraints, N_ALGO_RUNS, seed,
                                     ls_prob=0.0)

                ma_fits = [r["fitness"] for r in ma_full]
                ga_fits = [r["fitness"] for r in ga_only]

                all_results["ablation_multiseed"][key] = {
                    "network": network_name,
                    "scenario": scenario_name,
                    "seed": seed,
                    "ma_full": {"mean": float(np.mean(ma_fits)), "std": float(np.std(ma_fits)), "runs": ma_full},
                    "ga_only": {"mean": float(np.mean(ga_fits)), "std": float(np.std(ga_fits)), "runs": ga_only},
                }
                logger.info(
                    f"  MA: {np.mean(ma_fits):.0f} +/- {np.std(ma_fits):.0f} | "
                    f"GA: {np.mean(ga_fits):.0f} +/- {np.std(ga_fits):.0f}"
                )

    # =========================================================
    # PART 3: Fairness (3 seeds, 3 D_max, BOTH networks)
    # =========================================================
    logger.info("\n" + "=" * 60)
    logger.info("PART 3: Fairness analysis (3 seeds, 3 D_max)")
    logger.info("=" * 60)

    d_max_values = [None, 25, 12]  # None = infinity

    fairness_combos = [
        ("33bus", create_scenario, ["I3", "I4"]),
        ("69bus", create_scenario_69bus, ["I4"]),
    ]

    for network_name, create_fn, scenarios in fairness_combos:
        for scenario_name in scenarios:
            for seed in INSTANCE_SEEDS:
                for d_max in d_max_values:
                    d_label = "inf" if d_max is None else str(d_max)
                    key = f"{network_name}_{scenario_name}_s{seed}_d{d_label}"
                    logger.info(f"\n--- Fairness {key} ---")

                    net, projects, config = create_fn(scenario_name, seed=seed)

                    results = run_ma_runs(
                        net, projects, constraints, N_ALGO_RUNS, seed,
                        d_max=d_max,
                    )

                    fitnesses = [r["fitness"] for r in results]
                    all_results["fairness_multiseed"][key] = {
                        "network": network_name,
                        "scenario": scenario_name,
                        "seed": seed,
                        "d_max": d_max,
                        "mean": float(np.mean(fitnesses)),
                        "std": float(np.std(fitnesses)),
                        "runs": results,
                    }
                    logger.info(
                        f"  D_max={d_label}: {np.mean(fitnesses):.0f} +/- {np.std(fitnesses):.0f} kW"
                    )

    total_time = time.time() - t_start
    all_results["total_time_hours"] = total_time / 3600

    output_file = OUTPUT_DIR / f"experiment_69bus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nTotal time: {total_time/3600:.2f} hours")
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
