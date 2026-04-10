# -*- coding: utf-8 -*-
"""
run_full_experiment.py — Full experiment runner for both networks (Option A config).

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
    random_baseline,
)
from .evaluator import evaluate_permutation
from .memetic import MAConfig, run_memetic_algorithm
from .network import TechnicalConstraints
from .scenarios import SCENARIOS, create_scenario

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =====================================================================
# Configuration — Option A
# =====================================================================
MA_POP = 20
MA_GENS = 30
MA_CROSSOVER = 0.9
MA_MUTATION = 0.15
MA_LS_PROB = 0.3
MA_PATIENCE = 10

N_ALGO_RUNS = 10          # runs per (scenario, instance_seed)
INSTANCE_SEEDS = [42, 7, 13]  # 3 instance seeds for I3/I4
TRIVIAL_SCENARIOS = ["I1", "I2"]  # single run, single seed
RESTRICTIVE_SCENARIOS = ["I3", "I4"]

OUTPUT_DIR = Path("analysis-output")


def run_baselines_for_instance(net, projects, constraints, seed):
    """Run all deterministic baselines (no randomness)."""
    results = {}

    # FIFO
    res = fifo_baseline(net, projects, constraints)
    results["FIFO"] = {
        "fitness": res.fitness,
        "n_connected": res.n_connected,
        "n_rejected": res.n_rejected,
        "pf_calls": res.pf_calls,
        "time_s": res.eval_time_s,
    }

    # Greedy largest
    res = greedy_baseline(net, projects, constraints)
    results["Greedy_Largest"] = {
        "fitness": res.fitness,
        "n_connected": res.n_connected,
        "n_rejected": res.n_rejected,
        "pf_calls": res.pf_calls,
        "time_s": res.eval_time_s,
    }

    # Greedy smallest
    res = greedy_smallest_first(net, projects, constraints)
    results["Greedy_Smallest"] = {
        "fitness": res.fitness,
        "n_connected": res.n_connected,
        "n_rejected": res.n_rejected,
        "pf_calls": res.pf_calls,
        "time_s": res.eval_time_s,
    }

    # Random (100 samples)
    best, mean, std = random_baseline(net, projects, constraints, n_samples=100, seed=seed)
    results["Random"] = {
        "fitness_best": best.fitness,
        "fitness_mean": mean,
        "fitness_std": std,
        "n_connected": best.n_connected,
    }

    return results


def run_ma_multiple(net, projects, constraints, n_runs, base_seed):
    """Run MA n_runs times with different algorithm seeds."""
    all_results = []

    for run_idx in range(n_runs):
        algo_seed = base_seed * 1000 + run_idx
        ma_config = MAConfig(
            pop_size=MA_POP,
            n_generations=MA_GENS,
            crossover_rate=MA_CROSSOVER,
            mutation_rate=MA_MUTATION,
            local_search_prob=MA_LS_PROB,
            seed=algo_seed,
            convergence_patience=MA_PATIENCE,
        )

        logger.info(f"  MA run {run_idx+1}/{n_runs} (seed={algo_seed})...")
        result = run_memetic_algorithm(net, projects, constraints, ma_config)

        all_results.append({
            "run": run_idx,
            "algo_seed": algo_seed,
            "fitness": result.best_fitness,
            "n_connected": result.best_eval.n_connected,
            "n_rejected": result.best_eval.n_rejected,
            "pf_calls": result.total_pf_calls,
            "time_s": result.total_time_s,
            "generation_converged": result.generation_converged,
            "fitness_history": result.fitness_history,
        })

    # Summary statistics
    fitnesses = [r["fitness"] for r in all_results]
    n_conns = [r["n_connected"] for r in all_results]

    summary = {
        "fitness_mean": float(np.mean(fitnesses)),
        "fitness_std": float(np.std(fitnesses)),
        "fitness_min": float(np.min(fitnesses)),
        "fitness_max": float(np.max(fitnesses)),
        "n_connected_mean": float(np.mean(n_conns)),
        "n_connected_std": float(np.std(n_conns)),
        "n_runs": n_runs,
    }

    return all_results, summary


def run_scenario(scenario_name):
    """Run complete experiment for one scenario."""
    logger.info(f"\n{'='*60}")
    logger.info(f"SCENARIO {scenario_name}")
    logger.info(f"{'='*60}")

    is_trivial = scenario_name in TRIVIAL_SCENARIOS
    seeds = [42] if is_trivial else INSTANCE_SEEDS
    n_runs = 1 if is_trivial else N_ALGO_RUNS

    scenario_results = {
        "scenario": scenario_name,
        "label": SCENARIOS[scenario_name].label,
        "instances": [],
    }

    for inst_seed in seeds:
        logger.info(f"\n--- Instance seed={inst_seed} ---")
        net, projects, config = create_scenario(scenario_name, seed=inst_seed)

        total_dg = sum(p.p_kw for p in projects)
        total_load = float(net.load.p_mw.sum()) * 1000
        penetration = total_dg / total_load * 100

        # Baselines
        logger.info("Running baselines...")
        baselines = run_baselines_for_instance(net, projects, config.constraints, inst_seed)

        for name, res in baselines.items():
            f = res.get("fitness", res.get("fitness_best", 0))
            nc = res.get("n_connected", "?")
            logger.info(f"  {name:20s}: {f:8.1f} kW, connected={nc}")

        # MA runs
        logger.info(f"Running MA ({n_runs} runs)...")
        ma_runs, ma_summary = run_ma_multiple(
            net, projects, config.constraints, n_runs, inst_seed
        )

        logger.info(
            f"  MA summary: {ma_summary['fitness_mean']:.1f} +/- "
            f"{ma_summary['fitness_std']:.1f} kW, "
            f"n_connected={ma_summary['n_connected_mean']:.1f} +/- "
            f"{ma_summary['n_connected_std']:.1f}"
        )

        # Compute gains
        fifo_fitness = baselines["FIFO"]["fitness"]
        greedy_s_fitness = baselines["Greedy_Smallest"]["fitness"]
        gain_fifo = (
            (ma_summary["fitness_mean"] - fifo_fitness) / fifo_fitness * 100
            if fifo_fitness > 0 else 0
        )
        gain_greedy = (
            (ma_summary["fitness_mean"] - greedy_s_fitness) / greedy_s_fitness * 100
            if greedy_s_fitness > 0 else 0
        )
        logger.info(f"  Gain over FIFO: {gain_fifo:+.1f}%")
        logger.info(f"  Gain over Greedy-Smallest: {gain_greedy:+.1f}%")

        instance_result = {
            "instance_seed": inst_seed,
            "penetration_pct": penetration,
            "total_dg_kw": total_dg,
            "total_load_kw": total_load,
            "n_projects": len(projects),
            "baselines": baselines,
            "ma_runs": ma_runs,
            "ma_summary": ma_summary,
            "gains": {
                "vs_fifo_pct": gain_fifo,
                "vs_greedy_smallest_pct": gain_greedy,
            },
        }
        scenario_results["instances"].append(instance_result)

    return scenario_results


def main():
    t_start = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 60)
    logger.info("FULL EXPERIMENT — Option A (reduced config)")
    logger.info(f"Started at {datetime.now().isoformat()}")
    logger.info(f"MA config: pop={MA_POP}, gen={MA_GENS}, "
                f"runs={N_ALGO_RUNS}, seeds={INSTANCE_SEEDS}")
    logger.info("=" * 60)

    all_results = {
        "experiment": "Option_A",
        "timestamp": timestamp,
        "config": {
            "ma_pop": MA_POP,
            "ma_gens": MA_GENS,
            "ma_crossover": MA_CROSSOVER,
            "ma_mutation": MA_MUTATION,
            "ma_ls_prob": MA_LS_PROB,
            "n_algo_runs": N_ALGO_RUNS,
            "instance_seeds": INSTANCE_SEEDS,
            "network": "IEEE 33-bus (Baran-Wu)",
            "n_projects": 50,
        },
        "scenarios": {},
    }

    for scenario_name in ["I1", "I2", "I3", "I4"]:
        result = run_scenario(scenario_name)
        all_results["scenarios"][scenario_name] = result

    total_time = time.perf_counter() - t_start

    all_results["total_time_s"] = total_time
    all_results["total_time_h"] = total_time / 3600

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"experiment_optionA_{timestamp}.json"
    output_path = OUTPUT_DIR / fname

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Total time: {total_time/3600:.2f} hours ({total_time:.0f}s)")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 60)

    # Print final summary table
    logger.info("\nFINAL SUMMARY:")
    logger.info(f"{'Scenario':<10} {'Pen%':<8} {'FIFO':<12} {'Greedy-S':<12} {'MA mean':<15} {'vs FIFO':<10}")
    logger.info("-" * 67)
    for sname, sdata in all_results["scenarios"].items():
        inst = sdata["instances"][0]  # first instance
        pen = inst["penetration_pct"]
        fifo = inst["baselines"]["FIFO"]["fitness"]
        greedy = inst["baselines"]["Greedy_Smallest"]["fitness"]
        ma_mean = inst["ma_summary"]["fitness_mean"]
        ma_std = inst["ma_summary"]["fitness_std"]
        gain = inst["gains"]["vs_fifo_pct"]
        logger.info(
            f"{sname:<10} {pen:<8.0f} {fifo:<12.0f} {greedy:<12.0f} "
            f"{ma_mean:.0f}+/-{ma_std:.0f}   {gain:+.1f}%"
        )


if __name__ == "__main__":
    main()
