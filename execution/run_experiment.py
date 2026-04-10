# -*- coding: utf-8 -*-
"""
run_experiment.py — Main experiment runner for the IEEE 33-bus network.

Part of: Hybrid Memetic Algorithm for DG Queue Optimization
Paper: "Path-Dependent Hosting Capacity and Sequential Queue Optimization
        for Distributed Generation Grid Access"
Authors: Williams F. Fontinele, Pablo T. Caballero, Eduardo C. M. da Costa
License: MIT
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from .baselines import fifo_baseline, greedy_baseline, random_baseline, run_all_baselines
from .evaluator import evaluate_permutation
from .memetic import MAConfig, run_memetic_algorithm
from .network import TechnicalConstraints, load_generic_radial, run_power_flow
from .queue_generator import generate_queue, get_penetration_pct

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_quick_test():
    """Quick validation test with small instance."""
    logger.info("=" * 60)
    logger.info("QUICK TEST: Validating framework")
    logger.info("=" * 60)

    # Load network
    net = load_generic_radial()
    logger.info(f"Network: {len(net.bus)} buses, {len(net.line)} lines")

    # Validate power flow on base network
    converged = run_power_flow(net)
    logger.info(f"Base power flow converged: {converged}")
    if not converged:
        logger.error("Base power flow failed. Check network.")
        return False

    v_min = float(net.res_bus.vm_pu.min())
    v_max = float(net.res_bus.vm_pu.max())
    logger.info(f"Base voltages: min={v_min:.4f}, max={v_max:.4f} pu")

    # Generate small queue
    n_projects = 15
    projects = generate_queue(
        net, n_projects=n_projects, seed=42,
        power_distribution="uniform",
        p_min_kw=5.0, p_max_kw=50.0,
    )
    penetration = get_penetration_pct(net, projects)
    logger.info(f"Queue: {n_projects} projects, penetration={penetration:.1f}%")

    # Technical constraints
    constraints = TechnicalConstraints()

    # Run baselines
    logger.info("-" * 40)
    logger.info("Running baselines...")
    baselines = run_all_baselines(net, projects, constraints, seed=42)
    for name, res in baselines.items():
        logger.info(
            f"  {name:20s}: {res['fitness']:8.1f} kW, "
            f"n_connected={res['n_connected']}/{n_projects}"
        )

    # Run MA (small config for quick test)
    logger.info("-" * 40)
    logger.info("Running Memetic Algorithm (quick config)...")
    ma_config = MAConfig(
        pop_size=20,
        n_generations=30,
        crossover_rate=0.9,
        mutation_rate=0.15,
        local_search_prob=0.3,
        seed=42,
        convergence_patience=10,
    )

    ma_result = run_memetic_algorithm(net, projects, constraints, ma_config)
    logger.info(
        f"  MA result: {ma_result.best_fitness:.1f} kW, "
        f"n_connected={ma_result.best_eval.n_connected}/{n_projects}, "
        f"pf_calls={ma_result.total_pf_calls}, "
        f"time={ma_result.total_time_s:.1f}s"
    )

    # Summary
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    fifo_fitness = baselines["FIFO"]["fitness"]
    ma_fitness = ma_result.best_fitness
    improvement = ((ma_fitness - fifo_fitness) / fifo_fitness * 100
                   if fifo_fitness > 0 else 0)

    logger.info(f"FIFO:    {fifo_fitness:8.1f} kW")
    logger.info(f"MA:      {ma_fitness:8.1f} kW")
    logger.info(f"Gain:    {improvement:+.1f}%")
    logger.info(f"AOS operator usage history entries: {len(ma_result.aos_usage.get('2-opt', []))}")

    logger.info("=" * 60)
    logger.info("Quick test PASSED" if converged else "Quick test FAILED")
    return True


def run_experiment(
    n_projects: int = 50,
    n_generations: int = 100,
    pop_size: int = 50,
    seed: int = 42,
    power_dist: str = "uniform",
    d_max: float = float("inf"),
    output_dir: str = "analysis-output",
):
    """Run a full experiment."""
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT: N={n_projects}, gen={n_generations}, seed={seed}")
    logger.info("=" * 60)

    net = load_generic_radial()
    constraints = TechnicalConstraints()

    projects = generate_queue(
        net, n_projects=n_projects, seed=seed,
        power_distribution=power_dist,
    )
    penetration = get_penetration_pct(net, projects)
    logger.info(f"Penetration: {penetration:.1f}%")

    # Baselines
    baselines = run_all_baselines(net, projects, constraints, d_max=d_max, seed=seed)

    # MA
    ma_config = MAConfig(
        pop_size=pop_size,
        n_generations=n_generations,
        seed=seed,
        d_max=d_max,
    )
    ma_result = run_memetic_algorithm(net, projects, constraints, ma_config)

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = {
        "config": {
            "n_projects": n_projects,
            "n_generations": n_generations,
            "pop_size": pop_size,
            "seed": seed,
            "power_dist": power_dist,
            "d_max": d_max,
            "penetration_pct": penetration,
        },
        "baselines": baselines,
        "ma": {
            "fitness": ma_result.best_fitness,
            "n_connected": ma_result.best_eval.n_connected,
            "pf_calls": ma_result.total_pf_calls,
            "time_s": ma_result.total_time_s,
            "generation_converged": ma_result.generation_converged,
        },
    }

    fname = f"experiment_N{n_projects}_s{seed}.json"
    with open(out_path / fname, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {out_path / fname}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Queue Optimization Experiment")
    parser.add_argument("--quick-test", action="store_true", help="Run quick validation test")
    parser.add_argument("--n-projects", type=int, default=50)
    parser.add_argument("--n-generations", type=int, default=100)
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--power-dist", choices=["uniform", "bimodal"], default="uniform")
    parser.add_argument("--d-max", type=float, default=float("inf"))
    parser.add_argument("--output-dir", default="analysis-output")

    args = parser.parse_args()

    if args.quick_test:
        success = run_quick_test()
        sys.exit(0 if success else 1)
    else:
        run_experiment(
            n_projects=args.n_projects,
            n_generations=args.n_generations,
            pop_size=args.pop_size,
            seed=args.seed,
            power_dist=args.power_dist,
            d_max=args.d_max,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
