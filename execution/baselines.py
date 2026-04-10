# -*- coding: utf-8 -*-
"""
baselines.py — Baseline methods (FIFO, Greedy-Largest, Greedy-Smallest, Random, GA-only).

Part of: Hybrid Memetic Algorithm for DG Queue Optimization
Paper: "Path-Dependent Hosting Capacity and Sequential Queue Optimization
        for Distributed Generation Grid Access"
Authors: Williams F. Fontinele, Pablo T. Caballero, Eduardo C. M. da Costa
License: MIT
"""

import logging
import time
from typing import Optional

import numpy as np
import pandapower as pp

from .evaluator import EvalResult, evaluate_permutation
from .memetic import (
    MAConfig,
    MAResult,
    order_crossover,
    swap_mutation,
    insert_mutation,
    _tournament_select,
)
from .network import TechnicalConstraints
from .queue_generator import Project

logger = logging.getLogger(__name__)


def fifo_baseline(
    net_base: pp.pandapowerNet,
    projects: list[Project],
    constraints: TechnicalConstraints,
    d_max: float = float("inf"),
) -> EvalResult:
    """FIFO: chronological order (current Brazilian practice)."""
    perm = [p.id for p in sorted(projects, key=lambda p: p.chrono_position)]
    return evaluate_permutation(net_base, projects, perm, constraints, d_max)


def greedy_baseline(
    net_base: pp.pandapowerNet,
    projects: list[Project],
    constraints: TechnicalConstraints,
    d_max: float = float("inf"),
) -> EvalResult:
    """Greedy: sort by power descending (largest first).

    Intuition: connecting larger projects first maximizes total kW
    when constraints are not tight.
    """
    perm = [p.id for p in sorted(projects, key=lambda p: -p.p_kw)]
    return evaluate_permutation(net_base, projects, perm, constraints, d_max)


def greedy_smallest_first(
    net_base: pp.pandapowerNet,
    projects: list[Project],
    constraints: TechnicalConstraints,
    d_max: float = float("inf"),
) -> EvalResult:
    """Greedy smallest first: connect smallest projects first.

    Intuition: under tight constraints, many small projects may
    connect where few large ones would.
    """
    perm = [p.id for p in sorted(projects, key=lambda p: p.p_kw)]
    return evaluate_permutation(net_base, projects, perm, constraints, d_max)


def random_baseline(
    net_base: pp.pandapowerNet,
    projects: list[Project],
    constraints: TechnicalConstraints,
    n_samples: int = 100,
    seed: int = 42,
    d_max: float = float("inf"),
) -> tuple[EvalResult, float, float]:
    """Random: mean fitness over n_samples random permutations.

    Returns:
        Tuple of (best_result, mean_fitness, std_fitness).
    """
    rng = np.random.default_rng(seed)
    project_ids = [p.id for p in projects]

    results = []
    for _ in range(n_samples):
        perm = rng.permutation(project_ids).tolist()
        res = evaluate_permutation(net_base, projects, perm, constraints, d_max)
        results.append(res)

    fitnesses = [r.fitness for r in results]
    best_idx = int(np.argmax(fitnesses))

    return results[best_idx], float(np.mean(fitnesses)), float(np.std(fitnesses))


def ga_only_baseline(
    net_base: pp.pandapowerNet,
    projects: list[Project],
    constraints: TechnicalConstraints,
    config: MAConfig,
) -> MAResult:
    """GA without local search (ablation baseline).

    Same as MA but with local_search_prob = 0.
    """
    from .memetic import run_memetic_algorithm

    config_ga = MAConfig(
        pop_size=config.pop_size,
        n_generations=config.n_generations,
        crossover_rate=config.crossover_rate,
        mutation_rate=config.mutation_rate,
        local_search_prob=0.0,  # No local search
        elitism_rate=config.elitism_rate,
        d_max=config.d_max,
        seed=config.seed,
        convergence_patience=config.convergence_patience,
        convergence_threshold=config.convergence_threshold,
    )
    return run_memetic_algorithm(net_base, projects, constraints, config_ga)


def run_all_baselines(
    net_base: pp.pandapowerNet,
    projects: list[Project],
    constraints: TechnicalConstraints,
    d_max: float = float("inf"),
    seed: int = 42,
) -> dict[str, dict]:
    """Run all simple baselines and return results.

    Args:
        net_base: Base network.
        projects: Project queue.
        constraints: Technical constraints.
        d_max: Fairness constraint.
        seed: Random seed.

    Returns:
        Dictionary mapping baseline name to result dict.
    """
    results = {}

    logger.info("Running FIFO baseline...")
    fifo = fifo_baseline(net_base, projects, constraints, d_max)
    results["FIFO"] = {
        "fitness": fifo.fitness,
        "n_connected": fifo.n_connected,
        "pf_calls": fifo.pf_calls,
        "time_s": fifo.eval_time_s,
    }

    logger.info("Running Greedy (largest first) baseline...")
    greedy_l = greedy_baseline(net_base, projects, constraints, d_max)
    results["Greedy_Largest"] = {
        "fitness": greedy_l.fitness,
        "n_connected": greedy_l.n_connected,
        "pf_calls": greedy_l.pf_calls,
        "time_s": greedy_l.eval_time_s,
    }

    logger.info("Running Greedy (smallest first) baseline...")
    greedy_s = greedy_smallest_first(net_base, projects, constraints, d_max)
    results["Greedy_Smallest"] = {
        "fitness": greedy_s.fitness,
        "n_connected": greedy_s.n_connected,
        "pf_calls": greedy_s.pf_calls,
        "time_s": greedy_s.eval_time_s,
    }

    logger.info("Running Random baseline (100 samples)...")
    rand_best, rand_mean, rand_std = random_baseline(
        net_base, projects, constraints,
        n_samples=100, seed=seed, d_max=d_max,
    )
    results["Random"] = {
        "fitness": rand_best.fitness,
        "fitness_mean": rand_mean,
        "fitness_std": rand_std,
        "n_connected": rand_best.n_connected,
        "pf_calls": rand_best.pf_calls * 100,
        "time_s": rand_best.eval_time_s * 100,
    }

    return results
