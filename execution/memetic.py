# -*- coding: utf-8 -*-
"""
memetic.py — Hybrid memetic algorithm combining GA with AOS-driven local search.

Part of: Hybrid Memetic Algorithm for DG Queue Optimization
Paper: "Connection Order Matters: Path-Dependent Hosting Capacity in
        Distributed Generation Interconnection Queues"
Authors: Williams F. Fontinele, Pablo T. Caballero, Eduardo C. M. da Costa
License: MIT
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandapower as pp

from .evaluator import EvalResult, evaluate_permutation
from .network import TechnicalConstraints
from .queue_generator import Project

logger = logging.getLogger(__name__)


@dataclass
class MAConfig:
    """Configuration for the Memetic Algorithm."""
    pop_size: int = 20
    n_generations: int = 30
    crossover_rate: float = 0.9
    mutation_rate: float = 0.15
    local_search_prob: float = 0.3
    elitism_rate: float = 0.05
    aos_window: int = 20
    aos_min_prob: float = 0.1
    d_max: float = float("inf")
    seed: int = 42
    convergence_patience: int = 10
    convergence_threshold: float = 0.001  # 0.1% improvement


@dataclass
class MAResult:
    """Result of MA optimization."""
    best_permutation: list[int]
    best_fitness: float
    best_eval: EvalResult
    fitness_history: list[float]
    generation_converged: int
    total_pf_calls: int
    total_time_s: float
    aos_usage: dict[str, list[float]]


# =====================================================================
# Genetic Operators (Global Search)
# =====================================================================

def order_crossover(parent1: np.ndarray, parent2: np.ndarray, rng) -> np.ndarray:
    """Order Crossover (OX) for permutations."""
    n = len(parent1)
    child = np.full(n, -1, dtype=int)

    # Select crossover segment
    i, j = sorted(rng.choice(n, size=2, replace=False))
    child[i:j+1] = parent1[i:j+1]

    # Fill remaining from parent2 in order
    segment = set(parent1[i:j+1])
    fill_values = [v for v in parent2 if v not in segment]
    fill_idx = 0
    for k in range(n):
        if child[k] == -1:
            child[k] = fill_values[fill_idx]
            fill_idx += 1

    return child


def swap_mutation(perm: np.ndarray, rng) -> np.ndarray:
    """Swap two random positions."""
    result = perm.copy()
    i, j = rng.choice(len(result), size=2, replace=False)
    result[i], result[j] = result[j], result[i]
    return result


def insert_mutation(perm: np.ndarray, rng) -> np.ndarray:
    """Remove element at random position, insert at another."""
    result = list(perm)
    i = rng.integers(len(result))
    val = result.pop(i)
    j = rng.integers(len(result) + 1)
    result.insert(j, val)
    return np.array(result)


# =====================================================================
# Local Search Operators (Memetic Layer)
# =====================================================================

def ls_2opt(perm: np.ndarray, rng) -> np.ndarray:
    """2-opt: reverse a random subsequence."""
    result = perm.copy()
    n = len(result)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    result[i:j+1] = result[i:j+1][::-1]
    return result


def ls_or_opt(perm: np.ndarray, rng) -> np.ndarray:
    """Or-opt: relocate a block of 1-3 elements."""
    result = list(perm)
    n = len(result)
    block_size = rng.integers(1, min(4, n))
    i = rng.integers(n - block_size + 1)
    block = result[i:i+block_size]
    del result[i:i+block_size]
    j = rng.integers(len(result) + 1)
    for k, val in enumerate(block):
        result.insert(j + k, val)
    return np.array(result)


def ls_insert(perm: np.ndarray, rng) -> np.ndarray:
    """Insert: remove and reinsert one element."""
    return insert_mutation(perm, rng)


# =====================================================================
# Adaptive Operator Selection
# =====================================================================

class AdaptiveOperatorSelector:
    """Selects local search operator based on recent success rates."""

    def __init__(
        self,
        operators: list[tuple[str, Callable]],
        window: int = 20,
        min_prob: float = 0.1,
    ):
        self.operators = operators
        self.names = [name for name, _ in operators]
        self.funcs = [func for _, func in operators]
        self.n_ops = len(operators)
        self.window = window
        self.min_prob = min_prob

        # Track successes and applications
        self.successes = [0] * self.n_ops
        self.applications = [0] * self.n_ops
        self.total_applications = 0

        # Probabilities
        self.probs = [1.0 / self.n_ops] * self.n_ops

        # History for analysis
        self.prob_history: list[list[float]] = []

    def select(self, rng) -> tuple[int, Callable]:
        """Select an operator probabilistically."""
        idx = rng.choice(self.n_ops, p=self.probs)
        self.applications[idx] += 1
        self.total_applications += 1
        return idx, self.funcs[idx]

    def report_result(self, op_idx: int, improved: bool) -> None:
        """Report whether the operator improved the solution."""
        if improved:
            self.successes[op_idx] += 1

    def update_probabilities(self) -> None:
        """Recalculate selection probabilities based on success rates."""
        rates = []
        for i in range(self.n_ops):
            if self.applications[i] > 0:
                rates.append(self.successes[i] / self.applications[i])
            else:
                rates.append(0.0)

        total_rate = sum(rates)
        if total_rate > 0:
            raw_probs = [r / total_rate for r in rates]
        else:
            raw_probs = [1.0 / self.n_ops] * self.n_ops

        # Apply minimum probability floor
        self.probs = [
            max(p, self.min_prob) for p in raw_probs
        ]
        # Renormalize
        total = sum(self.probs)
        self.probs = [p / total for p in self.probs]

        # Record history
        self.prob_history.append(list(self.probs))

        # Reset counters for next window
        self.successes = [0] * self.n_ops
        self.applications = [0] * self.n_ops

    def get_usage_history(self) -> dict[str, list[float]]:
        """Return operator probability history."""
        result = {}
        for i, name in enumerate(self.names):
            result[name] = [h[i] for h in self.prob_history]
        return result


# =====================================================================
# Memetic Algorithm
# =====================================================================

def run_memetic_algorithm(
    net_base: pp.pandapowerNet,
    projects: list[Project],
    constraints: TechnicalConstraints,
    config: MAConfig,
) -> MAResult:
    """Run the hybrid memetic algorithm with AOS.

    Args:
        net_base: Base pandapower network.
        projects: List of MMGD projects.
        constraints: Technical constraints.
        config: MA configuration.

    Returns:
        MAResult with best solution and diagnostics.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    n = len(projects)
    project_ids = [p.id for p in projects]
    total_pf_calls = 0

    # Initialize AOS
    aos = AdaptiveOperatorSelector(
        operators=[
            ("2-opt", ls_2opt),
            ("or-opt", ls_or_opt),
            ("insert", ls_insert),
        ],
        window=config.aos_window,
        min_prob=config.aos_min_prob,
    )
    ls_applications_in_window = 0

    # --- Initialize population ---
    population = []
    fitnesses = []

    # First individual: chronological order (FIFO)
    chrono = np.array(project_ids)
    population.append(chrono)

    # Remaining: random permutations
    for _ in range(config.pop_size - 1):
        perm = rng.permutation(project_ids)
        population.append(perm)

    # Evaluate initial population
    for perm in population:
        result = evaluate_permutation(
            net_base, projects, perm.tolist(), constraints, config.d_max
        )
        fitnesses.append(result.fitness)
        total_pf_calls += result.pf_calls

    best_idx = int(np.argmax(fitnesses))
    best_fitness = fitnesses[best_idx]
    best_perm = population[best_idx].copy()
    fitness_history = [best_fitness]

    no_improve_count = 0
    generation_converged = 0

    logger.info(
        f"MA initialized: pop={config.pop_size}, "
        f"best_fitness={best_fitness:.1f} kW"
    )

    # --- Main loop ---
    for gen in range(config.n_generations):
        n_elite = max(1, int(config.pop_size * config.elitism_rate))

        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitnesses)[::-1]
        elite_indices = sorted_indices[:n_elite]

        new_population = []
        new_fitnesses = []

        # Preserve elite
        for ei in elite_indices:
            new_population.append(population[ei].copy())
            new_fitnesses.append(fitnesses[ei])

        # Generate offspring
        while len(new_population) < config.pop_size:
            # Tournament selection
            p1_idx = _tournament_select(fitnesses, rng)
            p2_idx = _tournament_select(fitnesses, rng)

            parent1 = population[p1_idx]
            parent2 = population[p2_idx]

            # Crossover
            if rng.random() < config.crossover_rate:
                child = order_crossover(parent1, parent2, rng)
            else:
                child = parent1.copy()

            # Mutation
            if rng.random() < config.mutation_rate:
                if rng.random() < 0.5:
                    child = swap_mutation(child, rng)
                else:
                    child = insert_mutation(child, rng)

            # Evaluate child first (always needed)
            res_child = evaluate_permutation(
                net_base, projects, child.tolist(),
                constraints, config.d_max,
            )
            child_fitness = res_child.fitness
            total_pf_calls += res_child.pf_calls

            # Local search (memetic layer) with AOS
            if rng.random() < config.local_search_prob:
                op_idx, op_func = aos.select(rng)
                candidate = op_func(child, rng)

                # Evaluate candidate only (child already evaluated above)
                res_candidate = evaluate_permutation(
                    net_base, projects, candidate.tolist(),
                    constraints, config.d_max,
                )
                total_pf_calls += res_candidate.pf_calls

                improved = res_candidate.fitness > child_fitness
                aos.report_result(op_idx, improved)

                if improved:
                    child = candidate
                    child_fitness = res_candidate.fitness

                ls_applications_in_window += 1
                if ls_applications_in_window >= config.aos_window:
                    aos.update_probabilities()
                    ls_applications_in_window = 0

            new_population.append(child)
            new_fitnesses.append(child_fitness)

        population = new_population
        fitnesses = new_fitnesses

        # Track best
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_fitness = fitnesses[gen_best_idx]

        if gen_best_fitness > best_fitness * (1 + config.convergence_threshold):
            best_fitness = gen_best_fitness
            best_perm = population[gen_best_idx].copy()
            no_improve_count = 0
            generation_converged = gen
        else:
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_perm = population[gen_best_idx].copy()
            no_improve_count += 1

        fitness_history.append(best_fitness)

        if gen % 50 == 0:
            logger.info(
                f"Gen {gen}: best={best_fitness:.1f} kW, "
                f"pf_calls={total_pf_calls}"
            )

        # Early stopping
        if no_improve_count >= config.convergence_patience:
            logger.info(f"Converged at generation {gen}.")
            break

    # Final evaluation of best permutation
    best_eval = evaluate_permutation(
        net_base, projects, best_perm.tolist(), constraints, config.d_max
    )
    total_pf_calls += best_eval.pf_calls
    total_time = time.perf_counter() - t0

    logger.info(
        f"MA finished: best={best_fitness:.1f} kW, "
        f"n_connected={best_eval.n_connected}/{n}, "
        f"pf_calls={total_pf_calls}, "
        f"time={total_time:.1f}s"
    )

    return MAResult(
        best_permutation=best_perm.tolist(),
        best_fitness=best_fitness,
        best_eval=best_eval,
        fitness_history=fitness_history,
        generation_converged=generation_converged,
        total_pf_calls=total_pf_calls,
        total_time_s=total_time,
        aos_usage=aos.get_usage_history(),
    )


def _tournament_select(fitnesses: list[float], rng, k: int = 2) -> int:
    """Binary tournament selection."""
    candidates = rng.choice(len(fitnesses), size=k, replace=False)
    best = candidates[0]
    for c in candidates[1:]:
        if fitnesses[c] > fitnesses[best]:
            best = c
    return int(best)
