# -*- coding: utf-8 -*-
"""
queue_generator.py — Random DG project queue generation with configurable distributions.

Part of: Hybrid Memetic Algorithm for DG Queue Optimization
Paper: "Path-Dependent Hosting Capacity and Sequential Queue Optimization
        for Distributed Generation Grid Access"
Authors: Williams F. Fontinele, Pablo T. Caballero, Eduardo C. M. da Costa
License: MIT
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .network import get_valid_buses

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Project:
    """A single MMGD project in the access queue."""
    id: int
    bus: int
    p_kw: float
    chrono_position: int  # original FIFO position (0-indexed)
    name: str = ""

    def __repr__(self) -> str:
        return f"Project(id={self.id}, bus={self.bus}, p_kw={self.p_kw:.1f})"


def generate_queue(
    net,
    n_projects: int,
    seed: int = 42,
    power_distribution: str = "uniform",
    p_min_kw: float = 1.0,
    p_max_kw: float = 75.0,
    p_micro_max_kw: float = 7.5,
    p_mini_min_kw: float = 10.0,
    p_mini_max_kw: float = 75.0,
    micro_fraction: float = 0.7,
) -> list[Project]:
    """Generate a synthetic MMGD project queue.

    Args:
        net: pandapower network (used to determine valid buses).
        n_projects: Number of projects to generate.
        seed: Random seed for reproducibility.
        power_distribution: "uniform" or "bimodal" (micro+mini).
        p_min_kw: Minimum power for uniform distribution.
        p_max_kw: Maximum power for uniform distribution.
        p_micro_max_kw: Max power for micro-generation (bimodal).
        p_mini_min_kw: Min power for mini-generation (bimodal).
        p_mini_max_kw: Max power for mini-generation (bimodal).
        micro_fraction: Fraction of micro projects (bimodal).

    Returns:
        List of Project objects in chronological order.
    """
    rng = np.random.default_rng(seed)
    valid_buses = get_valid_buses(net)

    if not valid_buses:
        raise ValueError("No valid buses for DG connection.")

    # Assign buses (with replacement — multiple projects can share a bus)
    buses = rng.choice(valid_buses, size=n_projects, replace=True)

    # Generate power ratings
    if power_distribution == "uniform":
        powers = rng.uniform(p_min_kw, p_max_kw, size=n_projects)
    elif power_distribution == "bimodal":
        n_micro = int(n_projects * micro_fraction)
        n_mini = n_projects - n_micro
        p_micro = rng.uniform(p_min_kw, p_micro_max_kw, size=n_micro)
        p_mini = rng.uniform(p_mini_min_kw, p_mini_max_kw, size=n_mini)
        powers = np.concatenate([p_micro, p_mini])
        # Shuffle to mix micro and mini in the queue
        order = rng.permutation(n_projects)
        powers = powers[order]
        buses = buses[order]
    else:
        raise ValueError(
            f"Unknown power distribution: {power_distribution}. "
            f"Use 'uniform' or 'bimodal'."
        )

    projects = []
    for i in range(n_projects):
        proj = Project(
            id=i,
            bus=int(buses[i]),
            p_kw=float(powers[i]),
            chrono_position=i,
            name=f"MMGD_{i:03d}",
        )
        projects.append(proj)

    total_kw = sum(p.p_kw for p in projects)
    logger.info(
        f"Generated queue: {n_projects} projects, "
        f"total={total_kw:.1f} kW, "
        f"distribution={power_distribution}, "
        f"seed={seed}"
    )
    return projects


def get_penetration_pct(net, projects: list[Project]) -> float:
    """Calculate DG penetration as percentage of total load.

    Args:
        net: pandapower network.
        projects: List of projects.

    Returns:
        Penetration percentage.
    """
    total_dg_kw = sum(p.p_kw for p in projects)
    total_load_kw = float(net.load.p_mw.sum()) * 1000.0
    if total_load_kw <= 0:
        return float("inf")
    return (total_dg_kw / total_load_kw) * 100.0
