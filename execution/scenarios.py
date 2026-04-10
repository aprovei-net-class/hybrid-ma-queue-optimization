# -*- coding: utf-8 -*-
"""
scenarios.py — Penetration scenario generation (I1--I4) for queue experiments.

Part of: Hybrid Memetic Algorithm for DG Queue Optimization
Paper: "Path-Dependent Hosting Capacity and Sequential Queue Optimization
        for Distributed Generation Grid Access"
Authors: Williams F. Fontinele, Pablo T. Caballero, Eduardo C. M. da Costa
License: MIT
"""

import logging
from dataclasses import dataclass

import pandapower as pp
import pandapower.networks as pn

from .network import TechnicalConstraints, create_ieee69, run_power_flow
from .queue_generator import Project, generate_queue, get_penetration_pct

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration for a single experimental scenario."""
    name: str
    label: str
    n_projects: int
    p_min_kw: float
    p_max_kw: float
    power_distribution: str
    target_penetration_pct: float
    constraints: TechnicalConstraints


# =====================================================================
# Scenario Definitions
# =====================================================================

SCENARIOS = {
    "I1": ScenarioConfig(
        name="I1",
        label="Low (~50%)",
        n_projects=50,
        p_min_kw=5.0,
        p_max_kw=70.0,
        power_distribution="uniform",
        target_penetration_pct=50.0,
        constraints=TechnicalConstraints(),
    ),
    "I2": ScenarioConfig(
        name="I2",
        label="Medium (~120%)",
        n_projects=50,
        p_min_kw=20.0,
        p_max_kw=160.0,
        power_distribution="uniform",
        target_penetration_pct=120.0,
        constraints=TechnicalConstraints(),
    ),
    "I3": ScenarioConfig(
        name="I3",
        label="High (~250%)",
        n_projects=50,
        p_min_kw=50.0,
        p_max_kw=320.0,
        power_distribution="uniform",
        target_penetration_pct=250.0,
        constraints=TechnicalConstraints(),
    ),
    "I4": ScenarioConfig(
        name="I4",
        label="Stress (~500%)",
        n_projects=50,
        p_min_kw=100.0,
        p_max_kw=640.0,
        power_distribution="uniform",
        target_penetration_pct=500.0,
        constraints=TechnicalConstraints(),
    ),
}


def load_test_network() -> pp.pandapowerNet:
    """Load the IEEE 33-bus (Baran-Wu) distribution network.

    This is a 12.66 kV radial distribution system with 33 buses,
    32 loads, and 37 lines. Total load: 3,715 kW.

    Widely used benchmark for DG allocation studies (1,800+ citations).
    Base V_min ~ 0.913 pu — sensitive to DG injection.

    Returns:
        pandapower network object.
    """
    net = pn.case33bw()
    logger.info(
        f"Loaded IEEE 33-bus (Baran-Wu): "
        f"{len(net.bus)} buses, {len(net.line)} lines, "
        f"{len(net.load)} loads, "
        f"total_load={net.load.p_mw.sum()*1000:.0f} kW"
    )
    return net


def create_scenario(
    scenario_name: str,
    seed: int = 42,
) -> tuple[pp.pandapowerNet, list[Project], ScenarioConfig]:
    """Create a complete scenario: network + project queue.

    Args:
        scenario_name: One of "I1", "I2", "I3", "I4".
        seed: Random seed for queue generation.

    Returns:
        Tuple of (network, projects, config).
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario: {scenario_name}. "
            f"Available: {list(SCENARIOS.keys())}"
        )

    config = SCENARIOS[scenario_name]
    net = load_test_network()

    # Validate base power flow
    converged = run_power_flow(net)
    if not converged:
        raise RuntimeError("Base network power flow did not converge.")

    total_load_kw = float(net.load.p_mw.sum()) * 1000.0

    # Generate queue
    projects = generate_queue(
        net,
        n_projects=config.n_projects,
        seed=seed,
        power_distribution=config.power_distribution,
        p_min_kw=config.p_min_kw,
        p_max_kw=config.p_max_kw,
    )

    actual_penetration = get_penetration_pct(net, projects)

    logger.info(
        f"Scenario {config.name} ({config.label}): "
        f"N={config.n_projects}, "
        f"penetration={actual_penetration:.1f}% "
        f"(target={config.target_penetration_pct:.0f}%), "
        f"total_dg={sum(p.p_kw for p in projects):.0f} kW, "
        f"total_load={total_load_kw:.0f} kW, "
        f"seed={seed}"
    )

    return net, projects, config


def create_scenario_69bus(
    scenario_name: str,
    seed: int = 42,
) -> tuple[pp.pandapowerNet, list[Project], ScenarioConfig]:
    """Create a scenario using the IEEE 69-bus network.

    Same scenario configs (I1-I4) but on the 69-bus network.

    Args:
        scenario_name: One of "I1", "I2", "I3", "I4".
        seed: Random seed for queue generation.

    Returns:
        Tuple of (network, projects, config).
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario: {scenario_name}. "
            f"Available: {list(SCENARIOS.keys())}"
        )

    config = SCENARIOS[scenario_name]
    net = create_ieee69()

    converged = run_power_flow(net)
    if not converged:
        raise RuntimeError("IEEE 69-bus base power flow did not converge.")

    total_load_kw = float(net.load.p_mw.sum()) * 1000.0

    projects = generate_queue(
        net,
        n_projects=config.n_projects,
        seed=seed,
        power_distribution=config.power_distribution,
        p_min_kw=config.p_min_kw,
        p_max_kw=config.p_max_kw,
    )

    actual_penetration = get_penetration_pct(net, projects)

    logger.info(
        f"Scenario {config.name} (69-bus, {config.label}): "
        f"N={config.n_projects}, "
        f"penetration={actual_penetration:.1f}%, "
        f"total_dg={sum(p.p_kw for p in projects):.0f} kW, "
        f"total_load={total_load_kw:.0f} kW, "
        f"seed={seed}"
    )

    return net, projects, config


def print_scenario_summary(
    net: pp.pandapowerNet,
    projects: list[Project],
    config: ScenarioConfig,
) -> None:
    """Print a formatted scenario summary."""
    total_load_kw = float(net.load.p_mw.sum()) * 1000.0
    total_dg_kw = sum(p.p_kw for p in projects)
    penetration = (total_dg_kw / total_load_kw * 100) if total_load_kw > 0 else 0
    powers = [p.p_kw for p in projects]

    print(f"\n{'='*50}")
    print(f"Scenario: {config.name} — {config.label}")
    print(f"{'='*50}")
    print(f"Network:     IEEE 33-bus (Baran-Wu)")
    print(f"Total load:  {total_load_kw:.0f} kW")
    print(f"Projects:    {len(projects)}")
    print(f"Total DG:    {total_dg_kw:.0f} kW")
    print(f"Penetration: {penetration:.1f}%")
    print(f"Power range: {min(powers):.1f} - {max(powers):.1f} kW")
    print(f"Mean power:  {sum(powers)/len(powers):.1f} kW")
    print(f"Constraints: V=[{config.constraints.v_min}, {config.constraints.v_max}] pu")
    print(f"             Line loading <= {config.constraints.max_line_loading}%")
    print(f"             Loss <= {config.constraints.max_total_loss_pct}%")
    print(f"{'='*50}\n")
