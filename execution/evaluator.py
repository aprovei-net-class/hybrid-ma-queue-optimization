# -*- coding: utf-8 -*-
"""
evaluator.py — Sequential permutation evaluation with dynamic hosting capacity update.

Part of: Hybrid Memetic Algorithm for DG Queue Optimization
Paper: "Path-Dependent Hosting Capacity and Sequential Queue Optimization
        for Distributed Generation Grid Access"
Authors: Williams F. Fontinele, Pablo T. Caballero, Eduardo C. M. da Costa
License: MIT
"""

import copy
import logging
import time
from dataclasses import dataclass

import numpy as np
import pandapower as pp

from .network import (
    TechnicalConstraints,
    add_sgen,
    check_constraints,
    remove_sgen,
    run_power_flow,
)
from .queue_generator import Project

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating a single permutation."""
    fitness: float              # total connected power (kW)
    n_connected: int            # number of connected projects
    n_rejected: int             # number of rejected projects
    connected_ids: list[int]    # IDs of connected projects
    rejected_ids: list[int]     # IDs of rejected projects
    total_power_kw: float       # same as fitness (for clarity)
    pf_calls: int               # number of power flow evaluations
    eval_time_s: float          # evaluation wall-clock time


def _check_no_new_violations(
    net: pp.pandapowerNet,
    v_before: np.ndarray,
    constraints: TechnicalConstraints,
) -> bool:
    """Check that adding a project did not introduce NEW violations.

    For DG (PV) which raises voltage, the key check is:
    - No bus exceeds v_max (overvoltage — the binding constraint)
    - No bus that was previously above v_min drops below it
    - Line loading does not exceed thermal limit

    Pre-existing undervoltage in the base network is accepted,
    since DG injection typically improves (raises) voltage.
    """
    v_after = net.res_bus.vm_pu.values

    # Check overvoltage (absolute — DG must not push any bus above v_max)
    v_valid = v_after[~np.isnan(v_after)]
    if len(v_valid) == 0:
        return False
    if float(np.max(v_valid)) > constraints.v_max:
        return False

    # Check that no bus NEWLY drops below v_min
    # (a bus already below v_min before is OK; a bus that was OK and now isn't is NOT OK)
    for i in range(len(v_after)):
        if np.isnan(v_after[i]) or np.isnan(v_before[i]):
            continue
        if v_before[i] >= constraints.v_min and v_after[i] < constraints.v_min:
            return False

    # Check thermal limits on lines
    if len(net.res_line) > 0 and "loading_percent" in net.res_line.columns:
        line_loading = net.res_line.loading_percent.values
        line_valid = line_loading[~np.isnan(line_loading)]
        if len(line_valid) > 0 and float(np.max(line_valid)) > constraints.max_line_loading:
            return False

    return True


def evaluate_permutation(
    net_base: pp.pandapowerNet,
    projects: list[Project],
    permutation: list[int],
    constraints: TechnicalConstraints,
    d_max: float = float("inf"),
) -> EvalResult:
    """Evaluate a permutation by sequential power flow with dynamic HC.

    For each project in the permutation order:
    1. Check fairness constraint (displacement ≤ d_max)
    2. Add project as sgen to the network
    3. Run power flow
    4. If all constraints satisfied: keep (dynamic HC update)
    5. If any constraint violated: remove and skip

    Args:
        net_base: Base network (will be deep-copied, not modified).
        projects: List of all projects.
        permutation: Order of project indices to evaluate.
        constraints: Technical constraint thresholds.
        d_max: Maximum displacement from chronological position.

    Returns:
        EvalResult with fitness and diagnostics.
    """
    t0 = time.perf_counter()
    net = copy.deepcopy(net_base)
    pf_calls = 0

    connected_ids = []
    rejected_ids = []
    total_power = 0.0

    # Map project ID to Project object
    proj_map = {p.id: p for p in projects}

    # Run initial power flow to get baseline state
    run_power_flow(net)
    pf_calls += 1
    # Save voltage snapshot for comparison
    v_snapshot = net.res_bus.vm_pu.values.copy()

    for new_pos, proj_id in enumerate(permutation):
        proj = proj_map[proj_id]

        # Fairness check: displacement from chronological position
        displacement = abs(new_pos - proj.chrono_position)
        if displacement > d_max:
            rejected_ids.append(proj_id)
            continue

        # Add project as static generator
        sgen_idx = add_sgen(net, proj.bus, proj.p_kw, name=proj.name)

        # Run power flow
        converged = run_power_flow(net)
        pf_calls += 1

        if not converged:
            remove_sgen(net, sgen_idx)
            rejected_ids.append(proj_id)
            continue

        # Check constraints: only NEW violations caused by this connection
        ok = _check_no_new_violations(net, v_snapshot, constraints)

        if ok:
            # Accept: keep sgen, update network state (dynamic HC)
            connected_ids.append(proj_id)
            total_power += proj.p_kw
            # Update voltage snapshot to current state
            v_snapshot = net.res_bus.vm_pu.values.copy()
        else:
            # Reject: remove sgen
            remove_sgen(net, sgen_idx)
            # Restore res_bus to previous snapshot (avoids extra power flow)
            net.res_bus.vm_pu = v_snapshot.copy()
            rejected_ids.append(proj_id)

    elapsed = time.perf_counter() - t0

    return EvalResult(
        fitness=total_power,
        n_connected=len(connected_ids),
        n_rejected=len(rejected_ids),
        connected_ids=connected_ids,
        rejected_ids=rejected_ids,
        total_power_kw=total_power,
        pf_calls=pf_calls,
        eval_time_s=elapsed,
    )


def evaluate_static_permutation(
    net_base: pp.pandapowerNet,
    projects: list[Project],
    permutation: list[int],
    constraints: TechnicalConstraints,
    d_max: float = float("inf"),
) -> EvalResult:
    """Evaluate with STATIC HC (no dynamic update).

    Same as evaluate_permutation but checks each project against
    the BASE network state (no cumulative updates). Used as baseline
    to quantify the value of dynamic HC.

    Args:
        Same as evaluate_permutation.

    Returns:
        EvalResult with fitness and diagnostics.
    """
    t0 = time.perf_counter()
    pf_calls = 0

    connected_ids = []
    rejected_ids = []
    total_power = 0.0

    proj_map = {p.id: p for p in projects}

    for new_pos, proj_id in enumerate(permutation):
        proj = proj_map[proj_id]

        displacement = abs(new_pos - proj.chrono_position)
        if displacement > d_max:
            rejected_ids.append(proj_id)
            continue

        # Test each project against base network (static HC)
        net_test = copy.deepcopy(net_base)

        # Add all previously connected projects
        for cid in connected_ids:
            cp = proj_map[cid]
            add_sgen(net_test, cp.bus, cp.p_kw, name=cp.name)

        # Add candidate project
        add_sgen(net_test, proj.bus, proj.p_kw, name=proj.name)

        converged = run_power_flow(net_test)
        pf_calls += 1

        if not converged:
            rejected_ids.append(proj_id)
            continue

        ok, _ = check_constraints(net_test, constraints)

        if ok:
            connected_ids.append(proj_id)
            total_power += proj.p_kw
        else:
            rejected_ids.append(proj_id)

    elapsed = time.perf_counter() - t0

    return EvalResult(
        fitness=total_power,
        n_connected=len(connected_ids),
        n_rejected=len(rejected_ids),
        connected_ids=connected_ids,
        rejected_ids=rejected_ids,
        total_power_kw=total_power,
        pf_calls=pf_calls,
        eval_time_s=elapsed,
    )
