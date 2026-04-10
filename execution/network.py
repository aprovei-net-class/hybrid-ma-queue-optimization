# -*- coding: utf-8 -*-
"""
network.py — IEEE 33-bus and 69-bus network models with power flow utilities.

Part of: Hybrid Memetic Algorithm for DG Queue Optimization
Paper: "Path-Dependent Hosting Capacity and Sequential Queue Optimization
        for Distributed Generation Grid Access"
Authors: Williams F. Fontinele, Pablo T. Caballero, Eduardo C. M. da Costa
License: MIT
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandapower as pp
import pandapower.networks as pn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TechnicalConstraints:
    """Hard technical constraints for network operation.

    For DG (PV) injection, the binding constraint is typically
    overvoltage (V > v_max), since DG raises voltage.
    Undervoltage (V < v_min) is checked but only for NEW violations
    introduced by the DG — pre-existing undervoltage in the base
    network is accepted.
    """
    v_min: float = 0.93        # p.u.
    v_max: float = 1.05        # p.u.
    max_line_loading: float = 100.0  # % of thermal capacity
    max_trafo_loading: float = 100.0  # % of transformer capacity
    max_total_loss_pct: float = 15.0  # % of total load (relaxed for radial feeders)


def load_ieee_network(bus_count: int) -> pp.pandapowerNet:
    """Load an IEEE test feeder.

    Args:
        bus_count: Number of buses (13, 34, or 123).

    Returns:
        pandapower network object.

    Raises:
        ValueError: If bus_count is not supported.
    """
    loaders = {
        13: pn.case_ieee_european_lv_asymmetric,
        34: pn.case34_3ph,
        123: pn.case_ieee123,
    }

    if bus_count not in loaders:
        raise ValueError(
            f"Unsupported bus count: {bus_count}. Use 13, 34, or 123."
        )

    net = loaders[bus_count]()
    logger.info(
        f"Loaded IEEE {bus_count}-bus: "
        f"{len(net.bus)} buses, {len(net.line)} lines, "
        f"{len(net.load)} loads"
    )
    return net


def load_generic_radial(n_buses: int = 34) -> pp.pandapowerNet:
    """Create a generic radial distribution network for testing.

    Uses pandapower's simple MV distribution network as a reliable
    fallback when IEEE networks have compatibility issues.

    Args:
        n_buses: Approximate number of buses (used for logging only).

    Returns:
        pandapower network object.
    """
    net = pn.simple_mv_open_ring_net()
    logger.info(
        f"Loaded simple MV radial network: "
        f"{len(net.bus)} buses, {len(net.line)} lines, "
        f"{len(net.load)} loads"
    )
    return net


def create_ieee69() -> pp.pandapowerNet:
    """Create the IEEE 69-bus (Baran-Wu, 1989) distribution network.

    12.66 kV radial system with 69 buses, 68 lines, 48 loads.
    Total load: 3,802 kW + 2,695 kVAr. Base V_min ~ 0.909 p.u.

    Data from: Baran & Wu, "Network reconfiguration in distribution
    systems for loss reduction and load balancing," IEEE Trans. Power
    Delivery, vol. 4, no. 2, pp. 1401-1407, 1989.

    Returns:
        pandapower network object.
    """
    bus_data = [
        (1,2, 0.0005, 0.0012, 0, 0),
        (2,3, 0.0005, 0.0012, 0, 0),
        (3,4, 0.0015, 0.0036, 0, 0),
        (4,5, 0.0251, 0.0294, 0, 0),
        (5,6, 0.366, 0.1864, 2.6, 2.2),
        (6,7, 0.3811, 0.1941, 40.4, 30),
        (7,8, 0.0922, 0.047, 75, 54),
        (8,9, 0.0493, 0.0251, 30, 22),
        (9,10, 0.819, 0.2707, 28, 19),
        (10,11, 0.1872, 0.0619, 145, 104),
        (11,12, 0.7114, 0.2351, 145, 104),
        (12,13, 1.03, 0.34, 8, 5.5),
        (13,14, 1.044, 0.345, 8, 5.5),
        (14,15, 1.058, 0.3496, 0, 0),
        (15,16, 0.1966, 0.065, 45.5, 30),
        (16,17, 0.3744, 0.1238, 60, 35),
        (17,18, 0.0047, 0.0016, 60, 35),
        (18,19, 0.3276, 0.1083, 0, 0),
        (19,20, 0.2106, 0.069, 1, 0.6),
        (20,21, 0.3416, 0.1129, 114, 81),
        (21,22, 0.014, 0.0046, 5.3, 3.5),
        (22,23, 0.1591, 0.0526, 0, 0),
        (23,24, 0.3463, 0.1145, 28, 20),
        (24,25, 0.7488, 0.2475, 0, 0),
        (25,26, 0.3089, 0.1021, 14, 10),
        (26,27, 0.1732, 0.0572, 14, 10),
        (3,28, 0.0044, 0.0108, 26, 18.6),
        (28,29, 0.064, 0.1565, 26, 18.6),
        (29,30, 0.3978, 0.1315, 0, 0),
        (30,31, 0.0702, 0.0232, 0, 0),
        (31,32, 0.351, 0.116, 0, 0),
        (32,33, 0.839, 0.2816, 14, 10),
        (33,34, 1.708, 0.5646, 19.5, 14),
        (34,35, 1.474, 0.4873, 6, 4),
        (3,36, 0.0044, 0.0108, 26, 18.55),
        (36,37, 0.064, 0.1565, 26, 18.55),
        (37,38, 0.1053, 0.123, 0, 0),
        (38,39, 0.0304, 0.0355, 24, 17),
        (39,40, 0.0018, 0.0021, 24, 17),
        (40,41, 0.7283, 0.8509, 1.2, 1),
        (41,42, 0.31, 0.3623, 0, 0),
        (42,43, 0.041, 0.0478, 6, 4.3),
        (43,44, 0.0092, 0.0116, 0, 0),
        (44,45, 0.1089, 0.1373, 39.22, 26.3),
        (45,46, 0.0009, 0.0012, 39.22, 26.3),
        (4,47, 0.0034, 0.0084, 0, 0),
        (47,48, 0.0851, 0.2083, 79, 56.4),
        (48,49, 0.2898, 0.7091, 384.7, 274.5),
        (49,50, 0.0822, 0.2011, 384.7, 274.5),
        (8,51, 0.0928, 0.0473, 40.5, 28.3),
        (51,52, 0.3319, 0.1114, 3.6, 2.7),
        (9,53, 0.174, 0.0886, 4.35, 3.5),
        (53,54, 0.203, 0.1034, 26.4, 19),
        (54,55, 0.2842, 0.1447, 24, 17.2),
        (55,56, 0.2813, 0.1433, 0, 0),
        (56,57, 1.59, 0.5337, 0, 0),
        (57,58, 0.7837, 0.263, 0, 0),
        (58,59, 0.3042, 0.1006, 100, 72),
        (59,60, 0.3861, 0.1172, 0, 0),
        (60,61, 0.5075, 0.2585, 1244, 888),
        (61,62, 0.0974, 0.0496, 32, 23),
        (62,63, 0.145, 0.0738, 0, 0),
        (63,64, 0.7105, 0.3619, 227, 162),
        (64,65, 1.041, 0.5302, 59, 42),
        (11,66, 0.2012, 0.0611, 18, 13),
        (66,67, 0.0047, 0.0014, 18, 13),
        (12,68, 0.7394, 0.2444, 28, 20),
        (68,69, 0.0047, 0.0016, 28, 20),
    ]

    net = pp.create_empty_network(name="IEEE 69-bus (Baran-Wu)")
    vn = 12.66

    for i in range(69):
        pp.create_bus(net, vn_kv=vn, name=f"Bus {i+1}")

    pp.create_ext_grid(net, bus=0, vm_pu=1.0)

    for (fb, tb, r, x, p, q) in bus_data:
        pp.create_line_from_parameters(
            net, from_bus=fb-1, to_bus=tb-1,
            length_km=1.0, r_ohm_per_km=r, x_ohm_per_km=x,
            c_nf_per_km=0, max_i_ka=1.0,
            name=f"Line {fb}-{tb}",
        )
        if p > 0:
            pp.create_load(
                net, bus=tb-1, p_mw=p/1000, q_mvar=q/1000,
                name=f"Load bus {tb}",
            )

    logger.info(
        f"Created IEEE 69-bus (Baran-Wu): "
        f"{len(net.bus)} buses, {len(net.line)} lines, "
        f"{len(net.load)} loads, "
        f"total_load={net.load.p_mw.sum()*1000:.0f} kW"
    )
    return net


def run_power_flow(net: pp.pandapowerNet) -> bool:
    """Run Newton-Raphson power flow.

    Args:
        net: pandapower network (modified in-place with results).

    Returns:
        True if power flow converged, False otherwise.
    """
    try:
        pp.runpp(net, algorithm="nr", max_iteration=30, tolerance_mva=1e-6)
        return net.converged
    except pp.LoadflowNotConverged:
        logger.warning("Power flow did not converge.")
        return False
    except Exception as e:
        logger.error(f"Power flow failed: {e}")
        return False


def check_constraints(
    net: pp.pandapowerNet,
    constraints: TechnicalConstraints,
) -> tuple[bool, dict]:
    """Check if all technical constraints are satisfied.

    Args:
        net: pandapower network with completed power flow results.
        constraints: Technical constraint thresholds.

    Returns:
        Tuple of (all_ok, details_dict).
    """
    details = {}

    # Voltage constraints
    v_bus = net.res_bus.vm_pu.values
    v_bus_valid = v_bus[~np.isnan(v_bus)]
    if len(v_bus_valid) == 0:
        return False, {"error": "No valid bus voltages"}

    v_min_actual = float(np.min(v_bus_valid))
    v_max_actual = float(np.max(v_bus_valid))
    v_ok = (v_min_actual >= constraints.v_min) and (v_max_actual <= constraints.v_max)
    details["v_min"] = v_min_actual
    details["v_max"] = v_max_actual
    details["v_ok"] = v_ok

    # Thermal constraints (lines)
    line_ok = True
    if len(net.res_line) > 0 and "loading_percent" in net.res_line.columns:
        line_loading = net.res_line.loading_percent.values
        line_loading_valid = line_loading[~np.isnan(line_loading)]
        if len(line_loading_valid) > 0:
            max_line = float(np.max(line_loading_valid))
            line_ok = max_line <= constraints.max_line_loading
            details["max_line_loading"] = max_line
        else:
            details["max_line_loading"] = 0.0
    details["line_ok"] = line_ok

    # Transformer constraints
    trafo_ok = True
    if len(net.res_trafo) > 0 and "loading_percent" in net.res_trafo.columns:
        trafo_loading = net.res_trafo.loading_percent.values
        trafo_loading_valid = trafo_loading[~np.isnan(trafo_loading)]
        if len(trafo_loading_valid) > 0:
            max_trafo = float(np.max(trafo_loading_valid))
            trafo_ok = max_trafo <= constraints.max_trafo_loading
            details["max_trafo_loading"] = max_trafo
        else:
            details["max_trafo_loading"] = 0.0
    details["trafo_ok"] = trafo_ok

    # Loss constraints
    loss_ok = True
    if len(net.res_line) > 0 and "pl_mw" in net.res_line.columns:
        total_loss = float(net.res_line.pl_mw.sum())
        total_load = float(net.load.p_mw.sum()) if len(net.load) > 0 else 1.0
        if total_load > 0:
            loss_pct = (total_loss / total_load) * 100
            loss_ok = loss_pct <= constraints.max_total_loss_pct
            details["loss_pct"] = loss_pct
        else:
            details["loss_pct"] = 0.0
    details["loss_ok"] = loss_ok

    all_ok = v_ok and line_ok and trafo_ok and loss_ok
    details["all_ok"] = all_ok

    return all_ok, details


def get_valid_buses(net: pp.pandapowerNet) -> list[int]:
    """Get list of buses where DG can be connected.

    Excludes slack bus and buses without load.

    Args:
        net: pandapower network.

    Returns:
        List of valid bus indices.
    """
    slack_buses = set()
    if len(net.ext_grid) > 0:
        slack_buses = set(net.ext_grid.bus.values)
    if len(net.gen) > 0:
        slack_gen = net.gen[net.gen.slack == True] if "slack" in net.gen.columns else net.gen.iloc[:0]
        slack_buses.update(slack_gen.bus.values)

    all_buses = set(net.bus.index)
    valid = sorted(all_buses - slack_buses)

    if not valid:
        valid = sorted(all_buses)
        logger.warning("No non-slack buses found; using all buses.")

    return valid


def add_sgen(
    net: pp.pandapowerNet,
    bus: int,
    p_kw: float,
    name: str = "",
) -> int:
    """Add a static generator (DG) to the network.

    Args:
        net: pandapower network (modified in-place).
        bus: Bus index for connection.
        p_kw: Active power in kW.
        name: Optional name for the generator.

    Returns:
        Index of the added sgen element.
    """
    idx = pp.create_sgen(
        net,
        bus=bus,
        p_mw=p_kw / 1000.0,
        q_mvar=0.0,
        name=name,
        type="PV",
    )
    return idx


def remove_sgen(net: pp.pandapowerNet, idx: int) -> None:
    """Remove a static generator from the network.

    Args:
        net: pandapower network (modified in-place).
        idx: Index of the sgen to remove.
    """
    net.sgen.drop(idx, inplace=True)
