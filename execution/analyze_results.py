# -*- coding: utf-8 -*-
"""
analyze_results.py — Result analysis, statistical summaries, and quality assessment.

Part of: Hybrid Memetic Algorithm for DG Queue Optimization
Paper: "Connection Order Matters: Path-Dependent Hosting Capacity in
        Distributed Generation Interconnection Queues"
Authors: Williams F. Fontinele, Pablo T. Caballero, Eduardo C. M. da Costa
License: MIT
"""

import json
import sys
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_results(filepath: str) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)


def analyze_scenario(scenario_data: dict) -> None:
    """Analyze and print results for one scenario."""
    name = scenario_data["scenario"]
    label = scenario_data["label"]
    instances = scenario_data["instances"]

    logger.info(f"\n{'='*70}")
    logger.info(f"Scenario {name} — {label}")
    logger.info(f"{'='*70}")

    for inst in instances:
        seed = inst["instance_seed"]
        pen = inst["penetration_pct"]
        n_proj = inst["n_projects"]
        baselines = inst["baselines"]
        ma = inst["ma_summary"]
        gains = inst["gains"]

        logger.info(f"\n  Instance seed={seed}, penetration={pen:.0f}%")
        logger.info(f"  {'Method':<22} {'Power (kW)':>12} {'Connected':>10} {'vs FIFO':>10}")
        logger.info(f"  {'-'*56}")

        fifo_f = baselines["FIFO"]["fitness"]
        fifo_n = baselines["FIFO"]["n_connected"]
        logger.info(f"  {'FIFO':<22} {fifo_f:>12.0f} {fifo_n:>10}/{n_proj}")

        gl_f = baselines["Greedy_Largest"]["fitness"]
        gl_n = baselines["Greedy_Largest"]["n_connected"]
        gl_gain = (gl_f - fifo_f) / fifo_f * 100 if fifo_f > 0 else 0
        logger.info(f"  {'Greedy-Largest':<22} {gl_f:>12.0f} {gl_n:>10}/{n_proj} {gl_gain:>+9.1f}%")

        gs_f = baselines["Greedy_Smallest"]["fitness"]
        gs_n = baselines["Greedy_Smallest"]["n_connected"]
        gs_gain = (gs_f - fifo_f) / fifo_f * 100 if fifo_f > 0 else 0
        logger.info(f"  {'Greedy-Smallest':<22} {gs_f:>12.0f} {gs_n:>10}/{n_proj} {gs_gain:>+9.1f}%")

        if "Random" in baselines:
            r_f = baselines["Random"].get("fitness_best", baselines["Random"].get("fitness", 0))
            r_mean = baselines["Random"].get("fitness_mean", r_f)
            r_n = baselines["Random"]["n_connected"]
            r_gain = (r_mean - fifo_f) / fifo_f * 100 if fifo_f > 0 else 0
            logger.info(f"  {'Random (mean)':<22} {r_mean:>12.0f} {r_n:>10}/{n_proj} {r_gain:>+9.1f}%")

        ma_f = ma["fitness_mean"]
        ma_std = ma["fitness_std"]
        ma_n = ma["n_connected_mean"]
        ma_gain = gains["vs_fifo_pct"]
        ma_gs_gain = gains["vs_greedy_smallest_pct"]
        logger.info(f"  {'MA (mean +/- std)':<22} {ma_f:>8.0f}+/-{ma_std:<3.0f} {ma_n:>7.1f}/{n_proj} {ma_gain:>+9.1f}%")

        logger.info(f"\n  MA vs FIFO: {ma_gain:+.1f}%")
        logger.info(f"  MA vs Greedy-Smallest: {ma_gs_gain:+.1f}%")


def print_consolidated_table(results: dict) -> None:
    """Print consolidated LaTeX-ready table."""
    logger.info(f"\n{'='*70}")
    logger.info("CONSOLIDATED RESULTS TABLE")
    logger.info(f"{'='*70}")

    logger.info(f"\n{'Scenario':<12} {'Pen%':<6} {'Seed':<6} {'FIFO':>8} {'Gre-S':>8} {'MA mean':>10} {'MA std':>7} {'vs FIFO':>9} {'vs Gre-S':>9}")
    logger.info("-" * 85)

    all_fifo_gains = []
    all_gs_gains = []

    for sname in ["I1", "I2", "I3", "I4"]:
        if sname not in results["scenarios"]:
            continue
        sdata = results["scenarios"][sname]
        for inst in sdata["instances"]:
            seed = inst["instance_seed"]
            pen = inst["penetration_pct"]
            fifo = inst["baselines"]["FIFO"]["fitness"]
            gs = inst["baselines"]["Greedy_Smallest"]["fitness"]
            ma_mean = inst["ma_summary"]["fitness_mean"]
            ma_std = inst["ma_summary"]["fitness_std"]
            g_fifo = inst["gains"]["vs_fifo_pct"]
            g_gs = inst["gains"]["vs_greedy_smallest_pct"]

            logger.info(
                f"{sname:<12} {pen:<6.0f} {seed:<6} {fifo:>8.0f} {gs:>8.0f} "
                f"{ma_mean:>10.0f} {ma_std:>7.0f} {g_fifo:>+9.1f} {g_gs:>+9.1f}"
            )

            if g_fifo != 0:
                all_fifo_gains.append(g_fifo)
            if g_gs != 0:
                all_gs_gains.append(g_gs)

    if all_fifo_gains:
        logger.info(f"\nMean gain over FIFO (restrictive scenarios): {np.mean(all_fifo_gains):+.1f}%")
        logger.info(f"Mean gain over Greedy-Smallest: {np.mean(all_gs_gains):+.1f}%")


def print_ma_run_details(results: dict) -> None:
    """Print MA run-level details for restrictive scenarios."""
    logger.info(f"\n{'='*70}")
    logger.info("MA RUN-LEVEL DETAILS (restrictive scenarios)")
    logger.info(f"{'='*70}")

    for sname in ["I3", "I4"]:
        if sname not in results["scenarios"]:
            continue
        sdata = results["scenarios"][sname]
        for inst in sdata["instances"]:
            seed = inst["instance_seed"]
            runs = inst["ma_runs"]
            fitnesses = [r["fitness"] for r in runs]
            n_conns = [r["n_connected"] for r in runs]
            times = [r["time_s"] for r in runs]
            pf_calls = [r["pf_calls"] for r in runs]

            logger.info(f"\n{sname} seed={seed}: {len(runs)} runs")
            logger.info(f"  Fitness: {np.mean(fitnesses):.0f} +/- {np.std(fitnesses):.0f} "
                        f"[{np.min(fitnesses):.0f}, {np.max(fitnesses):.0f}]")
            logger.info(f"  N_connected: {np.mean(n_conns):.1f} +/- {np.std(n_conns):.1f} "
                        f"[{np.min(n_conns)}, {np.max(n_conns)}]")
            logger.info(f"  Time: {np.mean(times):.0f} +/- {np.std(times):.0f} s")
            logger.info(f"  PF calls: {np.mean(pf_calls):.0f} +/- {np.std(pf_calls):.0f}")


def assess_quality(results: dict) -> bool:
    """Assess if results are good enough for the paper."""
    logger.info(f"\n{'='*70}")
    logger.info("QUALITY ASSESSMENT")
    logger.info(f"{'='*70}")

    issues = []
    good = []

    for sname in ["I3", "I4"]:
        if sname not in results["scenarios"]:
            issues.append(f"{sname}: MISSING")
            continue
        sdata = results["scenarios"][sname]
        for inst in sdata["instances"]:
            seed = inst["instance_seed"]
            g_fifo = inst["gains"]["vs_fifo_pct"]
            g_gs = inst["gains"]["vs_greedy_smallest_pct"]
            ma_std = inst["ma_summary"]["fitness_std"]
            ma_mean = inst["ma_summary"]["fitness_mean"]
            cv = (ma_std / ma_mean * 100) if ma_mean > 0 else 0

            # Check MA beats FIFO
            if g_fifo > 0:
                good.append(f"{sname}-s{seed}: MA +{g_fifo:.1f}% vs FIFO")
            else:
                issues.append(f"{sname}-s{seed}: MA does NOT beat FIFO ({g_fifo:+.1f}%)")

            # Check coefficient of variation
            if cv < 5:
                good.append(f"{sname}-s{seed}: Low CV ({cv:.1f}%) — stable")
            else:
                issues.append(f"{sname}-s{seed}: High CV ({cv:.1f}%) — unstable")

    logger.info("\nPositive findings:")
    for g in good:
        logger.info(f"  [+] {g}")

    if issues:
        logger.info("\nIssues:")
        for i in issues:
            logger.info(f"  [!] {i}")

    overall = len(issues) == 0
    logger.info(f"\nOverall assessment: {'PASS — results are publishable' if overall else 'NEEDS ATTENTION'}")
    return overall


def main():
    if len(sys.argv) < 2:
        # Find the latest results file
        results_dir = Path("analysis-output")
        files = sorted(results_dir.glob("experiment_optionA_*.json"))
        if not files:
            logger.error("No results files found in analysis-output/")
            sys.exit(1)
        filepath = str(files[-1])
        logger.info(f"Using latest results file: {filepath}")
    else:
        filepath = sys.argv[1]

    results = load_results(filepath)

    logger.info(f"Experiment: {results.get('experiment', '?')}")
    logger.info(f"Timestamp: {results.get('timestamp', '?')}")
    logger.info(f"Total time: {results.get('total_time_h', 0):.2f} hours")

    # Analyze each scenario
    for sname in ["I1", "I2", "I3", "I4"]:
        if sname in results["scenarios"]:
            analyze_scenario(results["scenarios"][sname])

    # Consolidated table
    print_consolidated_table(results)

    # MA run details
    print_ma_run_details(results)

    # Quality assessment
    assess_quality(results)


if __name__ == "__main__":
    main()
