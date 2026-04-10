# -*- coding: utf-8 -*-
"""
generate_figures.py — Publication-quality figure generation for main experiment results.

Part of: Hybrid Memetic Algorithm for DG Queue Optimization
Paper: "Path-Dependent Hosting Capacity and Sequential Queue Optimization
        for Distributed Generation Grid Access"
Authors: Williams F. Fontinele, Pablo T. Caballero, Eduardo C. M. da Costa
License: MIT
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Style configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'FIFO': '#d62728',
    'Greedy_Largest': '#ff7f0e',
    'Greedy_Smallest': '#2ca02c',
    'Random': '#7f7f7f',
    'MA': '#1f77b4',
    'MA_static': '#9467bd',
    'MA_fixed': '#8c564b',
    'GA_only': '#e377c2',
}

OUTPUT_DIR = Path("analysis-output/figures")


def load_results(filepath: str) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)


def find_latest_results(pattern: str = "experiment_optionA_*.json") -> str:
    results_dir = Path("analysis-output")
    files = sorted(results_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No results files matching {pattern}")
    return str(files[-1])


# =====================================================================
# Figure 1: Bar chart — Total connected power by scenario and method
# =====================================================================
def fig_barras_por_cenario(results: dict, output_dir: Path):
    """Bar chart comparing methods across scenarios (first seed only)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    scenarios = []
    fifo_vals = []
    gl_vals = []
    gs_vals = []
    ma_vals = []
    ma_errs = []

    for sname in ["I1", "I2", "I3", "I4"]:
        if sname not in results["scenarios"]:
            continue
        inst = results["scenarios"][sname]["instances"][0]
        scenarios.append(f"{sname}\n({inst['penetration_pct']:.0f}%)")
        fifo_vals.append(inst["baselines"]["FIFO"]["fitness"])
        gl_vals.append(inst["baselines"]["Greedy_Largest"]["fitness"])
        gs_vals.append(inst["baselines"]["Greedy_Smallest"]["fitness"])
        ma_vals.append(inst["ma_summary"]["fitness_mean"])
        ma_errs.append(inst["ma_summary"]["fitness_std"])

    x = np.arange(len(scenarios))
    width = 0.2

    ax.bar(x - 1.5*width, fifo_vals, width, label='FIFO', color=COLORS['FIFO'], edgecolor='black', linewidth=0.5)
    ax.bar(x - 0.5*width, gl_vals, width, label='Greedy-Largest', color=COLORS['Greedy_Largest'], edgecolor='black', linewidth=0.5)
    ax.bar(x + 0.5*width, gs_vals, width, label='Greedy-Smallest', color=COLORS['Greedy_Smallest'], edgecolor='black', linewidth=0.5)
    ax.bar(x + 1.5*width, ma_vals, width, label='MA (proposed)', color=COLORS['MA'], edgecolor='black', linewidth=0.5,
           yerr=ma_errs, capsize=3)

    ax.set_xlabel('Scenario (penetration level)')
    ax.set_ylabel('Total connected power (kW)')
    ax.set_title('Total Connected Power by Scenario and Method')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend(loc='upper left')
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    fig.savefig(output_dir / "fig_barras_cenarios.png")
    fig.savefig(output_dir / "fig_barras_cenarios.pdf")
    plt.close(fig)
    logger.info("Generated: fig_barras_cenarios")


# =====================================================================
# Figure 2: Bar chart — Multi-seed comparison for I3 and I4
# =====================================================================
def fig_barras_multiseed(results: dict, output_dir: Path):
    """Bar chart comparing methods across instance seeds for I3 and I4."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for idx, sname in enumerate(["I3", "I4"]):
        ax = axes[idx]
        if sname not in results["scenarios"]:
            continue

        instances = results["scenarios"][sname]["instances"]
        seeds = [f"s{inst['instance_seed']}" for inst in instances]
        n_seeds = len(seeds)

        fifo = [inst["baselines"]["FIFO"]["fitness"] for inst in instances]
        gs = [inst["baselines"]["Greedy_Smallest"]["fitness"] for inst in instances]
        ma = [inst["ma_summary"]["fitness_mean"] for inst in instances]
        ma_err = [inst["ma_summary"]["fitness_std"] for inst in instances]

        x = np.arange(n_seeds)
        width = 0.25

        ax.bar(x - width, fifo, width, label='FIFO', color=COLORS['FIFO'], edgecolor='black', linewidth=0.5)
        ax.bar(x, gs, width, label='Greedy-S', color=COLORS['Greedy_Smallest'], edgecolor='black', linewidth=0.5)
        ax.bar(x + width, ma, width, label='MA', color=COLORS['MA'], edgecolor='black', linewidth=0.5,
               yerr=ma_err, capsize=3)

        ax.set_xlabel('Instance seed')
        ax.set_ylabel('Total connected power (kW)')
        ax.set_title(f'Scenario {sname} ({instances[0]["penetration_pct"]:.0f}% pen.)')
        ax.set_xticks(x)
        ax.set_xticklabels(seeds)
        ax.legend()
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    fig.tight_layout()
    fig.savefig(output_dir / "fig_barras_multiseed.png")
    fig.savefig(output_dir / "fig_barras_multiseed.pdf")
    plt.close(fig)
    logger.info("Generated: fig_barras_multiseed")


# =====================================================================
# Figure 3: Boxplot — MA fitness distribution per scenario
# =====================================================================
def fig_boxplot_ma(results: dict, output_dir: Path):
    """Boxplot of MA fitness across runs for each restrictive instance."""
    fig, ax = plt.subplots(figsize=(10, 5))

    data = []
    labels = []

    for sname in ["I3", "I4"]:
        if sname not in results["scenarios"]:
            continue
        for inst in results["scenarios"][sname]["instances"]:
            seed = inst["instance_seed"]
            fitnesses = [r["fitness"] for r in inst["ma_runs"]]
            data.append(fitnesses)
            labels.append(f"{sname}-s{seed}")

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)

    colors_bp = ['#aec7e8', '#aec7e8', '#aec7e8', '#c5b0d5', '#c5b0d5', '#c5b0d5']
    for patch, color in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')

    # Add FIFO markers
    fifo_vals = []
    for sname in ["I3", "I4"]:
        if sname not in results["scenarios"]:
            continue
        for inst in results["scenarios"][sname]["instances"]:
            fifo_vals.append(inst["baselines"]["FIFO"]["fitness"])

    for i, fv in enumerate(fifo_vals):
        ax.plot(i + 1, fv, 'rv', markersize=10, label='FIFO' if i == 0 else None)

    # Add Greedy-S markers
    gs_vals = []
    for sname in ["I3", "I4"]:
        if sname not in results["scenarios"]:
            continue
        for inst in results["scenarios"][sname]["instances"]:
            gs_vals.append(inst["baselines"]["Greedy_Smallest"]["fitness"])

    for i, gv in enumerate(gs_vals):
        ax.plot(i + 1, gv, 'g^', markersize=10, label='Greedy-S' if i == 0 else None)

    ax.set_ylabel('Total connected power (kW)')
    ax.set_title('MA Fitness Distribution (10 runs per instance)')
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    fig.savefig(output_dir / "fig_boxplot_ma.png")
    fig.savefig(output_dir / "fig_boxplot_ma.pdf")
    plt.close(fig)
    logger.info("Generated: fig_boxplot_ma")


# =====================================================================
# Figure 4: Gain over FIFO (%) per instance
# =====================================================================
def fig_ganho_fifo(results: dict, output_dir: Path):
    """Horizontal bar chart showing % gain over FIFO."""
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = []
    gains_ma = []
    gains_gs = []

    for sname in ["I3", "I4"]:
        if sname not in results["scenarios"]:
            continue
        for inst in results["scenarios"][sname]["instances"]:
            seed = inst["instance_seed"]
            labels.append(f"{sname}-s{seed}")
            gains_ma.append(inst["gains"]["vs_fifo_pct"])

            fifo_f = inst["baselines"]["FIFO"]["fitness"]
            gs_f = inst["baselines"]["Greedy_Smallest"]["fitness"]
            gains_gs.append((gs_f - fifo_f) / fifo_f * 100 if fifo_f > 0 else 0)

    y = np.arange(len(labels))
    height = 0.35

    ax.barh(y - height/2, gains_ma, height, label='MA vs FIFO', color=COLORS['MA'], edgecolor='black', linewidth=0.5)
    ax.barh(y + height/2, gains_gs, height, label='Greedy-S vs FIFO', color=COLORS['Greedy_Smallest'], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Gain over FIFO (%)')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title('Improvement over Chronological Ordering')
    ax.legend()
    ax.axvline(x=0, color='black', linewidth=0.5)

    fig.savefig(output_dir / "fig_ganho_fifo.png")
    fig.savefig(output_dir / "fig_ganho_fifo.pdf")
    plt.close(fig)
    logger.info("Generated: fig_ganho_fifo")


# =====================================================================
# Figure 5: Convergence curves
# =====================================================================
def fig_convergencia(results: dict, output_dir: Path):
    """Convergence curves (best fitness over generations) for I3 and I4."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, sname in enumerate(["I3", "I4"]):
        ax = axes[idx]
        if sname not in results["scenarios"]:
            continue

        inst = results["scenarios"][sname]["instances"][0]  # first seed
        runs = inst["ma_runs"]

        # Collect all fitness histories
        histories = [r["fitness_history"] for r in runs]
        max_len = max(len(h) for h in histories)

        # Pad shorter histories
        padded = []
        for h in histories:
            padded.append(h + [h[-1]] * (max_len - len(h)))

        arr = np.array(padded)
        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)

        gens = np.arange(max_len)
        ax.plot(gens, mean_curve, color=COLORS['MA'], linewidth=2, label='MA mean')
        ax.fill_between(gens, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.2, color=COLORS['MA'])

        # FIFO line
        fifo_f = inst["baselines"]["FIFO"]["fitness"]
        ax.axhline(y=fifo_f, color=COLORS['FIFO'], linestyle='--', linewidth=1.5, label='FIFO')

        # Greedy-S line
        gs_f = inst["baselines"]["Greedy_Smallest"]["fitness"]
        ax.axhline(y=gs_f, color=COLORS['Greedy_Smallest'], linestyle='--', linewidth=1.5, label='Greedy-S')

        ax.set_xlabel('Generation')
        ax.set_ylabel('Best fitness (kW)')
        ax.set_title(f'{sname} (seed={inst["instance_seed"]}, {inst["penetration_pct"]:.0f}%)')
        ax.legend()
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    fig.tight_layout()
    fig.savefig(output_dir / "fig_convergencia.png")
    fig.savefig(output_dir / "fig_convergencia.pdf")
    plt.close(fig)
    logger.info("Generated: fig_convergencia")


# =====================================================================
# Figure 6: Connected projects count
# =====================================================================
def fig_n_conectados(results: dict, output_dir: Path):
    """Bar chart of number of connected projects."""
    fig, ax = plt.subplots(figsize=(10, 5))

    scenarios = []
    fifo_n = []
    gs_n = []
    ma_n = []
    ma_n_err = []

    for sname in ["I1", "I2", "I3", "I4"]:
        if sname not in results["scenarios"]:
            continue
        inst = results["scenarios"][sname]["instances"][0]
        scenarios.append(f"{sname}\n({inst['penetration_pct']:.0f}%)")
        fifo_n.append(inst["baselines"]["FIFO"]["n_connected"])
        gs_n.append(inst["baselines"]["Greedy_Smallest"]["n_connected"])
        ma_n.append(inst["ma_summary"]["n_connected_mean"])
        ma_n_err.append(inst["ma_summary"]["n_connected_std"])

    x = np.arange(len(scenarios))
    width = 0.25

    ax.bar(x - width, fifo_n, width, label='FIFO', color=COLORS['FIFO'], edgecolor='black', linewidth=0.5)
    ax.bar(x, gs_n, width, label='Greedy-S', color=COLORS['Greedy_Smallest'], edgecolor='black', linewidth=0.5)
    ax.bar(x + width, ma_n, width, label='MA', color=COLORS['MA'], edgecolor='black', linewidth=0.5,
           yerr=ma_n_err, capsize=3)

    ax.axhline(y=50, color='black', linestyle=':', linewidth=1, alpha=0.5, label='N=50 (total)')
    ax.set_xlabel('Scenario (penetration level)')
    ax.set_ylabel('Number of connected projects')
    ax.set_title('Connected Projects by Scenario')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylim(0, 55)
    ax.legend()

    fig.savefig(output_dir / "fig_n_conectados.png")
    fig.savefig(output_dir / "fig_n_conectados.pdf")
    plt.close(fig)
    logger.info("Generated: fig_n_conectados")


# =====================================================================
# Figure 7: Computational time per method
# =====================================================================
def fig_tempo_computacional(results: dict, output_dir: Path):
    """Bar chart of computational time per method."""
    fig, ax = plt.subplots(figsize=(8, 4))

    # Collect MA times from I3 and I4
    ma_times = []
    for sname in ["I3", "I4"]:
        if sname not in results["scenarios"]:
            continue
        for inst in results["scenarios"][sname]["instances"]:
            for run in inst["ma_runs"]:
                ma_times.append(run["time_s"])

    methods = ['FIFO', 'Greedy', 'Random\n(100 perm.)', 'MA\n(mean)']
    times = [0.7, 0.7, 70, np.mean(ma_times)]
    colors = [COLORS['FIFO'], COLORS['Greedy_Smallest'], COLORS['Random'], COLORS['MA']]

    bars = ax.bar(methods, times, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{t:.0f}s', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Time per evaluation (seconds)')
    ax.set_title('Computational Cost per Method')
    ax.set_yscale('log')
    ax.set_ylim(0.1, 1000)

    fig.savefig(output_dir / "fig_tempo.png")
    fig.savefig(output_dir / "fig_tempo.pdf")
    plt.close(fig)
    logger.info("Generated: fig_tempo")


# =====================================================================
# Main
# =====================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filepath = find_latest_results()
    logger.info(f"Loading results from: {filepath}")
    results = load_results(filepath)

    fig_barras_por_cenario(results, OUTPUT_DIR)
    fig_barras_multiseed(results, OUTPUT_DIR)
    fig_boxplot_ma(results, OUTPUT_DIR)
    fig_ganho_fifo(results, OUTPUT_DIR)
    fig_convergencia(results, OUTPUT_DIR)
    fig_n_conectados(results, OUTPUT_DIR)
    fig_tempo_computacional(results, OUTPUT_DIR)

    logger.info(f"\nAll figures saved to: {OUTPUT_DIR}")
    logger.info(f"Total: {len(list(OUTPUT_DIR.glob('*.png')))} PNG + {len(list(OUTPUT_DIR.glob('*.pdf')))} PDF")


if __name__ == "__main__":
    main()
