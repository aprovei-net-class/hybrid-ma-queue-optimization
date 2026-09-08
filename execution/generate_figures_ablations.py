# -*- coding: utf-8 -*-
"""
generate_figures_ablations.py — IEEE-format figure generation for ablation studies.

Part of: Hybrid Memetic Algorithm for DG Queue Optimization
Paper: "Connection Order Matters: Path-Dependent Hosting Capacity in
        Distributed Generation Interconnection Queues"
Authors: Williams F. Fontinele, Pablo T. Caballero, Eduardo C. M. da Costa
License: MIT
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

COL_W = 3.5
FULL_W = 7.16
FONT_SIZE = 8

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': FONT_SIZE,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.4,
    'axes.linewidth': 0.6,
})

OUTPUT_DIR = Path("analysis-output/figures-ieee")


def load_ablation_results():
    results_dir = Path("analysis-output")
    files = sorted(results_dir.glob("ablations_*.json"))
    if not files:
        raise FileNotFoundError("No ablation results found")
    with open(files[-1], "r") as f:
        return json.load(f)


# =====================================================================
# Fig 8: Ablation — MA full vs GA-only (Section 5.5)
# =====================================================================
def fig8_ablation_ls(results, out):
    """Compare MA full (with local search) vs GA-only (without)."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.2))

    for idx, sname in enumerate(["I3", "I4"]):
        ax = axes[idx]
        sdata = results["scenarios"][sname]
        fifo = sdata["baselines"]["FIFO"]["fitness"] / 1000
        gs = sdata["baselines"]["Greedy_Smallest"]["fitness"] / 1000

        ma_full = sdata["ablations"]["MA_full"]["summary"]
        ga_only = sdata["ablations"]["GA_only"]["summary"]

        methods = ['FIFO', 'Greedy-S', 'GA-only', 'MA full']
        values = [fifo, gs,
                  ga_only["fitness_mean"]/1000,
                  ma_full["fitness_mean"]/1000]
        errors = [0, 0,
                  ga_only["fitness_std"]/1000,
                  ma_full["fitness_std"]/1000]
        colors = ['#c0392b', '#27ae60', '#e377c2', '#2c3e50']
        hatches = ['//', 'xx', '..', '']

        bars = ax.bar(methods, values, color=colors, edgecolor='black',
                      linewidth=0.4, width=0.6, yerr=errors, capsize=3,
                      error_kw={'linewidth': 0.6})
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)

        ax.set_ylabel('Connected power (MW)')
        ax.text(0.02, 0.96, f'({chr(97+idx)}) {sname}',
                transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')

    fig.tight_layout(w_pad=1.5)
    fig.savefig(out / "fig8_ablation_ls.pdf")
    fig.savefig(out / "fig8_ablation_ls.png")
    plt.close(fig)
    logger.info("Fig 8: ablation local search (MA vs GA-only)")


# =====================================================================
# Fig 9: Fairness — D_max impact (Section 5.8)
# =====================================================================
def fig9_fairness(results, out):
    """Compare MA with different D_max values."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4))

    for idx, sname in enumerate(["I3", "I4"]):
        ax = axes[idx]
        sdata = results["scenarios"][sname]
        fifo = sdata["baselines"]["FIFO"]["fitness"] / 1000

        ma_full = sdata["ablations"]["MA_full"]["summary"]

        dmax_keys = [k for k in sdata["ablations"].keys() if "Dmax" in k]
        dmax_keys.sort(key=lambda k: -int(k.split("Dmax")[1]))  # descending

        labels = ['No limit']
        values = [ma_full["fitness_mean"]/1000]
        errors = [ma_full["fitness_std"]/1000]

        for dk in dmax_keys:
            d_val = dk.split("Dmax")[1]
            labels.append(f'D={d_val}')
            s = sdata["ablations"][dk]["summary"]
            values.append(s["fitness_mean"]/1000)
            errors.append(s["fitness_std"]/1000)

        colors = ['#2c3e50', '#3498db', '#e74c3c']
        x = np.arange(len(labels))

        bars = ax.bar(x, values, color=colors, edgecolor='black',
                      linewidth=0.4, width=0.5, yerr=errors, capsize=3,
                      error_kw={'linewidth': 0.6})

        ax.axhline(y=fifo, color='#c0392b', linestyle='--', linewidth=0.8, label='FIFO')

        ax.set_ylabel('Connected power (MW)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel('Fairness constraint (D_max)')
        ax.text(0.02, 0.96, f'({chr(97+idx)}) {sname}',
                transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')
        if idx == 0:
            ax.legend(loc='lower left')

        # Add percentage labels on bars
        for bar, v in zip(bars, values):
            gain = (v - fifo) / fifo * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errors[bars.index(bar)] + 0.05,
                    f'+{gain:.0f}%', ha='center', va='bottom', fontsize=6.5)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(out / "fig9_fairness.pdf")
    fig.savefig(out / "fig9_fairness.png")
    plt.close(fig)
    logger.info("Fig 9: fairness D_max impact")


# =====================================================================
# Fig 10: Combined ablation summary (single column)
# =====================================================================
def fig10_ablation_summary(results, out):
    """Grouped bar chart: all ablation variants for I4."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.8))

    sdata = results["scenarios"]["I4"]
    fifo = sdata["baselines"]["FIFO"]["fitness"] / 1000
    gs = sdata["baselines"]["Greedy_Smallest"]["fitness"] / 1000

    methods = ['FIFO', 'Greedy-S', 'GA-only', 'MA\nD=12', 'MA\nD=25', 'MA\n(full)']
    ablations = sdata["ablations"]

    values = [
        fifo, gs,
        ablations["GA_only"]["summary"]["fitness_mean"]/1000,
        ablations["MA_Dmax12"]["summary"]["fitness_mean"]/1000,
        ablations["MA_Dmax25"]["summary"]["fitness_mean"]/1000,
        ablations["MA_full"]["summary"]["fitness_mean"]/1000,
    ]
    errors = [
        0, 0,
        ablations["GA_only"]["summary"]["fitness_std"]/1000,
        ablations["MA_Dmax12"]["summary"]["fitness_std"]/1000,
        ablations["MA_Dmax25"]["summary"]["fitness_std"]/1000,
        ablations["MA_full"]["summary"]["fitness_std"]/1000,
    ]
    colors = ['#c0392b', '#27ae60', '#e377c2', '#e74c3c', '#3498db', '#2c3e50']

    bars = ax.bar(methods, values, color=colors, edgecolor='black',
                  linewidth=0.4, width=0.6, yerr=errors, capsize=2,
                  error_kw={'linewidth': 0.6})

    # Add gain labels
    for bar, v, e in zip(bars, values, errors):
        if v != fifo:
            gain = (v - fifo) / fifo * 100
            ax.text(bar.get_x() + bar.get_width()/2, v + e + 0.1,
                    f'+{gain:.0f}%', ha='center', va='bottom', fontsize=6.5)

    ax.set_ylabel('Connected power (MW)')
    ax.axhline(y=fifo, color='#c0392b', linestyle=':', linewidth=0.5, alpha=0.5)

    fig.savefig(out / "fig10_ablation_summary.pdf")
    fig.savefig(out / "fig10_ablation_summary.png")
    plt.close(fig)
    logger.info("Fig 10: ablation summary I4")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_ablation_results()
    logger.info("Generating ablation figures (IEEE format)...\n")

    fig8_ablation_ls(results, OUTPUT_DIR)
    fig9_fairness(results, OUTPUT_DIR)
    fig10_ablation_summary(results, OUTPUT_DIR)

    logger.info(f"\nAblation figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
