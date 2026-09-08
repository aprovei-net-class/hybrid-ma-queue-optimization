# -*- coding: utf-8 -*-
"""
generate_figures_ieee.py — IEEE-format figure generation for main experiment results.

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
import matplotlib.ticker as ticker
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# IEEE Style Configuration
# =====================================================================
COL_W = 3.5    # inches — single column
FULL_W = 7.16  # inches — full page width
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
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 1.0,
    'lines.markersize': 5,
})

# Grayscale-friendly colors (also work in color)
C_FIFO = '#c0392b'       # red
C_GREEDY_L = '#e67e22'   # orange
C_GREEDY_S = '#27ae60'   # green
C_RANDOM = '#7f8c8d'     # gray
C_MA = '#2c3e50'         # dark blue/black
C_MA_LIGHT = '#85c1e9'   # light blue for fill

# Hatching patterns for grayscale differentiation
H_FIFO = '//'
H_GREEDY_L = '\\\\'
H_GREEDY_S = 'xx'
H_MA = ''  # solid (stands out)

OUTPUT_DIR = Path("analysis-output/figures-ieee")


def load_results(pattern="experiment_optionA_*.json"):
    results_dir = Path("analysis-output")
    files = sorted(results_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No results files matching {pattern}")
    with open(files[-1], "r") as f:
        return json.load(f)


def fmt_kw(x, pos):
    """Format kW values for axis."""
    if x >= 1000:
        return f'{x/1000:.1f}k'
    return f'{x:.0f}'


# =====================================================================
# Fig 1: Connected power by scenario (single column)
# =====================================================================
def fig1_barras_cenarios(results, out):
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))

    scenarios = []
    fifo, gl, gs, ma, ma_err = [], [], [], [], []

    for sname in ["I1", "I2", "I3", "I4"]:
        if sname not in results["scenarios"]:
            continue
        inst = results["scenarios"][sname]["instances"][0]
        pen = inst['penetration_pct']
        scenarios.append(f"{sname} ({pen:.0f}%)")
        fifo.append(inst["baselines"]["FIFO"]["fitness"] / 1000)
        gl.append(inst["baselines"]["Greedy_Largest"]["fitness"] / 1000)
        gs.append(inst["baselines"]["Greedy_Smallest"]["fitness"] / 1000)
        ma.append(inst["ma_summary"]["fitness_mean"] / 1000)
        ma_err.append(inst["ma_summary"]["fitness_std"] / 1000)

    x = np.arange(len(scenarios))
    w = 0.19

    ax.bar(x - 1.5*w, fifo, w, label='FIFO', color=C_FIFO, hatch=H_FIFO, edgecolor='black', linewidth=0.4)
    ax.bar(x - 0.5*w, gl, w, label='Greedy-L', color=C_GREEDY_L, hatch=H_GREEDY_L, edgecolor='black', linewidth=0.4)
    ax.bar(x + 0.5*w, gs, w, label='Greedy-S', color=C_GREEDY_S, hatch=H_GREEDY_S, edgecolor='black', linewidth=0.4)
    ax.bar(x + 1.5*w, ma, w, label='MA (proposed)', color=C_MA, hatch=H_MA, edgecolor='black', linewidth=0.4,
           yerr=ma_err, capsize=2, error_kw={'linewidth': 0.6})

    ax.set_ylabel('Total connected power (MW)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend(loc='upper left', ncol=2)
    ax.set_ylim(0, max(ma) * 1.25)

    fig.savefig(out / "fig1_barras_cenarios.pdf")
    fig.savefig(out / "fig1_barras_cenarios.png")
    plt.close(fig)
    logger.info("Fig 1: barras_cenarios")


# =====================================================================
# Fig 2: Multi-seed comparison I3 and I4 (full width)
# =====================================================================
def fig2_multiseed(results, out):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.2))

    for idx, sname in enumerate(["I3", "I4"]):
        ax = axes[idx]
        if sname not in results["scenarios"]:
            continue

        instances = results["scenarios"][sname]["instances"]
        seeds = [f"s{inst['instance_seed']}" for inst in instances]
        pen = instances[0]['penetration_pct']

        fifo = [inst["baselines"]["FIFO"]["fitness"]/1000 for inst in instances]
        gs = [inst["baselines"]["Greedy_Smallest"]["fitness"]/1000 for inst in instances]
        ma_v = [inst["ma_summary"]["fitness_mean"]/1000 for inst in instances]
        ma_e = [inst["ma_summary"]["fitness_std"]/1000 for inst in instances]

        x = np.arange(len(seeds))
        w = 0.22

        ax.bar(x - w, fifo, w, label='FIFO', color=C_FIFO, hatch=H_FIFO, edgecolor='black', linewidth=0.4)
        ax.bar(x, gs, w, label='Greedy-S', color=C_GREEDY_S, hatch=H_GREEDY_S, edgecolor='black', linewidth=0.4)
        ax.bar(x + w, ma_v, w, label='MA', color=C_MA, edgecolor='black', linewidth=0.4,
               yerr=ma_e, capsize=2, error_kw={'linewidth': 0.6})

        ax.set_ylabel('Connected power (MW)')
        ax.set_xlabel('Instance seed')
        ax.set_xticks(x)
        ax.set_xticklabels(seeds)
        # Title as text annotation (IEEE style: label inside plot)
        ax.text(0.02, 0.96, f'({chr(97+idx)}) {sname} (~{pen:.0f}%)',
                transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')
        if idx == 0:
            ax.legend(loc='upper right')

    fig.tight_layout(w_pad=1.5)
    fig.savefig(out / "fig2_multiseed.pdf")
    fig.savefig(out / "fig2_multiseed.png")
    plt.close(fig)
    logger.info("Fig 2: multiseed")


# =====================================================================
# Fig 3: Boxplot MA fitness (single column)
# =====================================================================
def fig3_boxplot(results, out):
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))

    data, labels, fifo_v, gs_v = [], [], [], []

    for sname in ["I3", "I4"]:
        if sname not in results["scenarios"]:
            continue
        for inst in results["scenarios"][sname]["instances"]:
            seed = inst["instance_seed"]
            fitnesses = [r["fitness"]/1000 for r in inst["ma_runs"]]
            data.append(fitnesses)
            labels.append(f"{sname}\ns{seed}")
            fifo_v.append(inst["baselines"]["FIFO"]["fitness"]/1000)
            gs_v.append(inst["baselines"]["Greedy_Smallest"]["fitness"]/1000)

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5,
                    boxprops=dict(linewidth=0.6),
                    whiskerprops=dict(linewidth=0.5),
                    medianprops=dict(linewidth=1, color='black'),
                    flierprops=dict(markersize=3))

    for patch in bp['boxes']:
        patch.set_facecolor(C_MA_LIGHT)
        patch.set_edgecolor('black')

    for i, (fv, gv) in enumerate(zip(fifo_v, gs_v)):
        ax.plot(i + 1, fv, 'v', color=C_FIFO, markersize=6,
                label='FIFO' if i == 0 else None, zorder=5)
        ax.plot(i + 1, gv, '^', color=C_GREEDY_S, markersize=6,
                label='Greedy-S' if i == 0 else None, zorder=5)

    ax.set_ylabel('Total connected power (MW)')
    ax.legend(loc='lower left')

    fig.savefig(out / "fig3_boxplot.pdf")
    fig.savefig(out / "fig3_boxplot.png")
    plt.close(fig)
    logger.info("Fig 3: boxplot")


# =====================================================================
# Fig 4: Gain over FIFO (single column)
# =====================================================================
def fig4_ganho(results, out):
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))

    labels, g_ma, g_gs = [], [], []

    for sname in ["I3", "I4"]:
        if sname not in results["scenarios"]:
            continue
        for inst in results["scenarios"][sname]["instances"]:
            seed = inst["instance_seed"]
            labels.append(f"{sname}-s{seed}")
            g_ma.append(inst["gains"]["vs_fifo_pct"])
            fifo_f = inst["baselines"]["FIFO"]["fitness"]
            gs_f = inst["baselines"]["Greedy_Smallest"]["fitness"]
            g_gs.append((gs_f - fifo_f) / fifo_f * 100 if fifo_f > 0 else 0)

    y = np.arange(len(labels))
    h = 0.3

    ax.barh(y - h/2, g_ma, h, label='MA', color=C_MA, edgecolor='black', linewidth=0.4)
    ax.barh(y + h/2, g_gs, h, label='Greedy-S', color=C_GREEDY_S, hatch=H_GREEDY_S,
            edgecolor='black', linewidth=0.4)

    ax.set_xlabel('Gain over FIFO (%)')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.legend(loc='lower right')

    fig.savefig(out / "fig4_ganho.pdf")
    fig.savefig(out / "fig4_ganho.png")
    plt.close(fig)
    logger.info("Fig 4: ganho")


# =====================================================================
# Fig 5: Convergence curves (full width)
# =====================================================================
def fig5_convergencia(results, out):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.2))

    for idx, sname in enumerate(["I3", "I4"]):
        ax = axes[idx]
        if sname not in results["scenarios"]:
            continue

        inst = results["scenarios"][sname]["instances"][0]
        runs = inst["ma_runs"]
        pen = inst['penetration_pct']

        histories = [r["fitness_history"] for r in runs]
        max_len = max(len(h) for h in histories)
        padded = [h + [h[-1]] * (max_len - len(h)) for h in histories]

        arr = np.array(padded) / 1000  # Convert to MW
        mean_c = np.mean(arr, axis=0)
        std_c = np.std(arr, axis=0)
        gens = np.arange(max_len)

        ax.plot(gens, mean_c, color=C_MA, linewidth=1.2, label='MA (mean)')
        ax.fill_between(gens, mean_c - std_c, mean_c + std_c,
                        alpha=0.2, color=C_MA_LIGHT, linewidth=0)

        fifo_f = inst["baselines"]["FIFO"]["fitness"] / 1000
        gs_f = inst["baselines"]["Greedy_Smallest"]["fitness"] / 1000
        ax.axhline(y=fifo_f, color=C_FIFO, linestyle='--', linewidth=0.8, label='FIFO')
        ax.axhline(y=gs_f, color=C_GREEDY_S, linestyle='-.', linewidth=0.8, label='Greedy-S')

        ax.set_xlabel('Generation')
        ax.set_ylabel('Best fitness (MW)')
        ax.text(0.02, 0.96, f'({chr(97+idx)}) {sname} (~{pen:.0f}%)',
                transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')
        if idx == 0:
            ax.legend(loc='lower right')

    fig.tight_layout(w_pad=1.5)
    fig.savefig(out / "fig5_convergencia.pdf")
    fig.savefig(out / "fig5_convergencia.png")
    plt.close(fig)
    logger.info("Fig 5: convergencia")


# =====================================================================
# Fig 6: Connected projects (single column)
# =====================================================================
def fig6_n_conectados(results, out):
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))

    scenarios, fifo_n, gs_n, ma_n, ma_e = [], [], [], [], []

    for sname in ["I1", "I2", "I3", "I4"]:
        if sname not in results["scenarios"]:
            continue
        inst = results["scenarios"][sname]["instances"][0]
        pen = inst['penetration_pct']
        scenarios.append(f"{sname}\n({pen:.0f}%)")
        fifo_n.append(inst["baselines"]["FIFO"]["n_connected"])
        gs_n.append(inst["baselines"]["Greedy_Smallest"]["n_connected"])
        ma_n.append(inst["ma_summary"]["n_connected_mean"])
        ma_e.append(inst["ma_summary"]["n_connected_std"])

    x = np.arange(len(scenarios))
    w = 0.22

    ax.bar(x - w, fifo_n, w, label='FIFO', color=C_FIFO, hatch=H_FIFO, edgecolor='black', linewidth=0.4)
    ax.bar(x, gs_n, w, label='Greedy-S', color=C_GREEDY_S, hatch=H_GREEDY_S, edgecolor='black', linewidth=0.4)
    ax.bar(x + w, ma_n, w, label='MA', color=C_MA, edgecolor='black', linewidth=0.4,
           yerr=ma_e, capsize=2, error_kw={'linewidth': 0.6})

    ax.axhline(y=50, color='black', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.set_ylabel('Connected projects (of 50)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylim(0, 55)
    ax.legend(loc='lower left')

    fig.savefig(out / "fig6_n_conectados.pdf")
    fig.savefig(out / "fig6_n_conectados.png")
    plt.close(fig)
    logger.info("Fig 6: n_conectados")


# =====================================================================
# Fig 7: Computational time (single column)
# =====================================================================
def fig7_tempo(results, out):
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))

    ma_times = []
    for sname in ["I3", "I4"]:
        if sname not in results["scenarios"]:
            continue
        for inst in results["scenarios"][sname]["instances"]:
            for run in inst["ma_runs"]:
                ma_times.append(run["time_s"])

    methods = ['FIFO', 'Greedy', 'Random\n(100)', 'MA']
    times = [0.7, 0.7, 70, np.mean(ma_times)]
    colors = [C_FIFO, C_GREEDY_S, C_RANDOM, C_MA]

    bars = ax.bar(methods, times, color=colors, edgecolor='black', linewidth=0.4, width=0.6)

    for bar, t in zip(bars, times):
        label = f'{t:.0f} s' if t >= 1 else f'{t:.1f} s'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                label, ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('Time (s)')
    ax.set_yscale('log')
    ax.set_ylim(0.3, 800)

    fig.savefig(out / "fig7_tempo.pdf")
    fig.savefig(out / "fig7_tempo.png")
    plt.close(fig)
    logger.info("Fig 7: tempo")


# =====================================================================
# Main
# =====================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results()
    logger.info(f"Loaded results, generating IEEE-formatted figures...\n")

    fig1_barras_cenarios(results, OUTPUT_DIR)
    fig2_multiseed(results, OUTPUT_DIR)
    fig3_boxplot(results, OUTPUT_DIR)
    fig4_ganho(results, OUTPUT_DIR)
    fig5_convergencia(results, OUTPUT_DIR)
    fig6_n_conectados(results, OUTPUT_DIR)
    fig7_tempo(results, OUTPUT_DIR)

    n_png = len(list(OUTPUT_DIR.glob('*.png')))
    n_pdf = len(list(OUTPUT_DIR.glob('*.pdf')))
    logger.info(f"\nAll figures saved to: {OUTPUT_DIR}")
    logger.info(f"Total: {n_png} PNG + {n_pdf} PDF")


if __name__ == "__main__":
    main()
