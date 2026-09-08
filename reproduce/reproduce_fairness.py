# -*- coding: utf-8 -*-
"""Reproduces the fairness table (D_max impact, Bonferroni-corrected tests).

Reads the raw fairness JSON and recomputes, per row (network x scenario):
mean fitness under D_inf / D_25 / D_12, the relative costs, the per-seed
paired Wilcoxon tests D_inf vs D_12 (Bonferroni alpha = 0.05/9), the "(x/3)"
significance counts, and the Cohen's d ranges. Prints the published values
alongside for comparison.

Cost: seconds. Usage: python reproduce_fairness.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

AO = Path(__file__).resolve().parents[1] / "analysis-output"
with open(AO / "fairness_multiseed_20260410_054544.json", encoding="utf-8") as f:
    fm = json.load(f)["fairness_multiseed"]

rows = [
    ("33-bus", "I3", ["33bus_I3_s42", "33bus_I3_s7", "33bus_I3_s13"]),
    ("33-bus", "I4", ["33bus_I4_s42", "33bus_I4_s7", "33bus_I4_s13"]),
    ("69-bus", "I4", ["69bus_I4_s42", "69bus_I4_s7", "69bus_I4_s13"]),
]

manuscript = {
    ("33-bus", "I3"): dict(dinf=7941, d25=7876, d12=7733, c25=-0.8, c12=-2.6, sig="3/3", dr=(1.4, 1.9)),
    ("33-bus", "I4"): dict(dinf=10760, d25=10654, d12=10383, c25=-1.0, c12=-3.5, sig="2/3", dr=(0.2, 2.7)),
    ("69-bus", "I4"): dict(dinf=15098, d25=14873, d12=14679, c25=-1.5, c12=-2.8, sig="2/3", dr=(0.7, 2.0)),
}

ALPHA_BONF = 0.05 / 9

for net, scen, keys in rows:
    means = {"inf": [], "25": [], "12": []}
    n_sig, ds = 0, []
    for key in keys:
        vinf, v25, v12 = fm[f"{key}_dinf"], fm[f"{key}_d25"], fm[f"{key}_d12"]
        paired = ([r["seed"] for r in vinf["runs"]]
                  == [r["seed"] for r in v12["runs"]])
        fi = np.array([r["fitness"] for r in vinf["runs"]])
        f25 = np.array([r["fitness"] for r in v25["runs"]])
        f12 = np.array([r["fitness"] for r in v12["runs"]])
        means["inf"].append(fi.mean())
        means["25"].append(f25.mean())
        means["12"].append(f12.mean())
        _, p = stats.wilcoxon(fi, f12, alternative="two-sided")
        d = (fi - f12).mean() / (fi - f12).std(ddof=1)
        ds.append(abs(d))
        if p < ALPHA_BONF:
            n_sig += 1
        print(f"{net} {scen} seed={vinf['seed']:>3} paired={paired} "
              f"p={p:.4f} d={d:+.2f} sig={p < ALPHA_BONF}")
    mi, m25, m12 = (np.mean(means[k]) for k in ("inf", "25", "12"))
    m = manuscript[(net, scen)]
    print(f"-- ROW {net} {scen}: Dinf={mi:,.0f} D25={m25:,.0f} D12={m12:,.0f} "
          f"cost25={100*(m25-mi)/mi:+.1f}% cost12={100*(m12-mi)/mi:+.1f}% "
          f"sig={n_sig}/3 d_range=({min(ds):.1f},{max(ds):.1f})")
    print(f"   published: Dinf={m['dinf']} D25={m['d25']} D12={m['d12']} "
          f"c25={m['c25']}% c12={m['c12']}% sig={m['sig']} d={m['dr']}\n")
