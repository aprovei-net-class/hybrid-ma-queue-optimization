# -*- coding: utf-8 -*-
"""Reproduces the MA-vs-GA-only ablation table (Wilcoxon p, Cohen's d).

Reads the raw ablation JSON, verifies that MA and GA-only runs are paired by
shared algorithm seed (common random numbers), and recomputes the paired
Wilcoxon signed-rank test and Cohen's d on paired differences for each of the
9 configurations. Prints the published values alongside for comparison.

Cost: seconds. Usage: python reproduce_ablation.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

AO = Path(__file__).resolve().parents[1] / "analysis-output"
with open(AO / "ablation_multiseed_20260409_054353.json", encoding="utf-8") as f:
    am = json.load(f)["ablation_multiseed"]

manuscript = {
    "33bus_I3_s42": dict(ma=7600, ga=7601, diff=-2, p=0.910, d=-0.07),
    "33bus_I3_s7":  dict(ma=8433, ga=8428, diff=5,  p=0.383, d=0.18),
    "33bus_I3_s13": dict(ma=7789, ga=7764, diff=24, p=0.557, d=0.32),
    "33bus_I4_s42": dict(ma=10780, ga=10692, diff=88, p=0.322, d=0.28),
    "33bus_I4_s7":  dict(ma=11912, ga=11778, diff=134, p=0.160, d=0.43),
    "33bus_I4_s13": dict(ma=9588, ga=9659, diff=-71, p=0.492, d=-0.32),
    "69bus_I4_s42": dict(ma=15955, ga=15978, diff=-23, p=0.432, d=-0.18),
    "69bus_I4_s7":  dict(ma=15508, ga=15536, diff=-28, p=0.432, d=-0.19),
    "69bus_I4_s13": dict(ma=13832, ga=13780, diff=52, p=0.232, d=0.44),
}

print(f"{'config':16s} {'paired':>7s} {'MA':>9s} {'GA':>9s} {'diff':>8s} "
      f"{'p':>7s} {'d':>7s}  published(p, d)")
for key, v in am.items():
    ma_runs, ga_runs = v["ma_full"]["runs"], v["ga_only"]["runs"]
    paired = [r["seed"] for r in ma_runs] == [r["seed"] for r in ga_runs]
    ma = np.array([r["fitness"] for r in ma_runs])
    ga = np.array([r["fitness"] for r in ga_runs])
    diff = ma - ga
    _, p = stats.wilcoxon(ma, ga, alternative="two-sided")
    d = diff.mean() / diff.std(ddof=1)
    m = manuscript[key]
    print(f"{key:16s} {str(paired):>7s} {ma.mean():9.1f} {ga.mean():9.1f} "
          f"{diff.mean():8.1f} {p:7.3f} {d:+7.2f}  ({m['p']}, {m['d']:+.2f})")
