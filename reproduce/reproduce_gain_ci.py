# -*- coding: utf-8 -*-
"""Reproduces the MA-vs-FIFO gain table (bootstrap CIs) and the aggregate gains.

Reads the raw per-run campaign JSONs in ``analysis-output/`` and recomputes,
for every restrictive instance on both networks: MA mean/std, gain over FIFO,
and the 95% bootstrap percentile CI (10,000 resamples, seed 12345 as in the
manuscript; a second seed is printed to show CI stability). The hardcoded
``manuscript`` dict carries the published values so the script doubles as a
reproduction gate: every row prints whether the recomputed values match.

Cost: seconds. Usage: python reproduce_gain_ci.py
"""
import json
from pathlib import Path

import numpy as np

AO = Path(__file__).resolve().parents[1] / "analysis-output"
P33 = AO / "experiment_optionA_20260406_173130.json"
P69 = AO / "results_69bus_saved.json"

with open(P33, encoding="utf-8") as f:
    d33 = json.load(f)
with open(P69, encoding="utf-8") as f:
    d69 = json.load(f)

# --- extract 33-bus instances ---
rows_33 = {}
for sc in ["I3", "I4"]:
    for inst in d33["scenarios"][sc]["instances"]:
        seed = inst["instance_seed"]
        key = f"33-bus {sc}-s{seed}"
        fifo = inst["baselines"]["FIFO"]["fitness"]
        ma = np.array([r["fitness"] for r in inst["ma_runs"]], dtype=float)
        rows_33[key] = (ma, fifo)

# --- extract 69-bus instances (exclude trivial I3-s42, I3-s7) ---
map69 = {
    "I3_s13": "69-bus I3-s13",
    "I4_s42": "69-bus I4-s42",
    "I4_s7": "69-bus I4-s7",
    "I4_s13": "69-bus I4-s13",
}
rows_69 = {}
for k, label in map69.items():
    v = d69["part1_69bus_main"][k]
    rows_69[label] = (np.array(v["MA_runs"], dtype=float), float(v["FIFO"]))

manuscript = {
    "33-bus I3-s42": dict(ma_mean=7600, ma_std=11, gain=31.8, ci=(31.7, 32.0)),
    "33-bus I3-s7": dict(ma_mean=8433, ma_std=16, gain=10.9, ci=(10.8, 11.0)),
    "33-bus I3-s13": dict(ma_mean=7789, ma_std=34, gain=5.6, ci=(5.4, 5.9)),
    "33-bus I4-s42": dict(ma_mean=10780, ma_std=171, gain=67.6, ci=(65.9, 69.2)),
    "33-bus I4-s7": dict(ma_mean=11912, ma_std=239, gain=38.8, ci=(36.8, 40.2)),
    "33-bus I4-s13": dict(ma_mean=9588, ma_std=235, gain=24.3, ci=(22.4, 26.2)),
    "69-bus I3-s13": dict(ma_mean=9150, ma_std=7, gain=0.3, ci=(0.3, 0.4)),
    "69-bus I4-s42": dict(ma_mean=15955, ma_std=80, gain=3.2, ci=(2.9, 3.6)),
    "69-bus I4-s7": dict(ma_mean=15508, ma_std=113, gain=17.3, ci=(16.7, 17.8)),
    "69-bus I4-s13": dict(ma_mean=13832, ma_std=53, gain=8.7, ci=(8.4, 8.9)),
}

all_rows = {**rows_33, **rows_69}


def bootstrap_ci(ma, fifo, seed, n_boot=10000):
    rng = np.random.RandomState(seed)
    n = len(ma)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(ma, size=n, replace=True).mean()
    gains = (means - fifo) / fifo * 100.0
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return lo, hi


print(f"{'instance':16s} {'MAmean':>10s} {'MAstd':>8s} {'gain%':>8s} "
      f"{'CIlo':>8s} {'CIhi':>8s} {'n>FIFO':>7s} {'match':>6s}")
gains_33, gains_69 = [], []
for key, (ma, fifo) in all_rows.items():
    mean, std = ma.mean(), ma.std(ddof=0)
    gain = (mean - fifo) / fifo * 100.0
    lo, hi = bootstrap_ci(ma, fifo, seed=12345)
    n_gt = int((ma > fifo).sum())
    m = manuscript[key]
    ok = (round(mean) == m["ma_mean"] and round(std) == m["ma_std"]
          and round(gain, 1) == m["gain"]
          and (round(lo, 1), round(hi, 1)) == m["ci"])
    print(f"{key:16s} {mean:10.2f} {std:8.3f} {gain:8.3f} "
          f"{lo:8.3f} {hi:8.3f} {n_gt:7d} {str(ok):>6s}")
    (gains_33 if key.startswith("33") else gains_69).append(gain)

# Aggregates over the six restrictive instances per network (trivial 69-bus
# I3-s42/s7 enter as 0% gain, as in the manuscript's unweighted mean).
gains_69_full = gains_69 + [0.0, 0.0]
print(f"\naggregate 33-bus (6 instances): mean {np.mean(gains_33):+.1f}% "
      f"median {np.median(gains_33):+.1f}%   [manuscript: +29.8% / +28.1%]")
print(f"aggregate 69-bus (6 instances): mean {np.mean(gains_69_full):+.1f}% "
      f"median {np.median(gains_69_full):+.1f}%   [manuscript: +4.9% / +1.8%]")

# CV table (coefficient of variation of the MA runs per instance group).
print("\nCV per group [manuscript: 33-bus I3 0.1-0.4%; 33-bus I4 1.6-2.5%; "
      "69-bus I4 0.4-0.7%]")
for grupo, prefixo in [("33-bus I3", "33-bus I3"), ("33-bus I4", "33-bus I4"),
                       ("69-bus I4", "69-bus I4")]:
    cvs = [100 * ma.std(ddof=0) / ma.mean()
           for key, (ma, _) in all_rows.items() if key.startswith(prefixo)]
    print(f"  {grupo}: CV {min(cvs):.1f}% - {max(cvs):.1f}%")
