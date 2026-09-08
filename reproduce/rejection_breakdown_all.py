# -*- coding: utf-8 -*-
"""Reproduces the binding-constraint appendix table on ALL restrictive instances.

Replays the FIFO sequential evaluation with per-rejection constraint labeling
on the 6 restrictive instances of each network (12 replays total), extending
analysis-output/rejection_breakdown.py (which covers the 33-bus set plus one
69-bus instance) to the full 69-bus set. Confirms that 100% of rejections are
caused by overvoltage and reports the maximum loss level among accepted states
(manuscript: 8.6%, 69-bus I4-s13).

Cost: ~1 min. Usage: python rejection_breakdown_all.py
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "analysis-output"))

from rejection_breakdown import report_instance  # noqa: E402
from execution.scenarios import create_scenario, create_scenario_69bus  # noqa: E402

for scen in ("I3", "I4"):
    for seed in (42, 7, 13):
        net, projects, cfg = create_scenario(scen, seed=seed)
        report_instance(f"33-bus {scen} s{seed}", net, projects, cfg.constraints)
for scen in ("I3", "I4"):
    for seed in (42, 7, 13):
        net, projects, cfg = create_scenario_69bus(scen, seed=seed)
        report_instance(f"69-bus {scen} s{seed}", net, projects, cfg.constraints)
