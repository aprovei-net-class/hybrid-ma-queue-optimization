# -*- coding: utf-8 -*-
"""Reproduces the deterministic-ordering check on the trivial 69-bus scenarios.

Replays FIFO, Greedy-Largest and Greedy-Smallest on the 69-bus I1 and I2
instances (seed 42) and verifies the manuscript statement (Statistical
Protocol section): with no active constraint, the three orderings connect all
50 projects with identical total power. Compares against the reference output
`analysis-output/trivial_69bus_deterministic.json`.

Cost: ~1 min (6 sequential evaluations). Usage: python trivial_69bus.py
"""
import json
from pathlib import Path

from _comum import avaliar, instancia_original

REF = (Path(__file__).resolve().parents[1] / "analysis-output"
       / "trivial_69bus_deterministic.json")
TOL_KW = 1e-3

with open(REF, encoding="utf-8") as f:
    ref = json.load(f)["resultados"]

todos_ok = True
for cen in ("I1", "I2"):
    net, projetos, cfg = instancia_original("69bus", cen, 42)
    ids = [p.id for p in projetos]
    ordens = {
        "FIFO": ids,
        "GreedyL": sorted(ids, key=lambda i: -projetos[i].p_kw),
        "GreedyS": sorted(ids, key=lambda i: projetos[i].p_kw),
    }
    linha = {}
    for nome, ordem in ordens.items():
        r = avaliar(net, projetos, ordem, cfg.constraints)
        linha[nome] = (r.fitness, r.n_connected)
    fits = [v[0] for v in linha.values()]
    identicos = max(fits) - min(fits) < TOL_KW
    for nome, (fit, ncon) in linha.items():
        r = ref[f"69bus_{cen}_s42"][nome]
        ok = (abs(fit - r["fitness"]) < TOL_KW and ncon == r["n_connected"]
              and ncon == 50)
        todos_ok &= ok and identicos
        print(f"69bus {cen} s42 {nome:8s} fitness={fit:10.1f} kW "
              f"connected={ncon}/50  match={ok}")
    print(f"  -> identical totals across orderings (< {TOL_KW} kW): "
          f"{identicos}")
print(f"\nall rows match reference and manuscript statement: {todos_ok}")
