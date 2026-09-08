# -*- coding: utf-8 -*-
"""Shared utilities for the reproduction scripts (revision experiments).

Imports the frozen experiment code from ``execution/`` at the repository root
(path resolved relative to this file, so any clean checkout works) and provides
instance constructors. No published result is altered by this module.
"""
import sys
from pathlib import Path

RAIZ_CODIGO = str(Path(__file__).resolve().parents[1])
if RAIZ_CODIGO not in sys.path:
    sys.path.insert(0, RAIZ_CODIGO)

from execution.evaluator import evaluate_permutation  # noqa: E402
from execution.network import TechnicalConstraints  # noqa: E402
from execution.queue_generator import generate_queue, get_penetration_pct  # noqa: E402
from execution.scenarios import (  # noqa: E402
    SCENARIOS,
    create_scenario,
    create_scenario_69bus,
)

SAIDA = Path(__file__).parent

# Network factories by name
CRIADORES = {"33bus": create_scenario, "69bus": create_scenario_69bus}


def criar_instancia_reduzida(rede: str, cenario: str, seed: int, n_projetos: int):
    """Instance with N projects and powers scaled by 50/N.

    The scaling preserves the total penetration of the original scenario, so
    the constraint regime (trivial / restrictive) is reproduced at smaller N.
    """
    net, _, cfg = CRIADORES[rede](cenario, seed=seed)
    escala = 50.0 / n_projetos
    projetos = generate_queue(
        net,
        n_projects=n_projetos,
        seed=seed,
        power_distribution=cfg.power_distribution,
        p_min_kw=cfg.p_min_kw * escala,
        p_max_kw=cfg.p_max_kw * escala,
    )
    return net, projetos, cfg


def instancia_original(rede: str, cenario: str, seed: int):
    """Instance identical to the published experiments."""
    return CRIADORES[rede](cenario, seed=seed)


def avaliar(net, projetos, permutacao, constraints=None, d_max=float("inf")):
    if constraints is None:
        constraints = TechnicalConstraints()
    return evaluate_permutation(net, projetos, list(permutacao), constraints, d_max)
