# -*- coding: utf-8 -*-
"""A8/E3 — Medição do limiar de ativação da dependência de ordem (R1.4).

Para cada (rede, seed): parte da fila do cenário I2 e escala TODAS as potências
por um fator comum; busca binária no fator até localizar o menor nível de
penetração em que alguma das três ordens determinísticas (FIFO, Greedy-Largest,
Greedy-Smallest) rejeita ao menos um projeto. Abaixo desse nível, toda ordem
conecta 50/50 e a ordenação é trivialmente irrelevante.

Custo: ~14 iterações × 3 avaliações × ~0,8 s ≈ 35 s por (rede, seed).
"""
import json
import logging
import time

logging.disable(logging.INFO)

from _comum import SAIDA, avaliar, instancia_original  # noqa: E402
from execution.queue_generator import Project, get_penetration_pct  # noqa: E402


def escalar(projetos, fator):
    return [Project(p.id, p.bus, p.p_kw * fator, p.chrono_position, p.name)
            for p in projetos]


def alguma_rejeicao(net, projetos, constraints):
    ids = [p.id for p in projetos]
    ordens = [
        ids,
        sorted(ids, key=lambda i: -projetos[i].p_kw),
        sorted(ids, key=lambda i: projetos[i].p_kw),
    ]
    return any(avaliar(net, projetos, o, constraints).n_rejected > 0
               for o in ordens)


def medir(rede, seed, tol=0.01):
    net, base, cfg = instancia_original(rede, "I2", seed)
    lo, hi = 1.0, 6.0
    if alguma_rejeicao(net, escalar(base, lo), cfg.constraints):
        return {"erro": "ja ha rejeicao no fator 1.0"}
    while not alguma_rejeicao(net, escalar(base, hi), cfg.constraints):
        hi *= 1.5
    while (hi - lo) / lo > tol:
        mid = (lo + hi) / 2
        if alguma_rejeicao(net, escalar(base, mid), cfg.constraints):
            hi = mid
        else:
            lo = mid
    pen_lo = get_penetration_pct(net, escalar(base, lo))
    pen_hi = get_penetration_pct(net, escalar(base, hi))
    return {"fator": round(hi, 3), "penetracao_limiar_pct":
            round((pen_lo + pen_hi) / 2, 1),
            "intervalo_pct": [round(pen_lo, 1), round(pen_hi, 1)]}


def main():
    resultados = {}
    t0 = time.perf_counter()
    for rede in ("33bus", "69bus"):
        for seed in (42, 7, 13):
            r = medir(rede, seed)
            resultados[f"{rede}_s{seed}"] = r
            print(f"{rede} s{seed}: {r}")
    resultados["_meta"] = {"criterio": "menor penetracao com >=1 rejeicao em "
                           "FIFO/Greedy-L/Greedy-S (fila I2 escalada)",
                           "tempo_total_s": round(time.perf_counter() - t0, 1)}
    destino = SAIDA / "limiar_ativacao_resultados.json"
    destino.write_text(json.dumps(resultados, indent=2), encoding="utf-8")
    print(f"\nGravado em {destino} ({resultados['_meta']['tempo_total_s']} s)")


if __name__ == "__main__":
    main()
