# -*- coding: utf-8 -*-
"""E1c — Baseline de ranking ESTÁTICO multicritério (proxy de Fontinele2026).

Responde ao R3.5b ("simulate your proposed method with the closest reference").
O método mais próximo é o trabalho anterior dos autores: ranking multicritério
estático (AHP+TOPSIS) com avaliação de cada projeto em isolamento. Este proxy
reproduz o ASPECTO ESTRUTURAL contrastado no artigo — decisão por ranking
estático sobre a rede base — sem a elicitação AHP (pesos iguais, declarado):

  Critérios por projeto (medidos com o projeto SOZINHO na rede base):
    C1 (benefício): potência nominal do projeto (kW)
    C2 (custo):     elevação de tensão máxima causada na rede base (p.u.)
    C3 (custo):     posição cronológica (antiguidade preservada)

  Score TOPSIS clássico com pesos iguais; ordena por score decrescente;
  a ORDEM resultante é avaliada com o MESMO avaliador sequencial dinâmico
  dos demais métodos (mesmas instâncias, mesmas restrições).

Custo: 50 fluxos por instância + 1 avaliação sequencial (~1,5 s).
"""
import copy
import json
import logging
import time

import numpy as np

logging.disable(logging.INFO)

from _comum import SAIDA, avaliar, instancia_original  # noqa: E402
from execution.network import add_sgen, remove_sgen, run_power_flow  # noqa: E402


def criterios_estaticos(net_base, projetos):
    """Mede C2 (ΔVmax na rede base) conectando cada projeto sozinho."""
    net = copy.deepcopy(net_base)
    run_power_flow(net)
    v_base = net.res_bus.vm_pu.values.copy()
    delta_v = []
    for p in projetos:
        idx = add_sgen(net, p.bus, p.p_kw, name=p.name)
        ok = run_power_flow(net)
        dv = float(np.nanmax(net.res_bus.vm_pu.values - v_base)) if ok else float("inf")
        delta_v.append(dv)
        remove_sgen(net, idx)
    return delta_v


def topsis(matriz, beneficio):
    """TOPSIS clássico, pesos iguais. matriz: linhas=alternativas."""
    m = np.asarray(matriz, float)
    norma = np.linalg.norm(m, axis=0)
    norma[norma == 0] = 1.0
    r = m / norma
    ideal = np.where(beneficio, r.max(axis=0), r.min(axis=0))
    anti = np.where(beneficio, r.min(axis=0), r.max(axis=0))
    d_pos = np.linalg.norm(r - ideal, axis=1)
    d_neg = np.linalg.norm(r - anti, axis=1)
    denom = d_pos + d_neg
    denom[denom == 0] = 1.0
    return d_neg / denom


def main():
    resultados = {}
    t0 = time.perf_counter()
    for rede in ("33bus", "69bus"):
        for cen in ("I3", "I4"):
            for seed in (42, 7, 13):
                net, projetos, cfg = instancia_original(rede, cen, seed)
                dv = criterios_estaticos(net, projetos)
                matriz = [[p.p_kw, dv[i], p.chrono_position]
                          for i, p in enumerate(projetos)]
                scores = topsis(matriz, beneficio=[True, False, False])
                ordem = [projetos[i].id for i in np.argsort(-scores)]
                res = avaliar(net, projetos, ordem, cfg.constraints)
                fifo = avaliar(net, projetos, [p.id for p in projetos],
                               cfg.constraints)
                chave = f"{rede}_{cen}_s{seed}"
                resultados[chave] = {
                    "static_rank_fitness": res.fitness,
                    "static_rank_n_connected": res.n_connected,
                    "fifo_fitness": fifo.fitness,
                    "gain_vs_fifo_pct": (res.fitness - fifo.fitness)
                    / fifo.fitness * 100 if fifo.fitness else 0.0,
                }
                print(f"{chave}: static={res.fitness:.0f} kW "
                      f"({res.n_connected}) | FIFO={fifo.fitness:.0f} "
                      f"| ganho={resultados[chave]['gain_vs_fifo_pct']:+.1f}%")
    resultados["_meta"] = {
        "criterios": "P_kw (beneficio), dVmax rede base (custo), "
                     "posicao cronologica (custo); TOPSIS pesos iguais",
        "tempo_total_s": round(time.perf_counter() - t0, 1),
    }
    destino = SAIDA / "baseline_estatico_resultados.json"
    destino.write_text(json.dumps(resultados, indent=2), encoding="utf-8")
    print(f"\nGravado em {destino} ({resultados['_meta']['tempo_total_s']} s)")


if __name__ == "__main__":
    main()
