# -*- coding: utf-8 -*-
"""E1a — Optimality gap por enumeração exaustiva em instâncias N=8 (R1.5).

Enumera as 8! = 40.320 permutações de instâncias reduzidas (potências escaladas
por 50/8 para reproduzir o regime restritivo) e compara o ótimo global com
MA, GA-only e FIFO nas MESMAS instâncias, usando o MESMO avaliador sequencial.

Uso:
  python enum_n8.py --smoke          # 200 permutações, mede custo, não grava
  python enum_n8.py --instancia 33bus I4 42   # enumera 1 instância (grava)
  python enum_n8.py --full           # as 6 instâncias (Gate B aprovado)
"""
import argparse
import itertools
import json
import logging
import time

import numpy as np

logging.disable(logging.INFO)

from _comum import SAIDA, avaliar, criar_instancia_reduzida  # noqa: E402
from execution.memetic import MAConfig, run_memetic_algorithm  # noqa: E402

N = 8
INSTANCIAS = [("33bus", "I4", s) for s in (42, 7, 13)] + \
             [("69bus", "I4", s) for s in (42, 7, 13)]


def enumerar(net, projetos, constraints, limite=None):
    ids = [p.id for p in projetos]
    melhor, melhor_perm = -1.0, None
    fitness_fifo = avaliar(net, projetos, ids, constraints).fitness
    t0 = time.perf_counter()
    total = 0
    for perm in itertools.permutations(ids):
        res = avaliar(net, projetos, perm, constraints)
        if res.fitness > melhor:
            melhor, melhor_perm = res.fitness, list(perm)
        total += 1
        if limite and total >= limite:
            break
    return {
        "otimo_global": melhor,
        "perm_otima": melhor_perm,
        "fifo": fitness_fifo,
        "n_avaliadas": total,
        "tempo_s": round(time.perf_counter() - t0, 1),
    }


def rodar_ma_ga(net, projetos, constraints, n_runs=10):
    saida = {"ma": [], "ga": []}
    for i in range(n_runs):
        for rotulo, ls in (("ma", 0.3), ("ga", 0.0)):
            cfg = MAConfig(seed=42000 + i, local_search_prob=ls)
            r = run_memetic_algorithm(net, projetos, constraints, cfg)
            saida[rotulo].append(r.best_fitness)
    return saida


def uma_instancia(rede, cen, seed, smoke=False):
    net, projetos, cfg = criar_instancia_reduzida(rede, cen, seed, N)
    chave = f"{rede}_{cen}_s{seed}_N{N}"
    print(f"== {chave} ==")
    lim = 200 if smoke else None
    enum = enumerar(net, projetos, cfg.constraints, limite=lim)
    print(f"  enum: {enum['n_avaliadas']} perms em {enum['tempo_s']} s "
          f"({enum['tempo_s']/enum['n_avaliadas']*1000:.0f} ms/perm) "
          f"| melhor={enum['otimo_global']:.0f} | FIFO={enum['fifo']:.0f}")
    if smoke:
        est_h = enum["tempo_s"] / enum["n_avaliadas"] * 40320 / 3600
        print(f"  estimativa enumeração completa: {est_h:.1f} h/instância")
        return None
    algs = rodar_ma_ga(net, projetos, cfg.constraints)
    gap = {k: (enum["otimo_global"] - float(np.mean(v)))
           / enum["otimo_global"] * 100 for k, v in algs.items()}
    resultado = {**enum,
                 "ma_runs": algs["ma"], "ga_runs": algs["ga"],
                 "gap_medio_ma_pct": gap["ma"], "gap_medio_ga_pct": gap["ga"],
                 "ma_atingiu_otimo": int(sum(np.isclose(v, enum["otimo_global"])
                                             for v in algs["ma"]))}
    print(f"  MA gap médio: {gap['ma']:.2f}% | GA: {gap['ga']:.2f}% "
          f"| runs MA no ótimo: {resultado['ma_atingiu_otimo']}/10")
    return chave, resultado


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--instancia", nargs=3, metavar=("REDE", "CEN", "SEED"))
    args = ap.parse_args()

    if args.smoke:
        uma_instancia("33bus", "I4", 42, smoke=True)
        return

    alvo = ([tuple([args.instancia[0], args.instancia[1],
                    int(args.instancia[2])])] if args.instancia
            else INSTANCIAS if args.full else [])
    if not alvo:
        ap.error("use --smoke, --instancia ou --full")

    destino = SAIDA / "enum_n8_resultados.json"
    acumulado = json.loads(destino.read_text(encoding="utf-8")) \
        if destino.exists() else {}
    for rede, cen, seed in alvo:
        chave, resultado = uma_instancia(rede, cen, seed)
        acumulado[chave] = resultado
        destino.write_text(json.dumps(acumulado, indent=2), encoding="utf-8")
        print(f"  gravado em {destino}")


if __name__ == "__main__":
    main()
