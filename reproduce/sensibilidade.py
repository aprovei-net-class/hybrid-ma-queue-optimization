# -*- coding: utf-8 -*-
"""E1b — Sensibilidade paramétrica (R3.3/R3.5a), desenho OFAT declarado.

Braço A (algoritmo): em torno da configuração base (pop=20, gen=30), varia um
fator por vez: pop {10, 40} e gerações {15, 60}. A base já existe nos dados
publicados (10 runs por instância).

Braço B (modelo físico): limite de sobretensão {1.04, 1.06} pu (base: 1.05).

Instâncias: 33bus I4-s42 e 69bus I4-s7 (as de maior e média ordem-sensibilidade
com restrição ativa nas duas redes). 10 runs por célula, sementes 42000+i
(as mesmas da campanha original — common random numbers).

Uso:
  python sensibilidade.py --smoke        # 1 run curto, mede custo por célula
  python sensibilidade.py --celula pop10 33bus I4 42
  python sensibilidade.py --full         # todas as células (Gate B aprovado)
"""
import argparse
import json
import logging
import time

logging.disable(logging.INFO)

from _comum import SAIDA, avaliar, instancia_original  # noqa: E402
from execution.memetic import MAConfig, run_memetic_algorithm  # noqa: E402
from execution.network import TechnicalConstraints  # noqa: E402

INSTANCIAS = [("33bus", "I4", 42), ("69bus", "I4", 7)]

CELULAS = {
    "pop10":  {"pop_size": 10},
    "pop40":  {"pop_size": 40},
    "gen15":  {"n_generations": 15},
    "gen60":  {"n_generations": 60},
    "vmax104": {"vmax": 1.04},
    "vmax106": {"vmax": 1.06},
}


def rodar_celula(nome, rede, cen, seed, n_runs=10):
    net, projetos, cfg = instancia_original(rede, cen, seed)
    params = dict(CELULAS[nome])
    vmax = params.pop("vmax", None)
    constraints = (TechnicalConstraints(v_max=vmax) if vmax
                   else cfg.constraints)
    fifo = avaliar(net, projetos, [p.id for p in projetos], constraints)
    runs = []
    t0 = time.perf_counter()
    for i in range(n_runs):
        macfg = MAConfig(seed=42000 + i, **params)
        r = run_memetic_algorithm(net, projetos, constraints, macfg)
        runs.append(r.best_fitness)
    return {"runs": runs, "fifo": fifo.fitness,
            "fifo_n_connected": fifo.n_connected,
            "params": {**params, **({"v_max": vmax} if vmax else {})},
            "tempo_s": round(time.perf_counter() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--celula", nargs=4,
                    metavar=("NOME", "REDE", "CEN", "SEED"))
    args = ap.parse_args()

    if args.smoke:
        t0 = time.perf_counter()
        r = rodar_celula("gen15", "33bus", "I4", 42, n_runs=1)
        print(f"smoke gen15 1 run: {r['tempo_s']} s -> por célula de 10 runs "
              f"estimar ~{r['tempo_s']*10/60:.0f} min (células gen60/pop40 "
              f"custam ~4x a gen15)")
        return

    alvos = []
    if args.celula:
        nome, rede, cen, seed = args.celula
        alvos = [(nome, rede, cen, int(seed))]
    elif args.full:
        alvos = [(nome, rede, cen, seed)
                 for nome in CELULAS
                 for rede, cen, seed in INSTANCIAS]
    else:
        ap.error("use --smoke, --celula ou --full")

    destino = SAIDA / "sensibilidade_resultados.json"
    acumulado = json.loads(destino.read_text(encoding="utf-8")) \
        if destino.exists() else {}
    for nome, rede, cen, seed in alvos:
        chave = f"{nome}_{rede}_{cen}_s{seed}"
        if chave in acumulado:
            print(f"{chave}: ja feito, pulando")
            continue
        print(f"== {chave} ==")
        acumulado[chave] = rodar_celula(nome, rede, cen, seed)
        destino.write_text(json.dumps(acumulado, indent=2), encoding="utf-8")
        import numpy as np
        media = float(np.mean(acumulado[chave]["runs"]))
        print(f"  media={media:.0f} kW | FIFO={acumulado[chave]['fifo']:.0f} "
              f"| {acumulado[chave]['tempo_s']} s")


if __name__ == "__main__":
    main()
