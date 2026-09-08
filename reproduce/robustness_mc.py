# -*- coding: utf-8 -*-
"""Paired Monte Carlo robustness study under load and generation uncertainty.

Reproduces the robustness table and figure of the manuscript: re-evaluates the
ALREADY-OPTIMIZED orderings (no re-optimization) under multiplicative load and
generation-availability perturbation, in a PAIRED design (the same perturbation
sample is applied to the optimized ordering and to FIFO).

Método:
  Passo A — recuperação das permutações ótimas: as campanhas de abril/2026 não salvaram
    `best_permutation` nos JSONs (só fitness). Como o MA é determinístico dado o seed
    (rng único `np.random.default_rng(config.seed)` em memetic.py), re-rodar o MA com o
    `algo_seed` do melhor run registrado recupera a permutação. GATE DURO: o replay nominal
    da permutação recuperada deve reproduzir o fitness salvo (e o FIFO idem); senão a
    instância é descartada com o motivo registrado.
  Passo B — Monte Carlo pareado (K amostras, seed fixo): por amostra,
    - fator de carga por barra  f_j ~ U[0.90, 1.10] aplicado a p_mw e q_mvar (FP preservado);
    - fator de disponibilidade por projeto g_i ~ U[0.85, 1.00] aplicado à INJEÇÃO testada
      (a factibilidade é verificada em g_i·p_kw; o fitness conta o kW NOMINAL aceito,
      como em estudos de HC: a outorga é pela potência de placa).
    A mesma dupla (f, g) avalia as duas ordens → ganho pareado por amostra.

Usage (from anywhere): python reproduce/robustness_mc.py
Cost: ~2-3 h (six instances x 2 x 200 perturbed sequential replays).

Outputs (this folder): permutacoes_otimas_recuperadas.json,
robustness_mc_results.json, fig06_robustness.pdf. A reference copy of the
published results is analysis-output/robustness_mc_results.json.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

# Repository-local imports: the frozen experiment code lives in execution/ at
# the repository root, resolved relative to this file (any clean checkout works).
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from execution.evaluator import evaluate_permutation  # noqa: E402
from execution.evaluator import _check_no_new_violations  # noqa: E402
from execution.memetic import MAConfig, run_memetic_algorithm  # noqa: E402
from execution.network import (  # noqa: E402
    TechnicalConstraints,
    add_sgen,
    remove_sgen,
    run_power_flow,
)
from execution.scenarios import create_scenario, create_scenario_69bus  # noqa: E402

AQUI = Path(__file__).resolve().parent
K_AMOSTRAS = 200
SEED_MC = 20260727
CARGA_MIN, CARGA_MAX = 0.90, 1.10
GER_MIN, GER_MAX = 0.85, 1.00
TOL_KW = 0.01  # tolerância do gate de reprodução (kW)


def tolerancia_para(valor_salvo: float) -> float:
    """O JSON do 69 barras salvou fitness arredondado para inteiro; o do 33 não.

    Para valor salvo inteiro a comparação justa é contra o arredondamento (±0,5 kW);
    para valor com casas decimais, exige reprodução exata (±0,01 kW).
    """
    return 0.5 if float(valor_salvo).is_integer() else TOL_KW

# Instâncias: cobrem o espectro de ganho nominal (de +5,6% a +67,6% no 33b; 0,3% a 17,3% no 69b)
INSTANCIAS = [
    {"rede": "33bus", "cenario": "I3", "seed": 42},
    {"rede": "33bus", "cenario": "I4", "seed": 42},
    {"rede": "33bus", "cenario": "I4", "seed": 7},
    {"rede": "69bus", "cenario": "I3", "seed": 13},
    {"rede": "69bus", "cenario": "I4", "seed": 42},
    {"rede": "69bus", "cenario": "I4", "seed": 7},
]

JSON_33 = RAIZ / "analysis-output" / "experiment_optionA_20260406_173130.json"
JSON_69 = RAIZ / "analysis-output" / "results_69bus_saved.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seeds_e_fitness_salvos(inst: dict) -> dict:
    """Extrai do JSON da campanha: algo_seed do melhor run, fitness salvo do MA e do FIFO."""
    if inst["rede"] == "33bus":
        dados = json.loads(JSON_33.read_text(encoding="utf-8"))
        registros = dados["scenarios"][inst["cenario"]]["instances"]
        alvo = next(r for r in registros if r["instance_seed"] == inst["seed"])
        melhor = max(alvo["ma_runs"], key=lambda r: r["fitness"])
        return {
            "algo_seed": melhor["algo_seed"],
            "fitness_ma_salvo": melhor["fitness"],
            "fitness_fifo_salvo": alvo["baselines"]["FIFO"]["fitness"],
        }
    dados = json.loads(JSON_69.read_text(encoding="utf-8"))
    alvo = dados["part1_69bus_main"][f'{inst["cenario"]}_s{inst["seed"]}']
    runs = alvo["MA_runs"]
    i_melhor = int(np.argmax(runs))
    return {
        "algo_seed": inst["seed"] * 1000 + i_melhor,
        "fitness_ma_salvo": runs[i_melhor],
        "fitness_fifo_salvo": alvo["FIFO"],
    }


def montar_instancia(inst: dict):
    if inst["rede"] == "33bus":
        return create_scenario(inst["cenario"], seed=inst["seed"])
    return create_scenario_69bus(inst["cenario"], seed=inst["seed"])


def replay_perturbado(net_base, projetos, permutacao, restricoes,
                      fatores_carga: np.ndarray, fatores_ger: dict[int, float]) -> float:
    """Replay sequencial sob perturbação; retorna o kW NOMINAL total aceito.

    Mesma lógica de aceitação de evaluate_permutation (evaluator.py), com duas diferenças
    deliberadas: (1) a carga da rede é escalada por barra antes do laço; (2) a injeção
    testada é g_i·p_kw, mas o fitness soma o p_kw nominal do projeto aceito.
    """
    net = copy.deepcopy(net_base)
    net.load.p_mw *= fatores_carga
    net.load.q_mvar *= fatores_carga

    mapa = {p.id: p for p in projetos}
    if not run_power_flow(net):
        return float("nan")  # caso-base perturbado não convergiu (registrado como NaN)
    v_snapshot = net.res_bus.vm_pu.values.copy()

    total_nominal = 0.0
    for proj_id in permutacao:
        proj = mapa[proj_id]
        inj_kw = fatores_ger[proj_id] * proj.p_kw
        idx = add_sgen(net, proj.bus, inj_kw, name=proj.name)
        if not run_power_flow(net):
            remove_sgen(net, idx)
            continue
        if _check_no_new_violations(net, v_snapshot, restricoes):
            total_nominal += proj.p_kw
            v_snapshot = net.res_bus.vm_pu.values.copy()
        else:
            remove_sgen(net, idx)
            net.res_bus.vm_pu = v_snapshot.copy()
    return total_nominal


def main() -> int:
    t_ini = time.perf_counter()
    restricoes = TechnicalConstraints()
    resultado = {"config": {
        "K": K_AMOSTRAS, "seed_mc": SEED_MC,
        "carga": [CARGA_MIN, CARGA_MAX], "geracao": [GER_MIN, GER_MAX],
        "pareado": True, "fitness": "kW nominal aceito; factibilidade testada em g_i*p_kw",
    }, "instancias": []}
    permutacoes = {}

    for i_inst, inst in enumerate(INSTANCIAS):
        rotulo = f'{inst["rede"]} {inst["cenario"]}-s{inst["seed"]}'
        log(f"=== {rotulo} ===")
        salvo = seeds_e_fitness_salvos(inst)
        net, projetos, _cfg = montar_instancia(inst)

        # --- Passo A: recuperar a permutação ótima via seed determinística ---
        log(f"  recuperando permutação (algo_seed={salvo['algo_seed']})...")
        t0 = time.perf_counter()
        cfg = MAConfig(seed=salvo["algo_seed"])  # defaults == config das campanhas
        ma = run_memetic_algorithm(net, projetos, restricoes, cfg)
        log(f"  MA re-rodado em {time.perf_counter()-t0:.0f}s; fitness={ma.best_fitness:.1f} "
            f"(salvo: {salvo['fitness_ma_salvo']:.1f})")

        perm_ma = ma.best_permutation
        fifo = sorted(projetos, key=lambda p: p.chrono_position)
        perm_fifo = [p.id for p in fifo]

        # --- GATE DURO: replay nominal reproduz os fitness salvos ---
        nom_ma = evaluate_permutation(net, projetos, perm_ma, restricoes).fitness
        nom_fifo = evaluate_permutation(net, projetos, perm_fifo, restricoes).fitness
        gate_ma = abs(nom_ma - salvo["fitness_ma_salvo"]) <= tolerancia_para(salvo["fitness_ma_salvo"])
        gate_fifo = abs(nom_fifo - salvo["fitness_fifo_salvo"]) <= tolerancia_para(salvo["fitness_fifo_salvo"])
        if not (gate_ma and gate_fifo):
            log(f"  !! GATE FALHOU (MA {nom_ma:.2f} vs {salvo['fitness_ma_salvo']:.2f}; "
                f"FIFO {nom_fifo:.2f} vs {salvo['fitness_fifo_salvo']:.2f}) — instância descartada")
            resultado["instancias"].append({
                "instancia": rotulo, "descartada": True,
                "motivo": "replay nominal não reproduziu o fitness salvo",
                "nominal_ma": nom_ma, "salvo_ma": salvo["fitness_ma_salvo"],
                "nominal_fifo": nom_fifo, "salvo_fifo": salvo["fitness_fifo_salvo"],
            })
            continue
        log(f"  gate OK: MA {nom_ma:.1f} kW, FIFO {nom_fifo:.1f} kW (idênticos aos salvos)")
        permutacoes[rotulo] = {
            "algo_seed": salvo["algo_seed"], "fitness_nominal": nom_ma,
            "permutacao_ma": perm_ma, "permutacao_fifo": perm_fifo,
        }

        # --- Passo B: Monte Carlo pareado ---
        rng = np.random.default_rng(SEED_MC + i_inst)
        n_cargas = len(net.load)
        ids = [p.id for p in projetos]
        ganhos, fit_ma_mc, fit_fifo_mc, nao_convergiu = [], [], [], 0
        t0 = time.perf_counter()
        for k in range(K_AMOSTRAS):
            f_carga = rng.uniform(CARGA_MIN, CARGA_MAX, size=n_cargas)
            f_ger = dict(zip(ids, rng.uniform(GER_MIN, GER_MAX, size=len(ids))))
            fma = replay_perturbado(net, projetos, perm_ma, restricoes, f_carga, f_ger)
            ffo = replay_perturbado(net, projetos, perm_fifo, restricoes, f_carga, f_ger)
            if np.isnan(fma) or np.isnan(ffo) or ffo == 0:
                nao_convergiu += 1
                continue
            fit_ma_mc.append(fma)
            fit_fifo_mc.append(ffo)
            ganhos.append((fma - ffo) / ffo * 100.0)
            if (k + 1) % 25 == 0:
                por_amostra = (time.perf_counter() - t0) / (k + 1)
                log(f"  MC {k+1}/{K_AMOSTRAS} ({por_amostra:.2f}s/amostra; "
                    f"ganho médio parcial {np.mean(ganhos):+.2f}%)")

        g = np.array(ganhos)
        resultado["instancias"].append({
            "instancia": rotulo, "descartada": False,
            "ganho_nominal_pct": (nom_ma - nom_fifo) / nom_fifo * 100.0,
            "n_amostras_validas": int(len(g)), "n_nao_convergiu": nao_convergiu,
            "ganho_mc": {
                "media_pct": float(np.mean(g)), "mediana_pct": float(np.median(g)),
                "p2_5": float(np.percentile(g, 2.5)), "p97_5": float(np.percentile(g, 97.5)),
                "min": float(np.min(g)), "max": float(np.max(g)),
                "frac_ganho_positivo": float(np.mean(g > 0)),
                "frac_ganho_nao_negativo": float(np.mean(g >= 0)),
            },
            "fitness_ma_mc_media": float(np.mean(fit_ma_mc)),
            "fitness_fifo_mc_media": float(np.mean(fit_fifo_mc)),
            "ganhos_amostrais_pct": [round(float(x), 4) for x in g],
        })
        # dump incremental (crash-safe)
        (AQUI / "robustness_mc_results.json").write_text(
            json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
        (AQUI / "permutacoes_otimas_recuperadas.json").write_text(
            json.dumps(permutacoes, indent=2, ensure_ascii=False), encoding="utf-8")

    resultado["tempo_total_s"] = round(time.perf_counter() - t_ini, 1)
    (AQUI / "robustness_mc_results.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"CONCLUÍDO em {resultado['tempo_total_s']/60:.1f} min")

    # --- figura (boxplot dos ganhos pareados) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ok = [r for r in resultado["instancias"] if not r["descartada"]]
        fig, ax = plt.subplots(figsize=(7.0, 3.4))
        ax.boxplot([r["ganhos_amostrais_pct"] for r in ok],
                   tick_labels=[r["instancia"].replace("bus ", "-bus\n") for r in ok],
                   showfliers=True, whis=(2.5, 97.5))
        for j, r in enumerate(ok, start=1):
            ax.plot(j, r["ganho_nominal_pct"], marker="D", ms=5, color="tab:red", zorder=3)
        ax.axhline(0.0, color="gray", lw=0.8, ls="--")
        ax.set_ylabel("Paired gain over FIFO (%)")
        ax.set_title(f"Optimized order vs FIFO under load/generation uncertainty "
                     f"(K={K_AMOSTRAS}; diamonds: nominal gain)", fontsize=9)
        fig.tight_layout()
        fig.savefig(AQUI / "fig06_robustness.pdf")
        log("figura fig06_robustness.pdf gravada")
    except Exception as exc:  # noqa: BLE001 — figura é acessória; o JSON é o resultado
        log(f"figura falhou ({exc}); JSON está completo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
