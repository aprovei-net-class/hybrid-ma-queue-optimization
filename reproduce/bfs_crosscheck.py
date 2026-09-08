# -*- coding: utf-8 -*-
"""BFS-vs-Newton-Raphson cross-check cited in the manuscript (Implementation).

The manuscript justifies the full NR solver by citing max|dV| = 3.3e-3 pu
between an independent linearized backward/forward sweep implementation and
the pandapower NR solution on the base 33-bus network. This script reproduces
that number using the self-contained BFS in bfs_standalone.py (same folder).

Cost: seconds. Usage: python bfs_crosscheck.py
"""
from collections import deque

import numpy as np
import pandapower as pp
import pandapower.networks as pn

from bfs_standalone import RedeRadial, fluxo_potencia


def montar_rede_radial(net) -> RedeRadial:
    """Constrói a representação pai/filho a partir das linhas em serviço."""
    adj = {}
    for _, ln in net.line.iterrows():
        if not ln.in_service:
            continue
        # Linha com switch aberto está fora do caminho radial
        sw = net.switch
        aberta = False
        if len(sw):
            m = sw[(sw.et == "l") & (sw.element == ln.name) & (~sw.closed)]
            aberta = len(m) > 0
        if aberta:
            continue
        a, b = int(ln.from_bus), int(ln.to_bus)
        r = float(ln.r_ohm_per_km * ln.length_km)
        x = float(ln.x_ohm_per_km * ln.length_km)
        adj.setdefault(a, []).append((b, r, x))
        adj.setdefault(b, []).append((a, r, x))

    n = len(net.bus)
    pai = np.full(n, -1, dtype=int)
    r_tr = np.zeros(n)
    x_tr = np.zeros(n)
    ordem = []
    visto = {0}
    fila = deque([0])
    while fila:
        b = fila.popleft()
        ordem.append(b)
        for viz, r, x in adj.get(b, []):
            if viz not in visto:
                visto.add(viz)
                pai[viz] = b
                r_tr[viz], x_tr[viz] = r, x
                fila.append(viz)
    assert len(ordem) == n, f"rede nao radial/conexa: {len(ordem)}/{n}"
    return RedeRadial(
        n_barras=n, pai=pai, r_trecho_ohm=r_tr, x_trecho_ohm=x_tr,
        ordem_topo=np.array(ordem), v_base_kV=float(net.bus.vn_kv.iloc[0]),
    )


def main():
    net = pn.case33bw()
    pp.runpp(net, algorithm="nr", tolerance_mva=1e-6, max_iteration=30)
    v_nr = net.res_bus.vm_pu.values.copy()

    rede = montar_rede_radial(net)
    n = rede.n_barras
    carga_kw = np.zeros(n)
    carga_kvar = np.zeros(n)
    for _, ld in net.load.iterrows():
        carga_kw[int(ld.bus)] += ld.p_mw * 1000.0
        carga_kvar[int(ld.bus)] += ld.q_mvar * 1000.0

    res = fluxo_potencia(rede, carga_kw, carga_kvar, np.zeros(n))
    v_bf = np.asarray(res["V_pu"], dtype=float)
    delta = np.abs(v_bf - v_nr)
    print(f"max|dV| BFS vs NR (33 barras, caso base): {delta.max():.2e} pu")
    print("citado no manuscrito (secao 5.4): 3.3e-3 pu")


if __name__ == "__main__":
    main()
