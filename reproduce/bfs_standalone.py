"""Fluxo de potência backward-forward (varredura) para redes radiais.

Alternativa leve ao pandapower para alimentadores radiais de distribuição
(ex.: IEEE 34 barras): acumula potências das folhas para a raiz (backward)
e propaga quedas de tensão da raiz para as folhas (forward), em pu.

Origem: porte MATLAB→Python de ``MMGD_melhorias_SBA.m`` (Artigo - Pablo),
funções ``fluxo_potencia_v4`` (l.477) e ``calcular_tensao_final`` (l.733).
Autoria: Williams F. Fontinele et al.

Desvio deliberado do original: ``perdas_kW`` aqui é físico (Σ S²·R/V_LL²
por trecho, em kW). O MATLAB (l.504) usava potências em kW onde caberiam
watts e omitia o fator trifásico, produzindo um valor ~3,3e5 vezes menor
que só servia como critério relativo no TOPSIS. Ver ``fluxo_potencia``.

Convenção: índices de barra 0-based (barra 0 = subestação/referência),
diferentemente do MATLAB original (1-based).
"""
from dataclasses import dataclass

import numpy as np

__all__ = ["RedeRadial", "fluxo_potencia", "calcular_tensao_final"]


@dataclass(frozen=True, eq=False)
class RedeRadial:
    """Rede radial em representação pai/filho (índices 0-based).

    ``pai[b] == -1`` indica a raiz (subestação). ``ordem_topo`` é a ordem
    topológica obtida por BFS a partir da raiz.
    """

    n_barras: int
    pai: np.ndarray            # (n_barras,) int — pai de cada barra (-1 na raiz)
    r_trecho_ohm: np.ndarray   # (n_barras,) R do trecho pai→barra [ohm]
    x_trecho_ohm: np.ndarray   # (n_barras,) X do trecho pai→barra [ohm]
    ordem_topo: np.ndarray     # (n_barras,) int — ordem topológica (BFS)
    v_base_kV: float           # tensão de base linha-linha [kV]


def fluxo_potencia(
    rede: RedeRadial,
    carga_kW: np.ndarray,
    carga_kVAr: np.ndarray,
    geracao_kW: np.ndarray,
    v_nom_pu: float = 1.0,
    n_varreduras: int = 3,
) -> dict:
    """Executa a varredura backward-forward na rede radial.

    Args:
        rede: rede radial construída (ex.: via ``core.otimizacao.construir_rede``).
        carga_kW: carga ativa por barra [kW].
        carga_kVAr: carga reativa por barra [kVAr].
        geracao_kW: geração ativa injetada por barra [kW].
        v_nom_pu: tensão nominal (referência na subestação) [pu].
        n_varreduras: número de varreduras backward-forward (3 no original).

    Returns:
        dict com ``V_pu`` (tensões por barra), ``perdas_kW`` (perdas ativas
        trifásicas totais, físicas) e ``desvio_max_pu``.

    Notas:
        - ``perdas_kW`` = Σ S²·R/(V_pai·V_LL)² por trecho (kW), com S das
          potências acumuladas da última varredura. CORRIGE o cálculo do
          MATLAB original (l.504), que misturava kW/W e omitia o fator
          trifásico (valor ~3,3e5 vezes menor, usado lá apenas como
          critério relativo no TOPSIS — a otimização não é afetada).
        - Domínio de validade (validação cruzada com pandapower
          Newton-Raphson na IEEE 34, com e sem GD): desvio de tensão <1%
          para perfis com V ≥ 0,92 pu (faixa decisória PRODIST); sob
          subtensão severa (~0,86 pu) o erro da varredura linearizada
          chega a ~1,6%. Para relatórios/memórias de cálculo fora dessa
          faixa, prefira pandapower (Newton-Raphson).
    """
    pai = rede.pai
    topo = rede.ordem_topo
    r = rede.r_trecho_ohm
    x = rede.x_trecho_ohm

    v = np.full(rede.n_barras, v_nom_pu, dtype=float)
    p_net = np.asarray(carga_kW, dtype=float) - np.asarray(geracao_kW, dtype=float)
    q_net = np.asarray(carga_kVAr, dtype=float)

    p_acum = p_net.copy()
    q_acum = q_net.copy()
    for _ in range(n_varreduras):
        # Backward: acumula potências das folhas para a raiz.
        p_acum = p_net.copy()
        q_acum = q_net.copy()
        for b in topo[::-1]:
            p = pai[b]
            if p >= 0:
                p_acum[p] += p_acum[b]
                q_acum[p] += q_acum[b]
        # Forward: propaga quedas de tensão da raiz para as folhas.
        v[topo[0]] = v_nom_pu
        for b in topo[1:]:
            p = pai[b]
            if p < 0:
                continue
            dv = (p_acum[b] * r[b] + q_acum[b] * x[b]) / (
                v[p] * rede.v_base_kV**2 * 1000.0
            )
            v[b] = v[p] - dv

    # Perdas ativas trifásicas por trecho: P_loss = S²·R/V_LL² (S acumulada
    # da última varredura). kVA²·Ω/kV² = W; divide-se por 1000 → kW.
    # Corrige o MATLAB original (l.504) — ver docstring.
    filhos = topo[1:]
    pais = pai[filhos]
    mask = pais >= 0
    b_ok = filhos[mask]
    p_ok = pais[mask]
    s2_kVA2 = p_acum[b_ok] ** 2 + q_acum[b_ok] ** 2
    v_ll_kV2 = (v[p_ok] * rede.v_base_kV) ** 2
    perdas_kW = float(np.sum(s2_kVA2 * r[b_ok] / v_ll_kV2) / 1000.0)

    return {
        "V_pu": v,
        "perdas_kW": perdas_kW,
        "desvio_max_pu": float(np.max(np.abs(v - v_nom_pu))),
    }


def calcular_tensao_final(
    sigma: np.ndarray,
    barras_projeto: np.ndarray,
    p_projeto_kW: np.ndarray,
    rede: RedeRadial,
    carga_kW: np.ndarray,
    carga_kVAr: np.ndarray,
    v_nom_pu: float = 1.0,
    v_sup_pu: float = 1.05,
    v_inf_pu: float = 0.92,
) -> dict:
    """Tensões finais após conexão sequencial dos projetos viáveis.

    Percorre a permutação ``sigma`` e conecta cada projeto somente se as
    tensões permanecerem dentro dos limites PRODIST (0,92–1,05 pu).

    Args:
        sigma: permutação de índices de projetos (0-based).
        barras_projeto: barra (0-based) de cada projeto.
        p_projeto_kW: potência de cada projeto [kW].
        rede: rede radial.
        carga_kW: carga ativa por barra [kW].
        carga_kVAr: carga reativa por barra [kVAr].
        v_nom_pu: tensão nominal [pu].
        v_sup_pu: limite superior de tensão [pu].
        v_inf_pu: limite inferior de tensão [pu].

    Returns:
        dict com ``V_pu`` (perfil final), ``geracao_kW`` (por barra),
        ``n_conectados`` e ``perdas_kW`` (trifásicas, físicas — ver
        ``fluxo_potencia``).
    """
    geracao = np.zeros(rede.n_barras, dtype=float)
    n_conectados = 0
    for idx in np.asarray(sigma, dtype=int):
        ger_teste = geracao.copy()
        ger_teste[barras_projeto[idx]] += p_projeto_kW[idx]
        v = fluxo_potencia(rede, carga_kW, carga_kVAr, ger_teste, v_nom_pu)["V_pu"]
        if np.all(v >= v_inf_pu) and np.all(v <= v_sup_pu):
            geracao = ger_teste
            n_conectados += 1
    res = fluxo_potencia(rede, carga_kW, carga_kVAr, geracao, v_nom_pu)
    return {
        "V_pu": res["V_pu"],
        "geracao_kW": geracao,
        "n_conectados": n_conectados,
        "perdas_kW": res["perdas_kW"],
    }
