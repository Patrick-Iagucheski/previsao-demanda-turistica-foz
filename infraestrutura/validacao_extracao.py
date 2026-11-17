# -*- coding: utf-8 -*-
"""
Validação cruzada da extração dos desembarques (RNF02).

Confronta dois métodos independentes de obter os desembarques em Foz do Iguaçu
a partir dos mesmos arquivos da ANAC:

  MÉTODO A ("outro aeroporto") — o adotado pelo sistema. Conta os passageiros
  de voos que DECOLAM (NR_MOVIMENTO_TIPO='D') de aeroportos brasileiros da base, tendo SBFI
  como outro aeroporto do trecho (NR_VOO_OUTRO_AEROPORTO='SBFI'). Funciona em
  todos os anos, pois não exige que Foz conste como aeroporto de referência.

  MÉTODO B ("direto") — Foz como aeroporto de referência
  (NR_AEROPORTO_REFERENCIA='SBFI') com movimento de POUSO
  (NR_MOVIMENTO_TIPO='P'). Só é possível a partir de abr/2022, quando o
  aeroporto passa a figurar como referência na base.

A divergência esperada corresponde a voos internacionais cuja origem não consta
da base brasileira — Foz é predominantemente doméstico (~98%).

Uso:  python infraestrutura/validacao_extracao.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from dominio.regras import eh_pandemia
from infraestrutura.anac_csv_loader import _detectar_skiprows, listar_csvs

COLS = [
    "ANO", "MES", "NR_AEROPORTO_REFERENCIA", "NR_MOVIMENTO_TIPO",
    "NR_VOO_OUTRO_AEROPORTO", *config.COLUNAS_PAX,
]


def _ler(caminho: Path) -> pd.DataFrame:
    df = pd.read_csv(
        caminho,
        sep=";",
        skiprows=_detectar_skiprows(caminho),
        encoding="utf-8-sig",
        usecols=COLS,
        dtype={
            "NR_AEROPORTO_REFERENCIA": str,
            "NR_MOVIMENTO_TIPO": str,
            "NR_VOO_OUTRO_AEROPORTO": str,
        },
    )
    for c in config.COLUNAS_PAX:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["pax"] = df[config.COLUNAS_PAX].sum(axis=1)
    return df


def comparar(origem: Path = None) -> pd.DataFrame:
    arquivos = listar_csvs(origem)

    linhas = []
    for arq in arquivos:
        df = _ler(arq)

        a = df[
            (df["NR_VOO_OUTRO_AEROPORTO"] == config.ICAO)
            & (df["NR_MOVIMENTO_TIPO"] == config.MOVIMENTO_DECOLAGEM_ORIGEM)
        ]["pax"].sum()

        b = df[
            (df["NR_AEROPORTO_REFERENCIA"] == config.ICAO)
            & (df["NR_MOVIMENTO_TIPO"] == config.MOVIMENTO_DESEMBARQUE)
        ]["pax"].sum()

        ano, mes = int(df["ANO"].iloc[0]), int(df["MES"].iloc[0])
        linhas.append({"ano": ano, "mes": mes, "metodo_a": a, "metodo_b": b})

    return pd.DataFrame(linhas).sort_values(["ano", "mes"]).reset_index(drop=True)


if __name__ == "__main__":
    comp = comparar()

    print("=" * 68)
    print("COBERTURA DA BASE")
    print("=" * 68)
    esperados = {
        (a, m) for a in range(2019, 2026) for m in range(1, 13)
        if not (a == 2019 and m < 3)
    }
    presentes = set(zip(comp["ano"], comp["mes"]))
    faltantes = sorted(esperados - presentes)
    print(f"Competências esperadas (2019-03 a 2025-12): {len(esperados)}")
    print(f"Competências presentes na base:             {len(presentes)}")
    print(f"Ausentes: {[f'{a}-{m:02d}' for a, m in faltantes] or 'nenhuma'}")
    for a, m in faltantes:
        print(f"   {a}-{m:02d} — dentro da pandemia? "
              f"{'sim' if eh_pandemia(a, m) else 'NÃO'}")

    # o método B só existe a partir de abr/2022
    val = comp[comp["metodo_b"] > 0].copy()
    val["dif_%"] = (val["metodo_a"] - val["metodo_b"]) / val["metodo_b"] * 100
    val["concord_%"] = val[["metodo_a", "metodo_b"]].min(axis=1) / \
                       val[["metodo_a", "metodo_b"]].max(axis=1) * 100

    print()
    print("=" * 68)
    print("VALIDAÇÃO CRUZADA — MÉTODO A vs. MÉTODO B")
    print("=" * 68)
    print(f"Período comparável: {val['ano'].min()}-{val[val['ano'] == val['ano'].min()]['mes'].min():02d} "
          f"a {val['ano'].max()}-{val[val['ano'] == val['ano'].max()]['mes'].max():02d} "
          f"({len(val)} meses)")
    print()

    por_ano = val.groupby("ano").agg(
        metodo_a=("metodo_a", "sum"),
        metodo_b=("metodo_b", "sum"),
        meses=("mes", "count"),
    )
    por_ano["dif_%"] = (por_ano["metodo_a"] - por_ano["metodo_b"]) / por_ano["metodo_b"] * 100
    por_ano["concord_%"] = 100 - por_ano["dif_%"].abs()

    print(f"{'Ano':<6}{'Meses':>6}{'Método A':>13}{'Método B':>13}{'Dif. %':>9}{'Concord. %':>12}")
    print("-" * 68)
    for ano, r in por_ano.iterrows():
        print(f"{ano:<6}{int(r['meses']):>6}{int(r['metodo_a']):>13,}"
              f"{int(r['metodo_b']):>13,}{r['dif_%']:>8.2f}%{r['concord_%']:>11.2f}%"
              .replace(",", "."))

    ta, tb = val["metodo_a"].sum(), val["metodo_b"].sum()
    dif = (ta - tb) / tb * 100
    print("-" * 68)
    print(f"{'TOTAL':<6}{len(val):>6}{int(ta):>13,}{int(tb):>13,}"
          f"{dif:>8.2f}%{100 - abs(dif):>11.2f}%".replace(",", "."))
    print()
    print(f"Concordância mensal — mínima: {val['concord_%'].min():.2f}%  "
          f"média: {val['concord_%'].mean():.2f}%  máxima: {val['concord_%'].max():.2f}%")
