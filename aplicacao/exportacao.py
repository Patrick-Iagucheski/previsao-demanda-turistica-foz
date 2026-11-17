# -*- coding: utf-8 -*-
"""
Serviço de exportação (camada de aplicação) — RF22.

    exportar(execucao) -> conteúdo CSV
    exportar_lote(execucoes) -> conteúdo CSV com várias execuções

Formato pensado para abrir direto no Excel em português: separador ponto e
vírgula, vírgula decimal e marca de ordem de bytes UTF-8 (acentos corretos).
Cada linha traz uma coluna `secao` que indica seu conteúdo:

    execucao  metadados: modelo, semente, data, período de teste, hiperparâmetros
    previsao  um mês do teste: real, previsto e erro percentual
    metrica   uma métrica num escopo (global ou alta temporada)
"""
import csv
import io
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dominio.entidades import EscopoAvaliacaoEnum, Execucao, TipoMetricaEnum, TipoModeloEnum

COLUNAS = ["secao", "id_execucao", "modelo", "semente", "data_hora", "periodo_teste",
           "hiperparametros", "mes", "alta_temporada", "valor_real", "valor_previsto",
           "erro_percentual", "escopo", "metrica", "valor"]


def _num(v, casas=2):
    return "" if v is None else f"{v:.{casas}f}".replace(".", ",")


class ExportacaoCsvService:

    def exportar(self, execucao: Execucao) -> str:
        return self.exportar_lote([execucao])

    def exportar_lote(self, execucoes: Sequence[Execucao]) -> str:
        saida = io.StringIO()
        w = csv.writer(saida, delimiter=";", lineterminator="\r\n")
        w.writerow(COLUNAS)
        for e in execucoes:
            modelo = TipoModeloEnum(e.tipo_modelo).value
            comum = {"id_execucao": e.id, "modelo": modelo, "semente": e.semente_aleatoria}
            linhas = [{**comum, "secao": "execucao",
                       "data_hora": e.data_hora_execucao.isoformat(timespec="seconds"),
                       "periodo_teste": e.periodo_teste,
                       "hiperparametros": json.dumps(e.hiperparametros, ensure_ascii=False)}]
            for p in e.previsoes:
                erro = (None if p.valor_real in (None, 0)
                        else 100 * (p.valor_real - p.valor_previsto) / p.valor_real)
                linhas.append({**comum, "secao": "previsao", "mes": p.rotulo,
                               "alta_temporada": "sim" if p.alta_temporada else "não",
                               "valor_real": _num(p.valor_real, 0),
                               "valor_previsto": _num(p.valor_previsto, 0),
                               "erro_percentual": _num(erro, 2)})
            for m in e.metricas:
                linhas.append({**comum, "secao": "metrica",
                               "escopo": EscopoAvaliacaoEnum(m.escopo_avaliacao).value,
                               "metrica": TipoMetricaEnum(m.tipo_metrica).value,
                               "valor": _num(m.valor_metrica, 4)})
            for linha in linhas:
                w.writerow([linha.get(c, "") for c in COLUNAS])
        return "﻿" + saida.getvalue()
