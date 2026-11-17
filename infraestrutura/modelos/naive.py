# -*- coding: utf-8 -*-
"""
Baseline naïve sazonal — adaptador de IForecaster (RF11).

Prevê, para cada mês, o valor observado no mesmo mês do ano anterior. É uma
referência simples de desempenho: permite avaliar se o Random Forest e a LSTM
reduzem o erro em relação à repetição sazonal dos valores históricos
(Hyndman; Athanasopoulos, 2021). Não é um piso — um modelo pode errar mais do
que ele (Seção 3.7 do TCC).

Não há treinamento nem hiperparâmetros a ajustar: o modelo é aplicado como
referência determinística; fit() apenas memoriza a série observada até o fim
do treino.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dominio.entidades import SerieTemporal, TipoModeloEnum
from dominio.portas import IForecaster

DEFASAGEM_SAZONAL = 12


class NaiveSazonal(IForecaster):

    extensao = ".json"

    def __init__(self, semente: int = None):
        # Determinístico: a semente é aceita só para manter a mesma assinatura
        # dos demais modelos na fábrica do TreinamentoService.
        self._memoria: Dict[Tuple[int, int], float] = {}

    def nome(self) -> str:
        return TipoModeloEnum.NAIVE.value

    def hiperparametros(self) -> dict:
        return {"defasagem_sazonal": DEFASAGEM_SAZONAL}

    def fit(self, treino: SerieTemporal) -> None:
        # Usa todos os meses do treino, inclusive os que não têm a matriz de
        # atributos completa: o naïve não depende das defasagens curtas.
        self._memoria = {(r.ano, r.mes): r.desembarques for r in treino.registros}

    def predict(self, horizonte: SerieTemporal) -> List[float]:
        previsoes = []
        for r in horizonte.registros:
            chave = (r.ano - 1, r.mes)
            if chave not in self._memoria:
                raise ValueError(
                    f"naïve sazonal: {r.rotulo} exige o valor de "
                    f"{chave[0]}-{chave[1]:02d}, ausente do treino — o mesmo mês "
                    f"do ano anterior caiu no intervalo excluído ou fora da série"
                )
            previsoes.append(self._memoria[chave])
        return previsoes

    def salvar(self, caminho_modelo) -> None:
        caminho = Path(caminho_modelo)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        conteudo = [{"ano": a, "mes": m, "desembarques": v}
                    for (a, m), v in sorted(self._memoria.items())]
        caminho.write_text(json.dumps(conteudo, ensure_ascii=False, indent=1),
                           encoding="utf-8")

    def carregar(self, caminho_modelo) -> None:
        conteudo = json.loads(Path(caminho_modelo).read_text(encoding="utf-8"))
        self._memoria = {(x["ano"], x["mes"]): x["desembarques"] for x in conteudo}
