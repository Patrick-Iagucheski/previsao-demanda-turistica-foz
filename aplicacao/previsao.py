# -*- coding: utf-8 -*-
"""
Serviço de previsão (camada de aplicação) — RF15 e RF19.

    prever(execucao, periodo) -> List[Previsao]

Recarrega o modelo serializado de uma execução registrada e prevê o período
pedido **sem novo treinamento**, na ordem do Diagrama de Sequência "Prever"
(docs/diagramas/14_sequencia_prever.puml). É o que dá sentido ao campo
ds_caminho_modelo: sem ele, qualquer previsão exigiria repetir o treino.

A série é preparada pelo mesmo PreProcessamentoService usado no treinamento,
de modo que o modelo recarregado recebe exatamente os atributos com que foi
ajustado. Os modelos concretos chegam por uma fábrica injetada, como no
TreinamentoService (Ports & Adapters).
"""
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

from aplicacao.preprocessamento import PreProcessamentoService
from dominio.entidades import Execucao, Previsao, SerieTemporal, TipoModeloEnum
from dominio.portas import IForecaster, ISerieRepository


def intervalo_do_periodo(periodo: str) -> Tuple[int, int]:
    """'2025-01..2025-12' -> (índice absoluto inicial, final), inclusivos."""
    inicio, fim = (p.strip() for p in periodo.split(".."))
    a1, m1 = (int(x) for x in inicio.split("-"))
    a2, m2 = (int(x) for x in fim.split("-"))
    return a1 * 12 + (m1 - 1), a2 * 12 + (m2 - 1)


class PrevisaoService:

    def __init__(self, fabrica: Dict[TipoModeloEnum, Callable[..., IForecaster]],
                 repositorio_serie: ISerieRepository,
                 preprocessamento: Optional[PreProcessamentoService] = None):
        self.fabrica = fabrica
        self.repositorio_serie = repositorio_serie
        self.preprocessamento = preprocessamento or PreProcessamentoService()

    def prever(self, execucao: Execucao, periodo: Optional[str] = None) -> List[Previsao]:
        """Previsões do modelo salvo para o período (padrão: o período de
        teste da própria execução), com o valor real ao lado quando houver."""
        caminho = Path(execucao.caminho_modelo or "")
        if not execucao.caminho_modelo or not caminho.exists():
            raise FileNotFoundError(
                f"modelo da execução {execucao.id} não encontrado em disco: "
                f"{execucao.caminho_modelo or '(sem caminho registrado)'}")

        tipo = TipoModeloEnum(execucao.tipo_modelo)
        modelo = self.fabrica[tipo](semente=execucao.semente_aleatoria)
        modelo.carregar(caminho)

        preparada = self.preprocessamento.preparar(self.repositorio_serie.carregar_serie())
        inicio, fim = intervalo_do_periodo(periodo or execucao.periodo_teste)
        horizonte = SerieTemporal([r for r in preparada.com_atributos()
                                   if inicio <= r.indice_absoluto <= fim])
        valores = modelo.predict(horizonte)
        return [Previsao(r.ano, r.mes, float(v), r.desembarques, r.alta_temporada)
                for r, v in zip(horizonte.registros, valores)]
