# -*- coding: utf-8 -*-
"""
Serviço de treinamento (camada de aplicação) — RF10 a RF16 e RF19.

Executa um experimento completo para um modelo, na ordem do Diagrama de
Sequência "Treinar e avaliar" (docs/diagramas/13_sequencia_treinar.puml):

    PreProcessamentoService.preparar(serie)
    PreProcessamentoService.dividir(serie preparada) -> (treino, teste)
    IForecaster.fit(treino)
    IForecaster.salvar(caminho_modelo)
    IForecaster.predict(teste)
    AvaliacaoService.avaliar(previsoes)   # métricas nos escopos global e alta temporada
    IExecucaoRepository.salvar_execucao(execucao)
    IExecucaoRepository.salvar_previsoes(id, previsoes)

O teste de Diebold-Mariano não faz parte deste fluxo: ele compara execuções já
registradas (tela de comparação e notebook de resultados).

O serviço não conhece os modelos concretos: recebe uma fábrica que associa
cada TipoModeloEnum a um construtor de IForecaster. Quem monta essa fábrica
é o ponto de composição (a aplicação Flask ou o notebook de resultados). É
o que mantém a camada de aplicação livre de dependências da infraestrutura
(Ports & Adapters, Seção 4.6.2 do TCC).

Todos os modelos recebem o mesmo pré-processamento e são avaliados sobre o
mesmo horizonte de teste, condição da comparação justa (RN07).
"""
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from aplicacao.avaliacao import AvaliacaoService
from aplicacao.preprocessamento import PreProcessamentoService
from dominio.entidades import Execucao, Previsao, SerieTemporal, TipoModeloEnum
from dominio.portas import IExecucaoRepository, IForecaster

FabricaModelos = Dict[TipoModeloEnum, Callable[..., IForecaster]]


class TreinamentoService:

    def __init__(
        self,
        fabrica: FabricaModelos,
        repositorio: Optional[IExecucaoRepository] = None,
        preprocessamento: Optional[PreProcessamentoService] = None,
        avaliacao: Optional[AvaliacaoService] = None,
        pasta_modelos: Optional[Path] = None,
    ):
        self.fabrica = fabrica
        self.repositorio = repositorio
        self.preprocessamento = preprocessamento or PreProcessamentoService()
        self.avaliacao = avaliacao or AvaliacaoService()
        self.pasta_modelos = Path(pasta_modelos or config.MODELOS)

    def executar(
        self,
        serie: SerieTemporal,
        modelo: TipoModeloEnum,
        seed: int = config.SEED,
        hiperparametros: Optional[dict] = None,
        persistir: bool = True,
    ) -> Execucao:
        """Treina, prevê, avalia e (opcionalmente) registra uma execução.

        `persistir=False` executa sem gravar no banco — útil para conferências
        que não devem entrar no histórico do experimento.
        """
        tipo = TipoModeloEnum(modelo)
        forecaster = self.fabrica[tipo](semente=seed, **(hiperparametros or {}))

        # RF06–RF08 e RF10
        preparada = self.preprocessamento.preparar(serie)
        treino, teste = self.preprocessamento.dividir(preparada)
        # O horizonte avaliado é o mesmo para todos os modelos (RN07): os meses
        # de teste com a matriz de atributos completa.
        horizonte = SerieTemporal(teste.com_atributos())

        # RF11–RF13: ajuste e serialização
        forecaster.fit(treino)
        carimbo = datetime.now()
        extensao = getattr(forecaster, "extensao", ".bin")
        caminho = self.pasta_modelos / f"{tipo.value}_{carimbo:%Y%m%d-%H%M%S}{extensao}"
        if persistir:
            forecaster.salvar(caminho)

        # RF15: previsões, com o valor real materializado ao lado
        valores = forecaster.predict(horizonte)
        previsoes = [
            Previsao(r.ano, r.mes, float(v), r.desembarques, r.alta_temporada)
            for r, v in zip(horizonte.registros, valores)
        ]

        # RF16: métricas nos dois escopos
        metricas = self.avaliacao.avaliar(previsoes)

        regs = horizonte.registros
        execucao = Execucao(
            tipo_modelo=tipo,
            semente_aleatoria=seed,
            hiperparametros=forecaster.hiperparametros(),
            data_hora_execucao=carimbo,
            tamanho_janela_entrada=getattr(forecaster, "tamanho_janela_entrada", None),
            periodo_teste=f"{regs[0].rotulo}..{regs[-1].rotulo}" if regs else "",
            caminho_modelo=str(caminho) if persistir else "",
            previsoes=previsoes,
            metricas=metricas,
        )

        # RF19: registro do experimento
        if persistir and self.repositorio is not None:
            id_execucao = self.repositorio.salvar_execucao(execucao)
            self.repositorio.salvar_previsoes(id_execucao, previsoes)
        return execucao
