# -*- coding: utf-8 -*-
"""
Portas do domínio — os contratos que as demais camadas enxergam.

São interfaces abstratas, sem dependência de biblioteca externa: é o que
permite trocar a biblioteca de um modelo (scikit-learn, TensorFlow) ou o
SQLite pelo PostgreSQL sem tocar no núcleo (Martin, 2019; Percival; Gregory,
2020; Seção 4.6.2 do TCC). As métricas e o teste de Diebold-Mariano não
passam por porta: o AvaliacaoService usa scikit-learn, statsmodels e SciPy
diretamente, por serem cálculo, e não entrada e saída.

As assinaturas seguem o Diagrama de Classes: Arquitetura Implementada
(Seção 4.3.4 do TCC; docs/diagramas/06_classes_arquitetura_planejada.puml).

Adaptadores concretos:
    IForecaster          -> infraestrutura/modelos/{naive,random_forest,lstm}.py
    ISerieRepository     -> infraestrutura/repositorios/sqlite_repo.py
    IExecucaoRepository  -> infraestrutura/repositorios/sqlite_repo.py
"""
from abc import ABC, abstractmethod
from typing import List


class IForecaster(ABC):
    """Contrato comum aos três modelos comparados (RF11, RF12, RF13).

    Tratar os três pela mesma interface é o que torna a comparação justa
    (RN07): o serviço de treinamento não sabe qual modelo está executando, de
    modo que todos recebem o mesmo pré-processamento e a mesma partição.
    """

    @abstractmethod
    def fit(self, treino) -> None:
        """Ajusta o modelo sobre a partição de treino."""

    @abstractmethod
    def predict(self, horizonte) -> List[float]:
        """Devolve uma previsão por registro do horizonte informado."""

    @abstractmethod
    def salvar(self, caminho_modelo) -> None:
        """Serializa o modelo ajustado (RF19: prever depois sem retreinar)."""

    @abstractmethod
    def carregar(self, caminho_modelo) -> None:
        """Recarrega um modelo previamente serializado."""

    @abstractmethod
    def nome(self) -> str:
        """Identificador do modelo, entre os valores de TipoModeloEnum."""

    def hiperparametros(self) -> dict:
        """Hiperparâmetros efetivamente usados, para registro da execução
        (RF19 / RNF01: reprodutibilidade)."""
        return {}


class ISerieRepository(ABC):
    """Persistência da série mensal de desembarques."""

    @abstractmethod
    def salvar_serie(self, serie) -> None:
        ...

    @abstractmethod
    def carregar_serie(self):
        ...


class IExecucaoRepository(ABC):
    """Persistência dos experimentos — o que torna o estudo auditável
    (RF19, RF21: cada métrica rastreável até a execução e a semente).

    Como Execucao compõe suas Metricas (composição no diagrama de classes),
    salvar_execucao grava também as métricas associadas.
    """

    @abstractmethod
    def salvar_execucao(self, execucao) -> int:
        """Grava a execução e suas métricas; devolve o id atribuído."""

    @abstractmethod
    def salvar_previsoes(self, id_execucao: int, previsoes) -> None:
        ...

    @abstractmethod
    def listar_execucoes(self) -> List:
        ...

    @abstractmethod
    def buscar_execucao(self, id_execucao: int):
        ...
