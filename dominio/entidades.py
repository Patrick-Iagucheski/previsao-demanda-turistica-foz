from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple

from dominio.regras import eh_alta_temporada, mes_ciclico

# Entidades do domínio. Não dependem de pandas, banco ou biblioteca de ML:
# são o núcleo do sistema (Clean Architecture).
#
# Segue o Diagrama de Classes: Arquitetura do Sistema (Seção 4.3.3 do TCC;
# docs/diagramas/06_classes_arquitetura_planejada.puml). Os valores dos enums
# são exatamente os aceitos pelas restrições CHECK do esquema físico
# (infraestrutura/repositorios/schema.sql).


# Enumerações (domínios fechados, que espelham os CHECK do esquema físico)

# Os três modelos comparados: naïve sazonal (RF11), Random Forest (RF12) e LSTM (RF13).
class TipoModeloEnum(str, Enum):
    NAIVE = "naive"
    RANDOM_FOREST = "random_forest"
    LSTM = "lstm"

# Bloco temporal de cada mês, calculado por bloco_temporal() em regras.py:
# 0 = dentro da pandemia, 1 = pré-pandemia, 2 = pós-pandemia.
class BlocoTemporalEnum(IntEnum):
    PANDEMIA = 0
    PRE_PANDEMIA = 1
    POS_PANDEMIA = 2

# Escopo em que as métricas são calculadas: os 12 meses do teste (global) ou só
# janeiro, julho e dezembro (alta temporada, RN03).
class EscopoAvaliacaoEnum(str, Enum):
    GLOBAL = "global"
    ALTA_TEMPORADA = "alta_temporada"

# Métricas de erro calculadas pelo AvaliacaoService (RF16).
class TipoMetricaEnum(str, Enum):
    MAE = "MAE"
    RMSE = "RMSE"
    MAPE = "MAPE"


# Série temporal

# Um mês da série de desembarques do aeroporto.
@dataclass
class RegistroMensal:
    ano: int
    mes: int
    desembarques: float
    alta_temporada: bool = False
    pandemia: bool = False
    # Segmento temporal do mês, tipado pela enumeração do diagrama de classes
    # (Seção 4.3.3 do TCC). Por ser IntEnum, grava e compara como inteiro (0/1/2).
    bloco_temporal: BlocoTemporalEnum = BlocoTemporalEnum.PANDEMIA
    # Atributos derivados (RF08). Vazio até gerar_atributos() ser chamado e
    # nos meses sem defasagens completas dentro do próprio bloco.
    atributos: Dict[str, float] = field(default_factory=dict)
    # Marca os meses inseridos por interpolação intra-bloco (RF07).
    interpolado: bool = False

    # Retorna o mês em texto no formato "AAAA-MM" (ex.: "2025-01"), usado nas telas.
    @property
    def rotulo(self) -> str:
        return f"{self.ano}-{self.mes:02d}"

    # Retorna a posição do mês numa contagem mensal contínua (ano x 12 + mês - 1);
    # facilita medir lacunas: a diferença entre dois meses é o número de meses entre eles.
    @property
    def indice_absoluto(self) -> int:
        return self.ano * 12 + (self.mes - 1)

# A série mensal completa, como lista ordenada de RegistroMensal.
@dataclass
class SerieTemporal:
    registros: List[RegistroMensal] = field(default_factory=list)

    # Ordena os registros por ano e mês e retorna a própria série, para encadear chamadas.
    def ordenar(self):
        self.registros.sort(key=lambda r: (r.ano, r.mes))
        return self

    # Retorna o número de meses da série (é o que permite usar len(serie)).
    def __len__(self):
        return len(self.registros)

    # Retorna um resumo em texto: quantidade de meses, primeiro e último mês da série.
    def resumo(self) -> str:
        if not self.registros:
            return "série vazia"
        self.ordenar()
        p = self.registros[0].rotulo
        u = self.registros[-1].rotulo
        return f"{len(self)} meses ({p} a {u})"

    # Pré-processamento (RF06 a RF08, RF10)

    # RF06: remove o intervalo pandêmico (RN02), preservando os dois
    # blocos como segmentos independentes. Devolve uma nova série.
    # Como funciona: mantém só os meses que não estão marcados como pandemia
    # (a marca é feita na importação, por eh_pandemia() de regras.py).
    def excluir_pandemia(self) -> "SerieTemporal":
        self.ordenar()
        return SerieTemporal([r for r in self.registros if not r.pandemia])

    # Agrupa os registros por bloco temporal, cada grupo já ordenado.
    # Retorna um dicionário {bloco: lista de meses}: 1 = antes da pandemia,
    # 2 = depois da pandemia.
    def blocos(self) -> Dict[int, List[RegistroMensal]]:
        self.ordenar()
        agrupados: Dict[int, List[RegistroMensal]] = {}
        for r in self.registros:
            agrupados.setdefault(r.bloco_temporal, []).append(r)
        return agrupados

    # RF07: preenche meses ausentes por interpolação linear *dentro de
    # cada bloco*, sem atravessar o intervalo excluído (RNF03).
    # Como funciona: dentro de cada bloco, compara cada mês com o seguinte; se
    # houver meses faltando entre os dois, cria esses meses com valores em linha
    # reta entre o anterior e o seguinte e os marca como interpolados.
    def interpolar_por_bloco(self) -> "SerieTemporal":
        completos: List[RegistroMensal] = []
        for _, registros in sorted(self.blocos().items()):
            completos.append(registros[0])
            for anterior, atual in zip(registros, registros[1:]):
                lacuna = atual.indice_absoluto - anterior.indice_absoluto
                for passo in range(1, lacuna):
                    indice = anterior.indice_absoluto + passo
                    ano, mes = divmod(indice, 12)
                    mes += 1
                    proporcao = passo / lacuna
                    valor = (anterior.desembarques
                             + proporcao * (atual.desembarques - anterior.desembarques))
                    completos.append(RegistroMensal(
                        ano=ano,
                        mes=mes,
                        desembarques=valor,
                        alta_temporada=eh_alta_temporada(mes),
                        pandemia=False,
                        bloco_temporal=anterior.bloco_temporal,
                        interpolado=True,
                    ))
                completos.append(atual)
        return SerieTemporal(completos).ordenar()

    # RF08: gera mês do ano, indicador de alta temporada e defasagens.
    #
    # As defasagens usam apenas valores passados (RN06) e são buscadas
    # dentro do mesmo bloco: nenhuma janela atravessa a fronteira entre os
    # blocos (RNF03). Meses sem histórico suficiente no próprio bloco ficam
    # com `atributos` vazio e são descartados da matriz de aprendizado.
    #
    # O mês é codificado de forma cíclica (seno e cosseno) em vez de uma
    # variável indicadora por mês: com a amostra reduzida (43 meses; Seção 3.5.1), doze
    # colunas binárias consumiriam graus de liberdade que a série não tem.
    #
    # Como funciona: para cada mês, busca no mesmo bloco os desembarques de 1, 2
    # e 3 meses antes (lag_1, lag_2, lag_3); se faltar algum, o mês fica sem
    # atributos. Depois acrescenta o seno e o cosseno do mês (mes_ciclico()) e o
    # indicador de alta temporada (1 em jan, jul e dez; 0 nos demais).
    def gerar_atributos(self, lags: Tuple[int, ...] = (1, 2, 3)) -> "SerieTemporal":
        for registros in self.blocos().values():
            por_indice = {r.indice_absoluto: r for r in registros}
            for r in registros:
                defasagens = {}
                completo = True
                for lag in lags:
                    vizinho = por_indice.get(r.indice_absoluto - lag)
                    if vizinho is None:
                        completo = False
                        break
                    defasagens[f"lag_{lag}"] = vizinho.desembarques
                if not completo:
                    r.atributos = {}
                    continue
                seno, cosseno = mes_ciclico(r.mes)
                r.atributos = {
                    **defasagens,
                    "mes_sen": seno,
                    "mes_cos": cosseno,
                    "fl_alta_temporada": float(r.alta_temporada),
                }
        return self

    # Retorna só os meses com atributos completos: são as linhas usadas para
    # treinar e prever (os primeiros 3 meses de cada bloco ficam de fora).
    def com_atributos(self) -> List[RegistroMensal]:
        self.ordenar()
        return [r for r in self.registros if r.atributos]

    # RF10: particiona por critério estritamente cronológico (RN04).
    #
    # `corte` é o primeiro mês do conjunto de teste, no formato (ano, mês);
    # tudo anterior compõe o treino. Não há embaralhamento.
    # Como funciona: converte o corte na contagem contínua de meses; os meses
    # antes dele vão para o treino e os demais para o teste.
    def dividir_treino_teste(
        self, corte: Tuple[int, int]
    ) -> Tuple["SerieTemporal", "SerieTemporal"]:
        self.ordenar()
        limite = corte[0] * 12 + (corte[1] - 1)
        treino = [r for r in self.registros if r.indice_absoluto < limite]
        teste = [r for r in self.registros if r.indice_absoluto >= limite]
        return SerieTemporal(treino), SerieTemporal(teste)


# Experimento

# Um valor de erro, num escopo de avaliação (RF16).
@dataclass
class Metrica:
    escopo_avaliacao: EscopoAvaliacaoEnum
    tipo_metrica: TipoMetricaEnum
    valor_metrica: float

# Previsão de um mês, com o valor real materializado ao lado (RF15).
@dataclass
class Previsao:
    ano: int
    mes: int
    valor_previsto: float
    valor_real: Optional[float] = None
    alta_temporada: bool = False

    # Retorna o mês previsto em texto no formato "AAAA-MM".
    @property
    def rotulo(self) -> str:
        return f"{self.ano}-{self.mes:02d}"

    # Retorna o erro da previsão (real menos previsto); None quando ainda não há valor real.
    @property
    def erro(self) -> Optional[float]:
        if self.valor_real is None:
            return None
        return self.valor_real - self.valor_previsto

# Um treinamento completo, com suas previsões e métricas (RF19).
#
# É a unidade de rastreabilidade do experimento: guarda o modelo, a semente
# e o período de teste, de modo que qualquer número citado na monografia
# possa ser reproduzido (RNF01).
@dataclass
class Execucao:
    tipo_modelo: TipoModeloEnum
    semente_aleatoria: int
    id: Optional[int] = None
    hiperparametros: Dict = field(default_factory=dict)
    data_hora_execucao: datetime = field(default_factory=datetime.now)
    tamanho_janela_entrada: Optional[int] = None
    periodo_teste: str = ""
    caminho_modelo: str = ""
    previsoes: List[Previsao] = field(default_factory=list)
    metricas: List[Metrica] = field(default_factory=list)

    # Busca pontual de uma métrica já calculada: retorna o valor pedido (ex.: MAPE
    # na alta temporada) ou None se a execução não tiver essa métrica.
    def metrica(self, escopo: EscopoAvaliacaoEnum, tipo: TipoMetricaEnum):
        for m in self.metricas:
            if m.escopo_avaliacao == escopo and m.tipo_metrica == tipo:
                return m.valor_metrica
        return None
