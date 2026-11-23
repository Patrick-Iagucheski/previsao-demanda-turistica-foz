import json
import os
from pathlib import Path

# Configuração central do projeto: caminhos, aeroporto de interesse, colunas da
# base da ANAC usadas na extração (RN01) e semente de reprodutibilidade.
#
# As regras de negócio do domínio (RN02 pandemia, RN03 alta temporada e blocos
# temporais) ficam em dominio/regras.py, e não aqui.

# Caminhos

RAIZ = Path(__file__).resolve().parent  # pasta raiz do projeto (onde está este arquivo)
# Pasta padrão dos CSV da ANAC dentro do projeto; é o valor padrão de ANAC_ORIGEM.
DADOS_BRUTOS = RAIZ / "dados" / "brutos"

# Onde estão os CSV originais da ANAC. Padrão: dados/brutos, dentro do projeto.
# Outra pasta pode ser indicada pela variável de ambiente ANAC_ORIGEM
# (na máquina do autor: C:\Dados_Anac), sem editar o código, ou digitada no
# campo "Pasta dos CSV" da tela de importação.
ANAC_ORIGEM = Path(os.environ.get("ANAC_ORIGEM") or DADOS_BRUTOS)

# Banco SQLite da base real (série e execuções), usado pelo SqliteRepository quando
# nenhum outro caminho é informado.
BANCO = RAIZ / "dados" / "experimentos.db"

# Modelos treinados e serializados (ds_caminho_modelo aponta para cá).
MODELOS = RAIZ / "dados" / "modelos"

# Bases de dados selecionáveis na interface: a real e as de experimentos
# complementares (ex.: histórico simulado). Cada base é um arquivo SQLite próprio.
BASES = RAIZ / "bases.json"

# Lista das bases e a base padrão, lidas de bases.json.
def ler_bases() -> dict:
    return json.loads(BASES.read_text(encoding="utf-8"))

# Aeroporto de interesse

IATA = "IGU"  # código IATA do aeroporto de Foz do Iguaçu
ICAO = "SBFI"          # código usado na coluna NR_AEROPORTO_REFERENCIA
AEROPORTO_NOME = "Aeroporto Internacional de Foz do Iguaçu"  # nome exibido nas telas e gravado no banco

# Extração da base da ANAC

# RN01 (variável-alvo): desembarques de passageiros em Foz, em voos domésticos.
#
# Método adotado ("outro aeroporto"): um voo que DECOLA (D) de outro aeroporto da base
# com destino a SBFI carrega os passageiros que vão desembarcar em Foz. Isso
# funciona em TODOS os anos, pois não exige Foz como aeroporto de referência
# (que só passa a existir na base em abr/2022). Validado contra o método direto
# (Foz-referência + pouso "P") de abr/2022 a dez/2025: 98,28% de concordância.
# A diferença vem dos voos internacionais (cerca de 2%), cuja origem não está na base.
MOVIMENTO_DECOLAGEM_ORIGEM = "D"    # decolagem no aeroporto de origem, rumo a Foz
MOVIMENTO_DESEMBARQUE = "P"         # (método direto, usado só para validação)
# Colunas de passageiros somadas para formar os desembarques de cada voo (usadas no
# AnacCsvLoader e na validação da extração).
COLUNAS_PAX = [
    "QT_PAX_LOCAL",
    "QT_PAX_CONEXAO_DOMESTICO",
    "QT_PAX_CONEXAO_INTERNACIONAL",
]

# Reprodutibilidade (RNF01 / RF20)

SEED = 42  # semente padrão dos treinamentos: com ela o resultado se repete
