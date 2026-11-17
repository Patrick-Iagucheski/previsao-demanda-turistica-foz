# -*- coding: utf-8 -*-
"""
Configuração central do projeto: caminhos, aeroporto de interesse, colunas da
base da ANAC usadas na extração (RN01) e semente de reprodutibilidade.

As regras de negócio do domínio (RN02 pandemia, RN03 alta temporada e blocos
temporais) ficam em dominio/regras.py, e não aqui.
"""
import json
from pathlib import Path

# ----------------------------------------------------------------------
# Caminhos
# ----------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent
DADOS_BRUTOS = RAIZ / "dados" / "brutos"

# Onde estão os CSV originais da ANAC (fora do projeto, no OneDrive).
# O loader copia/lê a partir daqui.
ANAC_ORIGEM = Path(
    r"C:\Dados_Anac"
)

BANCO = RAIZ / "dados" / "experimentos.db"

# Modelos treinados e serializados (ds_caminho_modelo aponta para cá).
MODELOS = RAIZ / "dados" / "modelos"

# Bases de dados selecionáveis na interface: a real e as de experimentos
# complementares (ex.: histórico simulado). Cada base é um arquivo SQLite próprio.
BASES = RAIZ / "bases.json"


def ler_bases() -> dict:
    """Lista das bases e a base padrão, lidas de bases.json."""
    return json.loads(BASES.read_text(encoding="utf-8"))

# ----------------------------------------------------------------------
# Aeroporto de interesse
# ----------------------------------------------------------------------
IATA = "IGU"
ICAO = "SBFI"          # código usado na coluna NR_AEROPORTO_REFERENCIA
AEROPORTO_NOME = "Aeroporto Internacional de Foz do Iguaçu"

# ----------------------------------------------------------------------
# Extração da base da ANAC
# ----------------------------------------------------------------------
# RN01 — variável-alvo: desembarques de passageiros em Foz, em voos domésticos.
#
# Método adotado ("outro aeroporto"): um voo que DECOLA (D) de outro aeroporto da base
# com destino a SBFI carrega os passageiros que vão desembarcar em Foz. Isso
# funciona em TODOS os anos, pois não exige Foz como aeroporto de referência
# (que só passa a existir na base em abr/2022). Validado contra o método direto
# (Foz-referência + pouso "P") de abr/2022 a dez/2025: 98,28% de concordância.
# A diferença vem dos voos internacionais (cerca de 2%), cuja origem não está na base.
MOVIMENTO_DECOLAGEM_ORIGEM = "D"    # decolagem no aeroporto de origem, rumo a Foz
MOVIMENTO_DESEMBARQUE = "P"         # (método direto, usado só para validação)
COLUNAS_PAX = [
    "QT_PAX_LOCAL",
    "QT_PAX_CONEXAO_DOMESTICO",
    "QT_PAX_CONEXAO_INTERNACIONAL",
]

# ----------------------------------------------------------------------
# Reprodutibilidade (RNF01 / RF20)
# ----------------------------------------------------------------------
SEED = 42
