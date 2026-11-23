import math
from typing import Tuple

# Regras de negócio do domínio (RN02 e RN03) — fonte única para todo o sistema.
# Ficam no domínio, e não em config.py, porque são regras do negócio e não
# configuração (Martin, 2019, cap. 20; Percival; Gregory, 2020, cap. 1): o
# intervalo pandêmico excluído, os meses de alta temporada, o bloco temporal de
# cada mês e a codificação cíclica do mês usada como atributo pelos modelos.
# Como o restante do domínio, não depende de biblioteca externa.

# RN02 — período pandêmico excluído (da caracterização da pandemia ao fim da ESPII, OPAS): 11/03/2020 a 05/05/2023.
# Granularidade mensal: excluem-se mar/2020 a mai/2023 (inclusive).
# Constantes definidas para o período pandêmico, usadas no método eh_pandemia(), retornando true quando o mês está dentro do período pandêmico
PANDEMIA_INICIO = (2020, 3)     # (ano, mês)
PANDEMIA_FIM = (2023, 5)

# RN03 — alta temporada (calendário escolar SEED-PR): janeiro, julho e dezembro.
# Constante definida para alta temporada, usada no método eh_alta_temporada(), retornando true quando é alta temporada
MESES_ALTA_TEMPORADA = frozenset({1, 7, 12})

# True se (ano, mês) cai dentro do intervalo pandêmico excluído.
def eh_pandemia(ano: int, mes: int) -> bool:
    return PANDEMIA_INICIO <= (ano, mes) <= PANDEMIA_FIM

# Retorna true se o mês é de alta temporada (janeiro, julho ou dezembro).
def eh_alta_temporada(mes: int) -> bool:
    return mes in MESES_ALTA_TEMPORADA

# Retorna o bloco temporal do mês: 1 = pré-pandemia · 2 = pós-pandemia · 0 = dentro da pandemia.
def bloco_temporal(ano: int, mes: int) -> int:
    if eh_pandemia(ano, mes):
        return 0
    return 1 if (ano, mes) < PANDEMIA_INICIO else 2

# Codificação cíclica do mês (seno e cosseno) usada como atributo informado aos modelos pelos atributos de calendário 
# (codificação cíclica do RF08) dezembro fica vizinho de janeiro o que uma numeração de 1 a 12 não expressa.
def mes_ciclico(mes: int) -> Tuple[float, float]:
    angulo = 2 * math.pi * mes / 12
    return math.sin(angulo), math.cos(angulo)
