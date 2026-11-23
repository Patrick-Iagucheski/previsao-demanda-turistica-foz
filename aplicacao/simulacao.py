import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from aplicacao.preprocessamento import CORTE_TESTE
from dominio.entidades import BlocoTemporalEnum, RegistroMensal, SerieTemporal
from dominio.regras import bloco_temporal, eh_alta_temporada

# Serviço de histórico simulado (camada de aplicação): experimento complementar.
#
# Acrescenta, ANTES da série real, um histórico fictício com o mesmo
# comportamento da demanda, para medir o efeito do volume de dados sobre os
# modelos. O conjunto de teste continua sendo 2025 REAL. A série resultante é
# gravada numa base própria (bases.json), nunca na base real do experimento.
#
# Calibração: só com os meses de TREINO (antes de CORTE_TESTE), nunca com o
# teste (RN04, RN05):
#   - sazonalidade: índice mensal médio dos dois ciclos completos do treino
#     (mar/2019–fev/2020 e jan–dez/2024), normalizado para média 1;
#   - nível: média do ciclo mar/2019–fev/2020, recuado no tempo à taxa de
#     crescimento das chegadas a Foz do Iguaçu entre 1999 e 2009 medida por
#     Casanova et al. (2012, Tabela 3): de 764.917 para 866.242, ≈ 1,25% a.a.;
#   - ruído: desvio-padrão dos meses desses ciclos em relação a
#     nível × sazonalidade, aplicado como variação aleatória multiplicativa.
# A semente fixa torna a série simulada reprodutível (RNF01).
#
# Uso (cria a base "historico20" a partir da série da base real):
#     python aplicacao/simulacao.py

# Constante definida para a taxa de crescimento anual das chegadas a Foz
# (Casanova et al., 2012): de 764.917 (1999) para 866.242 (2009), cerca de 1,25%
# ao ano; usada no método gerar() para recuar o nível no tempo.
TAXA_ANUAL_CASANOVA = (866_242 / 764_917) ** (1 / 10) - 1
INICIO_REAL = (2019, 3)          # primeiro mês da base da ANAC usada no TCC
# Os dois ciclos completos de 12 meses do treino (mar/2019 a fev/2020 e jan a
# dez/2024), usados no método calibrar().
CICLOS_TREINO = (((2019, 3), (2020, 2)), ((2024, 1), (2024, 12)))

# Retorna a posição do mês numa contagem contínua (ano x 12 + mês - 1).
def _indice(ano: int, mes: int) -> int:
    return ano * 12 + mes - 1

# Gera o histórico fictício e o junta à série real.
class HistoricoSimuladoService:
    # Guarda a semente (para a simulação sair igual a cada vez), quantos anos
    # simular (padrão 20) e a taxa anual de crescimento.
    def __init__(self, semente: int = config.SEED, anos: int = 20,
                 taxa_anual: float = TAXA_ANUAL_CASANOVA):
        self.semente = semente
        self.anos = anos
        self.taxa_anual = taxa_anual

    # Parâmetros do gerador, estimados só nos ciclos completos do treino.
    # Como funciona: para cada mês do ano, calcula quanto ele fica acima ou
    # abaixo da média do seu ciclo (fator sazonal) e tira a média dos dois
    # ciclos; o nível é a média do ciclo 2019/20; o ruído é o desvio-padrão de
    # quanto cada mês real se afastou do esperado.
    def calibrar(self, serie_real: SerieTemporal) -> Dict[str, object]:
        por_mes = {(r.ano, r.mes): r.desembarques for r in serie_real.registros
                   if not r.pandemia and (r.ano, r.mes) < CORTE_TESTE}
        ciclos: List[List[Tuple[int, float]]] = []
        for inicio, fim in CICLOS_TREINO:
            ciclo = [(i % 12 + 1, por_mes.get((i // 12, i % 12 + 1)))
                     for i in range(_indice(*inicio), _indice(*fim) + 1)]
            if any(v is None for _, v in ciclo):
                raise ValueError(f"ciclo de treino incompleto: {inicio} a {fim}")
            ciclos.append(ciclo)

        fatores = {m: 0.0 for m in range(1, 13)}
        for ciclo in ciclos:
            media = statistics.fmean(v for _, v in ciclo)
            for mes, v in ciclo:
                fatores[mes] += v / media / len(ciclos)
        ajuste = statistics.fmean(fatores.values())
        fatores = {m: f / ajuste for m, f in fatores.items()}

        desvios = [v / (statistics.fmean(x for _, x in ciclo) * fatores[mes]) - 1
                   for ciclo in ciclos for mes, v in ciclo]
        return {
            "fatores": fatores,
            "nivel_2019": statistics.fmean(v for _, v in ciclos[0]),
            "centro_2019": _indice(*CICLOS_TREINO[0][0]) + 5.5,
            "ruido": statistics.stdev(desvios),
        }

    # Série real precedida de `anos` anos simulados (mar/1999 a fev/2019
    # com os valores padrão).
    # Como funciona: para cada mês simulado, recua o nível de 2019 pela taxa
    # anual (quanto mais antigo, menor), multiplica pelo fator sazonal do mês e
    # aplica uma variação aleatória; no fim, junta simulados e reais em ordem.
    def gerar(self, serie_real: SerieTemporal) -> SerieTemporal:
        p = self.calibrar(serie_real)
        rng = random.Random(self.semente)
        simulados = []
        primeiro = _indice(*INICIO_REAL)
        for i in range(primeiro - self.anos * 12, primeiro):
            ano, mes = i // 12, i % 12 + 1
            anos_atras = (p["centro_2019"] - i) / 12
            esperado = p["nivel_2019"] / (1 + self.taxa_anual) ** anos_atras * p["fatores"][mes]
            valor = max(0.0, esperado * (1 + rng.gauss(0, p["ruido"])))
            simulados.append(RegistroMensal(
                ano=ano, mes=mes, desembarques=float(round(valor)),
                alta_temporada=eh_alta_temporada(mes), pandemia=False,
                bloco_temporal=BlocoTemporalEnum(bloco_temporal(ano, mes)),
            ))
        reais = [RegistroMensal(r.ano, r.mes, r.desembarques, r.alta_temporada,
                                r.pandemia, r.bloco_temporal) for r in serie_real.registros]
        return SerieTemporal(simulados + reais).ordenar()


if __name__ == "__main__":
    from infraestrutura.repositorios.sqlite_repo import SqliteRepository

    base = next(b for b in config.ler_bases()["bases"] if b["id"] == "historico20")
    real = SqliteRepository(config.BANCO).carregar_serie()
    servico = HistoricoSimuladoService()
    p = servico.calibrar(real)
    simulada = servico.gerar(real)
    destino = SqliteRepository(config.RAIZ / base["arquivo"])
    destino.criar_esquema()
    destino.salvar_serie(simulada)
    print(f"taxa anual (Casanova et al., 2012): {servico.taxa_anual:.4%}")
    print(f"nível do ciclo mar/2019–fev/2020: {p['nivel_2019']:,.0f} | ruído: {p['ruido']:.4f}")
    print("fatores sazonais:", {m: round(f, 3) for m, f in p["fatores"].items()})
    print(f"base '{base['nome']}': {simulada.resumo()} -> {base['arquivo']}")
