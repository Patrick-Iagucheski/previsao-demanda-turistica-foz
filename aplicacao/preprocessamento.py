import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dominio.entidades import RegistroMensal, SerieTemporal

# Serviço de pré-processamento (camada de aplicação): RF06 a RF08 e RF10.
#
# Orquestra, nesta ordem, as operações que o domínio expõe em SerieTemporal:
#
#     excluir_pandemia()      corte efetivo do intervalo da RN02
#     interpolar_por_bloco()  preenchimento intra-bloco, sem cruzar o corte
#     gerar_atributos()       mês cíclico, alta temporada e defasagens
#     dividir_treino_teste()  partição estritamente cronológica (RN04)
#
# O serviço não calcula nada por conta própria: ele apenas encadeia as
# operações do domínio e devolve o material pronto para os modelos. A
# normalização exigida pela LSTM é feita no próprio adaptador
# (infraestrutura/modelos/lstm.py), também estimada só no treino (RN05),
# porque a infraestrutura não pode depender da camada de aplicação. Por isso
# não importa scikit-learn nem TensorFlow; as bibliotecas de modelagem ficam
# nos adaptadores de infraestrutura/modelos/.
#
# Decisão sobre as defasagens (registrada aqui porque afeta a leitura dos
# resultados): o bloco pré-pandêmico tem exatamente 12 meses. Com defasagem de
# 12 períodos e sem atravessar a fronteira entre blocos (RNF03), esse bloco não
# produz nenhuma linha, e a matriz de aprendizado cai de 37 para 19 observações,
# das quais apenas 7 sobram para treino. Por isso a sazonalidade anual é
# informada aos modelos pelos atributos de calendário (mês em codificação
# cíclica e indicador de alta temporada), conforme previsto na Seção 3.7 do
# TCC, e não por uma defasagem de doze meses. O baseline naïve sazonal é a
# exceção: ele usa o mesmo mês do ano anterior por definição (RF11).
#
# Execução isolada (relatório do pré-processamento sobre o banco):
#     python aplicacao/preprocessamento.py

# Defasagens 
# Número de meses anteriores a considerar
# Usada no método preparar(), que a repassa para gerar_atributos() da SerieTemporal
LAGS_PADRAO: Tuple[int, ...] = (1, 2, 3) 

# Primeiro mês do conjunto de teste. O ano inteiro de 2025 é reservado para
# teste por cobrir os três meses de alta temporada (RN03) numa única
# partição cronológica, usada uma só vez ao final (RN04).
# Constante usada no método dividir().
CORTE_TESTE: Tuple[int, int] = (2025, 1)

# Prepara a série bruta para o experimento: encadeia, na ordem, os passos do
# domínio (excluir a pandemia, interpolar, gerar atributos e dividir).
class PreProcessamentoService:
    # Guarda as defasagens e o mês de corte do teste; self.relatorio guarda os
    # números de cada etapa, para conferência.
    def __init__(self, lags: Tuple[int, ...] = LAGS_PADRAO,
                 corte_teste: Tuple[int, int] = CORTE_TESTE):
        self.lags = lags
        self.corte_teste = corte_teste
        self.relatorio: Dict[str, object] = {}

    # RF06 a RF08: devolve a série sem o intervalo pandêmico,
    # interpolada dentro de cada bloco e com os atributos derivados.
    # Como funciona: chama excluir_pandemia(), interpolar_por_bloco() e
    # gerar_atributos() da SerieTemporal e anota em self.relatorio quantos meses
    # sobraram em cada etapa.
    def preparar(self, serie: SerieTemporal) -> SerieTemporal:
        bruta = len(serie)
        sem_pandemia = serie.excluir_pandemia()
        interpolada = sem_pandemia.interpolar_por_bloco()
        interpolada.gerar_atributos(lags=self.lags)

        self.relatorio = {
            "meses_brutos": bruta,
            "meses_apos_corte": len(sem_pandemia),
            "meses_interpolados": sum(1 for r in interpolada.registros if r.interpolado),
            "blocos": {b: len(rs) for b, rs in sorted(interpolada.blocos().items())},
            "linhas_matriz": len(interpolada.com_atributos()),
            "lags": self.lags,
        }
        return interpolada

    # RF10: partição cronológica no corte configurado (RN04).
    # Como funciona: tudo antes de jan/2025 vira treino e 2025 vira teste, sem
    # embaralhar; anota no relatório quantas linhas ficaram em cada parte.
    def dividir(self, serie: SerieTemporal) -> Tuple[SerieTemporal, SerieTemporal]:
        treino, teste = serie.dividir_treino_teste(self.corte_teste)
        self.relatorio["treino"] = len(treino.com_atributos())
        self.relatorio["teste"] = len(teste.com_atributos())
        return treino, teste

    # Ordem canônica das colunas da matriz de atributos: retorna os nomes
    # (lag_1, lag_2, lag_3, mes_sen, mes_cos, fl_alta_temporada), lidos do
    # primeiro mês com atributos.
    @staticmethod
    def nomes_atributos(serie: SerieTemporal) -> List[str]:
        linhas = serie.com_atributos()
        return list(linhas[0].atributos.keys()) if linhas else []

    # Converte a série na matriz (X, y) e nos registros correspondentes.
    #
    # Só entram os meses com defasagens completas dentro do próprio bloco,
    # o que garante que nenhuma linha use informação de outro bloco (RNF03)
    # nem de um instante futuro (RN06).
    # Como funciona: X recebe, para cada mês, os valores dos atributos (as
    # entradas do modelo); y recebe os desembarques do próprio mês (o valor que
    # o modelo deve acertar).
    @staticmethod
    def matriz(serie: SerieTemporal):
        linhas = serie.com_atributos()
        if not linhas:
            return [], [], []
        colunas = list(linhas[0].atributos.keys())
        X = [[r.atributos[c] for c in colunas] for r in linhas]
        y = [r.desembarques for r in linhas]
        return X, y, linhas

    # Limites de detecção de outliers pelo intervalo interquartílico.
    #
    # Estimados **somente sobre o treino** (RN05). O método apenas
    # sinaliza candidatos: picos sazonais legítimos devem ser preservados,
    # conforme a Seção 3.7 do TCC, de modo que nenhum valor é removido
    # automaticamente.
    # Como funciona: calcula o 1º e o 3º quartil (Q1 e Q3) dos desembarques do
    # treino e retorna Q1 - 1,5 x IQR e Q3 + 1,5 x IQR, sendo IQR = Q3 - Q1.
    @staticmethod
    def limites_outliers(treino: SerieTemporal, fator: float = 1.5):
        valores = sorted(r.desembarques for r in treino.registros)
        if not valores:
            return None, None

        # Retorna o quantil p dos valores ordenados (ex.: p = 0,25 dá o Q1), com
        # interpolação linear entre as duas posições mais próximas.
        def quantil(p: float) -> float:
            pos = p * (len(valores) - 1)
            base = int(pos)
            resto = pos - base
            if base + 1 < len(valores):
                return valores[base] + resto * (valores[base + 1] - valores[base])
            return valores[base]

        q1, q3 = quantil(0.25), quantil(0.75)
        iqr = q3 - q1
        return q1 - fator * iqr, q3 + fator * iqr


if __name__ == "__main__":
    from infraestrutura.repositorios.sqlite_repo import SqliteRepository

    servico = PreProcessamentoService()
    serie = servico.preparar(SqliteRepository().carregar_serie())
    treino, teste = servico.dividir(serie)

    r = servico.relatorio
    print("Pré-processamento (RF06–RF08, RF10)")
    print(f"  meses na base ............. {r['meses_brutos']}")
    print(f"  após excluir a pandemia ... {r['meses_apos_corte']}")
    for bloco, n in r["blocos"].items():
        print(f"    bloco {bloco} ................. {n} meses")
    print(f"  meses interpolados ........ {r['meses_interpolados']}")
    print(f"  defasagens usadas ......... {r['lags']}")
    print(f"  linhas na matriz .......... {r['linhas_matriz']}")
    print(f"    treino .................. {r['treino']}")
    print(f"    teste ................... {r['teste']}")

    inferior, superior = PreProcessamentoService.limites_outliers(treino)
    print(f"  limites de outlier (IQR, só no treino): "
          f"{inferior:,.0f} a {superior:,.0f}")
    fora = [x for x in treino.registros
            if x.desembarques < inferior or x.desembarques > superior]
    print(f"    candidatos no treino .... {len(fora)}"
          + (f" ({', '.join(x.rotulo for x in fora)})" if fora else ""))

    print(f"  atributos ................. {servico.nomes_atributos(serie)}")
