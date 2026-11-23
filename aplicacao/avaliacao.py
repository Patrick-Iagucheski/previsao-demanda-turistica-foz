import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dominio.entidades import (
    EscopoAvaliacaoEnum, Execucao, Metrica, Previsao, TipoMetricaEnum, TipoModeloEnum,
)

# Serviço de avaliação (camada de aplicação): RF16, RF17 e RF18.
#
#     calcular_metricas(prev, real, escopo)  MAE, RMSE e MAPE num escopo
#     avaliar(previsoes)                     as três métricas nos dois escopos
#     diebold_mariano(erros_a, erros_b)      valor-p do teste de significância
#     comparar_pares(execucoes)              DM entre cada par e o veredito (RF18)
#     execucoes_padrao(execucoes, semente)   o que comparar quando nada é escolhido
#
# A decisão sobre o resultado do teste (significativo ou não, e qual modelo
# errou menos) é regra do caso de uso de comparação e fica aqui, não na tela:
# a interface só exibe o veredito (Percival; Gregory, 2020, cap. 4).
#
# As métricas são calculadas pelo scikit-learn, e não por fórmulas próprias,
# conforme o RNF04 (precisão dos cálculos): biblioteca consolidada, com
# implementações testadas, reduz o risco de erro. O teste de Diebold-Mariano não existe pronto nessas
# bibliotecas; é montado sobre a autocovariância do statsmodels e a
# distribuição t do SciPy, como prevê a Seção 3.7 do TCC.
#
# Execução isolada (autoteste com valores conhecidos):
#     python aplicacao/avaliacao.py

# Número mínimo de previsões para que o teste de Diebold-Mariano seja
# reportado como inferência. Adota-se um ciclo sazonal completo (12 meses):
# com menos que isso, o diferencial de perdas fica dominado pela estação
# específica que caiu no teste. Com o conjunto de teste de 2025, o escopo
# global (n = 12) é aplicável e o de alta temporada (n = 3) é apresentado de
# forma descritiva, como determina a Seção 3.7 do TCC.
# Constante usada em teste_diebold_mariano() para indicar se o teste é aplicável.
N_MINIMO_DM = 12

# Mede e compara o desempenho das previsões.
class AvaliacaoService:
    # RF16: métricas de erro

    # MAE, RMSE e MAPE de um escopo. O MAPE é devolvido em percentual.
    # Como funciona: confere se previsões e reais têm o mesmo tamanho e calcula
    # cada métrica com a função pronta do scikit-learn (o MAPE é multiplicado
    # por 100 para virar percentual).
    @staticmethod
    def calcular_metricas(
        prev: Sequence[float], real: Sequence[float], escopo: EscopoAvaliacaoEnum,
    ) -> List[Metrica]:
        from sklearn.metrics import (
            mean_absolute_error, mean_absolute_percentage_error,
            root_mean_squared_error,
        )

        if len(prev) != len(real):
            raise ValueError("previsões e valores reais com tamanhos diferentes")
        if not prev:
            return []
        return [
            Metrica(escopo, TipoMetricaEnum.MAE,
                    float(mean_absolute_error(real, prev))),
            Metrica(escopo, TipoMetricaEnum.RMSE,
                    float(root_mean_squared_error(real, prev))),
            Metrica(escopo, TipoMetricaEnum.MAPE,
                    float(mean_absolute_percentage_error(real, prev) * 100)),
        ]

    # As três métricas sobre todo o teste e restritas à alta temporada
    # (Seção 3.7 do TCC); o segundo escopo é o que responde diretamente à
    # pergunta de pesquisa.
    # Como funciona: separa os meses que têm valor real; destes, separa os de
    # alta temporada (jan, jul e dez); chama calcular_metricas() uma vez para cada
    # grupo e retorna as seis métricas juntas.
    def avaliar(self, previsoes: Sequence[Previsao]) -> List[Metrica]:
        validas = [p for p in previsoes if p.valor_real is not None]
        alta = [p for p in validas if p.alta_temporada]
        metricas = self.calcular_metricas(
            [p.valor_previsto for p in validas], [p.valor_real for p in validas],
            EscopoAvaliacaoEnum.GLOBAL,
        )
        metricas += self.calcular_metricas(
            [p.valor_previsto for p in alta], [p.valor_real for p in alta],
            EscopoAvaliacaoEnum.ALTA_TEMPORADA,
        )
        return metricas

    # RF17: significância estatística

    # Teste de Diebold e Mariano (1995) com a correção para amostras
    # finitas de Harvey, Leybourne e Newbold (1997).
    #
    # Hipótese nula: os dois modelos têm a mesma acurácia esperada.
    # Estatística negativa indica que o modelo A errou menos que o B.
    #
    # Devolve um dicionário com a estatística corrigida, o valor-p
    # bilateral, o tamanho da amostra e a indicação de aplicabilidade
    # (n >= N_MINIMO_DM).
    #
    # Como funciona: para cada mês, compara o erro dos dois modelos (diferencial
    # de perdas d = erro_a² - erro_b²); calcula a média de d e a variância dessa
    # média; divide uma pela raiz da outra (estatística DM), aplica a correção
    # para amostra pequena e tira o valor-p da distribuição t com n - 1 graus de
    # liberdade. Valor-p abaixo de 0,05 indica diferença significativa.
    @staticmethod
    def teste_diebold_mariano(
        erros_a: Sequence[float], erros_b: Sequence[float],
        horizonte: int = 1, perda: str = "quadratica",
    ) -> Dict[str, object]:
        import numpy as np
        from scipy import stats
        from statsmodels.tsa.stattools import acovf

        a = np.asarray(erros_a, dtype=float)
        b = np.asarray(erros_b, dtype=float)
        if a.shape != b.shape:
            raise ValueError("séries de erros com tamanhos diferentes")
        n = len(a)
        h = horizonte

        if perda == "quadratica":
            d = a ** 2 - b ** 2
        elif perda == "absoluta":
            d = np.abs(a) - np.abs(b)
        else:
            raise ValueError("perda deve ser 'quadratica' ou 'absoluta'")

        resultado = {
            "n": n, "horizonte": h, "perda": perda,
            "aplicavel": n >= N_MINIMO_DM,
            "estatistica": None, "valor_p": None,
        }
        if n < 2:
            return resultado

        media = d.mean()
        gamas = acovf(d, demean=True, fft=False, nlag=max(h - 1, 0))
        variancia = (gamas[0] + 2 * sum(gamas[1:h])) / n
        if variancia <= 0:
            # Diferencial de perdas constante: nenhuma evidência de diferença.
            resultado.update(estatistica=0.0, valor_p=1.0)
            return resultado

        dm = media / math.sqrt(variancia)
        correcao = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        dm_hln = dm * correcao
        valor_p = 2 * stats.t.sf(abs(dm_hln), df=n - 1)
        resultado.update(estatistica=float(dm_hln), valor_p=float(valor_p))
        return resultado

    # Assinatura do diagrama de classes: devolve só o valor-p.
    # Para estatística e aplicabilidade, use teste_diebold_mariano().
    # Como funciona: chama teste_diebold_mariano() e retorna só o campo valor_p.
    def diebold_mariano(self, erros_a: Sequence[float],
                        erros_b: Sequence[float]) -> float:
        return self.teste_diebold_mariano(erros_a, erros_b)["valor_p"]

    # RF18: comparação de execuções

    # Aplica o Diebold-Mariano a cada par de execuções e decide o
    # veredito: só há diferença quando o teste é aplicável (n >= N_MINIMO_DM)
    # e o valor-p fica abaixo do nível (Seção 3.7 do TCC).
    # Como funciona: forma todos os pares possíveis entre as execuções
    # escolhidas, roda o teste com os erros mês a mês de cada uma e escreve o
    # veredito: amostra insuficiente, diferença significativa (dizendo qual
    # errou menos) ou sem diferença significativa.
    def comparar_pares(self, execucoes: Sequence[Execucao],
                       nivel: float = 0.05) -> List[Dict[str, object]]:
        pares = []
        for a, b in combinations(execucoes, 2):
            r = self.teste_diebold_mariano([p.erro for p in a.previsoes],
                                           [p.erro for p in b.previsoes])
            significativo = bool(r["aplicavel"] and r["valor_p"] is not None
                                 and r["valor_p"] < nivel)
            if not r["aplicavel"] or r["valor_p"] is None:
                veredito = "amostra insuficiente para inferência"
            elif significativo:
                melhor = a if r["estatistica"] < 0 else b
                veredito = f"diferença significativa a {nivel:.0%}: nº {melhor.id} errou menos"
            else:
                veredito = f"sem diferença significativa a {nivel:.0%}"
            pares.append({"a": a, "b": b, "veredito": veredito,
                          "significativo": significativo, **r})
        return pares

    # Sem seleção, compara a execução mais recente de cada modelo com a
    # semente padrão do projeto (o resultado principal do Capítulo 5).
    # Como funciona: para naïve, RF e LSTM, pega a primeira execução da lista com
    # a semente padrão (42); se não houver, a primeira do modelo. Retorna os ids.
    @staticmethod
    def execucoes_padrao(execucoes: Sequence[Execucao], semente: int) -> List[int]:
        ids = []
        for tipo in (TipoModeloEnum.NAIVE, TipoModeloEnum.RANDOM_FOREST, TipoModeloEnum.LSTM):
            do_tipo = [e for e in execucoes if TipoModeloEnum(e.tipo_modelo) == tipo]
            padrao = [e for e in do_tipo if e.semente_aleatoria == semente]
            escolhida = (padrao or do_tipo or [None])[0]
            if escolhida:
                ids.append(escolhida.id)
        return ids


if __name__ == "__main__":
    servico = AvaliacaoService()

    # Autoteste das métricas com valores calculáveis de cabeça.
    real = [100.0, 200.0, 300.0, 400.0]
    prev = [110.0, 190.0, 330.0, 400.0]
    m = {x.tipo_metrica.value: x.valor_metrica
         for x in servico.calcular_metricas(prev, real, EscopoAvaliacaoEnum.GLOBAL)}
    # erros absolutos: 10, 10, 30, 0 -> MAE = 12.5
    # quadrados: 100, 100, 900, 0 -> RMSE = sqrt(275) = 16.583
    # percentuais: 10%, 5%, 10%, 0% -> MAPE = 6.25%
    assert abs(m["MAE"] - 12.5) < 1e-9, m
    assert abs(m["RMSE"] - math.sqrt(275)) < 1e-9, m
    assert abs(m["MAPE"] - 6.25) < 1e-9, m
    print(f"métricas ............ OK  (MAE {m['MAE']}, RMSE {m['RMSE']:.3f}, MAPE {m['MAPE']}%)")

    # Autoteste do DM: erros idênticos -> nenhuma diferença (p = 1).
    iguais = servico.teste_diebold_mariano([1, -2, 3] * 4, [1, -2, 3] * 4)
    assert iguais["valor_p"] == 1.0, iguais
    # A erra sistematicamente bem menos que B -> p pequeno, estatística < 0.
    pequeno = [1, -1, 2, -2, 1, -1, 2, -2, 1, -1, 2, -2]
    grande = [9, -8, 10, -9, 8, -10, 9, -8, 10, -9, 8, -10]
    forte = servico.teste_diebold_mariano(pequeno, grande)
    assert forte["estatistica"] < 0 and forte["valor_p"] < 0.05, forte
    assert forte["aplicavel"] is True
    curto = servico.teste_diebold_mariano([1, 2, 3], [3, 2, 1])
    assert curto["aplicavel"] is False
    print(f"Diebold-Mariano ..... OK  (erros iguais p=1.0; A melhor que B: "
          f"DM={forte['estatistica']:.2f}, p={forte['valor_p']:.2e}; n=3 inaplicável)")
