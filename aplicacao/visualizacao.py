import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dominio.entidades import Execucao, RegistroMensal, SerieTemporal, TipoModeloEnum

# Serviço de visualização (camada de aplicação): RF20.
#
#     grafico_previsto_real(execucoes) -> Figura
#     grafico_serie(registros) -> Figura
#
# Monta a geometria dos gráficos que as telas desenham em SVG: o previsto ×
# real da comparação e a série consolidada da importação. O gráfico é gerado pelo próprio sistema, sem biblioteca
# JavaScript nem recurso externo, para que a interface funcione sem internet
# (decisão registrada em docs/telas.md).
#
# Cores: valor real em tinta escura; cada modelo com a mesma cor das figuras
# do Capítulo 5 do TCC (naïve azul, Random Forest laranja, LSTM verde-água).
# Duas execuções do mesmo modelo distinguem-se pelo tracejado e pelo marcador.
# Os meses de alta temporada recebem faixa âmbar, como no restante do sistema.
# No gráfico da série, a alta temporada recebe faixa verde-clara (pontos verdes
# nas séries com mais de dez anos) e a pandemia, faixa cinza, as mesmas cores
# das linhas da tabela da tela de importação; o trecho simulado é tracejado.

# Abreviações dos meses, usadas nos rótulos dos gráficos e das telas.
MESES = ["jan", "fev", "mar", "abr", "mai", "jun",
         "jul", "ago", "set", "out", "nov", "dez"]
# Nome de cada modelo como aparece nas telas e nas legendas.
NOME_MODELO = {"naive": "Naïve sazonal", "random_forest": "Random Forest", "lstm": "LSTM"}
# Cor de cada modelo nos gráficos (as mesmas das figuras do Capítulo 5).
COR_MODELO = {"naive": "#2a78d6", "random_forest": "#eb6834", "lstm": "#1baf7a"}
COR_REAL = "#0b3c49"  # cor da linha do valor real
COR_SIMULADO = "#7aa9b2"   # trecho fictício da base simulada (tracejado)
COR_ALTA = "#1d9e5a"       # pontos de alta temporada nas séries longas
# Tracejado e marcador de cada execução do mesmo modelo (a 1ª com linha contínua
# e círculo, a 2ª tracejada e quadrado...), usados em grafico_previsto_real().
TRACEJADOS = ["", "7 4", "2 3"]
MARCADORES = ["circulo", "quadrado", "triangulo"]

# Área do desenho (unidades do viewBox)
LARGURA, ALTURA = 780, 262
X0, X1, Y0, Y1 = 72, 762, 18, 238

# Retorna a posição do mês numa contagem contínua (ano x 12 + mês - 1).
def _indice(ano: int, mes: int) -> int:
    return ano * 12 + mes - 1

# Formata o número com ponto de milhar e sem casas decimais (ex.: 76543 vira
# "76.543").
def milhar(v: float) -> str:
    return f"{v:,.0f}".replace(",", ".")

# Escolhe o intervalo entre as linhas do eixo vertical: o menor passo da lista
# que gere no máximo 6 divisões.
def _passo(amplitude: float) -> int:
    for passo in (1000, 2000, 2500, 5000, 10000, 20000, 25000, 50000):
        if amplitude / passo <= 6:
            return passo
    return 100000

# Calcula as posições (x, y) de tudo o que os gráficos das telas desenham em
# SVG (RF20).
class VisualizacaoService:

    # Geometria do gráfico. Sem execuções, desenha apenas a série
    # observada (últimos 12 meses fora da pandemia), se houver série.
    # Como funciona: junta os valores reais e previstos para achar a escala
    # vertical; converte cada mês em posição x e cada valor em posição y; monta
    # uma linha para o valor real e uma para cada execução e marca as faixas dos
    # meses de alta temporada.
    def grafico_previsto_real(self, execucoes: Sequence[Execucao],
                              serie: Optional[SerieTemporal] = None) -> Optional[Dict]:
        if execucoes:
            base = execucoes[0].previsoes
            meses = [(p.ano, p.mes, p.alta_temporada) for p in base]
            reais = [p.valor_real for p in base]
        elif serie is not None and len(serie):
            recentes = [r for r in serie.ordenar().registros if not r.pandemia][-12:]
            meses = [(r.ano, r.mes, r.alta_temporada) for r in recentes]
            reais = [r.desembarques for r in recentes]
        else:
            return None

        valores = [v for v in reais if v is not None]
        for e in execucoes:
            valores += [p.valor_previsto for p in e.previsoes]
        passo = _passo(max(valores) - min(valores))
        lo = math.floor(min(valores) / passo) * passo
        hi = math.ceil(max(valores) / passo) * passo
        if hi == lo:
            hi = lo + passo

        n = len(meses)
        dx = (X1 - X0) / max(1, n - 1) if n > 1 else 0
        px = lambda i: X0 + i * dx if n > 1 else (X0 + X1) / 2
        py = lambda v: Y1 - (v - lo) / (hi - lo) * (Y1 - Y0)
        rotulo = lambda a, m: f"{MESES[m - 1]}/{str(a)[2:]}"

        # Converte uma lista de valores em pontos do gráfico: x, y, valor e o texto da
        # dica mostrada ao passar o cursor.
        def pontos(vals, nome):
            return [{"x": round(px(i), 1), "y": round(py(v), 1), "valor": milhar(v),
                     "dica": f"{rotulo(a, m)} · {nome}: {milhar(v)}"}
                    for i, ((a, m, _), v) in enumerate(zip(meses, vals)) if v is not None]

        series = [{"nome": "Valor real", "cor": COR_REAL, "tracejado": "",
                   "largura": 2.8, "marcador": "circulo", "pontos": pontos(reais, "real")}]
        vistos: Dict[str, int] = {}
        for e in execucoes:
            tipo = TipoModeloEnum(e.tipo_modelo).value
            k = vistos.get(tipo, 0)
            vistos[tipo] = k + 1
            nome = f"{NOME_MODELO[tipo]} · nº {e.id} · semente {e.semente_aleatoria}"
            series.append({
                "nome": nome, "cor": COR_MODELO[tipo],
                "tracejado": TRACEJADOS[k % len(TRACEJADOS)], "largura": 2.0,
                "marcador": MARCADORES[k % len(MARCADORES)],
                "pontos": pontos([p.valor_previsto for p in e.previsoes], NOME_MODELO[tipo]),
            })

        return {
            "largura": LARGURA, "altura": ALTURA,
            "x0": X0, "x1": X1, "y0": Y0, "y1": Y1,
            "rotulos_x": [{"x": round(px(i), 1), "texto": rotulo(a, m)}
                          for i, (a, m, _) in enumerate(meses)],
            "ticks_y": [{"y": round(py(v), 1), "texto": milhar(v)}
                        for v in range(int(lo), int(hi) + 1, passo)],
            "faixas_alta": [{"x": round(px(i) - max(dx, 28) / 2, 1), "largura": round(max(dx, 28), 1)}
                            for i, (_, _, alta) in enumerate(meses) if alta],
            "series": series,
            "apenas_observada": not execucoes,
        }

    # Geometria da série consolidada (todos os meses, com a pandemia).
    # n_meses reserva o eixo horizontal para a série esperada: durante a
    # importação, o gráfico cresce da esquerda para a direita sem mudar de
    # escala a cada arquivo lido. Um mês ausente interrompe a linha.
    # simulado_ate ('AAAA-MM', base simulada) separa o trecho fictício, que é
    # desenhado tracejado. Cada ponto leva os dados da caixa de detalhes
    # exibida ao passar o cursor.
    # ano amplia um único ano: os 12 meses com o valor escrito em cada ponto
    # e o ano anterior como linha de comparação, na mesma escala vertical da
    # série inteira (para que anos diferentes possam ser comparados).
    # Como funciona: calcula a escala, cria um ponto por mês com ponto(), quebra
    # a linha onde falta mês ou onde o simulado vira real e monta as faixas de
    # pandemia e de alta temporada, os rótulos do eixo e a lista de anos.
    def grafico_serie(self, registros: Sequence[RegistroMensal], n_meses: int = 0,
                      simulado_ate: Optional[str] = None,
                      ano: Optional[int] = None) -> Optional[Dict]:
        todos = sorted(registros, key=lambda r: (r.ano, r.mes))
        if not todos:
            return None
        if ano is None:
            regs = todos
            inicio = _indice(regs[0].ano, regs[0].mes)
            n = max(_indice(regs[-1].ano, regs[-1].mes) - inicio + 1, n_meses, 2)
        else:
            regs = [r for r in todos if r.ano == ano]
            if not regs:
                return None
            inicio, n = _indice(ano, 1), 12
        if ano is None:
            dx = (X1 - X0) / (n - 1)
            px = lambda i: X0 + i * dx
        else:
            # cada mês no centro do seu espaço: jan e dez não encostam no eixo
            dx = (X1 - X0) / n
            px = lambda i: X0 + (i + 0.5) * dx
        passo = _passo(max(r.desembarques for r in todos))
        hi = max(passo, math.ceil(max(r.desembarques for r in todos) / passo) * passo)
        py = lambda v: Y1 - v / hi * (Y1 - Y0)
        largura = max(dx, 2)
        # em séries longas as faixas de alta temporada viram listras: usa pontos
        longa = ano is None and n > 120
        valor = {_indice(r.ano, r.mes): r.desembarques for r in todos}

        # Retorna a variação percentual entre o valor atual e o de comparação (mês
        # anterior ou mesmo mês do ano anterior) e o sinal, sobe ou desce, para a cor.
        def variacao(atual, base):
            if base is None or base <= 0:
                return {"texto": "—", "sinal": ""}
            v = (atual - base) / base * 100
            return {"texto": f"{v:+.1f}%".replace(".", ","), "sinal": "sobe" if v >= 0 else "desce"}

        # Monta um ponto da série: a posição no gráfico e os dados da caixa de detalhes
        # (valor, variações, alta temporada, pandemia, bloco e origem).
        def ponto(r, i):
            k = _indice(r.ano, r.mes)
            simulado = bool(simulado_ate) and r.rotulo <= simulado_ate
            return {
                "x": round(px(i), 1), "y": round(py(r.desembarques), 1),
                "ano": r.ano,
                "valor_curto": f"{r.desembarques / 1000:.1f} mil".replace(".", ","),
                "alta": r.alta_temporada and not r.pandemia,
                "titulo": f"{MESES[r.mes - 1]}/{r.ano}",
                "valor": milhar(r.desembarques),
                "var_mes": variacao(r.desembarques, valor.get(k - 1)),
                "rotulo_ano": f"{MESES[r.mes - 1]}/{r.ano - 1}",
                "var_ano": variacao(r.desembarques, valor.get(k - 12)),
                "alta_temporada": r.alta_temporada,
                "pandemia": r.pandemia,
                "bloco": "—" if r.pandemia else (
                    "1 (pré-pandemia)" if int(r.bloco_temporal) == 1 else "2 (pós-pandemia)"),
                "origem": ("simulado" if simulado else "dados reais da ANAC") if simulado_ate else "",
                "simulado": simulado,
            }

        # trechos contínuos da linha: um mês ausente ou a passagem do simulado
        # para o real começam um trecho novo (o real recomeça no último simulado,
        # para a linha não ter falha)
        segmentos: List[Dict] = []
        anterior = None
        for r in regs:
            i = _indice(r.ano, r.mes) - inicio
            p = ponto(r, i)
            if anterior is None or i != anterior[0] + 1:
                segmentos.append({"simulado": p["simulado"], "repete": False, "pontos": []})
            elif p["simulado"] != segmentos[-1]["simulado"]:
                segmentos.append({"simulado": p["simulado"], "repete": True, "pontos": [anterior[1]]})
            segmentos[-1]["pontos"].append(p)
            anterior = (i, p)

        # faixas contínuas de pandemia (uma por trecho) e meses de alta fora dela
        pandemia: List[Tuple[int, int]] = []
        for r in regs:
            if r.pandemia:
                i = _indice(r.ano, r.mes) - inicio
                if pandemia and i <= pandemia[-1][1] + 2:   # atravessa um mês ausente
                    pandemia[-1] = (pandemia[-1][0], i)
                else:
                    pandemia.append((i, i))

        # um rótulo por ano (janeiro) ou, em séries longas, a cada poucos anos;
        # no ano ampliado, um por mês
        rotulos = []
        if ano is None:
            intervalo = max(1, math.ceil(n / 12 / 10))
            for k in range(n):
                a, mes = divmod(inicio + k, 12)
                if mes == 0 and a % intervalo == 0:
                    rotulos.append({"x": round(px(k), 1),
                                    "texto": f"jan/{str(a)[2:]}" if intervalo == 1 else str(a)})
        else:
            rotulos = [{"x": round(px(k), 1), "texto": MESES[k]} for k in range(12)]

        # ano ampliado: o ano anterior, na mesma posição de mês, para comparação
        comparacao: List[List[Dict]] = []
        if ano is not None:
            for m in range(1, 13):
                v = valor.get(_indice(ano - 1, m))
                if v is None:
                    if comparacao and comparacao[-1]:
                        comparacao.append([])
                    continue
                if not comparacao:
                    comparacao.append([])
                comparacao[-1].append({"x": round(px(m - 1), 1), "y": round(py(v), 1)})
            comparacao = [c for c in comparacao if len(c) > 1]

        divisa = None
        sim = [ponto_["x"] for seg in segmentos if seg["simulado"] for ponto_ in seg["pontos"]]
        if sim and any(not seg["simulado"] for seg in segmentos):
            divisa = round(max(sim) + dx / 2, 1)

        anos_serie = sorted({r.ano for r in todos})

        # Segunda linha da lista de anos: origem, pandemia, meses e total.
        def resumo_do_ano(a):
            meses = [r for r in todos if r.ano == a]
            simulados = sum(1 for r in meses if simulado_ate and r.rotulo <= simulado_ate)
            origem = ("dados reais" if not simulados else "simulado" if simulados == len(meses)
                      else f"simulado até {MESES[simulados - 1 + meses[0].mes - 1]}, real depois")
            pandemia = " · pandemia" if any(r.pandemia for r in meses) else ""
            total = milhar(sum(r.desembarques for r in meses))
            return f"{origem}{pandemia} · {len(meses)} meses · {total} passageiros"
        if ano is None:
            titulo = f"Todos os anos · {anos_serie[0]} a {anos_serie[-1]}"
        else:
            origem = ""
            if simulado_ate:
                simulados = [r for r in regs if r.rotulo <= simulado_ate]
                if len(simulados) == len(regs):
                    origem = " · simulado"
                elif simulados:
                    origem = (f" · simulado até {MESES[simulados[-1].mes - 1]}, "
                              f"real a partir de {MESES[simulados[-1].mes % 12]}")
                else:
                    origem = " · dados reais"
            titulo = f"{ano}{origem}" + (" · pandemia" if any(r.pandemia for r in regs) else "")

        return {
            "largura": LARGURA, "altura": ALTURA,
            "x0": X0, "x1": X1, "y0": Y0, "y1": Y1,
            "ano": ano,
            "anos": anos_serie,
            "lista_anos": [{"ano": a, "texto": resumo_do_ano(a)} for a in reversed(anos_serie)],
            "titulo": titulo,
            "comparacao": comparacao,
            "rotulos_x": rotulos,
            "ticks_y": [{"y": round(py(v), 1), "texto": milhar(v)}
                        for v in range(0, int(hi) + 1, passo * 2 if hi / passo > 6 else passo)],
            "faixas_alta": [] if longa else [
                {"x": round(px(_indice(r.ano, r.mes) - inicio) - largura / 2, 1),
                 "largura": round(largura, 1)}
                for r in regs if r.alta_temporada and not r.pandemia],
            "faixas_pandemia": [{"x": round(px(a) - largura / 2, 1),
                                 "largura": round((b - a + 1) * dx, 1),
                                 "meio": round((px(a) + px(b)) / 2, 1)} for a, b in pandemia],
            "segmentos": segmentos,
            "divisa_simulado": divisa,
            "tem_simulado": bool(sim),
            "cor": COR_REAL,
            "cor_simulado": COR_SIMULADO,
            "cor_alta": COR_ALTA,
            "longa": longa,
        }
