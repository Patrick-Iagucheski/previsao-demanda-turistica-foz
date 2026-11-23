import os
import secrets
import sys
import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from flask import (Flask, Response, abort, flash, jsonify, redirect, render_template,
                   request, url_for)

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from aplicacao.avaliacao import N_MINIMO_DM, AvaliacaoService
from aplicacao.exportacao import ExportacaoCsvService
from aplicacao.ingestao import IngestaoService
from aplicacao.preprocessamento import CORTE_TESTE
from aplicacao.previsao import PrevisaoService
from aplicacao.treinamento import TreinamentoService
from aplicacao.visualizacao import MESES, NOME_MODELO, VisualizacaoService, milhar
from dominio.entidades import (EscopoAvaliacaoEnum as E, TipoMetricaEnum as M,
                               TipoModeloEnum as T)
from infraestrutura.anac_csv_loader import AnacCsvLoader
from infraestrutura.modelos.lstm import LSTMModel
from infraestrutura.modelos.naive import NaiveSazonal
from infraestrutura.modelos.random_forest import RandomForestModel
from infraestrutura.repositorios.sqlite_repo import SqliteRepository

# Interface web do sistema de previsão de demanda turística.
#
# As quatro telas estão ligadas à camada de aplicação. Este módulo é também o
# **ponto de composição** do sistema: é aqui, e só aqui, que os serviços da
# camada de aplicação recebem os adaptadores concretos da infraestrutura
# (carregador da ANAC, repositório SQLite e os três modelos). Os serviços
# conhecem apenas as portas do domínio (Ports & Adapters, Seção 4.6.2 do TCC).
#
# Rotas (mapa completo em docs/telas.md):
#     GET  /                       Importação (UC01): série gravada no banco e progresso
#     GET  /importacao/andamento   progresso da importação e gráfico parcial (JSON)
#     GET  /importacao/grafico     gráfico da série: todos os anos ou um ano (?ano=AAAA)
#     POST /importar               IngestaoService.importar(pasta), em segundo plano + validar(serie) (RF04)
#     GET  /treinamento            Treinamento (UC04): formulário e andamento
#     POST /treinar                TreinamentoService.executar, em segundo plano
#     GET  /comparacao             comparar(), Comparação (UC06, UC07, UC09): ?ids=1&ids=2
#     GET  /historico              Histórico (UC08): execuções do banco
#     GET  /execucao/<id>          Detalhes de uma execução
#     POST /execucao/<id>/prever   PrevisaoService.prever: modelo salvo, sem retreinar
#     POST /exportar               ExportacaoCsvService: download do CSV
#
# O treinamento roda numa thread separada porque a LSTM leva minutos; a tela
# de treinamento se atualiza sozinha até a execução terminar. Um treinamento
# por vez.
#
# Variáveis de ambiente opcionais (usadas nos testes, para não tocar no banco
# real): TCC_BANCO (caminho do banco), TCC_MODELOS (pasta dos modelos) e
# TCC_PORTA (porta do servidor; padrão 5000).
#
# Execução:
#     .venv\Scripts\activate
#     python apresentacao/app.py
#     http://localhost:5000

# Cria a aplicação web (Flask).
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)   # só para as mensagens de aviso (flash)

# Ponto de composição

# Fábrica de modelos: liga cada tipo à classe que cria o modelo; é passada ao
# TreinamentoService e ao PrevisaoService.
_FABRICA = {T.NAIVE: NaiveSazonal, T.RANDOM_FOREST: RandomForestModel, T.LSTM: LSTMModel}
# Serviços que não dependem da base escolhida (criados uma vez só).
_avaliacao = AvaliacaoService()
_visualizacao = VisualizacaoService()
_exportacao = ExportacaoCsvService()

# Cache da série lida do banco (evita reler a cada tela).
_CACHE = {"serie": None}

# Bases selecionáveis no cabeçalho (bases.json): a real e as de experimentos
# complementares. Cada uma é um arquivo SQLite próprio.
_BASES = {b["id"]: b for b in config.ler_bases()["bases"]}
_ATIVA = {"base": None}  # base em uso no momento (trocada em trocar_base())

# Cria o repositório da base escolhida e o injeta nos serviços (Martin,
# 2019, cap. 26; Percival; Gregory, 2020). Domínio, serviços e modelos não
# mudam: dependem só das interfaces. TCC_BANCO, quando definido (testes),
# tem prioridade sobre o arquivo da base.
# Como funciona: cria o SqliteRepository da base (e as tabelas, se faltarem),
# escolhe a pasta dos modelos e recria os serviços de ingestão, treinamento e
# previsão com esse repositório.
def _montar_servicos(id_base):
    global _REPO, _ingestao, _treinamento, _previsao
    base = _BASES[id_base]
    _REPO = SqliteRepository(os.environ.get("TCC_BANCO") or (config.RAIZ / base["arquivo"]))
    _REPO.criar_esquema()
    # os modelos de cada base simulada ficam à parte dos do experimento real
    pasta_modelos = os.environ.get("TCC_MODELOS") or (
        config.MODELOS if base["tipo"] == "real" else config.MODELOS / id_base)
    _ingestao = IngestaoService(AnacCsvLoader().carregar_serie, _REPO)
    _treinamento = TreinamentoService(_FABRICA, repositorio=_REPO, pasta_modelos=pasta_modelos)
    _previsao = PrevisaoService(_FABRICA, _REPO)
    _CACHE["serie"] = None
    _ATIVA["base"] = base

# Ao iniciar, monta os serviços com a base padrão de bases.json.
_montar_servicos(config.ler_bases()["padrao"])

# Envia a todas as telas a lista de bases e a base ativa (seletor do cabeçalho).
@app.context_processor
def _base_ativa():
    return {"bases": list(_BASES.values()), "base_ativa": _ATIVA["base"]}

# Treinamento em segundo plano: no máximo um por vez.
_TAREFA = {"atual": None}
_TRAVA = threading.Lock()  # trava para não iniciar dois treinos ou importações ao mesmo tempo

# Importação em segundo plano: no máximo uma por vez, e nunca junto com um treino
# (o treino leria a série enquanto ela é regravada).
_IMPORTACAO = {"atual": None}

# Tempo típico de cada execução, medido no experimento do Capítulo 5.
DURACAO = {
    "naive": "cerca de 3 segundos",
    "rf_auto": "de 15 a 25 segundos",
    "rf_fixo": "alguns segundos",
    "lstm_auto": "de 3 a 10 minutos",
    "lstm_fixo": "cerca de 1 minuto",
}


# Auxiliares

# Retorna a série da base ativa, lida do banco só na primeira vez (depois vem do
# _CACHE); None se o banco ainda estiver vazio.
def _serie():
    if _CACHE["serie"] is None:
        serie = _REPO.carregar_serie()
        _CACHE["serie"] = serie if len(serie) else None
    return _CACHE["serie"]

# Números da série já carregada, exibidos na tela de importação.
def _resumo(serie):
    if serie is None:
        return None
    regs = serie.registros
    fora = [r for r in regs if not r.pandemia]
    blocos = {}
    for r in fora:
        blocos.setdefault(r.bloco_temporal, []).append(r)
    faixas = []
    for b in sorted(blocos):
        ms = sorted(blocos[b], key=lambda x: (x.ano, x.mes))
        faixas.append({
            "bloco": b,
            "inicio": f"{ms[0].ano}-{ms[0].mes:02d}",
            "fim": f"{ms[-1].ano}-{ms[-1].mes:02d}",
            "meses": len(ms),
        })
    return {
        "iata": config.IATA,
        "icao": config.ICAO,
        "aeroporto": config.AEROPORTO_NOME,
        "meses_registro": len(regs),
        "utilizaveis": len(fora),
        "alta_temporada": sum(1 for r in fora if r.alta_temporada),
        "periodo": f"{MESES[regs[0].mes - 1]}/{regs[0].ano} a {MESES[regs[-1].mes - 1]}/{regs[-1].ano}",
        "faixas": faixas,
    }

# Resumo de uma execução para tabelas e listas de seleção.
def _linha(execucao):
    m = execucao.metrica
    tipo = T(execucao.tipo_modelo).value
    return {
        "id": execucao.id,
        "tipo": tipo,
        "modelo": NOME_MODELO[tipo],
        "semente": execucao.semente_aleatoria,
        "data": execucao.data_hora_execucao,
        "periodo": execucao.periodo_teste,
        "mae_g": m(E.GLOBAL, M.MAE), "rmse_g": m(E.GLOBAL, M.RMSE),
        "mape_g": m(E.GLOBAL, M.MAPE),
        "mae_a": m(E.ALTA_TEMPORADA, M.MAE), "rmse_a": m(E.ALTA_TEMPORADA, M.RMSE),
        "mape_a": m(E.ALTA_TEMPORADA, M.MAPE),
        "rotulo": f"nº {execucao.id} · {NOME_MODELO[tipo]} · semente {execucao.semente_aleatoria}",
    }

# Converte ids vindos de formulário/URL, sem repetição e na ordem.
def _ids(valores):
    vistos = []
    for v in valores:
        if str(v).strip().isdigit() and int(v) not in vistos:
            vistos.append(int(v))
    return vistos

# Retorna a execução do banco pelo id, ou None se ela não existir.
def _buscar(id_execucao):
    try:
        return _REPO.buscar_execucao(id_execucao)
    except KeyError:
        return None

# Retorna o texto do período de teste da tela de treinamento: de jan/2025
# (CORTE_TESTE) ao último mês da série.
def _periodo_teste(serie):
    fim = serie.registros[-1] if serie else None
    inicio = f"{CORTE_TESTE[0]}-{CORTE_TESTE[1]:02d}"
    return f"{inicio} a {fim.rotulo}" if fim else inicio


# Filtros de formatação (padrão brasileiro), usados nos templates HTML.
# milhar: número com ponto de milhar; mostra um traço quando não há valor.
@app.template_filter("milhar")
def _f_milhar(v):
    return "—" if v is None else milhar(v)

# pct: número em percentual com vírgula decimal (ex.: 4,52%).
@app.template_filter("pct")
def _f_pct(v, casas=2):
    return "—" if v is None else f"{v:.{casas}f}".replace(".", ",") + "%"

# dec: número com vírgula decimal e 3 casas (ex.: 0,048).
@app.template_filter("dec")
def _f_dec(v, casas=3):
    return "—" if v is None else f"{v:.{casas}f}".replace(".", ",")

# data_br: data e hora no formato dd/mm/aaaa hh:mm.
@app.template_filter("data_br")
def _f_data(d):
    return d.strftime("%d/%m/%Y %H:%M") if d else "—"

# '2019-03' -> 'mar/2019', a forma usada no TCC.
@app.template_filter("mes_ano")
def _f_mes_ano(competencia):
    ano, mes = str(competencia).split("-")
    return f"{MESES[int(mes) - 1]}/{ano}"


# Tela 1: Importação   (UC01 · RF01–RF06)

# Retorna True se a tarefa (importação ou treino) ainda está executando.
def _em_andamento(estado):
    tarefa = estado["atual"]
    return bool(tarefa and tarefa["estado"] == "executando")

# Roda a importação numa thread separada, para a tela não travar. Como funciona:
# chama _ingestao.importar() com ao_ler(), que a cada arquivo lido atualiza o
# progresso e os meses do gráfico parcial; no fim marca a tarefa como concluída ou
# com erro.
def _importar_em_segundo_plano(tarefa, pasta):
    # Chamada a cada arquivo lido: soma os meses do arquivo e atualiza o progresso.
    def ao_ler(lidos, total, nome, meses):
        # um arquivo pode trazer voos de outro mês (o de jul/2022 traz alguns de
        # ago/2022): soma, como a consolidação final, para o gráfico parcial
        # terminar igual à série gravada
        for r in meses:
            anterior = tarefa["meses"].get((r.ano, r.mes))
            tarefa["meses"][(r.ano, r.mes)] = r if anterior is None else replace(
                r, desembarques=anterior.desembarques + r.desembarques)
        tarefa.update(lidos=lidos, total=total, arquivo=nome)
    try:
        serie = _ingestao.importar(pasta, ao_ler=ao_ler)
        _CACHE["serie"] = None   # força a releitura a partir do banco
        regs = serie.ordenar().registros
        tarefa.update(estado="concluida",
                      resumo=f"{len(regs)} meses ({_f_mes_ano(regs[0].rotulo)} a "
                             f"{_f_mes_ano(regs[-1].rotulo)})")
    except (FileNotFoundError, ValueError) as e:
        # recusa prevista pela conferência dos arquivos: mensagem já escrita para o usuário
        tarefa.update(estado="erro", erro=str(e))
    except Exception as e:  # o erro é mostrado na tela, em vez de derrubar a thread
        tarefa.update(estado="erro", erro=f"Erro inesperado ({type(e).__name__}): {e}")
    finally:
        tarefa["fim"] = datetime.now()

# Segundos decorridos e percentual de arquivos lidos da importação.
def _andamento(tarefa):
    decorrido = int(((tarefa.get("fim") or datetime.now()) - tarefa["inicio"]).total_seconds())
    percentual = int(100 * tarefa["lidos"] / tarefa["total"]) if tarefa["total"] else 0
    return decorrido, percentual

# Gráfico com os meses lidos até agora, no eixo da série esperada.
def _grafico_parcial(tarefa):
    return _visualizacao.grafico_serie(list(tarefa["meses"].values()), n_meses=tarefa["total"])

# Tela de importação (GET /): série gravada, resumo, conferência de integridade
# (RF04), gráfico e, se houver, o progresso da importação.
@app.route("/", methods=["GET"])
def importacao():
    serie = _serie()
    tarefa = _IMPORTACAO["atual"]
    decorrido = percentual = None
    if tarefa:
        decorrido, percentual = _andamento(tarefa)
    em_andamento = _em_andamento(_IMPORTACAO)
    return render_template(
        "importacao.html",
        aba="importacao",
        origem=str(config.ANAC_ORIGEM),
        resumo=_resumo(serie),
        integridade=_ingestao.validar(serie) if serie else None,
        registros=(serie.registros if serie else []),
        figura_serie=_grafico_da_serie(serie) if serie else None,
        figura_parcial=_grafico_parcial(tarefa) if em_andamento else None,
        tarefa=tarefa,
        em_andamento=em_andamento,
        decorrido=decorrido,
        percentual=percentual,
    )

# Gráfico da série gravada: todos os anos ou um ano ampliado.
def _grafico_da_serie(serie, ano=None):
    return _visualizacao.grafico_serie(serie.registros, ano=ano,
                                       simulado_ate=_ATIVA["base"].get("simulado_ate"))

# Gráfico da aba "Gráfico", trocado sem recarregar a página quando se
# escolhe um ano (ou "Todos") no campo acima dele.
@app.route("/importacao/grafico", methods=["GET"])
def grafico_serie():
    serie = _serie()
    ano = request.args.get("ano", "").strip()
    if serie is None or (ano and not (ano.isdigit() and len(ano) == 4)):
        abort(400)
    figura = _grafico_da_serie(serie, int(ano) if ano else None)
    if figura is None:
        abort(404)   # ano fora da série
    return render_template("_grafico_serie.html", g=figura)

# Consultado pela tela a cada meio segundo durante a importação: devolve o
# progresso e o gráfico parcial já desenhado, sem recarregar a página.
@app.route("/importacao/andamento", methods=["GET"])
def andamento_importacao():
    tarefa = _IMPORTACAO["atual"]
    if not tarefa:
        return jsonify(estado="nenhuma")
    decorrido, percentual = _andamento(tarefa)
    figura = _grafico_parcial(tarefa) if tarefa["estado"] == "executando" else None
    return jsonify(
        estado=tarefa["estado"], lidos=tarefa["lidos"], total=tarefa["total"],
        arquivo=tarefa["arquivo"], decorrido=decorrido, percentual=percentual,
        meses=len(tarefa["meses"]),
        grafico=render_template("_grafico_serie.html", g=figura) if figura else "",
    )

# Recebe o formulário de importação (POST /importar). Como funciona: confere se a
# pasta existe e se não há importação nem treino em andamento e inicia
# _importar_em_segundo_plano() numa thread; a tela acompanha o progresso.
@app.route("/importar", methods=["POST"])
def importar():
    pasta = Path((request.form.get("pasta") or str(config.ANAC_ORIGEM)).strip())
    if not pasta.is_dir():
        flash(f"Pasta não encontrada: {pasta}", "erro")
        return redirect(url_for("importacao"))
    with _TRAVA:
        if _em_andamento(_IMPORTACAO):
            flash("Já há uma importação em andamento. Aguarde a conclusão.", "erro")
            return redirect(url_for("importacao"))
        if _em_andamento(_TAREFA):
            flash("Há um treinamento em andamento. Importe depois que ele terminar.", "erro")
            return redirect(url_for("importacao"))
        tarefa = {"estado": "executando", "lidos": 0, "total": 0, "arquivo": "", "meses": {},
                  "inicio": datetime.now(), "fim": None, "resumo": None, "erro": None}
        _IMPORTACAO["atual"] = tarefa
        threading.Thread(target=_importar_em_segundo_plano,
                         args=(tarefa, pasta), daemon=True).start()
    return redirect(url_for("importacao"))

# Troca a base ativa (seletor do cabeçalho). Bloqueada durante importação
# ou treino, que usam o repositório da base em uso.
@app.route("/base", methods=["POST"])
def trocar_base():
    with _TRAVA:
        if _em_andamento(_IMPORTACAO) or _em_andamento(_TAREFA):
            flash("Aguarde a importação ou o treinamento em andamento para trocar de base.", "erro")
        elif request.form.get("base") in _BASES:
            _montar_servicos(request.form["base"])
            _IMPORTACAO["atual"] = _TAREFA["atual"] = None   # o andamento era da base anterior
            flash(f"Base ativa: {_ATIVA['base']['nome']}.", "ok")
    return redirect(request.referrer or url_for("importacao"))


# Tela 2: Treinamento   (UC04 · RF10–RF16, RF19)

# Lê os campos condicionais do formulário. Valores fora das opções
# oferecidas são rejeitados.
# Retorna os hiperparâmetros do modelo, o texto da configuração e a duração
# esperada, mostrados na tela durante o treino.
def _hiperparametros(tipo, form):
    if tipo == T.RANDOM_FOREST and form.get("ajuste_rf") == "fixo":
        n = int(form.get("n_estimators", "200"))
        d = form.get("max_depth", "3")
        d = None if d == "none" else int(d)
        if n not in (200, 500) or d not in (2, 3, 5, None):
            raise ValueError("hiperparâmetros do Random Forest fora das opções")
        texto = f"{n} árvores; profundidade {d if d else 'sem limite'}"
        return {"grade": {"n_estimators": [n], "max_depth": [d],
                          "max_features": ["sqrt"]}}, texto, DURACAO["rf_fixo"]
    if tipo == T.LSTM and form.get("ajuste_lstm") == "fixo":
        u = int(form.get("unidades", "16"))
        dr = float(form.get("dropout", "0.2"))
        if u not in (8, 16, 32) or dr not in (0.0, 0.2):
            raise ValueError("hiperparâmetros da LSTM fora das opções")
        texto = f"{u} unidades; dropout {str(dr).replace('.', ',')}"
        return {"grade": {"unidades": [u], "dropout": [dr]}}, texto, DURACAO["lstm_fixo"]
    if tipo == T.NAIVE:
        return {}, "defasagem sazonal de 12 meses", DURACAO["naive"]
    if tipo == T.LSTM:
        return {}, "busca em grade por validação temporal (6 configurações)", DURACAO["lstm_auto"]
    return {}, "busca em grade por validação temporal (16 configurações)", DURACAO["rf_auto"]

# Roda o treinamento numa thread separada (a LSTM leva minutos) e anota na tarefa
# o id da execução gravada ou o erro.
def _treinar_em_segundo_plano(tarefa, serie, tipo, semente, hiper):
    try:
        execucao = _treinamento.executar(serie, tipo, seed=semente, hiperparametros=hiper)
        tarefa.update(estado="concluida", id_execucao=execucao.id)
    except Exception as e:  # o erro é mostrado na tela, em vez de derrubar a thread
        tarefa.update(estado="erro", erro=f"{type(e).__name__}: {e}")
    finally:
        tarefa["fim"] = datetime.now()

# Tela de treinamento (GET /treinamento): formulário, andamento do treino em curso
# e o resultado da última execução.
@app.route("/treinamento", methods=["GET"])
def treinamento():
    serie = _serie()
    tarefa = _TAREFA["atual"]
    em_andamento = bool(tarefa and tarefa["estado"] == "executando")
    resultado = None
    if tarefa and tarefa.get("id_execucao"):
        resultado = _buscar(tarefa["id_execucao"])
    if resultado is None:
        lista = _REPO.listar_execucoes()
        resultado = lista[0] if lista else None
    decorrido = None
    if tarefa:
        decorrido = int(((tarefa.get("fim") or datetime.now()) - tarefa["inicio"]).total_seconds())
    return render_template(
        "treinamento.html",
        aba="treinamento",
        serie_carregada=serie is not None,
        semente=config.SEED,
        periodo_teste=_periodo_teste(serie),
        tarefa=tarefa,
        em_andamento=em_andamento,
        decorrido=decorrido,
        resultado=_linha(resultado) if resultado else None,
        do_ultimo_treino=bool(tarefa and resultado and tarefa.get("id_execucao") == resultado.id),
        duracao=DURACAO,
    )

# Recebe o formulário de treino (POST /treinar). Como funciona: valida o modelo, a
# semente e os hiperparâmetros, confere se não há treino nem importação em
# andamento e inicia _treinar_em_segundo_plano() numa thread.
@app.route("/treinar", methods=["POST"])
def treinar():
    serie = _serie()
    if serie is None:
        flash("Importe a série antes de treinar (pré-requisito do caso de uso).", "erro")
        return redirect(url_for("treinamento"))
    try:
        tipo = T(request.form.get("modelo", "naive"))
        semente = int(request.form.get("semente", config.SEED))
        if not 0 <= semente <= 2 ** 31 - 1:
            raise ValueError("semente fora do intervalo")
        hiper, descricao, duracao = _hiperparametros(tipo, request.form)
    except ValueError as e:
        flash(f"Configuração inválida: {e}", "erro")
        return redirect(url_for("treinamento"))

    with _TRAVA:
        if _em_andamento(_TAREFA):
            flash("Já há um treinamento em andamento. Aguarde a conclusão.", "erro")
            return redirect(url_for("treinamento"))
        if _em_andamento(_IMPORTACAO):
            flash("Há uma importação em andamento. Treine depois que ela terminar.", "erro")
            return redirect(url_for("treinamento"))
        tarefa = {"estado": "executando", "modelo": NOME_MODELO[tipo.value],
                  "semente": semente, "configuracao": descricao, "duracao": duracao,
                  "inicio": datetime.now(), "fim": None, "id_execucao": None, "erro": None}
        _TAREFA["atual"] = tarefa
        threading.Thread(target=_treinar_em_segundo_plano,
                         args=(tarefa, serie, tipo, semente, hiper), daemon=True).start()
    return redirect(url_for("treinamento"))


# Tela 3: Comparação   (UC06, UC07, UC09 · RF17, RF18, RF20, RF22)

MAX_COMPARADAS = 3  # máximo de execuções mostradas na tela de comparação

# Tela de comparação (GET /comparacao). Como funciona: usa as execuções pedidas na
# URL (ou, sem pedido, as da semente 42 de cada modelo), monta o gráfico previsto x
# real e roda o teste de Diebold-Mariano entre cada par (comparar_pares()).
@app.route("/comparacao", methods=["GET"])
def comparar():
    lista = _REPO.listar_execucoes()
    pedidos = _ids(request.args.getlist("ids"))
    ids = pedidos or _avaliacao.execucoes_padrao(lista, config.SEED)
    if len(ids) > MAX_COMPARADAS:
        flash(f"A comparação mostra até {MAX_COMPARADAS} execuções; "
              f"foram consideradas as {MAX_COMPARADAS} primeiras selecionadas.", "erro")
        ids = ids[:MAX_COMPARADAS]
    selecionadas = [e for e in (_buscar(i) for i in ids) if e is not None]

    mesmo_periodo = len({e.periodo_teste for e in selecionadas}) <= 1
    if not mesmo_periodo:
        flash("As execuções selecionadas têm períodos de teste diferentes; "
              "o gráfico e o teste de significância usam só a primeira.", "erro")
    no_grafico = selecionadas if mesmo_periodo else selecionadas[:1]
    figura = _visualizacao.grafico_previsto_real(no_grafico, serie=_serie())

    n_global = len(no_grafico[0].previsoes) if no_grafico else 0
    n_alta = sum(1 for p in no_grafico[0].previsoes if p.alta_temporada) if no_grafico else 0
    testes = [{**par, "a": _linha(par["a"]), "b": _linha(par["b"])}
              for par in _avaliacao.comparar_pares(no_grafico)]

    return render_template(
        "comparacao.html",
        aba="comparacao",
        opcoes=[_linha(e) for e in lista],
        selecionados=ids + [None] * (MAX_COMPARADAS - len(ids)),
        linhas=[_linha(e) for e in selecionadas],
        figura=figura,
        testes=testes,
        n_global=n_global, n_alta=n_alta, n_minimo=N_MINIMO_DM,
        semente_padrao=config.SEED,
    )


# Tela 4: Histórico   (UC08 · RF21, RF22)

# Tela de histórico (GET /historico): lista as execuções gravadas, com as métricas.
@app.route("/historico", methods=["GET"])
def historico():
    return render_template(
        "historico.html",
        aba="historico",
        execucoes=[_linha(e) for e in _REPO.listar_execucoes()],
    )

# Detalhes de uma execução (GET /execucao/<id>); página 404 se o id não existir.
@app.route("/execucao/<int:id_execucao>", methods=["GET"])
def execucao(id_execucao):
    ex = _buscar(id_execucao)
    if ex is None:
        abort(404)
    return _render_execucao(ex)

# RF15, com o modelo persistido pelo RF19: recarrega o modelo salvo e prevê
# o período de teste sem retreinar, conferindo o resultado com as previsões
# registradas. Não grava nada no banco.
# Como funciona: chama _previsao.prever(), compara mês a mês o novo valor com o
# gravado e mostra a maior diferença (zero quando o modelo salvo reproduz o treino).
@app.route("/execucao/<int:id_execucao>/prever", methods=["POST"])
def prever(id_execucao):
    ex = _buscar(id_execucao)
    if ex is None:
        abort(404)
    try:
        novas = _previsao.prever(ex)
    except Exception as e:
        flash(f"Não foi possível prever com o modelo salvo: {e}", "erro")
        return _render_execucao(ex)
    novas_por_mes = {p.rotulo: p.valor_previsto for p in novas}
    diferenca = max((abs(p.valor_previsto - novas_por_mes.get(p.rotulo, float("inf")))
                     for p in ex.previsoes), default=0.0)
    return _render_execucao(ex, reprevisao=novas_por_mes, diferenca=diferenca)

# Monta a tela de detalhes da execução: hiperparâmetros, arquivo do modelo e a
# tabela mês a mês (real, previsto, erro % e, se houver, o valor reprevisto).
def _render_execucao(ex, reprevisao=None, diferenca=None):
    previsoes = [{
        "mes": p.rotulo, "alta": p.alta_temporada, "real": p.valor_real,
        "previsto": p.valor_previsto,
        "erro_pct": (100 * p.erro / p.valor_real) if p.valor_real else None,
        "recarregado": (reprevisao or {}).get(p.rotulo),
    } for p in ex.previsoes]
    return render_template(
        "execucao.html",
        aba="historico",
        e=_linha(ex),
        hiperparametros=ex.hiperparametros,
        caminho_modelo=ex.caminho_modelo,
        modelo_existe=bool(ex.caminho_modelo) and Path(ex.caminho_modelo).exists(),
        janela=ex.tamanho_janela_entrada,
        previsoes=previsoes,
        reprevisao=reprevisao is not None,
        diferenca=diferenca,
    )

# RF22: CSV com metadados, previsões e métricas das execuções escolhidas.
@app.route("/exportar", methods=["POST"])
def exportar():
    ids = _ids(request.form.getlist("ids"))
    execucoes = [e for e in (_buscar(i) for i in ids) if e is not None]
    if not execucoes:
        flash("Selecione ao menos uma execução para exportar.", "erro")
        return redirect(request.referrer or url_for("historico"))
    conteudo = _exportacao.exportar_lote(execucoes)
    nome = ("execucao_" if len(execucoes) == 1 else "execucoes_") + \
        "-".join(str(e.id) for e in execucoes) + ".csv"
    return Response(conteudo, mimetype="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{nome}"'})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("TCC_PORTA", 5000)),
            debug=False, threaded=True)
