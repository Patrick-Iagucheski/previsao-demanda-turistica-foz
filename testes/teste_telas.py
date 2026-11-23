import os, re, shutil, sys, tempfile, time
from pathlib import Path

# Teste ponta a ponta das telas, sobre uma CÓPIA do banco (o real não é tocado).
# Resultado na Seção 5.7 do TCC. Exige, além da base incluída no repositório, os CSV
# da ANAC (dados/brutos ou a pasta da variável ANAC_ORIGEM, parte 8), os modelos
# salvos das 11 execuções (dados/modelos, parte 4) e a base simulada (parte 9).

# Pasta raiz do projeto e pasta temporária com a cópia do banco e dos modelos
# treinados durante o teste.
RAIZ = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="tcc_telas_"))
shutil.copy(RAIZ / "dados" / "experimentos.db", TMP / "banco.db")
os.environ["TCC_BANCO"] = str(TMP / "banco.db")
os.environ["TCC_MODELOS"] = str(TMP / "modelos")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "apresentacao"))
sys.stdout.reconfigure(encoding="utf-8")

import config  # noqa: E402
import app as modulo  # noqa: E402
# Pasta dos CSV da ANAC, usada nos testes de importação.
ANAC = Path(config.ANAC_ORIGEM)
# Cliente de teste do Flask: acessa as rotas sem abrir o navegador.
c = modulo.app.test_client()
# ok() imprime [ok] ou [FALHOU] com a mensagem e retorna a condição; texto()
# retorna o HTML da resposta.
ok = lambda cond, msg: print(("  [ok] " if cond else "  [FALHOU] ") + msg) or cond
texto = lambda r: r.get_data(as_text=True)
falhas = 0

print("1. Telas respondem e não há mais 'aguarda conexão'")
for rota in ("/", "/treinamento", "/comparacao", "/historico", "/execucao/2"):
    r = c.get(rota)
    falhas += not ok(r.status_code == 200 and "aguarda conexão" not in texto(r) and "CONEXÃO PENDENTE" not in texto(r),
                     f"{rota} -> {r.status_code}")
falhas += not ok(c.get("/execucao/9999").status_code == 404, "execução inexistente -> 404")

print("2. Histórico lista o banco")
h = texto(c.get("/historico"))
falhas += not ok(h.count('name="ids"') == 11, f"11 execuções na tabela ({h.count('name=\"ids\"')})")

print("3. Comparação padrão = resultado principal (1, 2, 7)")
cp = texto(c.get("/comparacao"))
falhas += not ok(all(f'value="{i}" selected' in cp for i in (1, 2, 7)), "pré-seleciona 1, 2 e 7")
falhas += not ok(cp.count("<polyline") == 4, f"4 linhas no gráfico ({cp.count('<polyline')})")
falhas += not ok("sem diferença significativa a 5%" in cp and "Seção 3.7 do TCC" in cp, "veredito DM conforme a Seção 3.7")
falhas += not ok("0,179" in cp and "0,537" in cp, "valores-p iguais aos do Cap. 5 (0,179 e 0,537)")
dup = texto(c.get("/comparacao?ids=2&ids=3"))
falhas += not ok('stroke-dasharray="7 4"' in dup, "2 execuções do mesmo modelo: 2ª tracejada")

print("4. Detalhes + prever com o modelo salvo (sem retreinar)")
t0 = time.time()
pv = texto(c.post("/execucao/2/prever"))
falhas += not ok("previsões idênticas" in pv, f"RF recarregado reproduz as previsões ({time.time()-t0:.1f}s)")

print("5. Exportar CSV")
r = c.post("/exportar", data={"ids": ["1", "2"]})
falhas += not ok(r.status_code == 200 and "attachment" in r.headers.get("Content-Disposition", "")
                 and r.headers["Content-Disposition"].endswith('execucoes_1-2.csv"'), f"download {r.headers.get('Content-Disposition')}")
linhas = texto(r).lstrip("\ufeff").splitlines()
falhas += not ok(len(linhas) == 1 + 2 * (1 + 12 + 6), f"{len(linhas)} linhas (esperado 39)")
r = c.post("/exportar", data={}, follow_redirects=True)
falhas += not ok("Selecione ao menos uma" in texto(r), "exportar sem seleção avisa")

print("6. Validação do formulário de treino")
r = c.post("/treinar", data={"modelo": "random_forest", "semente": "42", "ajuste_rf": "fixo",
                             "n_estimators": "999", "max_depth": "3"}, follow_redirects=True)
falhas += not ok("Configuração inválida" in texto(r), "valor fora das opções é rejeitado")
r = c.post("/treinar", data={"modelo": "xgboost"}, follow_redirects=True)
falhas += not ok("Configuração inválida" in texto(r), "modelo desconhecido é rejeitado")

# Envia o formulário de treino pela tela, espera o treino em segundo plano terminar
# e confere se a execução apareceu na tela de treinamento.
def treinar(dados, rotulo, limite=900):
    t0 = time.time()
    c.post("/treinar", data=dados)
    pagina = texto(c.get("/treinamento"))
    andamento = "em andamento" in pagina and 'http-equiv="refresh"' in pagina
    while modulo._TAREFA["atual"]["estado"] == "executando" and time.time() - t0 < limite:
        time.sleep(2)
    t = modulo._TAREFA["atual"]
    pagina = texto(c.get("/treinamento"))
    return ok(andamento and t["estado"] == "concluida" and f"execução nº {t['id_execucao']}" in pagina,
              f"{rotulo}: estado={t['estado']} execução={t['id_execucao']} em {time.time()-t0:.0f}s {t['erro'] or ''}")

print("7. Treinamentos pela tela (em segundo plano)")
falhas += not treinar({"modelo": "naive", "semente": "42"}, "naïve")
falhas += not treinar({"modelo": "random_forest", "semente": "7", "ajuste_rf": "fixo",
                       "n_estimators": "200", "max_depth": "3"}, "RF fixo")
falhas += not treinar({"modelo": "lstm", "semente": "42", "ajuste_lstm": "fixo",
                       "unidades": "16", "dropout": "0.2"}, "LSTM fixa")
n = len(modulo._REPO.listar_execucoes())
falhas += not ok(n == 14, f"cópia do banco agora tem {n} execuções (11 + 3)")
c.post("/treinar", data={"modelo": "random_forest", "semente": "1"})
time.sleep(0.5)
r2 = c.post("/treinar", data={"modelo": "naive", "semente": "2"}, follow_redirects=True)
falhas += not ok("Já há um treinamento em andamento" in texto(r2) and modulo._TAREFA["atual"]["semente"] == 1,
                 "um treinamento por vez (RF em curso recusa o 2º pedido)")
while modulo._TAREFA["atual"]["estado"] == "executando":
    time.sleep(1)

print("8. Importação")
r = c.post("/importar", data={"pasta": r"C:\nao\existe"}, follow_redirects=True)
falhas += not ok("Pasta não encontrada" in texto(r), "pasta inexistente é recusada")

# Envia o formulário de importação com a pasta informada, espera terminar e
# retorna o HTML da tela de importação.
def importar_e_esperar(pasta):
    c.post("/importar", data={"pasta": str(pasta)})
    while modulo._IMPORTACAO["atual"]["estado"] == "executando":
        time.sleep(0.5)
    return texto(c.get("/"))


falhas += not ok("Nenhum arquivo da ANAC foi encontrado" in importar_e_esperar("C:\\"),
                 "pasta sem CSV da ANAC (C:\\) é recusada sem varrer o disco")
dup = Path(tempfile.mkdtemp())
for sub in ("x", "y"):
    (dup / sub).mkdir()
    shutil.copy(next(ANAC.rglob("Movimentacoes_Aeroportuarias_201903.csv")), dup / sub)
falhas += not ok("mais de um arquivo" in importar_e_esperar(dup),
                 "mês duplicado é recusado (não soma em dobro)")
shutil.rmtree(dup, ignore_errors=True)
ruim = Path(tempfile.mkdtemp())
(ruim / "Movimentacoes_Aeroportuarias_201904.csv").write_text("A;B\n1;2\n", encoding="utf-8")
falhas += not ok("não tem as colunas usadas" in importar_e_esperar(ruim),
                 "CSV sem as colunas da ANAC é recusado")
shutil.rmtree(ruim, ignore_errors=True)
real0 = __import__("sqlite3").connect(os.environ["TCC_BANCO"])
falhas += not ok(real0.execute("select qt_desembarques from tb_ser_registros_mensais "
                               "where nr_ano=2019 and nr_mes=3").fetchone()[0] == 96792,
                 "nenhuma importação recusada alterou a série (mar/2019 = 96.792)")
t0 = time.time()
r = c.post("/importar", data={"pasta": str(ANAC)}, follow_redirects=True)
falhas += not ok("Importação em andamento" in texto(r) and "Importar dados" in texto(r),
                 "importação roda em segundo plano e a tela mostra o andamento")
time.sleep(2)
r = c.get("/")
falhas += not ok(("Lendo arquivo" in texto(r) or "Consolidando" in texto(r)) and "<progress" in r.get_data(as_text=True),
                 f"barra de progresso: {modulo._IMPORTACAO['atual']['lidos']} de {modulo._IMPORTACAO['atual']['total']} arquivos")
a = c.get("/importacao/andamento").get_json()
falhas += not ok(a["estado"] == "executando" and a["meses"] > 0 and "<polyline" in a["grafico"],
                 f"gráfico parcial durante a leitura ({a['meses']} meses desenhados)")
r2 = c.post("/importar", data={"pasta": str(ANAC)}, follow_redirects=True)
falhas += not ok("Já há uma importação em andamento" in texto(r2), "uma importação por vez")
r3 = c.post("/treinar", data={"modelo": "naive", "semente": "3"}, follow_redirects=True)
falhas += not ok("Há uma importação em andamento" in texto(r3), "não treina durante a importação")
while modulo._IMPORTACAO["atual"]["estado"] == "executando":
    time.sleep(1)
r = c.get("/")
falhas += not ok("Importação concluída: 81 meses" in texto(r), f"IngestaoService importa os 81 meses ({time.time()-t0:.0f}s)")
parcial = {k: v.desembarques for k, v in modulo._IMPORTACAO["atual"]["meses"].items()}
falhas += not ok(parcial == {(x.ano, x.mes): x.desembarques for x in modulo._REPO.carregar_serie().registros},
                 "gráfico parcial termina igual à série gravada (81 meses)")
falhas += not ok('id="aba-grafico"' in texto(r) and texto(r).count('class="linha-alta"') == 11
                 and texto(r).count('class="linha-pandemia"') == 38,
                 "abas Gráfico/Tabela e linhas coloridas (11 de alta, 38 da pandemia)")

print("8b. Gráfico da série: um ano ampliado")
g = c.get("/importacao/grafico?ano=2024")
falhas += not ok(g.status_code == 200 and texto(g).count('class="alvo"') == 12
                 and 'stroke="#b9c6c9"' in texto(g) and "94,5 mil" in texto(g),
                 "ano 2024: 12 meses, 2023 de comparação e o valor de jan (94,5 mil)")
falhas += not ok(c.get("/importacao/grafico?ano=2030").status_code == 404
                 and c.get("/importacao/grafico?ano=abc").status_code == 400,
                 "ano fora da série (404) e ano inválido (400) são recusados")
falhas += not ok(texto(c.get("/importacao/grafico")).count('class="alvo"') == 81, "Todos: os 81 meses")

print("9. Seletor de base")
r = c.post("/base", data={"base": "historico20"}, follow_redirects=True)
falhas += not ok("Base ativa: ANAC + histórico simulado" in texto(r)
                 and "dados simulados até fev/2019" in texto(c.get("/")),
                 "troca para a base simulada e a tela sinaliza dados simulados")
c.post("/treinar", data={"modelo": "random_forest", "semente": "5"})
r = c.post("/base", data={"base": "real"}, follow_redirects=True)
falhas += not ok("Aguarde a importação ou o treinamento" in texto(r)
                 and modulo._ATIVA["base"]["id"] == "historico20",
                 "não troca de base durante um treinamento")
while modulo._TAREFA["atual"] and modulo._TAREFA["atual"]["estado"] == "executando":
    time.sleep(1)
r = c.post("/base", data={"base": "real"}, follow_redirects=True)
falhas += not ok("Base ativa: ANAC — dados reais" in texto(r) and "dados reais" in texto(c.get("/")),
                 "volta para a base real")

real = __import__("sqlite3").connect(RAIZ / "dados" / "experimentos.db")
falhas += not ok(real.execute("select count(*) from tb_exp_execucoes").fetchone()[0] == 11, "banco REAL intacto (11)")
print("\nRESULTADO:", "tudo certo" if not falhas else f"{falhas} falha(s)")
shutil.rmtree(TMP, ignore_errors=True)
