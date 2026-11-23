import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from sqlalchemy import create_engine, event, text

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import config
from dominio.entidades import (
    BlocoTemporalEnum, EscopoAvaliacaoEnum, Execucao, Metrica, Previsao,
    RegistroMensal, SerieTemporal, TipoMetricaEnum, TipoModeloEnum,
)
from dominio.portas import IExecucaoRepository, ISerieRepository

# Repositório SQLite: adaptador de persistência (camada de infraestrutura).
#
# Implementa as duas portas do domínio:
#     ISerieRepository     criar o esquema, gravar e reler a série mensal
#     IExecucaoRepository  salvar_execucao (com as métricas, por composição),
#                          salvar_previsoes, listar_execucoes, buscar_execucao
#
# O acesso é feito pelo SQLAlchemy, com SQL aceito pelo SQLite e pelo PostgreSQL
# (inclusive o upsert ON CONFLICT, que não é SQL padrão). Para usar o PostgreSQL,
# além da URL de conexão, é preciso retirar o PRAGMA ligado a cada conexão e trocar
# o uso de lastrowid, que não retorna o identificador naquele banco (RNF11).
#
# A tradução entre os nomes do domínio (desembarques, alta_temporada...) e os
# nomes das colunas (qt_desembarques, fl_alta_temporada...) acontece só aqui,
# para que o domínio não conheça o formato de armazenamento.
#
# Execução (cria o banco e carrega a série a partir dos CSV da ANAC):
#     python infraestrutura/repositorios/sqlite_repo.py

# Arquivo com os comandos CREATE TABLE do banco, lido em criar_esquema().
ESQUEMA = Path(__file__).resolve().parent / "schema.sql"

# Liga a verificação de chaves estrangeiras (FOREIGN KEY) em cada nova conexão.
def _ligar_chaves_estrangeiras(conexao_dbapi, _registro):
    # No SQLite a verificação de FK vem desligada e vale por conexão.
    conexao_dbapi.execute("PRAGMA foreign_keys = ON")

# Implementa as duas portas de persistência do domínio sobre o banco SQLite.
class SqliteRepository(ISerieRepository, IExecucaoRepository):
    # Abre a conexão com o banco (padrão: config.BANCO) e registra a função que liga
    # as chaves estrangeiras a cada conexão.
    def __init__(self, caminho: Path = None):
        self.caminho = Path(caminho or config.BANCO)
        self.engine = create_engine(f"sqlite:///{self.caminho}")
        event.listen(self.engine, "connect", _ligar_chaves_estrangeiras)

    # Cria as cinco tabelas (idempotente: CREATE TABLE IF NOT EXISTS).
    # Como funciona: lê o schema.sql, tira as linhas de comentário (--), separa os
    # comandos pelo ';' e executa um por um.
    def criar_esquema(self):
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        linhas = [
            l for l in ESQUEMA.read_text(encoding="utf-8").splitlines()
            if not l.strip().startswith("--")
        ]
        comandos = [c.strip() for c in "\n".join(linhas).split(";") if c.strip()]
        with self.engine.begin() as con:
            for comando in comandos:
                con.execute(text(comando))

    # ISerieRepository

    # Grava a série completa (inclusive os meses pandêmicos, bloco 0).
    # Reimportar o mesmo mês atualiza o registro em vez de duplicá-lo.
    # Como funciona: cadastra o aeroporto (se ainda não existir), pega o id dele e
    # grava todos os meses de uma vez; o ON CONFLICT ... DO UPDATE atualiza o mês
    # que já existe.
    def salvar_serie(self, serie: SerieTemporal):
        with self.engine.begin() as con:
            con.execute(
                text(
                    "INSERT INTO tb_ger_aeroportos (cd_iata, cd_icao, nm_aeroporto) "
                    "VALUES (:iata, :icao, :nome) ON CONFLICT (cd_icao) DO NOTHING"
                ),
                {"iata": config.IATA, "icao": config.ICAO, "nome": config.AEROPORTO_NOME},
            )
            id_aeroporto = con.execute(
                text("SELECT id_aeroporto FROM tb_ger_aeroportos WHERE cd_icao = :icao"),
                {"icao": config.ICAO},
            ).scalar_one()
            con.execute(
                text("""
                    INSERT INTO tb_ser_registros_mensais
                        (id_aeroporto, nr_ano, nr_mes, qt_desembarques,
                         fl_alta_temporada, fl_pandemia, tp_bloco_temporal)
                    VALUES (:id_aeroporto, :ano, :mes, :qt, :at, :pd, :bloco)
                    ON CONFLICT (id_aeroporto, nr_ano, nr_mes) DO UPDATE SET
                        qt_desembarques   = excluded.qt_desembarques,
                        fl_alta_temporada = excluded.fl_alta_temporada,
                        fl_pandemia       = excluded.fl_pandemia,
                        tp_bloco_temporal = excluded.tp_bloco_temporal
                """),
                [
                    {"id_aeroporto": id_aeroporto, "ano": r.ano, "mes": r.mes,
                     "qt": int(round(r.desembarques)), "at": int(r.alta_temporada),
                     "pd": int(r.pandemia), "bloco": int(r.bloco_temporal)}
                    for r in serie.registros
                ],
            )

    # Lê do banco a série do aeroporto de Foz, em ordem de ano e mês, e transforma
    # cada linha num RegistroMensal.
    def carregar_serie(self) -> SerieTemporal:
        with self.engine.connect() as con:
            linhas = con.execute(
                text("""
                    SELECT r.nr_ano, r.nr_mes, r.qt_desembarques, r.fl_alta_temporada,
                           r.fl_pandemia, r.tp_bloco_temporal
                    FROM tb_ser_registros_mensais r
                    JOIN tb_ger_aeroportos a ON a.id_aeroporto = r.id_aeroporto
                    WHERE a.cd_icao = :icao
                    ORDER BY r.nr_ano, r.nr_mes
                """),
                {"icao": config.ICAO},
            ).all()
        return SerieTemporal([
            RegistroMensal(
                ano=ano, mes=mes, desembarques=float(qt),
                alta_temporada=bool(at), pandemia=bool(pd),
                bloco_temporal=BlocoTemporalEnum(bloco),
            )
            for ano, mes, qt, at, pd, bloco in linhas
        ])

    # IExecucaoRepository (RF19, RF21)

    # Grava a execução e, por composição, as suas métricas. Atribui e
    # devolve o id gerado pelo banco.
    # Como funciona: insere a linha da execução, pega o id gerado (lastrowid) e
    # insere as métricas ligadas a esse id, tudo na mesma transação.
    def salvar_execucao(self, execucao: Execucao) -> int:
        with self.engine.begin() as con:
            resultado = con.execute(
                text("""
                    INSERT INTO tb_exp_execucoes
                        (tp_modelo, ds_hiperparametros, nr_semente_aleatoria,
                         dt_hora_execucao, nr_tamanho_janela_entrada,
                         ds_periodo_teste, ds_caminho_modelo)
                    VALUES (:modelo, :hiper, :semente, :dt, :janela, :periodo, :caminho)
                """),
                {
                    "modelo": TipoModeloEnum(execucao.tipo_modelo).value,
                    "hiper": json.dumps(execucao.hiperparametros, ensure_ascii=False),
                    "semente": execucao.semente_aleatoria,
                    "dt": execucao.data_hora_execucao.isoformat(timespec="seconds"),
                    "janela": execucao.tamanho_janela_entrada,
                    "periodo": execucao.periodo_teste,
                    "caminho": execucao.caminho_modelo,
                },
            )
            execucao.id = resultado.lastrowid
            if execucao.metricas:
                con.execute(
                    text("""
                        INSERT INTO tb_exp_metricas
                            (id_execucao, tp_escopo_avaliacao, tp_metrica, vl_metrica)
                        VALUES (:id, :escopo, :tipo, :valor)
                    """),
                    [
                        {"id": execucao.id,
                         "escopo": EscopoAvaliacaoEnum(m.escopo_avaliacao).value,
                         "tipo": TipoMetricaEnum(m.tipo_metrica).value,
                         "valor": m.valor_metrica}
                        for m in execucao.metricas
                    ],
                )
        return execucao.id

    # Grava as previsões mês a mês da execução (valor previsto, valor real e alta
    # temporada) na tb_exp_previsoes.
    def salvar_previsoes(self, id_execucao: int, previsoes) -> None:
        if not previsoes:
            return
        with self.engine.begin() as con:
            con.execute(
                text("""
                    INSERT INTO tb_exp_previsoes
                        (id_execucao, nr_ano, nr_mes, vl_previsto, vl_real,
                         fl_alta_temporada)
                    VALUES (:id, :ano, :mes, :prev, :real, :at)
                """),
                [
                    {"id": id_execucao, "ano": p.ano, "mes": p.mes,
                     "prev": p.valor_previsto, "real": p.valor_real,
                     "at": int(p.alta_temporada)}
                    for p in previsoes
                ],
            )

    # Transforma uma linha da tb_exp_execucoes num objeto Execucao e busca as
    # métricas dela; as previsões só são buscadas quando com_previsoes for True.
    def _montar_execucao(self, con, linha, com_previsoes: bool) -> Execucao:
        (id_exec, modelo, hiper, semente, dt, janela, periodo, caminho) = linha
        execucao = Execucao(
            id=id_exec,
            tipo_modelo=TipoModeloEnum(modelo),
            hiperparametros=json.loads(hiper) if hiper else {},
            semente_aleatoria=semente,
            data_hora_execucao=datetime.fromisoformat(dt),
            tamanho_janela_entrada=janela,
            periodo_teste=periodo or "",
            caminho_modelo=caminho or "",
        )
        execucao.metricas = [
            Metrica(EscopoAvaliacaoEnum(escopo), TipoMetricaEnum(tipo), valor)
            for escopo, tipo, valor in con.execute(
                text("SELECT tp_escopo_avaliacao, tp_metrica, vl_metrica "
                     "FROM tb_exp_metricas WHERE id_execucao = :id "
                     "ORDER BY id_metrica"),
                {"id": id_exec},
            )
        ]
        if com_previsoes:
            execucao.previsoes = [
                Previsao(ano, mes, prev, real, bool(at))
                for ano, mes, prev, real, at in con.execute(
                    text("SELECT nr_ano, nr_mes, vl_previsto, vl_real, fl_alta_temporada "
                         "FROM tb_exp_previsoes WHERE id_execucao = :id "
                         "ORDER BY nr_ano, nr_mes"),
                    {"id": id_exec},
                )
            ]
        return execucao

    # Colunas lidas da tb_exp_execucoes, na ordem que _montar_execucao() espera.
    _COLUNAS_EXECUCAO = (
        "id_execucao, tp_modelo, ds_hiperparametros, nr_semente_aleatoria, "
        "dt_hora_execucao, nr_tamanho_janela_entrada, ds_periodo_teste, "
        "ds_caminho_modelo"
    )

    # Execuções com as métricas (sem as previsões), da mais recente para
    # a mais antiga; é o que a tela de Histórico exibe.
    def listar_execucoes(self) -> List[Execucao]:
        with self.engine.connect() as con:
            linhas = con.execute(text(
                f"SELECT {self._COLUNAS_EXECUCAO} FROM tb_exp_execucoes "
                "ORDER BY id_execucao DESC"
            )).all()
            return [self._montar_execucao(con, l, com_previsoes=False) for l in linhas]

    # Execução completa, com métricas e previsões; dá erro (KeyError) se o id
    # não existir.
    def buscar_execucao(self, id_execucao: int) -> Execucao:
        with self.engine.connect() as con:
            linha = con.execute(
                text(f"SELECT {self._COLUNAS_EXECUCAO} FROM tb_exp_execucoes "
                     "WHERE id_execucao = :id"),
                {"id": id_execucao},
            ).first()
            if linha is None:
                raise KeyError(f"execução {id_execucao} não encontrada")
            return self._montar_execucao(con, linha, com_previsoes=True)


if __name__ == "__main__":
    from infraestrutura.anac_csv_loader import carregar_serie

    repo = SqliteRepository()
    repo.criar_esquema()
    repo.salvar_serie(carregar_serie())

    serie = repo.carregar_serie()
    print(f"Banco: {repo.caminho}")
    print(f"Série relida do banco: {serie.resumo()}")
    with repo.engine.connect() as con:
        tabelas = con.execute(text(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        )).scalars().all()
        for tabela in tabelas:
            n = con.execute(text(f"SELECT COUNT(*) FROM {tabela}")).scalar_one()
            print(f"  {tabela:<28}{n:>4} linhas")
        print("Meses por bloco (0 = pandemia):")
        for bloco, n, ini, fim in con.execute(text(
            "SELECT tp_bloco_temporal, COUNT(*), MIN(nr_ano*100+nr_mes), MAX(nr_ano*100+nr_mes) "
            "FROM tb_ser_registros_mensais GROUP BY tp_bloco_temporal"
        )):
            print(f"  bloco {bloco}: {n:>2} meses ({ini // 100}-{ini % 100:02d} a {fim // 100}-{fim % 100:02d})")
