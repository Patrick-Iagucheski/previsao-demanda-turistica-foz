-- =====================================================================
-- Esquema físico do banco de experimentos (SQLite)
-- TCC II — Previsão da Demanda Turística em Foz do Iguaçu
--
-- Convenção (Seção 4.4.2 do TCC):
--   tabelas -> tb_<módulo>_<entidade no plural>   (ger · ser · exp)
--   colunas -> <prefixo>_<nome>
--     id identificador · cd código · nm nome · nr número inteiro
--     qt quantidade · vl valor decimal · fl flag · dt data/hora
--     ds descrição/texto · tp domínio fechado (enumeração, via CHECK)
--
-- SQL compatível com o PostgreSQL (RNF11), com poucas adaptações: lá, remove-se o
-- PRAGMA abaixo, INTEGER PRIMARY KEY vira GENERATED ALWAYS AS IDENTITY, REAL vira
-- DOUBLE PRECISION e os CHECK de tp_ podem virar ENUM.
-- =====================================================================

PRAGMA foreign_keys = ON;

-- Cadastro geral: aeroportos (dimensão)
CREATE TABLE IF NOT EXISTS tb_ger_aeroportos (
    id_aeroporto  INTEGER PRIMARY KEY,
    cd_iata       TEXT    NOT NULL UNIQUE,
    cd_icao       TEXT    NOT NULL UNIQUE,
    nm_aeroporto  TEXT    NOT NULL
);

-- Série temporal: um registro por aeroporto e mês (RN01, RN02, RN03)
CREATE TABLE IF NOT EXISTS tb_ser_registros_mensais (
    id_registro_mensal  INTEGER PRIMARY KEY,
    id_aeroporto        INTEGER NOT NULL
                        REFERENCES tb_ger_aeroportos (id_aeroporto),
    nr_ano              INTEGER NOT NULL,
    nr_mes              INTEGER NOT NULL CHECK (nr_mes BETWEEN 1 AND 12),
    qt_desembarques     INTEGER NOT NULL CHECK (qt_desembarques >= 0),
    fl_alta_temporada   INTEGER NOT NULL CHECK (fl_alta_temporada IN (0, 1)),
    fl_pandemia         INTEGER NOT NULL CHECK (fl_pandemia IN (0, 1)),
    tp_bloco_temporal   INTEGER NOT NULL CHECK (tp_bloco_temporal IN (0, 1, 2)),
    UNIQUE (id_aeroporto, nr_ano, nr_mes)
);

-- Experimentos: uma linha por execução de treinamento (RF19, RNF01, RNF08)
CREATE TABLE IF NOT EXISTS tb_exp_execucoes (
    id_execucao                INTEGER PRIMARY KEY,
    tp_modelo                  TEXT    NOT NULL
                               CHECK (tp_modelo IN ('naive', 'random_forest', 'lstm')),
    ds_hiperparametros         TEXT,
    nr_semente_aleatoria       INTEGER NOT NULL,
    dt_hora_execucao           TEXT    NOT NULL,
    nr_tamanho_janela_entrada  INTEGER,
    ds_periodo_teste           TEXT,
    ds_caminho_modelo          TEXT
);

-- Previsões de cada execução, com o valor real materializado (RF15, RF20)
CREATE TABLE IF NOT EXISTS tb_exp_previsoes (
    id_previsao        INTEGER PRIMARY KEY,
    id_execucao        INTEGER NOT NULL
                       REFERENCES tb_exp_execucoes (id_execucao) ON DELETE CASCADE,
    nr_ano             INTEGER NOT NULL,
    nr_mes             INTEGER NOT NULL CHECK (nr_mes BETWEEN 1 AND 12),
    vl_previsto        REAL    NOT NULL,
    vl_real            REAL,
    fl_alta_temporada  INTEGER NOT NULL CHECK (fl_alta_temporada IN (0, 1)),
    UNIQUE (id_execucao, nr_ano, nr_mes)
);

-- Métricas de cada execução, por escopo (RF16)
CREATE TABLE IF NOT EXISTS tb_exp_metricas (
    id_metrica           INTEGER PRIMARY KEY,
    id_execucao          INTEGER NOT NULL
                         REFERENCES tb_exp_execucoes (id_execucao) ON DELETE CASCADE,
    tp_escopo_avaliacao  TEXT    NOT NULL
                         CHECK (tp_escopo_avaliacao IN ('global', 'alta_temporada')),
    tp_metrica           TEXT    NOT NULL CHECK (tp_metrica IN ('MAE', 'RMSE', 'MAPE')),
    vl_metrica           REAL    NOT NULL,
    UNIQUE (id_execucao, tp_escopo_avaliacao, tp_metrica)
);
