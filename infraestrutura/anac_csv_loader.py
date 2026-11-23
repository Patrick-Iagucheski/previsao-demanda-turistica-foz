import re
import sys
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

# permite rodar como script solto (python infraestrutura/anac_csv_loader.py)
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from dominio.entidades import BlocoTemporalEnum, RegistroMensal, SerieTemporal
from dominio.regras import bloco_temporal, eh_alta_temporada, eh_pandemia

# Adaptador de infraestrutura: lê os CSV brutos da ANAC e devolve a série
# mensal de DESEMBARQUES de passageiros do Aeroporto de Foz do Iguaçu.
#
# Estrutura dos arquivos da ANAC (Movimentações Aeroportuárias):
#   - 1ª linha: metadado "Atualizado em: ..." em parte dos meses (detectado)
#   - cabeçalho real; em 2022-07 as colunas vêm em outra ordem (lidas por nome)
#   - separador ';', campos entre aspas, codificação UTF-8 com BOM
#   - cada linha = um movimento de aeronave (pouso ou decolagem)
#
# Regra (RN01): desembarques em Foz = passageiros dos voos que DECOLAM ("D")
# de outro aeroporto com destino a SBFI (coluna NR_VOO_OUTRO_AEROPORTO),
# somando as três colunas de passageiros. É o "método A" da Seção 4.6.5 do
# TCC, validado contra o método direto (Foz como aeroporto de referência +
# pouso), aplicável só a partir de abr/2022: concordância de 98,28% em 45
# meses (infraestrutura/validacao_extracao.py).
#
# Uso: AnacCsvLoader().carregar_serie(origem), a classe do diagrama de
# classes. A função carregar_serie() é mantida como atalho equivalente.

# Colunas lidas de cada CSV (as demais são ignoradas); usadas em _ler_um_csv() e
# conferidas em _colunas_faltando().
COLS_USADAS = [
    "ANO", "MES", "NR_AEROPORTO_REFERENCIA", "NR_MOVIMENTO_TIPO",
    "NR_VOO_OUTRO_AEROPORTO",
    *config.COLUNAS_PAX,
]

# Os arquivos da ANAC variam entre meses: alguns trazem a linha de metadado
# 'Atualizado em:' no topo, outros começam direto no cabeçalho, e há meses
# (ex.: 2022-07) com as colunas em ordem diferente. A leitura por nome
# (usecols) resolve a ordem; aqui só decidimos se há linha a pular.
#
# Regra robusta: se a primeira linha já contém um nome de coluna conhecido,
# ela é o cabeçalho (skiprows=0); senão, é o metadado (skiprows=1).
def _detectar_skiprows(caminho: Path) -> int:
    with open(caminho, encoding="utf-8-sig") as f:
        primeira = f.readline().upper()
    return 0 if "NR_MOVIMENTO_TIPO" in primeira else 1

# Lê um CSV mensal da ANAC, já filtrado para os voos com destino a SBFI.
# Como funciona: lê só as colunas de COLS_USADAS, mantém as decolagens de outros
# aeroportos rumo a Foz e soma as três colunas de passageiros na coluna
# desembarques.
def _ler_um_csv(caminho: Path) -> pd.DataFrame:
    df = pd.read_csv(
        caminho,
        sep=";",
        skiprows=_detectar_skiprows(caminho),   # pula 'Atualizado em:' só se existir
        encoding="utf-8-sig",                    # trata o BOM
        usecols=COLS_USADAS,
        dtype={
            "NR_AEROPORTO_REFERENCIA": str,
            "NR_MOVIMENTO_TIPO": str,
            "NR_VOO_OUTRO_AEROPORTO": str,
        },
    )
    # DESEMBARQUES em Foz reconstruídos pelo "outro aeroporto":
    # um voo que DECOLA (D) de outro aeroporto rumo a SBFI leva os passageiros
    # que vão DESEMBARCAR em Foz. Este método funciona em TODOS os anos, pois
    # não depende de Foz constar como aeroporto de referência (o que só ocorre
    # a partir de abr/2022). Validado contra o método direto (Foz-referência +
    # pouso) de abr/2022 a dez/2025: concordância de 98,28%. A pequena diferença
    # são voos internacionais cuja origem não está na base brasileira da ANAC.
    df = df[
        (df["NR_VOO_OUTRO_AEROPORTO"] == config.ICAO)
        & (df["NR_MOVIMENTO_TIPO"] == config.MOVIMENTO_DECOLAGEM_ORIGEM)
    ].copy()

    # soma os passageiros das três colunas, tratando ausentes como 0
    for c in config.COLUNAS_PAX:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["desembarques"] = df[config.COLUNAS_PAX].sum(axis=1)
    return df[["ANO", "MES", "desembarques"]]


# Nome exato da ANAC: Movimentacoes_Aeroportuarias_AAAAMM.csv (aceita também a
# cópia "AAAAMM (1).csv" que o navegador cria ao baixar o mesmo arquivo de novo).
PADRAO_NOME = re.compile(r"^Movimentacoes_Aeroportuarias_(\d{4})(\d{2})(?: \(\d+\))?\.csv$", re.I)

# Colunas usadas na extração que não estão no cabeçalho do arquivo.
# Lê só o cabeçalho (pulando a linha 'Atualizado em:', se houver).
def _colunas_faltando(caminho: Path) -> List[str]:
    with open(caminho, encoding="utf-8-sig") as f:
        for _ in range(_detectar_skiprows(caminho)):
            f.readline()
        cabecalho = {c.strip().strip('"').upper() for c in f.readline().split(";")}
    return [c for c in COLS_USADAS if c not in cabecalho]

# Os CSV mensais da ANAC na origem (pasta padrão: config.ANAC_ORIGEM) e nas
# pastas por ano logo abaixo dela, em ordem. Antes de qualquer leitura confere
# o nome no padrão da ANAC, um único arquivo por mês (cópias somariam os
# passageiros em dobro) e o cabeçalho com todas as colunas usadas. Usada também
# pela validação da extração, para que os dois métodos leiam os mesmos arquivos.
def listar_csvs(origem: Path = None) -> List[Path]:
    origem = Path(origem or config.ANAC_ORIGEM)

    # só a pasta e um nível abaixo (ex.: brutos\2019\...), nunca o disco inteiro
    por_mes = {}
    for arq in [*origem.glob("*.csv"), *origem.glob("*/*.csv")]:
        m = PADRAO_NOME.match(arq.name)
        if m and 1 <= int(m.group(2)) <= 12:
            por_mes.setdefault((int(m.group(1)), int(m.group(2))), []).append(arq)
    # As mensagens vão direto para a tela: uma linha por parágrafo.
    if not por_mes:
        raise FileNotFoundError(
            f"Nenhum arquivo da ANAC foi encontrado em {origem}.\n"
            "Informe a pasta que contém os CSV mensais (Movimentacoes_Aeroportuarias_AAAAMM.csv), "
            "diretamente ou em pastas por ano. A pasta padrão é dados/brutos; outra pasta pode ser "
            "digitada no campo acima ou indicada pela variável de ambiente ANAC_ORIGEM.")

    repetidos = [f"{a}-{m:02d}: " + " e ".join(str(p) for p in ps)
                 for (a, m), ps in sorted(por_mes.items()) if len(ps) > 1]
    if repetidos:
        raise ValueError(
            "Há meses com mais de um arquivo, e a importação somaria os passageiros em dobro. "
            "Deixe apenas um arquivo para cada mês:\n" + "\n".join(repetidos))

    arquivos = [ps[0] for _, ps in sorted(por_mes.items())]
    for arq in arquivos:
        faltando = _colunas_faltando(arq)
        if faltando:
            raise ValueError(
                f"O arquivo {arq.name} não tem as colunas usadas na extração "
                f"({', '.join(faltando)}).\n"
                "Verifique se ele é mesmo um arquivo de movimentação aeroportuária da ANAC.")
    return arquivos


# Tipo da função de progresso: ao_ler(lidos, total, nome_do_arquivo, meses_do_arquivo)
AoLer = Callable[[int, int, str, List[RegistroMensal]], None]

# Um mês da série, já com as marcações das regras RN02 e RN03.
# Como funciona: usa eh_alta_temporada(), eh_pandemia() e bloco_temporal() de
# regras.py para marcar o mês.
def _registro(ano: int, mes: int, desembarques: float) -> RegistroMensal:
    return RegistroMensal(
        ano=ano,
        mes=mes,
        desembarques=float(desembarques),
        alta_temporada=eh_alta_temporada(mes),
        pandemia=eh_pandemia(ano, mes),
        bloco_temporal=BlocoTemporalEnum(bloco_temporal(ano, mes)),
    )

# Soma os desembarques por ano/mês (um arquivo é um mês, mas garantimos robustez)
# e transforma cada mês num RegistroMensal com _registro().
def _por_mes(df: pd.DataFrame) -> List[RegistroMensal]:
    mensal = (
        df.groupby(["ANO", "MES"], as_index=False)["desembarques"].sum()
        .sort_values(["ANO", "MES"])
    )
    return [_registro(int(linha["ANO"]), int(linha["MES"]), linha["desembarques"])
            for _, linha in mensal.iterrows()]

# Varre todos os CSV da origem e consolida a série mensal de desembarques.
# Se informado, ao_ler(lidos, total, nome_do_arquivo, meses_do_arquivo) é
# chamado após cada arquivo, com os meses que ele contém já somados; a tela
# de importação usa isso para mostrar o progresso e montar o gráfico durante
# a leitura.
# Como funciona: lista os arquivos (listar_csvs()), lê um por um (_ler_um_csv()),
# junta tudo e soma por mês (_por_mes()).
def carregar_serie(origem: Path = None, ao_ler: Optional[AoLer] = None) -> SerieTemporal:
    arquivos = listar_csvs(origem)

    partes = []
    for i, arq in enumerate(arquivos, 1):
        df = _ler_um_csv(arq)
        partes.append(df)
        if ao_ler:
            ao_ler(i, len(arquivos), arq.name, _por_mes(df))
    bruto = pd.concat(partes, ignore_index=True)

    serie = SerieTemporal()
    serie.registros.extend(_por_mes(bruto))
    return serie.ordenar()

# Adaptador de leitura da ANAC (diagrama de classes, camada de
# infraestrutura). Injetado no IngestaoService pelo ponto de composição.
class AnacCsvLoader:
    # Retorna a série lida dos CSV; só repassa para a função carregar_serie() acima.
    def carregar_serie(self, origem: Path = None, ao_ler: Optional[AoLer] = None) -> SerieTemporal:
        return carregar_serie(origem, ao_ler)


if __name__ == "__main__":
    serie = carregar_serie()
    print(f"Série carregada: {serie.resumo()}")
    print(f"{'Mês':<10}{'Desembarques':>14}  {'Alta temp.':<10}{'Pandemia'}")
    print("-" * 48)
    for r in serie.registros:
        flag_at = "SIM" if r.alta_temporada else ""
        flag_pd = "excluído" if r.pandemia else ""
        print(f"{r.rotulo:<10}{r.desembarques:>14,.0f}  {flag_at:<10}{flag_pd}")

    validos = [r for r in serie.registros if not r.pandemia]
    print("-" * 48)
    print(f"Total de meses: {len(serie)}")
    print(f"Meses fora da pandemia (utilizáveis): {len(validos)}")
