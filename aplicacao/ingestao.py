# -*- coding: utf-8 -*-
"""
Serviço de ingestão (camada de aplicação) — RF01 a RF05.

    importar(pasta) -> SerieTemporal
    validar(serie)  -> relatório de integridade (RF04)

Lê os arquivos da ANAC pelo carregador recebido, consolida a série mensal com
as flags de negócio e a grava pelo repositório (ISerieRepository). Assim como
o TreinamentoService, não conhece os adaptadores concretos: o carregador e o
repositório são injetados pelo ponto de composição (a aplicação Flask).
"""
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dominio.entidades import SerieTemporal
from dominio.portas import ISerieRepository
from dominio.regras import eh_pandemia


class IngestaoService:

    def __init__(self, carregador: Callable[..., SerieTemporal],
                 repositorio: ISerieRepository):
        self.carregador = carregador
        self.repositorio = repositorio

    def importar(self, pasta: Optional[Path] = None,
                 ao_ler: Optional[Callable[..., None]] = None) -> SerieTemporal:
        """Lê os CSV (pasta padrão: config.ANAC_ORIGEM) e persiste a série.
        Reimportar os mesmos meses atualiza os registros, sem duplicá-los.
        ao_ler, se informado, recebe o progresso da leitura arquivo a arquivo
        (e os meses de cada arquivo, já somados)."""
        argumentos = {"ao_ler": ao_ler} if ao_ler else {}
        serie = self.carregador(pasta, **argumentos) if pasta else self.carregador(**argumentos)
        self.repositorio.salvar_serie(serie)
        return serie

    @staticmethod
    def validar(serie: SerieTemporal) -> Dict[str, object]:
        """RF04 — verifica continuidade cronológica, valores inválidos ou
        negativos e duplicidades.

        Lacunas dentro do intervalo pandêmico são registradas, mas não
        comprometem a série: esse intervalo é excluído da modelagem (RN02).
        A série é considerada íntegra quando não há duplicidade, nem valor
        inválido, nem lacuna fora do intervalo pandêmico.
        """
        regs = sorted(serie.registros, key=lambda r: (r.ano, r.mes))
        contagem = Counter((r.ano, r.mes) for r in regs)
        duplicidades = [f"{a}-{m:02d}" for (a, m), n in sorted(contagem.items()) if n > 1]
        invalidos = [r.rotulo for r in regs
                     if r.desembarques is None or math.isnan(r.desembarques)
                     or r.desembarques < 0]
        lacunas = []
        if regs:
            presentes = {r.indice_absoluto for r in regs}
            for indice in range(regs[0].indice_absoluto, regs[-1].indice_absoluto + 1):
                if indice not in presentes:
                    ano, mes = divmod(indice, 12)
                    mes += 1
                    lacunas.append({"mes": f"{ano}-{mes:02d}",
                                    "na_pandemia": eh_pandemia(ano, mes)})
        fora = [l["mes"] for l in lacunas if not l["na_pandemia"]]
        return {
            "meses": len(regs),
            "duplicidades": duplicidades,
            "valores_invalidos": invalidos,
            "lacunas": lacunas,
            "lacunas_fora_da_pandemia": fora,
            "integra": not duplicidades and not invalidos and not fora,
        }


if __name__ == "__main__":
    from infraestrutura.repositorios.sqlite_repo import SqliteRepository
    r = IngestaoService.validar(SqliteRepository().carregar_serie())
    print(f"meses: {r['meses']} | duplicidades: {r['duplicidades']} | "
          f"inválidos: {r['valores_invalidos']} | lacunas: {r['lacunas']} | "
          f"íntegra: {r['integra']}")
