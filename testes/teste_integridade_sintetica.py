import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Testa a rotina de integridade (RF04, IngestaoService.validar) com séries sintéticas,
# sem banco: cada problema deve ser apontado e marcar a série como não íntegra; uma lacuna
# dentro do intervalo pandêmico só é registrada (RN02). Resultado na Seção 5.7 do TCC.
sys.stdout.reconfigure(encoding="utf-8")

from aplicacao.ingestao import IngestaoService  # noqa: E402
from dominio.entidades import RegistroMensal, SerieTemporal  # noqa: E402

# Monta uma SerieTemporal de teste a partir de uma lista de (ano, mês, desembarques).
def serie(meses):
    return SerieTemporal([RegistroMensal(a, m, v) for a, m, v in meses])

# Série base: os 12 meses de 2024, todos com 90.000 passageiros (íntegra).
base = [(2024, m, 90000.0) for m in range(1, 13)]
# caso: (meses, íntegra esperada, campo que deve apontar o problema)
casos = {
    "íntegra": (base, True, None),
    "mês duplicado": (base + [(2024, 5, 1.0)], False, "duplicidades"),
    "valor negativo": (base[:3] + [(2024, 4, -5.0)] + base[4:], False, "valores_invalidos"),
    "valor ausente (NaN)": (base[:3] + [(2024, 4, float("nan"))] + base[4:], False, "valores_invalidos"),
    "lacuna fora da pandemia": (base[:5] + base[6:], False, "lacunas_fora_da_pandemia"),
    "lacuna na pandemia": ([(2021, m, 1.0) for m in (1, 2, 4)], True, None),
}
falhas = 0
# Roda a validação em cada caso e confere se o resultado é o esperado.
for nome, (meses, integra, campo) in casos.items():
    r = IngestaoService.validar(serie(meses))
    ok = r["integra"] is integra and (campo is None or bool(r[campo]))
    falhas += not ok
    print(f"  [{'ok' if ok else 'FALHOU'}] {nome:25s} íntegra={r['integra']}"
          + (f" | {campo}={r[campo]}" if campo else ""))
print("RESULTADO:", "tudo certo" if not falhas else f"{falhas} falha(s)")
