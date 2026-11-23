import os, shutil, sys, tempfile, time
from pathlib import Path

# Recarrega os modelos salvos das 11 execuções (sem retreinar) e compara as previsões com
# as gravadas, sobre uma CÓPIA do banco (o real não é tocado). Esperado: diferença nula nas
# 132 previsões (RF15 e RF19). Resultado na Seção 5.7 do TCC.

# Pasta raiz do projeto e pasta temporária com a cópia do banco (o teste usa a
# cópia, nunca o banco real).
RAIZ = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="tcc_prever_"))
shutil.copy(RAIZ / "dados" / "experimentos.db", TMP / "banco.db")
os.environ["TCC_BANCO"] = str(TMP / "banco.db")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "apresentacao"))
sys.stdout.reconfigure(encoding="utf-8")

import app as m  # noqa: E402

total, falhas = 0, 0
# Para cada uma das 11 execuções: recarrega o modelo salvo, prevê de novo e mede a
# maior diferença em relação às previsões gravadas.
for i in range(1, 12):
    ex = m._buscar(i)
    t0 = time.time()
    novas = {p.rotulo: p.valor_previsto for p in m._previsao.prever(ex)}
    dif = max(abs(p.valor_previsto - novas[p.rotulo]) for p in ex.previsoes)
    total += len(ex.previsoes)
    falhas += dif != 0
    print(f"  [{'ok' if dif == 0 else 'FALHOU'}] execução {i:2d} ({ex.tipo_modelo.name}, semente {ex.semente_aleatoria}): "
          f"{len(ex.previsoes)} previsões, maior diferença {dif:.6f} ({time.time() - t0:.1f} s)")
print(f"RESULTADO: {total} previsões; " + ("tudo certo" if not falhas else f"{falhas} execução(ões) com diferença"))
