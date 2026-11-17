# -*- coding: utf-8 -*-
"""
LSTM — adaptador de IForecaster (RF13), com ajuste de hiperparâmetros por
validação temporal (RF14).

Arquitetura enxuta, como prevê o planejamento (docs/planejamento_sistema.md, Seção 4.4): uma camada LSTM
de 8 a 32 unidades, dropout e parada antecipada (EarlyStopping). Com a
amostra reduzida, redes maiores apenas memorizariam o treino.

Entrada sequencial: cada amostra é a sequência dos `lookback` meses
anteriores ao mês previsto, e cada passo traz o desembarque normalizado e o
calendário daquele mês (seno e cosseno do mês e indicador de alta
temporada). As sequências são montadas a partir das mesmas defasagens que o
Random Forest recebe, de modo que os dois modelos usam o mesmo conjunto de
informação e as mesmas linhas de treino e de teste (RN07). Pelo mesmo motivo
das defasagens (ver aplicacao/preprocessamento.py), o lookback é 3, e não
12: uma janela de 12 meses não caberia no bloco pré-pandêmico sem
atravessar o intervalo excluído (RNF03).

Normalização min-max do desembarque, estimada somente no treino (RN05) — a
convergência da LSTM é sensível à escala das entradas (Seção 3.7 do TCC).
Limitação conhecida e declarada no TCC (Seção 3.7 e Limitações do Cap. 5): os
limites são estimados sobre o treino completo ANTES da divisão nas partições da
validação cruzada; na 1ª partição eles refletem valores do trecho de validação
(nas outras três coincidem). Não envolve o teste; afeta só a seleção de
hiperparâmetros. Correção possível: estimar os limites dentro de cada partição.

Ajuste de hiperparâmetros: busca em grade sobre unidades e dropout, com
validação cruzada temporal (TimeSeriesSplit). Em cada partição, a parada
antecipada usa a validação para escolher o número de épocas; o modelo final
é reajustado sobre o treino inteiro com a configuração vencedora e com o
número médio de épocas encontrado nas partições — sem tocar no teste (RN04).

Reprodutibilidade (RNF01): a semente fixa o Python, o NumPy e o TensorFlow,
e as operações do TensorFlow são forçadas a ser determinísticas. Como redes
neurais ainda variam com a inicialização, o experimento roda a LSTM com
várias sementes e reporta a dispersão.
"""
import json
import os
import sys
from pathlib import Path
from typing import List

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dominio.entidades import RegistroMensal, SerieTemporal, TipoModeloEnum
from dominio.portas import IForecaster
from dominio.regras import eh_alta_temporada, mes_ciclico

GRADE_PADRAO = {"unidades": [8, 16, 32], "dropout": [0.0, 0.2]}
N_PARTICOES = 4
LOOKBACK = 3
MAX_EPOCAS = 400
PACIENCIA = 40
TAXA_APRENDIZADO = 0.005
TAMANHO_LOTE = 4


def _calendario(mes: int) -> List[float]:
    """Os mesmos atributos de calendário do RF08, lidos das regras do domínio."""
    return [*mes_ciclico(mes), float(eh_alta_temporada(mes))]


class LSTMModel(IForecaster):

    extensao = ".keras"

    def __init__(self, semente: int = 42, grade: dict = None,
                 n_particoes: int = N_PARTICOES, lookback: int = LOOKBACK):
        self.semente = semente
        self.grade = grade or GRADE_PADRAO
        self.n_particoes = n_particoes
        self.lookback = lookback
        self.unidades = None
        self.dropout = None
        self.epocas = None
        self._minimo = None
        self._maximo = None
        self._modelo = None
        self._busca = {}

    def nome(self) -> str:
        return TipoModeloEnum.LSTM.value

    @property
    def tamanho_janela_entrada(self):
        return self.lookback

    def hiperparametros(self) -> dict:
        return {
            "unidades": self.unidades,
            "dropout": self.dropout,
            "lookback": self.lookback,
            "epocas": self.epocas,
            "taxa_aprendizado": TAXA_APRENDIZADO,
            "tamanho_lote": TAMANHO_LOTE,
            "validacao": f"TimeSeriesSplit({self.n_particoes})",
            "criterio_selecao": "MAE",
            **self._busca,
        }

    # ------------------------------------------------------------------
    # Montagem das sequências
    # ------------------------------------------------------------------
    def _normalizar(self, valor: float) -> float:
        return (valor - self._minimo) / (self._maximo - self._minimo)

    def _desnormalizar(self, valor: float) -> float:
        return valor * (self._maximo - self._minimo) + self._minimo

    def _sequencia(self, r: RegistroMensal) -> List[List[float]]:
        """Os `lookback` meses anteriores a r, do mais antigo ao mais recente,
        lidos das defasagens já calculadas dentro do bloco (RNF03, RN06)."""
        passos = []
        for k in range(self.lookback, 0, -1):
            mes = (r.mes - k - 1) % 12 + 1
            passos.append([self._normalizar(r.atributos[f"lag_{k}"]), *_calendario(mes)])
        return passos

    def _tensores(self, registros: List[RegistroMensal]):
        import numpy as np
        X = np.array([self._sequencia(r) for r in registros], dtype="float32")
        y = np.array([self._normalizar(r.desembarques) for r in registros], dtype="float32")
        return X, y

    # ------------------------------------------------------------------
    # Rede
    # ------------------------------------------------------------------
    def _semear(self):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(self.semente)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

    def _construir(self, unidades: int, dropout: float, n_atributos: int):
        import tensorflow as tf
        modelo = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.lookback, n_atributos)),
            tf.keras.layers.LSTM(unidades, dropout=dropout),
            tf.keras.layers.Dense(1),
        ])
        modelo.compile(optimizer=tf.keras.optimizers.Adam(TAXA_APRENDIZADO), loss="mse")
        return modelo

    def _ajustar(self, X, y, unidades, dropout, X_val=None, y_val=None, epocas=None):
        import tensorflow as tf
        self._semear()
        modelo = self._construir(unidades, dropout, X.shape[2])
        if X_val is not None:
            parada = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PACIENCIA, restore_best_weights=True)
            historico = modelo.fit(X, y, validation_data=(X_val, y_val),
                                   epochs=MAX_EPOCAS, batch_size=TAMANHO_LOTE,
                                   shuffle=False, verbose=0, callbacks=[parada])
            melhor_epoca = int(min(range(len(historico.history["val_loss"])),
                                   key=historico.history["val_loss"].__getitem__)) + 1
            return modelo, melhor_epoca
        modelo.fit(X, y, epochs=epocas, batch_size=TAMANHO_LOTE,
                   shuffle=False, verbose=0)
        return modelo, epocas

    # ------------------------------------------------------------------
    # IForecaster
    # ------------------------------------------------------------------
    def fit(self, treino: SerieTemporal) -> None:
        import numpy as np
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import TimeSeriesSplit

        linhas = treino.com_atributos()
        # RN05: limites da normalização estimados só no treino (alvos e defasagens).
        # São calculados antes do laço das partições (ver limitação no docstring).
        valores = [r.desembarques for r in linhas] + [
            r.atributos[f"lag_{k}"] for r in linhas for k in range(1, self.lookback + 1)]
        self._minimo, self._maximo = min(valores), max(valores)
        X, y = self._tensores(linhas)

        resultados = []
        divisor = TimeSeriesSplit(n_splits=self.n_particoes)
        for unidades in self.grade["unidades"]:
            for dropout in self.grade["dropout"]:
                erros, epocas = [], []
                for idx_tr, idx_val in divisor.split(X):
                    modelo, melhor = self._ajustar(
                        X[idx_tr], y[idx_tr], unidades, dropout,
                        X_val=X[idx_val], y_val=y[idx_val])
                    prev = modelo.predict(X[idx_val], verbose=0).ravel()
                    erros.append(mean_absolute_error(
                        [self._desnormalizar(v) for v in y[idx_val]],
                        [self._desnormalizar(v) for v in prev]))
                    epocas.append(melhor)
                resultados.append((float(np.mean(erros)), unidades, dropout,
                                   int(round(np.mean(epocas)))))

        mae, self.unidades, self.dropout, self.epocas = min(resultados)
        self._modelo, _ = self._ajustar(X, y, self.unidades, self.dropout,
                                        epocas=self.epocas)
        self._busca = {"mae_validacao": round(mae, 2),
                       "configuracoes_testadas": len(resultados)}

    def predict(self, horizonte: SerieTemporal) -> List[float]:
        import numpy as np
        X = np.array([self._sequencia(r) for r in horizonte.registros], dtype="float32")
        return [self._desnormalizar(float(v))
                for v in self._modelo.predict(X, verbose=0).ravel()]

    def salvar(self, caminho_modelo) -> None:
        caminho = Path(caminho_modelo)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        self._modelo.save(caminho)
        # A normalização e os hiperparâmetros acompanham o modelo, em arquivo
        # irmão: sem eles a rede recarregada não saberia desfazer a escala.
        caminho.with_suffix(".json").write_text(json.dumps({
            "minimo": self._minimo, "maximo": self._maximo,
            "lookback": self.lookback, "unidades": self.unidades,
            "dropout": self.dropout, "epocas": self.epocas, "busca": self._busca,
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    def carregar(self, caminho_modelo) -> None:
        import tensorflow as tf
        caminho = Path(caminho_modelo)
        self._modelo = tf.keras.models.load_model(caminho)
        meta = json.loads(caminho.with_suffix(".json").read_text(encoding="utf-8"))
        self._minimo, self._maximo = meta["minimo"], meta["maximo"]
        self.lookback, self.unidades = meta["lookback"], meta["unidades"]
        self.dropout, self.epocas = meta["dropout"], meta["epocas"]
        self._busca = meta.get("busca", {})
