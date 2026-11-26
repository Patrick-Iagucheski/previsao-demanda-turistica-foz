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

# LSTM: adaptador de IForecaster (RF13), com ajuste de hiperparâmetros por
# validação temporal (RF14).
#
# Arquitetura enxuta, como prevê o planejamento (docs/planejamento_sistema.md, Seção 4.4): uma camada LSTM
# de 8 a 32 unidades, dropout e parada antecipada (EarlyStopping). Com a
# amostra reduzida, redes maiores apenas memorizariam o treino.
#
# Entrada sequencial: cada amostra é a sequência dos `lookback` meses
# anteriores ao mês previsto, e cada passo traz o desembarque normalizado e o
# calendário daquele mês (seno e cosseno do mês e indicador de alta
# temporada). As sequências são montadas a partir das mesmas defasagens que o
# Random Forest recebe, de modo que os dois modelos usam o mesmo conjunto de
# informação e as mesmas linhas de treino e de teste (RN07). Pelo mesmo motivo
# das defasagens (ver aplicacao/preprocessamento.py), o lookback é 3, e não
# 12: uma janela de 12 meses não caberia no bloco pré-pandêmico sem
# atravessar o intervalo excluído (RNF03).
#
# Normalização min-max do desembarque, estimada somente no treino (RN05), porque a
# convergência da LSTM é sensível à escala das entradas (Seção 3.7 do TCC).
# Limitação conhecida e declarada no TCC (Seção 3.7 e Limitações do Cap. 5): os
# limites são estimados sobre o treino completo ANTES da divisão nas partições da
# validação cruzada; na 1ª partição eles refletem valores do trecho de validação
# (nas outras três coincidem). Não envolve o teste; afeta só a seleção de
# hiperparâmetros. Correção possível: estimar os limites dentro de cada partição.
#
# Ajuste de hiperparâmetros: busca em grade sobre unidades e dropout, com
# validação cruzada temporal (TimeSeriesSplit). Em cada partição, a parada
# antecipada usa a validação para escolher o número de épocas; o modelo final
# é reajustado sobre o treino inteiro com a configuração vencedora e com o
# número médio de épocas encontrado nas partições, sem tocar no teste (RN04).
#
# Reprodutibilidade (RNF01): a semente fixa o Python, o NumPy e o TensorFlow,
# e as operações do TensorFlow são forçadas a ser determinísticas. Como redes
# neurais ainda variam com a inicialização, o experimento roda a LSTM com
# várias sementes e reporta a dispersão.

# Espaço de busca dos hiperparâmetros: unidades da camada LSTM e taxa de dropout
# (3 x 2 = 6 combinações), testadas em fit().
GRADE_PADRAO = {"unidades": [8, 16, 32], "dropout": [0.0, 0.2]}
# Número de partições da validação cruzada temporal (TimeSeriesSplit) usada em fit().
N_PARTICOES = 4

# Usada no método _sequencia(), que monta a entrada com os 3 meses anteriores ao
# mês previsto (as mesmas defasagens do Random Forest).
LOOKBACK = 3 # Número de meses anteriores a considerar
# Limite de épocas (passadas completas pelo treino) na busca; a parada antecipada
# costuma interromper antes.
MAX_EPOCAS = 400
# Quantas épocas sem melhora na validação a parada antecipada (EarlyStopping)
# espera antes de parar.
PACIENCIA = 40
# Taxa de aprendizado do otimizador Adam, usada em _construir().
TAXA_APRENDIZADO = 0.005
# Quantas sequências a rede vê antes de cada atualização dos pesos (lote), usado
# em _ajustar().
TAMANHO_LOTE = 4

# Os mesmos atributos de calendário do RF08, lidos das regras do domínio.
# Retorna [seno do mês, cosseno do mês, 1 se for alta temporada ou 0 se não for].
def _calendario(mes: int) -> List[float]:
    return [*mes_ciclico(mes), float(eh_alta_temporada(mes))]

# Modelo LSTM (rede neural recorrente) que segue a interface IForecaster (RF13, RF14).
class LSTMModel(IForecaster):

    extensao = ".keras"

    # Guarda a semente, a grade de busca, as partições e o lookback; os hiperparâmetros
    # escolhidos e os limites da normalização só são preenchidos em fit() ou carregar().
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

    # Retorna o identificador do modelo ("lstm"), o mesmo do TipoModeloEnum.
    def nome(self) -> str:
        return TipoModeloEnum.LSTM.value

    # Retorna quantos meses anteriores entram na sequência (3), gravado na execução.
    @property
    def tamanho_janela_entrada(self):
        return self.lookback

    # Retorna os hiperparâmetros escolhidos e o resultado da busca, gravados com a
    # execução (RF19).
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

    # Montagem das sequências

    # Leva o valor para a escala de 0 a 1 (min-max), com os limites do treino.
    def _normalizar(self, valor: float) -> float:
        return (valor - self._minimo) / (self._maximo - self._minimo)

    # Faz o caminho inverso: volta o valor da escala de 0 a 1 para passageiros.
    def _desnormalizar(self, valor: float) -> float:
        return valor * (self._maximo - self._minimo) + self._minimo

    # Os `lookback` meses anteriores a r, do mais antigo ao mais recente,
    # lidos das defasagens já calculadas dentro do bloco (RNF03, RN06).
    # Como funciona: para k = 3, 2 e 1, pega o lag_k (normalizado) e o calendário
    # daquele mês; a entrada fica com 3 passos de 4 valores cada.
    def _sequencia(self, r: RegistroMensal) -> List[List[float]]:
        passos = []
        for k in range(self.lookback, 0, -1):
            mes = (r.mes - k - 1) % 12 + 1
            passos.append([self._normalizar(r.atributos[f"lag_{k}"]), *_calendario(mes)])
        return passos

    # Monta as matrizes da rede: X com a sequência de cada mês e y com o desembarque
    # do próprio mês, os dois normalizados.
    def _tensores(self, registros: List[RegistroMensal]):
        import numpy as np
        X = np.array([self._sequencia(r) for r in registros], dtype="float32")
        y = np.array([self._normalizar(r.desembarques) for r in registros], dtype="float32")
        return X, y

    # Rede

    # Fixa a semente do Python, do NumPy e do TensorFlow e liga o modo
    # determinístico, para o treino sair igual a cada execução (RNF01). Também cala os
    # avisos de desempenho do TensorFlow ("retracing"), que aparecem quando a busca cria
    # várias redes seguidas; eles não mudam os resultados.
    def _semear(self):
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        tf.keras.utils.set_random_seed(self.semente)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

    # Cria a rede: entrada de 3 meses x 4 valores, uma camada LSTM com as unidades e
    # o dropout testados e uma saída com o valor previsto; o erro minimizado é o
    # quadrático médio (mse).
    def _construir(self, unidades: int, dropout: float, n_atributos: int):
        import tensorflow as tf
        modelo = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.lookback, n_atributos)),
            tf.keras.layers.LSTM(unidades, dropout=dropout),
            tf.keras.layers.Dense(1),
        ])
        modelo.compile(optimizer=tf.keras.optimizers.Adam(TAXA_APRENDIZADO), loss="mse")
        return modelo

    # Treina uma rede. Com validação (na busca), usa a parada antecipada e retorna
    # também a melhor época; sem validação (treino final), treina pelo número de
    # épocas informado.
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

    # IForecaster

    # Treina a LSTM (RF13) escolhendo os hiperparâmetros por validação temporal (RF14).
    # Como funciona: calcula os limites da normalização no treino; para cada
    # combinação da grade, treina e valida nas 4 partições e guarda o MAE médio e a
    # média das melhores épocas; escolhe a combinação de menor MAE e treina a rede
    # final no treino inteiro com esse número de épocas.
    def fit(self, treino: SerieTemporal) -> None:
        import numpy as np
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import TimeSeriesSplit

        linhas = treino.com_atributos()
        # RN05: limites da normalização estimados só no treino (alvos e defasagens).
        # São calculados antes do laço das partições (ver a limitação no comentário do início do arquivo).
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

    # Retorna a previsão de cada mês do horizonte: monta a sequência dos 3 meses
    # anteriores, a rede prevê na escala de 0 a 1 e o valor volta para passageiros.
    def predict(self, horizonte: SerieTemporal) -> List[float]:
        import numpy as np
        X = np.array([self._sequencia(r) for r in horizonte.registros], dtype="float32")
        return [self._desnormalizar(float(v))
                for v in self._modelo.predict(X, verbose=0).ravel()]

    # Salva a rede no arquivo .keras e, num .json com o mesmo nome, os limites da
    # normalização e os hiperparâmetros (RF19).
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

    # Recarrega a rede do .keras e lê do .json os limites da normalização e os
    # hiperparâmetros, para prever sem treinar de novo.
    def carregar(self, caminho_modelo) -> None:
        import tensorflow as tf
        caminho = Path(caminho_modelo)
        self._modelo = tf.keras.models.load_model(caminho)
        meta = json.loads(caminho.with_suffix(".json").read_text(encoding="utf-8"))
        self._minimo, self._maximo = meta["minimo"], meta["maximo"]
        self.lookback, self.unidades = meta["lookback"], meta["unidades"]
        self.dropout, self.epocas = meta["dropout"], meta["epocas"]
        self._busca = meta.get("busca", {})
