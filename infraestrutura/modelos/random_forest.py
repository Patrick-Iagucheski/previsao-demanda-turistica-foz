# -*- coding: utf-8 -*-
"""
Random Forest — adaptador de IForecaster (RF12), com ajuste de
hiperparâmetros por validação temporal (RF14).

Treina sobre a matriz de atributos gerada no pré-processamento (defasagens
1 a 3, mês em codificação cíclica e indicador de alta temporada).

Ajuste de hiperparâmetros (Seção 3.7 do TCC): busca em grade sobre o número
de árvores, a profundidade máxima e o número de atributos considerados em
cada divisão, avaliada por validação cruzada temporal (TimeSeriesSplit, em
janela expansível). Em cada partição o modelo é treinado num trecho inicial e
validado no trecho imediatamente posterior. Como a busca ocorre dentro de
fit(), ela só enxerga o treino: o conjunto de teste não participa da escolha
dos parâmetros (RN04). A melhor configuração é então reajustada sobre o
treino inteiro.

O Random Forest não exige normalização das entradas: as divisões das árvores
dependem apenas da ordem dos valores, não da escala.
"""
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dominio.entidades import SerieTemporal, TipoModeloEnum
from dominio.portas import IForecaster

# Espaço de busca (Seção 3.7 do TCC; docs/planejamento_sistema.md, Seção 4.4).
GRADE_PADRAO = {
    "n_estimators": [200, 500],
    "max_depth": [2, 3, 5, None],
    "max_features": [1.0, "sqrt"],
}

# Partições da validação temporal. Com 25 linhas de treino, quatro partições
# deixam 5 meses para validar em cada uma e 5 meses no menor treino.
N_PARTICOES = 4


class RandomForestModel(IForecaster):

    extensao = ".joblib"

    def __init__(self, semente: int = 42, grade: dict = None,
                 n_particoes: int = N_PARTICOES):
        self.semente = semente
        self.grade = grade or GRADE_PADRAO
        self.n_particoes = n_particoes
        self.n_estimators = None
        self.max_depth = None
        self._modelo = None
        self._colunas: List[str] = []
        self._busca = {}

    def nome(self) -> str:
        return TipoModeloEnum.RANDOM_FOREST.value

    @property
    def tamanho_janela_entrada(self):
        return sum(1 for c in self._colunas if c.startswith("lag_")) or None

    def hiperparametros(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "max_features": self._modelo.max_features if self._modelo else None,
            "atributos": self._colunas,
            "validacao": f"TimeSeriesSplit({self.n_particoes})",
            "criterio_selecao": "MAE",
            **self._busca,
        }

    # ------------------------------------------------------------------
    def _matriz(self, serie: SerieTemporal):
        linhas = serie.com_atributos()
        if not self._colunas:
            self._colunas = list(linhas[0].atributos.keys())
        X = [[r.atributos[c] for c in self._colunas] for r in linhas]
        y = [r.desembarques for r in linhas]
        return X, y

    def fit(self, treino: SerieTemporal) -> None:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

        self._colunas = []
        X, y = self._matriz(treino)
        busca = GridSearchCV(
            RandomForestRegressor(random_state=self.semente),
            param_grid=self.grade,
            cv=TimeSeriesSplit(n_splits=self.n_particoes),
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            refit=True,
        )
        busca.fit(X, y)
        self._modelo = busca.best_estimator_
        self.n_estimators = self._modelo.n_estimators
        self.max_depth = self._modelo.max_depth
        self._busca = {
            "mae_validacao": round(float(-busca.best_score_), 2),
            "configuracoes_testadas": len(busca.cv_results_["params"]),
        }

    def predict(self, horizonte: SerieTemporal) -> List[float]:
        X = [[r.atributos[c] for c in self._colunas] for r in horizonte.registros]
        return [float(v) for v in self._modelo.predict(X)]

    def importancia_atributos(self) -> dict:
        """Importância média de cada atributo na floresta (impureza)."""
        return dict(zip(self._colunas,
                        (float(v) for v in self._modelo.feature_importances_)))

    def salvar(self, caminho_modelo) -> None:
        import joblib
        caminho = Path(caminho_modelo)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"modelo": self._modelo, "colunas": self._colunas,
                     "busca": self._busca}, caminho)

    def carregar(self, caminho_modelo) -> None:
        import joblib
        conteudo = joblib.load(caminho_modelo)
        self._modelo = conteudo["modelo"]
        self._colunas = conteudo["colunas"]
        self._busca = conteudo.get("busca", {})
        self.n_estimators = self._modelo.n_estimators
        self.max_depth = self._modelo.max_depth
