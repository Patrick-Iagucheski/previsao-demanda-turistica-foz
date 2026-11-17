# Previsão da demanda turística em Foz do Iguaçu: Random Forest e LSTM

Código do Trabalho de Conclusão de Curso *Aplicação de algoritmos de aprendizado de máquina
para prever a demanda turística no período de alta temporada*, de Patrick Iagucheski de
Oliveira, curso de Sistemas de Informação do Centro Universitário Dinâmica das Cataratas
(UDC), 2026.

O projeto compara o Random Forest e a rede LSTM na previsão mensal dos desembarques de
passageiros no Aeroporto de Foz do Iguaçu, com dados abertos da ANAC de 2019 a 2025, sem o
período da pandemia. As referências de comparação são o naïve sazonal (repete o mesmo mês do
ano anterior) e a persistência (repete o mês anterior). Um sistema web, organizado em camadas,
importa os dados, treina e compara os modelos e registra cada execução em banco de dados.

## Estrutura

```
dominio/          entidades e interfaces (núcleo, sem dependências externas)
aplicacao/        serviços: ingestão, pré-processamento, treinamento, previsão, avaliação
infraestrutura/   adaptadores: CSV da ANAC, modelos (naïve, Random Forest, LSTM), banco SQLite
apresentacao/     aplicação Flask: rotas, telas e estilo
notebooks/        caracterização da série, experimento e referência de persistência
dados/            experimentos.db: a série mensal e as 11 execuções do TCC
config.py         caminhos e parâmetros (semente padrão 42, pasta dos CSV da ANAC)
bases.json        bases que podem ser escolhidas na interface
```

## Instalação

Desenvolvido no Windows 11 com Python 3.12.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dados da ANAC

Os arquivos originais da ANAC (cerca de 2 GB) não estão no repositório. Eles são públicos e
estão no portal de dados abertos da agência, em
[Dados de Movimentação Aeroportuária](https://sistemas.anac.gov.br/dadosabertos/Operador%20Aeroportu%C3%A1rio/Dados%20de%20Movimenta%C3%A7%C3%A3o%20Aeroportu%C3%A1rias/).
A pasta onde ficam é definida em `config.py` (`ANAC_ORIGEM`, hoje `C:\Dados_Anac`).

Os CSV só são necessários para importar a série de novo: a base incluída
(`dados/experimentos.db`) já tem a série mensal e as 11 execuções do experimento.

## Como rodar

Sistema web (abre em http://localhost:5000):

```
python apresentacao\app.py
```

Telas: Importação, Treinar, Comparar e Histórico. As outras bases do seletor (demonstração e
simulada) são criadas vazias quando escolhidas.

Notebooks:

```
jupyter notebook notebooks
```

| Notebook | O que faz |
|---|---|
| `01_caracterizacao_serie` | Estatísticas da série, índice sazonal de janeiro, julho e dezembro, decomposição, autocorrelação e testes ADF e KPSS |
| `02_experimento_resultados` | Executa o experimento: naïve sazonal, Random Forest e LSTM em cinco sementes, métricas, Diebold-Mariano e importância dos atributos. **Ao ser executado, treina de novo e grava mais 11 execuções na base** (cerca de 40 minutos) |
| `03_baseline_persistencia` | Calcula a persistência e o teste de Diebold-Mariano contra as 11 execuções gravadas. Abre a base só para leitura |

Os notebooks gravam os resultados em `docs/resultados`, pasta criada na primeira execução.

## Reprodutibilidade

- Semente padrão 42 (`config.SEED`); o experimento usa as sementes 42 a 46.
- Cada execução fica registrada em `dados/experimentos.db` com modelo, hiperparâmetros,
  semente, previsões e métricas (tabelas `tb_exp_execucoes`, `tb_exp_previsoes` e
  `tb_exp_metricas`).
- Os modelos treinados não estão no repositório. Podem ser gerados de novo pela tela Treinar
  ou pelo notebook 02.

## De onde vêm os números do TCC

| Parte do TCC | Fonte |
|---|---|
| Caracterização da série (Seção 5.1) | Notebook 01 |
| Experimento, métricas, sementes, Diebold-Mariano e importância dos atributos (Seções 5.2 a 5.6) | Notebook 02 e as execuções 1 a 11 da base |
| Referência de persistência | Notebook 03 |
| Validação da extração dos dados (Seção 4.6.5) | `python infraestrutura\validacao_extracao.py` (exige os CSV da ANAC) |
