# Titanic — EDA, Pipeline de Classificação e Bônus de Regressão

Pipeline de Machine Learning reproduzível, ponta a ponta, para prever a sobrevivência de passageiros do Titanic: análise exploratória orientada a hipóteses, feature engineering, comparação de modelos com validação cruzada, tuning e avaliação honesta em holdout — sem vazamento de dados. Um bônus final aplica as mesmas técnicas a um problema de regressão com target assimétrico (`Fare`), no estilo *House Prices — Ames*.

**Dataset:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

## O que o notebook cobre

1. **EDA orientada a hipóteses** — taxas de sobrevivência por sexo, classe, idade e tarifa.
2. **Feature engineering** — `Title` (extraído de `Name`), `FamilySize`/`IsAlone`, `Deck` (a partir de `Cabin`).
3. **Pipeline sem vazamento** — `ColumnTransformer` + `Pipeline` do scikit-learn, com imputação e encoding aprendidos só no treino.
4. **Comparação de modelos** — Regressão Logística, Random Forest, Gradient Boosting e Hist. Gradient Boosting, via validação cruzada estratificada de 5 folds.
5. **Tuning** (`GridSearchCV`) e **avaliação no holdout** — classification report, matriz de confusão, ROC AUC.
6. **Interpretabilidade** — importância por permutação no holdout.
7. **Submissão Kaggle** — `submission.csv` no formato oficial.
8. **Bônus de regressão** — `Fare` como target assimétrico, transformação `log1p` + `TransformedTargetRegressor`, Ridge vs. Hist. Gradient Boosting.

**Resultado:** melhor modelo (Regressão Logística) com acurácia de validação cruzada de **0,8245**; taxa de sobrevivência prevista no teste de 40,2%, acima do baseline público de referência (~0,7655).

## Como rodar

```bash
git clone https://github.com/kagenash/titanic-classificacao-ml.git
cd titanic-classificacao-ml
pip install -r requirements.txt
```

Baixe `train.csv` e `test.csv` em [kaggle.com/c/titanic](https://www.kaggle.com/c/titanic) e coloque-os na raiz do projeto antes de abrir o notebook.

Requer Python 3.10+.

## Estrutura

```
titanic-classificacao-ml/
├── titanic_pipeline_classificacao.ipynb
├── requirements.txt
└── README.md
```

## Licença

MIT — use, adapte e compartilhe à vontade.
