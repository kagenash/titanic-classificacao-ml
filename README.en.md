🇧🇷 [Português](README.md) | 🇺🇸 **English**

# Titanic — EDA, Classification Pipeline & Regression Bonus

Reproducible, end-to-end Machine Learning pipeline to predict Titanic passenger survival: hypothesis-driven exploratory analysis, feature engineering, model comparison with cross-validation, tuning and honest holdout evaluation — with no data leakage. A final bonus applies the same techniques to a regression problem with a skewed target (`Fare`), in the style of *House Prices — Ames*.

**Dataset:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

## What the notebook covers

1. **Hypothesis-driven EDA** — survival rates by sex, class, age and fare.
2. **Feature engineering** — `Title` (extracted from `Name`), `FamilySize`/`IsAlone`, `Deck` (derived from `Cabin`).
3. **Leakage-free pipeline** — `ColumnTransformer` + scikit-learn `Pipeline`, with imputation and encoding learned only on the training fold.
4. **Model comparison** — Logistic Regression, Random Forest, Gradient Boosting and Hist. Gradient Boosting, via 5-fold stratified cross-validation.
5. **Tuning** (`GridSearchCV`) and **holdout evaluation** — classification report, confusion matrix, ROC AUC.
6. **Interpretability** — permutation importance on the holdout set.
7. **Kaggle submission** — `submission.csv` in the official format.
8. **Regression bonus** — `Fare` as a skewed target, `log1p` transform + `TransformedTargetRegressor`, Ridge vs. Hist. Gradient Boosting.

**Result:** best model (Logistic Regression) with cross-validated accuracy of **0.8245**; predicted test-set survival rate of 40.2%, above the public leaderboard reference baseline (~0.7655).

## How to run

```bash
git clone https://github.com/kagenash/titanic-classificacao-ml.git
cd titanic-classificacao-ml
pip install -r requirements.txt
```

Download `train.csv` and `test.csv` from [kaggle.com/c/titanic](https://www.kaggle.com/c/titanic) and place them in the project root before opening the notebook.

Requires Python 3.10+.

## Structure

```
titanic-classificacao-ml/
├── titanic_pipeline_classificacao.ipynb
├── requirements.txt
└── README.md
```

## License

MIT — use, adapt and share freely.
