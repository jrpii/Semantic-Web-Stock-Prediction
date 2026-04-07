# Model Performance Comparison

Evaluation of Logistic Regression, Linear SVM, and XGBoost across intervals.

## Overall Results

| dataset        | model              |   roc_auc |       f1 |   precision |   recall |   accuracy |
|:---------------|:-------------------|----------:|---------:|------------:|---------:|-----------:|
| ALL_COMBINED   | XGBoost            |  0.517083 | 0.521934 |    0.502438 | 0.543003 |   0.51135  |
| ALL_COMBINED   | LogisticRegression |  0.509625 | 0.50718  |    0.500632 | 0.513902 |   0.509399 |
| ALL_COMBINED   | LinearSVM          |  0.509576 | 0.507466 |    0.501117 | 0.513978 |   0.509887 |
| BREAKOUTS_ONLY | LogisticRegression |  0.527288 | 0.552017 |    0.513158 | 0.597243 |   0.528316 |
| BREAKOUTS_ONLY | LinearSVM          |  0.527055 | 0.552017 |    0.513158 | 0.597243 |   0.528316 |
| BREAKOUTS_ONLY | XGBoost            |  0.51966  | 0.5235   |    0.49589  | 0.554364 |   0.508942 |

## Transparency and Explainability
Models and their corresponding feature weights (`*importances.csv`) to gauge transparency are saved in `analysis_outputs/models/`.
