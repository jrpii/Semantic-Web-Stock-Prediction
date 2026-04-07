# Model Performance Comparison

Evaluation of Logistic Regression, Linear SVM, and XGBoost across intervals.

## Overall Results

| dataset      | model              |   roc_auc |       f1 |   precision |   recall |   accuracy |
|:-------------|:-------------------|----------:|---------:|------------:|---------:|-----------:|
| ALL_COMBINED | XGBoost            |  0.512365 | 0.48907  |    0.501082 | 0.47762  |   0.509774 |
| ALL_COMBINED | LogisticRegression |  0.507012 | 0.510885 |    0.497324 | 0.525206 |   0.505985 |
| ALL_COMBINED | LinearSVM          |  0.505219 | 0.498844 |    0.495146 | 0.502597 |   0.503921 |

## Transparency and Explainability
Models and their corresponding feature weights (`*importances.csv`) to gauge transparency are saved in `analysis_outputs/models/`.
