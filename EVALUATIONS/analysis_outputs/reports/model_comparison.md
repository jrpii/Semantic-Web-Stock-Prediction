# Model Performance Comparison

Evaluation of Logistic Regression, Linear SVM, and XGBoost across intervals.

## Overall Results

| dataset         | model              |   roc_auc |       f1 |   precision |   recall |   accuracy |
|:----------------|:-------------------|----------:|---------:|------------:|---------:|-----------:|
| 1440m_ALL       | XGBoost            |  0.489023 | 0.488987 |    0.509174 | 0.470339 |   0.479821 |
| 1440m_ALL       | LogisticRegression |  0.468281 | 0.567164 |    0.506667 | 0.644068 |   0.479821 |
| 1440m_ALL       | LinearSVM          |  0.467333 | 0.556818 |    0.503425 | 0.622881 |   0.475336 |
| 1440m_BREAKOUTS | LinearSVM          |  0.503968 | 0.357143 |    0.714286 | 0.238095 |   0.454545 |
| 1440m_BREAKOUTS | XGBoost            |  0.472222 | 0.387097 |    0.6      | 0.285714 |   0.424242 |
| 1440m_BREAKOUTS | LogisticRegression |  0.468254 | 0.4      |    0.666667 | 0.285714 |   0.454545 |
| 15m_ALL         | LinearSVM          |  0.518205 | 0.4888   |    0.51046  | 0.468903 |   0.511316 |
| 15m_ALL         | LogisticRegression |  0.517902 | 0.49016  |    0.512186 | 0.469951 |   0.512883 |
| 15m_ALL         | XGBoost            |  0.510941 | 0.495663 |    0.502332 | 0.489168 |   0.504004 |
| 15m_BREAKOUTS   | LinearSVM          |  0.534599 | 0.441441 |    0.550562 | 0.368421 |   0.505976 |
| 15m_BREAKOUTS   | LogisticRegression |  0.533325 | 0.443439 |    0.556818 | 0.368421 |   0.50996  |
| 15m_BREAKOUTS   | XGBoost            |  0.529056 | 0.463519 |    0.54     | 0.406015 |   0.501992 |
| 240m_ALL        | LinearSVM          |  0.562795 | 0.627035 |    0.510957 | 0.811355 |   0.521779 |
| 240m_ALL        | LogisticRegression |  0.561563 | 0.625889 |    0.511628 | 0.805861 |   0.522686 |
| 240m_ALL        | XGBoost            |  0.517095 | 0.535959 |    0.503215 | 0.57326  |   0.508167 |
| 240m_BREAKOUTS  | LinearSVM          |  0.540895 | 0.25     |    0.5      | 0.166667 |   0.5      |
| 240m_BREAKOUTS  | XGBoost            |  0.539352 | 0.491803 |    0.6      | 0.416667 |   0.569444 |
| 240m_BREAKOUTS  | LogisticRegression |  0.531636 | 0.25     |    0.5      | 0.166667 |   0.5      |
| 30m_ALL         | LogisticRegression |  0.505178 | 0.512291 |    0.482632 | 0.545833 |   0.496104 |
| 30m_ALL         | LinearSVM          |  0.503812 | 0.508371 |    0.478466 | 0.542262 |   0.491486 |
| 30m_ALL         | XGBoost            |  0.496876 | 0.533262 |    0.485352 | 0.591667 |   0.497835 |
| 30m_BREAKOUTS   | XGBoost            |  0.583793 | 0.486486 |    0.537313 | 0.444444 |   0.565714 |
| 30m_BREAKOUTS   | LogisticRegression |  0.555687 | 0.324324 |    0.6      | 0.222222 |   0.571429 |
| 30m_BREAKOUTS   | LinearSVM          |  0.553586 | 0.318584 |    0.5625   | 0.222222 |   0.56     |
| 5m_ALL          | LogisticRegression |  0.520555 | 0.494305 |    0.508413 | 0.480958 |   0.520026 |
| 5m_ALL          | LinearSVM          |  0.520432 | 0.493842 |    0.508266 | 0.480214 |   0.519881 |
| 5m_ALL          | XGBoost            |  0.519336 | 0.517172 |    0.498208 | 0.537638 |   0.510376 |
| 5m_BREAKOUTS    | LogisticRegression |  0.529481 | 0.536585 |    0.492105 | 0.589905 |   0.529155 |
| 5m_BREAKOUTS    | LinearSVM          |  0.528772 | 0.536585 |    0.492105 | 0.589905 |   0.529155 |
| 5m_BREAKOUTS    | XGBoost            |  0.503347 | 0.532787 |    0.46988  | 0.615142 |   0.501458 |
| 60m_ALL         | LogisticRegression |  0.529868 | 0.547406 |    0.514286 | 0.585086 |   0.520833 |
| 60m_ALL         | LinearSVM          |  0.528133 | 0.541947 |    0.510566 | 0.577438 |   0.516572 |
| 60m_ALL         | XGBoost            |  0.523381 | 0.548791 |    0.515993 | 0.586042 |   0.522727 |
| 60m_BREAKOUTS   | XGBoost            |  0.507949 | 0.632258 |    0.544444 | 0.753846 |   0.544    |
| 60m_BREAKOUTS   | LinearSVM          |  0.429744 | 0.651429 |    0.518182 | 0.876923 |   0.512    |
| 60m_BREAKOUTS   | LogisticRegression |  0.425897 | 0.651429 |    0.518182 | 0.876923 |   0.512    |

## Transparency and Explainability
Models and their corresponding feature weights (`*importances.csv`) to gauge transparency are saved in `analysis_outputs/models/`.
