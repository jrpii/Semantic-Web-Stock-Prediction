import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

def main():
    os.makedirs('analysis_outputs/models', exist_ok=True)
    os.makedirs('analysis_outputs/reports', exist_ok=True)

    # Load News
    news_df = pd.read_csv('analysis_outputs/cleaned/news/news_aligned_by_bar.csv')
    news_df.rename(columns={'aligned_bar_utc': 'timestamp_utc'}, inplace=True)

    # Features and Target
    features = [
        'volume', 'return_pct', 'volatility_10', 'rsi_14',
        'dist_to_sma_5', 'dist_to_sma_10', 'dist_to_sma_20',
        'dist_to_ema_10', 'high_low_spread_pct',
        'article_count', 'weighted_article_count', 'avg_word_count', 'unique_sites',
        'news_volatility_interaction', 'news_momentum'
    ]

    results = []
    stocks_dir = 'analysis_outputs/cleaned/stocks'
    
    print("Loading, merging, and scaling datasets per ticker/interval...")
    all_dfs = []
    for file in sorted(os.listdir(stocks_dir)):
        if not file.endswith('.csv'): 
            continue
        df_part = pd.read_csv(os.path.join(stocks_dir, file))
        
        # Merge news features
        df_part = df_part.merge(news_df, on=['ticker', 'interval_minutes', 'timestamp_utc'], how='left')
        
        # Engineer relative features to avoid absolute pricing
        df_part['dist_to_sma_5'] = (df_part['close'] - df_part['sma_5']) / df_part['sma_5']
        df_part['dist_to_sma_10'] = (df_part['close'] - df_part['sma_10']) / df_part['sma_10']
        df_part['dist_to_sma_20'] = (df_part['close'] - df_part['sma_20']) / df_part['sma_20']
        df_part['dist_to_ema_10'] = (df_part['close'] - df_part['ema_10']) / df_part['ema_10']
        df_part['high_low_spread_pct'] = (df_part['high'] - df_part['low']) / df_part['close']
        
        # Fill NaN for news columns for bars with no news
        news_cols = ['article_count', 'weighted_article_count', 'avg_word_count', 'unique_sites']
        df_part[news_cols] = df_part[news_cols].fillna(0)
        
        # Create News-Volatility Interaction Features
        df_part['news_volatility_interaction'] = df_part['weighted_article_count'] * df_part['volatility_10']
        df_part['news_momentum'] = df_part['article_count'] * df_part['return_pct']
        
        # Drop rows with NaN resulting from rolling technical indicators
        df_part = df_part.dropna(subset=features + ['target_up_next_bar', 'split'])
        
        train_idx = df_part['split'] == 'train'
        test_idx = df_part['split'] == 'test'
        
        if train_idx.sum() == 0 or test_idx.sum() == 0:
            continue
            
        scaler = StandardScaler()
        # Fit on train, transform train and test to prevent data leakage
        df_part.loc[train_idx, features] = scaler.fit_transform(df_part.loc[train_idx, features])
        df_part.loc[test_idx, features] = scaler.transform(df_part.loc[test_idx, features])
        
        all_dfs.append(df_part)
        
    if not all_dfs:
        print("No stock data found.")
        return
        
    df = pd.concat(all_dfs, ignore_index=True)
    
    # One-Hot Encode Ticker and Interval (context for XGBoost)
    df = pd.get_dummies(df, columns=['ticker', 'interval_minutes'], dtype=float)
    
    # Define final features including the newly created dummy columns
    cat_features = [col for col in df.columns if col.startswith('ticker_') or col.startswith('interval_minutes_')]
    final_features = features + cat_features
    
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("Skipping due to missing train/test rows.")
        return
        
    X_train_scaled = train_df[final_features]
    y_train = train_df['target_up_next_bar']
    X_test_scaled = test_df[final_features]
    y_test = test_df['target_up_next_bar']
    
    models = {
        'LogisticRegression': LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000, C=1.0, random_state=42),
        'LinearSVM': LinearSVC(penalty='l1', loss='squared_hinge', dual=False, max_iter=3000, C=1.0, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, 
                                     eval_metric='logloss', random_state=42)
    }
    
    for model_name, model in models.items():
        try:
            print(f"Training {model_name} on combined dataset...")
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            
            # Proba for ROC-AUC
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_test_scaled)[:, 1]
            else:
                probs = model.decision_function(X_test_scaled)
            
            auc = roc_auc_score(y_test, probs)
            f1 = f1_score(y_test, preds, zero_division=0)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            acc = accuracy_score(y_test, preds)
            
            results.append({
                'dataset': 'ALL_COMBINED',
                'model': model_name,
                'roc_auc': auc,
                'f1': f1,
                'precision': prec,
                'recall': rec,
                'accuracy': acc
            })
            
            # Extract and save Feature Weights / Importances
            if model_name in ['LogisticRegression', 'LinearSVM']:
                importances = model.coef_[0]
            elif model_name == 'XGBoost':
                importances = model.feature_importances_
            
            imp_df = pd.DataFrame({'feature': final_features, 'importance': importances})
            imp_df.to_csv(f'analysis_outputs/models/ALL_COMBINED_{model_name}_importances.csv', index=False)
            
            # Save Model
            joblib.dump(model, f'analysis_outputs/models/ALL_COMBINED_{model_name}.joblib')
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")

    # Generate Report
    res_df = pd.DataFrame(results)
    
    # Sort for best output formatting
    res_df = res_df.sort_values(by=['dataset', 'roc_auc'], ascending=[True, False])
    res_df.to_csv('analysis_outputs/reports/model_results.csv', index=False)
    
    with open('analysis_outputs/reports/model_comparison.md', 'w') as f:
        f.write("# Model Performance Comparison\n\n")
        f.write("Evaluation of Logistic Regression, Linear SVM, and XGBoost across intervals.\n\n")
        f.write("## Overall Results\n\n")
        f.write(res_df.to_markdown(index=False) + "\n\n")
        f.write("## Transparency and Explainability\n")
        f.write("Models and their corresponding feature weights (`*importances.csv`) to gauge transparency are saved in `analysis_outputs/models/`.\n")

    print(f"\nCompleted modeling for {len(res_df)} model configurations. Report saved to analysis_outputs/reports/model_comparison.md")

if __name__ == "__main__":
    main()