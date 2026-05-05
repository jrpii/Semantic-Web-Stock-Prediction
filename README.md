# Semantic Web Stock Prediction

CSE 573, Group 15, Project 6: Directional Prediction of Stocks.

This project combines OHLCV stock data, financial news metadata, technical indicators, and semantic news signals to predict next-bar stock direction for AAPL and AMZN across multiple time intervals.

## Project Structure

- `CODE/scripts/` - cleaning, feature engineering, model training, evaluation, and plotting scripts.
- `CODE/pipeline/` - data collection scripts for OHLCV and news data.
- `CODE/dashboard/` - static data visualization webpage for exploring project results.
- `DATA/datasets/` - raw stock chart CSVs and raw news JSON files.
- `DATA/papers/` - project reference papers.
- `EVALUATIONS/analysis_outputs/cleaned/` - cleaned stock and news datasets.
- `EVALUATIONS/analysis_outputs/reports/` - generated EDA summaries, split metadata, and model result tables.
- `EVALUATIONS/analysis_outputs/models/` - trained model artifacts and feature importance CSVs.
- `EVALUATIONS/analysis_outputs/charts/` - generated price and prediction plots.
- `EVALUATIONS/analysis_outputs/visualizations/` - generated ROC AUC visualizations.

## Data

Stock ticker CSV data is stored in OHLCV format:

```text
Date, Time, Open, High, Low, Close, Volume
```

Available intervals:

- 5 minutes
- 15 minutes
- 30 minutes
- 1 hour
- 4 hours
- 1 day

Stock timestamps are standardized to UTC-style outputs during cleaning. News articles are cleaned, deduplicated, filtered for ticker relevance, and aligned to the next valid market bar.

## Environment Setup

Create and activate a Python environment:

```powershell
conda create -n stock-prediction python=3.10
conda activate stock-prediction
```

Install PyTorch and project dependencies:

```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

If CUDA is not available, install the CPU-compatible PyTorch build from the official PyTorch instructions, then run `pip install -r requirements.txt`.

## Common Workflows

Clean data and generate EDA outputs:

```powershell
python CODE/scripts/clean_and_eda.py
```

Train and evaluate classic ML models:

```powershell
python CODE/scripts/train_evaluate_models.py
```

Run LSTM experiments:

```powershell
python CODE/scripts/run_lstm_all_intervals.py
python CODE/scripts/run_lstm_news_all_intervals.py
python CODE/scripts/run_lstm_finbert_all_intervals.py
```

Multi-interval fusion LSTM (one model, six aligned resolutions; multi-task next-bar labels):

```powershell
python CODE/scripts/train_lstm_multi_interval.py
python CODE/scripts/run_lstm_multi_interval.py
```

`train_lstm_multi_interval.py` owns training and console metrics; `run_lstm_multi_interval.py` calls it once and **appends** rows to `EVALUATIONS/analysis_outputs/reports/lstm_multi_interval_results.csv`. Use `--save-model` on the train script to write checkpoints under `EVALUATIONS/analysis_outputs/models/lstm_multi/`.

Generate plots:

```powershell
python CODE/scripts/plot_lstm_results.py
python CODE/scripts/plot_lstm_finbert_results.py
python CODE/scripts/plot_price_with_model_predictions.py
```

## Dashboard

The dashboard is a static webpage that visualizes the generated outputs in `EVALUATIONS/analysis_outputs`, including:

- model performance comparisons
- LSTM, LSTM + News, and LSTM + FinBERT results
- cleaned stock coverage and volatility
- chronological train/test splits
- news volume and FinBERT sentiment
- generated chart and ROC AUC image outputs
- EDA and model comparison reports

Run a local web server from the repository root:

```powershell
python -m http.server 8000 --bind 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8000/CODE/dashboard/
```

The server is required because the dashboard loads CSV, Markdown, and PNG files from `EVALUATIONS/analysis_outputs`; opening `CODE/dashboard/index.html` directly may block those file reads in the browser.

## Key Outputs

- `EVALUATIONS/analysis_outputs/reports/eda_summary.md`
- `EVALUATIONS/analysis_outputs/reports/model_comparison.md`
- `EVALUATIONS/analysis_outputs/reports/model_results.csv`
- `EVALUATIONS/analysis_outputs/reports/lstm_interval_results.csv`
- `EVALUATIONS/analysis_outputs/reports/lstm_news_interval_results.csv`
- `EVALUATIONS/analysis_outputs/reports/lstm_finbert_interval_results_ExNeEn.csv`
- `EVALUATIONS/analysis_outputs/reports/lstm_multi_interval_results.csv`
- `EVALUATIONS/analysis_outputs/reports/stock_file_summary.csv`
- `EVALUATIONS/analysis_outputs/reports/chronological_splits.csv`
- `EVALUATIONS/analysis_outputs/charts/AAPL_1440m_price_with_model_predictions.png`
- `EVALUATIONS/analysis_outputs/visualizations/*.png`
