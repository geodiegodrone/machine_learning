# Bike Sharing Demand Forecasting

A leakage-aware hourly rental-demand forecasting pipeline using calendar, weather, and operational context. A chronological holdout compares a seasonal-naive baseline with HistGradientBoosting, with MLflow tracking for reproducible model comparison.

## Business question
How many bikes should an operator make available by hour to reduce unmet demand and excess fleet allocation?

## Approach
UCI hourly records are ordered by timestamp and split chronologically. `casual` and `registered` are excluded because they sum to the target `cnt`. Evaluation reports MAE, RMSE, and R2 on a final unseen time period. The dataset spans a limited historical period, so it is not a production forecast without temporal backtesting and local recalibration.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
mlflow ui --backend-store-uri ./mlruns
```
Windows activation: `.venv\\Scripts\\activate`. Outputs: `models/`, `reports/`, `mlruns/`. Set `MLFLOW_TRACKING_URI` for a remote server.

## Data
[UCI Bike Sharing](https://doi.org/10.24432/C5W894), CC BY 4.0. Target: hourly total rentals (`cnt`).
