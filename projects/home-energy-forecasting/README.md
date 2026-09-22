# Home Energy Forecasting

A time-aware regression case study estimating household appliance energy consumption from indoor/outdoor sensor readings and calendar context. It establishes a persistence-style mean baseline, evaluates HistGradientBoosting on a chronological holdout, logs experiments to MLflow, and persists the complete fitted pipeline.

## Business question
Can short-horizon energy estimates support operational monitoring and identify periods of unusually high consumption?

## Design
UCI measurements are ordered by timestamp; the final 20% is held out. Calendar variables are derived from the timestamp. Random variables `rv1` and `rv2` are excluded as non-informative. Metrics include MAE, RMSE, and R2. A random split would overstate generalization, so chronological evaluation is used; future deployments need rolling-origin backtests and drift checks.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
mlflow ui --backend-store-uri ./mlruns
```
Windows activation: `.venv\\Scripts\\activate`. Generated outputs are in ignored `models/`, `reports/`, and `mlruns/` directories. Configure a remote server with `MLFLOW_TRACKING_URI`.

## Data
[UCI Appliances Energy Prediction](https://doi.org/10.24432/C5VC8G), CC BY 4.0. This is observational data from one home over a limited period, not a general-purpose consumption benchmark.
