# Bank Marketing Response

An imbalanced binary-classification project for prioritizing outreach to clients likely to subscribe to a term deposit. It uses a mixed-type preprocessing pipeline, excludes post-contact `duration` leakage, selects a decision threshold on validation data, and reports untouched test performance with MLflow experiment lineage.

## Business question
Can a bank allocate limited campaign capacity toward customers with higher response likelihood while controlling precision/recall trade-offs?

## Evaluation design
The UCI Bank Marketing dataset is split into train, validation, and test partitions using stratification. `duration` is excluded because it is only known after a call. A logistic-regression baseline is evaluated with ROC AUC, average precision, precision, recall, and F1. The threshold is tuned for F1 on validation only; test results are reported once. Scores are decision support, not grounds for exclusionary treatment.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
mlflow ui --backend-store-uri ./mlruns
```
On Windows activate with `.venv\\Scripts\\activate`. Outputs are ignored in Git under `data/`, `models/`, `reports/`, and `mlruns/`.

## Data
[UCI Bank Marketing](https://doi.org/10.24432/C5K306), CC BY 4.0. Limitations include historical campaign context, class imbalance, and possible temporal/customer-level dependence.
