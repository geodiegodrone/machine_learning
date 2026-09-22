# SMS Spam Classifier

A compact NLP workflow classifying short messages as spam or legitimate. It uses a TF-IDF and balanced logistic-regression pipeline, compares against a majority-class baseline, evaluates a stratified holdout, logs model and metrics to MLflow, and saves a reusable inference artifact.

## Business question
Can incoming short messages be prioritized for review while keeping false positives visible and measurable?

## Method and safeguards
The UCI SMS Spam Collection is split into train/test with stratification. Text is vectorized inside the pipeline to prevent vocabulary leakage. Results include precision, recall, F1, ROC AUC, and average precision. No message samples are logged to MLflow. This benchmark does not represent current multilingual or channel-specific spam and should not automatically block communications.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
mlflow ui --backend-store-uri ./mlruns
```
Windows activation: `.venv\\Scripts\\activate`. Data, trained models, reports, and MLflow runs are generated locally and ignored by Git. Set `MLFLOW_TRACKING_URI` to use a shared tracking server.

## Data
[UCI SMS Spam Collection](https://doi.org/10.24432/C5CC84), CC BY 4.0.
