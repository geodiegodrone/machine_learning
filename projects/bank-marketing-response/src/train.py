from pathlib import Path
from io import BytesIO
import os
import urllib.request
import zipfile

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"


def main():
    raw = ROOT / "data/raw"; raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "bank.zip"
    if not archive.exists(): urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as zf:
        nested_name = next(n for n in zf.namelist() if n.endswith("bank.zip"))
        with zipfile.ZipFile(BytesIO(zf.read(nested_name))) as bank_zip:
            csv_name = next(n for n in bank_zip.namelist() if n.endswith("bank-full.csv"))
            with bank_zip.open(csv_name) as f: data = pd.read_csv(f, sep=";")
    y = data.pop("y").map({"no": 0, "yes": 1})
    data = data.drop(columns=["duration"])
    X_train, X_rem, y_train, y_rem = train_test_split(data, y, test_size=.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=.5, stratify=y_rem, random_state=42)
    cats = X_train.select_dtypes(include="object").columns.tolist()
    nums = X_train.columns.difference(cats).tolist()
    prep = ColumnTransformer([("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cats),
                              ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                                 ("scale", StandardScaler())]), nums)])
    model = Pipeline([("prep", prep), ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))])
    model.fit(X_train, y_train)
    val_prob = model.predict_proba(X_val)[:, 1]
    p, r, thresholds = precision_recall_curve(y_val, val_prob)
    f1s = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    threshold = float(thresholds[int(np.argmax(f1s))])
    test_prob = model.predict_proba(X_test)[:, 1]; pred = (test_prob >= threshold).astype(int)
    metrics = {"roc_auc": roc_auc_score(y_test, test_prob), "average_precision": average_precision_score(y_test, test_prob),
               "precision": precision_score(y_test, pred, zero_division=0), "recall": recall_score(y_test, pred, zero_division=0),
               "f1": f1_score(y_test, pred, zero_division=0)}
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", str(ROOT / "mlruns")))
    mlflow.set_experiment("bank-marketing-response")
    with mlflow.start_run(run_name="balanced-logistic-regression"):
        mlflow.log_params({"model": "logistic_regression", "class_weight": "balanced", "threshold": threshold,
                           "seed": 42, "duration_excluded": True, "split": "stratified_60_20_20"})
        mlflow.log_metrics(metrics); mlflow.sklearn.log_model(model, "model")
    out = ROOT / "reports"; out.mkdir(exist_ok=True)
    pd.DataFrame([{"threshold": threshold, **metrics}]).to_csv(out / "test_metrics.csv", index=False)
    (ROOT / "models").mkdir(exist_ok=True); joblib.dump({"pipeline": model, "threshold": threshold}, ROOT / "models/response_model.joblib")
    print({"threshold": threshold, **metrics})


if __name__ == "__main__": main()
