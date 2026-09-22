from pathlib import Path
import os
import urllib.request
import zipfile

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def main():
    raw = ROOT / "data/raw"; raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "sms.zip"
    if not archive.exists(): urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as zf:
        name = next(n for n in zf.namelist() if "SMSSpamCollection" in n)
        with zf.open(name) as f: data = pd.read_csv(f, sep="\t", header=None, names=["label", "text"])
    y = (data.label == "spam").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(data.text, y, test_size=.2, stratify=y, random_state=42)
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr.to_frame(), ytr)
    model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                                                 strip_accents="unicode")),
                      ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))])
    model.fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:, 1]; pred = (prob >= .5).astype(int)
    base = dummy.predict(Xte.to_frame())
    def scores(y, p, s): return {"precision": precision_score(y, p, zero_division=0), "recall": recall_score(y, p, zero_division=0),
                                  "f1": f1_score(y, p, zero_division=0), "roc_auc": roc_auc_score(y, s),
                                  "average_precision": average_precision_score(y, s)}
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", str(ROOT / "mlruns")))
    mlflow.set_experiment("sms-spam-classifier")
    for name, estimator, metrics in [("majority_baseline", None, scores(yte, base, base)),
                                     ("tfidf_balanced_logistic", model, scores(yte, pred, prob))]:
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "split": "stratified_80_20", "seed": 42})
            mlflow.log_metrics(metrics)
            if estimator is not None: mlflow.sklearn.log_model(estimator, "model")
    out = ROOT / "reports"; out.mkdir(exist_ok=True)
    pd.DataFrame([{"model": "tfidf_balanced_logistic", **scores(yte, pred, prob)}]).to_csv(out / "test_metrics.csv", index=False)
    (ROOT / "models").mkdir(exist_ok=True); joblib.dump(model, ROOT / "models/spam_classifier.joblib")
    print(scores(yte, pred, prob))


if __name__ == "__main__": main()
