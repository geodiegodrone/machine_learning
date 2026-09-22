from pathlib import Path
import os
import urllib.request
import zipfile

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/292/wholesale+customers.zip"
FEATURES = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]


def main():
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "wholesale.zip"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith("data.csv"))
        with zf.open(name) as f:
            data = pd.read_csv(f)
    X = data[FEATURES]
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", str(ROOT / "mlruns")))
    mlflow.set_experiment("wholesale-customer-segmentation")
    best = (-1.0, None, None)
    for k in range(2, 9):
        pipeline = Pipeline([("log", FunctionTransformer(lambda x: __import__("numpy").log1p(x), validate=False)),
                             ("scale", RobustScaler()), ("cluster", KMeans(n_clusters=k, n_init=20, random_state=42))])
        labels = pipeline.fit_predict(X)
        score = silhouette_score(pipeline[:-1].transform(X), labels)
        with mlflow.start_run(run_name=f"kmeans-k{k}"):
            mlflow.log_params({"algorithm": "KMeans", "k": k, "n_init": 20, "seed": 42, "features": ",".join(FEATURES)})
            mlflow.log_metrics({"silhouette": score, "customers": len(X)})
            mlflow.sklearn.log_model(pipeline, "model")
        if score > best[0]:
            best = (score, k, pipeline)
    labels = best[2].predict(X)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    profile = X.assign(segment=labels).groupby("segment")[FEATURES].agg(["count", "mean", "median"])
    profile.to_csv(out / "segment_profiles.csv")
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(best[2], ROOT / "models/segmenter.joblib")
    print(f"Selected k={best[1]}, silhouette={best[0]:.4f}")


if __name__ == "__main__":
    main()
