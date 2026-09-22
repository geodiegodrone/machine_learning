from pathlib import Path
import os
import urllib.request
import zipfile

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"


def main():
    raw = ROOT / "data/raw"; raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "energy.zip"
    if not archive.exists(): urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith("energydata_complete.csv"))
        with zf.open(name) as f: data = pd.read_csv(f)
    data["date"] = pd.to_datetime(data["date"])
    data["hour"] = data.date.dt.hour; data["day_of_week"] = data.date.dt.dayofweek
    data["month"] = data.date.dt.month
    target = "Appliances"
    features = data.drop(columns=[target, "date", "rv1", "rv2"])
    n = int(len(data) * .8); Xtr, Xte = features.iloc[:n], features.iloc[n:]
    ytr, yte = data[target].iloc[:n], data[target].iloc[n:]
    model = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(max_iter=200, learning_rate=.08, random_state=42))
    model.fit(Xtr, ytr)
    baseline = np.repeat(ytr.mean(), len(yte)); pred = model.predict(Xte)
    def metrics(y, p): return {"mae": mean_absolute_error(y, p), "rmse": mean_squared_error(y, p) ** .5, "r2": r2_score(y, p)}
    results = [{"model": "training_mean_baseline", **metrics(yte, baseline)}, {"model": "hist_gradient_boosting", **metrics(yte, pred)}]
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", str(ROOT / "mlruns")))
    mlflow.set_experiment("home-energy-forecasting")
    for row in results:
        with mlflow.start_run(run_name=row["model"]):
            mlflow.log_params({"model": row["model"], "split": "chronological_80_20", "seed": 42, "excluded": "rv1,rv2"})
            mlflow.log_metrics({k: v for k, v in row.items() if k != "model"})
            if row["model"] == "hist_gradient_boosting": mlflow.sklearn.log_model(model, "model")
    out = ROOT / "reports"; out.mkdir(exist_ok=True); pd.DataFrame(results).to_csv(out / "metrics.csv", index=False)
    (ROOT / "models").mkdir(exist_ok=True); joblib.dump(model, ROOT / "models/energy_model.joblib")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__": main()
