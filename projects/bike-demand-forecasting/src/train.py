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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


def main():
    raw = ROOT / "data/raw"; raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "bike.zip"
    if not archive.exists(): urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as zf:
        with zf.open(next(n for n in zf.namelist() if n.endswith("hour.csv"))) as f: data = pd.read_csv(f)
    data["datetime"] = pd.to_datetime(data["dteday"]) + pd.to_timedelta(data["hr"], unit="h")
    data = data.sort_values("datetime").reset_index(drop=True)
    data["month"] = data.datetime.dt.month; data["day_of_year"] = data.datetime.dt.dayofyear
    target = "cnt"
    cats = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
    nums = ["temp", "atemp", "hum", "windspeed", "day_of_year"]
    n = int(len(data) * .8); train, test = data.iloc[:n], data.iloc[n:]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cats),
                             ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), nums)])
    models = {"seasonal_naive": None, "hist_gradient_boosting": make_pipeline(pre, HistGradientBoostingRegressor(max_iter=200, learning_rate=.08, l2_regularization=1, random_state=42))}
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", str(ROOT / "mlruns")))
    mlflow.set_experiment("bike-demand-forecasting")
    results = []
    for name, model in models.items():
        if model is None:
            # Hour-of-week median baseline uses training observations only.
            lookup = train.assign(hour_week=(train.weekday * 24 + train.hr)).groupby("hour_week")[target].median()
            pred = (test.weekday * 24 + test.hr).map(lookup).fillna(train[target].median()).to_numpy()
        else:
            model.fit(train[cats + nums], train[target]); pred = model.predict(test[cats + nums])
        metrics = {"mae": mean_absolute_error(test[target], pred), "rmse": mean_squared_error(test[target], pred) ** .5,
                   "r2": r2_score(test[target], pred)}
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "split": "chronological_80_20", "seed": 42, "excluded_leakage": "casual,registered"})
            mlflow.log_metrics(metrics)
            if model is not None: mlflow.sklearn.log_model(model, "model")
        results.append({"model": name, **metrics})
    out = ROOT / "reports"; out.mkdir(exist_ok=True); pd.DataFrame(results).to_csv(out / "metrics.csv", index=False)
    (ROOT / "models").mkdir(exist_ok=True); joblib.dump(models["hist_gradient_boosting"], ROOT / "models/demand_model.joblib")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__": main()
