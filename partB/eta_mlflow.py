import argparse
import importlib.metadata as md
import json
import os
import time

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import eta_data

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = "eta-model"
RUSH_HOURS = [7, 8, 9, 16, 17, 18, 19]
EPS = 1e-12                                   
class EtaPyfunc(mlflow.pyfunc.PythonModel):
    """Wraps a {model, feature_columns, version} joblib bundle."""

    def load_context(self, context):
        bundle = joblib.load(context.artifacts["bundle"])
        self.model = bundle["model"]
        self.cols = bundle["feature_columns"]
        self.version = bundle["version"]

    def predict(self, context, model_input, params=None):
        X = pd.DataFrame(model_input)[self.cols]
        preds = np.round(self.model.predict(X), 1)
        return pd.DataFrame({"eta_minutes": preds, "model_version": self.version})


def _pins() -> list[str]:
    pkgs = ["mlflow", "scikit-learn", "pandas", "numpy", "joblib", "cloudpickle"]
    return [f"{p}=={md.version(p)}" for p in pkgs]

def register(data_dir="data") -> list[dict]:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("eta")
    client = MlflowClient()
    example = eta_data.make_data(5, seed=1)[eta_data.FEATURES]
    versions = {}
    for v in ("v1", "v2"):
        path = f"{data_dir}/eta_{v}.joblib"
        bundle = joblib.load(path)
        out = pd.DataFrame({"eta_minutes": np.round(bundle["model"].predict(example), 1),
                            "model_version": bundle["version"]})
        with mlflow.start_run(run_name=f"eta-{v}"):
            info = mlflow.pyfunc.log_model(
                name="model", python_model=EtaPyfunc(), artifacts={"bundle": path},
                signature=infer_signature(example, out), input_example=example,
                pip_requirements=_pins(), code_paths=[os.path.abspath(__file__)],
                registered_model_name=MODEL_NAME)
            versions[v] = info.registered_model_version
    client.set_registered_model_alias(MODEL_NAME, "prod", versions["v1"])
    client.set_registered_model_alias(MODEL_NAME, "candidate", versions["v2"])
    rows = []
    for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
        full = client.get_model_version(MODEL_NAME, mv.version)          
        rows.append({"version": int(full.version), "aliases": sorted(full.aliases)})
    rows.sort(key=lambda r: r["version"])
    print(f"{'version':<8}{'alias':<12}")
    for r in rows:
        print(f"{r['version']:<8}{','.join(r['aliases']) or '-':<12}")
    return rows

def decide(prod: dict, cand: dict, min_rel_gain: float = 0.03, max_p95_ms: float = 10.0,
           max_slice_regression: float = 0.05) -> dict:
    if prod["mae"] > 0:
        gain = (prod["mae"] - cand["mae"]) / prod["mae"]
    else:
        gain = 0.0 if cand["mae"] <= 0 else float("-inf")
    criteria = {
        "overall_mae_gain": bool(gain >= min_rel_gain - EPS),
        "latency_p95": bool(cand["p95_ms"] <= max_p95_ms + EPS),
        "no_regression_rush": bool(cand["mae_rush"] <= prod["mae_rush"] * (1 + max_slice_regression) + EPS),
        "no_regression_rain": bool(cand["mae_rain"] <= prod["mae_rain"] * (1 + max_slice_regression) + EPS),
    }
    return {"criteria": criteria, "promote": all(criteria.values())}


def _evaluate(model, fresh: pd.DataFrame) -> dict:
    X, y = fresh[eta_data.FEATURES], fresh["eta_minutes"].to_numpy()
    pred = model.predict(X)["eta_minutes"].to_numpy()
    err = np.abs(pred - y)
    rush = fresh["hour"].isin(RUSH_HOURS).to_numpy()
    rain = (fresh["rain"] == 1).to_numpy()
    rows = [X.iloc[[i]] for i in range(200)]
    for r in rows[:20]:                                   
        model.predict(r)
    lat = []
    for r in rows:
        t0 = time.perf_counter()
        model.predict(r)
        lat.append((time.perf_counter() - t0) * 1000)
    return {"mae": float(err.mean()), "mae_rush": float(err[rush].mean()), "mae_rain": float(err[rain].mean()),
            "p95_ms": float(np.percentile(lat, 95))}


def promotion_gate(**thresholds) -> dict:
    mlflow.set_tracking_uri(TRACKING_URI)
    prod = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@prod")
    cand = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@candidate")
    fresh = eta_data.make_data(2000, seed=99)
    mp, mc = _evaluate(prod, fresh), _evaluate(cand, fresh)
    result = decide(mp, mc, **thresholds)
    print("thresholds:", thresholds or "defaults")
    print("prod     :", {k: round(v, 3) for k, v in mp.items()})
    print("candidate:", {k: round(v, 3) for k, v in mc.items()})
    print(f"relative MAE gain: {(mp['mae'] - mc['mae']) / mp['mae']:.3f}")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["register", "gate"])
    ap.add_argument("--min-rel-gain", type=float)
    ap.add_argument("--max-p95-ms", type=float)
    ap.add_argument("--max-slice-regression", type=float)
    a = ap.parse_args()
    if a.cmd == "register":
        register()
    else:
        th = {k: v for k, v in dict(min_rel_gain=a.min_rel_gain, max_p95_ms=a.max_p95_ms,
                                    max_slice_regression=a.max_slice_regression).items() if v is not None}
        promotion_gate(**th)
