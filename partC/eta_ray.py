import asyncio
import hashlib
import os
from typing import Optional, Union

import joblib
import pandas as pd
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from ray import serve

def _data_dir() -> str:
    if os.environ.get("ETA_DATA_DIR"):
        return os.environ["ETA_DATA_DIR"]
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "data"), os.path.join(os.path.dirname(here), "data"), os.path.join(os.getcwd(), "data")):
        if os.path.exists(os.path.join(cand, "eta_v1.joblib")):
            return cand
    return os.path.join(here, "data")


DATA_DIR = _data_dir()
FEATURES = ["distance_km", "n_stops", "hour", "weekday", "traffic_index", "rain", "weight_kg"]
CPU_ROUTER, CPU_REPLICA = 0.25, 0.30
SALT = "eta-canary-v1"                        

def bucket(customer_id: str) -> float:
    digest = hashlib.sha256(f"{SALT}:{customer_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def arm_for(customer_id: str, weight: float) -> str:
    return "v2" if bucket(customer_id) < weight else "v1"


@serve.deployment(max_ongoing_requests=5)
class EtaModel:
    def __init__(self, bundle_path: str, extra_work_ms: float = 0.0):
        bundle = joblib.load(bundle_path)                 
        self.model, self.cols, self.version = bundle["model"], bundle["feature_columns"], bundle["version"]
        self.extra_work_s = extra_work_ms / 1000.0

    async def predict(self, rows: list) -> dict:
        if self.extra_work_s:
            await asyncio.sleep(self.extra_work_s)        
        X = pd.DataFrame(rows)[self.cols]
        preds = [round(float(p), 1) for p in self.model.predict(X)]
        return {"predictions": preds, "model_version": self.version}

class PredictIn(BaseModel):
    customer_id: Union[str, int]
    instances: list[dict] = Field(min_length=1, max_length=500)


api = FastAPI(title="SwiftParcel ETA canary router")


@serve.deployment(ray_actor_options={"num_cpus": CPU_ROUTER})
@serve.ingress(api)
class Router:
    def __init__(self, prod, cand, weight: float):
        self.prod, self.cand, self.weight = prod, cand, float(weight)

    @api.post("/v1/models/eta:predict")
    async def predict(self, req: PredictIn, x_beta: Optional[str] = Header(default=None)):
        missing = {f for r in req.instances for f in FEATURES if f not in r}
        if missing:
            return JSONResponse({"error": "validation_failed", "missing_features": sorted(missing)}, status_code=422)
        forced = (x_beta or "").strip() == "1"
        arm = "v2" if forced else arm_for(str(req.customer_id), self.weight)
        handle = self.cand if arm == "v2" else self.prod
        try:
            out = await handle.predict.remote(req.instances)
        except Exception:
            return JSONResponse({"error": "internal_error"}, status_code=500)
        return {**out, "arm": arm}

    @api.get("/healthz")
    async def healthz(self):
        return {"status": "ok", "weight": self.weight}


def build(weight: float = 0.1, extra_work_ms: float = 0.0):
    prod = EtaModel.options(
        name="EtaProd", num_replicas=2, ray_actor_options={"num_cpus": CPU_REPLICA},
    ).bind(os.path.join(DATA_DIR, "eta_v1.joblib"), extra_work_ms)
    cand = EtaModel.options(
        name="EtaCandidate", ray_actor_options={"num_cpus": CPU_REPLICA},
        autoscaling_config={"min_replicas": 1, "max_replicas": 3, "target_ongoing_requests": 2,
                            "upscale_delay_s": 5, "downscale_delay_s": 30},
    ).bind(os.path.join(DATA_DIR, "eta_v2.joblib"), extra_work_ms)
    return Router.bind(prod, cand, weight)
