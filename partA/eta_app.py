import os
import threading
import time
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, model_validator

MODEL_PATH = os.environ.get("ETA_MODEL", "data/eta_v1.joblib")
_bundle = joblib.load(MODEL_PATH)              
model = _bundle["model"]                     
FEATURE_COLUMNS = _bundle["feature_columns"]
MODEL_VERSION = _bundle["version"]

app = FastAPI(title="SwiftParcel ETA service")
class Instance(BaseModel):
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def _feasible(self):
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery: distance_km > 150 together with n_stops > 20")
        return self


class PredictRequest(BaseModel):
    instances: list[Instance] = Field(min_length=1, max_length=500)

BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
_lock = threading.Lock()
_counts: dict[tuple[str, int], int] = {}
_bucket_counts = [0] * len(BUCKETS)            
_hist_inf = 0                                
_hist_sum = 0.0
_hist_count = 0
KNOWN_PATHS = {"/v1/models/eta:predict", "/healthz", "/metrics"}


def _observe(path: str, status: int, seconds: float) -> None:
    global _hist_inf, _hist_sum, _hist_count
    path = path if path in KNOWN_PATHS else "other"          
    with _lock:
        _counts[(path, status)] = _counts.get((path, status), 0) + 1
        _hist_sum += seconds
        _hist_count += 1
        for i, b in enumerate(BUCKETS):
            if seconds <= b:
                _bucket_counts[i] += 1
                return
        _hist_inf += 1


def _render_metrics() -> str:
    lines = ["# HELP eta_requests_total Requests by path and status.",
             "# TYPE eta_requests_total counter"]
    with _lock:
        for (path, status), n in sorted(_counts.items()):
            lines.append(f'eta_requests_total{{path="{path}",status="{status}"}} {n}')
        lines += ["# HELP eta_request_latency_seconds Request latency.",
                  "# TYPE eta_request_latency_seconds histogram"]
        cum = 0
        for b, n in zip(BUCKETS, _bucket_counts):
            cum += n
            lines.append(f'eta_request_latency_seconds_bucket{{le="{b}"}} {cum}')
        lines.append(f'eta_request_latency_seconds_bucket{{le="+Inf"}} {cum + _hist_inf}')
        lines.append(f"eta_request_latency_seconds_sum {_hist_sum:.6f}")
        lines.append(f"eta_request_latency_seconds_count {_hist_count}")
    return "\n".join(lines) + "\n"

@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:                                       
        response = JSONResponse({"error": "internal_error", "request_id": rid}, status_code=500)
    response.headers["X-Request-ID"] = rid
    if request.url.path != "/metrics":                       
        _observe(request.url.path, response.status_code, time.perf_counter() - t0)
    return response

@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    details = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"] if p != "body")
        msg = str(err["msg"]).removeprefix("Value error, ")
        details.append({"field": field, "msg": msg})
    return JSONResponse(status_code=422,
                        content={"error": "validation_failed", "details": details,
                                 "request_id": request.state.request_id})

@app.post("/v1/models/eta:predict")
def predict(req: PredictRequest, request: Request):
    rid = request.state.request_id
    try:
        frame = pd.DataFrame([i.model_dump() for i in req.instances])[FEATURE_COLUMNS]
        preds = [round(float(p), 1) for p in model.predict(frame)]
    except Exception:
        return JSONResponse({"error": "internal_error", "request_id": rid}, status_code=500)
    return {"predictions": preds, "model_version": MODEL_VERSION, "request_id": rid}


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(_render_metrics(), media_type="text/plain; version=0.0.4")
