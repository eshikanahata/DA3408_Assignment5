"""test_eta_app.py -- run from the folder that contains data/:  pytest -q test_eta_app.py"""
import re

import pytest
from fastapi.testclient import TestClient

import eta_app

client = TestClient(eta_app.app)
URL = "/v1/models/eta:predict"
GOOD = dict(distance_km=12.5, n_stops=5, hour=8, weekday=2, traffic_index=0.4, rain=0, weight_kg=10.0)


def post(rows, **kw):
    return client.post(URL, json={"instances": rows}, **kw)


def test_valid_request():
    r = post([GOOD])
    assert r.status_code == 200
    j = r.json()
    assert len(j["predictions"]) == 1 and isinstance(j["predictions"][0], float)
    assert j["predictions"][0] == round(j["predictions"][0], 1)          # one decimal
    assert j["model_version"] == eta_app.MODEL_VERSION and j["request_id"]


def test_batch_order_preserved():
    rows = [dict(GOOD, distance_km=d) for d in (5, 40, 10, 80)]
    batch = post(rows).json()["predictions"]
    singles = [post([r]).json()["predictions"][0] for r in rows]
    assert batch == singles                                              # same order, same values


@pytest.mark.parametrize("field,value", [("distance_km", 0.05), ("distance_km", 501), ("n_stops", 0),
                                         ("n_stops", 61), ("hour", 24), ("weekday", 7),
                                         ("traffic_index", 1.2), ("rain", 2), ("weight_kg", 0),
                                         ("weight_kg", 1001)])
def test_out_of_range_rejected(field, value):
    r = post([dict(GOOD, **{field: value})])
    assert r.status_code == 422
    j = r.json()
    assert j["error"] == "validation_failed" and j["request_id"]
    assert any(field in d["field"] for d in j["details"])


def test_missing_field_rejected():
    bad = {k: v for k, v in GOOD.items() if k != "hour"}
    r = post([bad])
    assert r.status_code == 422 and any("hour" in d["field"] for d in r.json()["details"])


def test_wrong_type_rejected():
    r = post([dict(GOOD, distance_km="far")])
    assert r.status_code == 422 and r.json()["error"] == "validation_failed"


def test_cross_field_infeasible():
    r = post([dict(GOOD, distance_km=151, n_stops=21)])
    assert r.status_code == 422
    assert "infeasible" in r.text
    # boundaries are feasible: strictly greater-than is required on both fields
    assert post([dict(GOOD, distance_km=150, n_stops=21)]).status_code == 200
    assert post([dict(GOOD, distance_km=151, n_stops=20)]).status_code == 200


def test_batch_size_limits():
    assert post([]).status_code == 422                                   # below 1
    assert post([GOOD] * 500).status_code == 200                         # exactly 500
    assert post([GOOD] * 501).status_code == 422                         # above 500


def test_request_id_echo_and_generation():
    r = post([GOOD], headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123" and r.json()["request_id"] == "abc-123"
    r = post([GOOD])
    rid = r.headers["X-Request-ID"]
    assert re.fullmatch(r"[0-9a-f-]{36}", rid) and r.json()["request_id"] == rid


def test_request_id_in_error_response():
    r = post([dict(GOOD, hour=99)], headers={"X-Request-ID": "err-1"})
    assert r.status_code == 422
    assert r.headers["X-Request-ID"] == "err-1" and r.json()["request_id"] == "err-1"
    r = post([dict(GOOD, hour=99)])                                      # generated id in an error
    assert r.headers["X-Request-ID"] == r.json()["request_id"]


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_version": eta_app.MODEL_VERSION}


def _metric(text, name, **labels):
    for line in text.splitlines():
        if line.startswith(name + "{") and all(f'{k}="{v}"' in line for k, v in labels.items()):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def test_metrics_output_and_not_self_counting():
    before = client.get("/metrics").text
    n0 = _metric(before, "eta_requests_total", path="/v1/models/eta:predict", status="200")
    post([GOOD]); post([GOOD])
    text = client.get("/metrics").text
    assert _metric(text, "eta_requests_total", path="/v1/models/eta:predict", status="200") == n0 + 2
    assert 'le="+Inf"' in text and len(re.findall(r"eta_request_latency_seconds_bucket\{le=", text)) >= 4
    assert "eta_request_latency_seconds_sum" in text and "eta_request_latency_seconds_count" in text
    assert 'path="/metrics"' not in text                                 # /metrics is never counted


def test_generic_500_hides_internals(monkeypatch):
    class Boom:
        def predict(self, X):
            raise RuntimeError("secret internal detail /opt/models/x.pkl")
    monkeypatch.setattr(eta_app, "model", Boom())
    r = post([GOOD], headers={"X-Request-ID": "boom-1"})
    assert r.status_code == 500
    assert "secret" not in r.text and "Traceback" not in r.text and "RuntimeError" not in r.text
    assert r.headers["X-Request-ID"] == "boom-1"
