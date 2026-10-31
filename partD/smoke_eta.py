import argparse
import sys
import time

import numpy as np
import requests

BASE = dict(distance_km=20.0, n_stops=5, hour=11, weekday=2, traffic_index=0.4, rain=0, weight_kg=10.0)
CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


class Ctx:
    def __init__(self, a):
        self.url, self.expect, self.max_p95 = a.url, a.expect_version, a.max_p95_ms
        self.headers = dict(h.split(":", 1) for h in (a.header or []))
        self.headers = {k.strip(): v.strip() for k, v in self.headers.items()}
        self.s = requests.Session()

    def post(self, rows, extra_headers=None):
        body = {"customer_id": "smoke-test", "instances": rows}
        return self.s.post(self.url, json=body, headers={**self.headers, **(extra_headers or {})}, timeout=10)

    def eta(self, **over):
        r = self.post([{**BASE, **over}])
        r.raise_for_status()
        return r.json()["predictions"][0]


@check("valid request -> 200 with one numeric prediction per instance")
def c_valid(c):
    r = c.post([BASE, {**BASE, "distance_km": 30}])
    j = r.json()
    ok = r.status_code == 200 and len(j["predictions"]) == 2 and all(isinstance(p, (int, float)) for p in j["predictions"])
    return ok, f"status={r.status_code} predictions={j.get('predictions')}"


@check("served model_version equals --expect-version")
def c_version(c):
    j = c.post([BASE]).json()
    return j.get("model_version") == c.expect, f"got {j.get('model_version')!r}, expected {c.expect!r}"


@check("sanity: predictions are plausible (0 < ETA < 1000 min)")
def c_range(c):
    p = c.eta()
    return 0 < p < 1000, f"eta={p}"


@check("sanity: longer distance must not give a shorter ETA (5 < 25 < 60 < 120 km)")
def c_distance(c):
    etas = [c.eta(distance_km=d) for d in (5, 25, 60, 120)]
    return all(b >= a for a, b in zip(etas, etas[1:])), f"etas={etas}"


@check("sanity: heavier traffic must not shorten the ETA (0.1 -> 0.5 -> 0.9)")
def c_traffic(c):
    etas = [c.eta(distance_km=40, hour=8, traffic_index=t) for t in (0.1, 0.5, 0.9)]
    return all(b >= a for a, b in zip(etas, etas[1:])), f"etas={etas}"


@check("batch order preserved (batch == the same rows sent one by one)")
def c_batch(c):
    rows = [{**BASE, "distance_km": d} for d in (60, 5, 120, 25)]
    batch = c.post(rows).json()["predictions"]
    singles = [c.eta(distance_km=d) for d in (60, 5, 120, 25)]
    return batch == singles, f"batch={batch} singles={singles}"


@check("invalid input (hour=99) -> HTTP 422")
def c_422(c):
    r = c.post([{**BASE, "hour": 99}])
    return r.status_code == 422, f"status={r.status_code}"


@check("infeasible delivery (distance 151 km & 21 stops) -> HTTP 422")
def c_infeasible(c):
    r = c.post([{**BASE, "distance_km": 151, "n_stops": 21}])
    return r.status_code == 422, f"status={r.status_code}"


@check("latency: p95 of 40 single-row requests <= --max-p95-ms")
def c_latency(c):
    for _ in range(5):
        c.post([BASE])                                      
    lat = []
    for _ in range(40):
        t0 = time.perf_counter()
        c.post([BASE]).raise_for_status()
        lat.append((time.perf_counter() - t0) * 1000)
    p95 = float(np.percentile(lat, 95))
    return p95 <= c.max_p95, f"p95={p95:.1f} ms (limit {c.max_p95} ms)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--expect-version", required=True)
    ap.add_argument("--max-p95-ms", type=float, default=80.0)
    ap.add_argument("--header", action="append", help='extra header, e.g. "X-Beta: 1" (repeatable)')
    a = ap.parse_args()
    c = Ctx(a)
    failed = 0
    for name, fn in CHECKS:
        try:
            ok, detail = fn(c)
        except Exception as e:                            
            ok, detail = False, f"{type(e).__name__}: {str(e)[:120]}"
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed -> {'RELEASE OK' if not failed else 'RELEASE BLOCKED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
