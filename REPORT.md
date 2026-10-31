# Module 4 — Assignment 5: SwiftParcel ETA service — Report
**Roll no.: DA24B004**

**Machine:** IdeaPad laptop, AMD Ryzen 7 5700U, 16 logical CPUs (`lscpu`), 12 GiB RAM (about 1 GiB free during the first load-test attempt), Linux, Python 3.14.4, mlflow 3.17.2, ray 2.59.0. The load generator ran on the same machine as the server. All latency/throughput numbers are single 4-second runs, not repeated.

**Sources and AI assistance:** `eta_data.py` and `load_test.py` are the provided starter files. The code follows the patterns from the lecture notebooks (REST service, pyfunc + alias registry, Ray Serve router/handle composition); no notebook cells were copied. TODO-CONFIRM: state here how an AI assistant was used and whether the course policy requires disclosure.

**Reproduce:** see `README.md` (clone, `pip install -r requirements.txt`, `source env.sh`, `python eta_data.py`, then the per-part commands; all commands run from the repo root). Tests: `pytest -q partA/test_eta_app.py partB/test_gate.py`.

---
## Part A — REST service

**A1/A2.** `eta_app.py` loads the bundle once at import time (path from `ETA_MODEL`). Validation uses pydantic (ranges, batch 1–500, cross-field infeasible rule). A middleware sets `X-Request-ID` on every response (client's value or a UUID) and records metrics (`/metrics` itself is not counted; unknown paths are folded into `path="other"` to bound label cardinality). Prediction errors return a generic 500 without internals. `test_eta_app.py`: **21 tests passed** (`21 passed in 6.26s`). During development I also broke the code on purpose in three places (infeasible threshold, batch limit, request-id header) and each change made at least one test fail.

**A3.1 Load test** (v1, one uvicorn worker, 4 s per level, 0 errors):

| conc. | requests | rps | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|
| 1 | 722 | 180.5 | 5.5 | 6.8 | 7.1 |
| 2 | 887 | 221.8 | 8.8 | 11.4 | 14.3 |
| 4 | 908 | 227.0 | 17.2 | 22.9 | 29.3 |
| 8 | 958 | 239.5 | 32.7 | 45.9 | 57.5 |
| 16 | 1076 | 269.0 | 57.6 | 84.0 | 102.4 |

An earlier run was discarded: it hit another process on port 8000 (every request failed), and a second attempt was too noisy (concurrency 2 slower than 1) because of background load. The table above is a clean rerun.

**A3.2 Saturation and capacity.** Throughput rises from 180 to 222 rps between concurrency 1 and 2 and then barely grows (227, 240, 269 rps) while latency roughly doubles with each doubling of concurrency, so the knee is at about concurrency 2: one Python process is saturated and extra clients only queue. Little's Law (L = λ·W) agrees: at concurrency 16, 16 / 269 rps = 59.5 ms against a measured p50 of 57.6 ms; at concurrency 4, 4 / 227 = 17.6 ms against 17.2 ms (I use p50 as a proxy for the mean, which load_test.py does not report). Under the SLO p95 ≤ 80 ms the largest passing level is concurrency 8 (p95 45.9 ms, 239.5 rps); concurrency 16 fails (84.0 ms). **Capacity ≈ 240 rps**; the exact limit lies between concurrency 8 and 16, which I did not test.

**A3.3 Batch cost** (concurrency 1):

| batch | rps | ms/request | ms/row |
|---|---|---|---|
| 1 | 190.2 | 5.26 | 5.26 |
| 100 | 142.2 | 7.03 | 0.070 |

Batch 100 is about 75× cheaper per row. A 100-row request takes only 1.8 ms longer than a 1-row request; from these two points the fixed per-request cost is roughly 5.2 ms (HTTP handling, validation, DataFrame construction, the predict call, JSON) and the marginal cost about 0.02 ms per row (an estimate from two points). Batching amortises the fixed cost.

**A3.4 Change: 4 worker processes** (`uvicorn eta_app:app --workers 4`). Reason: the GIL confines one process to about one core, while the machine has 16 logical CPUs. The baseline above was measured against a server started without `--workers`, and this table against one started with `--workers 4`.

| conc. | 1 worker rps / p95 ms | 4 workers rps / p95 ms |
|---|---|---|
| 1 | 180.5 / 6.8 | 173.0 / 7.1 |
| 2 | 221.8 / 11.4 | 321.8 / 7.9 |
| 4 | 227.0 / 22.9 | 471.2 / 12.4 |
| 8 | 239.5 / 45.9 | 565.5 / 22.0 |
| 16 | 269.0 / 84.0 | 580.5 / 46.4 |

Peak throughput rose 2.2× (269 → 580 rps). Capacity under the SLO rose from about 240 rps to at least 580 rps (≥ 2.4×; concurrency 16 still met p95 ≤ 80 ms, so this is a lower bound — I did not test higher concurrency). It did not reach 4×: throughput plateaus around 570 rps. Likely causes are the load generator (a Python program on the same machine) and the machine having fewer physical cores than logical CPUs (hyper-threads); I did not measure either, so these are hypotheses. Single-request latency is unchanged (concurrency 1), as expected. Limitation: each worker keeps its own `/metrics` counters, so a scrape sees only one worker; Prometheus multiprocess mode would be needed in production.

**A3.5 Why p95/p99, not the mean.** The mean hides the slow tail — a few very slow requests barely move it, but they are what users notice, and under queueing the tail grows much faster than the mean. SLOs are written on percentiles because they describe what the slowest 5% or 1% of requests experience.

---
## Part B — MLflow

**B1 Registration.** `eta_mlflow.py` defines `EtaPyfunc`, wrapping a bundle and returning a DataFrame with `eta_minutes` (one decimal) and `model_version`. Both bundles are logged with an inferred signature, an input example, pinned `pip_requirements` (exact installed versions of mlflow, scikit-learn, pandas, numpy, joblib, cloudpickle) and registered as `eta-model` in a SQLite store (`sqlite:///mlflow.db`, or `MLFLOW_TRACKING_URI`).

| version | alias | model |
|---|---|---|
| 1 | prod | v1 (Ridge) |
| 2 | candidate | v2 (GradientBoosting) |

**B2 Gate.** `decide()` is pure. Slice regression is **relative**: candidate slice MAE ≤ prod slice MAE × (1 + `max_slice_regression`) — the brief does not say relative or absolute; I chose relative to be consistent with `min_rel_gain`. `test_gate.py`: 9 tests passed, including values exactly on and just outside each threshold. `promotion_gate()` evaluates on `make_data(2000, seed=99)` and measures p95 over 200 single-row in-process predictions (not HTTP). Output (`gate_output.txt`):

| | MAE | MAE rush | MAE rain | p95 ms |
|---|---|---|---|---|
| prod (v1) | 11.955 | 19.680 | 11.623 | 2.549 |
| candidate (v2) | 6.105 | 8.580 | 6.285 | 3.000 |

Relative MAE gain 0.489. Default thresholds: all four criteria true → **promote = true**. With `min_rel_gain=0.60`: `overall_mae_gain` false, the others true → **promote = false** (p95 2.639 / 3.059 ms in that run).

**B3 Shadow.** The same 200 single-row requests from `data/eta_fresh.csv` were sent alternately to prod (5101) and candidate (5102), both served with `mlflow models serve --env-manager local`:
(i) mean absolute difference **9.48 min** (median 7.10, max 51.70); (ii) **35.0%** of requests (70/200) differ by more than 10 min; (iii) p95 latency **10.8 ms** prod and **11.2 ms** candidate (p50 8.4 / 9.2 ms). Extra check using ground truth from the same rows: MAE 11.42 (prod) vs 6.11 (candidate).

*Decision.* I would promote v2, but gradually through the canary rather than all at once. The models disagree a lot (mean 9.5 min, 35% of requests by more than 10 min), so this is a real change in behaviour, but the labelled rows show the disagreement comes from v1 being less accurate (MAE 11.42 vs 6.11). Latency is essentially the same (10.8 vs 11.2 ms), and the offline gate passes on all four criteria. The caveat is that in real shadow mode labels are not yet available, so disagreement shows how much behaviour changes, not which model is right; here the labels come from the offline fresh data.

*Why moving an alias does not change a running server.* An alias is only a pointer in the registry database. `models:/eta-model@prod` is resolved to a concrete version and artifact once, when the server starts; the server then keeps that model in memory and never consults the registry again. Demonstrated: after setting `prod` to version 2, the running server on port 5101 still answered `model_version: v1` with the same prediction (51.6). The alias was set back to version 1 afterwards.

*Two ways to pick up a new version without dropping requests.* (1) Rolling / blue-green deployment: start new replicas on the new version, wait until their health check passes, shift traffic, then stop the old replicas (as Ray Serve or Kubernetes do). (2) In-process hot reload: the service re-resolves the alias periodically (or on a signal/admin call), loads the new model in the background and swaps the reference atomically, so in-flight requests finish on the old model.

---
## Part C — Ray Serve sticky canary

**C1.** `EtaModel` loads a bundle once per replica. Prod = 2 replicas; candidate uses `autoscaling_config` (min 1, max 3). A FastAPI ingress `Router` takes `{"customer_id", "instances"}`, returns the model output plus `"arm"`. Bucketing: `SHA-256("eta-canary-v1:" + customer_id)`, first 8 bytes as a fraction in [0, 1); the customer goes to the candidate iff bucket < `weight`. This is sticky and monotone in `weight`. `X-Beta: 1` forces the candidate. CPU reservation: router 0.25 + prod 2 × 0.30 + candidate ≤ 3 × 0.30 = **1.75 ≤ 2** even at maximum scale.

**C2 Verification** (`verify_output.txt`, Ray started with `num_cpus=2`, 1000 customer ids `cust-0000`…`cust-0999`):

| check | evidence | result |
|---|---|---|
| (a) share at weight 0.2 within ±4 pp | 0.190 on v2 (1.0 pp off) | pass |
| (b) 200 customers, same arm on second call | 0 changed | pass |
| (c) same after redeploying identical app | 0 changed | pass |
| (d) weight 0.2 → 0.5, nobody leaves candidate | 190 were on v2, 0 moved back; share now 0.507 | pass |
| (e) `X-Beta: 1` for 50 requests | 50/50 answered by v2 (without header: v1) | pass |

At the end the candidate had 2 running replicas and 0.55 CPU was free.

**C3.**
1. Python's `hash()` for strings is randomised per process (hash randomisation), so different replicas, processes or restarts would compute different values for the same customer id; stickiness would break across replicas and after a redeploy. SHA-256 gives the same value everywhere.
2. Stickiness matters because (i) for the product, a customer who sees two different ETAs for the same delivery loses trust in the number; (ii) for the experiment, the unit is the customer — a customer who sees both models contaminates both arms, so their outcome cannot be attributed to either, and per-arm metrics are no longer clean.
3. `target_ongoing_requests=2` is kept below `max_ongoing_requests=5` to leave headroom for bursts. `upscale_delay_s=5` reacts to sustained load within seconds but ignores single blips; `downscale_delay_s=30` avoids flapping, since each new replica must reload the model. `max_replicas=3` is set by the CPU budget (3 × 0.3 + router 0.25 + prod 0.6 = 1.75 of 2 CPUs); `min_replicas=1` suffices because the candidate takes only about 10% of traffic.

---
## Part D — Release gate

**D1.1** `smoke_eta.py --url … --expect-version … [--max-p95-ms 80] [--header …]` runs 9 checks (valid request; served version; plausible range; longer distance never shortens ETA; heavier traffic never shortens ETA; batch equals single requests; `hour=99` → 422; infeasible delivery → 422; p95 of 40 single requests ≤ limit). Exit 0 on success, 1 on any failure (including an unreachable server). Run against my Part A service with v1: **9/9 passed, exit 0**. Run with `--expect-version v2` against the same v1 server: **8/9, version check failed, exit 1**.

**D1.2 Abort criteria for a 10% canary of v2** (thresholds are my proposals and were not exercised):
1. **Errors:** candidate 5xx rate > 1% (or > 3× prod's rate) over a rolling 5-minute window with at least 200 requests.
2. **Latency:** candidate p95 > 80 ms (the SLO) or > 1.25× prod's p95 for 3 consecutive 1-minute windows.
3. **Prediction sanity:** more than 1% of candidate predictions outside 1–480 minutes, or its mean predicted ETA differing from prod's by more than 20% on a comparable traffic mix, over a 30-minute window.
4. **Accuracy on real outcomes:** once delivery times are known, candidate MAE over the last 1,000 completed deliveries more than 5% worse than prod's overall, or in the rush-hour or rain slice (same tolerance as the promotion gate), evaluated hourly.

---
## Not tested
- Why throughput did not reach 4× with 4 workers (load generator or physical-core limit are hypotheses, not measured); capacity with 4 workers beyond concurrency 16.
- Repeated runs / confidence intervals for any latency or throughput number; the discarded runs are described in A3.1.
- Autoscaling under heavy load (`extra_work_ms` was never used); only the replica count at the end of `verify_ray.py` was observed.
- `smoke_eta.py` against the Part C router and the Part B MLflow servers (the latter is not supported).
- The abort-criteria thresholds; Prometheus multiprocess aggregation of `/metrics`.
- Operating system and kernel versions were not recorded; physical core count was not measured (only logical CPUs).
