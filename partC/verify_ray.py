import argparse
import concurrent.futures as cf
import sys

import ray
import requests
from ray import serve

import eta_ray

ROW = dict(distance_km=12.5, n_stops=5, hour=8, weekday=2, traffic_index=0.4, rain=0, weight_kg=10.0)
N, W1, W2 = 1000, 0.2, 0.5
IDS = [f"cust-{i:04d}" for i in range(N)]
PASSED = []


def call_all(url, ids, headers=None):
    def one(cid):
        r = requests.post(url, json={"customer_id": cid, "instances": [ROW]}, headers=headers, timeout=30)
        r.raise_for_status()
        j = r.json()
        return cid, (j["arm"], j["model_version"])
    with cf.ThreadPoolExecutor(8) as ex:
        return dict(ex.map(one, ids))


def check(label, cond, evidence):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}\n       {evidence}")
    assert cond, f"{label} failed: {evidence}"
    PASSED.append(label)


def deploy(weight):
    serve.run(eta_ray.build(weight=weight), name="eta", route_prefix="/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}/v1/models/eta:predict"

    ray.init(num_cpus=2, include_dashboard=False, logging_level="ERROR", log_to_driver=False)
    serve.start(http_options={"host": "127.0.0.1", "port": a.port})
    print(f"Ray cluster CPUs: {ray.cluster_resources()['CPU']}")
    worst = eta_ray.CPU_ROUTER + 2 * eta_ray.CPU_REPLICA + 3 * eta_ray.CPU_REPLICA
    print(f"Reserved CPUs at maximum autoscale: router {eta_ray.CPU_ROUTER} + prod 2x{eta_ray.CPU_REPLICA} "
          f"+ candidate 3x{eta_ray.CPU_REPLICA} = {worst:.2f} <= 2.0")
    assert worst <= 2.0

    try:
        deploy(W1)
        print(f"Deployed with weight={W1}. CPUs still free in the cluster: {ray.available_resources().get('CPU', 0):.2f}")
        requests.post(url, json={"customer_id": "warm", "instances": [ROW]}, timeout=30).raise_for_status()

        first = call_all(url, IDS)
        share = sum(arm == "v2" for arm, _ in first.values()) / N
        check(f"(a) candidate share within +-4pp of {W1}", abs(share - W1) <= 0.04,
              f"{N} distinct customers -> {share:.3f} on v2 (target {W1}, diff {100 * abs(share - W1):.1f} pp)")
        check("(a') arm label matches the model that answered",
              all(arm == ver for arm, ver in first.values()), "every response: arm == model_version")

        sample = IDS[::5]                                              
        second = call_all(url, sample)
        moved = [c for c in sample if second[c] != first[c]]
        check("(b) 200 customers get the same arm on a second call", not moved,
              f"{len(sample)} customers, {len(moved)} changed arm; on v2: {sum(second[c][0] == 'v2' for c in sample)}")

        deploy(W1)
        third = call_all(url, sample)
        moved = [c for c in sample if third[c] != first[c]]
        check("(c) same arms after redeploying the identical app", not moved,
              f"{len(sample)} customers re-queried after serve.run(build(weight={W1})), {len(moved)} changed arm")

        deploy(W2)
        after = call_all(url, IDS)
        was_v2 = [c for c in IDS if first[c][0] == "v2"]
        back = [c for c in was_v2 if after[c][0] != "v2"]
        new_share = sum(arm == "v2" for arm, _ in after.values()) / N
        check(f"(d) weight {W1} -> {W2}: no candidate customer moves back to prod", not back,
              f"{len(was_v2)} customers were on v2, {len(back)} moved back; v2 share now {new_share:.3f}; "
              f"{sum(1 for c in IDS if first[c][0] == 'v1' and after[c][0] == 'v2')} prod customers newly moved to v2")

        on_prod = [c for c in IDS if after[c][0] == "v1"][:50]
        plain = call_all(url, on_prod)
        forced = call_all(url, on_prod, headers={"X-Beta": "1"})
        check("(e) X-Beta: 1 forces the candidate for 50 requests",
              all(v == ("v1", "v1") for v in plain.values()) and all(v == ("v2", "v2") for v in forced.values()),
              f"50 prod-assigned customers: without header {set(plain.values())}, with header {set(forced.values())}")
        reps = {n: dict(d.replica_states) for n, d in serve.status().applications["eta"].deployments.items()}
        print(f"\nreplica states at the end (candidate autoscales 1..3, prod fixed at 2): {reps}")
        print(f"CPUs still free in the 2-CPU cluster: {ray.available_resources().get('CPU', 0):.2f}")
        print(f"\nALL {len(PASSED)} CHECKS PASSED")
    finally:
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    try:
        main()
    except AssertionError:
        sys.exit(1)
