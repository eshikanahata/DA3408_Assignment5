import argparse
import time

import numpy as np
import pandas as pd
import requests

import eta_data


def call(sess, url, rec):
    t0 = time.perf_counter()
    r = sess.post(url, json={"dataframe_records": [rec]}, timeout=30)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    p = r.json()["predictions"][0]
    return float(p["eta_minutes"]), p["model_version"], ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", default="http://127.0.0.1:5101/invocations")
    ap.add_argument("--cand", default="http://127.0.0.1:5102/invocations")
    ap.add_argument("--data", default="data/eta_fresh.csv")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    rows = pd.read_csv(a.data)[eta_data.FEATURES].head(a.n).to_dict(orient="records")
    sp, sc = requests.Session(), requests.Session()
    for rec in rows[:10]:                                  
        call(sp, a.prod, rec); call(sc, a.cand, rec)
    pp, pc, lp, lc, vp, vc = [], [], [], [], set(), set()
    for rec in rows:
        e1, v1, m1 = call(sp, a.prod, rec)
        e2, v2, m2 = call(sc, a.cand, rec)
        pp.append(e1); pc.append(e2); lp.append(m1); lc.append(m2); vp.add(v1); vc.add(v2)
    pp, pc = np.array(pp), np.array(pc)
    d = np.abs(pp - pc)
    print(f"requests compared (identical rows to both servers): {len(rows)}")
    print(f"versions answering: prod={sorted(vp)} candidate={sorted(vc)}")
    print(f"(i)   mean |prod - candidate| = {d.mean():.2f} min   (median {np.median(d):.2f}, max {d.max():.2f})")
    print(f"(ii)  requests differing by > 10 min = {100 * (d > 10).mean():.1f}%  ({int((d > 10).sum())}/{len(d)})")
    print(f"(iii) p95 latency  prod={np.percentile(lp, 95):.1f} ms   candidate={np.percentile(lc, 95):.1f} ms"
          f"   (p50 prod={np.median(lp):.1f}, candidate={np.median(lc):.1f})")
    truth = pd.read_csv(a.data)["eta_minutes"].head(a.n).to_numpy()
    print(f"extra (uses ground truth, not asked): MAE on these rows prod={np.abs(pp - truth).mean():.2f}  "
          f"candidate={np.abs(pc - truth).mean():.2f}")


if __name__ == "__main__":
    main()
