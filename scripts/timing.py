"""Per-stage timing for the local pipeline.

Runs normalize -> cascade -> aggregate -> build_data on 10, 100 and 377 review
subsets, parses cascade's stage marks, and measures the /api/data recompute.

    python scripts/timing.py --reviews data/sample/reviews.csv

Output: docs/data/latency.json (aggregate timings only, no review text).
"""
import argparse
import csv
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PIPELINE = os.path.join(ROOT, "app", "pipeline")
MARK = re.compile(r"\[\s*([\d.]+)s\]\s*(.*)")


def subset(src, n, path):
    with open(src, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        w.writeheader()
        w.writerows(rows[:n])
    return min(n, len(rows))


def run_step(args, env):
    cmd = [sys.executable] + args
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=ROOT, env=env, text=True, encoding="utf-8",
                       errors="replace", capture_output=True)
    dt = time.perf_counter() - t0
    if p.returncode:
        sys.stderr.write(p.stdout)
        sys.stderr.write(p.stderr)
        raise SystemExit(p.returncode)
    return dt, p.stdout


def parse_marks(out):
    marks = [(float(t), msg) for t, msg in MARK.findall(out)]
    if not marks:
        return {}
    start = marks[0][0]
    total = marks[-1][0] - start

    def at(word):
        for t, msg in marks:
            if word in msg:
                return t
        return None

    times = {
        "probe_load": at("loaded probe"),
        "segment": at("segmented"),
        "classify": at("categorized"),
        "embed_facets": at("embedded"),
        "sentiment": at("sentiment"),
    }
    stages, prev = {}, start
    for key in ("probe_load", "segment", "classify", "embed_facets", "sentiment"):
        t = times[key]
        stages[key] = max(0.0, (t - prev)) if t is not None else 0.0
        if t is not None:
            prev = t
    stages["rules_facets_write"] = max(0.0, total - (times["sentiment"] - start if times["sentiment"] is not None else 0.0))
    stages["total"] = total
    return stages


def measure_api(db, stats_path):
    sys.path.insert(0, PIPELINE)
    import aggregate
    import build_data
    aggregate.DB = db
    build_data.DB = db
    aggregate.STATS = stats_path
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        out, _ = aggregate.compute("cat")
        build_data.assemble(out)
        ts.append(time.perf_counter() - t0)
    return ts


def summary(values):
    values = sorted(values)
    return {
        "median_ms": round(statistics.median(values) * 1000, 1),
        "p95_ms": round(values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))] * 1000, 1),
        "min_ms": round(values[0] * 1000, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", default=os.path.join(ROOT, "data", "sample", "reviews.csv"))
    ap.add_argument("--sizes", default="10,100,377")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "data", "latency.json"))
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",") if x.strip()]

    results = {}
    for n in sizes:
        runs = []
        for r in range(a.runs):
            with tempfile.TemporaryDirectory(prefix=f"timing-{n}-") as tmp:
                csv_path = os.path.join(tmp, "reviews.csv")
                subset(a.reviews, n, csv_path)
                db = os.path.join(tmp, "reviews.sqlite")
                env = dict(os.environ)
                env["REVIEWS_DATA_DIR"] = tmp
                env["REVIEWS_DATA_JS"] = os.path.join(tmp, "data.js")
                env["PYTHONUNBUFFERED"] = "1"

                t_norm, _ = run_step([os.path.join(PIPELINE, "normalize.py"),
                                      "--input", csv_path, "--db", db], env)
                t_cas, out = run_step([os.path.join(PIPELINE, "cascade_extract.py"),
                                       "--key", "cat", "--rebuild"], env)
                t_agg, _ = run_step([os.path.join(PIPELINE, "aggregate.py"),
                                     "--run", "cat"], env)
                t_build, _ = run_step([os.path.join(PIPELINE, "build_data.py")], env)
                api = measure_api(db, os.path.join(tmp, "stats.json"))
                stages = parse_marks(out)
                stages.update(normalize=t_norm, aggregate=t_agg, build_data=t_build,
                              cascade_total=t_cas, api_data=statistics.median(api))
                stages["startup_overhead"] = max(0.0, t_cas - stages["total"])
                stages["import_total"] = t_norm + t_cas + t_agg + t_build
                runs.append(stages)
                print(f"n={n} run={r+1}: total={stages['import_total']:.2f}s "
                      f"cascade={t_cas:.2f}s api={stages['api_data']*1000:.0f}ms")
        keys = runs[0].keys()
        results[str(n)] = {
            "input_rows": n,
            "runs": a.runs,
            "summary": {k: summary([r[k] for r in runs]) for k in keys},
        }

    out = {
        "generated": time.strftime("%Y-%m-%d"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": __import__("torch").__version__,
            "cuda": __import__("torch").cuda.is_available(),
            "gpu": __import__("torch").cuda.get_device_name(0) if __import__("torch").cuda.is_available() else "",
        },
        "note": "cold model load each run; /api/data is a warm recompute from the built database",
        "sizes": results,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
