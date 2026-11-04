"""Build the static showcase cache from data/showcase/reviews.csv.

The GitHub Pages page cannot run Python, so this script runs the same pipeline in a
temporary data directory and writes docs/data/showcase.json. Re-run it whenever the
ten reviews change.

    python scripts/build_showcase.py
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PIPELINE = os.path.join(ROOT, "app", "pipeline")
DEFAULT_REVIEWS = os.path.join(ROOT, "data", "showcase", "reviews.csv")
DEFAULT_OUT = os.path.join(ROOT, "docs", "data", "showcase.json")


def run_step(args, env):
    cmd = [sys.executable, os.path.join(PIPELINE, args[0])] + args[1:]
    print(" ".join([os.path.basename(cmd[0])] + args), flush=True)
    result = subprocess.run(cmd, cwd=ROOT, env=env, text=True, encoding="utf-8",
                            errors="replace", capture_output=True)
    if result.returncode:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    if result.stdout.strip():
        print(result.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", default=DEFAULT_REVIEWS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    sys.path.insert(0, PIPELINE)
    import aggregate
    import build_data

    with tempfile.TemporaryDirectory(prefix="showcase-") as tmp:
        db = os.path.join(tmp, "reviews.sqlite")
        env = dict(os.environ)
        env["REVIEWS_DATA_DIR"] = tmp
        env["REVIEWS_DATA_JS"] = os.path.join(tmp, "data.js")
        env["PYTHONUNBUFFERED"] = "1"
        run_step(["normalize.py", "--input", a.reviews, "--db", db], env)
        run_step(["cascade_extract.py", "--key", "showcase", "--rebuild"], env)

        aggregate.DB = db
        aggregate.STATS = os.path.join(tmp, "stats.json")
        out, key = aggregate.compute("showcase")
        build_data.DB = db
        build_data.RUN = "showcase"
        data = build_data.assemble(out, "showcase")

    data["mentions"] = [m for m in data.get("mentions", [])
                        if m[0] != "Location" and (m[2] or "") != "General Experience"]
    totals = {"negative": 0, "neutral": 0, "positive": 0}
    for m in data["mentions"]:
        if m[1] in totals:
            totals[m[1]] += 1
    data["hotel"]["mention_totals"] = totals

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"wrote {a.out}: reviews={len(data['reviews'])} mentions={len(data['mentions'])}")


if __name__ == "__main__":
    main()
