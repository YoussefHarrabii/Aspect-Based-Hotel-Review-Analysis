"""Turn stats.json + the reviews table into web/data.js for the dashboard.

    python build_data.py

Emits window.DATA = { generated, hotel:{...}, reviews:[...], mentions:[...],
taxonomy:{...} } - the shape the UI consumes.
"""
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)           # app/
ROOT = os.path.dirname(APP)           # repo root
sys.path.insert(0, ROOT)
sys.path.insert(0, APP)
sys.path.insert(0, HERE)

DATA = os.getenv("REVIEWS_DATA_DIR") or os.path.join(ROOT, "data")
DB = os.path.join(DATA, "reviews.sqlite")
STATS = os.path.join(DATA, "stats.json")
BUCKETS = os.path.join(ROOT, "config", "buckets.json")
OUT = os.getenv("REVIEWS_DATA_JS") or os.path.join(APP, "web", "data.js")
RUN = "cat"


def assemble(stats, run=None):
    run = run or RUN
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT review_id, rating, bucket, text, lang, "
        "published_date, owner_response FROM reviews "
        "ORDER BY published_date DESC"
    ).fetchall()
    stats["reviews"] = [{"id": rid, "rating": rating, "bucket": b,
                         "text": text, "language": lang, "date": pdate, "owner": owner}
                        for rid, rating, b, text, lang, pdate, owner in rows]

    # [aspect, sentiment, facet, facet_sim, rating, date, evidence, review_id]
    mrows = conn.execute(
        "SELECT m.aspect, m.sentiment, m.facet, m.facet_sim, r.rating, "
        "r.published_date, m.evidence, m.review_id "
        "FROM mentions m JOIN reviews r ON r.review_id = m.review_id "
        "WHERE m.run_key = ? AND r.text != '' AND m.aspect != 'Location'", (run,)
    ).fetchall()
    stats["mentions"] = [[a, s, f, round(float(sim or 0.0), 4), rating, (pd or "")[:10], ev, rid]
                         for a, s, f, sim, rating, pd, ev, rid in mrows]

    stats["taxonomy"] = json.load(open(BUCKETS, encoding="utf-8"))["buckets"]

    conn.close()
    return stats


def main():
    stats = json.load(open(STATS, encoding="utf-8"))
    stats = assemble(stats)
    with open(OUT, "w", encoding="utf-8") as f:
        if OUT.endswith(".json"):
            json.dump(stats, f, ensure_ascii=False)
        else:
            f.write("window.DATA = ")
            json.dump(stats, f, ensure_ascii=False)
            f.write(";\n")
    print(f"wrote {OUT}: reviews={len(stats['reviews'])} mentions={len(stats['mentions'])}")


if __name__ == "__main__":
    main()
