"""Load one local review CSV into the SQLite table used by the pipeline.

The CSV contract is ``review_id,date,rating,text``. Ingestion is local only:
no network access, no scraper. See ``docs/data-schema.md``.

    python app/pipeline/normalize.py --input data/sample/reviews.csv
"""
import argparse
import csv
import os
import re
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)           # app/
ROOT = os.path.dirname(APP)           # repo root

DATA = os.getenv("REVIEWS_DATA_DIR") or os.path.join(ROOT, "data")
DB = os.path.join(DATA, "reviews.sqlite")
DEFAULT_CSV = os.path.join(DATA, "sample", "reviews.csv")


def bucket(rating):
    if rating is None:
        return "unknown"
    if rating <= 2:
        return "critical"
    if rating == 3:
        return "mixed"
    return "praise"


def init(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS reviews(
        review_id TEXT PRIMARY KEY, rating INTEGER, bucket TEXT,
        text TEXT, lang TEXT, published_date TEXT,
        reviewer TEXT, owner_response TEXT, url TEXT,
        original_text TEXT, translated TEXT)""")
    conn.execute("CREATE TABLE IF NOT EXISTS extracted(review_id TEXT PRIMARY KEY, ts TEXT)")
    conn.execute("""CREATE TABLE IF NOT EXISTS aspects(
        review_id TEXT, topic TEXT, polarity TEXT, severity INTEGER, quote TEXT)""")
    have = {r[1] for r in conn.execute("PRAGMA table_info(reviews)")}
    for col in ("original_text", "translated"):
        if col not in have:
            conn.execute(f"ALTER TABLE reviews ADD COLUMN {col} TEXT")
    conn.commit()


def _first(row, *names):
    for n in names:
        if n in row and str(row[n]).strip() not in ("", "None"):
            return str(row[n]).strip()
    return ""


def _rating(raw):
    m = re.search(r"-?\d+", str(raw))
    return int(m.group()) if m else None


def read_csv(path):
    """Read the dashboard CSV contract. Unknown extra columns are ignored."""
    out = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            rating = _rating(_first(r, "rating", "review_rating", "stars", "score"))
            out.append(dict(
                review_id=_first(r, "review_id", "id") or f"review-{i}",
                rating=rating, bucket=bucket(rating),
                text=_first(r, "text", "review_text", "content", "body", "comment"),
                lang=_first(r, "language", "lang"),
                published_date=_first(r, "date", "published_date", "created_at"),
                reviewer=_first(r, "reviewer", "author", "author_name", "name"),
                owner_response=_first(r, "owner_response", "owner_response_text", "reply"),
                url=_first(r, "url", "review_url", "permalink"),
                original_text=_first(r, "original_text"),
                translated=_first(r, "translated")))
    return out


def _norm_text(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def dedup(conn, min_len=30, threshold=0.9):
    """Drop duplicate reviews. Exact normalized matches are always dropped; fuzzy
    token-set matches are dropped at the same threshold. Idempotent."""
    rows = conn.execute("SELECT review_id, text FROM reviews ORDER BY rowid").fetchall()
    kept, drop = [], []
    for rid, text in rows:
        k = _norm_text(text)
        if len(k) < min_len:
            continue
        toks = set(k.split())
        dup = False
        for nk, nt in kept:
            if nk == k:
                dup = True
                break
            inter = len(toks & nt)
            union = len(toks | nt)
            if union and inter / union >= threshold:
                dup = True
                break
            mn = min(len(toks), len(nt))
            if mn >= 15 and inter / mn >= threshold and 0.8 <= len(toks) / len(nt) <= 1.25:
                dup = True
                break
        if dup:
            drop.append(rid)
        else:
            kept.append((k, toks))
    conn.executemany("DELETE FROM reviews WHERE review_id=?", [(r,) for r in drop])
    if drop:
        try:
            conn.executemany("DELETE FROM mentions WHERE review_id=?", [(r,) for r in drop])
        except sqlite3.OperationalError:
            pass
    conn.commit()
    return len(drop)


def upsert(conn, rows):
    cols = ["review_id", "rating", "bucket", "text", "lang",
            "published_date", "reviewer", "owner_response",
            "url", "original_text", "translated"]
    sql = f"INSERT OR REPLACE INTO reviews ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
    conn.executemany(sql, [[r[c] for c in cols] for r in rows])
    conn.commit()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_CSV)
    ap.add_argument("--db", default=DB)
    a = ap.parse_args(argv)

    os.makedirs(os.path.dirname(a.db) or ".", exist_ok=True)
    conn = sqlite3.connect(a.db)
    init(conn)
    rows = read_csv(a.input)
    upsert(conn, rows)
    removed = dedup(conn)
    print(f"{os.path.basename(a.input)}: {len(rows)} rows")
    print(f"deduped duplicates: {removed}")
    print(f"total in db: {conn.execute('SELECT COUNT(*) FROM reviews').fetchone()[0]}")
    conn.close()


if __name__ == "__main__":
    main()
