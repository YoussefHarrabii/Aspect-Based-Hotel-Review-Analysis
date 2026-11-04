"""Aggregate sentence mentions into the numbers the dashboard shows.

Priority = negative mentions weighted by rating severity and recency
(exponential decay, half-life 9 months), with evidence quotes.

    python aggregate.py            # uses the most-mentioned run, else pass --run
    python aggregate.py --run cat

Output (data/stats.json):
    { "generated": ..., "hotel": {<block>} }
where <block> carries: hotel, place, period, totals, verdict, buckets,
priorities, topics, mention_totals.
"""
import argparse
import collections
import datetime as dt
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
HOTEL_NAME = "Your hotel"
BUCKETS = [("critical", "Critical", "1\u20132\u2605"),
           ("mixed", "Mixed", "3\u2605"),
           ("praise", "Praise", "4\u20135\u2605")]
WEIGHTS = {"critical": 1.5, "mixed": 1.0, "praise": 0.3}
SEV = {1: 3.0, 2: 2.5, 3: 1.5, 4: 1.0, 5: 1.0}
HALF_LIFE_DAYS = 270
RECENT_DAYS = 90
TOP_THEMES = 6
TOP_PRIORITIES = 6
MAX_QUOTES = 3
DROP = {"Location"}


def band(r):
    if r is None:
        return "praise"
    return "critical" if r <= 2 else "mixed" if r == 3 else "praise"


def clamp3(x):
    return max(1, min(3, int(round(x))))


def parse_date(s):
    try:
        return dt.date.fromisoformat((s or "")[:10])
    except ValueError:
        return None


def decay(pdate, today):
    d = parse_date(pdate)
    if not d:
        return 0.5
    return 0.5 ** (max(0, (today - d).days) / HALF_LIFE_DAYS)


def pick_quotes(items, by_rating=False, by_sim=False, limit=MAX_QUOTES):
    if by_sim:
        key = lambda m: (-m.get("sim", 0.0), m["date"] or "")
    elif by_rating:
        key = lambda m: (-(m["rating"] or 0), m["date"] or "")
    else:
        key = lambda m: (-m["severity"], m["date"] or "")
    out, seen = [], set()
    for m in sorted(items, key=key):
        k = m["text"].strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append({"text": m["text"], "rating": m["rating"],
                    "date": m["date"], "id": m.get("id")})
        if len(out) >= limit:
            break
    return out


def _block(rows, all_ratings, per_review, today, name, place):
    """Compute the dashboard block for one scope (a hotel, or the whole set).

    rows        : (review_id, rating, published_date, text) for reviews WITH text
    all_ratings : every review's rating in scope (includes rating-only reviews)
    per_review  : {review_id: [(aspect, sentiment, evidence, facet, facet_sim)]}
    """
    meta = {rid: (rating, pd) for rid, rating, pd, _ in rows}
    total_all = len(all_ratings)
    with_text = len(rows)
    rating_only = total_all - with_text
    n_by_bucket = collections.Counter(band(r) for r in all_ratings)
    ratings = [r for r in all_ratings if r]
    avg_all = round(sum(ratings) / len(ratings), 2) if ratings else 0
    dates = [pd for _, _, pd, _ in rows if pd]

    facet_neg = collections.defaultdict(collections.Counter)
    facet_pos = collections.defaultdict(collections.Counter)

    neg, pos, neu = (collections.defaultdict(set) for _ in range(3))
    sev_by = collections.defaultdict(list)
    neg_weight = collections.defaultdict(float)
    neg_m = collections.Counter()
    neutral_m = collections.Counter()
    neg_ids = collections.defaultdict(set)
    recent_ids = collections.defaultdict(set)
    quotes = collections.defaultdict(list)
    neutral_quotes = collections.defaultdict(list)
    facet_quotes = collections.defaultdict(list)

    for rid, (rating, pd) in meta.items():
        b = band(rating)
        seen = set()
        for aspect, sen, ev, facet, fsim in per_review.get(rid, []):
            theme = facet or "Other"
            if sen == "negative":
                neg[(b, aspect)].add(rid)
                neg_m[aspect] += 1
                facet_neg[aspect][theme] += 1
                sev_by[(b, aspect)].append(SEV.get(rating, 1.0))
                neg_ids[aspect].add(rid)
                if aspect not in seen:
                    neg_weight[aspect] += WEIGHTS.get(b, 1.0) * SEV.get(rating, 1.0) * decay(pd, today)
                    seen.add(aspect)
                d = parse_date(pd)
                if d and (today - d).days <= RECENT_DAYS:
                    recent_ids[aspect].add(rid)
                if ev:
                    q = {"text": ev, "severity": SEV.get(rating, 1.0), "sim": fsim,
                         "rating": rating, "date": pd, "id": rid}
                    quotes[aspect].append(q)
                    facet_quotes[(aspect, "negative", theme)].append(q)
            elif sen == "positive":
                pos[(b, aspect)].add(rid)
                facet_pos[aspect][theme] += 1
                if ev:
                    facet_quotes[(aspect, "positive", theme)].append(
                        {"text": ev, "severity": SEV.get(rating, 1.0), "sim": fsim,
                         "rating": rating, "date": pd, "id": rid})
            else:
                neu[(b, aspect)].add(rid)
                neutral_m[aspect] += 1
                if ev:
                    neutral_quotes[aspect].append(
                        {"text": ev, "severity": SEV.get(rating, 1.0), "sim": fsim,
                         "rating": rating, "date": pd, "id": rid})

    buckets_out = []
    for k, bname, stars in BUCKETS:
        n = n_by_bucket.get(k, 0)
        polarity = {"critical": "negative", "mixed": "neutral", "praise": "positive"}[k]
        topics = {t for (bb, t) in list(neg) + list(pos) + list(neu) if bb == k}
        scored = []
        for t in topics:
            if k == "critical":
                count = len(neg[(k, t)])
            elif k == "praise":
                count = len(pos[(k, t)])
            else:
                count = len(neg[(k, t)] | pos[(k, t)] | neu[(k, t)])
            if count == 0:
                continue
            share = round(count / n * 100, 1) if n else 0
            sev = clamp3(sum(sev_by[(k, t)]) / len(sev_by[(k, t)])) if sev_by[(k, t)] else clamp3(share / 20)
            scored.append((count, t, share, sev))
        scored.sort(key=lambda x: (-x[0], x[1]))
        themes = [{"topic": t, "pct": sh, "count": c, "severity": sev}
                  for c, t, sh, sev in scored[:TOP_THEMES]]
        buckets_out.append({"key": k, "name": bname, "stars": stars, "n": n,
                            "polarity": polarity, "themes": themes})

    neg_items = [(t, c) for t, c in neg_m.items() if t not in DROP]
    max_c = max((c for _, c in neg_items), default=1)
    prios = []
    for t, cnt in sorted(neg_items, key=lambda kv: kv[1], reverse=True)[:TOP_PRIORITIES]:
        picked = pick_quotes(quotes.get(t, []))
        allsev = [s for (b, tt), vals in sev_by.items() if tt == t for s in vals]
        prios.append({"rank": 0, "topic": t, "score": round(cnt / max_c * 100),
                      "polarity": "negative", "mentions": cnt,
                      "recent": len(recent_ids[t]),
                      "avg_severity": round(sum(allsev) / len(allsev), 1) if allsev else 0,
                      "quotes": picked})
    for i, p in enumerate(prios, 1):
        p["rank"] = i

    def facet_rows(counter, aspect, polarity, by_rating):
        rows_ = [f for f in counter.most_common(8) if f[0] and f[0] != "Other"]
        return [{"theme": f, "count": n,
                 "quotes": pick_quotes(facet_quotes[(aspect, polarity, f)], by_sim=True)}
                for f, n in rows_[:6]]

    topics = []
    for c in sorted(set(facet_neg) | set(facet_pos) | set(neutral_m)):
        if c in DROP:
            continue
        topics.append({
            "topic": c,
            "neg": sum(facet_neg[c].values()),
            "pos": sum(facet_pos[c].values()),
            "neutral": neutral_m[c],
            "neutral_quotes": pick_quotes(neutral_quotes[c], limit=3),
            "complaints": facet_rows(facet_neg[c], c, "negative", False),
            "praise": facet_rows(facet_pos[c], c, "positive", True),
        })
    topics.sort(key=lambda t: -t["neg"])

    vol_total = sum(t["neg"] + t["pos"] for t in topics)
    floor = 0.10 * vol_total
    cand = [t for t in topics if (t["neg"] + t["pos"]) >= floor] or topics
    weakest = max(cand, key=lambda t: ((t["neg"] / (t["neg"] + t["pos"])) if (t["neg"] + t["pos"]) else 0.0,
                                       t["neg"])) if cand else None
    if weakest:
        wvol = weakest["neg"] + weakest["pos"]
        detail = (f"{weakest['neg'] / wvol * 100:.0f}% of {weakest['topic']} mentions are negative "
                  f"({weakest['neg']} of {wvol}).") if wvol else ""
    else:
        detail = ""
    mention_totals = {
        "negative": sum(t["neg"] for t in topics),
        "positive": sum(t["pos"] for t in topics),
        "neutral": sum(t["neutral"] for t in topics),
    }

    return {
        "hotel": name, "place": place,
        "period": (f"{min(dates)} to {max(dates)}" if dates else ""),
        "totals": {"reviews": total_all, "with_text": with_text, "rating_only": rating_only,
                   "avg": avg_all, "avg_all": avg_all,
                   "buckets": {"critical": n_by_bucket.get("critical", 0),
                               "mixed": n_by_bucket.get("mixed", 0),
                               "praise": n_by_bucket.get("praise", 0)}},
        "verdict": {"headline": (f"{weakest['topic']} is the weakest area in this range."
                                 if weakest else "No complaints detected."),
                    "detail": detail, "topic": weakest["topic"] if weakest else ""},
        "buckets": buckets_out, "priorities": prios, "topics": topics,
        "mention_totals": mention_totals,
    }


def compute(run=None):
    conn = sqlite3.connect(DB)
    today = dt.date.today()

    if run:
        key = run
    else:
        row = conn.execute("SELECT run_key FROM mentions GROUP BY run_key "
                           "ORDER BY COUNT(*) DESC LIMIT 1").fetchone()
        key = row[0] if row else None

    reviews = conn.execute("SELECT review_id, rating, published_date, text "
                           "FROM reviews WHERE text != ''").fetchall()
    all_ratings = [r[0] for r in conn.execute("SELECT rating FROM reviews")]

    per_review = collections.defaultdict(list)
    for rid, aspect, sen, ev, facet, fsim in conn.execute(
            "SELECT review_id, aspect, sentiment, evidence, facet, facet_sim FROM mentions "
            "WHERE run_key=?", (key,)):
        per_review[rid].append((aspect, sen, ev, facet, float(fsim or 0.0)))

    block = _block(reviews, all_ratings, per_review, today, HOTEL_NAME, "")
    out = {"generated": today.strftime("%d %b %Y"), "hotel": block}
    conn.close()
    return out, key


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    args = ap.parse_args(argv)
    out, key = compute(args.run)
    json.dump(out, open(STATS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    h = out["hotel"]
    print(f"wrote {STATS}: run={key} reviews={h['totals']['reviews']} "
          f"priorities={len(h['priorities'])} topics={len(h['topics'])}")


if __name__ == "__main__":
    main()
