"""Runtime cascade (NO training):

review -> segment -> categorize (one saved seed) -> e5 embeddings -> sentiment probe
-> magnet routing -> mentions.

No judge, no LLM call, no scraper. Incremental: only reviews not yet processed (or
whose text changed) are handled; already-processed reviews keep their mentions.
``--rebuild`` reprocesses everything.

    python app/pipeline/cascade_extract.py [--limit N] [--key cat] [--rebuild]
"""
import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time

import joblib
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)           # app/
ROOT = os.path.dirname(APP)           # repo root
sys.path.insert(0, ROOT)
sys.path.insert(0, APP)
sys.path.insert(0, HERE)
import restore_seg
import catmodel as M
import cat_rules
from sentence_transformers import SentenceTransformer

DATA = os.getenv("REVIEWS_DATA_DIR") or os.path.join(ROOT, "data")
DB = os.path.join(DATA, "reviews.sqlite")
ART = os.path.join(ROOT, "models", "cascade")
ART_FALLBACK = os.path.join(ROOT, "config", "cascade")
CATS = list(json.load(open(os.path.join(ROOT, "config", "taxonomy_v5.json"), encoding="utf-8"))["categories"])
E5 = "intfloat/multilingual-e5-large-instruct"
INSTR = "Represent the specific topic of this hotel review sentence for clustering."
SENT_INSTR = ("Instruct: Classify the sentiment of this hotel review sentence as positive or "
              "negative.\nQuery: ")
MIN_UNIT_WORDS = 2
BAND = 0.10

_BANK = json.load(open(os.path.join(ROOT, "config", "magnets.json"), encoding="utf-8"))
MAGNETS = _BANK["buckets"]


def _sha(text):
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def _to_process(rows, done):
    """rows = [(review_id, text)], done = {review_id: text_hash}."""
    return [r for r in rows if done.get(r[0]) != _sha(r[1])]


def _asset(name):
    for base in (ART, ART_FALLBACK):
        path = os.path.join(base, name)
        if os.path.isfile(path):
            return path
    return os.path.join(ART_FALLBACK, name)


def _thresholds(model_source):
    """User refit, then next to the model, then the shipped fallback."""
    candidates = [os.path.join(ART, "thresholds.json")]
    src = M.resolve(model_source)
    if os.path.isdir(src):
        candidates.append(os.path.join(src, "thresholds.json"))
    candidates += [os.path.join(ROOT, "config", "thresholds.json"),
                   os.path.join(ART_FALLBACK, "thresholds.json")]
    for path in candidates:
        if os.path.isfile(path):
            values = json.load(open(path, encoding="utf-8"))
            if len(values) != len(CATS):
                raise SystemExit(f"threshold count {len(values)} != {len(CATS)} labels: {path}")
            return values, path
    return [0.5] * len(CATS), "default 0.5"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

T0 = time.time()


def mark(msg):
    print(f"[{time.time() - T0:6.3f}s] {msg}", flush=True)


def _magnet_matrices(enc):
    """Embed every magnet word once, grouped by (category, bucket)."""
    all_words, slices, pos = [], {}, 0
    for cat, bank in MAGNETS.items():
        for b, words in bank.items():
            slices[(cat, b)] = (pos, pos + len(words))
            all_words += words
            pos += len(words)
    emb = enc.encode([f"Instruct: {INSTR}\nQuery: {w}" for w in all_words],
                     normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    return {k: emb[a:b] for k, (a, b) in slices.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--key", default="cat")
    ap.add_argument("--model-path", default=None,
                    help="model directory or Hugging Face id (default: models/cat, else the configured id)")
    ap.add_argument("--rebuild", action="store_true", help="reprocess every review")
    a = ap.parse_args(argv)
    local = os.path.join(ROOT, "models", "cat")
    model_source = a.model_path or (local if os.path.isdir(local) else M.DEFAULT_MODEL)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mark("start")

    conn = sqlite3.connect(DB)
    conn.execute("CREATE TABLE IF NOT EXISTS processed_reviews ("
                 "review_id TEXT, run_key TEXT, text_hash TEXT, PRIMARY KEY (review_id, run_key))")
    conn.execute("CREATE TABLE IF NOT EXISTS mentions (review_id TEXT, aspect TEXT, "
                 "sentiment TEXT, evidence TEXT, run_key TEXT, term TEXT, "
                 "score REAL, facet TEXT, facet_sim REAL, confidence REAL)")
    if a.rebuild:
        conn.execute("DELETE FROM mentions WHERE run_key=?", (a.key,))
        conn.execute("DELETE FROM processed_reviews WHERE run_key=?", (a.key,))
        conn.commit()
    done = dict(conn.execute("SELECT review_id, text_hash FROM processed_reviews WHERE run_key=?",
                             (a.key,)))
    rows = conn.execute("SELECT review_id, text FROM reviews WHERE text != '' "
                        "ORDER BY published_date DESC").fetchall()
    todo = _to_process(rows, done)
    if a.limit:
        todo = todo[:a.limit]
    mark(f"{len(rows)} reviews with text, {len(todo)} new/changed to process")
    if not todo:
        mark("nothing new; mentions unchanged")
        conn.close()
        return

    probe = joblib.load(_asset("probe.joblib"))
    poscol = list(probe.classes_).index("positive")
    th, th_path = _thresholds(model_source)
    mark(f"loaded probe + thresholds ({th_path})")

    pairs = []
    for rid, text in todo:
        for u in restore_seg.units(text):
            if len(u.split()) >= MIN_UNIT_WORDS:
                pairs.append((rid, u))
    texts = [p[1] for p in pairs]
    mark(f"segmented {len(todo)} reviews -> {len(pairs)} units")
    restore_seg._PM = None
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    probs = M.ensemble_probs([model_source], texts, CATS, dev=dev).numpy()
    pred = probs > np.asarray(th, dtype=np.float32)
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    mark("categorized")

    enc = SentenceTransformer(E5)
    Xu = enc.encode([f"Instruct: {INSTR}\nQuery: {t}" for t in texts],
                    normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    mats = _magnet_matrices(enc)
    bath = cat_rules.encode_bathroom(enc, f"Instruct: {INSTR}\nQuery: ")
    mark(f"embedded {len(pairs)} units + {sum(len(v) for v in MAGNETS.values())} magnets (e5-instruct)")
    Xp = enc.encode([SENT_INSTR + t for t in texts], normalize_embeddings=True,
                    batch_size=96, show_progress_bar=False)
    sp = probe.predict_proba(Xp)[:, poscol]
    del enc, Xp
    gc.collect()
    torch.cuda.empty_cache()
    mark("sentiment (probe) + freed e5")

    sent = np.where(sp > 0.5, "positive", "negative").astype(object)
    band = np.where(np.abs(sp - 0.5) < BAND)[0]
    sent[band] = "neutral"
    mark(f"neutral band |p-0.5| < {BAND}: {len(band)} units")

    pred = cat_rules.veto_staff(pred, texts)
    pred = cat_rules.apply_rules(pred, probs, texts)
    pred = cat_rules.water_to_room(pred, Xu, texts, bath)
    mark("category rules applied (off-prop veto + room convention + water->Room)")

    facet_of, margin_of = {}, {}
    for cat, bank in MAGNETS.items():
        ci = CATS.index(cat)
        buckets = list(bank)
        for pol in ("positive", "negative"):
            idx = [i for i in range(len(pairs)) if pred[i, ci] and sent[i] == pol]
            if not idx:
                continue
            X = Xu[idx]
            comb = np.stack([(X @ mats[(cat, b)].T).max(1) for b in buckets], 1)
            order = np.argsort(-comb, axis=1)
            for r, i in enumerate(idx):
                facet_of[(i, cat)] = buckets[int(order[r, 0])]
                margin_of[(i, cat)] = (float(comb[r, order[r, 0]] - comb[r, order[r, 1]])
                                       if len(buckets) > 1 else 0.0)

    mark(f"routed facets (magnet nearest-neighbour); "
         f"neutral = {sum(1 for s in sent if s == 'neutral')}")

    conn.execute("CREATE TABLE IF NOT EXISTS mentions (review_id TEXT, aspect TEXT, "
                 "sentiment TEXT, evidence TEXT, run_key TEXT, term TEXT, "
                 "score REAL, facet TEXT, facet_sim REAL, confidence REAL)")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(mentions)")}
    for col in ("term", "score", "facet", "facet_sim", "confidence"):
        if col not in cols:
            conn.execute(f"ALTER TABLE mentions ADD COLUMN {col} TEXT")
    rids = sorted({rid for rid, _ in todo})
    conn.executemany("DELETE FROM mentions WHERE run_key=? AND review_id=?",
                     [(a.key, r) for r in rids])
    nins = 0
    for i, (rid, u) in enumerate(pairs):
        for j, c in enumerate(CATS):
            if pred[i, j]:
                sen = sent[i]
                facet = facet_of.get((i, c), "") if sen in ("positive", "negative") else ""
                fsim = margin_of.get((i, c), 0.0)
                conn.execute("INSERT INTO mentions (review_id, aspect, sentiment, evidence, run_key, "
                             "score, facet, facet_sim, confidence) "
                             "VALUES (?,?,?,?,?,?,?,?,?)",
                             (rid, c, sen, u[:200], a.key, float(round(sp[i], 3)), facet, fsim,
                              float(probs[i, j])))
                nins += 1
    conn.executemany("INSERT OR REPLACE INTO processed_reviews (review_id, run_key, text_hash) "
                     "VALUES (?,?,?)",
                     [(rid, a.key, _sha(text)) for rid, text in todo])
    conn.commit()
    mark(f"wrote {nins} mentions -> run {a.key}")
    mark(f"DONE {time.time()-T0:.1f}s for {len(todo)} reviews "
         f"({(time.time()-T0)/max(1,len(todo)):.2f}s/review)")
    conn.close()


def _selftest():
    rows = [("a", "hello", "web"), ("b", "world", "partner"), ("c", "new", "web")]
    done = {"a": _sha("hello"), "b": _sha("OLD")}
    assert _to_process(rows, done) == [("b", "world", "partner"), ("c", "new", "web")]
    assert _to_process(rows, {r[0]: _sha(r[1]) for r in rows}) == []
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
