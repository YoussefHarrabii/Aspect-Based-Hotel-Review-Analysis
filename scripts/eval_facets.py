"""Facet-routing benchmark on a hand-labelled gold set.

The gold file and the production magnet bank are external and are not part of
this repository. Pass them explicitly:

    python scripts/eval_facets.py \
        --gold path/to/magnets_combined.jsonl \
        --magnets path/to/config/magnets.json

Gold rows: {id, sentence, category, polarity, acceptable:[bucket], origin:dev|fresh}
Protocol:
  * prediction is one bucket inside the row's category;
  * top-1 coverage = the single prediction is in `acceptable`;
  * bucket coverage = top-1 coverage conditioned on each bucket, then macro.

The output is aggregate metrics only. No sentence text is written.
"""
import argparse
import collections
import datetime as dt
import json
import os
import random
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "app", "pipeline"))
from catmodel import DEFAULT_MODEL  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

E5 = "intfloat/multilingual-e5-large-instruct"
INSTR = "Represent the specific topic of this hotel review sentence for clustering."
KNN_K = 5


def load_jsonl(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            row["acceptable"] = [b for b in row.get("acceptable", []) if b]
            if row["acceptable"]:
                rows.append(row)
    return rows


def load_magnets(path):
    data = json.load(open(path, encoding="utf-8"))
    return data["buckets"]


def score(rows, pred):
    n = len(rows)
    cover = sum(p in r["acceptable"] for r, p in zip(rows, pred)) / n if n else 0.0

    buckets = sorted({b for r in rows for b in r["acceptable"]}
                     | {p for p in pred if p})
    covs = []
    for b in buckets:
        idx = [i for i, r in enumerate(rows) if b in r["acceptable"]]
        if not idx:
            continue
        covs.append(sum(pred[i] in rows[i]["acceptable"] for i in idx) / len(idx))
    return {
        "coverage": round(cover, 4),
        "bucket_coverage": round(float(np.mean(covs)), 4) if covs else 0.0,
    }


def per_category(rows, pred):
    out = {}
    for cat in sorted({r["category"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["category"] == cat]
        out[cat] = {
            "n": len(idx),
            "coverage": round(sum(pred[i] in rows[i]["acceptable"] for i in idx) / len(idx), 4),
        }
    return out


def bucket_matrix(enc, magnets):
    mats = {}
    for cat, bank in magnets.items():
        for bucket, words in bank.items():
            v = enc.encode([f"Instruct: {INSTR}\nQuery: {w}" for w in words],
                           normalize_embeddings=True, batch_size=64, show_progress_bar=False)
            mats[(cat, bucket)] = np.asarray(v, dtype=np.float32)
    return mats


def magnet_max(rows, X, mats):
    out = []
    for r, x in zip(rows, X):
        buckets = list(mats.keys())
        buckets = [b for c, b in buckets if c == r["category"]]
        scores = [float((x @ mats[(r["category"], b)].T).max()) for b in buckets]
        out.append(buckets[int(np.argmax(scores))])
    return out


def magnet_centroid(rows, X, mats):
    cent = {}
    for (cat, bucket), v in mats.items():
        c = v.mean(0)
        cent[(cat, bucket)] = c / (np.linalg.norm(c) or 1.0)
    out = []
    for r, x in zip(rows, X):
        buckets = [b for (c, b) in cent if c == r["category"]]
        scores = [float(x @ cent[(r["category"], b)]) for b in buckets]
        out.append(buckets[int(np.argmax(scores))])
    return out


def tfidf_centroid(rows, magnets):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    corpus = [r["sentence"] for r in rows]
    words = [w for bank in magnets.values() for ws in bank.values() for w in ws]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    vec.fit(corpus + words)
    cent = {}
    for cat, bank in magnets.items():
        for bucket, ws in bank.items():
            m = vec.transform(ws).toarray()
            c = m.mean(0)
            cent[(cat, bucket)] = c / (np.linalg.norm(c) or 1.0)
    X = normalize(vec.transform(corpus))
    out = []
    for i, r in enumerate(rows):
        buckets = list(magnets[r["category"]])
        xv = np.asarray(X[i].todense()).ravel()
        scores = [float(xv @ cent[(r["category"], b)]) for b in buckets]
        out.append(buckets[int(np.argmax(scores))])
    return out


def _labels(rows, buckets):
    idx = {b: i for i, b in enumerate(buckets)}
    y = np.zeros((len(rows), len(buckets)), dtype=np.int8)
    for i, r in enumerate(rows):
        for b in r["acceptable"]:
            if b in idx:
                y[i, idx[b]] = 1
    return y


def supervised(rows, X, method):
    """Train per category on dev, predict fresh."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import LinearSVC
    dev = [i for i, r in enumerate(rows) if r.get("origin") == "dev"]
    test = [i for i, r in enumerate(rows) if r.get("origin") == "fresh"]
    pred = [None] * len(rows)
    fit_s = 0.0
    for cat in sorted({rows[i]["category"] for i in test}):
        d = [i for i in dev if rows[i]["category"] == cat]
        t = [i for i in test if rows[i]["category"] == cat]
        buckets = sorted({b for i in d for b in rows[i]["acceptable"]} |
                         {b for i in t for b in rows[i]["acceptable"]})
        y = _labels([rows[i] for i in d], buckets)
        if method == "logreg":
            model = OneVsRestClassifier(LogisticRegression(max_iter=3000, class_weight="balanced"))
        else:
            model = OneVsRestClassifier(LinearSVC(class_weight="balanced", max_iter=8000))
        t0 = time.perf_counter()
        model.fit(X[d], y)
        fit_s += time.perf_counter() - t0
        scores = model.predict_proba(X[t]) if method == "logreg" else model.decision_function(X[t])
        for row_i, s in zip(t, scores):
            pred[row_i] = buckets[int(np.argmax(s))]
    return pred, fit_s


def knn(rows, X):
    dev = [i for i, r in enumerate(rows) if r.get("origin") == "dev"]
    test = [i for i, r in enumerate(rows) if r.get("origin") == "fresh"]
    pred = [None] * len(rows)
    for cat in sorted({rows[i]["category"] for i in test}):
        d = [i for i in dev if rows[i]["category"] == cat]
        t = [i for i in test if rows[i]["category"] == cat]
        if not d:
            for i in t:
                pred[i] = rows[i]["acceptable"][0]
            continue
        sims = X[t] @ X[d].T
        for k, i in enumerate(t):
            order = np.argsort(-sims[k])[:KNN_K]
            votes = collections.defaultdict(float)
            for j in order:
                for b in rows[d[j]]["acceptable"]:
                    votes[b] += float(sims[k][j])
            pred[i] = max(votes.items(), key=lambda kv: kv[1])[0] if votes else rows[i]["acceptable"][0]
    return pred


def cluster_map(rows, X, method):
    from sklearn.cluster import AgglomerativeClustering, KMeans
    dev = [i for i, r in enumerate(rows) if r.get("origin") == "dev"]
    test = [i for i, r in enumerate(rows) if r.get("origin") == "fresh"]
    pred = [None] * len(rows)
    fit_s = 0.0
    for cat in sorted({rows[i]["category"] for i in test}):
        d = [i for i in dev if rows[i]["category"] == cat]
        t = [i for i in test if rows[i]["category"] == cat]
        buckets = sorted({b for i in d for b in rows[i]["acceptable"]})
        k = min(len(buckets), len(d))
        if method == "kmeans":
            model = KMeans(n_clusters=k, n_init=10, random_state=42)
            t0 = time.perf_counter(); labels = model.fit_predict(X[d]); fit_s += time.perf_counter() - t0
            centers = model.cluster_centers_
        else:
            model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
            t0 = time.perf_counter(); labels = model.fit_predict(X[d]); fit_s += time.perf_counter() - t0
            centers = np.stack([X[d][labels == c].mean(0) for c in range(k)])
        counts = collections.defaultdict(collections.Counter)
        for j, lab in zip(d, labels):
            for b in rows[j]["acceptable"]:
                counts[int(lab)][b] += 1
        cmap = {lab: c.most_common(1)[0][0] for lab, c in counts.items()}
        assign = np.argmax(X[t] @ centers.T, axis=1)
        for row_i, lab in zip(t, assign):
            pred[row_i] = cmap.get(int(lab), rows[row_i]["acceptable"][0])
    return pred, fit_s


def topic_map(rows, method, magnets):
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    dev = [i for i, r in enumerate(rows) if r.get("origin") == "dev"]
    test = [i for i, r in enumerate(rows) if r.get("origin") == "fresh"]
    pred = [None] * len(rows)
    fit_s = 0.0
    for cat in sorted({rows[i]["category"] for i in test}):
        d = [i for i in dev if rows[i]["category"] == cat]
        t = [i for i in test if rows[i]["category"] == cat]
        buckets = sorted({b for i in d for b in rows[i]["acceptable"]})
        k = min(len(buckets), len(d))
        vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1) if method == "lda" else TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        V = vectorizer.fit_transform([rows[i]["sentence"] for i in d])
        model = (LatentDirichletAllocation(n_components=k, random_state=42, max_iter=20)
                 if method == "lda" else NMF(n_components=k, init="nndsvda", random_state=42, max_iter=400))
        t0 = time.perf_counter(); W = model.fit_transform(V); fit_s += time.perf_counter() - t0
        counts = collections.defaultdict(collections.Counter)
        for j, row in enumerate(d):
            for comp, w in enumerate(W[j]):
                for b in rows[row]["acceptable"]:
                    counts[int(comp)][b] += float(w)
        cmap = {comp: c.most_common(1)[0][0] for comp, c in counts.items()}
        Wt = model.transform(vectorizer.transform([rows[i]["sentence"] for i in t]))
        for row_i, comp in zip(t, Wt.argmax(1)):
            pred[row_i] = cmap.get(int(comp), rows[row_i]["acceptable"][0])
    return pred, fit_s


def random_router(rows, magnets):
    rng = random.Random(42)
    return [rng.choice(list(magnets[r["category"]])) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=os.getenv("REVIEWS_FACET_GOLD"), required=False)
    ap.add_argument("--magnets", default=os.getenv("REVIEWS_PRODUCTION_MAGNETS"), required=False)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "data", "facet-eval.json"))
    ap.add_argument("--limit", type=int, default=0, help="debug only")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if not a.gold or not os.path.isfile(a.gold):
        raise SystemExit("pass --gold or REVIEWS_FACET_GOLD (external gold, not in this repo)")
    if not a.magnets or not os.path.isfile(a.magnets):
        raise SystemExit("pass --magnets or REVIEWS_PRODUCTION_MAGNETS (production bank, not in this repo)")

    rows = load_jsonl(a.gold)
    if a.limit:
        rows = rows[:a.limit]
    magnets = load_magnets(a.magnets)
    missing = {b for r in rows for b in r["acceptable"]} - {b for c in magnets.values() for b in c}
    if missing:
        raise SystemExit("gold buckets missing from magnet bank: " + ", ".join(sorted(missing)))

    dev = sum(1 for r in rows if r.get("origin") == "dev")
    fresh = sum(1 for r in rows if r.get("origin") == "fresh")
    device = "cpu" if a.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"gold={len(rows)} dev={dev} fresh={fresh} categories={len({r['category'] for r in rows})} device={device}")

    t0 = time.perf_counter()
    enc = SentenceTransformer(E5, device=device)
    texts = [f"Instruct: {INSTR}\nQuery: {r['sentence']}" for r in rows]
    X = np.asarray(enc.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False))
    embed_s = time.perf_counter() - t0
    mats = bucket_matrix(enc, magnets)

    methods, preds = [], {}
    fresh_idx = [i for i, r in enumerate(rows) if r.get("origin") == "fresh"]
    fresh_rows = [rows[i] for i in fresh_idx]
    def add(name, family, pred, fit_s=0.0, method_s=0.0):
        preds[name] = [pred[i] for i in fresh_idx]
        s = score(fresh_rows, preds[name])
        ms = method_s / max(1, len(rows)) * 1000
        methods.append({"name": name, "family": family, **s,
                        "fit_s": round(fit_s, 3), "method_ms_per_sentence": round(ms, 4)})
        print(f"{name:22s} top1={s['coverage']:.3f} bucket={s['bucket_coverage']:.3f} "
              f"fit={fit_s:.2f}s method={ms:.3f}ms")

    t = time.perf_counter(); add("magnet max (current)", "unsupervised", magnet_max(rows, X, mats), method_s=time.perf_counter()-t)
    t = time.perf_counter(); add("magnet centroid", "unsupervised", magnet_centroid(rows, X, mats), method_s=time.perf_counter()-t)
    t = time.perf_counter(); add("tf-idf centroid", "lexical", tfidf_centroid(rows, magnets), method_s=time.perf_counter()-t)
    t = time.perf_counter(); p, fit = supervised(rows, X, "logreg"); add("logistic regression", "supervised", p, fit, time.perf_counter()-t-fit)
    t = time.perf_counter(); p, fit = supervised(rows, X, "svm"); add("linear SVM", "supervised", p, fit, time.perf_counter()-t-fit)
    t = time.perf_counter(); add("kNN", "supervised", knn(rows, X), method_s=time.perf_counter()-t)
    t = time.perf_counter(); p, fit = cluster_map(rows, X, "kmeans"); add("k-means", "clustering", p, fit, time.perf_counter()-t-fit)
    t = time.perf_counter(); p, fit = cluster_map(rows, X, "agglomerative"); add("agglomerative", "clustering", p, fit, time.perf_counter()-t-fit)
    t = time.perf_counter(); p, fit = topic_map(rows, "nmf", magnets); add("NMF (tf-idf)", "topic", p, fit, time.perf_counter()-t-fit)
    t = time.perf_counter(); p, fit = topic_map(rows, "lda", magnets); add("LDA (counts)", "topic", p, fit, time.perf_counter()-t-fit)
    t = time.perf_counter(); add("random", "baseline", random_router(rows, magnets), method_s=time.perf_counter()-t)

    out = {
        "generated": dt.date.today().isoformat(),
        "gold": {"rows": len(rows), "dev": dev, "fresh": fresh,
                 "evaluated": len(fresh_rows),
                 "categories": len({r["category"] for r in rows}),
                 "buckets": len({b for r in rows for b in r["acceptable"]})},
        "protocol": "held-out fresh split; one predicted bucket per sentence; top-1 coverage counts a hit when the prediction is in the acceptable set; bucket coverage macro",
        "embedding_s": round(embed_s, 3),
        "embedding_ms_per_sentence": round(embed_s / max(1, len(rows)) * 1000, 3),
        "environment": {"device": device,
                        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "",
                        "torch": torch.__version__},
        "algorithms": methods,
        "per_category": {name: per_category(fresh_rows, p) for name, p in preds.items()},
    }

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
