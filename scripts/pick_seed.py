"""Score each saved seed on the held-out gold set and print the comparison.

The gold file is not in this repository. Point ``--gold`` (or ``REVIEWS_GOLD``)
at a local JSONL file with ``{"id", "mentions":[{"snippet","category","sentiment"}]}``.
Point ``--seeds`` (or ``REVIEWS_SEED_DIRS``, semicolon-separated) at the saved
model directories. Each seed is scored with its own ``thresholds.json``.

Protocol:
  * one example per (review id, snippet);
  * a neutral mention counts as category-present (three-way collapse);
  * six actionable categories, ``Location`` excluded from the score;
  * per-label thresholds from that seed's ``thresholds.json``.

    python scripts/pick_seed.py --seeds DIR1 DIR2 DIR3 --gold PATH
"""
import argparse
import collections
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "app", "pipeline"))
import catmodel  # noqa: E402

EXCLUDED = {"Location", "__NONE__"}


def load_gold(path):
    """{snippet_key: (text, set(labels))} for every gold sentence."""
    bucket = collections.OrderedDict()
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        for m in row.get("mentions", []):
            snippet = str(m.get("snippet", "")).strip()
            if not snippet:
                continue
            key = (str(row.get("id", "")), snippet)
            _, labels = bucket.setdefault(key, (snippet, set()))
            cat = str(m.get("category", ""))
            if cat not in EXCLUDED:
                labels.add(cat)
    return list(bucket.values())


def prf(y_true, y_pred):
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if t and p:
            tp += 1
        elif p:
            fp += 1
        elif t:
            fn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def score(seed, examples, labels, dev):
    idx = [i for i, label in enumerate(labels) if label not in EXCLUDED]
    texts = [text for text, _ in examples]
    probs = catmodel.model_probs(seed, texts, labels, dev=dev)
    th = json.load(open(os.path.join(seed, "thresholds.json"), encoding="utf-8"))
    if len(th) != len(labels):
        raise SystemExit(f"threshold count {len(th)} != labels {len(labels)} in {seed}")
    pred = (probs > torch.tensor(th)).tolist()
    truth = [[labels[i] in gold for i in idx] for _, gold in examples]
    guessed = [[row[i] for i in idx] for row in pred]

    flat_t = [x for row in truth for x in row]
    flat_p = [x for row in guessed for x in row]
    micro = prf(flat_t, flat_p)
    per = []
    for k, label in enumerate(labels[i] for i in idx):
        t = [row[k] for row in truth]
        p = [row[k] for row in guessed]
        per.append((label,) + prf(t, p))
    macro = tuple(sum(x[k] for x in per) / len(per) for k in (1, 2, 3))
    return len(examples), micro, macro, per


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", default=None,
                    help="saved model directories (or REVIEWS_SEED_DIRS)")
    ap.add_argument("--gold", default=os.getenv("REVIEWS_GOLD"),
                    help="gold JSONL file (or REVIEWS_GOLD)")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)

    seeds = a.seeds or [p for p in os.getenv("REVIEWS_SEED_DIRS", "").split(";") if p.strip()]
    if not seeds:
        raise SystemExit("no seeds: pass --seeds or set REVIEWS_SEED_DIRS")
    if not a.gold:
        raise SystemExit("no gold file: pass --gold or set REVIEWS_GOLD")
    if not os.path.isfile(a.gold):
        raise SystemExit(f"gold file not found: {a.gold}")

    examples = load_gold(a.gold)
    labels = catmodel.labels_of(seeds[0])
    dev = "cpu" if a.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"gold {len(examples)} examples | {len(labels)} labels | device {dev}")
    print(f"{'seed':>16} {'n':>5} | {'micro P':>7} {'R':>6} {'F1':>6} | "
          f"{'macro P':>7} {'R':>6} {'F1':>6}")

    rows = []
    for seed in seeds:
        n, micro, macro, per = score(seed, examples, labels, dev)
        name = os.path.basename(os.path.normpath(seed))
        rows.append((name, micro, macro, per))
        print(f"{name:>16} {n:5d} | {micro[0]:7.3f} {micro[1]:6.3f} {micro[2]:6.3f} | "
              f"{macro[0]:7.3f} {macro[1]:6.3f} {macro[2]:6.3f}")

    winner = max(rows, key=lambda r: r[2][2])
    spread = max(r[2][2] for r in rows) - min(r[2][2] for r in rows)
    print(f"\nper category, winner {winner[0]}:")
    print(f"{'category':30s} {'P':>6} {'R':>6} {'F1':>6}")
    for label, p, r, f1 in winner[3]:
        print(f"{label:30s} {p:6.3f} {r:6.3f} {f1:6.3f}")
    print(f"\nWINNER {winner[0]} | category-macro F1 {winner[2][2]:.3f} | spread {spread:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
