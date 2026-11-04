"""Fit the sentiment probe and optional category thresholds on a local corpus.

  - sentiment probe: e5-instruct embeddings + LogisticRegression -> probe.joblib
  - category thresholds (tuned on val, optional)                 -> thresholds.json

    python training/fit_probe.py --data data/train --out models/cascade

``--data`` needs ``train.jsonl`` and (for thresholds) ``val.jsonl`` in the same
``{"id", "mentions":[{"snippet","category","sentiment"}]}`` shape as the evaluation
gold set. The corpus is not distributed with this repository.
"""
import argparse
import json
import os
import sys

import joblib
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "app", "pipeline"))
import shared as T
from catmodel import model_probs
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

D = os.path.join(ROOT, "data", "train")
OUT = os.path.join(ROOT, "models", "cascade")
CATS = list(json.load(open(os.path.join(ROOT, "config", "taxonomy_v5.json"), encoding="utf-8"))["categories"])
MODEL = os.path.join(ROOT, "models", "cat")
E5 = "intfloat/multilingual-e5-large-instruct"
SENT_INSTR = ("Instruct: Classify the sentiment of this hotel review sentence as positive or "
              "negative.\nQuery: ")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=D, help="directory with train.jsonl and optional val.jsonl")
    ap.add_argument("--out", default=OUT, help="output directory for probe.joblib")
    ap.add_argument("--model", default=MODEL, help="single classifier used to tune thresholds")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    trs, trl, seen = [], [], set()
    for line in open(os.path.join(a.data, "train.jsonl"), encoding="utf-8"):
        if not line.strip():
            continue
        for m in json.loads(line)["mentions"]:
            s = m["snippet"].strip()
            p = str(m.get("sentiment", "")).lower()
            if s and p in ("positive", "negative") and (s, p) not in seen:
                seen.add((s, p))
                trs.append(s)
                trl.append(p)
    print(f"sentiment training units: {len(trs)}")
    X = SentenceTransformer(E5).encode([SENT_INSTR + t for t in trs], normalize_embeddings=True,
                                       batch_size=96, show_progress_bar=False)
    probe = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X, trl)
    joblib.dump(probe, os.path.join(a.out, "probe.joblib"))
    print(f"probe classes {list(probe.classes_)} -> {os.path.join(a.out, 'probe.joblib')}")

    val = os.path.join(a.data, "val.jsonl")
    if not os.path.isfile(val) or not os.path.isdir(a.model):
        print("no val.jsonl or no local model: thresholds unchanged")
        return
    lab2i = {f"{c}|{s}": i for i, (c, s) in enumerate(
        [(c, s) for s in ("positive", "negative", "neutral") for c in CATS])}
    n = len(CATS)
    va_t, va_y = T.load_examples(val, lab2i)
    th, _ = T.tune_thresholds(model_probs(a.model, va_t, CATS, dev=dev),
                              torch.tensor([[max(r[i], r[n + i], r[2 * n + i]) for i in range(n)] for r in va_y]), 5)
    json.dump([float(x) for x in th], open(os.path.join(a.out, "thresholds.json"), "w"))
    print("thresholds ->", [round(float(x), 2) for x in th])


if __name__ == "__main__":
    main()
