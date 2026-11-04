"""Shared helpers for the training scripts.

Data loading, threshold tuning and metrics, used by ``train_cat.py`` and
``fit_probe.py``. Inference comes from the runtime classifier module, so the
trainers and the dashboard score labels the same way.
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # repo root
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "app", "pipeline"))
from catmodel import predict  # noqa: E402  (shared inference, runtime module)

GRID = [round(0.30 + 0.05 * i, 2) for i in range(13)]  # 0.30 .. 0.90 (floor = discipline)


def load_examples(path, lab2i):
    """One example per (review, sentence); labels = set of category|sentiment."""
    bucket = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        for m in r["mentions"]:
            sent = m["snippet"].strip()
            if not sent:
                continue
            lab = f"{m['category']}|{m['sentiment']}"
            ctx = str(m.get("context", "")).strip()
            if lab in lab2i:
                bucket.setdefault((r["id"], sent), [set(), ctx])[0].add(lab)
            elif m.get("category") == "__NONE__":
                bucket.setdefault((r["id"], sent), [set(), ctx])
    texts, labels = [], []
    for (_rid, sent), (labs, ctx) in bucket.items():
        texts.append((ctx + " " + sent).strip() if ctx else sent)
        y = [0.0] * len(lab2i)
        for l in labs:
            y[lab2i[l]] = 1.0
        labels.append(y)
    return texts, labels


def prf(y_true, y_pred):
    tp = int((y_true * y_pred).sum())
    fp = int((y_pred * (1 - y_true)).sum())
    fn = int(((1 - y_pred) * y_true).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def apply_thresholds(probs, th):
    return (probs > th).float()


def tune_thresholds(probs, y, min_val=5):
    """Per-label threshold (max label F1), global fallback for thin labels."""
    glob = max(GRID, key=lambda t: prf(y, apply_thresholds(probs, t))[2])
    th = torch.full((y.shape[1],), glob)
    for j in range(y.shape[1]):
        if int(y[:, j].sum()) < min_val:
            continue
        best = (-1.0, glob)
        for t in GRID:
            f = prf(y[:, j:j + 1], apply_thresholds(probs[:, j:j + 1], t))[2]
            if f > best[0]:
                best = (f, t)
        th[j] = best[1]
    return th, glob
