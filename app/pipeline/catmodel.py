"""Saved category-model inference (runtime, no training).

Loads one saved classifier, either from a local directory or from a Hugging Face
model id. Training/eval lives in training/.

Set ``REVIEWS_MODEL`` or pass a path / id to override the default.
"""
import json
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL = os.getenv("REVIEWS_MODEL") or "hf:<your-account>/deberta-review-categories"


def resolve(source=None):
    """Strip the ``hf:`` marker; both forms are accepted by transformers."""
    src = (source or DEFAULT_MODEL).strip()
    return src[3:] if src.startswith("hf:") else src


def labels_of(model_dir):
    id2label = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))["id2label"]
    return [id2label[str(i)] for i in range(len(id2label))]


def load(source=None, dev=None):
    src = resolve(source)
    dev = torch.device(dev or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src, dtype=dtype).to(dev).eval()
    id2label = model.config.id2label
    labels = [id2label.get(str(i), id2label.get(i)) for i in range(len(id2label))]
    return model, tok, labels, src


@torch.no_grad()
def predict(model, ids, mask, bs=64, dev=None):
    dev = torch.device(dev) if dev is not None else next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(ids), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(ids[i:i + bs].to(dev), mask[i:i + bs].to(dev)).logits
        out.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(out)


@torch.no_grad()
def model_probs(source, texts, labels=None, dev=None, max_length=128):
    """Load one model, score ``texts``, and release it again."""
    model, tok, model_labels, src = load(source, dev)
    if labels is not None and model_labels != list(labels):
        raise SystemExit(f"label mismatch: {src}")
    enc = tok(texts, truncation=True, max_length=max_length, padding="max_length",
              return_tensors="pt")
    p = predict(model, enc["input_ids"], enc["attention_mask"], dev=next(model.parameters()).device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return p

def ensemble_probs(model_dirs, texts, labels, dev):
    """Compatibility name from the source build. This repository ships one seed."""
    dirs = model_dirs if isinstance(model_dirs, (list, tuple)) else [model_dirs]
    if len(dirs) != 1:
        raise SystemExit("this build ships one seed, not an ensemble")
    return model_probs(dirs[0], texts, labels, dev=dev)
