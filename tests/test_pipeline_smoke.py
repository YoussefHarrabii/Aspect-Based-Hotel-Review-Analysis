"""CSV -> normalize -> segment -> classifier -> mentions -> aggregate -> data.json.

Runs with no network and no downloaded weights: the classifier, e5 encoder and
sentiment probe are replaced with deterministic local fakes. The sample CSV is
generated from templates, so no real review text is used.
"""
import json
import re
import sqlite3
import sys

import numpy as np
import pytest
import torch

import aggregate
import build_data
import cascade_extract
import make_sample_data
import normalize

KEYWORDS = {
    "Room & Property": ["room", "bathroom", "bed", "shower", "balcony", "chambre", "salle de bain"],
    "Food & Drink": ["food", "breakfast", "buffet", "dinner", "restaurant", "repas", "déjeuner", "dîner"],
    "Staff & Service": ["staff", "reception", "waiter", "service", "personnel", "réception"],
    "Location": ["beach", "centre", "taxi", "plage", "ville"],
    "Pool & Beach": ["pool", "beach", "piscine", "plage"],
    "Spa & Wellness": ["spa", "massage", "sauna", "soin"],
    "Entertainment & Animation": ["show", "animation", "kids", "spectacle", "musique"],
}
NEG = {"bad", "rude", "dirty", "cold", "noisy", "slow", "crowded", "disappointed", "problem",
       "broken", "leak", "stain", "bland", "overpriced", "far", "mauvais", "sale", "froid",
       "bruyant", "lent", "déçu", "problème", "cassé", "fuite", "tache", "loin"}


def fake_units(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def fake_model_probs(source, texts, labels=None, dev=None, max_length=128):
    rows = []
    for text in texts:
        low = text.lower()
        rows.append([0.9 if any(k in low for k in KEYWORDS.get(cat, [])) else 0.05 for cat in labels])
    return torch.tensor(rows, dtype=torch.float32)


def fake_encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False):
    out = []
    for text in texts:
        low = text.lower()
        v = np.zeros(8, dtype=np.float32)
        v[0] = 3.0 if any(w in low for w in NEG) else -3.0
        v[1] = (len(low) % 7) / 7.0
        v[2] = 1.0
        v /= np.linalg.norm(v)
        out.append(v)
    return np.stack(out)


class FakeST:
    def __init__(self, *a, **k):
        pass

    def encode(self, texts, **k):
        return fake_encode(texts)


class FakeProbe:
    classes_ = np.array(["negative", "positive"])

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-(X[:, 0] * 4.0)))
        return np.stack([1 - p, p], axis=1)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    db = tmp_path / "reviews.sqlite"
    csv_path = tmp_path / "reviews.csv"
    make_sample_data.write_csv(str(csv_path), rows=400, seed=20261104)

    for mod in (normalize, cascade_extract, aggregate, build_data):
        monkeypatch.setattr(mod, "DB", str(db))
    monkeypatch.setattr(aggregate, "STATS", str(tmp_path / "stats.json"))
    monkeypatch.setattr(build_data, "STATS", str(tmp_path / "stats.json"))
    monkeypatch.setattr(build_data, "OUT", str(tmp_path / "data.json"))
    monkeypatch.setattr(build_data, "RUN", "test")

    monkeypatch.setattr(cascade_extract.restore_seg, "units", fake_units)
    monkeypatch.setattr(cascade_extract, "_thresholds", lambda source: ([0.5] * len(cascade_extract.CATS), "test"))
    monkeypatch.setattr(cascade_extract, "SentenceTransformer", FakeST)
    monkeypatch.setattr(cascade_extract.joblib, "load", lambda path: FakeProbe())
    monkeypatch.setattr(cascade_extract.M, "model_probs", fake_model_probs)
    return {"db": db, "csv": csv_path, "tmp": tmp_path}


def test_pipeline_smoke(workspace, capsys):
    db = workspace["db"]
    normalize.main(["--input", str(workspace["csv"]), "--db", str(db)])
    out = capsys.readouterr().out
    assert "400 rows" in out

    conn = sqlite3.connect(db)
    reviews = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    conn.close()
    print(f"reviews after normalize: {reviews}")
    assert reviews >= 350

    cascade_extract.main(["--rebuild", "--key", "test"])
    conn = sqlite3.connect(db)
    mentions = conn.execute("SELECT COUNT(*) FROM mentions WHERE run_key='test'").fetchone()[0]
    units = len({r[0] for r in conn.execute("SELECT evidence FROM mentions WHERE run_key='test'")})
    conn.close()
    print(f"mentions: {mentions} across {units} distinct sentences")
    assert mentions > 0

    stats, key = aggregate.compute("test")
    print(f"aggregate run: {key} reviews: {stats['hotel']['totals']['reviews']}")
    assert stats["hotel"]["totals"]["reviews"] == reviews

    json.dump(stats, open(workspace["tmp"] / "stats.json", "w", encoding="utf-8"), ensure_ascii=False)
    build_data.main()
    data = json.load(open(workspace["tmp"] / "data.json", encoding="utf-8"))
    print(f"data.json: reviews={len(data['reviews'])} mentions={len(data['mentions'])}")
    assert data["hotel"]["totals"]["reviews"] == reviews
    assert len(data["reviews"]) == reviews
    assert len(data["mentions"]) > 0
    present = {m[0] for m in data["mentions"]}
    assert "Location" not in present
    for cat in ("Room & Property", "Food & Drink", "Staff & Service",
                "Pool & Beach", "Spa & Wellness", "Entertainment & Animation"):
        assert cat in present, f"missing category {cat}"


def test_pipeline_is_deterministic(workspace):
    a = make_sample_data.build_rows(rows=40, seed=11)
    b = make_sample_data.build_rows(rows=40, seed=11)
    assert a == b
    assert len(a) == 40
    assert all(2 <= len(r["text"].split(". ")) <= 6 for r in a)
