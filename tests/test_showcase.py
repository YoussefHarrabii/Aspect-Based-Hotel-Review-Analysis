import json
from pathlib import Path


def test_showcase_cache_shape():
    root = Path(__file__).resolve().parents[1]
    data = json.load(open(root / "docs" / "data" / "showcase.json", encoding="utf-8"))
    assert len(data["reviews"]) == 10
    assert data["hotel"]["totals"]["reviews"] == 10
    assert len(data["mentions"]) > 0
    assert all(len(m) == 8 for m in data["mentions"])
    assert {m[0] for m in data["mentions"]}
    ids = {r["id"] for r in data["reviews"]}
    assert all(m[7] in ids for m in data["mentions"])
    assert all("source" not in r for r in data["reviews"])


def test_showcase_omits_location_and_general_experience():
    root = Path(__file__).resolve().parents[1]
    data = json.load(open(root / "docs" / "data" / "showcase.json", encoding="utf-8"))
    assert all(m[0] != "Location" for m in data["mentions"])
    assert all((m[2] or "") != "General Experience" for m in data["mentions"])
