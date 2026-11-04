import json

import pick_seed


def test_load_gold_collapses_and_excludes_location(tmp_path):
    path = tmp_path / "gold.jsonl"
    rows = [
        {"id": "a", "mentions": [
            {"snippet": "The room was clean.", "category": "Room & Property", "sentiment": "positive"},
            {"snippet": "The room was clean.", "category": "Room & Property", "sentiment": "neutral"},
            {"snippet": "The room was clean.", "category": "Location", "sentiment": "positive"},
        ]},
        {"id": "b", "mentions": [
            {"snippet": "Nothing specific.", "category": "__NONE__", "sentiment": "none"},
        ]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    examples = pick_seed.load_gold(str(path))
    assert len(examples) == 2
    assert examples[0][0] == "The room was clean."
    assert examples[0][1] == {"Room & Property"}
    assert examples[1][1] == set()


def test_prf():
    p, r, f = pick_seed.prf([1, 1, 0, 0], [1, 0, 1, 0])
    assert round(p, 3) == 0.5
    assert round(r, 3) == 0.5
    assert round(f, 3) == 0.5
