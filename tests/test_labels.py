import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_taxonomy_matches_model_config():
    model = ROOT / "models" / "cat" / "config.json"
    if not model.is_file():
        pytest.skip("local model weights are not present")
    cfg = json.load(open(model, encoding="utf-8"))
    labels = [cfg["id2label"][str(i)] for i in range(len(cfg["id2label"]))]
    tax = json.load(open(ROOT / "config" / "taxonomy_v5.json", encoding="utf-8"))["categories"]
    assert labels == tax


def test_threshold_count_matches_taxonomy():
    tax = json.load(open(ROOT / "config" / "taxonomy_v5.json", encoding="utf-8"))["categories"]
    th = json.load(open(ROOT / "config" / "thresholds.json", encoding="utf-8"))
    assert len(th) == len(tax)
