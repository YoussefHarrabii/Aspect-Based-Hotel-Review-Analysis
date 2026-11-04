import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_facet_benchmark_cache():
    data = json.load(open(ROOT / "docs" / "data" / "facet-eval.json", encoding="utf-8"))
    assert data["gold"]["evaluated"] >= 500
    assert len(data["algorithms"]) >= 10
    current = next(a for a in data["algorithms"] if a["name"].startswith("magnet max"))
    assert current["coverage"] > 0.9
    assert '"sentence":' not in json.dumps(data)


def test_latency_cache():
    data = json.load(open(ROOT / "docs" / "data" / "latency.json", encoding="utf-8"))
    assert {"10", "100", "377"} <= set(data["sizes"])
    for size in data["sizes"].values():
        assert "import_total" in size["summary"]
        assert "api_data" in size["summary"]


def test_method_docs_exist():
    for name in ("method.html", "facets.html", "latency.html", "decisions.html"):
        assert (ROOT / "docs" / name).is_file()
    for name in ("pipeline.svg", "mention.svg", "facet-routing.svg"):
        assert (ROOT / "docs" / "assets" / "diagrams" / name).is_file()
