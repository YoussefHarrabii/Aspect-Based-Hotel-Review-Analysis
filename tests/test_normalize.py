import sqlite3

import normalize


def test_bucket():
    assert normalize.bucket(None) == "unknown"
    assert normalize.bucket(1) == "critical"
    assert normalize.bucket(2) == "critical"
    assert normalize.bucket(3) == "mixed"
    assert normalize.bucket(4) == "praise"
    assert normalize.bucket(5) == "praise"


def test_read_csv_and_dedup(tmp_path):
    path = tmp_path / "tiny.csv"
    path.write_text(
        "review_id,date,rating,text\n"
        "1,2025-01-01,4 stars,The room was clean and the staff were helpful.\n"
        "2,2025-01-02,2,The room was clean and the staff were helpful.\n"
        "3,2025-01-03,,A different property with a completely different review.\n",
        encoding="utf-8")
    rows = normalize.read_csv(str(path))
    assert len(rows) == 3
    assert rows[0]["rating"] == 4
    assert rows[0]["bucket"] == "praise"
    assert rows[2]["rating"] is None
    assert rows[2]["bucket"] == "unknown"

    db = tmp_path / "reviews.sqlite"
    conn = sqlite3.connect(db)
    normalize.init(conn)
    normalize.upsert(conn, rows)
    removed = normalize.dedup(conn)
    assert removed == 1
    assert conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 2
    conn.close()
