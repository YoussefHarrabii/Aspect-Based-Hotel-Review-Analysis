from pathlib import Path

import serve


def write(tmp_path, text):
    p = tmp_path / "reviews.csv"
    p.write_text(text, encoding="utf-8")
    return p


def test_valid_csv(tmp_path):
    p = write(tmp_path,
              "review_id,date,rating,text\n"
              "1,2026-01-01,4,The room was clean.\n"
              "2,2026-01-02,,Rating only.\n")
    assert serve.validate_csv(str(p)) == []


def test_extra_columns_are_ignored(tmp_path):
    p = write(tmp_path,
              "hotel_id,review_id,date,rating,text\n"
              "hotel-a,1,2026-01-01,4,The room was clean.\n")
    assert serve.validate_csv(str(p)) == []


def test_missing_columns(tmp_path):
    p = write(tmp_path, "hotel_id,text\nhotel-a,hello\n")
    problems = serve.validate_csv(str(p))
    assert problems and "review_id" in problems[0]


def test_bad_rows(tmp_path):
    p = write(tmp_path,
              "review_id,date,rating,text\n"
              "1,2026-13-01,4,The room was clean.\n"
              "1,2026-01-01,9,Different text.\n"
              ",2026-01-01,2,Different text again.\n")
    problems = serve.validate_csv(str(p))
    joined = "\n".join(problems)
    assert "date must be YYYY-MM-DD" in joined
    assert "rating must be an integer from 1 to 5" in joined
    assert "duplicate review_id 1" in joined
    assert "review_id is empty" in joined


def test_header_only(tmp_path):
    p = write(tmp_path, "review_id,date,rating,text\n")
    problems = serve.validate_csv(str(p))
    assert problems and "no data rows" in problems[0]
