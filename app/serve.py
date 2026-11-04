"""Local dashboard server: static UI, GET /api/data, POST /api/import.

    python app/serve.py                 # browser at http://127.0.0.1:8765
    python app/serve.py --no-browser --port 8137
    python app/serve.py --native        # pywebview window, browser fallback

The page asks for a CSV in the browser. The server writes it under data/imports,
replaces the current single-hotel dataset, runs the pipeline, and returns when the
dashboard can reload. The server exposes the UI, GET /api/data and POST /api/import.
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEB = os.path.join(HERE, "web")
PIPELINE = os.path.join(HERE, "pipeline")
DATA = os.getenv("REVIEWS_DATA_DIR") or os.path.join(ROOT, "data")
DB = os.path.join(DATA, "reviews.sqlite")
IMPORTS = os.path.join(DATA, "imports")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from pipeline import aggregate, build_data

MAX_IMPORT_BYTES = 20 * 1024 * 1024
REQUIRED_COLUMNS = {"review_id", "date", "rating", "text"}
_IMPORT_LOCK = threading.Lock()


def _pipeline_env():
    env = dict(os.environ)
    env["REVIEWS_DATA_DIR"] = DATA
    env["REVIEWS_DATA_JS"] = os.path.join(WEB, "data.js")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run_step(args, env):
    cmd = [sys.executable, os.path.join(PIPELINE, args[0])] + args[1:]
    result = subprocess.run(cmd, cwd=ROOT, env=env, text=True, encoding="utf-8",
                            errors="replace", capture_output=True)
    if result.returncode:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-6:]
        raise RuntimeError(f"{args[0]} failed: " + " | ".join(tail))
    return result.stdout


def _empty_payload():
    return {"generated": None, "hotel": None, "reviews": [], "mentions": [], "taxonomy": {}}


def validate_csv(path):
    """Return a list of human-readable problems; empty means valid."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return ["CSV is empty. Expected a header: review_id, date, rating, text."]
        header = [(h or "").strip().lower() for h in reader.fieldnames]
        missing = REQUIRED_COLUMNS - set(header)
        if missing:
            return ["CSV header is missing: " + ", ".join(sorted(missing)) +
                    ". Expected: review_id, date, rating, text."]
        rows = []
        for raw in reader:
            rows.append({(k or "").strip().lower(): (v if isinstance(v, str) else "") for k, v in raw.items()})

    if not rows:
        return ["CSV has a header but no data rows."]

    errors, seen = [], set()
    for i, row in enumerate(rows, start=2):
        rid = (row.get("review_id") or "").strip()
        date = (row.get("date") or "").strip()
        rating = (row.get("rating") or "").strip()
        if not rid:
            errors.append(f"row {i}: review_id is empty")
        elif rid in seen:
            errors.append(f"row {i}: duplicate review_id {rid}")
        else:
            seen.add(rid)
        if date:
            try:
                if len(date) != 10:
                    raise ValueError
                dt.date.fromisoformat(date)
            except ValueError:
                errors.append(f"row {i}: date must be YYYY-MM-DD")
        if rating:
            try:
                value = int(rating)
                if not 1 <= value <= 5:
                    raise ValueError
            except ValueError:
                errors.append(f"row {i}: rating must be an integer from 1 to 5")
        if len(errors) >= 10:
            errors.append("more errors follow; fix the first ten and import again")
            break
    return errors


def import_csv(raw, filename="reviews.csv"):
    """Validate, replace the local dataset, and run the full pipeline."""
    if not raw:
        raise ValueError("empty file")
    if len(raw) > MAX_IMPORT_BYTES:
        raise ValueError("file is larger than 20 MB")
    os.makedirs(IMPORTS, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", os.path.basename(filename or "reviews.csv"))
    path = os.path.join(IMPORTS, f"{stamp}-{safe}")
    with open(path, "wb") as f:
        f.write(raw)

    problems = validate_csv(path)
    if problems:
        raise ValueError("CSV format is invalid:\n- " + "\n- ".join(problems))

    if os.path.exists(DB):
        conn = sqlite3.connect(DB, timeout=20)
        for table in ("reviews", "mentions", "processed_reviews", "extracted", "aspects"):
            try:
                conn.execute(f"DELETE FROM {table}")
            except sqlite3.OperationalError:
                pass
        conn.commit()
        conn.close()
    for stale in (os.path.join(DATA, "stats.json"), os.path.join(WEB, "data.js")):
        try:
            os.remove(stale)
        except OSError:
            pass

    env = _pipeline_env()
    _run_step(["normalize.py", "--input", path, "--db", DB], env)
    _run_step(["cascade_extract.py", "--key", build_data.RUN, "--rebuild"], env)
    _run_step(["aggregate.py", "--run", build_data.RUN], env)
    _run_step(["build_data.py"], env)

    conn = sqlite3.connect(DB)
    reviews = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    mentions = conn.execute("SELECT COUNT(*) FROM mentions WHERE run_key=?", (build_data.RUN,)).fetchone()[0]
    conn.close()
    return {"reviews": reviews, "mentions": mentions}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=WEB, **k)

    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, obj, status=200):
        b = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _not_implemented(self):
        b = b"This endpoint is not part of the review dashboard.\n"
        self.send_response(501)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/api/data"):
            try:
                out, _ = aggregate.compute(build_data.RUN)
                self._json(build_data.assemble(out))
            except sqlite3.OperationalError:
                self._json(_empty_payload())
            return
        if self.path.startswith("/api/"):
            self._not_implemented()
            return
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/api/"):
            self.send_error(404)
            return
        if self.path.startswith("/api/import"):
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                self._json({"ok": False, "error": "empty upload"}, 400)
                return
            if length > MAX_IMPORT_BYTES:
                self._json({"ok": False, "error": "file is larger than 20 MB"}, 413)
                return
            raw = self.rfile.read(length)
            filename = unquote(self.headers.get("X-Filename") or "reviews.csv")
            if not _IMPORT_LOCK.acquire(blocking=False):
                self._json({"ok": False, "error": "an import is already running"}, 409)
                return
            try:
                counts = import_csv(raw, filename)
                self._json({"ok": True, **counts})
            except ValueError as e:
                self._json({"ok": False, "error": str(e)}, 400)
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 500)
            finally:
                _IMPORT_LOCK.release()
            return
        self._not_implemented()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--native", action="store_true",
                    help="open in a native window instead of the browser (needs pywebview)")
    ap.add_argument("--windowed", action="store_true",
                    help="native window as a normal window instead of maximized")
    args = ap.parse_args(argv)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Aspect-Based Hotel Review Analysis: {url}  (Ctrl+C to stop)")

    if args.native:
        try:
            import webview
        except ImportError:
            print("pywebview not installed (pip install pywebview); opening in the browser instead.")
            args.native = False

    if args.native:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        webview.create_window("Aspect-Based Hotel Review Analysis", url, maximized=not args.windowed)
        webview.start()
        return

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
