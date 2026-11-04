"""One command to rebuild the dashboard data.

    python app/pipeline/build.py
    = cascade_extract (review -> category -> sentiment -> facets)
      -> aggregate (mentions -> stats.json)
      -> build_data (stats.json + reviews -> web/data.js)
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [["cascade_extract.py"],
         ["aggregate.py", "--run", "cat"],
         ["build_data.py"]]


def main():
    for s in STEPS:
        cmd = [sys.executable, os.path.join(HERE, s[0])] + s[1:]
        print(f"\n=== {' '.join(s)} ===", flush=True)
        r = subprocess.run(cmd)
        if r.returncode:
            sys.exit(r.returncode)
    print("\nbuilt -> app/web/data.js")


if __name__ == "__main__":
    main()
