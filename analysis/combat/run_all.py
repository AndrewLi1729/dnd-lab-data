"""Rebuild every combat-analysis table from the raw repo files.

Usage (from the repo root):
    python analysis/combat/run_all.py . out_combat/
Needs pandas, numpy, scipy, statsmodels. Takes about 4 minutes (bootstraps).
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def run(*args):
    print(f"\n$ {' '.join(map(str, args))}", flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True)


def main(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    run(root / "clean_chars.py", root / "dnd_chars_unique.tsv", out)
    run(HERE / "prep.py", root, out)
    run(HERE / "c1_strength.py", root, out, 1000)
    run(HERE / "c1_events.py", out, 1000)
    run(HERE / "c2_danger.py", root, out)
    run(HERE / "c3_multiclass.py", out, 1000)
    run(HERE / "sanity.py", out)
    print("\nAll tables written to", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
