"""Rebuild every result table for the character deep dive from the raw TSV.

Usage (from the repo root):
    python analysis/char_deep_dive/run_all.py dnd_chars_unique.tsv out/
Needs pandas, numpy, scipy, statsmodels.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent


def run(*args):
    print(f"\n$ {' '.join(map(str, args))}", flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True)


def main(raw, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    clean, long = out / "chars_clean.csv", out / "char_classes_long.csv"
    run(ROOT / "clean_chars.py", raw, out)
    run(HERE / "a_rules.py", raw, clean, long, out)
    run(HERE / "b_adoption.py", raw, clean, out)
    run(HERE / "c_misc.py", raw, clean, out)
    run(HERE / "d_players.py", clean, out, 1000)
    run(HERE / "e_popular.py", raw, clean, out)
    print("\nAll tables written to", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
