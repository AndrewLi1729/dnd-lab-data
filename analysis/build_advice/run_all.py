"""Rebuild the build-advice tables from the raw repo files.

Usage (from the repo root):
    python analysis/build_advice/run_all.py . out_build/
Runs the upstream steps it needs from analysis/spells and analysis/combat, then ba.py. About 6 minutes.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SPELLS, COMBAT = HERE.parent / "spells", HERE.parent / "combat"


def run(*args):
    print(f"\n$ {' '.join(map(str, args))}", flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True)


def main(root, out):
    root, out = Path(root), Path(out)
    s, c = out / "spells", out / "combat"
    for d in (out, s, c):
        d.mkdir(parents=True, exist_ok=True)
    run(root / "clean_chars.py", root / "dnd_chars_unique.tsv", s)
    run(SPELLS / "spell_names.py", root, s)
    run(SPELLS / "attrs.py", root, s)
    run(COMBAT / "prep.py", root, c)
    run(COMBAT / "c2_danger.py", root, c)
    run(HERE / "ba.py", c, s, out)
    print("\nAll tables written to", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
