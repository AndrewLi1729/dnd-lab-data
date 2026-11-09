"""Rebuild every spell-analysis table from the raw repo files.

Usage (from the repo root):
    python analysis/spells/run_all.py . out_spells/
Needs pandas, numpy, scipy, statsmodels, scikit-learn.
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
    run(HERE / "spell_names.py", root, out)
    run(HERE / "attrs.py", root, out)
    run(HERE / "s1_learn.py", root, out)
    run(HERE / "s1_learn.py", root, out, "--srd-only")
    run(HERE / "s2_learn_vs_use.py", root, out)
    run(HERE / "s2_learn_vs_use.py", root, out, "--drop-new")
    run(HERE / "s3_effect.py", root, out)
    run(HERE / "s4_schools.py", out, 200)
    run(HERE / "recon.py", out)
    print("\nAll tables written to", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
