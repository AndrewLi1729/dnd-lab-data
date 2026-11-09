"""Reconcile the most-learned levelled spells with D&D Beyond's published popular spells.

Source: ComicBook.com on D&D Beyond's Dev Update
https://comicbook.com/gaming/news/dungeons-and-dragons-most-popular-spells
Usage: python recon.py <out_dir>   -> writes recon.csv
"""
import sys
from pathlib import Path

import pandas as pd

DDB = {  # class -> spell(s) D&D Beyond reported as most popular for that class
    "All": ["cure wounds", "healing word"],
    "Sorcerer": ["shield", "magic missile"],
    "Wizard": ["shield", "magic missile"],
    "Warlock": ["hellish rebuke", "hex"],
    "Ranger": ["hunters mark", "cure wounds"],
    "Bard": ["healing word"],
    "Cleric": ["healing word"],
    "Druid": ["cure wounds"],
    "Paladin": ["cure wounds"],
}


def main(out_dir):
    out = Path(out_dir)
    L = pd.read_csv(out / "learned.csv")
    L = L[L.spell_level >= 1]
    rows = []
    for cls, ddb in DDB.items():
        g = L if cls == "All" else L[L.main_class == cls]
        ours = g.spell.value_counts().head(3).index.tolist()
        rows.append(dict(main_class=cls, ddb=", ".join(ddb), ours_top3=", ".join(ours),
                         ddb_first_in_our_top3=ddb[0] in ours, our_first_in_ddb=ours[0] in ddb))
    r = pd.DataFrame(rows)
    r.to_csv(out / "recon.csv", index=False)
    print(r.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
