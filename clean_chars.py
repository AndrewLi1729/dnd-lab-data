"""Clean dnd_chars_unique.tsv for stage 4 (race, class, combos).

Usage: python3 clean_chars.py <path/to/dnd_chars_unique.tsv> <out_dir>

Outputs
- chars_clean.csv        one row per character
- char_classes_long.csv  one row per character x class (with that class's level)

Rules agreed with Andrew (9 Nov 2026)
1. Keep every character. `finger` (device hash) is the closest thing to a player;
   device_char_rank lets us cap characters per device in a sensitivity check.
2. Race: use processedRace; fix the one "Aaracokra" typo; other unmapped races stay
   missing (excluded from race analysis only). Custom stays its own group.
3. Race at family level (Elf = Wood/High/Dark...), matching D&D Beyond.
4. Classes: Revised Ranger -> Ranger; non-official classes -> Other.
5. Multiclass characters are KEPT. main_class = highest-level class
   (tie -> first listed); class_combo = classes sorted alphabetically.
6. Drop good/lawful (100% empty). Flag, don't fix, impossible stats.
"""
import re
import sys
from pathlib import Path

import pandas as pd

OFFICIAL = {
    "Artificer", "Barbarian", "Bard", "Cleric", "Druid", "Fighter", "Monk",
    "Paladin", "Ranger", "Rogue", "Sorcerer", "Warlock", "Wizard",
}
CLASS_MAP = {"Revised Ranger": "Ranger"}


def norm_class(name: str) -> str:
    name = CLASS_MAP.get(name.strip(), name.strip())
    return name if name in OFFICIAL else "Other"


def parse_classes(s: str) -> list[tuple[str, int]]:
    """'Fighter 16|Cleric 4' -> [('Fighter', 16), ('Cleric', 4)]"""
    out = []
    for part in s.split("|"):
        m = re.match(r"^(.*?)\s+(\d+)$", part.strip())
        out.append((norm_class(m.group(1)), int(m.group(2))) if m else (norm_class(part), 0))
    # merge duplicates created by mapping (e.g. two non-official -> Other)
    merged: dict[str, int] = {}
    for c, lv in out:
        merged[c] = merged.get(c, 0) + lv
    return list(merged.items())


def main(src: str, out_dir: str) -> None:
    df = pd.read_csv(src, sep="\t", low_memory=False)
    n_raw = len(df)

    # --- race
    race = df["processedRace"].copy()
    race[df["race"].str.strip().eq("Aaracokra")] = "Aarakocra"
    df["race_clean"] = race

    # --- classes
    parsed = df["class"].astype(str).apply(parse_classes)
    df["n_classes"] = parsed.apply(len)
    df["is_multiclass"] = df["n_classes"] > 1
    df["main_class"] = parsed.apply(lambda p: max(p, key=lambda x: x[1])[0])  # max keeps first on tie
    df["main_class_level"] = parsed.apply(lambda p: max(x[1] for x in p))
    df["class_combo"] = parsed.apply(lambda p: " / ".join(sorted(c for c, _ in p)))
    df["dip_levels"] = df["level"] - df["main_class_level"]

    # --- player proxy
    df["device_char_rank"] = (
        df.sort_values("date").groupby("finger", dropna=True).cumcount() + 1
    ).reindex(df.index)

    # --- flags (not fixed; irrelevant to stage 4)
    stats = ["Str", "Dex", "Con", "Int", "Wis", "Cha"]
    df["stats_suspect"] = (
        (df[stats] < 1).any(axis=1) | (df[stats] > 30).any(axis=1)
        | (df["HP"] <= 0) | (df["AC"] > 30)
    )
    df["gimmick_build"] = df["n_classes"] >= 4

    keep = [
        "hash", "finger", "date", "country", "level", "race_clean", "processedRace", "race",
        "main_class", "main_class_level", "class_combo", "n_classes", "is_multiclass",
        "dip_levels", "class", "subclass", "background", "device_char_rank",
        "stats_suspect", "gimmick_build",
    ]
    clean = df[keep].rename(columns={"hash": "char_id", "finger": "device_id", "race": "race_raw"})

    long = (
        pd.DataFrame({"char_id": df["hash"], "p": parsed, "main": df["main_class"]})
        .explode("p")
        .assign(cls=lambda d: d["p"].str[0], class_level=lambda d: d["p"].str[1])
        .assign(is_main=lambda d: d["cls"].eq(d["main"]))
        .rename(columns={"cls": "class"})[["char_id", "class", "class_level", "is_main"]]
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out / "chars_clean.csv", index=False)
    long.to_csv(out / "char_classes_long.csv", index=False)

    # --- reconciliation report
    print(f"rows in: {n_raw}  rows out: {len(clean)}  long rows: {len(long)}")
    print(f"race missing: {clean.race_clean.isna().sum()}  races: {clean.race_clean.nunique()}")
    print(f"multiclass: {clean.is_multiclass.sum()} ({clean.is_multiclass.mean():.1%})")
    print(f"main_class counts:\n{clean.main_class.value_counts().to_string()}")
    print(f"level check (sum of class levels == level): "
          f"{(long.groupby('char_id').class_level.sum().reindex(clean.char_id).values == clean.level.values).all()}")
    print(f"stats_suspect: {clean.stats_suspect.sum()}  gimmick_build: {clean.gimmick_build.sum()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
