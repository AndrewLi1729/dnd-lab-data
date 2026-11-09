"""Reference tables: racial ability score increases (ASI) and book releases.

An ASI "option" is (fixed, floats, excl):
  fixed  dict ability -> bonus that always applies
  floats list of bonuses the player places on distinct abilities
  excl   abilities a float may not go to
"""
import re
from itertools import permutations
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ABILITIES = ["Str", "Dex", "Con", "Int", "Wis", "Cha"]
TASHA = pd.Timestamp("2020-11-17", tz="UTC")

# Feats that also raise one ability by 1 (PHB, XGtE, TCoE). Treated as one extra
# free +1 when the character has them at levels 1-3 (variant human, custom lineage).
HALF_FEATS = {
    "Actor", "Athlete", "Durable", "Heavily Armored", "Heavy Armor Master", "Keen Mind",
    "Lightly Armored", "Linguist", "Moderately Armored", "Observant", "Resilient",
    "Tavern Brawler", "Weapon Master", "Elven Accuracy", "Dragon Fear", "Dragon Hide",
    "Dwarven Fortitude", "Fade Away", "Fey Teleportation", "Flames of Phlegethos",
    "Infernal Constitution", "Orcish Fury", "Squat Nimbleness", "Second Chance",
    "Fey Touched", "Shadow Touched", "Telekinetic", "Telepathic", "Chef", "Crusher",
    "Piercer", "Slasher", "Skill Expert", "Gunner",
}


def load_asi() -> pd.DataFrame:
    t = pd.read_csv(HERE / "race_asi.csv", keep_default_na=False)

    def parse_fixed(s):
        return {k: int(v) for k, v in (p.split(":") for p in s.split(";") if p)}

    t["fixed_d"] = t["fixed"].apply(parse_fixed)
    t["floats_l"] = t["floats"].apply(lambda s: [int(x) for x in str(s).split(";") if x])
    t["excl_s"] = t["float_excl"].apply(lambda s: set(x for x in s.split(";") if x))
    return t


def match_rows(asi: pd.DataFrame, family: str, race_raw: str):
    """Return (candidate rows, guessed flag) for one character."""
    fam = asi[asi.family == family]
    raw = (race_raw or "").lower()
    hit = fam[(fam["match"] != "") & fam["match"].apply(lambda m: bool(m) and re.search(m, raw) is not None)]
    if len(hit):
        rows, guessed = hit, False
    else:
        default = fam[fam["match"] == ""]
        if len(default) and family not in {"Aasimar", "Orc", "Elf", "Dwarf", "Shifter"}:
            rows, guessed = default, False
        else:
            rows, guessed = fam, len(fam) > 1
    if family == "Human" and "Standard" in set(rows.subrace):
        rows = fam[fam.subrace.isin(["Standard", "Variant"])]  # variant human is rarely labelled
    return rows, guessed


def options(rows: pd.DataFrame, after_tasha: bool, half_feat: bool):
    """Yield (fixed, floats, excl) options. After Tasha every positive increase may move."""
    for _, r in rows.iterrows():
        fixed, floats, excl = dict(r.fixed_d), list(r.floats_l), set(r.excl_s)
        if half_feat:
            floats = floats + [1]
        yield fixed, floats, excl
        if after_tasha:
            pos = sorted([v for v in fixed.values() if v > 0] + list(r.floats_l), reverse=True)
            neg = {k: v for k, v in fixed.items() if v < 0}
            yield neg, pos + ([1] if half_feat else []), set()


def placements(fixed, floats, excl):
    """All bonus vectors (dict ability -> total bonus) from one option."""
    free = [a for a in ABILITIES if a not in excl]
    seen = set()
    for combo in permutations(free, len(floats)):
        bonus = dict.fromkeys(ABILITIES, 0)
        for a, v in fixed.items():
            bonus[a] += v
        for a, v in zip(combo, floats):
            bonus[a] += v
        key = tuple(bonus[a] for a in ABILITIES)
        if key not in seen:
            seen.add(key)
            yield key


def load_releases() -> pd.DataFrame:
    r = pd.read_csv(HERE / "releases.csv", keep_default_na=False)
    r["date"] = pd.to_datetime(r["date"], utc=True)
    return r
