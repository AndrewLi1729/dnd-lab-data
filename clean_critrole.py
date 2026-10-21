"""Combine and clean the Critical Role Stats "All Rolls" exports into one table.

Usage:
    python clean_critrole.py <campaign 1 csv> <campaign 2 csv> <output csv>

Inputs are the "All Episodes" tabs of the All Rolls sheets by CritRoleStats:
campaign 1 (Tal'Dorei, Vox Machina) and campaign 2 (Wildemount, Mighty Nein).
"""
import re
import sys

import pandas as pd

ROLL_TYPE_FIXES = {
    "Steath": "Stealth", "Wisdom Saving": "Wisdom Save", "Dex Save": "Dexterity Save",
    "Consitution Save": "Constitution Save", "Inevstigation": "Investigation", "D20": "d20",
    "Ressurection Roll": "Resurrection Roll", "Beard Check": "Beard",
    "Alchemist Kit": "Alchemy Kit", "Alchemist Tools": "Alchemy Kit", "Alchemy": "Alchemy Kit",
    "Forgery Tools": "Forgery Kit", "Forgery": "Forgery Kit", "Percentage": "d100", "Percentile": "d100",
}
CHARACTER_FIXES = {"Tibierus": "Tiberius", "PIke": "Pike", "Willie": "Willi"}
ABILITIES = ["Strength", "Dexterity", "Constitution", "Intelligence", "Wisdom", "Charisma"]
SKILLS = ["Acrobatics", "Animal Handling", "Arcana", "Athletics", "Deception", "History", "Insight",
          "Intimidation", "Investigation", "Medicine", "Nature", "Perception", "Performance",
          "Persuasion", "Religion", "Sleight of Hand", "Stealth", "Survival"]


def roll_group(t):
    if t in ("Attack", "Spell Attack"):
        return "attack"
    if t == "Damage":
        return "damage"
    if t in ("Healing", "Hit Dice", "Hit Points"):
        return "healing"
    if t == "Initiative":
        return "initiative"
    if t.endswith(" Save") or t == "Death Save":
        return "save"
    if t in SKILLS:
        return "skill check"
    if t in ABILITIES:
        return "ability check"
    return "other"


def num(s):
    s = str(s).strip() if pd.notna(s) else ""
    return int(s) if re.fullmatch(r"-?\d+", s) else None


def clean(path, campaign, damage_col):
    d = pd.read_csv(path, dtype=str)
    out = pd.DataFrame()
    ep = d["Episode"].fillna("").str.extract(r"(\d+)$")[0]
    out["episode"] = pd.to_numeric(ep, errors="coerce").astype("Int64")
    out["campaign"] = campaign
    out["time"] = d["Time"]
    out["character"] = d["Character"].str.strip().replace(CHARACTER_FIXES)
    rt = d["Type of Roll"].fillna("Unknown").str.strip().replace(ROLL_TYPE_FIXES)
    out["roll_type"] = rt
    out["roll_group"] = rt.map(roll_group)

    totals, naturals, outcomes = [], [], []
    for tot, nat in zip(d["Total Value"], d["Natural Value"]):
        t_raw = str(tot).strip() if pd.notna(tot) else ""
        total, natural, outcome = num(t_raw), num(nat), ""
        m = re.fullmatch(r"(?:Nat|Natural )\s*(\d+)", t_raw, flags=re.I)
        if m:  # natural roll typed into the total column; the modified total wasn't recorded
            natural = natural if natural is not None else int(m[1])
            total = None
        elif t_raw.lower() in ("success", "fail"):
            outcome = t_raw.lower()
        totals.append(total)
        naturals.append(natural)
        outcomes.append(outcome)
    out["total"] = pd.array(totals, dtype="Int64")
    out["natural_raw"] = pd.array(naturals, dtype="Int64")
    valid = (out["natural_raw"] >= 1) & (out["natural_raw"] <= 20)
    out["natural_d20"] = out["natural_raw"].where(valid.fillna(False))
    out["outcome"] = outcomes
    crit = d["Crit?"].fillna("").str.strip().str.lower()
    out["crit"] = crit.isin(["y", "yes", "1"])
    out["damage_text"] = d[damage_col]
    out["damage_amount"] = pd.to_numeric(d[damage_col].fillna("").str.extract(r"^\s*(\d+)")[0],
                                         errors="coerce").astype("Int64")
    out["kills"] = pd.to_numeric(d["# Kills"].replace("Y", "1"), errors="coerce")
    notes = d["Notes"].fillna("")
    out["disregarded"] = notes.str.contains(r"disregard|discard|not used|ignored", case=False)
    out["notes"] = d["Notes"]
    return out


def main(c1, c2, dst):
    df = pd.concat([clean(c1, 1, "Damage"), clean(c2, 2, "Damage Dealt")], ignore_index=True)
    df = df[df["character"].notna() | df["episode"].notna()]
    cols = ["campaign", "episode", "time", "character", "roll_type", "roll_group", "total", "natural_d20",
            "natural_raw", "outcome", "crit", "damage_text", "damage_amount", "kills", "disregarded", "notes"]
    df[cols].to_csv(dst, index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
