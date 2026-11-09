"""Standard spell names shared by the character table, FIREBALL and the attribute table.

Usage: python spell_names.py <repo root> <out_dir>
Writes learned.csv (one row per character x spell), casts.csv (one row per FIREBALL character spell
action) and name_match.csv (match rates and the top unmatched names).
"""
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
OFFICIAL = {"Artificer", "Barbarian", "Bard", "Cleric", "Druid", "Fighter", "Monk", "Paladin", "Ranger",
            "Rogue", "Sorcerer", "Warlock", "Wizard"}


def norm(s: str) -> str:
    s = str(s).lower().replace("’", "").replace("'", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def load_alias() -> dict:
    a = pd.read_csv(HERE / "spell_alias.csv")
    return dict(zip(a.alias, a.standard))


def standardise(names: pd.Series, known: set, alias: dict) -> pd.Series:
    """norm -> alias -> strip 'mystic arcanum' prefix and ' with <item>' suffix when that yields a known name."""
    def one(n):
        n = norm(n)
        n = alias.get(n, n)
        if n.startswith("mystic arcanum "):
            n = alias.get(n[15:], n[15:])
        if n not in known and " with " in n:
            head = n.split(" with ")[0].strip()
            head = alias.get(head, head)
            if head in known:
                n = head
        return n
    return names.map(one)


def tier(level: pd.Series) -> pd.Series:
    return pd.cut(level, [0, 4, 10, 16, 20], labels=["1-4", "5-10", "11-16", "17-20"]).astype(str)


def main(root, out_dir):
    root, out = Path(root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    alias = load_alias()
    srd = set(pd.read_csv(root / "srd_spells.csv").name.map(norm))

    # learned: character table (needs chars_clean.csv for main_class)
    raw = pd.read_csv(root / "dnd_chars_unique.tsv", sep="\t", low_memory=False)
    clean = pd.read_csv(out / "chars_clean.csv")
    sp = raw[["hash", "processedSpells"]].dropna()
    sp = sp.assign(item=sp.processedSpells.str.split("|")).explode("item")
    sp["spell_level"] = pd.to_numeric(sp["item"].str.extract(r"\*(\d+)$")[0], errors="coerce")
    sp["raw_name"] = sp["item"].str.replace(r"\*\d*$", "", regex=True).str.strip()
    dropped = sp.groupby("hash").raw_name.apply(lambda s: (s == "").any()).rename("has_dropped_spell")
    sp = sp[sp.raw_name != ""]

    a = pd.read_csv(root / "fireball_actions.csv", low_memory=False)
    fc = a[(a.verb == "spell") & (a.actor_kind == "character")].copy()
    fc_all = len(fc)
    fc = fc[fc.actor_main_class.isin(OFFICIAL)].copy()

    known = srd | set(alias.values()) | set(sp.raw_name.map(norm)) | set(fc.action_name.map(norm))
    sp["spell"] = standardise(sp.raw_name, known, alias)
    fc["spell"] = standardise(fc.action_name, known, alias)

    learned = sp.rename(columns={"hash": "char_id"}).merge(
        clean[["char_id", "main_class", "level", "is_multiclass"]], on="char_id")
    learned = learned[learned.main_class.isin(OFFICIAL)]
    learned = learned.drop_duplicates(["char_id", "spell"])
    learned["tier"] = tier(learned.level)
    learned = learned.merge(dropped, left_on="char_id", right_index=True, how="left")
    learned[["char_id", "main_class", "level", "tier", "is_multiclass", "spell", "spell_level", "raw_name",
             "has_dropped_spell"]].to_csv(out / "learned.csv", index=False)

    fc["tier"] = tier(fc.actor_level)
    fc["source_file"] = fc.session_id.str.split("-").str[0]
    keep = ["session_id", "source_file", "turn", "action_no", "actor_name", "actor_main_class", "actor_level", "tier",
            "spell", "action_name", "n_targets", "attacks", "hits", "crits", "misses", "saves", "saves_failed",
            "save_ability", "damage", "healing"]
    fc[keep].to_csv(out / "casts.csv", index=False)

    # match report
    L, F = learned.spell, fc.spell
    rows = [
        dict(metric="fireball character spell casts (all classes)", value=fc_all),
        dict(metric="share by official classes", value=len(fc) / fc_all),
        dict(metric="learned picks", value=len(L)),
        dict(metric="distinct names learned (raw norm)", value=sp.raw_name.map(norm).nunique()),
        dict(metric="distinct names learned (standard)", value=L.nunique()),
        dict(metric="distinct names cast (raw norm)", value=fc.action_name.map(norm).nunique()),
        dict(metric="distinct names cast (standard)", value=F.nunique()),
        dict(metric="share of casts whose spell is also learned somewhere", value=F.isin(set(L)).mean()),
        dict(metric="share of learned picks whose spell is also cast somewhere", value=L.isin(set(F)).mean()),
        dict(metric="share of casts that are SRD spells", value=F.isin(srd).mean()),
        dict(metric="share of learned picks that are SRD spells", value=L.isin(srd).mean()),
        dict(metric="share of casters with a dropped spell (upstream)", value=dropped.mean()),
    ]
    unmatched = F[~F.isin(set(L))].value_counts().head(20)
    rows += [dict(metric=f"top unmatched cast: {k}", value=v) for k, v in unmatched.items()]
    rep = pd.DataFrame(rows)
    rep.to_csv(out / "name_match.csv", index=False)
    print(rep.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
