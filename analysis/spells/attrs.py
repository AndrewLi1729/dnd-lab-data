"""Spell attribute table: SRD rows (verified) + spell_attrs_extra.csv (mostly from memory, flagged).

Only rule attributes are stored, never description text.
Usage: python attrs.py <repo root> <out_dir>   -> writes spell_attrs.csv and attr_coverage.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from spell_names import norm  # noqa: E402

HERE = Path(__file__).parent
COLS = ["spell", "level", "school", "classes", "damage_type", "save", "attack_type", "concentration",
        "ritual", "heals", "source", "verified", "note"]


def load_attrs(root) -> pd.DataFrame:
    s = pd.read_csv(Path(root) / "srd_spells.csv")
    srd = pd.DataFrame({
        "spell": s.name.map(norm), "level": s.level, "school": s.school.str.lower(),
        "classes": s.classes.str.replace(" ", ""), "damage_type": s.damage_type.str.lower(),
        "save": s.save, "attack_type": s.attack_type, "concentration": s.concentration,
        "ritual": s.ritual, "heals": s.heals, "source": "SRD", "verified": "yes", "note": ""})
    e = pd.read_csv(HERE / "spell_attrs_extra.csv", keep_default_na=False)
    e = e.rename(columns={"standard": "spell"})
    e["level"] = pd.to_numeric(e.level, errors="coerce")
    for b in ["concentration", "ritual", "heals"]:
        e[b] = e[b].map({"True": True, "False": False})
    for c in ["damage_type", "save", "attack_type", "school", "classes"]:
        e[c] = e[c].replace("", pd.NA)
    out = pd.concat([srd, e[COLS]], ignore_index=True).drop_duplicates("spell")
    out["homebrew"] = out.source.eq("homebrew")
    out["damaging"] = out.damage_type.notna()
    out["needs_save"] = out.save.notna()
    out["attack_roll"] = out.attack_type.notna()
    return out


def main(root, out_dir):
    out = Path(out_dir)
    a = load_attrs(root)
    a.to_csv(out / "spell_attrs.csv", index=False)
    known = set(a[~a.homebrew].spell)
    L = pd.read_csv(out / "learned.csv").spell
    F = pd.read_csv(out / "casts.csv").spell
    hb = set(a[a.homebrew].spell)
    cov = pd.DataFrame([
        dict(table="learned", rows=len(L), share_with_attrs=L.isin(known).mean(), share_homebrew=L.isin(hb).mean()),
        dict(table="casts", rows=len(F), share_with_attrs=F.isin(known).mean(), share_homebrew=F.isin(hb).mean()),
    ])
    cov.to_csv(out / "attr_coverage.csv", index=False)
    print(a.groupby(["source", "verified"]).size().to_string())
    print(cov.round(4).to_string(index=False))
    assert (cov.share_with_attrs + cov.share_homebrew >= 0.90).all(), "attribute coverage below 90%"


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
