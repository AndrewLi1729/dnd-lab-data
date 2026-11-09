"""S1: which spells are learned, and which spell attributes go with being learned more.

Usage: python s1_learn.py <repo root> <out_dir> [--srd-only]
Writes s1_rates.csv (class x spell learning rates), s1_overall.csv (spell-level rates) and s1_model.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def spell_levels(learned: pd.DataFrame, attrs: pd.DataFrame) -> pd.Series:
    lv = attrs.set_index("spell").level.dropna()
    obs = learned.groupby("spell").spell_level.agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    return lv.combine_first(obs)


def eligibility(learned: pd.DataFrame, level: pd.Series) -> pd.DataFrame:
    """Casters per class able to cast each spell level (their list holds a spell of that level or higher)."""
    L = learned.assign(lv=learned.spell.map(level).fillna(learned.spell_level))
    top = L.groupby(["char_id", "main_class"]).lv.max().reset_index()
    return pd.DataFrame({lv: top[top.lv >= lv].groupby("main_class").size() for lv in range(10)}).fillna(0)


def class_rates(learned: pd.DataFrame, level: pd.Series, elig: pd.DataFrame) -> pd.DataFrame:
    """Learning rate per class x spell among casters of that class who can cast that spell level."""
    r = learned.groupby(["main_class", "spell"]).char_id.nunique().rename("learners").reset_index()
    r["level"] = r.spell.map(level)
    r = r.dropna(subset=["level"])
    r["eligible"] = [elig.at[c, int(lv)] for c, lv in zip(r.main_class, r.level)]
    r["rate"] = r.learners / r.eligible
    return r


def overall(rates: pd.DataFrame, attrs: pd.DataFrame, elig: pd.DataFrame) -> pd.DataFrame:
    """Pool over the classes whose list has the spell (off-list learners are excluded)."""
    cl = attrs.set_index("spell").classes.dropna().str.split(";")
    on = rates[[c in cl.get(s, []) for c, s in zip(rates.main_class, rates.spell)]]
    # eligible counts must include classes on the list that never learned the spell
    o = on.groupby("spell").agg(learners=("learners", "sum"), level=("level", "first"))
    o["eligible"] = [elig[int(row.level)].reindex(cl.get(sp, [])).fillna(0).sum() for sp, row in o.iterrows()]
    o = o[o.eligible > 0]
    o["rate"] = o.learners / o.eligible
    return o.reset_index()


def model(o: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    d = o.merge(attrs, on="spell", suffixes=("", "_a"))
    d = d[(d.eligible >= 50) & ~d.homebrew].copy()
    d["y"] = np.log((d.learners + 0.5) / (d.eligible - d.learners + 0.5))
    d["school"] = d.school.fillna("unknown")
    d["new_book"] = ~d.source.isin(["SRD", "PHB"])
    for b in ["concentration", "ritual", "heals", "damaging", "needs_save", "attack_roll"]:
        d[b] = d[b].fillna(False).astype(int)
    d["new_book"] = d.new_book.astype(int)
    f = ("y ~ C(level) + C(school, Treatment('evocation')) + concentration + ritual + heals + damaging"
         " + needs_save + attack_roll" + (" + new_book" if d.new_book.nunique() > 1 else ""))
    fit = smf.ols(f, d).fit(cov_type="HC3")
    t = pd.DataFrame({"term": fit.params.index, "coef": fit.params.values, "se": fit.bse.values,
                      "p": fit.pvalues.values})
    t = t[~t.term.str.startswith(("Intercept", "C(level)"))].copy()
    t["p_holm"] = multipletests(t.p, method="holm")[1]
    t["odds_ratio"] = np.exp(t.coef)
    t["n_spells"] = len(d)
    t["r2"] = fit.rsquared
    return t


def main(root, out_dir, srd_only=False):
    sys.path.insert(0, str(Path(__file__).parent))
    from attrs import load_attrs
    out = Path(out_dir)
    learned = pd.read_csv(out / "learned.csv")
    attrs = load_attrs(root)
    if srd_only:
        attrs = attrs[attrs.source == "SRD"]
    level = spell_levels(learned, attrs)
    elig = eligibility(learned, level)
    r = class_rates(learned, level, elig)
    o = overall(r, attrs, elig)
    m = model(o, attrs)
    tag = "_srd" if srd_only else ""
    if not srd_only:
        r.to_csv(out / "s1_rates.csv", index=False)
        o.to_csv(out / "s1_overall.csv", index=False)
    m.to_csv(out / f"s1_model{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    if not srd_only:
        top = o[o.eligible >= 100].sort_values("rate", ascending=False)
        for lv in [0, 1, 2, 3]:
            t = top[top.level == lv].head(6)
            print(f"level {lv}:", {s: f"{x:.0%}" for s, x in zip(t.spell, t.rate)})
    print(m.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--srd-only" in sys.argv)
