"""Topic C: smaller findings (alignment, optimisation, level 20, country).

Usage: python c_misc.py <raw.tsv> <chars_clean.csv> <out_dir>
Writes c_findings.csv: one row per test with effect size and p-value.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ABILITIES = ["Str", "Dex", "Con", "Int", "Wis", "Cha"]
PRIMARY = {"Barbarian": ["Str"], "Bard": ["Cha"], "Cleric": ["Wis"], "Druid": ["Wis"],
           "Fighter": ["Str", "Dex"], "Monk": ["Dex"], "Paladin": ["Str"], "Ranger": ["Dex"],
           "Rogue": ["Dex"], "Sorcerer": ["Cha"], "Warlock": ["Cha"], "Wizard": ["Int"], "Artificer": ["Int"]}


def cramers_v(ct):
    chi2, p, _, _ = stats.chi2_contingency(ct)
    return chi2, p, float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))


def main(raw_p, clean_p, out_dir):
    raw = pd.read_csv(raw_p, sep="\t", low_memory=False)
    c = pd.read_csv(clean_p).merge(raw[["hash", "processedAlignment", "feats"] + ABILITIES],
                                   left_on="char_id", right_on="hash")
    c = c[c.main_class != "Other"]
    rows = []

    # 1 alignment
    al = c[c.processedAlignment.notna()].copy()
    al["law"], al["moral"] = al.processedAlignment.str[0], al.processedAlignment.str[1]
    rows.append(dict(topic="alignment", test="share filled", value=len(al) / len(c), n=len(c)))
    for k, v in (al.processedAlignment.value_counts(normalize=True)).items():
        rows.append(dict(topic="alignment", test=f"share {k}", value=v, n=len(al)))
    for axis in ["law", "moral"]:
        for by in ["main_class", "race_clean"]:
            d = al[al[by].isin(al[by].value_counts()[lambda s: s >= 40].index)]
            chi2, p, v = cramers_v(pd.crosstab(d[by], d[axis]))
            rows.append(dict(topic="alignment", test=f"{axis} ~ {by}", value=v, p=p, n=len(d), metric="Cramer V"))
    pal = al.law.eq("L").groupby(al.main_class).mean()
    rows.append(dict(topic="alignment", test="lawful share Paladin vs rest", value=pal["Paladin"],
                     ref=al[al.main_class != "Paladin"].law.eq("L").mean(), n=(al.main_class == "Paladin").sum()))
    ev = al.moral.eq("E").groupby(al.race_clean).agg(["mean", "size"])
    rows.append(dict(topic="alignment", test="evil share Tiefling vs rest", value=ev.loc["Tiefling", "mean"],
                     ref=al[al.race_clean != "Tiefling"].moral.eq("E").mean(), n=int(ev.loc["Tiefling", "size"])))

    # 2 optimisation: primary ability at levels 1-3, single class, plausible stats
    o = c[(c.level <= 3) & ~c.is_multiclass & ~c.stats_suspect].copy()
    o["prim"] = o.apply(lambda r: max(r[a] for a in PRIMARY[r.main_class]), axis=1)
    rows.append(dict(topic="optimisation", test="primary >= 16 at L1-3", value=(o.prim >= 16).mean(), n=len(o)))
    rows.append(dict(topic="optimisation", test="primary < 14 at L1-3", value=(o.prim < 14).mean(), n=len(o)))
    for cls, g in o.groupby("main_class"):
        rows.append(dict(topic="optimisation", test=f"primary < 14: {cls}", value=(g.prim < 14).mean(), n=len(g)))

    # 3 level-20 characters vs level 15-19
    a, b = c[c.level == 20], c[c.level.between(15, 19)]
    for lab, fa, fb in [("multiclass share", a.is_multiclass, b.is_multiclass),
                        ("has any feat", a.feats.notna(), b.feats.notna())]:
        t = [[fa.sum(), len(fa) - fa.sum()], [fb.sum(), len(fb) - fb.sum()]]
        rows.append(dict(topic="level20", test=lab, value=fa.mean(), ref=fb.mean(), p=stats.fisher_exact(t)[1],
                         n=len(a)))
    rows.append(dict(topic="level20", test="count L20 vs L19", value=len(a), ref=(c.level == 19).sum()))

    # 4 country (top 6 countries)
    top = c[c.country.isin(c.country.value_counts().head(6).index)]
    for k in ["race_clean", "main_class"]:
        ct = pd.crosstab(top.country, top[k])
        ct = ct.loc[:, ct.sum() >= 30]
        chi2, p, v = cramers_v(ct)
        rows.append(dict(topic="country", test=f"{k} ~ country", value=v, p=p, n=int(ct.values.sum()), metric="Cramer V"))
    ct = pd.crosstab(top.country, top.is_multiclass)
    chi2, p, v = cramers_v(ct)
    rows.append(dict(topic="country", test="multiclass ~ country", value=v, p=p, n=int(ct.values.sum()), metric="Cramer V"))

    out = pd.DataFrame(rows)
    out.to_csv(Path(out_dir) / "c_findings.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
