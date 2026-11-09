"""Topic E: most popular specific choices per class, and each class's typical build.

Usage: python e_popular.py <raw.tsv> <chars_clean.csv> <out_dir>
Writes e_choices.csv (long: class, kind, item, n, denom, share) and e_typical.csv.
"""
import sys
from pathlib import Path

import pandas as pd

ABILITIES = ["Str", "Dex", "Con", "Int", "Wis", "Cha"]


def split_col(d, col, sep="|"):
    x = d[["char_id", "main_class", col]].dropna()
    return x.assign(item=x[col].str.split(sep)).explode("item").assign(item=lambda t: t["item"].str.strip())


def tally(long, kind, denom_by_class, top=10):
    long = long[long["item"].ne("")].drop_duplicates(["char_id", "item"])
    rows = []
    for scope, g in [("All", long)] + list(long.groupby("main_class")):
        den = denom_by_class.get(scope, 0)
        vc = g["item"].value_counts().head(top)
        rows += [dict(main_class=scope, kind=kind, item=k, n=int(v), denom=int(den), share=v / den if den else None)
                 for k, v in vc.items()]
    return rows


def main(raw_p, clean_p, out_dir):
    raw = pd.read_csv(raw_p, sep="\t", low_memory=False)
    c = pd.read_csv(clean_p).merge(
        raw[["hash", "feats", "skills", "processedWeapons", "processedSpells", "choices"] + ABILITIES],
        left_on="char_id", right_on="hash")
    c = c[c.main_class != "Other"]
    allc = c.groupby("main_class").size().to_dict()
    allc["All"] = len(c)
    rows = []

    # subclass: only the main class's subclass is not recoverable for multiclass, so single-class only
    s = c[~c.is_multiclass & c.subclass.notna()].assign(item=lambda d: d.subclass.str.strip())
    den = s.groupby("main_class").size().to_dict()
    den["All"] = len(s)
    rows += tally(s[["char_id", "main_class", "item"]], "subclass", den)

    for col, kind in [("feats", "feat"), ("skills", "skill"), ("processedWeapons", "weapon")]:
        rows += tally(split_col(c, col), kind, allc, top=10 if kind != "feat" else 20)
    rows += tally(c[["char_id", "main_class"]].assign(item=c.background), "background", allc)

    ch = split_col(c, "choices")
    ch = ch[ch["item"].str.contains("/")]
    ch[["ctype", "val"]] = ch["item"].str.split("/", n=1, expand=True)
    fs = ch[ch.ctype.str.lower() == "fighting style"].assign(item=lambda d: d.val)
    den = fs.groupby("main_class").char_id.nunique().to_dict()
    den["All"] = fs.char_id.nunique()
    rows += tally(fs[["char_id", "main_class", "item"]], "fighting_style", den)
    ex = ch[ch.ctype.str.lower() == "expertise"].assign(item=lambda d: d.val.str.split("*")).explode("item")
    den = ex.groupby("main_class").char_id.nunique().to_dict()
    den["All"] = ex.char_id.nunique()
    rows += tally(ex[["char_id", "main_class", "item"]], "expertise", den)

    sp = split_col(c, "processedSpells")
    sp["lvl"] = pd.to_numeric(sp["item"].str.extract(r"\*(\d+)$")[0], errors="coerce")
    sp["name"] = sp["item"].str.replace(r"\*\d*$", "", regex=True).str.strip()
    dropped = (sp["name"] == "").groupby(sp.char_id).any()
    sp = sp[sp["name"] != ""]
    for band, lo, hi in [("cantrip", 0, 0), ("spell_1_3", 1, 3), ("spell_4_9", 4, 9)]:
        b = sp[sp.lvl.between(lo, hi)]
        den = b.groupby("main_class").char_id.nunique().to_dict()
        den["All"] = b.char_id.nunique()
        rows += tally(b[["char_id", "main_class"]].assign(item=b["name"]), band, den, top=15)

    out = pd.DataFrame(rows)
    out.to_csv(Path(out_dir) / "e_choices.csv", index=False)

    # typical build per class
    typ = []
    for cls, g in c.groupby("main_class"):
        top_stat = g[ABILITIES].idxmax(axis=1).mode()[0]
        race = g.race_clean.mode()[0]
        sub = s[s.main_class == cls].item.mode()
        sub = sub[0] if len(sub) else None
        bg = g.background.mode()[0]
        sk = split_col(g, "skills")["item"].value_counts().head(3).index.tolist()
        ft = split_col(g, "feats")["item"].value_counts()
        match = g[(g.race_clean == race) & (g.background == bg) & (g.subclass.fillna("").str.strip() == sub)]
        typ.append(dict(main_class=cls, n=len(g), race=race, subclass=sub, background=bg, top_stat=top_stat,
                        skills=", ".join(sk), feat=ft.index[0] if len(ft) else None,
                        exact_race_sub_bg=len(match), exact_share=len(match) / len(g)))
    typ = pd.DataFrame(typ)
    typ.to_csv(Path(out_dir) / "e_typical.csv", index=False)
    print(f"casters with a dropped (unmapped) spell: {dropped.mean():.1%} of {dropped.size}")
    pd.set_option("display.width", 250)
    print(typ.round(3).to_string(index=False))
    for k in ["subclass", "feat", "background", "fighting_style", "expertise", "cantrip", "spell_1_3", "spell_4_9", "weapon", "skill"]:
        t = out[(out.main_class == "All") & (out.kind == k)].head(8)
        print(f"\n{k} (denom {t.denom.iloc[0] if len(t) else 0}):", {r.item: round(r.share * 100, 1) for r in t.itertuples()})


if __name__ == "__main__":
    main(*sys.argv[1:4])
