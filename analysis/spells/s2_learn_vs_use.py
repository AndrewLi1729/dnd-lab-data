"""S2: learned (character table) vs used (FIREBALL), aligned by class and tier.

Usage: python s2_learn_vs_use.py <repo root> <out_dir> [--drop-new]
Writes s2_spells.csv (one row per spell: weighted learn and use rates, use index, 95% interval)
and s2_summary.csv.

Cell = class x tier. Learn rate = share of the cell's casters (character table) who list the spell.
Use rate = share of the cell's caster-sessions (FIREBALL: a character in a session with at least one
spell cast) in which the spell was cast. Both are averaged over cells with FIREBALL caster-session
weights, so the two sides describe the same class and level mix.
Uncertainty: parametric bootstrap, binomial redraw of every cell count, 1000 reps.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

NEW_SOURCES = {"TCoE", "FToD", "SCC", "EGtW"}


def main(root, out_dir, drop_new=False, reps=1000, seed=7):
    sys.path.insert(0, str(Path(__file__).parent))
    from attrs import load_attrs
    out = Path(out_dir)
    L = pd.read_csv(out / "learned.csv")
    C = pd.read_csv(out / "casts.csv")
    attrs = load_attrs(root).set_index("spell")
    keys = ["main_class", "tier"]
    C = C.rename(columns={"actor_main_class": "main_class"})

    lc = L.groupby(keys).char_id.nunique().rename("n_learn")
    ls = L.groupby(keys + ["spell"]).char_id.nunique().rename("k_learn")
    sess = C.drop_duplicates(["session_id", "actor_name"])
    uc = sess.groupby(keys).size().rename("n_use")
    us = C.drop_duplicates(["session_id", "actor_name", "spell"]).groupby(keys + ["spell"]).size().rename("k_use")

    cells = pd.concat([lc, uc], axis=1).fillna(0)
    cells = cells[(cells.n_learn >= 20) & (cells.n_use >= 20)]
    spells = sorted(set(ls.index.get_level_values("spell")) | set(us.index.get_level_values("spell")))
    if drop_new:
        new = set(attrs.index[attrs.source.isin(NEW_SOURCES)])
        spells = [s for s in spells if s not in new]

    idx = pd.MultiIndex.from_product([cells.index.get_level_values(0).unique(), cells.index.get_level_values(1).unique(), spells],
                                     names=keys + ["spell"])
    g = pd.DataFrame(index=idx).join(ls).join(us).fillna(0).reset_index()
    g = g.merge(cells.reset_index(), on=keys)
    w = g.n_use  # weights: FIREBALL caster-sessions in the cell

    wsum = w.groupby(g.spell).sum()

    def agg(kl, ku):
        d = pd.DataFrame({"spell": g.spell, "lw": kl / g.n_learn * w, "uw": ku / g.n_use * w})
        s = d.groupby("spell")[["lw", "uw"]].sum()
        return pd.DataFrame({"learn_rate": s.lw / wsum, "use_rate": s.uw / wsum})

    base = agg(g.k_learn, g.k_use)
    tot = g.groupby("spell")[["k_learn", "k_use"]].sum()
    base = base.join(tot)
    keep = (base.k_learn >= 30) & (base.k_use >= 30)
    base["log_ratio"] = np.log(base.use_rate / base.learn_rate)
    med = base.loc[keep, "log_ratio"].median()
    base["use_index"] = base.log_ratio - med

    rng = np.random.default_rng(seed)
    kept = base.index[keep]
    sims = []
    for _ in range(reps):
        kl = rng.binomial(g.n_learn.astype(int), (g.k_learn / g.n_learn).clip(0, 1))
        ku = rng.binomial(g.n_use.astype(int), (g.k_use / g.n_use).clip(0, 1))
        b = agg(pd.Series(kl, index=g.index), pd.Series(ku, index=g.index)).loc[kept]
        lr = np.log(b.use_rate / b.learn_rate)
        sims.append(lr - lr.median())
    sims = pd.concat(sims, axis=1)
    base.loc[kept, "ci_low"] = sims.quantile(0.025, axis=1)
    base.loc[kept, "ci_high"] = sims.quantile(0.975, axis=1)
    base["reported"] = keep
    base = base.join(attrs[["level", "source"]], how="left")
    rho, p = stats.spearmanr(base.loc[keep, "learn_rate"], base.loc[keep, "use_rate"])
    summ = pd.DataFrame([dict(spells_reported=int(keep.sum()), cells=len(cells), spearman_rho=rho, spearman_p=p,
                              median_log_ratio=med, drop_new=drop_new,
                              sig_more_used=int((base.ci_low > 0).sum()), sig_less_used=int((base.ci_high < 0).sum()))])
    tag = "_nonew" if drop_new else ""
    base.reset_index().to_csv(out / f"s2_spells{tag}.csv", index=False)
    summ.to_csv(out / f"s2_summary{tag}.csv", index=False)
    pd.set_option("display.width", 220)
    print(summ.round(3).to_string(index=False))
    r = base[keep].sort_values("use_index")
    cols = ["learn_rate", "use_rate", "use_index", "ci_low", "ci_high", "k_learn", "k_use", "level", "source"]
    print("least used vs learned:\n", r[cols].head(12).round(3).to_string())
    print("most used vs learned:\n", r[cols].tail(12).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--drop-new" in sys.argv)
