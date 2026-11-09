"""S3: what spells do in play (FIREBALL), always within tier and spell level.

damage_per_cast is a 5% trimmed mean: some casts sum many sub-1,000 hits into impossible totals
(homebrew or jokes), so the raw mean is kept only for reference.

Usage: python s3_effect.py <repo root> <out_dir>
Writes s3_effect.csv (spell x tier metrics), s3_effect_vs_use.csv (Spearman per level x tier group)
and s3_source_check.csv (leave-largest-source-out sensitivity).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MIN_CASTS = 30


def metrics(c: pd.DataFrame) -> pd.DataFrame:
    g = c.groupby(["spell", "tier"])
    m = pd.DataFrame({
        "casts": g.size(),
        "sources": g.source_file.nunique(),
        "top_source_share": g.source_file.agg(lambda s: s.value_counts(normalize=True).iloc[0]),
        "damage_per_cast": g.damage.apply(lambda x: stats.trim_mean(x, 0.05)),
        "damage_mean_raw": g.damage.mean(),
        "damage_median": g.damage.median(),
        "targets_per_cast": g.n_targets.mean(),
        "damage_per_target": g.apply(lambda x: x.damage.sum() / x.n_targets.clip(lower=1).sum()),
        "save_fail_rate": g.apply(lambda x: x.saves_failed.sum() / x.saves.sum() if x.saves.sum() else np.nan),
        "hit_rate": g.apply(lambda x: x.hits.sum() / x.attacks.sum() if x.attacks.sum() else np.nan),
        "healing_per_cast": g.healing.mean(),
    })
    return m.reset_index()


def main(root, out_dir):
    sys.path.insert(0, str(Path(__file__).parent))
    from attrs import load_attrs
    out = Path(out_dir)
    C = pd.read_csv(out / "casts.csv")
    attrs = load_attrs(root).set_index("spell")
    for col in ["damage", "healing", "n_targets", "saves", "saves_failed", "attacks", "hits"]:
        C[col] = pd.to_numeric(C[col], errors="coerce").fillna(0)
    m = metrics(C)
    m = m.join(attrs[["level", "damaging", "heals", "source"]], on="spell")
    m["reported"] = m.casts >= MIN_CASTS
    # use share within tier = casts of the spell / all spell casts in that tier
    tier_tot = C.groupby("tier").size()
    m["use_share"] = m.casts / m.tier.map(tier_tot)
    m.to_csv(out / "s3_effect.csv", index=False)

    # effect vs popularity: damaging spells, same level and tier, >= MIN_CASTS, groups of >= 5 spells
    d = m[m.reported & (m.damage_per_cast > 0) & m.level.notna()]
    rows = []
    for (lv, t), g in d.groupby(["level", "tier"]):
        if len(g) < 5:
            continue
        rho, p = stats.spearmanr(g.damage_per_cast, g.casts)
        z_eff = (g.damage_per_cast - g.damage_per_cast.mean()) / g.damage_per_cast.std()
        z_use = (np.log(g.casts) - np.log(g.casts).mean()) / np.log(g.casts).std()
        gap = (z_eff - z_use)
        rows.append(dict(level=lv, tier=t, n_spells=len(g), spearman_rho=rho, p=p,
                         underused=g.spell.loc[gap.idxmax()], overused=g.spell.loc[gap.idxmin()]))
    ev = pd.DataFrame(rows)
    if len(ev):
        ev["p_holm"] = __import__("statsmodels.stats.multitest", fromlist=["x"]).multipletests(ev.p, method="holm")[1]
    ev.to_csv(out / "s3_effect_vs_use.csv", index=False)

    # sensitivity: does any single Discord source drive a spell's numbers?
    big = m[m.reported].sort_values("casts", ascending=False).head(25)
    chk = []
    for r in big.itertuples():
        sub = C[(C.spell == r.spell) & (C.tier == r.tier)]
        top = sub.source_file.value_counts().index[0]
        rest = sub[sub.source_file != top]
        chk.append(dict(spell=r.spell, tier=r.tier, casts=r.casts, top_source_share=r.top_source_share,
                        damage_per_cast=r.damage_per_cast,
                        damage_per_cast_without_top=stats.trim_mean(rest.damage, 0.05) if len(rest) else np.nan))
    pd.DataFrame(chk).to_csv(out / "s3_source_check.csv", index=False)

    pd.set_option("display.width", 220)
    show = m[m.reported & m.level.isin([0, 1, 3])].sort_values(["level", "tier", "damage_per_cast"], ascending=[True, True, False])
    for (lv, t), g in show.groupby(["level", "tier"]):
        if t in ["1-4", "5-10"]:
            print(f"\nlevel {int(lv)} tier {t}:")
            print(g[["spell", "casts", "damage_per_cast", "damage_per_target", "targets_per_cast", "save_fail_rate",
                     "hit_rate", "healing_per_cast"]].head(8).round(2).to_string(index=False))
    print(ev.round(3).to_string(index=False))
    print(pd.DataFrame(chk).round(2).head(12).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
