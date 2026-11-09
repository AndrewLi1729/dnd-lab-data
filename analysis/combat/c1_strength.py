"""C1: per-class damage, healing and control per action, within tier, with character-level bootstrap.

Usage: python c1_strength.py <repo root> <out_dir> [reps]
Writes c1_class_tier.csv (point estimates + 95% intervals), c1_vs_rest.csv (each class vs the rest of its
tier, Holm-corrected) and c1_popularity.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

TIERS = ["1-4", "5-10", "11-16", "17-20"]


def metrics(d: pd.DataFrame) -> dict:
    n = len(d)
    return {
        "actions": n,
        "dmg_trim": stats.trim_mean(d.damage, 0.05),
        "dmg_median": d.damage.median(),
        "heal_per_action": d.healing.mean(),
        "hard_per10": 10 * d.hard_t.sum() / n,
        "soft_per10": 10 * d.soft_t.sum() / n,
        "group_ctrl_per10": 10 * (d.ctrl_group == "group").sum() / n,
        "prevented_per_action": d.prevented.sum() / n,
        "buff_per10": 10 * d.buff.sum() / n,
    }


def main(root, out_dir, reps=1000, seed=3):
    out = Path(out_dir)
    c = pd.read_csv(out / "char_actions.csv", low_memory=False)
    s = pd.read_csv(out / "sessions.csv").set_index("session_id")
    c["hard_t"] = np.where(c.control == "hard", c.ctrl_targets, 0)
    c["soft_t"] = np.where(c.control == "soft", c.ctrl_targets, 0)
    c["prevented"] = c.hard_t * c.session_id.map(s.creature_dmg_per_action).fillna(0)
    rng = np.random.default_rng(seed)

    rows, boots = [], {}
    for (cls, t), g in c.groupby(["actor_main_class", "tier"]):
        chars = g.groupby("char_key")
        keys = list(chars.groups)
        if len(keys) < 20:
            continue
        idx = [chars.indices[k] for k in keys]
        est = metrics(g)
        samp = []
        for _ in range(reps):
            pick = rng.integers(0, len(keys), len(keys))
            samp.append(metrics(g.iloc[np.concatenate([idx[i] for i in pick])]))
        samp = pd.DataFrame(samp)
        boots[(cls, t)] = samp
        row = {"main_class": cls, "tier": t, "characters": len(keys), **est}
        for m in ["dmg_trim", "heal_per_action", "hard_per10", "soft_per10", "prevented_per_action"]:
            row[m + "_lo"], row[m + "_hi"] = samp[m].quantile([0.025, 0.975])
        rows.append(row)
    r = pd.DataFrame(rows)
    r.to_csv(out / "c1_class_tier.csv", index=False)

    # each class vs the action-weighted mean of the other classes in its tier (bootstrap difference)
    vs = []
    for t in TIERS:
        sub = r[r.tier == t]
        for cls in sub.main_class:
            others = [k for k in boots if k[1] == t and k[0] != cls]
            if not others:
                continue
            w = np.array([r[(r.main_class == o[0]) & (r.tier == t)].actions.iloc[0] for o in others], float)
            for m in ["dmg_trim", "hard_per10"]:
                rest = sum(boots[o][m].values * wi for o, wi in zip(others, w)) / w.sum()
                diff = boots[(cls, t)][m].values - rest
                se = diff.std(ddof=1)
                p = 2 * stats.norm.sf(abs(np.mean(diff)) / se) if se > 0 else 1.0  # normal approx, bootstrap SE
                vs.append(dict(tier=t, main_class=cls, metric=m, diff=np.median(diff),
                               lo=np.quantile(diff, 0.025), hi=np.quantile(diff, 0.975), p=p))
    vs = pd.DataFrame(vs)
    vs["p_holm"] = multipletests(vs.p, method="holm")[1]
    vs.to_csv(out / "c1_vs_rest.csv", index=False)

    # popularity (character table) vs strength (tier 5-10)
    clean = pd.read_csv(out / "chars_clean.csv")
    share = clean.main_class.value_counts(normalize=True)
    p = r[r.tier == "5-10"].set_index("main_class")
    p["char_table_share"] = share.reindex(p.index)
    pops = []
    for m in ["dmg_trim", "heal_per_action", "hard_per10", "prevented_per_action"]:
        rho, pv = stats.spearmanr(p.char_table_share, p[m])
        pops.append(dict(metric=m, spearman_rho=rho, p=pv, n_classes=len(p)))
    pd.DataFrame(pops).to_csv(out / "c1_popularity.csv", index=False)

    pd.set_option("display.width", 250)
    cols = ["main_class", "characters", "actions", "dmg_trim", "dmg_trim_lo", "dmg_trim_hi", "dmg_median",
            "heal_per_action", "hard_per10", "soft_per10", "group_ctrl_per10", "prevented_per_action", "buff_per10"]
    print(r[r.tier == "5-10"].sort_values("dmg_trim", ascending=False)[cols].round(2).to_string(index=False))
    print(vs[(vs.tier == "5-10") & (vs.p_holm < 0.05)].round(3).to_string(index=False))
    print(pd.DataFrame(pops).round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:4]))
