"""Underrated-but-useful spells and per-level picks, judged by in-play usefulness (FIREBALL), not popularity.

Usage: python picks.py <repo root> <combat_out> <spells_out> <out_dir> [reps]
  combat_out: output of analysis/combat (char_actions.csv, actions_all.csv, sessions.csv)
  spells_out: output of analysis/spells (spell_attrs.csv, s1_overall.csv)

Categories (a spell can be in several), from how its casts actually played out:
  damage  >= 50% of casts deal damage        metric: 5% trimmed mean damage per cast
  control >= 20% of casts control a target   metric: hard targets + 0.5 x soft targets per cast
  heal    >= 50% of casts heal               metric: mean healing per cast
  buff    >= 50% of casts apply a buff       metric: allies' per-action damage change in the next round,
                                             minus the same change at every non-control, non-buff moment
                                             of the session (event study as in analysis/combat)
Relative score (damage/control/heal): metric / median metric of same-category spells in the same
level x tier cell (cells with >= 5 casts and >= 2 rated spells), averaged over tiers weighted by casts.
Buff uplift is reported as exploratory only and not used for picks (see sanity note in main).
Only spells with >= 15 casts are rated; 15-29 are flagged "small sample".
Underrated: learning rate (character table) OR share of same-level casts below the median of rated
same-level same-category spells; judged only when the level x category has >= 4 rated spells. Spells absent from the character table use the cast share only.
Writes picks_scores.csv (every rated spell x category) and picks_top.csv (top 3 per level x category).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "spells"))
sys.path.insert(0, str(HERE.parent / "combat"))
from spell_names import load_alias, norm, standardise  # noqa: E402
from c1_events import change, window_stats  # noqa: E402

MIN_CASTS, SMALL, MIN_CELL = 15, 30, 5


def buff_uplift(casts, A, S, ctrl_or_buff):
    sess = {sid: {"seq": g.seq.values, "is_char": g.is_char.values.astype(bool), "damage": g.damage.values,
                  "attacks": g.attacks.values, "hits": g.hits.values, "crits": g.crits.values,
                  "actor": g.actor_name.astype(str).values} for sid, g in A[A.session_id.isin(casts.session_id)].groupby("session_id")}
    out = []
    for r in casts.itertuples():
        d = sess[r.session_id]
        w = max(1, int(round(S.at[r.session_id, "max_combatants"])))
        ch = change(*window_stats(d, d["actor"], r.seq, w, str(r.actor_name)))
        if ch is None:
            continue
        busy = ctrl_or_buff.get(r.session_id, set())
        pl = [change(*window_stats(d, d["actor"], s0, w, str(me))) for s0, me in zip(d["seq"][d["is_char"]], d["actor"][d["is_char"]])
              if s0 not in busy]
        pl = [x for x in pl if x is not None]
        if not pl:
            continue
        out.append(dict(session_id=r.session_id, spell=r.spell, tier=r.tier,
                        uplift=ch["d_dmg"] - np.mean([x["d_dmg"] for x in pl])))
    return pd.DataFrame(out)


def main(root, combat_out, spells_out, out_dir, reps=300, seed=21):
    root, co, so, out = map(Path, (root, combat_out, spells_out, out_dir))
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    C = pd.read_csv(co / "char_actions.csv", low_memory=False)
    A = pd.read_csv(co / "actions_all.csv", low_memory=False)
    S = pd.read_csv(co / "sessions.csv").set_index("session_id")
    attrs = pd.read_csv(so / "spell_attrs.csv")
    learn = pd.read_csv(so / "s1_overall.csv").set_index("spell").rate

    alias = load_alias()
    sp = C[C.verb == "spell"].copy()
    known = set(attrs.spell) | set(sp.action_name.map(norm))
    sp["spell"] = standardise(sp.action_name, known, alias)
    sp = sp.merge(attrs[["spell", "level", "homebrew"]], on="spell", how="inner")
    sp = sp[~sp.homebrew.astype(bool) & sp.level.notna()]
    sp["hard_t"] = np.where(sp.control == "hard", sp.ctrl_targets, 0)
    sp["soft_t"] = np.where(sp.control == "soft", sp.ctrl_targets, 0)
    sp["ctrl_score"] = sp.hard_t + 0.5 * sp.soft_t

    g = sp.groupby("spell")
    prof = pd.DataFrame({"casts": g.size(), "level": g["level"].first(),
                         "p_damage": g.damage.apply(lambda x: (x > 0).mean()),
                         "p_control": g.control.apply(lambda x: (x != "none").mean()),
                         "p_heal": g.healing.apply(lambda x: (x > 0).mean()),
                         "p_buff": g.buff.apply(lambda x: x.astype(bool).mean())})
    prof = prof[prof.casts >= MIN_CASTS]
    cats = {"damage": prof.index[prof.p_damage >= 0.5], "control": prof.index[prof.p_control >= 0.2],
            "heal": prof.index[prof.p_heal >= 0.5], "buff": prof.index[prof.p_buff >= 0.5]}
    metric = {"damage": lambda x: stats.trim_mean(x.damage, 0.05), "control": lambda x: x.ctrl_score.mean(),
              "heal": lambda x: x.healing.mean()}

    rows = []
    for cat in ["damage", "control", "heal"]:
        d = sp[sp.spell.isin(cats[cat])]
        cell = d.groupby(["spell", "level", "tier"]).apply(metric[cat]).rename("m").reset_index()
        cell["n"] = d.groupby(["spell", "level", "tier"]).size().values
        cell = cell[cell.n >= MIN_CELL]
        med = cell.groupby(["level", "tier"]).m.median().rename("med")
        peers = cell.groupby(["level", "tier"]).size().rename("peers")
        cell = cell.join(med, on=["level", "tier"]).join(peers, on=["level", "tier"])
        cell = cell[(cell.med > 0) & (cell.peers >= 2)]
        cell["rel"] = cell.m / cell.med
        for spell, cs in cell.groupby("spell"):
            w = cs.n.values
            rel = np.average(cs.rel, weights=w)
            # bootstrap: resample this spell's casts by session within each cell, peer medians fixed
            sub = d[d.spell == spell]
            by = {k: v for k, v in sub.groupby("session_id")}
            sids = list(by)
            bs = []
            for _ in range(reps):
                b = pd.concat([by[sids[i]] for i in rng.integers(0, len(sids), len(sids))])
                vals, ws = [], []
                for r in cs.itertuples():
                    bt = b[b.tier == r.tier]
                    if len(bt) >= 1:
                        vals.append(metric[cat](bt) / r.med)
                        ws.append(len(bt))
                if ws:
                    bs.append(np.average(vals, weights=ws))
            top_tier = cs.sort_values("n").iloc[-1]
            rows.append(dict(category=cat, spell=spell, level=int(cs.level.iloc[0]), casts=int(prof.at[spell, "casts"]),
                             rated_casts=int(w.sum()), rel_score=rel, rel_lo=np.quantile(bs, 0.025),
                             rel_hi=np.quantile(bs, 0.975), main_tier=top_tier.tier, metric_main_tier=top_tier.m,
                             peer_median_main_tier=top_tier.med))

    # spells that qualify for a category but have no same-level peer: report the raw metric, no score
    rated = {(r["category"], r["spell"]) for r in rows}
    for cat in ["damage", "control", "heal"]:
        for spell in cats[cat]:
            if (cat, spell) in rated:
                continue
            x = sp[sp.spell == spell]
            mt = x.groupby("tier").size().idxmax()
            rows.append(dict(category=cat, spell=spell, level=int(prof.at[spell, "level"]), casts=int(prof.at[spell, "casts"]),
                             rated_casts=0, rel_score=np.nan, rel_lo=np.nan, rel_hi=np.nan, main_tier=mt,
                             metric_main_tier=metric[cat](x[x.tier == mt]), peer_median_main_tier=np.nan, no_peers=True))

    # buffs: event study uplift
    busy = (C[(C.control != "none") | C.buff.astype(bool)].groupby("session_id").seq.apply(set).to_dict())
    bc = sp[sp.spell.isin(cats["buff"])]
    up = buff_uplift(bc, A, S, busy)
    for spell, u in up.groupby("spell"):
        if len(u) < MIN_CASTS:
            continue
        by = {k: v.uplift.values for k, v in u.groupby("session_id")}
        sids = list(by)
        bs = [np.concatenate([by[sids[i]] for i in rng.integers(0, len(sids), len(sids))]).mean() for _ in range(reps)]
        rows.append(dict(category="buff", spell=spell, level=int(prof.at[spell, "level"]), casts=int(prof.at[spell, "casts"]),
                         rated_casts=len(u), rel_score=u.uplift.mean(), rel_lo=np.quantile(bs, 0.025),
                         rel_hi=np.quantile(bs, 0.975), main_tier=u.tier.mode()[0], metric_main_tier=u.uplift.mean(),
                         peer_median_main_tier=np.nan))

    R = pd.DataFrame(rows)
    lvl_casts = sp.groupby("level").size()
    R["use_share"] = R.casts / R.level.map(lvl_casts)
    R["learn_rate"] = R.spell.map(learn)
    R["small_sample"] = R.casts < SMALL
    grp = R.groupby(["category", "level"])
    R["learn_below_median"] = R.learn_rate < grp.learn_rate.transform("median")
    R["use_below_median"] = R.use_share < grp.use_share.transform("median")
    R["peers_in_level"] = grp.spell.transform("size")
    # with fewer than 4 rated peers, "below the median" says little (with 2 peers one is always below)
    R["underrated"] = (R.learn_below_median | R.use_below_median) & (R.peers_in_level >= 4)
    R["underrated_and_better"] = R.underrated & (R.rel_lo > 1)
    R = R.sort_values(["category", "level", "rel_score"], ascending=[True, True, False], na_position="last")
    # buff uplift is not used for picks: defensive buffs cannot raise damage, so a positive value (e.g. death
    # ward) signals confounding (cast before big fights). Buffs stay in picks_scores.csv as exploratory.
    R["exploratory"] = R.category.eq("buff")
    R["no_peers"] = R.get("no_peers", pd.Series(False, index=R.index)).fillna(False).astype(bool)
    R.to_csv(out / "picks_scores.csv", index=False)
    top = R[~R.exploratory].groupby(["category", "level"]).head(3)
    top.to_csv(out / "picks_top.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["category", "level", "spell", "casts", "rel_score", "rel_lo", "rel_hi", "main_tier", "metric_main_tier",
            "peer_median_main_tier", "learn_rate", "use_share", "underrated", "small_sample", "peers_in_level"]
    print(top[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:5], *(int(a) for a in sys.argv[5:6]))
