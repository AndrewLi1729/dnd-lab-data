"""C1 part 2-3: what a control action does for the rest of the party.

Usage: python c1_events.py <out_dir> [reps]
Event study: for every control action, compare allies' damage per action, hit rate and crit rate in the round
after it with the round before it (round = the session's max_combatants actions, scaled by 0.5/1/2), windows
clipped at the session edges; events need >= 2 ally actions on each side. Allies = other characters. Control: every eligible non-control character action in the same session (placebo, no sampling).
Effect per event = event change - mean placebo change; averaged over events; 95% interval from a
session-level cluster bootstrap.
Also: early hard control vs fight length (descriptive).
Writes c1_events.csv and c1_fight_length.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


MIN_ALLY_ACTIONS = 2


def window_stats(arr, actor, e, w, me):
    """Ally (characters other than `me`) totals in [e-w, e-1] and [e+1, e+w], clipped to the session.

    Returns ((n, damage, attacks, hits, crits) before, (...) after).
    """
    def part(lo, hi):
        m = (arr["seq"] >= lo) & (arr["seq"] <= hi) & arr["is_char"] & (actor != me)
        return m.sum(), arr["damage"][m].sum(), arr["attacks"][m].sum(), arr["hits"][m].sum(), arr["crits"][m].sum()
    return part(e - w, e - 1), part(e + 1, e + w)


def change(pre, post):
    """Per-ally-action damage change and hit/crit rate changes; None if too few ally actions."""
    (pn, pd_, pa, ph, pc), (qn, qd, qa, qh, qc) = pre, post
    if pn < MIN_ALLY_ACTIONS or qn < MIN_ALLY_ACTIONS:
        return None
    return dict(d_dmg=qd / qn - pd_ / pn,
                d_hit=(qh / qa - ph / pa) if pa and qa else np.nan,
                d_crit=(qc / qa - pc / pa) if pa and qa else np.nan)


def main(out_dir, reps=1000, seed=5):
    out = Path(out_dir)
    A = pd.read_csv(out / "actions_all.csv", low_memory=False)
    C = pd.read_csv(out / "char_actions.csv", low_memory=False)
    S = pd.read_csv(out / "sessions.csv").set_index("session_id")
    rng = np.random.default_rng(seed)
    sess = {sid: {"seq": g.seq.values, "is_char": g.is_char.values.astype(bool), "damage": g.damage.values,
                  "attacks": g.attacks.values, "hits": g.hits.values, "crits": g.crits.values,
                  "actor": g.actor_name.astype(str).values} for sid, g in A.groupby("session_id")}
    ctrl_seq = C[C.control != "none"].groupby("session_id").seq.apply(set).to_dict()
    char_pool = C[C.control == "none"].groupby("session_id")[["seq", "actor_name"]].apply(lambda g: g.values).to_dict()

    rows = []
    for mult in [0.5, 1.0, 2.0]:
        for kind in ["hard", "soft"]:
            for r in C[C.control == kind].itertuples():
                d = sess[r.session_id]
                w = max(1, int(round(mult * S.at[r.session_id, "max_combatants"])))
                base = dict(mult=mult, kind=kind, group=r.ctrl_group, tier=r.tier, session_id=r.session_id)
                ch = change(*window_stats(d, d["actor"], r.seq, w, str(r.actor_name)))
                if ch is None:
                    rows.append(base | dict(dropped=True))
                    continue
                # placebo = every eligible non-control character action in the same session (no sampling)
                pl = [change(*window_stats(d, d["actor"], s0, w, str(me)))
                      for s0, me in char_pool.get(r.session_id, []) if s0 not in ctrl_seq.get(r.session_id, set())]
                pl = pd.DataFrame([x for x in pl if x is not None])
                if pl.empty:
                    rows.append(base | dict(dropped=True))
                    continue
                rows.append(base | dict(dropped=False, n_placebo=len(pl), event_d_dmg=ch["d_dmg"],
                                        placebo_d_dmg=pl.d_dmg.mean(),
                                        d_dmg=ch["d_dmg"] - pl.d_dmg.mean(),
                                        d_hit=ch["d_hit"] - pl.d_hit.mean(),
                                        d_crit=ch["d_crit"] - pl.d_crit.mean()))
    E = pd.DataFrame(rows)

    res = []
    for (mult, kind), g in E.groupby(["mult", "kind"]):
        for label, sub in [("all", g), ("single", g[g.group == "single"]), ("group", g[g.group == "group"])]:
            used = sub[~sub.dropped]
            if len(used) < 30:
                continue
            row = dict(window_rounds=mult, kind=kind, subset=label, events_used=len(used),
                       events_dropped=int(sub.dropped.sum()), median_placebo_windows=used.n_placebo.median(),
                       event_d_dmg=used.event_d_dmg.mean(), placebo_d_dmg=used.placebo_d_dmg.mean())
            by = {s: x for s, x in used.groupby("session_id")}
            sids = list(by)
            boot = []
            for _ in range(reps):
                pick = rng.choice(len(sids), len(sids), replace=True)
                b = pd.concat([by[sids[i]] for i in pick])
                boot.append([b.d_dmg.mean(), b.d_hit.mean(), b.d_crit.mean()])
            boot = np.array(boot)
            for j, m in enumerate(["d_dmg", "d_hit", "d_crit"]):
                row[m] = used[m].mean()
                row[m + "_lo"], row[m + "_hi"] = np.nanquantile(boot[:, j], [0.025, 0.975])
            res.append(row)
    res = pd.DataFrame(res)
    res.to_csv(out / "c1_events.csv", index=False)

    # fight length vs early hard control (descriptive)
    tot = A.groupby("session_id").seq.max() + 1
    hard = C[C.control == "hard"].groupby("session_id").seq.min()
    f = pd.DataFrame({"actions": tot}).join(hard.rename("first_hard"))
    f = f[f.actions >= 9]
    f["early_hard"] = f.first_hard < f.actions / 3
    f["remaining_after_third"] = f.actions - np.ceil(f.actions / 3)
    u, p = stats.mannwhitneyu(f[f.early_hard].remaining_after_third, f[~f.early_hard].remaining_after_third)
    fl = pd.DataFrame([dict(sessions=len(f), early_hard=int(f.early_hard.sum()),
                            median_remaining_early=f[f.early_hard].remaining_after_third.median(),
                            median_remaining_none=f[~f.early_hard].remaining_after_third.median(),
                            mean_remaining_early=f[f.early_hard].remaining_after_third.mean(),
                            mean_remaining_none=f[~f.early_hard].remaining_after_third.mean(), mannwhitney_p=p)])
    fl.to_csv(out / "c1_fight_length.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
    print(fl.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:3]))
