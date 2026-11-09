"""C2: who drops into danger (HP <= 20% of max) or to 0.

Usage: python c2_danger.py <repo root> <out_dir>
HP trajectory: anchor at each observed HP (the character's own actions), add later damage/healing aimed at
that character (fireball_damage, by seq), cap at max. Each observation re-anchors.
Validation: predict each observation from the previous one + the deltas in between.
Exposure: actions present = session actions from the character's first to last appearance.
Writes c2_validation.csv, c2_character_sessions.csv, c2_model.csv, c2_class_tier.csv, c2_hp_quartile.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

DANGER = 0.20


def main(root, out_dir):
    root, out = Path(root), Path(out_dir)
    acts = pd.read_csv(out / "char_actions.csv", low_memory=False)
    A = pd.read_csv(out / "actions_all.csv", low_memory=False)
    S = pd.read_csv(out / "sessions.csv").set_index("session_id")
    D = pd.read_csv(root / "fireball_damage.csv", low_memory=False)
    D = D[~D.outlier.astype(bool)]
    order = A[["session_id", "turn", "action_no", "seq"]]
    D = D.merge(order, on=["session_id", "turn", "action_no"], how="inner")
    D["delta"] = np.where(D.type == "damage", -D.amount, D.amount)
    dmap = {k: g.sort_values("seq")[["seq", "delta"]].values for k, g in D.groupby(["session_id", "target_name"])}

    acts = acts.dropna(subset=["actor_hp", "actor_max_hp"])
    acts = acts[acts.actor_max_hp > 0]
    val, cs = [], []
    for (sid, name), g in acts.sort_values("seq").groupby(["session_id", "actor_name"]):
        obs = g[["seq", "actor_hp", "actor_max_hp"]].values
        mx = obs[0, 2]
        ev = dmap.get((sid, name), np.empty((0, 2)))
        danger_now = obs[0, 1] <= DANGER * mx
        down_now = obs[0, 1] <= 0
        danger_entries = int(danger_now)
        down_entries = int(down_now)
        min_ratio = obs[0, 1] / mx
        for i in range(len(obs)):
            start_seq, hp = obs[i, 0], obs[i, 1]
            end_seq = obs[i + 1, 0] if i + 1 < len(obs) else np.inf
            seg = ev[(ev[:, 0] > start_seq) & (ev[:, 0] < end_seq)] if len(ev) else ev
            cur = hp
            for _, dlt in seg:
                cur = min(mx, cur + dlt)
                min_ratio = min(min_ratio, cur / mx)
                if cur <= DANGER * mx and not danger_now:
                    danger_entries += 1
                danger_now = cur <= DANGER * mx
                if cur <= 0 and not down_now:
                    down_entries += 1
                down_now = cur <= 0
            if i + 1 < len(obs):
                nxt = obs[i + 1, 1]
                val.append((cur, nxt, mx))
                # re-anchor on the observation
                if nxt <= DANGER * mx and not danger_now:
                    danger_entries += 1
                danger_now = nxt <= DANGER * mx
                if nxt <= 0 and not down_now:
                    down_entries += 1
                down_now = nxt <= 0
                min_ratio = min(min_ratio, nxt / mx)
        o_ratio = obs[:, 1] / mx
        o_danger = o_ratio <= DANGER
        obs_danger_entries = int(o_danger[0]) + int(((~o_danger[:-1]) & o_danger[1:]).sum())
        r0 = g.iloc[0]
        cs.append(dict(session_id=sid, actor_name=name, char_key=r0.char_key, main_class=r0.actor_main_class,
                       level=r0.actor_level, tier=r0.tier, max_hp=mx, multiclass=r0.actor_multiclass,
                       first_seq=obs[0, 0], last_seq=max(obs[-1, 0], ev[-1, 0] if len(ev) else 0),
                       danger_entries=danger_entries, down_entries=down_entries, min_hp_ratio=min_ratio,
                       obs_danger_entries=obs_danger_entries))
    V = pd.DataFrame(val, columns=["pred", "obs", "mx"])
    pd_, od = V.pred <= DANGER * V.mx, V.obs <= DANGER * V.mx
    vrow = dict(pairs=len(V), exact=(V.pred == V.obs).mean(), within_10pct=((V.pred - V.obs).abs() <= 0.1 * V.mx).mean(),
                danger_accuracy=(pd_ == od).mean(), danger_sensitivity=(pd_ & od).sum() / max(od.sum(), 1),
                danger_specificity=(~pd_ & ~od).sum() / max((~od).sum(), 1), observed_danger_share=od.mean())
    pd.DataFrame([vrow]).to_csv(out / "c2_validation.csv", index=False)

    X = pd.DataFrame(cs)
    X["exposure"] = (X.last_seq - X.first_seq + 1).clip(lower=1)
    X = X.join(S[["max_combatants"]], on="session_id")
    X.to_csv(out / "c2_character_sessions.csv", index=False)

    rates = X.groupby(["main_class", "tier"]).agg(char_sessions=("exposure", "size"), exposure=("exposure", "sum"),
                                                   danger=("danger_entries", "sum"), down=("down_entries", "sum"))
    rates["danger_per10"] = 10 * rates.danger / rates.exposure
    rates["down_per10"] = 10 * rates.down / rates.exposure
    rates.reset_index().to_csv(out / "c2_class_tier.csv", index=False)

    M = X[X.tier.isin(["1-4", "5-10", "11-16", "17-20"])].copy()
    M["log_hp"] = np.log(M.max_hp)
    M["log_exposure"] = np.log(M.exposure)
    rows = []
    for y in ["danger_entries", "down_entries", "obs_danger_entries"]:
        fit = smf.glm(f"{y} ~ C(main_class, Treatment('Fighter')) + C(tier) + log_hp + max_combatants", M,
                      family=__import__("statsmodels.api", fromlist=["x"]).families.Poisson(),
                      offset=M.log_exposure).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(M.char_key)[0]})
        t = pd.DataFrame({"outcome": y, "term": fit.params.index, "rate_ratio": np.exp(fit.params.values),
                          "lo": np.exp(fit.conf_int()[0].values), "hi": np.exp(fit.conf_int()[1].values), "p": fit.pvalues.values})
        cls = t.term.str.startswith("C(main_class")
        t.loc[cls, "p_holm"] = multipletests(t.loc[cls, "p"], method="holm")[1]
        rows.append(t)
    mod = pd.concat(rows)
    mod.to_csv(out / "c2_model.csv", index=False)

    q = []
    for t, g in M.groupby("tier"):
        g = g.assign(hp_q=pd.qcut(g.max_hp, 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"], duplicates="drop"))
        for k, h in g.groupby("hp_q", observed=True):
            q.append(dict(tier=t, hp_quartile=k, max_hp_median=h.max_hp.median(), char_sessions=len(h),
                          danger_per10=10 * h.danger_entries.sum() / h.exposure.sum(),
                          down_per10=10 * h.down_entries.sum() / h.exposure.sum()))
    pd.DataFrame(q).to_csv(out / "c2_hp_quartile.csv", index=False)

    pd.set_option("display.width", 250)
    print(pd.DataFrame([vrow]).round(3).to_string(index=False))
    print(rates.reset_index().query("tier == '5-10'").round(3).sort_values("danger_per10").to_string(index=False))
    print(mod[mod.term.str.contains("main_class|log_hp|tier")].round(3).to_string(index=False))
    print(pd.DataFrame(q).round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
