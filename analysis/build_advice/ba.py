"""Build advice: decisions for a melee striker and a ranged controller, from FIREBALL.

Usage: python ba.py <combat_out> <spells_out> <out_dir>
Needs combat prep outputs (char_actions.csv, actions_all.csv) and c2_character_sessions.csv in <combat_out>,
and casts.csv + spell_attrs.csv in <spells_out>.

Unit of analysis: a character at one level ("unit") with at least MIN_TURNS turns.
A turn = a run of consecutive actions by the same actor in a session (one character's go).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

MIN_TURNS = 5
REPS = 1000
SEED = 7
MELEE_CLASSES = ["Barbarian", "Fighter", "Paladin", "Rogue", "Monk", "Ranger"]
CASTER_CLASSES = ["Wizard", "Sorcerer", "Bard", "Druid", "Warlock", "Cleric"]
ARMOUR_DIPS = {"Cleric", "Fighter", "Paladin", "Artificer", "Barbarian", "Ranger"}

TWO = re.compile(r"great ?sword|great ?axe|\bmaul\b|glaive|halberd|\bpike\b|\blance\b|greatclub|2-handed|two[- ]handed")
ONE = re.compile(r"longsword|rapier|scimitar|shortsword|battle ?axe|war ?hammer|war ?pick|\bflail\b|morningstar"
                 r"|\bmace\b|hand ?axe|dagger|\bclub\b|\bspear\b|trident|\bwhip\b|sickle|katana|1-handed|one[- ]handed"
                 r"|shadow blade")
UNARMED = re.compile(r"unarmed|\bclaws?\b|\bfist")
RANGED = re.compile(r"longbow|shortbow|crossbow|\bsling\b|\bdarts?\b|blowgun|pistol|musket|rifle|pepperbox")


def weapon_kind(name: str) -> str:
    n = str(name).lower()
    if RANGED.search(n):
        return "ranged"
    if TWO.search(n):
        return "two"
    if ONE.search(n):
        return "one"
    if UNARMED.search(n):
        return "unarmed"
    return ""


def parse_classes(raw: str) -> dict:
    out = {}
    for part in str(raw).split("/"):
        m = re.match(r"\s*([A-Za-z]+)\s+(\d+)", part)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0) + int(m.group(2))
    return out


def boot_mean(df, col, key="char_key", reps=REPS, seed=SEED, stat=np.mean):
    """Character-cluster bootstrap of stat(df[col]). Returns (estimate, lo, hi)."""
    df = df.dropna(subset=[col])
    if len(df) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    groups = [g[col].values for _, g in df.groupby(key)]
    est = stat(df[col].values)
    bs = []
    for _ in range(reps):
        idx = rng.integers(0, len(groups), len(groups))
        bs.append(stat(np.concatenate([groups[i] for i in idx])))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi


def tmean(x):
    return trim_mean(x, 0.05)


def build_turns(cout: Path) -> pd.DataFrame:
    c = pd.read_csv(cout / "char_actions.csv", low_memory=False)
    a = pd.read_csv(cout / "actions_all.csv", low_memory=False).sort_values(["session_id", "seq"])
    new_run = (a.actor_name != a.actor_name.shift()) | (a.session_id != a.session_id.shift())
    a["run"] = new_run.cumsum()
    c = c.merge(a[["session_id", "seq", "run"]], on=["session_id", "seq"], how="left")
    c["hard_t"] = np.where(c.control == "hard", c.ctrl_targets, 0)
    c["soft_t"] = np.where(c.control == "soft", c.ctrl_targets, 0)
    c["group_hard"] = (c.control == "hard") & (c.ctrl_group == "group")
    c["is_attack"] = c.verb == "attack"
    c["wkind"] = np.where(c.verb == "attack", c.action_name.map(weapon_kind), "")
    T = c.groupby("run").agg(session_id=("session_id", "first"), char_key=("char_key", "first"),
                             main=("actor_main_class", "first"), level=("actor_level", "first"),
                             tier=("tier", "first"), multiclass=("actor_multiclass", "first"),
                             class_raw=("actor_class_raw", "first"), race=("actor_race", "first"),
                             max_hp=("actor_max_hp", "first"), dmg=("damage", "sum"), hard=("hard_t", "sum"),
                             soft=("soft_t", "sum"), group_hard=("group_hard", "any"), n_actions=("seq", "size"),
                             attack_turn=("is_attack", "any"))
    T["dmg_att"] = np.where(T.attack_turn, T.dmg, np.nan)
    return c, T


def build_units(c, T, cout: Path) -> pd.DataFrame:
    U = T.groupby(["char_key", "level"]).agg(main=("main", "first"), tier=("tier", "first"),
                                             multiclass=("multiclass", "first"), class_raw=("class_raw", "first"),
                                             max_hp=("max_hp", "median"), turns=("dmg", "size"),
                                             dmg=("dmg", "mean"), dmg_att=("dmg_att", "mean"),
                                             attack_share=("attack_turn", "mean"), hard=("hard", "mean"), soft=("soft", "mean"),
                                             group_hard=("group_hard", "mean")).reset_index()
    U = U[U.turns >= MIN_TURNS].copy()
    U["ctrl"] = U.hard + 0.5 * U.soft
    d = pd.read_csv(cout / "c2_character_sessions.csv")
    dz = d.groupby(["char_key", "level"])[["danger_entries", "down_entries", "exposure"]].sum().reset_index()
    U = U.merge(dz, on=["char_key", "level"], how="left")
    U["down10"] = 10 * U.down_entries / U.exposure
    U["danger10"] = 10 * U.danger_entries / U.exposure
    # weapon profile per character
    w = c[c.wkind != ""].groupby("char_key").wkind.value_counts().unstack(fill_value=0)
    for k in ["two", "one", "unarmed", "ranged"]:
        if k not in w:
            w[k] = 0
    w["n_w"] = w[["two", "one", "unarmed", "ranged"]].sum(axis=1)
    w["melee_share"] = (w.two + w.one + w.unarmed) / w.n_w
    mel = w.two + w.one + w.unarmed

    def style(r):
        m = r.two + r.one + r.unarmed
        if m == 0:
            return ""
        for k in ["two", "one", "unarmed"]:
            if r[k] / m >= 0.7:
                return k
        return "mixed"
    w["style"] = w.apply(style, axis=1)
    w["melee"] = (w.n_w >= 5) & (w.melee_share >= 0.7) & (mel >= 5)
    U = U.merge(w[["melee", "style", "melee_share", "n_w"]], left_on="char_key", right_index=True, how="left")
    U["melee"] = U.melee.fillna(False).astype(bool)
    U["style"] = U["style"].fillna("")
    return U


def class_table(U, classes, tiers, metrics, mask):
    rows = []
    for t in tiers:
        for cl in classes:
            g = U[mask & (U.main == cl) & (U.tier == t) & (~U.multiclass)]
            if g.char_key.nunique() < 12:
                continue
            row = {"tier": t, "class": cl, "units": len(g), "chars": g.char_key.nunique(),
                   "max_hp": g.max_hp.median()}
            for m in metrics:
                est, lo, hi = boot_mean(g, m, stat=tmean if m.startswith("dmg") else np.mean)
                row.update({m: est, f"{m}_lo": lo, f"{m}_hi": hi})
            rows.append(row)
    return pd.DataFrame(rows)


def matched_diff(U, treat_mask, base_mask, cell, metrics, label_col=None, min_units=12, min_chars=8):
    """Each treated unit minus the mean of base units in the same cell; summarise by label."""
    base = U[base_mask].groupby(cell)[metrics].mean()
    nb = U[base_mask].groupby(cell).size()
    tr = U[treat_mask].copy()
    tr = tr.join(base, on=cell, rsuffix="_base").join(nb.rename("n_base"), on=cell)
    tr = tr[tr.n_base >= 3]
    for m in metrics:
        tr[f"d_{m}"] = tr[m] - tr[f"{m}_base"]
    labels = [None] if label_col is None else sorted(tr[label_col].dropna().unique())
    rows = []
    for lab in labels:
        g = tr if lab is None else tr[tr[label_col] == lab]
        if len(g) < min_units or g.char_key.nunique() < min_chars:
            continue
        row = {"label": lab if lab is not None else "all", "units": len(g), "chars": g.char_key.nunique()}
        for m in metrics:
            est, lo, hi = boot_mean(g, f"d_{m}")
            row.update({f"d_{m}": est, f"d_{m}_lo": lo, f"d_{m}_hi": hi, f"base_{m}": g[f"{m}_base"].mean()})
        rows.append(row)
    return pd.DataFrame(rows)


def dip_label(raw, main):
    cls = parse_classes(raw)
    cls.pop(main, None)
    if len(cls) != 1:
        return None
    (k, v), = cls.items()
    return f"{main}+{k}"


def main(cout, sout, out_dir):
    cout, sout, out = Path(cout), Path(sout), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    c, T = build_turns(cout)
    U = build_units(c, T, cout)
    U.to_csv(out / "ba_units.csv", index=False)
    print(f"turns {len(T)}, units {len(U)}, melee units {U.melee.sum()}")
    tiers = ["5-10", "11-16"]

    # M1 melee class
    m1 = class_table(U, MELEE_CLASSES, tiers, ["dmg", "dmg_att", "attack_share", "down10", "danger10"], U.melee)
    m1.to_csv(out / "m1_melee_class.csv", index=False)

    # M2 dips for melee mains
    mm = U.melee & U.main.isin(MELEE_CLASSES)
    U["dip"] = [dip_label(r, m) if mc else None for r, m, mc in zip(U.class_raw, U.main, U.multiclass)]
    m2 = matched_diff(U, mm & U.multiclass & U.dip.notna(), mm & ~U.multiclass, ["main", "level"],
                      ["dmg", "down10"], label_col="dip", min_units=10, min_chars=6)
    m2b = matched_diff(U, mm & U.multiclass, mm & ~U.multiclass, ["main", "level"], ["dmg", "down10"])
    m2 = pd.concat([m2b.assign(label="any dip"), m2])
    m2.to_csv(out / "m2_melee_dips.csv", index=False)

    # M3 weapon style within melee single-class: two-handed vs one-handed, cell = main x tier
    ms = U.melee & ~U.multiclass & U.main.isin(MELEE_CLASSES)
    rows = []
    for cl in ["all"] + MELEE_CLASSES:
        cm = ms if cl == "all" else ms & (U.main == cl)
        r = matched_diff(U, cm & (U["style"] == "two"), cm & (U["style"] == "one"), ["main", "tier"],
                         ["dmg", "down10"], min_units=10, min_chars=8)
        if len(r):
            r["label"] = cl
            rows.append(r)
    m3 = pd.concat(rows) if rows else pd.DataFrame()
    m3.to_csv(out / "m3_two_vs_one.csv", index=False)

    # C1 caster class
    c1 = class_table(U, CASTER_CLASSES, tiers, ["ctrl", "group_hard", "dmg", "down10"], ~U.melee)
    c1["group_hard10"] = c1.group_hard * 10
    c1.to_csv(out / "c1_caster_class.csv", index=False)

    # C2 save ability fail rates (character spell casts that forced saves)
    k = pd.read_csv(sout / "casts.csv", low_memory=False)
    k = k[(k.saves > 0) & k.save_ability.notna()].copy()
    k["char_key"] = k.actor_name.astype(str)
    rows = []
    rng = np.random.default_rng(SEED)
    for t in ["1-4", "5-10", "11-16", "17-20"]:
        kt = k[k.tier == t]
        S = kt.pivot_table(index="session_id", columns="save_ability", values="saves", aggfunc="sum", fill_value=0)
        F = kt.pivot_table(index="session_id", columns="save_ability", values="saves_failed", aggfunc="sum",
                           fill_value=0).reindex_like(S)
        Sv, Fv, n = S.values, F.values, len(S)
        bs = []
        for _ in range(REPS):
            cnt = np.bincount(rng.integers(0, n, n), minlength=n)
            with np.errstate(invalid="ignore", divide="ignore"):
                bs.append((cnt @ Fv) / (cnt @ Sv))
        bs = pd.DataFrame(bs, columns=S.columns)
        g = kt.groupby("save_ability").agg(casts=("saves", "size"), saves=("saves", "sum"),
                                           failed=("saves_failed", "sum"))
        for ab, r in g.iterrows():
            rows.append({"tier": t, "ability": ab, "casts": r.casts, "saves": r.saves,
                         "fail_rate": r.failed / r.saves, "lo": bs[ab].quantile(0.025), "hi": bs[ab].quantile(0.975)})
    c2 = pd.DataFrame(rows)
    # within-caster: deviation from the caster's own overall fail rate (removes DC differences)
    kk = k.copy()
    kk["key"] = kk.actor_name.astype(str) + "|" + kk.actor_level.astype(str)
    tot = kk.groupby("key")[["saves", "saves_failed"]].sum()
    tot["own"] = tot.saves_failed / tot.saves
    per = kk.groupby(["key", "save_ability"])[["saves", "saves_failed"]].sum().reset_index()
    per = per.join(tot.own, on="key")
    multi = per.groupby("key").save_ability.nunique()
    per = per[per.key.map(multi) >= 2]
    per["dev"] = per.saves_failed / per.saves - per.own
    w = per.groupby("save_ability").apply(lambda g: np.average(g.dev, weights=g.saves)).rename("within_dev")
    c2 = c2.merge(w, left_on="ability", right_index=True, how="left")
    c2.to_csv(out / "c2_save_fail.csv", index=False)

    # C3 control spells: which land control, which save they use, which classes get them
    attrs = pd.read_csv(sout / "spell_attrs.csv")
    ca = pd.read_csv(sout / "casts.csv", low_memory=False)
    ca = ca.merge(c[["session_id", "turn", "action_no", "control", "ctrl_targets", "ctrl_group"]],
                  on=["session_id", "turn", "action_no"], how="inner")
    ca["hard_t"] = np.where(ca.control == "hard", ca.ctrl_targets, 0)
    ca["soft_t"] = np.where(ca.control == "soft", ca.ctrl_targets, 0)
    ca["landed"] = ca.control.isin(["hard", "soft"])
    sp = ca.groupby("spell").agg(casts=("spell", "size"), landed=("landed", "mean"), hard=("hard_t", "mean"),
                                 soft=("soft_t", "mean"), dmg=("damage", "mean"),
                                 tier=("tier", lambda x: x.mode().iat[0] if x.notna().any() else "")).reset_index()
    sp["ctrl"] = sp.hard + 0.5 * sp.soft
    sp = sp.merge(attrs[["spell", "level", "save", "classes", "concentration"]], on="spell", how="left")
    sp = sp[(sp.casts >= 15) & (sp.landed >= 0.2)].sort_values(["level", "ctrl"], ascending=[True, False])
    sp.to_csv(out / "c3_control_spells.csv", index=False)

    # C4 caster survival: armour dip vs single class, same main and level
    cm = ~U.melee & U.main.isin(["Wizard", "Sorcerer", "Warlock", "Bard", "Druid"])

    def has_armour_dip(raw, main):
        cls = parse_classes(raw)
        cls.pop(main, None)
        return bool(set(cls) & ARMOUR_DIPS)
    U["armour_dip"] = [has_armour_dip(r, m) if mc else False for r, m, mc in zip(U.class_raw, U.main, U.multiclass)]
    c4 = matched_diff(U, cm & U.armour_dip, cm & ~U.multiclass, ["main", "level"],
                      ["down10", "danger10", "max_hp", "ctrl", "dmg"], min_units=10, min_chars=8)
    c4.to_csv(out / "c4_caster_armour.csv", index=False)
    for name, df in [("m1", m1), ("m2", m2), ("m3", m3), ("c1", c1), ("c2", c2), ("c3", sp.head(40)), ("c4", c4)]:
        print(f"\n== {name}")
        print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
