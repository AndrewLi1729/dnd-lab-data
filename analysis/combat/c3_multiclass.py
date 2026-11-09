"""C3: do multiclass characters do more in play than single-class ones of the same main class and level?

Usage: python c3_multiclass.py <out_dir> [reps]
Unit = character at a level (char_key x level) with >= 5 actions. Each multiclass unit is compared with the
mean of single-class units of the same main class and total level. Differences are averaged; 95% intervals
come from resampling multiclass units (pair bootstrap). Top 5 combos are reported separately.
Sensitivity: level 5+ only; drop the controller (speaker) contributing the most multiclass units.
Writes c3_overall.csv and c3_combos.csv.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MIN_ACTIONS = 5


def combo(raw: str) -> str:
    cls = sorted({re.sub(r"\s*\d+$", "", p.strip()) for p in str(raw).split("/")})
    return " / ".join(cls)


def units(c: pd.DataFrame, danger: pd.DataFrame) -> pd.DataFrame:
    g = c.groupby(["char_key", "actor_level"])
    u = pd.DataFrame({
        "main_class": g.actor_main_class.first(), "multiclass": g.actor_multiclass.first().astype(bool),
        "class_raw": g.actor_class_raw.first(), "speaker": g.speaker_id.first(), "actions": g.size(),
        "dmg_trim": g.damage.apply(lambda x: stats.trim_mean(x, 0.05)),
        "hard_per10": g.apply(lambda x: 10 * np.where(x.control == "hard", x.ctrl_targets, 0).sum() / len(x)),
    }).reset_index()
    d = danger.groupby(["char_key", "level"]).agg(danger=("danger_entries", "sum"), exposure=("exposure", "sum"))
    d["danger_per10"] = 10 * d.danger / d.exposure
    u = u.join(d.danger_per10, on=["char_key", "actor_level"])
    u = u[u.actions >= MIN_ACTIONS]
    u["combo"] = np.where(u.multiclass, u.class_raw.map(combo), "")
    return u


def matched(u: pd.DataFrame) -> pd.DataFrame:
    ref = u[~u.multiclass].groupby(["main_class", "actor_level"])[["dmg_trim", "hard_per10", "danger_per10"]].mean()
    n_ref = u[~u.multiclass].groupby(["main_class", "actor_level"]).size().rename("n_ref")
    m = u[u.multiclass].join(ref, on=["main_class", "actor_level"], rsuffix="_ref").join(n_ref, on=["main_class", "actor_level"])
    m = m.dropna(subset=["dmg_trim_ref"])
    for k in ["dmg_trim", "hard_per10", "danger_per10"]:
        m["d_" + k] = m[k] - m[k + "_ref"]
    return m


def summarise(m: pd.DataFrame, label: str, rng, reps) -> dict:
    row = {"subset": label, "multiclass_units": len(m), "matched_cells": m[["main_class", "actor_level"]].drop_duplicates().shape[0]}
    for k in ["d_dmg_trim", "d_hard_per10", "d_danger_per10"]:
        v = m[k].dropna().values
        if len(v) < 10:
            continue
        bs = [rng.choice(v, len(v)).mean() for _ in range(reps)]
        row[k] = v.mean()
        row[k + "_lo"], row[k + "_hi"] = np.quantile(bs, [0.025, 0.975])
    return row


def main(out_dir, reps=1000, seed=13):
    out = Path(out_dir)
    c = pd.read_csv(out / "char_actions.csv", low_memory=False)
    danger = pd.read_csv(out / "c2_character_sessions.csv")
    rng = np.random.default_rng(seed)
    u = units(c, danger)
    m = matched(u)
    top_speaker = m.speaker.value_counts().index[0]
    rows = [summarise(m, "all", rng, reps),
            summarise(m[m.actor_level >= 5], "level 5+", rng, reps),
            summarise(m[m.speaker != top_speaker], "without top controller", rng, reps)]
    o = pd.DataFrame(rows)
    o.to_csv(out / "c3_overall.csv", index=False)
    top = m.combo.value_counts().head(5).index
    cb = pd.DataFrame([summarise(m[m.combo == k], k, rng, reps) for k in top])
    cb.to_csv(out / "c3_combos.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"units: {len(u)} (multiclass {u.multiclass.sum()}), matched multiclass units: {len(m)}")
    print(o.round(3).to_string(index=False))
    print(cb.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:3]))
