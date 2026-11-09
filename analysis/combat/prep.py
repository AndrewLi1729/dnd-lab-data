"""Prepare FIREBALL tables for the combat analysis.

Usage: python prep.py <repo root> <out_dir>
Writes:
  actions_all.csv   every action (characters and creatures) with order keys, used for windows
  char_actions.csv  character actions (official classes) with control and buff flags
  sessions.csv      per session: combatants, actions, creature damage per action
  control_review.csv  the 30 most common matched control effects, for manual review
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OFFICIAL = {"Artificer", "Barbarian", "Bard", "Cleric", "Druid", "Fighter", "Monk", "Paladin", "Ranger",
            "Rogue", "Sorcerer", "Warlock", "Wizard"}
HARD = re.compile(r"^(Paralyzed|Stunned|Stunning Strike|Incapacitated|Banished|Asleep|Unconscious|Petrified"
                  r"|Charmed \(Hypnotic Pattern\))(?![A-Za-z])")
SOFT = re.compile(r"^(Frightened|Prone|Restrained|Grappled|Slowed|Blinded|Charmed)(?![A-Za-z])")
BUFFS = re.compile(r"^(Blessed|Hasted|Aid|Shielded by Faith|Heroism|Bardic Inspiration|Death Ward|Greater Invisibility"
                   r"|Invisible|Enlarged|Guidance)\b")


def tier(level):
    return pd.cut(level, [0, 4, 10, 16, 20], labels=["1-4", "5-10", "11-16", "17-20"]).astype(str)


def classify(effects: str):
    """Return (control level, is_buff) for one action's effects_gained string."""
    if not isinstance(effects, str):
        return "none", False
    parts = [p.strip() for chunk in effects.split(";") for p in chunk.split(",")]
    hard = any(HARD.match(p) for p in parts)
    soft = any(SOFT.match(p) for p in parts)
    buff = any(BUFFS.match(p) for p in parts)
    return ("hard" if hard else "soft" if soft else "none"), buff


def controlled_targets(r) -> int:
    if r.saves > 0:
        return int(r.saves_failed)
    if r.hits > 0:
        return int(r.hits)
    if r.n_targets > 0:
        return int(r.n_targets)
    return 1


def main(root, out_dir):
    root, out = Path(root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(root / "fireball_actions.csv", low_memory=False)
    for col in ["damage", "healing", "attacks", "hits", "crits", "misses", "saves", "saves_failed", "n_targets"]:
        a[col] = pd.to_numeric(a[col], errors="coerce").fillna(0)
    a = a.sort_values(["session_id", "turn", "action_no"]).reset_index(drop=True)
    a["seq"] = a.groupby("session_id").cumcount()
    a["is_char"] = (a.actor_kind == "character") & a.actor_main_class.isin(OFFICIAL)
    a["char_key"] = np.where(a.is_char, a.actor_name.astype(str) + "|" + a.speaker_id.astype(str), "")
    a[["session_id", "seq", "turn", "action_no", "actor_name", "actor_kind", "is_char", "damage", "attacks", "hits",
       "crits"]].to_csv(out / "actions_all.csv", index=False)

    s = pd.read_csv(root / "fireball_sessions.csv")
    cre = a[a.actor_kind == "creature"].groupby("session_id").damage.mean().rename("creature_dmg_per_action")
    s = s.join(cre, on="session_id")
    s[["session_id", "source_file", "turns", "actions", "players", "max_combatants", "creature_dmg_per_action"]].to_csv(
        out / "sessions.csv", index=False)

    c = a[a.is_char].copy()
    bad_level = ~c.actor_level.between(1, 20)
    print(f"dropped {bad_level.sum()} character actions with level outside 1-20 "
          f"({c[bad_level].char_key.nunique()} characters)")
    c = c[~bad_level]
    cl = c.effects_gained.map(classify)
    c["control"] = cl.str[0]
    c["buff"] = cl.str[1]
    c["ctrl_targets"] = np.where(c.control != "none", c.apply(controlled_targets, axis=1), 0)
    c.loc[(c.control != "none") & (c.ctrl_targets == 0), "control"] = "none"  # nobody affected
    c["ctrl_group"] = np.where(c.control == "none", "", np.where(c.ctrl_targets >= 2, "group", "single"))
    c["tier"] = tier(c.actor_level)
    keep = ["session_id", "seq", "turn", "action_no", "char_key", "actor_name", "speaker_id", "actor_main_class",
            "actor_level", "tier", "actor_multiclass", "actor_class_raw", "actor_race", "actor_hp", "actor_max_hp",
            "verb", "action_name", "damage", "healing", "attacks", "hits", "crits", "saves", "saves_failed",
            "effects_gained", "control", "ctrl_targets", "ctrl_group", "buff"]
    c[keep].to_csv(out / "char_actions.csv", index=False)

    m = c[c.control != "none"].assign(eff=c.effects_gained.str.split(r"\s*;\s*")).explode("eff")
    rev = (m.groupby(["eff", "control"]).agg(actions=("eff", "size"), targets=("ctrl_targets", "sum"))
           .sort_values("actions", ascending=False).head(30).reset_index())
    rev.to_csv(out / "control_review.csv", index=False)
    print(f"character actions: {len(c)}; control actions: {(c.control != 'none').sum()} "
          f"(hard {(c.control == 'hard').sum()}, soft {(c.control == 'soft').sum()}); "
          f"group share: {(c.ctrl_group == 'group').sum() / max((c.control != 'none').sum(), 1):.1%}; "
          f"buff actions: {c.buff.sum()}")
    print(rev.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
