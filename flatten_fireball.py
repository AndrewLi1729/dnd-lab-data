"""Flatten the FIREBALL dataset (Zhu et al., ACL 2023) into four CSV tables.

Usage:
    python flatten_fireball.py <folder of filtered/*.jsonl files> <output folder>

Download the input from https://huggingface.co/datasets/lara-martin/FIREBALL
(the filtered/ folder: 1,000 .jsonl files, one per game session).

Outputs:
    fireball_sessions.csv    one row per session
    fireball_actions.csv     one row per action resolved by the Avrae bot
    fireball_damage.csv      one row per damage or healing event on a target
    fireball_combatants.csv  one row per creature or character per session

Damage or healing over 1,000 in one hit is flagged as an outlier in
fireball_damage.csv and left out of every damage and healing total. Values over
1,000,000 are blanked.

Actions are parsed from the bot's result text, which is machine-generated and
consistent, rather than from player commands, which vary a lot.
"""
import csv
import glob
import json
import os
import re
import sys

HEADER = re.compile(
    r"^(?P<who>.+?) (?:"
    r"attacks with (?:an? |the )?(?P<attack>.+?)"
    r"|casts (?P<spell>.+?)(?: with an? (?:an? )?.+?)?"
    r"|uses (?P<ability>.+?)"
    r"|begins (?:an? )?(?P<begins>.+?)"
    r")!$"
)
ATTACKED = re.compile(r"^.+? attacked (?P<target>.+?) (?:and (?P<res>hit|crit)|but (?P<miss>missed))")
TOOK = re.compile(r"^(?P<target>.+?) took (?P<n>-?\d+) damage\.$")
HEALED = re.compile(r"^(?P<target>.+?) healed for (?P<n>-?\d+) health\.$")
SAVE = re.compile(r"^.+? rolled an? (?P<ability>\w+) save (?:and (?P<ok>succeeded)|but (?P<fail>failed))")
GAINED = re.compile(r"^.+? gained (?P<effect>.+?)\.$")
TEMP = re.compile(r"^\d+ temp HP$")
HP = re.compile(r"<(?P<cur>-?\d+)/(?P<max>\d+) HP; (?P<status>[^>]+)>(?: \(\+(?P<temp>\d+) temp\))?")
# Real D&D damage or healing never approaches this; larger values are jokes, tests or "kill" tricks.
OUTLIER = 1000
IMPOSSIBLE = 10**6
CLASS = re.compile(r"([A-Za-z' -]+?)\s+(\d+)")


def parse_hp(s):
    m = HP.search(s or "")
    if not m:
        return "", "", ""
    return int(m["cur"]), int(m["max"]), m["status"]


def parse_class(s):
    """'Sorcerer 18/Warlock 2' -> ('Sorcerer', 20, True)."""
    if not s:
        return "", "", False
    parts = [(c.strip(), int(n)) for c, n in CLASS.findall(s)]
    if not parts:
        return s.strip(), "", False
    main = max(parts, key=lambda p: p[1])[0]
    return main, sum(n for _, n in parts), len(parts) > 1


def actor_fields(a):
    a = a or {}
    cur, mx, status = parse_hp(a.get("hp"))
    main, level, multi = parse_class(a.get("class"))
    return {
        "name": a.get("name") or "",
        "kind": "character" if a.get("class") else "creature",
        "class_raw": a.get("class") or "",
        "main_class": main,
        "level": level,
        "multiclass": multi,
        "race": a.get("race") or "",
        "hp": cur,
        "max_hp": mx,
        "hp_status": status,
        "controller_id": a.get("controller_id") or "",
    }


def add_event(out, target, typ, amount, crit):
    outlier = amount > OUTLIER
    if not outlier:
        out[typ] += amount
    out["events"].append((target, typ, "" if amount > IMPOSSIBLE else amount, crit, outlier))


def parse_result(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    out = {"verb": "other", "action_name": "", "attacks": 0, "hits": 0, "crits": 0,
           "misses": 0, "saves": 0, "saves_failed": 0, "save_ability": "",
           "damage": 0, "healing": 0, "effects": [], "events": [], "targets": set()}
    if not lines:
        return out
    m = HEADER.match(lines[0])
    if not m and not (ATTACKED.match(lines[0]) or SAVE.match(lines[0]) or TOOK.match(lines[0])
                      or HEALED.match(lines[0]) or GAINED.match(lines[0])):
        # Homebrew actions with custom wording, e.g. "Wren might sneeze!"
        out["verb"] = "custom"
        out["action_name"] = lines[0][:120]
        lines = lines[1:]
    elif m:
        for verb in ("attack", "spell", "ability", "begins"):
            if m[verb]:
                out["verb"] = {"begins": "ability"}.get(verb, verb)
                out["action_name"] = m[verb].strip()
                break
    crit_pending = False
    for l in lines[1:] if m else lines:  # lines already trimmed for custom headers
        if (a := ATTACKED.match(l)):
            out["attacks"] += 1
            out["targets"].add(a["target"])
            if a["miss"]:
                out["misses"] += 1
            else:
                out["hits"] += 1
                crit_pending = a["res"] == "crit"
                out["crits"] += crit_pending
        elif (s := SAVE.match(l)):
            out["saves"] += 1
            out["saves_failed"] += bool(s["fail"])
            out["save_ability"] = s["ability"]
        elif (t := TOOK.match(l)):
            n = int(t["n"])
            out["targets"].add(t["target"])
            typ, amt, crit = ("damage", n, crit_pending) if n >= 0 else ("healing", -n, False)
            add_event(out, t["target"], typ, amt, crit)
            crit_pending = False
        elif (h := HEALED.match(l)):
            # The bot logs healing as negative damage ("healed for -8 health").
            out["targets"].add(h["target"])
            add_event(out, h["target"], "healing", abs(int(h["n"])), False)
        elif (g := GAINED.match(l)):
            if not TEMP.match(g["effect"]):
                out["effects"].append(g["effect"])
    return out


def main(src, dst):
    os.makedirs(dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, "*.jsonl")))
    A = ["session_id", "turn", "action_no", "speaker_id", "actor_name", "actor_kind", "actor_main_class",
         "actor_level", "actor_multiclass", "actor_class_raw", "actor_race", "actor_hp", "actor_max_hp",
         "actor_hp_status", "verb", "action_name", "n_targets", "attacks", "hits", "crits", "misses",
         "saves", "saves_failed", "save_ability", "damage", "healing", "effects_gained", "combatants"]
    D = ["session_id", "turn", "action_no", "actor_name", "target_name", "type", "amount", "on_crit", "outlier"]
    C = ["session_id", "name", "kind", "main_class", "level", "multiclass", "class_raw", "race", "max_hp",
         "controller_id", "first_turn", "turns_present"]
    S = ["session_id", "source_file", "turns", "actions", "players", "characters", "creatures", "max_combatants",
         "total_damage", "total_healing"]
    fa = open(os.path.join(dst, "fireball_actions.csv"), "w", newline="")
    fd = open(os.path.join(dst, "fireball_damage.csv"), "w", newline="")
    wa, wd = csv.DictWriter(fa, A), csv.DictWriter(fd, D)
    wa.writeheader(); wd.writeheader()
    combatants, sessions = [], []
    def flush(sid, file_id, seen, players, turns, n_actions, dmg_total, heal_total, max_comb):
        if not turns:
            return
        combatants.extend(seen.values())
        kinds = [c["kind"] for c in seen.values()]
        sessions.append({"session_id": sid, "source_file": file_id, "turns": turns, "actions": n_actions,
                         "players": len(players), "characters": kinds.count("character"),
                         "creatures": kinds.count("creature"), "max_combatants": max_comb,
                         "total_damage": dmg_total, "total_healing": heal_total})

    for path in files:
        file_id = os.path.basename(path).split(".")[0]
        n_sess, last_idx = 0, None
        sid = None
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            idx = r.get("before_state_idx")
            # Each file holds many games. The raw-event index restarts when a new game begins.
            if sid is None or (idx is not None and last_idx is not None and idx < last_idx):
                if sid is not None:
                    flush(sid, file_id, seen, players, turns, n_actions, dmg_total, heal_total, max_comb)
                n_sess += 1
                sid = f"{file_id[:12]}-{n_sess:02d}"
                seen, players = {}, set()
                n_actions = dmg_total = heal_total = max_comb = 0
                turns = 0
            if idx is not None:
                last_idx = idx
            turn = turns
            turns += 1
            players.add(r["speaker_id"])
            state = r.get("combat_state_after") or r.get("combat_state_before") or []
            max_comb = max(max_comb, len(state))
            for a in state:
                f = actor_fields(a)
                key = f["name"]
                if key not in seen:
                    seen[key] = {"session_id": sid, "first_turn": turn, "turns_present": 0,
                                 **{k: f[k] for k in ("name", "kind", "main_class", "level", "multiclass",
                                                      "class_raw", "race", "max_hp", "controller_id")}}
                seen[key]["turns_present"] += 1
            actor = actor_fields(r.get("caster_after"))
            for i, res in enumerate(r["automation_results"]):
                p = parse_result(res)
                if p["verb"] == "other" and not (p["attacks"] or p["saves"] or p["damage"] or p["healing"] or p["effects"]):
                    continue
                n_actions += 1
                dmg_total += p["damage"]; heal_total += p["healing"]
                wa.writerow({"session_id": sid, "turn": turn, "action_no": i, "speaker_id": r["speaker_id"],
                             "actor_name": actor["name"], "actor_kind": actor["kind"],
                             "actor_main_class": actor["main_class"], "actor_level": actor["level"],
                             "actor_multiclass": actor["multiclass"], "actor_class_raw": actor["class_raw"],
                             "actor_race": actor["race"], "actor_hp": actor["hp"], "actor_max_hp": actor["max_hp"],
                             "actor_hp_status": actor["hp_status"], "verb": p["verb"],
                             "action_name": p["action_name"], "n_targets": len(p["targets"]),
                             "attacks": p["attacks"], "hits": p["hits"], "crits": p["crits"],
                             "misses": p["misses"], "saves": p["saves"], "saves_failed": p["saves_failed"],
                             "save_ability": p["save_ability"], "damage": p["damage"], "healing": p["healing"],
                             "effects_gained": "; ".join(p["effects"]), "combatants": len(state)})
                for tgt, typ, n, crit, outlier in p["events"]:
                    wd.writerow({"session_id": sid, "turn": turn, "action_no": i, "actor_name": actor["name"],
                                 "target_name": tgt, "type": typ, "amount": n, "on_crit": crit,
                                 "outlier": outlier})
        flush(sid, file_id, seen, players, turns, n_actions, dmg_total, heal_total, max_comb)
    fa.close(); fd.close()
    for name, cols, rows in (("fireball_combatants.csv", C, combatants), ("fireball_sessions.csv", S, sessions)):
        with open(os.path.join(dst, name), "w", newline="") as f:
            w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
