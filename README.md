# dnd-lab-data

Clean, flat D&D 5e datasets for analysis. One row per item, ready for SQL, pandas or a BI tool.

Part of a side project exploring D&D data: what players build, what the rules allow, and what happens at the table.

## Contents

| File | Rows | What it is |
| --- | --- | --- |
| `dnd_chars_unique.tsv` | 7,946 | Player characters submitted to a fan-made character-sheet app, Apr 2018 to Aug 2022 |
| `srd_spells.csv` | 319 | Spells from the 2014 SRD |
| `srd_monsters.csv` | 334 | Monsters from the 2014 SRD |
| `srd_classes.csv` | 12 | Classes from the 2014 SRD |
| `srd_races.csv` | 9 | Races from the 2014 SRD |
| `flatten_srd.py` | | Turns the SRD JSON files into the CSVs above |
| `fireball_sessions.csv` | 8,749 | Real D&D combat sessions played on Discord with the Avrae bot |
| `fireball_actions.csv` | 106,776 | Every attack, spell and ability used in those sessions, with its outcome |
| `fireball_damage.csv` | 89,453 | Every damage or healing event, one row per target |
| `fireball_combatants.csv` | 75,049 | Every character and creature in each session |
| `flatten_fireball.py` | | Builds the four FIREBALL tables from the raw dataset |

## Characters

Source: [oganm/dnddata](https://github.com/oganm/dnddata), unique-character version. Tab-separated.

Useful columns:

- `processedRace`: cleaned race (use this, not `race`)
- `justClass`: class, with multiclass characters joined by `|` (for example `Rogue|Fighter`)
- `level`: total character level
- `Str`, `Dex`, `Con`, `Int`, `Wis`, `Cha`, `HP`, `AC`
- `processedSpells`, `feats`, `skills`, `background`, `processedAlignment`, `country`
- `name`: a hash, not the real name

Known limits:

- Self-selected users of one app, not all D&D players
- About 22% of rows are from Canada and 29% have no country
- 26% of characters are level 1, so this mostly shows new characters
- 181 rows have no race

## SRD tables

Source: [5e-bits/5e-database](https://github.com/5e-bits/5e-database), `src/2014/en`. Long description text is left out.

Lists that don't fit one cell are joined with `; ` (for example `Sorcerer; Wizard`).

| File | Columns |
| --- | --- |
| `srd_spells.csv` | name, level, school, casting_time, range, duration, concentration, ritual, components, classes, damage_type, base_damage, attack_type, save, heals |
| `srd_monsters.csv` | name, size, type, alignment, armor_class, hit_points, hit_dice, challenge_rating, xp, speed_walk, speed_fly, legendary, str, dex, con, int, wis, cha |
| `srd_classes.csv` | name, hit_die, saving_throws, spellcasting_ability, subclasses |
| `srd_races.csv` | name, size, speed, ability_bonuses, subraces |

`base_damage` is the damage at the spell's lowest slot level, or at character level 1 for cantrips.

To rebuild:

```bash
git clone https://github.com/5e-bits/5e-database.git
python flatten_srd.py 5e-database/src/2014/en .
```

## FIREBALL: real game sessions

Source: [FIREBALL](https://huggingface.co/datasets/lara-martin/FIREBALL) (Zhu et al., ACL 2023), the published `filtered/` version: 1,000 files, 105,068 turns. Players and characters are anonymised in the source.

How the tables were built:

- Each source file bundles many games. A new session starts where the raw-event index resets, giving 8,749 sessions (median 3 players, 8 turns).
- Actions are parsed from the Avrae bot's result text, which is machine-generated and consistent, not from the players' commands.
- `actor_kind` is `character` when the actor has a class, otherwise `creature`. For multiclass characters, `actor_main_class` is the class with the most levels and `actor_level` is the total.
- `verb` is `attack`, `spell`, `ability`, or `custom` for homebrew actions with their own wording (the text is kept in `action_name`).
- Damage or healing over 1,000 in one hit (45 events, mostly jokes or "kill my character" tricks) is marked `outlier` in `fireball_damage.csv` and left out of all totals. Two impossible values are blank.

Tables join on `session_id`. Actions and damage also join on `turn` and `action_no`.

| File | Columns |
| --- | --- |
| `fireball_sessions.csv` | session_id, source_file, turns, actions, players, characters, creatures, max_combatants, total_damage, total_healing |
| `fireball_actions.csv` | session_id, turn, action_no, speaker_id, actor_name, actor_kind, actor_main_class, actor_level, actor_multiclass, actor_class_raw, actor_race, actor_hp, actor_max_hp, actor_hp_status, verb, action_name, n_targets, attacks, hits, crits, misses, saves, saves_failed, save_ability, damage, healing, effects_gained, combatants |
| `fireball_damage.csv` | session_id, turn, action_no, actor_name, target_name, type, amount, on_crit, outlier |
| `fireball_combatants.csv` | session_id, name, kind, main_class, level, multiclass, class_raw, race, max_hp, controller_id, first_turn, turns_present |

Known limits: only fights where players used the bot, so roleplay-only sessions are missing; homebrew content is mixed in; one Discord community can contribute many sessions.

To rebuild, download the `filtered/` folder from Hugging Face, then:

```bash
python flatten_fireball.py path/to/filtered fireball_tables
```

## Licences and attribution

- **Characters**: from oganm/dnddata, Copyright (c) 2019 Burak Ogan Mancarci, MIT License. The full notice is in `LICENSE-dnddata`.
- **SRD tables**: This work includes material taken from the System Reference Document 5.1 ("SRD 5.1") by Wizards of the Coast LLC, available at https://dnd.wizards.com/resources/systems-reference-document. The SRD 5.1 is licensed under the Creative Commons Attribution 4.0 International License, available at https://creativecommons.org/licenses/by/4.0/legalcode. Data structured by the 5e-bits/5e-database project (MIT).

- **FIREBALL tables**: derived from FIREBALL by Andrew Zhu, Karmanya Aggarwal, Alexander Feng, Lara J. Martin and Chris Callison-Burch, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Changes: flattened into four tables, games split into sessions, outlier values flagged. Paper: Zhu et al. (2023), "FIREBALL: A Dataset of Dungeons and Dragons Actual-Play with Structured Game State Information", ACL 2023, https://aclanthology.org/2023.acl-long.229/

Only open-licence SRD content is included. No text from published books.
