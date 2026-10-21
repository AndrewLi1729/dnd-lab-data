# dnd-lab-data

Clean, flat D&D 5e datasets for analysis. One row per item, ready for SQL, pandas or a BI tool.

Part of a side project exploring D&D data: what players build, what the rules allow, and what happens at the table.

## Contents

| Folder | File | Rows | What it is |
| --- | --- | --- | --- |
| `characters/` | `dnd_chars_unique.tsv` | 7,946 | Player characters submitted to a fan-made character-sheet app, Apr 2018 to Aug 2022 |
| `srd/2014/` | `srd_spells.csv` | 319 | Spells from the 2014 SRD |
| `srd/2014/` | `srd_monsters.csv` | 334 | Monsters from the 2014 SRD |
| `srd/2014/` | `srd_classes.csv` | 12 | Classes from the 2014 SRD |
| `srd/2014/` | `srd_races.csv` | 9 | Races from the 2014 SRD |
| `scripts/` | `flatten_srd.py` | | Turns the SRD JSON files into the CSVs above |

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
python scripts/flatten_srd.py 5e-database/src/2014/en srd/2014
```

## Licences and attribution

- **Characters**: from oganm/dnddata, Copyright (c) 2019 Burak Ogan Mancarci, MIT License. The full notice is in `characters/LICENSE`.
- **SRD tables**: This work includes material taken from the System Reference Document 5.1 ("SRD 5.1") by Wizards of the Coast LLC, available at https://dnd.wizards.com/resources/systems-reference-document. The SRD 5.1 is licensed under the Creative Commons Attribution 4.0 International License, available at https://creativecommons.org/licenses/by/4.0/legalcode. Data structured by the 5e-bits/5e-database project (MIT).

Only open-licence SRD content is included. No text from published books.
