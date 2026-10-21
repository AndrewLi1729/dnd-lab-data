import json, sys, os, pandas as pd
src, out = sys.argv[1], sys.argv[2]
L = lambda f: json.load(open(os.path.join(src, f'5e-SRD-{f}.json')))
names = lambda xs: '; '.join(x['name'] for x in (xs or []))
sp = []
for s in L('Spells'):
    ds = s.get('damage') or []
    ds = ds if isinstance(ds, list) else [ds]
    dtype = '; '.join(x['damage_type']['name'] for x in ds if 'damage_type' in x)
    def base(x):
        for k in ('damage_at_slot_level', 'damage_at_character_level'):
            if k in x: v = x[k]; return v[min(v, key=int)]
        return ''
    dice = '; '.join(base(x) for x in ds if base(x))
    sp.append({'name': s['name'], 'level': s['level'], 'school': s['school']['name'],
      'casting_time': s.get('casting_time'), 'range': s.get('range'), 'duration': s.get('duration'),
      'concentration': s.get('concentration'), 'ritual': s.get('ritual'),
      'components': ''.join(s.get('components', [])), 'classes': names(s.get('classes')),
      'damage_type': dtype, 'base_damage': dice, 'attack_type': s.get('attack_type', ''),
      'save': s.get('dc', {}).get('dc_type', {}).get('name', ''),
      'heals': 'heal_at_slot_level' in s})
pd.DataFrame(sp).to_csv(f'{out}/srd_spells.csv', index=False)
mo = []
for m in L('Monsters'):
    ac = m.get('armor_class'); ac = ac[0]['value'] if isinstance(ac, list) else ac
    r = {'name': m['name'], 'size': m.get('size'), 'type': m.get('type'), 'alignment': m.get('alignment'),
      'armor_class': ac, 'hit_points': m.get('hit_points'), 'hit_dice': m.get('hit_dice'),
      'challenge_rating': m.get('challenge_rating'), 'xp': m.get('xp'),
      'speed_walk': m.get('speed', {}).get('walk', ''), 'speed_fly': m.get('speed', {}).get('fly', ''),
      'legendary': bool(m.get('legendary_actions'))}
    for a in ['strength','dexterity','constitution','intelligence','wisdom','charisma']: r[a[:3]] = m.get(a)
    mo.append(r)
pd.DataFrame(mo).to_csv(f'{out}/srd_monsters.csv', index=False)
cl = [{'name': c['name'], 'hit_die': c['hit_die'], 'saving_throws': names(c.get('saving_throws')),
       'spellcasting_ability': c.get('spellcasting', {}).get('spellcasting_ability', {}).get('name', ''),
       'subclasses': names(c.get('subclasses'))} for c in L('Classes')]
pd.DataFrame(cl).to_csv(f'{out}/srd_classes.csv', index=False)
ra = [{'name': r['name'], 'size': r['size'], 'speed': r['speed'],
       'ability_bonuses': '; '.join(f"{b['ability_score']['name']} +{b['bonus']}" for b in r.get('ability_bonuses', [])),
       'subraces': names(r.get('subraces'))} for r in L('Races')]
pd.DataFrame(ra).to_csv(f'{out}/srd_races.csv', index=False)
for f in sorted(os.listdir(out)):
    df = pd.read_csv(f'{out}/{f}'); print(f, df.shape); print(df.head(2).to_string()[:400]); print()
