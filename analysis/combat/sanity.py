"""Plausibility check against rule expectations (2014 rules, from memory, not looked up).

Usage: python sanity.py <out_dir>   -> writes sanity.csv
Expectations for single-class level 5-10 characters:
  Fighter per hit: longsword d8+4 (+2 dueling) = 10.5, greatsword 2d6+4 = 11; crits and feats push it higher
  Barbarian per hit: greataxe d12+4 + rage 2 = 12.5; greatsword 2d6+4+2 = 13
  Attacks per Attack action from level 5 (Extra Attack): 2
"""
import sys
from pathlib import Path

import pandas as pd

RULES = {"Fighter": (10.5, 15.0), "Barbarian": (12.5, 15.0)}


def main(out_dir):
    out = Path(out_dir)
    c = pd.read_csv(out / "char_actions.csv", low_memory=False)
    rows = []
    for cls, (lo, hi) in RULES.items():
        g = c[(c.actor_main_class == cls) & (c.tier == "5-10") & (c.verb == "attack") & ~c.actor_multiclass.astype(bool)
              & (c.attacks > 0)]
        per_hit = g.damage.sum() / g.hits.sum()
        rows.append(dict(main_class=cls, attack_actions=len(g), attacks_per_action=g.attacks.mean(),
                         expected_attacks_per_action=2, hit_rate=g.hits.sum() / g.attacks.sum(),
                         crit_rate=g.crits.sum() / g.attacks.sum(), damage_per_hit=per_hit,
                         rule_low=lo, rule_high=hi, within_rule_range=lo <= per_hit <= hi))
    r = pd.DataFrame(rows)
    r.to_csv(out / "sanity.csv", index=False)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
