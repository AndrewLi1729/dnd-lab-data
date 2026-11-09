"""Topic A: rule compliance of ability scores (levels 1-3) and hit points (all levels).

Usage: python a_rules.py <raw.tsv> <chars_clean.csv> <char_classes_long.csv> <out_dir>
Writes a_abilities.csv, a_hp.csv (one row per character) and prints summaries.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from refs import ABILITIES, HALF_FEATS, TASHA, load_asi, match_rows, options, placements  # noqa: E402

STD_ARRAY = (8, 10, 12, 13, 14, 15)
PB_COST = {8: 0, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 7, 15: 9}
HIT_DIE = {"Barbarian": 12, "Fighter": 10, "Paladin": 10, "Ranger": 10, "Bard": 8, "Cleric": 8,
           "Druid": 8, "Monk": 8, "Rogue": 8, "Warlock": 8, "Artificer": 8, "Sorcerer": 6,
           "Wizard": 6}
ORDER = ["standard_array", "point_buy", "rolled_plausible", "beyond_rules"]


def classify_scores(scores: tuple, bonus_vectors) -> tuple[str, tuple]:
    """Best (most rule-abiding) category over all legal bonus placements."""
    best, best_base = 3, None
    for b in bonus_vectors:
        base = tuple(s - x for s, x in zip(scores, b))
        if tuple(sorted(base)) == STD_ARRAY:
            return "standard_array", base
        if all(8 <= v <= 15 for v in base) and sum(PB_COST[v] for v in base) <= 27:
            if best > 1:
                best, best_base = 1, base
        elif all(3 <= v <= 18 for v in base) and best > 2:
            best, best_base = 2, base
    return ORDER[best], best_base


def abilities(c: pd.DataFrame, asi: pd.DataFrame) -> pd.DataFrame:
    d = c[(c.level <= 3) & c.race_clean.notna() & ~c.stats_suspect].copy()
    out = []
    for r in d.itertuples():
        rows, guessed = match_rows(asi, r.race_clean, r.race_raw if isinstance(r.race_raw, str) else "")
        feats = set(str(r.feats).split("|")) if isinstance(r.feats, str) else set()
        half = bool(feats & HALF_FEATS)
        after = r.date >= TASHA
        vecs = {v for o in options(rows, after, half) for v in placements(*o)}
        scores = tuple(int(getattr(r, a)) for a in ABILITIES)
        cat, base = classify_scores(scores, vecs)
        out.append((r.char_id, cat, guessed, after, half, sum(scores), base))
    res = pd.DataFrame(out, columns=["char_id", "ability_class", "subrace_guessed", "after_tasha",
                                     "half_feat", "score_total", "base_scores"])
    assert len(res) == len(d), "every eligible character must be classified"
    return res


def hp(c: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """Min / average / max legal HP from class levels, Con and known HP bonuses."""
    lv = long.merge(c[["char_id", "class"]], on="char_id")
    lv = lv[lv["class_x"].isin(HIT_DIE)].rename(columns={"class_x": "cls"})
    # starting class = first listed in the raw `class` string
    first = c.set_index("char_id")["class"].str.split(r"\s\d").str[0].str.strip()
    lv["is_first"] = lv.cls.eq(lv.char_id.map(first))
    lv["die"] = lv.cls.map(HIT_DIE)
    lv["first_lv"] = lv.is_first.astype(int)  # one level at max die
    lv["rest"] = lv.class_level - lv.first_lv
    agg = lv.assign(
        mn=lv.first_lv * lv.die + lv.rest * 1,
        av=lv.first_lv * lv.die + lv.rest * (lv.die // 2 + 1),
        mx=lv.class_level * lv.die,
        sorc=np.where(lv.cls == "Sorcerer", lv.class_level, 0),
    ).groupby("char_id")[["mn", "av", "mx", "sorc", "class_level"]].sum()
    d = c.set_index("char_id").join(agg, how="inner")
    d = d[d.class_level == d.level]  # drop characters with non-official classes
    con = (d.Con - 10) // 2
    feats = d.feats.fillna("")
    sub = d.subclass.fillna("")
    extra = (d.level * 2 * feats.str.contains("Tough")
             + d.level * d.race_raw.fillna("").str.contains("Hill", case=False)
             + d.sorc * sub.str.contains("Draconic"))
    for k in ["mn", "av", "mx"]:
        d[k] = (d[k] + d.level * con + extra).clip(lower=d.level)  # at least 1 HP per level
    d["hp_class"] = np.select(
        [d.HP > d.mx, d.HP == d.av, (d.HP > d.av) & (d.HP <= d.mx), (d.HP < d.av) & (d.HP >= d.mn)],
        ["above_max", "average", "between_avg_max", "below_avg"], "below_min")
    d.loc[(d.level == 1) & (d.HP == d.mx), "hp_class"] = "average"  # level 1 is fixed: max die
    d["hp_ratio"] = d.HP / d.av
    return d.reset_index()[["char_id", "level", "is_multiclass", "HP", "mn", "av", "mx", "hp_ratio", "hp_class"]]


def dice_benchmark(a: pd.DataFrame, c: pd.DataFrame, n=200_000, seed=1) -> pd.DataFrame:
    """Compare 'rolled' base scores with 4d6-drop-lowest pushed through the same classifier.

    Uses fixed-bonus races before Tasha with no half-feat, so the base scores are unambiguous.
    """
    import ast
    rng = np.random.default_rng(seed)
    floating = {"Human", "Half-Elf", "Custom", "Warforged", "Changeling", "Simic hybrid"}
    d = a.merge(c[["char_id", "race_clean"]], on="char_id")
    keep = ~d.race_clean.isin(floating) & ~d.after_tasha & ~d.half_feat & ~d.subrace_guessed
    obs = d[keep & (d.ability_class == "rolled_plausible")].base_scores.apply(lambda v: ast.literal_eval(v) if isinstance(v, str) else v)
    rows = [dict(source="observed rolled class", n=len(obs), median_sum=obs.apply(sum).median(),
                 share_with_18=(obs.apply(max) == 18).mean())]

    def classify_rolled(b):
        inpb = ((b >= 8) & (b <= 15)).all(1)
        cost = np.vectorize(lambda v: PB_COST.get(v, 99))(b).sum(1)
        return ~(inpb & (cost <= 27))

    for k in [1, 2, 3]:
        dice = rng.integers(1, 7, (n * k, 6, 4))
        sets = (dice.sum(2) - dice.min(2)).reshape(n, k, 6)
        best = sets[np.arange(n), sets.sum(2).argmax(1)]
        r = classify_rolled(best)
        rows.append(dict(source=f"4d6 drop lowest, best of {k} set(s), same classifier", n=int(r.sum()),
                         median_sum=float(np.median(best[r].sum(1))), share_with_18=(best[r].max(1) == 18).mean()))
    return pd.DataFrame(rows)


def generosity(a: pd.DataFrame, h: pd.DataFrame, c: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ability class x HP class (levels 2-3), and per-device consistency of ability method."""
    from scipy import stats
    d = a.merge(h[["char_id", "hp_class"]], on="char_id").merge(c[["char_id", "level", "device_id"]], on="char_id")
    d["by_book"] = d.ability_class.isin(["standard_array", "point_buy"])
    d["generous_hp"] = d.hp_class.isin(["between_avg_max", "above_max"])
    e = d[d.level >= 2]
    t = pd.crosstab(~e.by_book, e.generous_hp).values
    odds, p = stats.fisher_exact(t)
    gen = pd.DataFrame([dict(n=len(e), generous_if_rolled=e[~e.by_book].generous_hp.mean(),
                             generous_if_by_book=e[e.by_book].generous_hp.mean(), odds_ratio=odds, p=p)])
    m = d[d.device_id.notna()].groupby("device_id").filter(lambda g: len(g) >= 3)
    share = m.groupby("device_id").by_book.mean()
    size = m.groupby("device_id").size()
    q = d.by_book.mean()
    dev = pd.DataFrame([dict(devices=len(share),
                             always_by_book=(share == 1).mean(), always_rolled=(share == 0).mean(),
                             mixed=((share > 0) & (share < 1)).mean(),
                             expected_always_by_book=(q ** size).mean(),
                             expected_always_rolled=((1 - q) ** size).mean())])
    return gen, dev


def main(raw_p, clean_p, long_p, out_dir):
    raw = pd.read_csv(raw_p, sep="\t", low_memory=False)
    c = pd.read_csv(clean_p).merge(
        raw[["hash"] + ABILITIES + ["HP", "feats"]], left_on="char_id", right_on="hash")
    c["date"] = pd.to_datetime(c["date"], utc=True)
    long = pd.read_csv(long_p)
    out = Path(out_dir)

    a = abilities(c, load_asi())
    a.to_csv(out / "a_abilities.csv", index=False)
    h = hp(c[~c.stats_suspect], long)
    h.to_csv(out / "a_hp.csv", index=False)
    bench = dice_benchmark(a, c)
    bench.to_csv(out / "a_dice_benchmark.csv", index=False)
    gen, dev = generosity(a, h, c)
    gen.to_csv(out / "a_generosity.csv", index=False)
    dev.to_csv(out / "a_device_habit.csv", index=False)
    print(bench.round(3).to_string(index=False))
    print(gen.round(4).to_string(index=False))
    print(dev.round(3).to_string(index=False))
    print("abilities n =", len(a))
    print((a.ability_class.value_counts(normalize=True) * 100).round(1).to_dict())
    print("hp n =", len(h))
    print((h.hp_class.value_counts(normalize=True) * 100).round(1).to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:5])
