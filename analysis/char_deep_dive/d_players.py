"""Topic D: are players (devices) loyal to a race/class or do they try new things?

Usage: python d_players.py <chars_clean.csv> <out_dir> [n_reps]
Compares observed diversity per device with random draws of the same size from all characters.
Writes d_players.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def distinct_ratio(values: np.ndarray) -> float:
    return len(set(values)) / len(values)


def main(clean_p, out_dir, reps=1000):
    reps = int(reps)
    c = pd.read_csv(clean_p)
    c = c[c.device_id.notna() & c.race_clean.notna() & (c.main_class != "Other")]
    sizes = c.groupby("device_id").size()
    devs = sizes[sizes >= 3].index
    d = c[c.device_id.isin(devs)]
    rng = np.random.default_rng(42)
    rows = []
    for col in ["race_clean", "main_class", "class_combo"]:
        obs = d.groupby("device_id")[col].apply(lambda s: distinct_ratio(s.values))
        pool = c[col].values
        n = sizes[devs].values
        sims = np.array([np.mean([distinct_ratio(rng.choice(pool, k, replace=False)) for k in n]) for _ in range(reps)])
        p = (np.sum(sims <= obs.mean()) + 1) / (reps + 1)
        # share of devices that repeat their most common value at least half the time
        loyal = d.groupby("device_id")[col].apply(lambda s: s.value_counts().iloc[0] / len(s) >= 0.5).mean()
        rows.append(dict(attribute=col, devices=len(devs), chars=len(d), observed_distinct_ratio=obs.mean(),
                         random_distinct_ratio=sims.mean(), random_p05=np.quantile(sims, 0.05),
                         p_value_lower=p, share_devices_half_same=loyal))
    out = pd.DataFrame(rows)
    out.to_csv(Path(out_dir) / "d_players.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
