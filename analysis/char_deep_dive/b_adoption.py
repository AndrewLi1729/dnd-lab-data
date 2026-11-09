"""Topic B: adoption of new book content and the Human share.

Usage: python b_adoption.py <raw.tsv> <chars_clean.csv> <out_dir>
Writes b_monthly.csv (monthly shares), b_adoption.csv (per book), b_its.csv (segmented regression).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from refs import TASHA, load_releases  # noqa: E402


def monthly(c: pd.DataFrame, rel: pd.DataFrame) -> pd.DataFrame:
    c = c.copy()
    c["month"] = c.date.dt.tz_convert(None).dt.to_period("M")
    sub = c.subclass.fillna("")
    for r in rel.itertuples():
        names = [s for s in r.adds_subclasses.split(";") if s]
        c[f"sub_{r.code}"] = sub.apply(lambda s: any(n in s for n in names)) if names else False
    has_sub = sub != ""
    low = c.level <= 3
    feat_early = c.feats.notna() & low
    c["human"] = c.race_clean.eq("Human")
    c["human_variant_l3"] = c.human & feat_early
    c["human_standard_l3"] = c.human & low & ~feat_early
    c["custom"] = c.race_clean.eq("Custom")
    g = c.groupby("month")
    out = pd.DataFrame({"n": g.size(), "n_low": g.apply(lambda x: (x.level <= 3).sum()),
                        "n_with_subclass": g.apply(lambda x: (x.subclass.fillna("") != "").sum())})
    out["human_share"] = g.human.mean()
    out["custom_share"] = g.custom.mean()
    out["human_variant_share_l3"] = g.human_variant_l3.sum() / out.n_low
    out["human_standard_share_l3"] = g.human_standard_l3.sum() / out.n_low
    out["custom_share_l3"] = g.apply(lambda x: (x.custom & (x.level <= 3)).sum()) / out.n_low
    for r in rel.itertuples():
        out[f"sub_{r.code}"] = c[has_sub].groupby("month")[f"sub_{r.code}"].mean()
    return out.reset_index()


def adoption(m: pd.DataFrame, rel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in rel.itertuples():
        col = f"sub_{r.code}"
        if not r.adds_subclasses:
            continue
        rm = pd.Period(r.date.tz_convert(None), "M")
        k = (m.month - rm).apply(lambda x: x.n)
        pre = m.loc[k < 0, col].mean()
        post = {f"m{h}": m.loc[k.between(h - 1, h + 1), col].mean() for h in [1, 3, 6, 12]}
        plateau = m.loc[k.between(12, 24), col].mean()
        first = m.loc[(m[col] > 0), "month"].min()
        half = m.loc[(k >= 0) & (m[col] >= plateau / 2), "month"].min() if plateau > 0 else None
        rows.append(dict(book=r.code, release=r.date.date(), first_seen=str(first), pre_share=pre, **post,
                         plateau_12_24=plateau,
                         months_to_half=(half - rm).n if half is not None and pd.notna(half) else None))
    return pd.DataFrame(rows)


def its(m: pd.DataFrame, col: str, when: pd.Timestamp, label: str) -> dict:
    """Segmented regression with Newey-West (HAC, 3 lags) errors, weighted by monthly n."""
    d = m.dropna(subset=[col]).copy()
    t = np.arange(len(d))
    rm = pd.Period(when.tz_convert(None), "M")
    post = (d.month >= rm).astype(int).values
    t_post = np.where(post == 1, t - t[post.argmax()], 0)
    X = sm.add_constant(np.column_stack([t, post, t_post]))
    fit = sm.WLS(d[col].values, X, weights=d.n.values).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    names = ["const", "trend", "level_change", "slope_change"]
    res = {"series": col, "event": label}
    for i, nm in enumerate(names):
        res[nm] = fit.params[i]
        res[nm + "_p"] = fit.pvalues[i]
    res["pre_mean"] = d.loc[post == 0, col].mean()
    res["post_mean"] = d.loc[post == 1, col].mean()
    return res


def main(raw_p, clean_p, out_dir):
    raw = pd.read_csv(raw_p, sep="\t", low_memory=False)
    c = pd.read_csv(clean_p).merge(raw[["hash", "feats"]], left_on="char_id", right_on="hash")
    c["date"] = pd.to_datetime(c["date"], utc=True)
    rel = load_releases()
    m = monthly(c, rel)
    a = adoption(m, rel)
    tests = [its(m, "human_share", TASHA, "Tasha"),
             its(m, "human_variant_share_l3", TASHA, "Tasha"),
             its(m, "human_standard_share_l3", TASHA, "Tasha"),
             its(m, "custom_share", TASHA, "Tasha"),
             its(m, "sub_TCoE", TASHA, "Tasha")]
    out = Path(out_dir)
    m.assign(month=m.month.astype(str)).to_csv(out / "b_monthly.csv", index=False)
    a.to_csv(out / "b_adoption.csv", index=False)
    pd.DataFrame(tests).to_csv(out / "b_its.csv", index=False)
    pd.set_option("display.width", 200)
    print(a.round(3).to_string(index=False))
    print(pd.DataFrame(tests)[["series", "pre_mean", "post_mean", "level_change", "level_change_p",
                               "slope_change", "slope_change_p", "trend", "trend_p"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
