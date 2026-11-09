"""S4: do casters of one class fall into stable 'schools' of spell choice?

Usage: python s4_schools.py <out_dir> [n_boot]
Per class: characters level 5+ with 6+ spells; spells learned by >= 5% of them; Jaccard distance,
average-linkage hierarchical clustering, k = 2..6 by silhouette. Stability = Hennig clusterboot:
mean best Jaccard match of each cluster across bootstrap reclusterings (> 0.75 stable, < 0.6 dissolved).
Writes s4_clusters.csv (one row per cluster) and s4_classes.csv (one row per class).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

MIN_CHARS = 100


def cluster(X: np.ndarray, k: int) -> np.ndarray:
    Z = linkage(pdist(X, "jaccard"), "average")
    return fcluster(Z, k, "maxclust")


def best_k(X: np.ndarray):
    D = squareform(pdist(X, "jaccard"))
    best = (None, -1, None)
    for k in range(2, 7):
        lab = cluster(X, k)
        if len(set(lab)) < 2 or min(np.bincount(lab)[1:]) < 10:
            continue
        s = silhouette_score(D, lab, metric="precomputed")
        if s > best[1]:
            best = (k, s, lab)
    return best


def stability(X, lab, k, n_boot, rng):
    n = len(X)
    clusters = [set(np.where(lab == c)[0]) for c in sorted(set(lab))]
    scores = np.zeros((n_boot, len(clusters)))
    for b in range(n_boot):
        idx = np.unique(rng.integers(0, n, n))
        bl = cluster(X[idx], k)
        bc = [set(idx[bl == c]) for c in set(bl)]
        for i, c in enumerate(clusters):
            ci = c & set(idx)
            if not ci:
                continue
            scores[b, i] = max(len(ci & d) / len(ci | d) for d in bc)
    return scores.mean(0)


def main(out_dir, n_boot=200, seed=11):
    out = Path(out_dir)
    L = pd.read_csv(out / "learned.csv")
    sub = pd.read_csv(out / "chars_clean.csv").set_index("char_id").subclass.fillna("?")
    L = L[L.level >= 5]
    size = L.groupby("char_id").spell.nunique()
    L = L[L.char_id.isin(size[size >= 6].index)]
    rng = np.random.default_rng(seed)
    crow, kro = [], []
    for cls, g in L.groupby("main_class"):
        n_chars = g.char_id.nunique()
        if n_chars < MIN_CHARS:
            kro.append(dict(main_class=cls, n_chars=n_chars, k=None, silhouette=None, note="too few characters"))
            continue
        M = pd.crosstab(g.char_id, g.spell).clip(upper=1)
        M = M.loc[:, M.mean() >= 0.05]
        M = M[M.sum(axis=1) > 0]
        X = M.values.astype(bool)
        # sensitivity: spectral clustering on Jaccard similarity, k = 3
        D = squareform(pdist(X, "jaccard"))
        sp = SpectralClustering(3, affinity="precomputed", random_state=0).fit_predict(1 - D)
        sp_purity = [sub.reindex(M.index[sp == c]).value_counts(normalize=True).iloc[0] for c in set(sp)]
        sp_info = dict(spectral_k3_silhouette=silhouette_score(D, sp, metric="precomputed"),
                       spectral_k3_sizes=sorted(np.bincount(sp).tolist(), reverse=True),
                       spectral_k3_min_purity=min(sp_purity))
        k, sil, lab = best_k(X)
        if k is None:
            kro.append(dict(main_class=cls, n_chars=n_chars, k=None, silhouette=None, note="no k with clusters >= 10", **sp_info))
            continue
        stab = stability(X, lab, k, n_boot, rng)
        base = M.mean()
        for i, c in enumerate(sorted(set(lab))):
            members = M[lab == c]
            lift = (members.mean() - base).sort_values(ascending=False)
            sc = sub.reindex(members.index).str.split("|").str[0].value_counts(normalize=True)
            crow.append(dict(main_class=cls, cluster=int(c), size=len(members), share=len(members) / len(M),
                             top_subclass=sc.index[0], top_subclass_share=sc.iloc[0],
                             stability=stab[i], stable=stab[i] > 0.75, dissolved=stab[i] < 0.6,
                             signature=", ".join(f"{s} (+{v:.0%})" for s, v in lift.head(5).items())))
        kro.append(dict(main_class=cls, n_chars=len(M), n_spells=M.shape[1], k=k, silhouette=sil,
                        stable_clusters=int(sum(stab > 0.75)), note="", **sp_info))
    cr, kr = pd.DataFrame(crow), pd.DataFrame(kro)
    cr.to_csv(out / "s4_clusters.csv", index=False)
    kr.to_csv(out / "s4_classes.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print(kr.round(3).to_string(index=False))
    print(cr.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:3]))
