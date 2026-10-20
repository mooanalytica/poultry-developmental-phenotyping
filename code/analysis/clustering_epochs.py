"""Per-room k-means on 6 h window means to test whether clustering recovers the day-70 boundary."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import datasets as D
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 0
SENSORS = ("morph", "yolo", "flow", "audio_bio", "audio_equip")


def inputs():
    X, cols, _ = D.build("A", "fused", augment=False)
    feat = [f"{c}__m" for c in cols]
    return X, [c for c in feat if c in X.columns and c.split("__")[0] in SENSORS]


def room_space(X, FEAT, room):
    g = X[X.room == room].sort_values("hour")
    V = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(g[FEAT]))
    return g, PCA(n_components=20, random_state=SEED).fit_transform(V)


def best_split(ages, young):
    best, bd = -1, np.nan
    for t in np.arange(35, 175, 1.0):
        b = balanced_accuracy_score((ages < t).astype(int), young)
        if b > best:
            best, bd = b, t
    return bd, best


def transition_day(ages, is_early):
    d = pd.DataFrame({"age": ages, "early": is_early})
    sm = d.groupby(d.age.round()).early.mean().rolling(7, center=True, min_periods=3).mean()
    below = sm[(sm < 0.5) & (sm.index > 35)]
    return float(below.index[0]) if len(below) else np.nan


def early_labels(ages, lab):
    return (lab == int(pd.Series(ages).groupby(lab).median().idxmin())).astype(int)


def run(out):
    X, FEAT = inputs()
    rows, seeds, db_rows = [], [], []
    curves = []
    for room in sorted(X.room.unique()):
        g, Pc = room_space(X, FEAT, room)
        ages = g.age_days.values
        sil = {
            k: silhouette_score(Pc, KMeans(k, n_init=10, random_state=SEED).fit_predict(Pc))
            for k in range(2, 7)
        }
        kbest = max(sil, key=sil.get)
        lab2 = KMeans(2, n_init=10, random_state=SEED).fit_predict(Pc)
        young = early_labels(ages, lab2)
        bd, bacc = best_split(ages, young)
        d_ = pd.DataFrame({"age": ages, "early": young})
        daily = d_.groupby(d_.age.round()).early.mean()
        curves.append(
            pd.DataFrame(
                {
                    "room": room,
                    "age_day": daily.index,
                    "early_fraction": daily.values,
                    "early_fraction_smoothed": daily.rolling(7, center=True, min_periods=3)
                    .mean()
                    .values,
                    "n_windows": d_.groupby(d_.age.round()).size().values,
                }
            )
        )
        rows.append(
            dict(
                room=room,
                n=len(g),
                n_features=len(FEAT),
                k_best=kbest,
                silhouette_kbest=sil[kbest],
                k2_boundary_day=bd,
                k2_balanced_accuracy=bacc,
                k2_ARI_vs_day70=adjusted_rand_score((ages < 70).astype(int), young),
                k2_early_cluster_frac=float(young.mean()),
                primary_transition_day=transition_day(ages, young),
                **{f"silhouette_k{k}": v for k, v in sil.items()},
            )
        )
        for s in range(10):
            l = KMeans(2, n_init=10, random_state=s).fit_predict(Pc)
            seeds.append(
                dict(
                    room=room,
                    seed=s,
                    silhouette=silhouette_score(Pc, l),
                    transition_day=transition_day(ages, early_labels(ages, l)),
                )
            )
        best = None
        for eps in np.round(np.arange(1.0, 8.1, 0.5), 1):
            dl = DBSCAN(eps=eps, min_samples=20).fit_predict(Pc)
            n_cl = len(set(dl)) - (1 if -1 in dl else 0)
            noise = (dl == -1).mean()
            if 2 <= n_cl <= 6 and noise < 0.5:
                s_ = silhouette_score(Pc[dl != -1], dl[dl != -1]) if (dl != -1).sum() > 50 else -1
                if best is None or s_ > best[1]:
                    best = (eps, s_, n_cl, noise)
        db_rows.append(
            dict(
                room=room,
                dbscan_eps=best[0] if best else np.nan,
                dbscan_clusters=best[2] if best else np.nan,
                dbscan_noise_frac=best[3] if best else np.nan,
                dbscan_silhouette=best[1] if best else np.nan,
            )
        )
    R = pd.DataFrame(rows).merge(pd.DataFrame(db_rows), on="room")
    S = pd.DataFrame(seeds)
    pd.concat(curves).to_csv(out / "set4_clustering_curves.csv", index=False)
    R.to_csv(out / "set4_clustering_summary.csv", index=False)
    S.to_csv(out / "set4_clustering_seed_stability.csv", index=False)
    st = (
        S.groupby("room")
        .agg(
            transition_day_mean=("transition_day", "mean"),
            transition_day_min=("transition_day", "min"),
            transition_day_max=("transition_day", "max"),
            silhouette_min=("silhouette", "min"),
            silhouette_max=("silhouette", "max"),
        )
        .reset_index()
    )
    st.to_csv(out / "set4_clustering_seed_summary.csv", index=False)
    return R, st


if __name__ == "__main__":
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    R, st = run(out)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))
    print(st.round(2).to_string(index=False))
