"""Paired per-window MAE differences between models on identical rows, with a room-day block bootstrap.
Usage: python3 code/analysis/paired_fusion.py --primary PREDS --v2 PREDS --out CSV
"""

import sys

import numpy as np
import pandas as pd

ROOMS = ["room2", "room6", "room7"]
N_BOOT, SEED = 2000, 0
CONTRASTS = {
    "fused - morphometry-only": (("primary", "fused"), ("primary", "morph_only")),
    "fused - V3 (body-size only)": (("primary", "fused"), ("primary", "morph_body_only")),
    "V2 fused - morphometry-only": (("v2", "fused"), ("primary", "morph_only")),
    "V2 fused - V3 (body-size only)": (("v2", "fused"), ("primary", "morph_body_only")),
}


def load(path):
    d = pd.read_csv(path, parse_dates=["hour"])
    d["abs_err"] = (d.pred - d.y).abs()
    d["day"] = d.hour.dt.floor("D")
    return d


def paired_frame(srcs, a, b):
    A = srcs[a[0]][srcs[a[0]].model == a[1]].set_index(["room", "hour"])
    B = srcs[b[0]][srcs[b[0]].model == b[1]].set_index(["room", "hour"])
    assert A.index.equals(B.index) or (
        len(A) == len(B) and set(A.index) == set(B.index)
    ), "contrast is not on identical rows"
    B = B.reindex(A.index)
    assert np.allclose(A.y.values, B.y.values), "targets differ"
    df = pd.DataFrame(
        {
            "d": A.abs_err.values - B.abs_err.values,
            "eA": A.abs_err.values,
            "eB": B.abs_err.values,
            "day": A.day.values,
        },
        index=A.index,
    ).reset_index()
    return df


def contrast_rows(name, df, rng):
    """Per-room and macro paired differences with room-day bootstrap intervals for one contrast."""
    by = {r: df[df.room == r].groupby("day").d.agg(["sum", "count"]) for r in ROOMS}
    boot = {r: [] for r in ROOMS + ["macro"]}
    for _ in range(N_BOOT):
        diffs = []
        for r in ROOMS:
            t = by[r].iloc[rng.integers(0, len(by[r]), len(by[r]))]
            v = t["sum"].sum() / t["count"].sum()
            boot[r].append(v)
            diffs.append(v)
        boot["macro"].append(float(np.mean(diffs)))
    point = {r: float(df[df.room == r].d.mean()) for r in ROOMS}
    point["macro"] = float(np.mean(list(point.values())))
    rows = []
    for r in ROOMS + ["macro"]:
        lo, hi = np.percentile(boot[r], [2.5, 97.5])
        g = df if r == "macro" else df[df.room == r]
        macro = r == "macro"
        rows.append(
            dict(
                contrast=name,
                heldout=r,
                n_pairs=len(g),
                n_days=int(sum(len(by[x]) for x in ROOMS)) if macro else int(g.day.nunique()),
                MAE_A=(
                    float(np.mean([df[df.room == x].eA.mean() for x in ROOMS]))
                    if macro
                    else float(g.eA.mean())
                ),
                MAE_B=(
                    float(np.mean([df[df.room == x].eB.mean() for x in ROOMS]))
                    if macro
                    else float(g.eB.mean())
                ),
                MAE_diff=point[r],
                ci_lo=float(lo),
                ci_hi=float(hi),
                ci_excludes_zero=bool(lo > 0 or hi < 0),
                n_boot=N_BOOT,
                seed=SEED,
                note="95% room-day block bootstrap; conditional on this cohort and these three rooms, not new farms; same rows for both models",
            )
        )
    return rows


def run(primary, v2, out):
    srcs = {"primary": load(primary), "v2": load(v2)}
    rng = np.random.default_rng(SEED)
    rows = []
    for name, (a, b) in CONTRASTS.items():
        rows += contrast_rows(name, paired_frame(srcs, a, b), rng)
    R = pd.DataFrame(rows)
    R.to_csv(out, index=False)
    return R


if __name__ == "__main__":
    a = sys.argv
    R = run(a[a.index("--primary") + 1], a[a.index("--v2") + 1], a[a.index("--out") + 1])
    pd.set_option("display.width", 220)
    print(R.drop(columns=["note"]).round(3).to_string(index=False))
