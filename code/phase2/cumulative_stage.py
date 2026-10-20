"""Cumulative Step 0 to Step 2 states of the pipeline corrections (LORO, age-block, season)."""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

np.random.seed(0)
import models as M
import preprocessing as P

OUT = Path(os.environ["PHASE2_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
tag = os.environ["CUM_TAG"]
AB, SE = {"step0": ("anchor", "none"), "step1": ("anchor", "none"), "step2": ("window", "window")}[
    tag
]
rows = []


def add(analysis, model, fold, metric, value):
    rows.append(
        dict(tag=tag, analysis=analysis, model=model, fold=fold, metric=metric, value=value)
    )


for fs in ("fused", "morph_only"):
    per = {}
    for r in P.ROOMS:
        res = P.fit_fold("LORO", r, fs)
        per[r] = res["met"]
        for k in ("MAE", "RMSE", "R2", "n"):
            add("LORO", fs, r, k, res["met"][k])
    for k in ("MAE", "RMSE", "R2"):
        add("LORO", fs, "macro", k, float(np.mean([per[r][k] for r in P.ROOMS])))
ab = []
for i, b in enumerate(P.BLOCKS):
    res = P.fit_fold("AGEBLOCK", i, "fused", purge=AB, embargo_h=12.0)
    ab.append(res["met"])
    for k in ("MAE", "R2"):
        add("AGEBLOCK_12h", "fused", b, k, res["met"][k])
for k in ("MAE", "R2"):
    add("AGEBLOCK_12h", "fused", "pooled fold-mean", k, float(np.mean([a[k] for a in ab])))
    add(
        "AGEBLOCK_12h",
        "fused",
        "pooled fold-mean excl. block 1",
        k,
        float(np.mean([a[k] for a in ab[1:]])),
    )
for a, bs in (("summer", "fall"), ("fall", "summer")):
    d = P.fold("SEASON", bs, "fused", purge=SE, embargo_h=12.0)
    fc = [c for c in d["fcols"] if c.endswith(("__m", "__s"))]
    Xtr, Xte = d["Xw"][d["tr"]], d["Xw"][d["te"]]
    met = M.metrics(Xte.age_days, M.hgb().fit(Xtr[fc], Xtr.age_days).predict(Xte[fc]))
    for k in ("MAE", "R2"):
        add("SEASON_12h", "fused", f"train_{a}_test_{bs}", k, met[k])
pd.DataFrame(rows).to_csv(OUT / f"cum_{tag}.csv", index=False)
