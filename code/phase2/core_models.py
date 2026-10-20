"""Leave-one-room-out, age-block and seasonal runs of the primary pipeline and the V2 and V3 variants."""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

np.random.seed(0)
import datasets as D
import models as M
import preprocessing as P
import scale_filter as S

OUT = Path(os.environ["PHASE2_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
tag = os.environ["STEP4_TAG"]
rows = []
cfg = dict(
    tag=tag,
    audio_ref=P.AUDIO_REF,
    scale_filter=S.MODE,
    slope_mode=D.SLOPE_MODE,
    slope_min_frac=D.SLOPE_MIN_FRAC,
)
assert (S.MODE, D.SLOPE_MODE, D.SLOPE_MIN_FRAC) == (
    "B_exclude",
    "time",
    0.5,
), "primary pipeline settings required"


def add(**k):
    rows.append({**cfg, **k})


preds = []
ab_preds = []
for fs in os.environ.get("SETS", "fused,morph_only,morph_body_only").split(","):
    per = {}
    for r in P.ROOMS:
        res = P.fit_fold("LORO", r, fs)
        per[r] = res["met"]
        for k in ("MAE", "RMSE", "R2", "n"):
            add(analysis="LORO", model=fs, fold=r, metric=k, value=res["met"][k])
        add(analysis="LORO", model=fs, fold=r, metric="n_train", value=res["n_train"])
        add(analysis="LORO", model=fs, fold=r, metric="n_features", value=len(res["fcols"]))
        preds.append(res["pred"].assign(model=fs))
    for k in ("MAE", "RMSE", "R2"):
        add(
            analysis="LORO",
            model=fs,
            fold="macro",
            metric=k,
            value=float(np.mean([per[r][k] for r in P.ROOMS])),
        )
if preds:
    pd.concat(preds).to_csv(OUT / f"preds_loro_{tag}.csv", index=False)
for fs in os.environ.get("AB_SETS", "fused,morph_body_only").split(","):
    for emb in (12.0, 24.0):
        ab = []
        for i, b in enumerate(P.BLOCKS):
            res = P.fit_fold("AGEBLOCK", i, fs, purge="window", embargo_h=emb)
            ab.append(res["met"])
            if emb == 12.0:
                ab_preds.append(res["pred"].assign(model=fs, block=b))
            for k in ("MAE", "RMSE", "R2", "n"):
                add(analysis=f"AGEBLOCK_{emb:g}h", model=fs, fold=b, metric=k, value=res["met"][k])
            add(
                analysis=f"AGEBLOCK_{emb:g}h",
                model=fs,
                fold=b,
                metric="n_train",
                value=res["n_train"],
            )
        for k in ("MAE", "RMSE", "R2"):
            add(
                analysis=f"AGEBLOCK_{emb:g}h",
                model=fs,
                fold="pooled fold-mean",
                metric=k,
                value=float(np.mean([a[k] for a in ab])),
            )
            add(
                analysis=f"AGEBLOCK_{emb:g}h",
                model=fs,
                fold="pooled fold-mean excl. block 1",
                metric=k,
                value=float(np.mean([a[k] for a in ab[1:]])),
            )
        for a, bs in (("summer", "fall"), ("fall", "summer")):
            d = P.fold("SEASON", bs, fs, purge="window", embargo_h=emb)
            fc = [c for c in d["fcols"] if c.endswith(("__m", "__s"))]
            Xtr, Xte = d["Xw"][d["tr"]], d["Xw"][d["te"]]
            met = M.metrics(Xte.age_days, M.hgb().fit(Xtr[fc], Xtr.age_days).predict(Xte[fc]))
            for k in ("MAE", "RMSE", "R2", "n"):
                add(
                    analysis=f"SEASON_{emb:g}h",
                    model=fs,
                    fold=f"train_{a}_test_{bs}",
                    metric=k,
                    value=met[k],
                )
            add(
                analysis=f"SEASON_{emb:g}h",
                model=fs,
                fold=f"train_{a}_test_{bs}",
                metric="n_train",
                value=len(Xtr),
            )
pd.DataFrame(rows).to_csv(OUT / f"core_{tag}.csv", index=False)
if ab_preds:
    pd.concat(ab_preds).to_csv(OUT / f"preds_ageblock_{tag}.csv", index=False)
