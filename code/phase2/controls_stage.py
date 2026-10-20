"""Negative-control reference points (clock-only, sensor-health-only, global mean, fused without acoustic level)."""

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

OUT = Path(os.environ["PHASE2_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
SENS = (
    "det_conf_mean",
    "frac_dark_frames",
    "frac_speed_valid",
    "frames_processed",
    "video_files_contributing",
    "n_det_sampled",
)
rows = []


def put(control, room, met, nf):
    for k in ("MAE", "RMSE", "R2", "n"):
        rows.append(dict(control=control, fold=room, metric=k, value=met[k], n_features=nf))


for room in P.ROOMS:
    d = P.fold("LORO", room)
    Xd = d["Xw"]
    tr = Xd.room != room
    te = (Xd.room == room) & (Xd.window_h == D.WINDOWS_EVAL)
    Xtr, Xte = Xd[tr], Xd[te]
    for name, fc in (
        ("clock-only (hour of day)", ["hour_sin", "hour_cos"]),
        ("sensor-health-only", [c for c in Xd.columns if any(k in c for k in SENS)]),
    ):
        m = M.hgb().fit(Xtr[fc], Xtr.age_days)
        put(name, room, M.metrics(Xte.age_days, m.predict(Xte[fc])), len(fc))
    E = Xd[Xd.window_h == D.WINDOWS_EVAL]
    Etr = E[E.room != room]
    put(
        "global-mean baseline",
        room,
        M.metrics(Xte.age_days, np.full(len(Xte), Etr.age_days.mean())),
        0,
    )
    put(
        "fused without acoustic level",
        room,
        P.fit_fold("LORO", room, "fused_bio_audio")["met"],
        len(P.fit_fold("LORO", room, "fused_bio_audio")["fcols"]),
    )
R = pd.DataFrame(rows)
mac = (
    R[R.metric.isin(["MAE", "RMSE", "R2"])]
    .groupby(["control", "metric"], sort=False)
    .value.mean()
    .reset_index()
    .assign(fold="macro", n_features=np.nan)
)
pd.concat([R, mac], ignore_index=True).to_csv(OUT / "controls.csv", index=False)
