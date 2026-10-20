"""Thermal-leakage diagnostics for the negative-controls figure, on the primary pipeline."""

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
from common import ROOMS, THERMAL_DIR

OUT = Path(os.environ["PHASE2_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
rows = []
TCOLS = [
    "t_mean_c",
    "t_p10_c",
    "t_p90_c",
    "t_max_c",
    "hotspot_frac",
    "t_spatial_std_c",
    "bird_frac",
    "ambient_est_c",
    "frame_mean_c",
]
th = pd.read_csv(THERMAL_DIR / "thermal_frame_features.csv")
th["hour"] = pd.to_datetime(th.timestamp, errors="coerce").dt.floor("h")
th = th.dropna(subset=["hour"])
th["t_bird_minus_ambient"] = th.t_p90_c - th.ambient_est_c
use = TCOLS + ["t_bird_minus_ambient"]
broadcast = th.groupby("hour")[use].mean().reset_index()
tc = [f"therm__{c}" for c in use]


def attach(X, table, tol_days):
    return pd.merge_asof(
        X.sort_values("hour"),
        table.sort_values("hour").rename(columns={c: f"therm__{c}" for c in use}),
        on="hour",
        direction="nearest",
        tolerance=pd.Timedelta(days=tol_days),
    )


def go(tag, Xd_fn, fc_fn):
    for room in ROOMS:
        d = P.fold("LORO", room)
        Xd = Xd_fn(d["Xw"])
        tr = Xd.room != room
        te = (Xd.room == room) & (Xd.window_h == D.WINDOWS_EVAL)
        Xtr, Xte = Xd[tr], Xd[te]
        if len(Xte) < 30 or len(Xtr) < 100:
            continue
        uc = [c for c in fc_fn(d) if Xtr[c].notna().sum() >= 20 and Xtr[c].nunique() > 1]
        if not uc:
            continue
        m = M.hgb().fit(Xtr[uc], Xtr.age_days)
        met = M.metrics(Xte.age_days, m.predict(Xte[uc]))
        rows.append(dict(test=tag, fold=f"holdout_{room}", n_feat=len(uc), **met))


base = lambda d: [c for c in d["fcols"] if c.endswith(("__m", "__s"))] + ["hour_sin", "hour_cos"]
ident = lambda X: X
TOL = 7
go("thermal_diag_fused_no_thermal", ident, base)
go(
    "thermal_diag_fused_plus_thermal_real",
    lambda X: attach(X, broadcast, TOL),
    lambda d: base(d) + tc,
)
go("thermal_diag_thermal_only_real", lambda X: attach(X, broadcast, TOL), lambda d: tc)
for seed in (0, 1, 2):
    rng = np.random.default_rng(seed)
    shuf = broadcast.copy()
    shuf[use] = shuf[use].values[rng.permutation(len(shuf))]
    go(
        f"thermal_diag_thermal_only_shuffled_s{seed}",
        lambda X, s=shuf: attach(X, s, TOL),
        lambda d: tc,
    )
for seed in (0, 1, 2):

    def rand_assign(X, sd=seed):
        rng = np.random.default_rng(100 + sd)
        Xr = X.copy()
        pick = rng.integers(0, len(broadcast), len(Xr))
        for c in use:
            Xr[f"therm__{c}"] = broadcast[c].values[pick]
        return Xr

    go(f"thermal_diag_thermal_only_random_s{seed}", rand_assign, lambda d: tc)
    go(f"thermal_diag_fused_plus_thermal_random_s{seed}", rand_assign, lambda d: base(d) + tc)
pd.DataFrame(rows).to_csv(OUT / "controls_thermal.csv", index=False)
