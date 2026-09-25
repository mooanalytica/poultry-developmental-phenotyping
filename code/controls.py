"""Negative controls and the thermal leakage test.

Age and calendar date are collinear (one flock cycle, one hatch date), so any feature that
tracks the calendar can predict age without biology. Controls, all leave-one-room-out
unless stated:
  clock_only            hour-of-day features only
  permuted_label        ages permuted across rooms within each timestamp, plus noise
  sensor_health_only    detector/recording-health columns only
  fused_reference       fused model on the same feature list (no hour/presence flags)
  season extrapolation  train on one season, test on the other (the held-out season is
                        treated as a held-out age range in preprocessing)
  fused_without_acoustic_level   fused model without the equipment-sensitive audio block
Thermal: one house-wide thermal trace is joined to every room at 1, 7 and 30-day
tolerance; diagnostics shuffle the readings across hours, assign them at random per row,
use one feature at a time, or use only ambient-corrected features.

Outputs (results/): negative_controls_folds.csv, negative_controls.csv
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from common import RESULTS, ROOMS, THERMAL_DIR
import datasets as D
import models as M
import preprocessing as P

rows = []


def loro_eval(tag, fcols_fn, Xd_fn=None, label="age_days"):
    for room in ROOMS:
        d = P.fold("LORO", room)
        Xd = d["Xw"] if Xd_fn is None else Xd_fn(d["Xw"])
        fc = fcols_fn(d, Xd)
        tr = Xd.room != room
        te = (Xd.room == room) & (Xd.window_h == D.WINDOWS_EVAL)
        Xtr, Xte = Xd[tr], Xd[te]
        if len(Xte) < 30 or len(Xtr) < 100:
            continue
        m = M.hgb().fit(Xtr[fc], Xtr[label])
        rows.append(dict(test=tag, fold=f"holdout_{room}", n_feat=len(fc),
                         **M.metrics(Xte[label], m.predict(Xte[fc]))))
    print(f"  {tag}", flush=True)


base_all = lambda d, Xd: [c for c in d["fcols"] if c.endswith(("__m", "__s"))]

loro_eval("clock_only", lambda d, Xd: ["hour_sin", "hour_cos"])


def permute(Xw):
    Xp = Xw.copy()
    rng = np.random.default_rng(0)
    Xp["age_days"] = Xp.groupby("hour").age_days.transform(lambda s: rng.permutation(s.values))
    Xp["age_days"] = Xp.age_days + rng.normal(0, 40, len(Xp))
    return Xp


loro_eval("permuted_label", base_all, permute)
SENS = ("det_conf_mean", "frac_dark_frames", "frac_speed_valid", "frames_processed",
        "video_files_contributing", "n_det_sampled")
loro_eval("sensor_health_only", lambda d, Xd: [c for c in Xd.columns if any(k in c for k in SENS)])
loro_eval("fused_reference", base_all)

for a, b in (("summer", "fall"), ("fall", "summer")):
    d = P.fold("SEASON", b)
    Xw = d["Xw"]
    fc = [c for c in d["fcols"] if c.endswith(("__m", "__s"))]
    tr, te = Xw[Xw.season == a], Xw[(Xw.season == b) & (Xw.window_h == D.WINDOWS_EVAL)]
    m = M.hgb().fit(tr[fc], tr.age_days)
    rows.append(dict(test=f"train_{a}_test_{b}", fold="-", n_feat=len(fc),
                     **M.metrics(te.age_days, m.predict(te[fc]))))
    print(f"  train_{a}_test_{b}", flush=True)

for room in ROOMS:
    res = P.fit_fold("LORO", room, "fused_bio_audio")
    rows.append(dict(test="fused_without_acoustic_level", fold=f"holdout_{room}",
                     n_feat=len(res["fcols"]), **res["met"]))

# ---------------------------------------------------------------- thermal
TCOLS = ["t_mean_c", "t_p10_c", "t_p90_c", "t_max_c", "hotspot_frac", "t_spatial_std_c",
         "bird_frac", "ambient_est_c", "frame_mean_c"]
AMBIENT_FREE = ["t_bird_minus_ambient", "bird_frac", "hotspot_frac", "t_spatial_std_c"]
th = pd.read_csv(THERMAL_DIR / "thermal_frame_features.csv")
th["hour"] = pd.to_datetime(th.timestamp, errors="coerce").dt.floor("h")
th = th.dropna(subset=["hour"])
th["t_bird_minus_ambient"] = th.t_p90_c - th.ambient_est_c
use = TCOLS + ["t_bird_minus_ambient"]
broadcast = th.groupby("hour")[use].mean().reset_index()
tc = [f"therm__{c}" for c in use]


def attach(X, table, tol_days):
    return pd.merge_asof(X.sort_values("hour"), table.sort_values("hour")
                         .rename(columns={c: f"therm__{c}" for c in use}),
                         on="hour", direction="nearest", tolerance=pd.Timedelta(days=tol_days))


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
        rows.append(dict(test=tag, fold=f"holdout_{room}", n_feat=len(uc),
                         **M.metrics(Xte.age_days, m.predict(Xte[uc]))))
    print(f"  {tag}", flush=True)


base = lambda d: [c for c in d["fcols"] if c.endswith(("__m", "__s"))] + ["hour_sin", "hour_cos"]
ident = lambda X: X
go("thermal_join_fused_no_thermal", ident, base)
for tol in (1, 7, 30):
    go(f"thermal_join_fused_plus_thermal_tol{tol}d", lambda X, t=tol: attach(X, broadcast, t), lambda d: base(d) + tc)
    go(f"thermal_join_thermal_only_tol{tol}d", lambda X, t=tol: attach(X, broadcast, t), lambda d: tc)

TOL = 7
go("thermal_diag_fused_no_thermal", ident, base)
go("thermal_diag_fused_plus_thermal_real", lambda X: attach(X, broadcast, TOL), lambda d: base(d) + tc)
go("thermal_diag_thermal_only_real", lambda X: attach(X, broadcast, TOL), lambda d: tc)
for seed in (0, 1, 2):
    rng = np.random.default_rng(seed)
    shuf = broadcast.copy()
    shuf[use] = shuf[use].values[rng.permutation(len(shuf))]
    go(f"thermal_diag_thermal_only_shuffled_s{seed}", lambda X, s=shuf: attach(X, s, TOL), lambda d: tc)
    go(f"thermal_diag_fused_plus_thermal_shuffled_s{seed}", lambda X, s=shuf: attach(X, s, TOL),
       lambda d: base(d) + tc)
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
for c in use:
    go(f"thermal_diag_thermal_only_single_{c}", lambda X: attach(X, broadcast, TOL), lambda d, cc=c: [f"therm__{cc}"])
go("thermal_diag_thermal_only_ambient_free", lambda X: attach(X, broadcast, TOL),
   lambda d: [f"therm__{c}" for c in AMBIENT_FREE])
go("thermal_diag_fused_plus_ambient_free", lambda X: attach(X, broadcast, TOL),
   lambda d: base(d) + [f"therm__{c}" for c in AMBIENT_FREE])

R = pd.DataFrame(rows)
R.to_csv(RESULTS / "negative_controls_folds.csv", index=False)
S = R.groupby("test", sort=False)[["MAE", "RMSE", "R2"]].mean().reset_index()
S.to_csv(RESULTS / "negative_controls.csv", index=False)
print(S.round(3).to_string(index=False))
