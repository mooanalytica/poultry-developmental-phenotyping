"""Sensitivity to the complete-case restriction (leave-one-room-out).

Row sets: complete (audio, tracking and morphometry observed; primary analysis), any (at
least one modality observed) and morph (morphometry observed). Fused and morphometry-only
boosting use native missing-value handling; extremely randomised trees use median
imputation with the per-modality presence flags (obs_*) as missing-data indicators. Every
model is scored on its own held-out hours and on shared subsets of hours, because models
evaluated on different hours are not directly comparable.

Output (results/): sensitivity_data_availability.csv
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from common import RESULTS, ROOMS
import datasets as D
import models as M
import preprocessing as P

OBS = P.OBS


def et():
    return make_pipeline(SimpleImputer(strategy="median"),
                         ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0))


preds = []
for room in ROOMS:
    for model, fs, rows in (("Gradient boosting, fused", "fused", "any"),
                            ("Gradient boosting, morphometry", "morph_only", "morph"),
                            ("Extremely randomised trees, fused", "fused", "any"),
                            ("Extremely randomised trees, fused", "fused", "complete")):
        d = P.fold("LORO", room, fs, rows)
        Xw, tr, te, fc = d["Xw"], d["tr"], d["te"], d["fcols"]
        Xte = Xw[te]
        if model.startswith("Gradient"):
            mdl = M.hgb().fit(Xw.loc[tr, fc], Xw.loc[tr, "age_days"])
        else:
            E = Xw[Xw.window_h == D.WINDOWS_EVAL]
            Etr = E[E.room != room]
            mdl = et().fit(Etr[fc], Etr.age_days)
        p = Xte[["room", "hour", "age_days"]].copy()
        p["pred"], p["model"], p["rows"] = mdl.predict(Xte[fc]), model, rows
        preds.append(p)
        print(f"  {room} {model} ({rows})", flush=True)
    for fs, label in (("fused", "Gradient boosting, fused"), ("morph_only", "Gradient boosting, morphometry")):
        r = P.fit_fold("LORO", room, fs)["pred"].rename(columns={"y": "age_days"})
        r["model"], r["rows"] = label, "complete"
        preds.append(r)

Pd = pd.concat(preds, ignore_index=True)
Pd["hour"] = pd.to_datetime(Pd.hour)
cc = Pd[(Pd.model == "Gradient boosting, fused") & (Pd.rows == "complete")][["room", "hour"]].assign(in_cc=True)
mo = Pd[(Pd.model == "Gradient boosting, morphometry") & (Pd.rows == "morph")][["room", "hour"]].assign(in_morph=True)
Pd = Pd.merge(cc, on=["room", "hour"], how="left").merge(mo, on=["room", "hour"], how="left")
Pd[["in_cc", "in_morph"]] = Pd[["in_cc", "in_morph"]].fillna(False).astype(bool)

subsets = {"own held-out hours": lambda d: d, "complete-case hours": lambda d: d[d.in_cc],
           "hours with morphometry": lambda d: d[d.in_morph],
           "hours without morphometry": lambda d: d[~d.in_morph]}
out = []
for (model, rows), g in Pd.groupby(["model", "rows"]):
    for sname, sel in subsets.items():
        s = sel(g)
        if len(s) == 0:
            continue
        per = {r: {"MAE": mean_absolute_error(q.age_days, q.pred), "R2": r2_score(q.age_days, q.pred)}
               for r in ROOMS for q in [s[s.room == r]] if len(q) >= 30}
        row = dict(model=model, training_rows=rows, evaluated_on=sname,
                   MAE_mean_of_rooms=np.mean([per[r]["MAE"] for r in per]) if per else np.nan,
                   R2_mean_of_rooms=np.mean([per[r]["R2"] for r in per]) if per else np.nan,
                   n_hours=len(s))
        for r in ROOMS:
            row[f"MAE_{r}"] = per[r]["MAE"] if r in per else np.nan
            row[f"R2_{r}"] = per[r]["R2"] if r in per else np.nan
            row[f"n_{r}"] = int((s.room == r).sum())
        out.append(row)
T = pd.DataFrame(out)
T.to_csv(RESULTS / "sensitivity_data_availability.csv", index=False)
print(T[["model", "training_rows", "evaluated_on", "MAE_mean_of_rooms", "R2_mean_of_rooms"]].round(3).to_string(index=False))
