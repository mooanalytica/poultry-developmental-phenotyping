"""Primary evaluation: leave-one-room-out for every feature set and the single-feature
baselines, leave-one-age-block-out for the fused model, and error by growth phase.

Outputs (results/):
  loro_performance.csv       per feature set / baseline: MAE, RMSE, R2 per room and the
                             mean of the three rooms
  loro_predictions.csv       per-hour predictions for every feature set
  ageblock_performance.csv   per withheld age block (all rooms pooled within the block, and
                             the mean of per-room errors within the block)
  ageblock_predictions.csv   per-hour age-block predictions
  growth_phase_mae.csv       MAE for true age <= 70 d and > 70 d, mean of the three rooms
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from common import RESULTS, ROOMS
import datasets as D
import models as M
import preprocessing as P

LABELS = {"fused": "Fused (all modalities)", "fused_bio_audio": "Fused without acoustic level",
          "video_all": "Video (morphometry + spatial behaviour + optical flow)",
          "morph_only": "Morphometry", "yolo_motion_only": "Spatial behaviour",
          "flow_only": "Optical flow", "audio_all": "Audio (bioacoustics + acoustic level)",
          "audio_bio_only": "Bioacoustics"}
BASELINES = {"Baseline: global mean": None,
             "Baseline: median body height (isotonic)": "morph__bbox_h_p50_rel__m",
             "Baseline: audio mid-band fraction (isotonic)": "audio_bio__band_mid_frac_mean__m",
             "Baseline: spectral centroid (isotonic)": "audio_bio__spectral_centroid_mean__m"}
CUT = 70.0

rows, preds = [], []
for fs in D.FEATURE_SETS:
    for r in ROOMS:
        res = P.fit_fold("LORO", r, fs)
        rows.append(dict(model=LABELS[fs], feature_set=fs, room=r, **res["met"]))
        preds.append(res["pred"].assign(feature_set=fs))
    print(f"  LORO {fs}", flush=True)

for r in ROOMS:
    d = P.fold("LORO", r, "fused")
    E = d["Xw"][d["Xw"].window_h == D.WINDOWS_EVAL]
    Etr, Ete = E[E.room != r], E[E.room == r]
    for name, col in BASELINES.items():
        if col is None:
            rows.append(dict(model=name, feature_set="-", room=r,
                             **M.metrics(Ete.age_days, np.full(len(Ete), Etr.age_days.mean()))))
            continue
        a, b = Etr[[col, "age_days"]].dropna(), Ete[[col, "age_days"]].dropna()
        iso = IsotonicRegression(out_of_bounds="clip").fit(a[col], a.age_days)
        rows.append(dict(model=name, feature_set="-", room=r, **M.metrics(b.age_days, iso.predict(b[col]))))

R = pd.DataFrame(rows)
mean = R.groupby(["model", "feature_set"], sort=False)[["MAE", "RMSE", "R2"]].mean().reset_index()
mean["room"], mean["n"] = "mean of 3 rooms", R.groupby(["model", "feature_set"], sort=False).n.sum().values
LP = pd.concat([R, mean], ignore_index=True)
LP.to_csv(RESULTS / "loro_performance.csv", index=False)
PR = pd.concat(preds, ignore_index=True).rename(columns={"y": "true_age", "pred": "predicted_age"})
PR[["feature_set", "room", "hour", "true_age", "predicted_age"]].to_csv(RESULTS / "loro_predictions.csv", index=False)

# ---------------------------------------------------------------- age blocks (fused)
ab, abp = [], []
for i, blk in enumerate(P.BLOCKS):
    res = P.fit_fold("AGEBLOCK", i)
    p = res["pred"].assign(block=blk)
    abp.append(p)
    per = {r: M.metrics(p[p.room == r].y, p[p.room == r].pred) for r in ROOMS if (p.room == r).any()}
    row = dict(block=blk, block_type="extrapolation" if i in (0, 4) else "interpolation",
               MAE=res["met"]["MAE"], RMSE=res["met"]["RMSE"], R2=res["met"]["R2"], n=res["met"]["n"],
               MAE_mean_of_rooms=np.mean([per[r]["MAE"] for r in per]),
               R2_mean_of_rooms=np.mean([per[r]["R2"] for r in per]))
    for r in per:
        row[f"MAE_{r}"], row[f"R2_{r}"], row[f"n_{r}"] = per[r]["MAE"], per[r]["R2"], per[r]["n"]
    ab.append(row)
AB = pd.DataFrame(ab)
AB = pd.concat([AB, pd.DataFrame([dict(block="mean of 5 blocks", MAE=AB.MAE.mean(), RMSE=AB.RMSE.mean(),
                                       R2=AB.R2.mean(), MAE_mean_of_rooms=AB.MAE_mean_of_rooms.mean(),
                                       R2_mean_of_rooms=AB.R2_mean_of_rooms.mean())])], ignore_index=True)
AB.to_csv(RESULTS / "ageblock_performance.csv", index=False)
pd.concat(abp, ignore_index=True).rename(columns={"y": "true_age", "pred": "predicted_age"}) \
    [["block", "room", "hour", "true_age", "predicted_age"]].to_csv(RESULTS / "ageblock_predictions.csv", index=False)

# ---------------------------------------------------------------- growth phase vs plateau
gp = []
for fs in ["morph_only", "yolo_motion_only", "flow_only", "audio_bio_only", "video_all", "fused"]:
    v = PR[PR.feature_set == fs]
    row = dict(model=LABELS[fs], feature_set=fs)
    for phase, sel in (("growth", v.true_age <= CUT), ("plateau", v.true_age > CUT)):
        q = v[sel]
        per = {r: (q[q.room == r].predicted_age - q[q.room == r].true_age).abs().mean() for r in ROOMS}
        row[f"{phase}_MAE"] = np.mean(list(per.values()))
        for r in ROOMS:
            row[f"{phase}_MAE_{r}"] = per[r]
    gp.append(row)
pd.DataFrame(gp).to_csv(RESULTS / "growth_phase_mae.csv", index=False)

print(LP[LP.room == "mean of 3 rooms"][["model", "MAE", "RMSE", "R2"]].round(3).to_string(index=False))
print(AB[["block", "MAE", "R2", "MAE_mean_of_rooms"]].round(3).to_string(index=False))
