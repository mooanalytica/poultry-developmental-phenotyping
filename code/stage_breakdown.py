"""Error by developmental stage for the fused model and the single-modality models.

Stages are the five attribution age bands (<45, 45-75, 75-105, 105-140, >140 d; right-
inclusive). Coverage (distinct days and hours per room per stage) is reported first; a
room/stage cell with fewer than 5 distinct days is flagged as unreliable and the mean of
rooms is also given without it. These five bands are finer than the three categories of
the stage classifier (<60 / 60-110 / >110 d, comparators.py), which serves a different
purpose.

Outputs (results/): stage_coverage.csv, stage_mae.csv, stage_fusion_vs_best_single.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from common import RESULTS, ROOMS

SETS = ["fused", "morph_only", "video_all", "audio_bio_only"]
SINGLE = ["morph_only", "video_all", "audio_bio_only"]
EDGES, LABELS = [0, 45, 75, 105, 140, 200], ["<45 d", "45-75 d", "75-105 d", "105-140 d", ">140 d"]
MIN_DAYS = 5

P = pd.read_csv(RESULTS / "loro_predictions.csv", parse_dates=["hour"])
P = P[P.feature_set.isin(SETS)].copy()
P["stage"] = pd.cut(P.true_age, EDGES, labels=LABELS)
P["day"] = P.hour.dt.floor("D")

fz = P[P.feature_set == "fused"]
cov = (fz.groupby(["stage", "room"], observed=False)
         .agg(n_hours=("hour", "size"), n_days=("day", "nunique")).reset_index())
cov["unreliable"] = cov.n_days < MIN_DAYS
cov.to_csv(RESULTS / "stage_coverage.csv", index=False)

rows = []
for (fs, st), g in P.groupby(["feature_set", "stage"], observed=True):
    row = dict(feature_set=fs, stage=st)
    maes, rmses, maes_rel, flags = [], [], [], []
    for r in ROOMS:
        q = g[g.room == r]
        c = cov[(cov.stage == st) & (cov.room == r)].iloc[0]
        e = q.predicted_age - q.true_age
        mae = e.abs().mean() if len(q) else np.nan
        rmse = float(np.sqrt((e ** 2).mean())) if len(q) else np.nan
        row[f"MAE_{r}"], row[f"RMSE_{r}"] = mae, rmse
        row[f"n_hours_{r}"], row[f"n_days_{r}"] = int(c.n_hours), int(c.n_days)
        row[f"flag_{r}"] = "unreliable, small sample" if c.unreliable else ""
        if len(q):
            maes.append(mae); rmses.append(rmse)
            (flags.append(r) if c.unreliable else maes_rel.append(mae))
    row["MAE_mean_of_rooms"] = np.mean(maes)
    row["RMSE_mean_of_rooms"] = np.mean(rmses)
    row["MAE_mean_of_reliable_rooms"] = np.mean(maes_rel) if maes_rel else np.nan
    row["n_reliable_rooms"] = len(maes_rel)
    row["unreliable_rooms"] = ",".join(flags)
    rows.append(row)
S = pd.DataFrame(rows)
S["stage"] = pd.Categorical(S.stage, LABELS, ordered=True)
S = S.sort_values(["stage", "feature_set"])
S.to_csv(RESULTS / "stage_mae.csv", index=False)

fus = []
for st in LABELS:
    s = S[S.stage == st].set_index("feature_set")
    best = s.loc[SINGLE, "MAE_mean_of_rooms"].idxmin()
    rec = dict(stage=st, best_single=best, fused_MAE=s.loc["fused", "MAE_mean_of_rooms"],
               best_single_MAE=s.loc[best, "MAE_mean_of_rooms"])
    rec["fusion_advantage"] = rec["best_single_MAE"] - rec["fused_MAE"]
    signs = []
    for r in ROOMS:
        adv = s.loc[best, f"MAE_{r}"] - s.loc["fused", f"MAE_{r}"]
        rec[f"advantage_{r}"] = adv
        rec[f"flag_{r}"] = s.loc["fused", f"flag_{r}"]
        signs.append(np.sign(adv))
    rec["same_sign_all_rooms"] = len(set(signs)) == 1
    rel = [np.sign(rec[f"advantage_{r}"]) for r in ROOMS if not rec[f"flag_{r}"]]
    rec["same_sign_reliable_rooms"] = len(set(rel)) == 1 if rel else np.nan
    fus.append(rec)
pd.DataFrame(fus).to_csv(RESULTS / "stage_fusion_vs_best_single.csv", index=False)
print(cov.pivot(index="stage", columns="room", values="n_days").to_string())
