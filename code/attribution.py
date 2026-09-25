"""SHAP attribution of the fused leave-one-room-out models.

TreeExplainer on every evaluation hour of each held-out room; |SHAP| is summed within
each modality block per hour. Age-band summaries are the mean of the three rooms: per
room, the mean over that room's hours in the band, then the mean over rooms. The global
ranking uses the per-room mean |SHAP| of each feature, averaged over rooms.

Outputs (results/):
  attribution_per_hour.csv     per room-hour block |SHAP| sums
  attribution_by_age_band.csv  share of |SHAP| and mean |SHAP| (days) per block and age band
  attribution_day70.csv        block shares for true age <= 70 d and > 70 d
  feature_attribution.csv      per-feature mean |SHAP|, per room and mean of rooms
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import shap
from common import RESULTS, ROOMS
import preprocessing as P

BLOCKS = ["morph", "yolo", "flow", "audio_bio", "audio_equip", "time/flags"]
BANDS = ([0, 45, 75, 105, 140, 200], ["<45d", "45-75d", "75-105d", "105-140d", ">140d"])


def band_tables(S, edges, labels):
    S = S.copy()
    S["band"] = pd.cut(S.age_days, edges, labels=labels)
    per = (S.groupby(["room", "band", "block"], observed=True).shap_sum.mean()
             .unstack("block").reindex(columns=BLOCKS))
    share = per.div(per.sum(axis=1), axis=0)
    n_rooms = per.groupby(level="band", observed=True).size()
    return (per.groupby(level="band", observed=True).mean(),
            share.groupby(level="band", observed=True).mean(), per, share, n_rooms)


per_hour, per_feat = [], []
for r in ROOMS:
    d = P.fold("LORO", r)
    res = P.fit_fold("LORO", r)
    Xte = d["Xw"][d["te"]].sort_values("hour")
    sv = np.abs(shap.TreeExplainer(res["model"]).shap_values(Xte[res["fcols"]]))
    blk = np.array([P.block_of(c) for c in res["fcols"]])
    rec = pd.DataFrame({"room": r, "hour": Xte.hour.values, "age_days": Xte.age_days.values})
    for b in BLOCKS:
        rec[b] = sv[:, blk == b].sum(axis=1)
    per_hour.append(rec)
    per_feat.append(pd.DataFrame({"room": r, "feature": res["fcols"], "block": blk,
                                  "mean_abs_shap": sv.mean(axis=0)}))
    print(f"  {r}: {sv.shape[0]} hours x {sv.shape[1]} features", flush=True)

S = pd.concat(per_hour).melt(id_vars=["room", "hour", "age_days"], value_vars=BLOCKS,
                             var_name="block", value_name="shap_sum")
S.to_csv(RESULTS / "attribution_per_hour.csv", index=False)

absm, sharem, per_abs, per_share, nroom = band_tables(S, *BANDS)
rows = []
for band in BANDS[1]:
    for b in BLOCKS:
        rs = per_share.xs(band, level="band")[b]
        rows.append(dict(age_band=band, block=b, n_rooms=int(nroom[band]),
                         share=sharem.loc[band, b], share_min_room=rs.min(), share_max_room=rs.max(),
                         mean_abs_shap_days=absm.loc[band, b]))
pd.DataFrame(rows).to_csv(RESULTS / "attribution_by_age_band.csv", index=False)

_, share70, _, _, _ = band_tables(S, [0, 70, 200], ["<=70d", ">70d"])
share70.to_csv(RESULTS / "attribution_day70.csv")

F = pd.concat(per_feat)
G = (F.groupby(["feature", "block"]).mean_abs_shap.mean().reset_index()
       .rename(columns={"mean_abs_shap": "mean_abs_shap_mean_of_rooms"})
       .sort_values("mean_abs_shap_mean_of_rooms", ascending=False))
wide = F.pivot_table(index="feature", columns="room", values="mean_abs_shap")
wide.columns = [f"mean_abs_shap_{c}" for c in wide.columns]
G.merge(wide, left_on="feature", right_index=True).to_csv(RESULTS / "feature_attribution.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
