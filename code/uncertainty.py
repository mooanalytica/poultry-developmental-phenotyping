"""Uncertainty of the headline leave-one-room-out metrics (fused model).

Room level: with three rooms a bootstrap is not meaningful, so the three per-room values
and their range are reported. Day level: whole room-days are resampled with replacement
(1,000 replicates, seed 0), either within each room (stratified) or pooled across rooms,
and the 2.5th/97.5th percentiles of each statistic are reported. MAE and R2 are computed
exactly from per-room-day sufficient statistics.

Output (results/): loro_uncertainty.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from common import RESULTS, ROOMS

N_BOOT = 1000


def room_day_table(p):
    p = p.copy()
    p["room_day"] = p.room + "_" + p.hour.dt.floor("D").astype(str)
    p["ae"] = (p.pred - p.y).abs()
    p["se"] = (p.pred - p.y) ** 2
    p["y2"] = p.y ** 2
    return (p.groupby(["room", "room_day"])
              .agg(n=("y", "size"), s_ae=("ae", "sum"), s_se=("se", "sum"),
                   s_y=("y", "sum"), s_y2=("y2", "sum")).reset_index())


def mae_r2(n, s_ae, s_se, s_y, s_y2):
    return s_ae / n, 1.0 - s_se / (s_y2 - s_y ** 2 / n)


def bootstrap(rd, stratified, rng):
    A = rd[["n", "s_ae", "s_se", "s_y", "s_y2"]].to_numpy(float)
    room_idx = {r: np.where(rd.room.to_numpy() == r)[0] for r in ROOMS}
    all_idx = np.arange(len(rd))
    reps = []
    for _ in range(N_BOOT):
        if stratified:
            draw = {r: rng.choice(ix, size=len(ix), replace=True) for r, ix in room_idx.items()}
        else:
            samp = rng.choice(all_idx, size=len(all_idx), replace=True)
            rooms_of = rd.room.to_numpy()[samp]
            draw = {r: samp[rooms_of == r] for r in ROOMS}
        per = {r: mae_r2(*A[draw[r]].sum(axis=0)) for r in ROOMS}
        tot = sum(A[draw[r]].sum(axis=0) for r in ROOMS)
        reps.append(dict(**{f"MAE_{r}": per[r][0] for r in ROOMS}, **{f"R2_{r}": per[r][1] for r in ROOMS},
                         MAE_mean_of_rooms=np.mean([per[r][0] for r in ROOMS]),
                         R2_mean_of_rooms=np.mean([per[r][1] for r in ROOMS]),
                         MAE_pooled=mae_r2(*tot)[0], R2_pooled=mae_r2(*tot)[1]))
    return pd.DataFrame(reps)


L = pd.read_csv(RESULTS / "loro_predictions.csv", parse_dates=["hour"])
L = L[L.feature_set == "fused"].rename(columns={"true_age": "y", "predicted_age": "pred"})
rd = room_day_table(L)
pt = {r: mae_r2(*rd[rd.room == r][["n", "s_ae", "s_se", "s_y", "s_y2"]].sum().to_numpy(float)) for r in ROOMS}
tot = rd[["n", "s_ae", "s_se", "s_y", "s_y2"]].sum().to_numpy(float)
point = {**{f"MAE_{r}": pt[r][0] for r in ROOMS}, **{f"R2_{r}": pt[r][1] for r in ROOMS},
         "MAE_mean_of_rooms": np.mean([pt[r][0] for r in ROOMS]),
         "R2_mean_of_rooms": np.mean([pt[r][1] for r in ROOMS]),
         "MAE_pooled": mae_r2(*tot)[0], "R2_pooled": mae_r2(*tot)[1]}

out = []
for stratified in (True, False):
    rng = np.random.default_rng(0)
    B = bootstrap(rd, stratified, rng)
    for stat, v in point.items():
        out.append(dict(resampling="room-days within each room" if stratified else "room-days pooled",
                        statistic=stat, point=v, ci_2_5=np.percentile(B[stat], 2.5),
                        ci_97_5=np.percentile(B[stat], 97.5), n_boot=N_BOOT, n_room_days=len(rd)))
U = pd.DataFrame(out)
U.to_csv(RESULTS / "loro_uncertainty.csv", index=False)
print(U[U.statistic.isin(["MAE_mean_of_rooms", "R2_mean_of_rooms"])].round(3).to_string(index=False))
