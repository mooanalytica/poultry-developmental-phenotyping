"""Sensitivity of the fused model to boosting hyperparameters (leave-one-room-out).

Grid: learning_rate {0.03, 0.06, 0.1} x max_leaf_nodes {15, 31, 63} x min_samples_leaf
{20, 40, 80}; all other settings as models.hgb(). Selection never uses the held-out room:
per fold, the configuration with the lowest internal early-stopping validation RMSE is
chosen, excluding configurations whose validation RMSE is >= 1.5x their training RMSE.
Only the selected configuration is scored on the held-out room. As a diagnostic (not used
for selection), each configuration is also scored room-blocked inside the training rooms.
HGB's squared-error loss is half the mean squared error, so RMSE = sqrt(-2 * score).

Outputs (results/): sensitivity_hyperparameters_all.csv, sensitivity_hyperparameters_selected.csv
"""
import itertools, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from common import RESULTS, ROOMS
import datasets as D
import models as M
import preprocessing as P

GRID = list(itertools.product([0.03, 0.06, 0.1], [15, 31, 63], [20, 40, 80]))
CURRENT = (0.06, 31, 40)
FLAG_RATIO = 1.5


def make(lr, leaves, msl):
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=lr, max_depth=6, max_leaf_nodes=leaves,
                                         min_samples_leaf=msl, l2_regularization=1.0, early_stopping=True,
                                         validation_fraction=0.15, random_state=0)


rows = []
for room in ROOMS:
    d = P.fold("LORO", room)
    Xw, FC = d["Xw"], d["fcols"]
    Xtr = Xw[d["tr"]]
    for lr, leaves, msl in GRID:
        m = make(lr, leaves, msl).fit(Xtr[FC], Xtr.age_days)
        tr_rmse = float(np.sqrt(-2 * m.train_score_[-1]))
        va_rmse = float(np.sqrt(-2 * m.validation_score_[-1]))
        inner = []
        for h in [r for r in ROOMS if r != room]:
            a = Xtr[Xtr.room != h]
            b = Xtr[(Xtr.room == h) & (Xtr.window_h == D.WINDOWS_EVAL)]
            mi = make(lr, leaves, msl).fit(a[FC], a.age_days)
            inner.append(float(np.abs(mi.predict(b[FC]) - b.age_days).mean()))
        rows.append(dict(held_out_room=room, learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl,
                         n_iter=m.n_iter_, internal_train_rmse=tr_rmse, internal_val_rmse=va_rmse,
                         val_over_train=va_rmse / tr_rmse,
                         insample_train_mae=float(np.abs(m.predict(Xtr[FC]) - Xtr.age_days).mean()),
                         overfit_flag=va_rmse >= FLAG_RATIO * tr_rmse,
                         roomblocked_inner_mae=float(np.mean(inner)),
                         is_default=(lr, leaves, msl) == CURRENT))
    print(f"  {room}: {len(GRID)} configurations", flush=True)

R = pd.DataFrame(rows)
R.to_csv(RESULTS / "sensitivity_hyperparameters_all.csv", index=False)

sel = []
for room in ROOMS:
    g = R[(R.held_out_room == room) & ~R.overfit_flag]
    best = g.sort_values("internal_val_rmse").iloc[0]
    cur = R[(R.held_out_room == room) & R.is_default].iloc[0]
    d = P.fold("LORO", room)
    Xw, FC = d["Xw"], d["fcols"]
    Xtr, Xte = Xw[d["tr"]], Xw[d["te"]]
    m = make(best.learning_rate, int(best.max_leaf_nodes), int(best.min_samples_leaf)).fit(Xtr[FC], Xtr.age_days)
    met = M.metrics(Xte.age_days, m.predict(Xte[FC]))
    ref = P.fit_fold("LORO", room)["met"]
    rr = R[R.held_out_room == room]
    sel.append(dict(held_out_room=room, learning_rate=best.learning_rate,
                    max_leaf_nodes=int(best.max_leaf_nodes), min_samples_leaf=int(best.min_samples_leaf),
                    internal_val_rmse=best.internal_val_rmse, default_internal_val_rmse=cur.internal_val_rmse,
                    n_flagged=int(rr.overfit_flag.sum()),
                    selected_rank_roomblocked=int(rr.roomblocked_inner_mae.rank().loc[best.name]),
                    spearman_internal_vs_roomblocked=spearmanr(rr.internal_val_rmse, rr.roomblocked_inner_mae).correlation,
                    MAE_selected=met["MAE"], R2_selected=met["R2"], MAE_default=ref["MAE"], R2_default=ref["R2"]))
S = pd.DataFrame(sel)
S.to_csv(RESULTS / "sensitivity_hyperparameters_selected.csv", index=False)
print(S.round(3).to_string(index=False))
