"""Time-bounded GRU (5 seeds) with its HGB reference and the Revision 3 construction."""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
import datasets as D
import gru_comparator as G
import models as M
import preprocessing as P
from common import ROOMS

OUT = Path(os.environ["PHASE2_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
rows, spans = [], []
for room in ROOMS:
    d = P.fold("LORO", room)
    FC = d["fcols"]
    E = (
        d["Xw"][d["Xw"].window_h == D.WINDOWS_EVAL]
        .dropna(subset=["age_days"])
        .sort_values(["room", "hour"])
        .reset_index(drop=True)
    )
    med, mu, sd = G.fit_imputer_scaler(E[E.room != room], FC)
    X = G.transform(E, FC, med, mu, sd)
    S, L, Y, R, H, SP = G.build_sequences(E, X)
    tr, te = R != room, R == room
    for seed in range(5):
        G.HP["seed"] = seed
        m = M.metrics(Y[te], G.train_predict(S[tr], L[tr], Y[tr], S[te], L[te]))
        rows.append(
            dict(
                heldout=room,
                model="GRU time-bounded",
                seed=seed,
                n_eval=m["n"],
                n_train_sequences=int(tr.sum()),
                MAE=m["MAE"],
                RMSE=m["RMSE"],
                R2=m["R2"],
                median_span_h=float(np.median(SP[te])),
                max_span_h=float(SP[te].max()),
                median_obs=float(np.median(L[te])),
            )
        )
    G.HP["seed"] = 0
    hg = P.fit_fold("LORO", room)["pred"].copy()
    hg["hour"] = pd.to_datetime(hg.hour)
    mg = (
        pd.DataFrame({"hour": pd.to_datetime(H[te]), "y": Y[te]})
        .merge(hg[["hour", "pred"]], on="hour", how="left")
        .dropna()
    )
    hs = M.metrics(mg.y, mg.pred)
    rows.append(
        dict(
            heldout=room,
            model="HGB fused, same anchors",
            seed=-1,
            n_eval=hs["n"],
            MAE=hs["MAE"],
            RMSE=hs["RMSE"],
            R2=hs["R2"],
        )
    )
    ha = P.fit_fold("LORO", room)["met"]
    rows.append(
        dict(
            heldout=room,
            model="HGB fused, all windows",
            seed=-1,
            n_eval=ha["n"],
            MAE=ha["MAE"],
            RMSE=ha["RMSE"],
            R2=ha["R2"],
        )
    )
    lm, nseq = G.run_legacy_gru(E, FC, room)
    rows.append(
        dict(
            heldout=room,
            model="GRU Revision 3 construction (seed 0)",
            seed=0,
            n_eval=lm["n"],
            n_train_sequences=nseq,
            MAE=lm["MAE"],
            RMSE=lm["RMSE"],
            R2=lm["R2"],
        )
    )
    sa = G.legacy_span_audit(E, FC)
    sa["fold"] = room
    spans.append(sa)
pd.DataFrame(rows).to_csv(OUT / "gru_stage.csv", index=False)
SP_ = pd.concat(spans)
SP_ = SP_[SP_.fold == "room2"]
SP_.groupby("room").span_h.agg(
    n_candidate_sequences="size",
    median_span_h="median",
    p90_span_h=lambda s: s.quantile(0.9),
    max_span_h="max",
    share_gt_24h=lambda s: (s > 24).mean(),
    share_gt_72h=lambda s: (s > 72).mean(),
).reset_index().to_csv(OUT / "gru_legacy_spans.csv", index=False)
