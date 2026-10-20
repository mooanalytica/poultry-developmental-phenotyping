"""Fusion under ExtraTrees and HGB on four feature sets, with paired room-day bootstrap differences."""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import numpy as np
import pandas as pd

np.random.seed(0)
import datasets as D
import models as M
import paired_fusion as PF
import preprocessing as P
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

OUT = Path(os.environ["PHASE2_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
SETS = {
    "fused": "Fused",
    "video_all": "Video only",
    "morph_only": "Morphometry only",
    "audio_all": "Audio only",
}
mrows, prows = [], []
for fs, nm in SETS.items():
    for room in P.ROOMS:
        d = P.fold("LORO", room, fs)
        Xw, FC = d["Xw"], d["fcols"]
        E = Xw[Xw.window_h == D.WINDOWS_EVAL]
        Xtr, Xte = E[E.room != room], E[E.room == room]
        et = make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=300, min_samples_leaf=3, n_jobs=1, random_state=M.SEED
            ),
        ).fit(Xtr[FC], Xtr.age_days)
        p = et.predict(Xte[FC])
        mrows.append(
            dict(
                learner="ExtraTrees",
                feature_set=nm,
                fold=room,
                n_features=len(FC),
                **M.metrics(Xte.age_days, p),
            )
        )
        prows.append(
            pd.DataFrame(
                dict(
                    room=room, hour=Xte.hour.values, y=Xte.age_days.values, pred=p, model=f"ET {nm}"
                )
            )
        )
        r = P.fit_fold("LORO", room, fs)
        mrows.append(
            dict(
                learner="HistGradientBoosting",
                feature_set=nm,
                fold=room,
                n_features=len(r["fcols"]),
                **r["met"],
            )
        )
        prows.append(r["pred"].assign(model=f"HGB {nm}"))
pd.DataFrame(mrows).to_csv(OUT / "et_fusion_metrics.csv", index=False)
Pr = pd.concat(prows, ignore_index=True)
Pr.to_csv(OUT / "et_fusion_preds.csv", index=False)
src = {"x": PF.load(OUT / "et_fusion_preds.csv")}
CONTR = {
    "ExtraTrees: fused - morphometry-only": ("ET Fused", "ET Morphometry only"),
    "ExtraTrees: fused - video-only": ("ET Fused", "ET Video only"),
    "HGB: fused - morphometry-only": ("HGB Fused", "HGB Morphometry only"),
    "HGB: fused - video-only": ("HGB Fused", "HGB Video only"),
}
rng = np.random.default_rng(PF.SEED)
rows = []
for name, (a, b) in CONTR.items():
    rows += PF.contrast_rows(name, PF.paired_frame(src, ("x", a), ("x", b)), rng)
rows = pd.DataFrame(rows)[
    ["contrast", "heldout", "n_pairs", "n_days", "MAE_diff", "ci_lo", "ci_hi"]
]
rows.to_csv(OUT / "et_fusion_paired.csv", index=False)
