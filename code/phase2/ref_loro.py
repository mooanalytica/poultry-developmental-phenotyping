"""Phase 2 stage: Step 0 reference (centred filter, row slope) and Step 1/2 state LORO macro, for the old-vs-new comparisons. Env: SCALE_FILTER_MODE, SLOPE_MODE, REF_TAG, PHASE2_OUT."""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

np.random.seed(0)
import preprocessing as P

OUT = Path(os.environ["PHASE2_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
tag = os.environ["REF_TAG"]
rows = []
for fs in ("fused", "morph_only"):
    per = {}
    for r in P.ROOMS:
        res = P.fit_fold("LORO", r, fs)
        per[r] = res["met"]
        for k in ("MAE", "RMSE", "R2", "n"):
            rows.append(dict(tag=tag, model=fs, fold=r, metric=k, value=res["met"][k]))
    for k in ("MAE", "RMSE", "R2"):
        rows.append(
            dict(
                tag=tag,
                model=fs,
                fold="macro",
                metric=k,
                value=float(np.mean([per[r][k] for r in P.ROOMS])),
            )
        )
pd.DataFrame(rows).to_csv(OUT / f"ref_loro_{tag}.csv", index=False)
