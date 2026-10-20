"""Phase 3 stage: Table 2 baselines on the primary pipeline, computed as code/evaluate.py does: global mean (training-room mean age) and isotonic regression on median relative body height. LORO, 6 h evaluation windows."""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets as D
import models as M
import numpy as np
import pandas as pd
import preprocessing as P
from sklearn.isotonic import IsotonicRegression

OUT = Path(os.environ["PHASE2_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
COL = "morph__bbox_h_p50_rel__m"
rows = []
for room in P.ROOMS:
    d = P.fold("LORO", room)
    E = d["Xw"][d["Xw"].window_h == D.WINDOWS_EVAL]
    tr, te = E[E.room != room], E[E.room == room]
    rows.append(
        dict(
            baseline="global mean",
            fold=room,
            **M.metrics(te.age_days, np.full(len(te), tr.age_days.mean()))
        )
    )
    a, b = tr[[COL, "age_days"]].dropna(), te[[COL, "age_days"]].dropna()
    iso = IsotonicRegression(out_of_bounds="clip").fit(a[COL], a.age_days)
    rows.append(
        dict(
            baseline="median body height (isotonic)",
            fold=room,
            **M.metrics(b.age_days, iso.predict(b[COL]))
        )
    )
pd.DataFrame(rows).to_csv(OUT / "baselines.csv", index=False)
