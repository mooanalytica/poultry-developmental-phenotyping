"""Flag each 6 h evaluation window by the number of observed hours it contains and the share of window-mean features present."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import preprocessing as P

OUT = Path(os.environ["PHASE2_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
frames = []
for room in P.ROOMS:
    d = P.fold("LORO", room, "fused")
    X = d["Xw"]
    X = X[(X.room == room) & (X.window_h == 6)]
    hours = np.sort(d["f"].loc[d["f"].room == room, "hour"].values)
    at = X.hour.values
    n_obs = np.searchsorted(hours, at, side="right") - np.searchsorted(
        hours, at - np.timedelta64(6, "h"), side="right"
    )
    mean_cols = [c for c in d["fcols"] if c.endswith("__m") and not c.startswith(("hour", "obs"))]
    share = X[mean_cols].notna().mean(axis=1).values
    frames.append(
        pd.DataFrame(
            {
                "room": room,
                "hour": pd.to_datetime(X.hour).values,
                "valid": share > 0.5,
                "n_obs_hours": n_obs,
                "share_mean_features_present": share,
            }
        )
    )
pd.concat(frames).to_csv(OUT / "valid_windows.csv", index=False)
