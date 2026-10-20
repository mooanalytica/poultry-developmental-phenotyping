"""Flag each 6 h evaluation window as valid when more than half of its window-mean features are present (used only to thin the scatter plots)."""

import os
import sys
from pathlib import Path

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
    mean_cols = [c for c in d["fcols"] if c.endswith("__m") and not c.startswith(("hour", "obs"))]
    valid = (X[mean_cols].notna().mean(axis=1) > 0.5).values
    frames.append(
        pd.DataFrame({"room": room, "hour": pd.to_datetime(X.hour).values, "valid": valid})
    )
pd.concat(frames).to_csv(OUT / "valid_windows.csv", index=False)
