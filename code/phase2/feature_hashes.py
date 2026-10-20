"""SHA-256 of the final windowed feature matrix per fold."""

import hashlib
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
import preprocessing as P

label = os.environ["HASH_LABEL"]
sets = os.environ["HASH_SETS"].split(",")


def mat_hash(d):
    X = d["Xw"].copy()
    X["_tr"] = d["tr"].values.astype(float)
    X["_te"] = d["te"].values.astype(float)
    X = X.sort_values(["room", "window_h", "hour"], kind="mergesort")
    cols = d["fcols"]
    h = hashlib.sha256()
    h.update(("|".join(cols)).encode())
    h.update(X.room.astype(str).str.cat(sep=",").encode())
    h.update(X.hour.astype("int64").values.tobytes())
    h.update(X.window_h.astype("int64").values.tobytes())
    num = X[["age_days", "_tr", "_te"] + cols].astype("float64").to_numpy()
    num = np.where(np.isnan(num), np.float64("nan"), num)
    h.update(np.ascontiguousarray(num).astype("<f8").tobytes())
    return h.hexdigest(), len(X), len(cols)


lines = []
folds = (
    [("LORO", r) for r in P.ROOMS]
    + [("AGEBLOCK", i) for i in range(5)]
    + [("SEASON", s) for s in ("summer", "fall")]
)
for fs in sets:
    allh = hashlib.sha256()
    for protocol, fid in folds:
        d = P.fold(protocol, fid, fs, purge="window", embargo_h=12.0)
        hx, n, k = mat_hash(d)
        allh.update(hx.encode())
        lines.append(f"{label}\t{fs}\t{protocol}\t{fid}\trows={n}\tcols={k}\tsha256={hx}")
    lines.append(f"{label}\t{fs}\tALL_FOLDS_COMBINED\t-\t-\t-\tsha256={allh.hexdigest()}")
open(__import__("os").environ["PHASE2_OUT"] + f"/feature_hashes_{label}.txt", "w").write(
    "\n".join(lines) + "\n"
)
