"""Morphometry reference week: first recorded morphometry per room and the body-size reference used by each fold."""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import preprocessing as P

OUT = Path(os.environ["PHASE2_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
rows = []
m = P.MASTER
for room in P.ROOMS:
    g = m[m.room == room]
    first_seen = g[g.morph__n_det_sampled.notna()].age_days.min()
    first_kept = g[g.morph__bbox_h_p50.notna()].age_days.min()
    rows += [
        dict(
            room=room, fold="all", metric="first recorded morphometry (age, days)", value=first_seen
        ),
        dict(
            room=room,
            fold="all",
            metric="end of the first morphometry week (age, days)",
            value=first_seen + 7,
        ),
        dict(
            room=room,
            fold="all",
            metric="first morphometry kept after QC (age, days)",
            value=first_kept,
        ),
    ]
folds = (
    [("LORO", r) for r in P.ROOMS]
    + [("AGEBLOCK", i) for i in range(5)]
    + [("SEASON", s) for s in ("summer", "fall")]
)
cols = ["room", "hour", "age_days", "morph__n_det_sampled"] + [
    f"morph__{c}" for c in P.MORPH_RAW_COLS
]
frame = m[cols].copy()
frame.columns = ["room", "hour", "age_days", "n_det_sampled"] + P.MORPH_RAW_COLS
for protocol, fid in folds:
    ref = P.fold_reference(protocol, fid, frame)
    label = f"block {int(fid) + 1}" if protocol == "AGEBLOCK" else str(fid)
    for room in P.ROOMS:
        rows.append(
            dict(
                room=room,
                fold=f"{protocol} | {label}",
                metric="body-size reference, bbox height (px)",
                value=ref[room][0],
            )
        )
pd.DataFrame(rows).to_csv(OUT / "reconcile.csv", index=False)
