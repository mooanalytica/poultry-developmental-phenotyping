"""Fold manifests for every fold, protocol and purge variant: held-out range, window counts, embargo removals and QC-removed hours."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import preprocessing as P
import scale_filter as S
from common import DATA

OUT = Path(os.environ["PHASE2_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
raw = pd.read_csv(
    DATA / S.MASTER_FILE,
    usecols=["room", "hour", "obs_audio", "obs_yolo", "morph__bbox_h_p50"],
    parse_dates=["hour"],
)
pre = raw[(raw.obs_audio == 1) & (raw.obs_yolo == 1) & raw.morph__bbox_h_p50.notna()][
    ["room", "hour"]
]
pre["age"] = (pre.hour - P.HATCH).dt.total_seconds() / 86400.0
FOLDS = (
    [("LORO", r) for r in P.ROOMS]
    + [("AGEBLOCK", i) for i in range(5)]
    + [("SEASON", s) for s in ("summer", "fall")]
)
VARIANTS = [("window", 12.0), ("window", 24.0), ("anchor", 12.0), ("none", 12.0)]
rows = []
for protocol, fid in FOLDS:
    for purge, embargo in VARIANTS:
        if protocol == "LORO" and (purge, embargo) != ("window", 12.0):
            continue
        d = P.fold(protocol, fid, purge=purge, embargo_h=embargo, check=False)
        Xw, tr, te, f = d["Xw"], d["tr"], d["te"], d["f"]
        held = P._held_out(protocol, fid, Xw)
        gone = pre.merge(f[["room", "hour"]], on=["room", "hour"], how="left", indicator=True)
        gone = gone.query("_merge=='left_only'")
        if protocol == "LORO":
            gone_held = gone.room == fid
        elif protocol == "AGEBLOCK":
            lo, hi = P.EDGES[fid], P.EDGES[fid + 1]
            gone_held = (gone.age >= lo) & ((gone.age < hi) if fid < 4 else (gone.age <= hi))
        else:
            gone_held = (gone.age < P.SEASON_CUT) if fid == "summer" else (gone.age >= P.SEASON_CUT)
        ages = (np.nan, np.nan) if protocol == "LORO" else P.held_out_ages(protocol, fid)
        row = dict(
            protocol=protocol,
            fold_id=fid,
            purge=purge,
            embargo_h=embargo if purge != "none" else 0,
            heldout_lo_age_d=ages[0],
            heldout_hi_age_d=ages[1],
            n_train=int(tr.sum()),
            n_test=int(te.sum()),
            n_heldout_windows_all_W=int(held.sum()),
            n_removed_by_embargo=d["purge_info"]["removed_total"],
            n_removed_anchor_in_zone=d["purge_info"]["removed_anchor_in_zone"],
            n_removed_reach_only=d["purge_info"]["removed_reach_only"],
            n_removed_by_QC_hours=len(gone),
            n_removed_by_QC_hours_heldout=int(gone_held.sum()),
            n_removed_by_QC_hours_train=int((~gone_held).sum()),
        )
        for label, hours in (("heldout_obs", 0.0), ("heldout_or_embargo_obs", embargo)):
            key = f"train_windows_reading_{label}"
            row[key] = (
                0
                if protocol == "LORO"
                else P.assert_purged(d, protocol, fid, hours, strict=False)["violations"]
            )
        for room in P.ROOMS:
            in_room = Xw.room == room
            row[f"{room}_n_train"] = int((tr & in_room).sum())
            row[f"{room}_n_test"] = int((te & in_room).sum())
            row[f"{room}_n_removed_by_embargo"] = (
                0 if protocol == "LORO" else int(((~held) & ~tr & in_room).sum())
            )
            row[f"{room}_n_removed_by_QC_hours"] = int((gone.room == room).sum())
        rows.append(row)
pd.DataFrame(rows).to_csv(OUT / "fold_manifests.csv", index=False)
