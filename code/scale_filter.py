"""Past-only camera-scale artefact filter (9-day baseline, ratio limit 1.35) and the master-table loader."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from common import DATA

WINDOW, MIN_HIST, LIMIT = 9, 3, 1.35
STAT = "morph__bbox_h_p50_rel"
MASTER_FILE = (
    "master_hourly_oct15.csv"  # original table + the 81 hours the centred filter had blanked
)
OLD_FLAG = "morph__scale_artifact_centred_orig"

# (variant, treatment of unchecked days)
MODES = {
    "B_exclude": ("B", "exclude"),  # PRIMARY
    "B_keep": ("B", "keep"),  # sensitivity: keep burn-in days
    "A_exclude": ("A", "exclude"),  # sensitivity: baseline from all past days
    "A_keep": ("A", "keep"),
    "centred_retro": ("centred", None),
}  # retrospective: original centred flags + legacy trailing check
MODE = os.environ.get("SCALE_FILTER_MODE", "B_exclude")
LEGACY = MODE == "centred_retro"

# morphometric columns the original pipeline left empty on flagged hours (det_conf_mean and
# n_det_sampled were never blanked)
BLANK_COLS = [
    "morph__" + c
    for c in [
        "bbox_h_mean",
        "bbox_h_p50",
        "bbox_h_p10",
        "bbox_h_p90",
        "bbox_h_std",
        "bbox_h_iqr",
        "bbox_h_skew",
        "bbox_w_p50",
        "area_p50",
        "area_p90",
        "aspect_p50",
        "aspect_iqr",
        "bbox_h_mean_rel",
        "bbox_h_p50_rel",
        "bbox_h_p10_rel",
        "bbox_h_p90_rel",
        "bbox_h_std_rel",
        "bbox_h_iqr_rel",
        "bbox_w_p50_rel",
        "area_p50_rel",
        "area_p90_rel",
    ]
]


def decide(day_med, variant):
    """Flag each day against the median of the preceding accepted days; returns [day_med, ref, status]."""
    s = day_med.dropna().sort_index()
    lim = np.log(LIMIT)
    ref, status = [], []
    if variant == "A":
        r = s.rolling(f"{WINDOW}D", closed="left", min_periods=MIN_HIST).median()
        for v, rv in zip(s.values, r.values):
            ref.append(rv)
            status.append(
                "unchecked" if np.isnan(rv) else ("flag" if abs(np.log(v / rv)) > lim else "ok")
            )
    elif variant == "B":
        accepted = {}
        for day, v in s.items():
            hist = [x for dd, x in accepted.items() if day - pd.Timedelta(days=WINDOW) <= dd < day]
            if len(hist) >= MIN_HIST:
                rv = float(np.median(hist))
                fl = abs(np.log(v / rv)) > lim
                ref.append(rv)
                status.append("flag" if fl else "ok")
                if not fl:
                    accepted[day] = v
            else:  # not enough accepted history: cannot judge
                ref.append(np.nan)
                status.append("unchecked")
                accepted[day] = v
    else:
        raise ValueError(variant)
    return pd.DataFrame({"day_med": s.values, "ref": ref, "status": status}, index=s.index)


def daily_decisions(m, variant, stat=STAT):
    """m: master-like frame with room, hour, `stat`. One row per (room, day) that has morphometry."""
    d = m.assign(day=m.hour.dt.floor("D"))
    out = []
    for room, g in d.groupby("room"):
        dm = g.groupby("day")[stat].median().dropna()
        t = decide(dm, variant).reset_index().rename(columns={"index": "day"})
        t.insert(0, "room", room)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def load_master(mode=None):
    """Master table with the scale filter applied for `mode`; the file on disk is never modified."""
    mode = mode or MODE
    variant, unchecked = MODES[mode]
    m = pd.read_csv(DATA / MASTER_FILE, parse_dates=["hour", "date"])
    old = m.pop(OLD_FLAG)  # keep out of the feature columns (prefix morph__)
    if variant == "centred":  # reconstruct the original table exactly
        bad = old == 1
        m.loc[bad, BLANK_COLS] = np.nan
        m.loc[bad, "morph__scale_artifact"] = 1.0
        m.loc[bad, "obs_morph"] = 0
        m["qc__scale_status"] = np.where(bad, "flag", "ok")
        return m
    dd = daily_decisions(m, variant)
    m["day"] = m.hour.dt.floor("D")
    m = m.merge(
        dd[["room", "day", "status"]].rename(columns={"status": "qc__scale_status"}),
        on=["room", "day"],
        how="left",
    ).drop(columns="day")
    has = m.morph__n_det_sampled.notna()
    drop = (m.qc__scale_status == "flag") | (
        (m.qc__scale_status == "unchecked") & (unchecked == "exclude")
    )
    m.loc[drop, BLANK_COLS] = np.nan
    m.loc[drop, "obs_morph"] = 0
    m["morph__scale_artifact"] = np.where(has, (m.qc__scale_status == "flag").astype(float), np.nan)
    return m
