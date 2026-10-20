"""Hourly morphometric features from subsampled track rows, normalised per room to its first observed week."""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import numpy as np
import pandas as pd
from common import DATA

HATCH = pd.Timestamp("2025-06-05 00:00:00")
raw_dir = DATA / "morpho_raw"
frames = []

for f in sorted(raw_dir.glob("*.csv")):
    room, season = f.stem.split("_")
    d = pd.read_csv(
        f,
        header=None,
        names=["unix", "bbox_h", "bbox_w", "conf"],
        dtype={"unix": "float64", "bbox_h": "float32", "bbox_w": "float32", "conf": "float32"},
    )
    d = d[(d.bbox_h > 5) & (d.bbox_w > 5) & (d.bbox_h < 1500) & (d.bbox_w < 1500)]
    d["hour"] = pd.to_datetime(d.unix, unit="s").dt.floor("h")
    d["area"] = d.bbox_h * d.bbox_w
    d["aspect"] = d.bbox_h / d.bbox_w

    g = d.groupby("hour")
    agg = pd.DataFrame(
        {
            "bbox_h_mean": g.bbox_h.mean(),
            "bbox_h_p50": g.bbox_h.median(),
            "bbox_h_p10": g.bbox_h.quantile(0.10),
            "bbox_h_p90": g.bbox_h.quantile(0.90),
            "bbox_h_std": g.bbox_h.std(),
            "bbox_h_iqr": g.bbox_h.quantile(0.75) - g.bbox_h.quantile(0.25),
            "bbox_h_skew": g.bbox_h.skew(),
            "bbox_w_p50": g.bbox_w.median(),
            "area_p50": g.area.median(),
            "area_p90": g.area.quantile(0.90),
            "aspect_p50": g.aspect.median(),
            "aspect_iqr": g.aspect.quantile(0.75) - g.aspect.quantile(0.25),
            "det_conf_mean": g.conf.mean(),
            "n_det_sampled": g.size(),
        }
    ).reset_index()
    agg["room"], agg["season"] = room, season
    # hours with too few sampled detections give unstable quantiles
    agg = agg[agg.n_det_sampled >= 30]
    frames.append(agg)

m = pd.concat(frames, ignore_index=True)

# Reference = median bbox height over the room's first 7 observed days, so the feature transfers across cameras.
m["age_days"] = (m.hour - HATCH).dt.total_seconds() / 86400.0
ref = (
    m[m.age_days <= m.groupby("room").age_days.transform("min") + 7]
    .groupby("room")[["bbox_h_p50", "area_p50"]]
    .median()
    .rename(columns={"bbox_h_p50": "ref_h", "area_p50": "ref_a"})
)
m = m.merge(ref, on="room", how="left")
for c in [
    "bbox_h_mean",
    "bbox_h_p50",
    "bbox_h_p10",
    "bbox_h_p90",
    "bbox_h_std",
    "bbox_h_iqr",
    "bbox_w_p50",
]:
    m[c + "_rel"] = m[c] / m.ref_h
for c in ["area_p50", "area_p90"]:
    m[c + "_rel"] = m[c] / m.ref_a
m = m.drop(columns=["ref_h", "ref_a", "age_days"])

# A mount or zoom change makes the apparent size jump without biology; days departing from the room's own
# trailing baseline are flagged and blanked (past-only, see scale_filter.py).
import scale_filter

variant, unchecked = scale_filter.MODES[scale_filter.MODE]
assert (
    variant != "centred"
), "the centred filter is retrospective-only (applied in scale_filter.load_master)"
dd = scale_filter.daily_decisions(m, variant, stat="bbox_h_p50_rel")
m["day"] = m.hour.dt.floor("D")
m = m.merge(dd[["room", "day", "status"]], on=["room", "day"], how="left").drop(columns=["day"])
m["scale_artifact"] = (m.status == "flag").astype(int)
blank = (m.status == "flag") | ((m.status == "unchecked") & (unchecked == "exclude"))
n_bad, n_unchk = int((m.status == "flag").sum()), int((m.status == "unchecked").sum())
morph_cols = [
    c
    for c in m.columns
    if c
    not in ("hour", "room", "season", "scale_artifact", "status", "n_det_sampled", "det_conf_mean")
]
m.loc[blank, morph_cols] = np.nan
m = m.drop(columns=["status"])

m.to_csv(DATA / "morpho_hourly.csv", index=False)
