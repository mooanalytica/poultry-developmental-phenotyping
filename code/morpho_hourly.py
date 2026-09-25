"""Hourly morphometric features from the subsampled track rows.

bbox height is the primary growth signal, but raw pixel height is only
comparable *within* a camera (fixed mount, fisheye, unknown focal geometry).
We therefore emit BOTH the raw pixel summaries and a per-room-normalised
version referenced to that room's own earliest-week median, so that the
feature means "how many times bigger than a day-28 bird" rather than "pixels".
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *

HATCH = pd.Timestamp("2025-06-05 00:00:00")
raw_dir = DATA/"morpho_raw"
frames = []

for f in sorted(raw_dir.glob("*.csv")):
    room, season = f.stem.split("_")
    d = pd.read_csv(f, header=None, names=["unix","bbox_h","bbox_w","conf"],
                    dtype={"unix":"float64","bbox_h":"float32","bbox_w":"float32","conf":"float32"})
    d = d[(d.bbox_h > 5) & (d.bbox_w > 5) & (d.bbox_h < 1500) & (d.bbox_w < 1500)]
    d["hour"] = pd.to_datetime(d.unix, unit="s").dt.floor("h")
    d["area"]   = d.bbox_h * d.bbox_w
    d["aspect"] = d.bbox_h / d.bbox_w

    g = d.groupby("hour")
    agg = pd.DataFrame({
        "bbox_h_mean":  g.bbox_h.mean(),
        "bbox_h_p50":   g.bbox_h.median(),
        "bbox_h_p10":   g.bbox_h.quantile(0.10),
        "bbox_h_p90":   g.bbox_h.quantile(0.90),
        "bbox_h_std":   g.bbox_h.std(),
        "bbox_h_iqr":   g.bbox_h.quantile(0.75) - g.bbox_h.quantile(0.25),
        "bbox_h_skew":  g.bbox_h.skew(),
        "bbox_w_p50":   g.bbox_w.median(),
        "area_p50":     g.area.median(),
        "area_p90":     g.area.quantile(0.90),
        "aspect_p50":   g.aspect.median(),
        "aspect_iqr":   g.aspect.quantile(0.75) - g.aspect.quantile(0.25),
        "det_conf_mean": g.conf.mean(),
        "n_det_sampled": g.size(),
    }).reset_index()
    agg["room"], agg["season"] = room, season
    # hours with too few sampled detections give unstable quantiles
    agg = agg[agg.n_det_sampled >= 30]
    frames.append(agg)
    print(f"{f.stem}: {len(d):>9,} rows -> {len(agg):>5} hours")

m = pd.concat(frames, ignore_index=True)

# --- per-room scale normalisation -------------------------------------------
# Reference = median bbox height over that room's first 7 observed days.
# Removes fixed camera-geometry differences between rooms so the morphometric
# transfers under leave-one-room-out.
m["age_days"] = (m.hour - HATCH).dt.total_seconds()/86400.0
ref = (m[m.age_days <= m.groupby("room").age_days.transform("min") + 7]
         .groupby("room")[["bbox_h_p50","area_p50"]].median()
         .rename(columns={"bbox_h_p50":"ref_h","area_p50":"ref_a"}))
m = m.merge(ref, on="room", how="left")
for c in ["bbox_h_mean","bbox_h_p50","bbox_h_p10","bbox_h_p90","bbox_h_std","bbox_h_iqr","bbox_w_p50"]:
    m[c+"_rel"] = m[c] / m.ref_h
for c in ["area_p50","area_p90"]:
    m[c+"_rel"] = m[c] / m.ref_a
m = m.drop(columns=["ref_h","ref_a","age_days"])

# --- camera-scale artifact detection ------------------------------------------
# Apparent bird size is only a growth signal while the camera geometry is fixed.
# Between 2025-08-19 and 2025-08-22 the median bbox height roughly DOUBLES in all
# three rooms simultaneously and then returns - a mount/zoom change, not biology.
# Rather than hard-coding those dates we flag any hour whose daily median departs
# sharply from that room's own 7-day trend, and blank the morphometrics there.
d = (m.assign(day=m.hour.dt.floor("D"))
       .groupby(["room","day"]).bbox_h_p50_rel.median().rename("day_med").reset_index())
d["trend"] = (d.groupby("room").day_med
                .transform(lambda s: s.rolling(9, center=True, min_periods=3).median()))
d["scale_artifact"] = (np.log(d.day_med / d.trend).abs() > np.log(1.35)).astype(int)
m["day"] = m.hour.dt.floor("D")
m = m.merge(d[["room","day","scale_artifact"]], on=["room","day"], how="left").drop(columns=["day"])
m["scale_artifact"] = m.scale_artifact.fillna(0).astype(int)
n_bad = int(m.scale_artifact.sum())
morph_cols = [c for c in m.columns if c not in ("hour","room","season","scale_artifact","n_det_sampled","det_conf_mean")]
m.loc[m.scale_artifact == 1, morph_cols] = np.nan
print(f"scale-artifact hours blanked: {n_bad} of {len(m)} "
      f"({n_bad/len(m):.1%}); dates: "
      f"{sorted(set(m.loc[m.scale_artifact==1,'hour'].dt.date.astype(str)))[:8]}")

m.to_csv(DATA/"morpho_hourly.csv", index=False)
print(f"\n-> {DATA/'morpho_hourly.csv'}  {len(m)} room-hours")
print(m.groupby("room").bbox_h_p50_rel.describe()[["count","mean","min","max"]].to_string())
