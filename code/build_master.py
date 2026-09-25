"""Build the fused hourly master table for rooms 2/6/7, keyed on UTC hour.

Alignment note: audio `time`, the master snapshot's `hour_local`, and the YOLO
`unix_timestamp_utc` were all verified to be the same UTC clock. Hatch date is
2025-06-05 00:00 UTC and is IDENTICAL for all three rooms -> a single flock
cycle, so age is exactly collinear with calendar date (see p8_confound).

Audio handling, per the mic-sensitivity requirement:
  * `bio_*`   : shape/ratio features, gain-invariant, used raw.
  * `equip_*` : absolute level/energy features, robust-z WITHIN room so only
                the deviation from that room's own baseline survives.
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *

HATCH = pd.Timestamp("2025-06-05 00:00:00")

# ---------------------------------------------------------------- 1. audio
au = []
for r in ROOMS:
    d = pd.read_csv(AUDIO_DIR/f"audio_hourly_{r}_denoised.csv")
    d["hour"] = pd.to_datetime(d.time).dt.floor("h")
    d["room"] = r
    au.append(d)
au = pd.concat(au, ignore_index=True).drop(columns=["time"])

feat_cols = [c for c in au.columns if c not in ("room","hour","bird_age_days","n_seconds_recorded","coverage_frac")]
bio   = [c for c in feat_cols if audio_class(c) == "bio"]
equip = [c for c in feat_cols if audio_class(c) == "equip"]
print(f"audio: {len(bio)} bio (gain-invariant), {len(equip)} equipment-sensitive")

# mic-sensitivity correction on the level features only
au = robust_z_by_room(au, equip, room_col="room")
au = au.rename(columns={**{c: f"audio_bio__{c}"   for c in bio},
                        **{c: f"audio_equip__{c}" for c in equip},
                        "coverage_frac": "audio__coverage_frac"})
au = au.drop(columns=["n_seconds_recorded","bird_age_days"])

# ---------------------------------------------------------------- 2. yolo video
yo = []
for r in ROOMS:
    for s in SEASONS:
        f = RAW_TRACKING/f"{r}_{s}"/"hourly_features.csv"
        if not f.exists(): continue
        d = pd.read_csv(f)
        d["hour"] = pd.to_datetime(d.unix_timestamp_utc, unit="s").dt.floor("h")
        d["room"], d["season"] = r, s
        yo.append(d.drop(columns=["hour_start_local","unix_timestamp_utc","video_files_contributing"]))
yo = pd.concat(yo, ignore_index=True)
# summer and fall batches overlap on the changeover hour (2025-09-01 16/17:00)
yo = yo.drop_duplicates(subset=["room", "hour"], keep="first")
yo = yo.rename(columns={c: f"yolo__{c}" for c in yo.columns
                        if c not in ("room","hour","season")})
# room7 is included here; it was absent from the previous ALL_ROOMS merge.
print("yolo rooms:", yo.groupby('room').size().to_dict())

# ---------------------------------------------------------------- 3. morphometrics
mo = pd.read_csv(DATA/"morpho_hourly.csv", parse_dates=["hour"]).drop(columns=["season"])
mo = mo.rename(columns={c: f"morph__{c}" for c in mo.columns if c not in ("room","hour")})

# ---------------------------------------------------------------- 4. flow video
fl = []
for r in ROOMS:
    f = RAW_FLOW/f"video_welfare_spatial_{r}_JUN_DEC.csv"
    if not f.exists(): continue
    d = pd.read_csv(f)
    d["hour"] = pd.to_datetime(d.time).dt.floor("h"); d["room"] = r
    keep = [c for c in d.columns if c.endswith(("_mean","_p50","_std"))]
    fl.append(d[["room","hour"]+keep])
fl = pd.concat(fl, ignore_index=True)
fl = fl.rename(columns={c: f"flow__{c}" for c in fl.columns if c not in ("room","hour")})

# ---------------------------------------------------------------- 5. merge
m = au.merge(yo, on=["room","hour"], how="outer") \
      .merge(mo, on=["room","hour"], how="outer") \
      .merge(fl, on=["room","hour"], how="outer")

m["age_days"]  = (m.hour - HATCH).dt.total_seconds()/86400.0
m["date"]      = m.hour.dt.floor("D")
m["hour_utc"]  = m.hour.dt.hour
m["hour_sin"]  = np.sin(2*np.pi*m.hour_utc/24)
m["hour_cos"]  = np.cos(2*np.pi*m.hour_utc/24)
m["season"]    = np.where(m.age_days < 88, "summer", "fall")
m = m.drop_duplicates(subset=["room", "hour"], keep="first")
m = m[(m.age_days >= 0) & (m.age_days <= 200)].sort_values(["room","hour"]).reset_index(drop=True)

# observation flags - kept as explicit features so imputation is never invisible
m["obs_audio"] = m["audio_bio__spectral_centroid_mean"].notna().astype(int)
m["obs_yolo"]  = m["yolo__bird_count_mean"].notna().astype(int)
m["obs_morph"] = m["morph__bbox_h_p50"].notna().astype(int)
m["obs_flow"]  = m["flow__occupancy_frac_mean"].notna().astype(int)

m.to_csv(DATA/"master_hourly.csv", index=False)

cov = m.groupby("room")[["obs_audio","obs_yolo","obs_morph","obs_flow"]].agg(["sum","mean"])
print("\n", cov.to_string())
comp = m[(m.obs_audio==1) & (m.obs_yolo==1) & (m.obs_morph==1)]
print(f"\ntotal room-hours      : {len(m)}")
print(f"complete-case (a+y+m) : {len(comp)}  ({len(comp)/len(m):.1%})")
print(comp.groupby('room').size().to_string())
pd.DataFrame({"n_rows":[len(m)],"complete_case":[len(comp)]}).to_csv(RESULTS/"data_coverage_summary.csv", index=False)
cov.to_csv(RESULTS/"data_coverage_by_room.csv")
