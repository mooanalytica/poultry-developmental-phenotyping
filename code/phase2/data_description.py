"""Data-description outputs (coverage, key dates) and the feature dictionary."""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import preprocessing as P
import scale_filter as S
from common import DATA, HATCH, ROOMS

OUT = Path(os.environ["PHASE2_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
M = P.MASTER.copy()
raw = pd.read_csv(
    DATA / S.MASTER_FILE,
    usecols=["room", "hour", "morph__n_det_sampled", "morph__bbox_h_p50"],
    parse_dates=["hour"],
)
M = M.merge(
    raw[["room", "hour", "morph__bbox_h_p50"]].rename(
        columns={"morph__bbox_h_p50": "morph_raw_p50"}
    ),
    on=["room", "hour"],
    how="left",
)
M["raw_morph"] = M.morph_raw_p50.notna().astype(int)
M["any_mod"] = (
    (M.obs_audio == 1) | (M.obs_yolo == 1) | (M.raw_morph == 1) | (M.obs_flow == 1)
).astype(int)
M["cc"] = ((M.obs_audio == 1) & (M.obs_yolo == 1) & (M.obs_morph == 1)).astype(int)
M["cc_preQC"] = ((M.obs_audio == 1) & (M.obs_yolo == 1) & (M.raw_morph == 1)).astype(int)
MODS = {
    "audio": "obs_audio",
    "tracking (YOLO)": "obs_yolo",
    "optical flow": "obs_flow",
    "morphometry (before QC filter)": "raw_morph",
    "morphometry (after QC filter)": "obs_morph",
}
rows, kd = [], []


def add(room, item, den_name, den, num, **k):
    rows.append(
        dict(
            room=room,
            quantity=item,
            denominator_definition=den_name,
            denominator=den,
            count=num,
            percent=(100.0 * num / den if den else np.nan),
            **k,
        )
    )


for room in ROOMS + ["all rooms"]:
    g = M if room == "all rooms" else M[M.room == room]
    rooms_ = ROOMS if room == "all rooms" else [room]
    span_h = sum(
        int((x.hour.max() - HATCH) / pd.Timedelta(hours=1)) + 1
        for _, x in [(r, M[M.room == r]) for r in rooms_]
    )
    span_d = sum(
        (M[M.room == r].hour.max().normalize() - HATCH.normalize()).days + 1 for r in rooms_
    )
    add(
        room,
        "calendar days from placement (5 Jun 2025) to the room's last observed day",
        "-",
        span_d,
        span_d,
    )
    add(room, "hours from placement to the room's last observed hour", "-", span_h, span_h)
    add(
        room,
        "hours with at least one modality observed",
        "hours from placement to last observed hour",
        span_h,
        int(g.any_mod.sum()),
    )
    for name, col in MODS.items():
        first_last = sum(
            int(
                (
                    M[(M.room == r) & (M[col] == 1)].hour.max()
                    - M[(M.room == r) & (M[col] == 1)].hour.min()
                )
                / pd.Timedelta(hours=1)
            )
            + 1
            for r in rooms_
        )
        add(
            room,
            f"observed hours: {name}",
            "hours from placement to last observed hour",
            span_h,
            int(g[col].sum()),
        )
        add(
            room,
            f"observed hours: {name}",
            "hours from the modality's first to last observed hour",
            first_last,
            int(g[col].sum()),
        )
        add(
            room,
            f"days with data: {name}",
            "calendar days from placement to last observed day",
            span_d,
            int(sum(M[(M.room == r) & (M[col] == 1)].hour.dt.floor("D").nunique() for r in rooms_)),
        )
    add(
        room,
        "complete-case hours, primary (audio + tracking + morphometry after QC)",
        "hours from placement to last observed hour",
        span_h,
        int(g.cc.sum()),
    )
    add(
        room,
        "complete-case hours, primary (audio + tracking + morphometry after QC)",
        "hours with at least one modality observed",
        int(g.any_mod.sum()),
        int(g.cc.sum()),
    )
    add(
        room,
        "complete-case hours before QC filter (audio + tracking + morphometry)",
        "hours from placement to last observed hour",
        span_h,
        int(g.cc_preQC.sum()),
    )
    add(
        room,
        "hours removed by the QC filter among complete-case-eligible hours",
        "complete-case hours before QC filter",
        int(g.cc_preQC.sum()),
        int(g.cc_preQC.sum() - g.cc.sum()),
    )
    add(
        room,
        "complete-case hours that also have optical flow",
        "complete-case hours, primary",
        int(g.cc.sum()),
        int(((g.cc == 1) & (g.obs_flow == 1)).sum()),
    )
pd.DataFrame(rows).to_csv(OUT / "coverage.csv", index=False)
kd.append(
    dict(
        item="placement (age 0)",
        room="all",
        value="2025-06-05",
        age_days=0.0,
        note="day_zero; age = days since placement",
    )
)
for room in ROOMS:
    g = M[M.room == room]
    for name, col in MODS.items():
        t0 = g[g[col] == 1].hour.min()
        kd.append(
            dict(
                item=f"sensor start: {name}",
                room=room,
                value=str(t0),
                age_days=round((t0 - HATCH) / pd.Timedelta(days=1), 2),
                note="first observed hour",
            )
        )
    t0 = g[g.raw_morph == 1].hour.min()
    kd.append(
        dict(
            item="morphometry first-week reference period (body-size reference)",
            room=room,
            value=f"{t0} to {t0 + pd.Timedelta(days=7)}",
            age_days=round((t0 - HATCH) / pd.Timedelta(days=1), 2),
            note="median over hours with age <= first morphometry hour + 7 d; includes hours later removed by the QC burn-in",
        )
    )
    t0 = g[g.obs_audio == 1].hour.min()
    kd.append(
        dict(
            item="audio first-week reference period (V2 only)",
            room=room,
            value=f"{t0} to {t0 + pd.Timedelta(days=7)}",
            age_days=round((t0 - HATCH) / pd.Timedelta(days=1), 2),
            note="first 7 days of the room's audio; not used by the primary pipeline",
        )
    )
    kd.append(
        dict(
            item="day-28 baseline: age of the first morphometry hour (start of the body-size reference)",
            room=room,
            value=str(g[g.raw_morph == 1].hour.min()),
            age_days=round((g[g.raw_morph == 1].hour.min() - HATCH) / pd.Timedelta(days=1), 2),
            note="the reference week covers ages ~28.7-35.7 d",
        )
    )
    c = g[g.cc == 1]
    kd.append(
        dict(
            item="analysed age range (complete-case hours, primary)",
            room=room,
            value=f"{c.age_days.min():.2f} to {c.age_days.max():.2f} d",
            age_days=np.nan,
            note=f"{len(c)} hours",
        )
    )
    kd.append(
        dict(
            item="last observed hour (any modality)",
            room=room,
            value=str(g.hour.max()),
            age_days=round((g.hour.max() - HATCH) / pd.Timedelta(days=1), 2),
            note="",
        )
    )
c = M[M.cc == 1]
kd.append(
    dict(
        item="analysed age range (complete-case hours, all rooms)",
        room="all",
        value=f"{c.age_days.min():.2f} to {c.age_days.max():.2f} d",
        age_days=np.nan,
        note=f"{len(c)} hours",
    )
)
kd.append(
    dict(
        item="season cut",
        room="all",
        value="2025-09-01 00:00 (age 88 d): summer < 88 d, fall >= 88 d",
        age_days=88.0,
        note="used by the seasonal folds",
    )
)
kd.append(
    dict(
        item="plateau boundary",
        room="all",
        value="70 d (growth <= 70 d, plateau > 70 d)",
        age_days=70.0,
        note="used by growth/plateau reporting and detection",
    )
)
pd.DataFrame(kd).to_csv(OUT / "key_dates.csv", index=False)
fcols_all, v3, mo, per_fold = set(), set(), set(), {}
folds = (
    [("LORO", r) for r in ROOMS]
    + [("AGEBLOCK", i) for i in range(5)]
    + [("SEASON", s) for s in ("summer", "fall")]
)
for proto, fid in folds:
    for fs, store in (("fused", fcols_all), ("morph_only", mo), ("morph_body_only", v3)):
        cols = P.fold(proto, fid, fs, purge="window", embargo_h=12.0)["fcols"]
        store.update(cols)
        if fs == "fused":
            per_fold[(proto, fid)] = set(cols)
UNITS = {
    "bbox_h": "pixels",
    "bbox_w": "pixels",
    "area": "pixels^2",
    "aspect": "ratio (height / width)",
    "det_conf": "detector confidence (0-1)",
    "n_det_sampled": "count of sampled detections in the hour",
    "scale_artifact": "flag (0/1)",
    "bird_count": "birds per frame (count)",
    "nn_dist": "nearest-neighbour distance (image units; not documented in the repository)",
    "spatial_spread": "image units (not documented in the repository)",
    "speed_bl_per_s": "body lengths per second",
    "n_high_activity_subwindows": "count",
    "n_tracks": "count",
    "median_track_length_s": "seconds",
    "frac_speed_valid": "fraction",
    "frac_dark_frames": "fraction",
    "frames_processed": "count",
    "occupancy_frac": "fraction",
    "activity_per_occ": "not documented in the repository",
    "occupancy_centroids": "count of occupancy centroids",
    "disp_": "optical-flow displacement (units not documented)",
    "huddle_index": "index (dimensionless)",
    "spread_x": "not documented in the repository",
    "spread_y": "not documented in the repository",
    "wall_frac": "fraction",
    "center_frac": "fraction",
    "spatial_entropy": "entropy (units not documented)",
    "flow_mean": "optical-flow magnitude (units not documented)",
    "flow_var": "optical-flow variance (units not documented)",
    "flow_skew": "dimensionless",
    "flow_kurt": "dimensionless",
    "moving_frac": "fraction",
    "flow_p90": "optical-flow magnitude (units not documented)",
    "zero_crossing_rate": "fraction of sample pairs",
    "spectral_centroid": "Hz",
    "spectral_rolloff": "Hz",
    "spectral_bandwidth": "Hz",
    "spectral_flux": "arbitrary units",
    "band_fan_frac": "fraction of band energy",
    "band_mid_frac": "fraction of band energy",
    "band_voc_frac": "fraction of band energy",
    "band_high_frac": "fraction of band energy",
    "voc_mech_ratio": "ratio",
    "rms": "linear amplitude (relative to full scale)",
    "band_fan_db": "dB",
    "band_mid_db": "dB",
    "band_voc_db": "dB",
    "band_high_db": "dB",
}
SRC = {
    "morph": "YOLO + ByteTrack bounding boxes, hourly aggregate of subsampled detections (morpho_hourly.py)",
    "yolo": "YOLO + ByteTrack tracking, hourly aggregate (chicktrack hourly_features)",
    "flow": "whole-image optical flow / occupancy, hourly aggregate",
    "audio_bio": "audio, hourly aggregate of denoised recordings (gain-invariant shape/ratio features)",
    "audio_equip": "audio, hourly aggregate of denoised recordings (absolute level features)",
}
STAT = ("mean", "std", "p10", "p50", "p90", "iqr", "skew", "max")
dic = []
for c in sorted(fcols_all | mo | v3):
    if not c.endswith(("__m", "__s")):
        dic.append(
            dict(
                column=c,
                block="time/flags",
                source_signal={
                    "hour_sin": "hour of day (UTC), sine",
                    "hour_cos": "hour of day (UTC), cosine",
                    "obs_flow": "optical-flow availability flag for the hour",
                }.get(c, c),
                window="-",
                window_statistic="-",
                base_statistic="-",
                units="dimensionless" if c.startswith("hour") else "flag (0/1)",
                referencing="none",
                in_primary_fused=c in fcols_all,
                in_V2_fused=c in fcols_all,
                in_morph_only=c in mo,
                in_V3=c in v3,
                n_folds_present=sum(c in v for v in per_fold.values()),
                n_folds_total=len(per_fold),
            )
        )
        continue
    stem, kind = c.rsplit("__", 1)
    block, rest = stem.split("__", 1)
    base = rest.rsplit("_", 1)[-1] if rest.rsplit("_", 1)[-1] in STAT else "-"
    sig = rest[: -(len(base) + 1)] if base != "-" else rest
    key = next((k for k in sorted(UNITS, key=len, reverse=True) if sig.startswith(k)), None)
    unit = UNITS.get(key, "not documented in the repository")
    if block == "morph" and sig.endswith("_rel"):
        unit, ref = (
            "ratio to the room's first-week reference (dimensionless)",
            "divided by the room's median over its first 7 observed morphometry days (LORO: held-out room uses the mean of the two training rooms' references)",
        )
    elif block == "morph":
        ref = "none (raw)"
    elif block == "audio_equip":
        ref = "primary: (x - median) / IQR of the room's own recording (held-out room: pooled median/IQR of the training rooms); V2: ratio to the room's own first-week median (difference for dB features)"
    elif block == "audio_bio":
        ref = "primary: none (raw, gain-invariant); V2: ratio to the room's own first-week median"
    else:
        ref = "none"
    if block == "audio_equip":
        unit = unit + " (standardised in the primary model: units of the room's IQR)"
    if kind == "s":
        unit = unit + " per hour (OLS slope)"
    dic.append(
        dict(
            column=c,
            block=block,
            source_signal=f"{sig} ({SRC[block]})",
            window="trailing 1, 3, 6, 12 h windows (rows of all four sizes; evaluation on 6 h)",
            window_statistic=(
                "window mean"
                if kind == "m"
                else "OLS slope per hour against the actual time stamps (NaN for 1 h windows; >= 3 observations and >= 50% of the window's hours)"
            ),
            base_statistic=base,
            units=unit,
            referencing=ref,
            in_primary_fused=c in fcols_all,
            in_V2_fused=c in fcols_all,
            in_morph_only=c in mo,
            in_V3=c in v3,
            n_folds_present=sum(c in v for v in per_fold.values()),
            n_folds_total=len(per_fold),
        )
    )
pd.DataFrame(dic).to_csv(OUT / "feature_dictionary.csv", index=False)
