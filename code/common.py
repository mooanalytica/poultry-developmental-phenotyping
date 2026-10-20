"""Shared paths, constants and helpers."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
import os

CACHE = Path(os.environ.get("POULTRY_CACHE", ROOT / ".cache"))
for _p in (RESULTS, FIGURES, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

RAW_TRACKING = DATA / "raw" / "tracking"
RAW_FLOW = DATA / "raw" / "flow"
AUDIO_DIR = DATA / "audio"
THERMAL_DIR = DATA / "thermal"

ROOMS = ["room2", "room6", "room7"]
SEASONS = ["summer", "fall"]
HATCH = pd.Timestamp("2025-06-05 00:00:00")

# Equipment-sensitive: absolute level / energy. These depend on mic sensitivity,
# gain and noise floor, which are NOT calibrated across rooms. They are only used
# after within-room standardisation.
AUDIO_EQUIP_PREFIXES = (
    "rms",
    "band_fan_db",
    "band_mid_db",
    "band_voc_db",
    "band_high_db",
)
# Biological / shape-based: ratios, spectral shape, temporal structure. These
# are (to first order) invariant to a constant gain on the waveform.
AUDIO_BIO_PREFIXES = (
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_rolloff",
    "spectral_bandwidth",
    "spectral_flux",
    "band_fan_frac",
    "band_mid_frac",
    "band_voc_frac",
    "band_high_frac",
    "voc_mech_ratio",
    "spectral_entropy",
    "spectral_contrast",
    "mfcc",
)
STATS = ("mean", "std", "p10", "p50", "p90")


def audio_class(col: str) -> str:
    """Classify an audio feature column as 'equip' (gain-sensitive) or 'bio'."""
    base = col
    for s in STATS:
        if base.endswith("_" + s):
            base = base[: -len(s) - 1]
            break
    if base in ("band_fan_db", "band_mid_db", "band_voc_db", "band_high_db"):
        return "equip"
    for p in AUDIO_EQUIP_PREFIXES:
        if base == p or base.startswith(p):
            return "equip"
    for p in AUDIO_BIO_PREFIXES:
        if base == p or base.startswith(p):
            return "bio"
    return "other"


def robust_z_by_room(df, cols, room_col="room"):
    """Median/IQR standardisation within room, over the whole recording."""
    out = df.copy()
    for c in cols:
        g = out.groupby(room_col)[c]
        med = g.transform("median")
        q75 = g.transform(lambda s: s.quantile(0.75))
        q25 = g.transform(lambda s: s.quantile(0.25))
        iqr = (q75 - q25).replace(0, np.nan)
        out[c] = (out[c] - med) / iqr
    return out


def hour_floor(s):
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None).dt.floor("h")
