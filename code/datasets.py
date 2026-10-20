"""Feature blocks, window augmentation and missing-data regimes."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

WINDOWS_TRAIN = [1, 3, 6, 12]
WINDOWS_EVAL = 6
STALENESS_CAP = 3

# Slope definition (Step 3). "time": OLS slope of the feature against its actual hour stamps, using only the
# observations in the trailing window (anchor - W h, anchor]; the window is valid when it holds at least
# SLOPE_MIN_OBS observations of that feature AND at least SLOPE_MIN_FRAC of its W hours (0.5 primary, 0.75
# sensitivity). "row": the previous definition (x[t] - x[t-(W-1) rows]) / (W-1), kept only to reproduce the
# Step 0-2 results; it reaches back W-1 ROWS, i.e. far beyond W hours across gaps.
SLOPE_MODE = os.environ.get("SLOPE_MODE", "time")
SLOPE_MIN_OBS = 3
SLOPE_MIN_FRAC = float(os.environ.get("SLOPE_MIN_FRAC", "0.5"))
assert SLOPE_MODE in ("time", "row")


def window_min_obs(W):
    """Minimum number of observations a W-hour window needs for a valid slope."""
    return max(SLOPE_MIN_OBS, int(np.ceil(SLOPE_MIN_FRAC * W - 1e-9)))


def ols_slope_time(g, cols, W):
    """OLS slope (feature units per hour) of each column against actual hour stamps, over (anchor-W h, anchor]. g: one room, index = hour stamps (on the hour, unique, sorted)."""
    hrs = g.index
    assert hrs.is_unique and hrs.is_monotonic_increasing and (hrs == hrs.floor("h")).all()
    pos = ((hrs - hrs[0]) / pd.Timedelta(hours=1)).astype(int).values
    C = len(cols)
    grid = np.full((pos[-1] + 1 + W - 1, C), np.nan)
    grid[pos + W - 1] = g[cols].values
    win = np.lib.stride_tricks.sliding_window_view(grid, W, axis=0)[pos]
    k = np.arange(W, dtype=float)
    m = ~np.isnan(win)
    n = m.sum(-1)
    x = np.where(m, win, 0.0)
    tm = (m * k).sum(-1) / np.maximum(n, 1)
    xm = x.sum(-1) / np.maximum(n, 1)
    num = (x * k).sum(-1) - n * tm * xm
    den = (m * k * k).sum(-1) - n * tm * tm
    ok = (n >= window_min_obs(W)) & (den > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sl = np.where(ok, num / den, np.nan)
    return pd.DataFrame(sl, index=hrs, columns=[f"{c}__s" for c in cols])


BLOCKS = {
    "audio_bio": lambda c: c.startswith("audio_bio__"),
    "audio_equip": lambda c: c.startswith("audio_equip__"),
    "morph": lambda c: c.startswith("morph__"),
    "yolo": lambda c: c.startswith("yolo__"),
    "flow": lambda c: c.startswith("flow__"),
}
FEATURE_SETS = {
    "audio_bio_only": ["audio_bio"],
    "audio_all": ["audio_bio", "audio_equip"],
    "morph_only": ["morph"],
    "yolo_motion_only": ["yolo"],
    "flow_only": ["flow"],
    "video_all": ["morph", "yolo", "flow"],
    "fused": ["audio_bio", "audio_equip", "morph", "yolo", "flow"],
    "fused_bio_audio": ["audio_bio", "morph", "yolo", "flow"],
}


def load_master():
    import scale_filter

    m = scale_filter.load_master()
    return m.sort_values(["room", "hour"]).reset_index(drop=True)


def block_cols(df, blocks):
    cols = []
    for b in blocks:
        cols += [c for c in df.columns if BLOCKS[b](c)]
    return cols


def apply_regime(m, regime):
    """Return (frame, note). Regime A drops incomplete rows; B interpolates."""
    if regime == "A":
        keep = (m.obs_audio == 1) & (m.obs_yolo == 1) & (m.obs_morph == 1)
        return m[keep].copy(), "complete-case (audio+yolo+morph observed)"
    out = []
    feat = block_cols(m, BLOCKS.keys())
    for room, g in m.groupby("room"):
        g = g.set_index("hour").sort_index()
        full = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))
        # limit=STALENESS_CAP on each side => only gaps up to ~2*cap are bridged
        full[feat] = full[feat].interpolate(
            method="time", limit=STALENESS_CAP, limit_direction="both", limit_area="inside"
        )
        full["room"] = room
        out.append(full.rename_axis("hour").reset_index())
    f = pd.concat(out, ignore_index=True)
    f["age_days"] = (f.hour - pd.Timestamp("2025-06-05")).dt.total_seconds() / 86400.0
    for c in ["obs_audio", "obs_yolo", "obs_morph", "obs_flow"]:
        f[c] = f[c].fillna(0)
    f["hour_utc"] = f.hour.dt.hour
    f["hour_sin"] = np.sin(2 * np.pi * f.hour_utc / 24)
    f["hour_cos"] = np.cos(2 * np.pi * f.hour_utc / 24)
    f["date"] = f.hour.dt.floor("D")
    f["season"] = np.where(f.age_days < 88, "summer", "fall")
    f = f.dropna(subset=["age_days"])
    # keep rows that ended up with at least one modality present
    f = f[f[feat].notna().any(axis=1)]
    return f, f"interpolated, <= {STALENESS_CAP} h gaps bridged"


def windowise(df, cols, windows):
    """Overlapping-window feature construction. Anchor = last hour of window."""
    parts = []
    for room, g in df.groupby("room"):
        g = g.set_index("hour").sort_index()
        for W in windows:
            if W == 1:
                X = g[cols].copy()
                X.columns = [f"{c}__m" for c in cols]
                for c in cols:
                    X[f"{c}__s"] = np.nan
            else:
                r = g[cols].rolling(f"{W}h", min_periods=max(2, W // 2))
                mean = r.mean()
                mean.columns = [f"{c}__m" for c in cols]
                # linear slope over the window == cov(x,t)/var(t), computed cheaply
                if SLOPE_MODE == "time":
                    slope = ols_slope_time(g, cols, W)
                else:
                    slope = (g[cols] - g[cols].shift(W - 1)) / (W - 1)
                    slope.columns = [f"{c}__s" for c in cols]
                X = pd.concat([mean, slope], axis=1)
            X["age_days"] = g["age_days"]
            # age of the OLDEST observation any feature of this window reads (used by the temporal purge).
            # "time" slope: the mean and the slope both read only observations in (anchor - W h, anchor].
            # "row" slope (legacy): the slope also reads the observation W-1 ROWS back.
            ages = g["age_days"].values
            hrs = g.index.values
            first = (
                np.searchsorted(hrs, hrs - np.timedelta64(W, "h"), side="right")
                if W > 1
                else np.arange(len(ages))
            )
            if W > 1 and SLOPE_MODE == "row":
                back = np.arange(len(ages)) - (W - 1)
                first = np.where(back >= 0, np.minimum(first, np.clip(back, 0, None)), first)
            X["first_age_days"] = ages[first]
            X["room"] = room
            X["season"] = g["season"]
            X["window_h"] = W
            X["hour_sin"], X["hour_cos"] = g["hour_sin"], g["hour_cos"]
            for c in ["obs_audio", "obs_yolo", "obs_morph", "obs_flow"]:
                X[c] = g[c]
            parts.append(X.reset_index())
    out = pd.concat(parts, ignore_index=True)
    return out.dropna(subset=["age_days"])


def build(regime, feature_set, augment=True):
    m = load_master()
    f, note = apply_regime(m, regime)
    cols = block_cols(f, FEATURE_SETS[feature_set])
    cols = [c for c in cols if f[c].notna().sum() > 50]
    wins = WINDOWS_TRAIN if augment else [WINDOWS_EVAL]
    X = windowise(f, cols, wins)
    return X, cols, note


if __name__ == "__main__":
    for regime in ("A", "B"):
        m = load_master()
        f, note = apply_regime(m, regime)
