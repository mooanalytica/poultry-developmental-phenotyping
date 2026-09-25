"""Modelling matrices: feature blocks, window augmentation and missing-data regimes.

Regime A (primary)  : complete-case. Rows where audio, YOLO and morphometrics
                      are all genuinely observed. Nothing invented.
Regime B (secondary): all rows, short gaps filled by time interpolation under a
                      3 h staleness cap, longer gaps left as NaN for the tree
                      model to route natively. Observation flags retained.

Augmentation: overlapping temporal windows. For each anchor hour we summarise
the preceding W hours (mean + linear slope) for W in {1,3,6,12}. Each W yields a
separate training row, so a given hour contributes several views at different
smoothing scales. Legitimate here because age varies slowly; leakage is
controlled by blocking every split on whole rooms or whole age-blocks.
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *

WINDOWS_TRAIN = [1, 3, 6, 12]
WINDOWS_EVAL  = 6
STALENESS_CAP = 3           # hours

BLOCKS = {
    "audio_bio":   lambda c: c.startswith("audio_bio__"),
    "audio_equip": lambda c: c.startswith("audio_equip__"),
    "morph":       lambda c: c.startswith("morph__"),
    "yolo":        lambda c: c.startswith("yolo__"),
    "flow":        lambda c: c.startswith("flow__"),
}
FEATURE_SETS = {
    "audio_bio_only":      ["audio_bio"],
    "audio_all":           ["audio_bio", "audio_equip"],
    "morph_only":          ["morph"],
    "yolo_motion_only":    ["yolo"],
    "flow_only":           ["flow"],
    "video_all":           ["morph", "yolo", "flow"],
    "fused":               ["audio_bio", "audio_equip", "morph", "yolo", "flow"],
    "fused_bio_audio":     ["audio_bio", "morph", "yolo", "flow"],
}


def load_master():
    m = pd.read_csv(DATA/"master_hourly.csv", parse_dates=["hour", "date"])
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
        full[feat] = full[feat].interpolate(method="time", limit=STALENESS_CAP,
                                            limit_direction="both", limit_area="inside")
        full["room"] = room
        out.append(full.rename_axis("hour").reset_index())
    f = pd.concat(out, ignore_index=True)
    f["age_days"] = (f.hour - pd.Timestamp("2025-06-05")).dt.total_seconds()/86400.0
    for c in ["obs_audio", "obs_yolo", "obs_morph", "obs_flow"]:
        f[c] = f[c].fillna(0)
    f["hour_utc"] = f.hour.dt.hour
    f["hour_sin"] = np.sin(2*np.pi*f.hour_utc/24); f["hour_cos"] = np.cos(2*np.pi*f.hour_utc/24)
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
                r = g[cols].rolling(f"{W}h", min_periods=max(2, W//2))
                mean = r.mean(); mean.columns = [f"{c}__m" for c in cols]
                # linear slope over the window == cov(x,t)/var(t), computed cheaply
                slope = (g[cols] - g[cols].shift(W-1)) / (W-1)
                slope.columns = [f"{c}__s" for c in cols]
                X = pd.concat([mean, slope], axis=1)
            X["age_days"] = g["age_days"]
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
        m = load_master(); f, note = apply_regime(m, regime)
        print(f"regime {regime}: {len(f):>6} rows  ({note})")
        print("   per room:", f.groupby('room').size().to_dict())
        print("   age range:", round(f.age_days.min(),1), "-", round(f.age_days.max(),1))
