"""Body-size trajectory: daily median relative body height per room as a ratio to its first-week baseline."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import scale_filter as sf

OUT = Path(os.environ["PHASE2_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
STAT, PLACEMENT = "morph__bbox_h_p50_rel", pd.Timestamp("2025-06-05")
m = sf.load_master()
daily = (
    m.assign(age_day=(m.hour.dt.floor("D") - PLACEMENT).dt.days)
    .groupby(["room", "age_day"])[STAT]
    .median()
    .dropna()
    .reset_index(name="daily_median")
)
frames, rows = [], []
for room, g in daily.groupby("room"):
    s = g.sort_values("age_day").set_index("age_day").daily_median
    base = float(s[s.index < s.index.min() + 7].median())
    ratio = s / base
    smooth = ratio.rolling(7, center=True, min_periods=3).median()
    peak = float(smooth.max())
    rows += [
        dict(
            room=room,
            metric="first-week baseline (median of the first 7 observed daily medians, relative-height units)",
            value=base,
        ),
        dict(
            room=room,
            metric="peak ratio (maximum of the 7-day centred rolling median of the daily ratio)",
            value=peak,
        ),
        dict(room=room, metric="age at peak ratio (days)", value=float(smooth.idxmax())),
        dict(
            room=room,
            metric="age at which the smoothed ratio first reaches 90% of its peak (days)",
            value=float(smooth[smooth >= 0.9 * peak].index.min()),
        ),
        dict(
            room=room,
            metric="age at which the smoothed ratio first reaches 95% of its peak (days)",
            value=float(smooth[smooth >= 0.95 * peak].index.min()),
        ),
        dict(room=room, metric="days with data", value=float(len(s))),
    ]
    frames.append(
        pd.DataFrame(
            dict(room=room, age_day=ratio.index, ratio=ratio.values, ratio_smooth7=smooth.values)
        )
    )
D = pd.concat(frames, ignore_index=True)
D.to_csv(OUT / "body_size_daily.csv", index=False)
M_ = pd.DataFrame(rows)
for metric, g in M_.groupby("metric", sort=False):
    if not metric.startswith(("first-week", "days with")):
        rows.append(dict(room="mean of rooms", metric=metric, value=float(g.value.mean())))
w = D.pivot(index="age_day", columns="room", values="ratio").sort_index().loc[28:110].dropna()
rng = (w.max(axis=1) - w.min(axis=1)) / w.mean(axis=1) * 100
rows += [
    dict(
        room="between rooms",
        metric="relative between-room difference, days 28-110: mean over days of (max - min) / mean of the three rooms' daily ratios (%)",
        value=float(rng.mean()),
    ),
    dict(
        room="between rooms",
        metric="relative between-room difference, days 28-110: maximum over days (%)",
        value=float(rng.max()),
    ),
    dict(
        room="between rooms",
        metric="age of the maximum between-room difference (days)",
        value=float(rng.idxmax()),
    ),
    dict(
        room="between rooms", metric="days 28-110 with data in all three rooms", value=float(len(w))
    ),
]
for a, b in (("room2", "room6"), ("room2", "room7"), ("room6", "room7")):
    p = (w[a] - w[b]).abs() / ((w[a] + w[b]) / 2) * 100
    rows += [
        dict(
            room=f"{a} vs {b}",
            metric="relative difference, days 28-110: mean over days (%)",
            value=float(p.mean()),
        ),
        dict(
            room=f"{a} vs {b}",
            metric="relative difference, days 28-110: maximum over days (%)",
            value=float(p.max()),
        ),
    ]
pd.DataFrame(rows).to_csv(OUT / "body_size_metrics.csv", index=False)
