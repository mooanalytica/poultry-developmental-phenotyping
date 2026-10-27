"""QC timing check for the age-block evaluation: how much the day-level QC history leaks across the split, and age-block results with QC built inside the training folds or strictly past-only by hour."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models as M
import preprocessing as P
import scale_filter as SF
from common import DATA

OUT = Path(os.environ["PHASE2_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
STAT, LIM, WINDOW = SF.STAT, np.log(SF.LIMIT), pd.Timedelta(days=SF.WINDOW)
EMBARGO_DAYS = 0.5
raw = pd.read_csv(DATA / SF.MASTER_FILE, parse_dates=["hour", "date"])
raw.pop(SF.OLD_FLAG)
raw["day"] = raw.hour.dt.floor("D")
orig = SF.daily_decisions(raw.drop(columns="day"), "B")
pre = raw[(raw.obs_audio == 1) & (raw.obs_yolo == 1) & raw.morph__bbox_h_p50.notna()]
age = lambda t: (t - P.HATCH).dt.total_seconds() / 86400.0


def with_age(m):
    m = m.copy()
    m["age_days"] = age(m.hour)
    return m


def day_range(day, lo, hi):
    a = (day - P.HATCH).days
    return a < hi and a + 1 > lo


def fold_internal(lo, hi):
    """Statuses of training days rebuilt from training days only; withheld and embargo days keep their original status."""
    parts = []
    for room, g in raw.groupby("room"):
        dm = g.groupby("day")[STAT].median().dropna()
        gone = np.array([day_range(d, lo - EMBARGO_DAYS, hi + EMBARGO_DAYS) for d in dm.index])
        new = SF.decide(dm[~gone], "B")
        o = orig[orig.room == room].set_index("day")
        st = o.status.copy()
        st.loc[new.index] = new.status
        parts.append(pd.DataFrame({"room": room, "day": st.index, "status": st.values}))
    return pd.concat(parts, ignore_index=True)


def hourly_master():
    """Each hour judged on the running median of its own day so far against the baseline from earlier accepted days."""
    m = raw.copy()
    m["run"] = m.groupby(["room", "day"])[STAT].transform(lambda s: s.expanding().median())
    d = orig.set_index(["room", "day"])
    ref = pd.Series(
        d.ref.reindex(pd.MultiIndex.from_frame(m[["room", "day"]])).values, index=m.index
    )
    day_status = pd.Series(
        d.status.reindex(pd.MultiIndex.from_frame(m[["room", "day"]])).values, index=m.index
    )
    flag = (np.log(m.run / ref).abs() > LIM) & m.run.notna() & ref.notna()
    drop = flag | (day_status == "unchecked")
    out = m.drop(columns=["run", "day"]).copy()
    out.loc[drop.values, SF.BLANK_COLS] = np.nan
    out.loc[drop.values, "obs_morph"] = 0
    return out


def use(master):
    P.MASTER = with_age(master)
    P._assembled.clear()


def evaluate(fold):
    d = P.fold("AGEBLOCK", fold, "fused", purge="window", embargo_h=12.0, check=False)
    Xtr, Xte = d["Xw"][d["tr"]], d["Xw"][d["te"]]
    model = M.hgb().fit(Xtr[d["fcols"]], Xtr.age_days)
    return M.metrics(Xte.age_days, model.predict(Xte[d["fcols"]]))


def excluded(master, lo, hi):
    post = master[(master.obs_audio == 1) & (master.obs_yolo == 1) & (master.obs_morph == 1)]
    gone = pre.merge(post[["room", "hour"]], on=["room", "hour"], how="left", indicator=True)
    gone = gone[gone._merge == "left_only"]
    a = age(gone.hour)
    held = (a >= lo) & (a < hi) if hi < P.EDGES[-1] else (a >= lo)
    return len(gone), int((~held).sum()), int(held.sum())


rows, leak = [], []
raw_nd = raw.drop(columns="day")
variants = {
    "original (QC built once on the full series)": with_age(SF.apply_status(raw_nd, orig)),
    "hourly past-only (running median of the day so far)": with_age(hourly_master()),
}
for fold in range(5):
    lo, hi = P.EDGES[fold], P.EDGES[fold + 1]
    internal = fold_internal(lo, hi)
    held_days = {(r, d) for r, d in zip(orig.room, orig.day) if day_range(d, lo, hi)}
    embargo_days = {
        (r, d)
        for r, d in zip(orig.room, orig.day)
        if day_range(d, lo - EMBARGO_DAYS, hi + EMBARGO_DAYS)
    }
    accepted = orig[orig.status != "flag"]
    acc_by_room = {r: g.day.values for r, g in accepted.groupby("room")}
    n_train = n_leak = 0
    for r, d in zip(orig.room, orig.day):
        if (r, d) in embargo_days:
            continue
        n_train += 1
        hist = acc_by_room[r][(acc_by_room[r] >= d - WINDOW) & (acc_by_room[r] < d)]
        n_leak += any((r, h) in held_days for h in hist)
    changed = orig.merge(internal, on=["room", "day"], suffixes=("_orig", "_fold"))
    changed = changed[[(r, d) not in embargo_days for r, d in zip(changed.room, changed.day)]]
    n_changed = int((changed.status_orig != changed.status_fold).sum())
    leak += [
        dict(fold=fold + 1, metric="training room-days", value=n_train),
        dict(
            fold=fold + 1,
            metric="training room-days whose accepted-day history includes withheld-block days",
            value=n_leak,
        ),
        dict(
            fold=fold + 1,
            metric="training room-days whose QC status changes when built from training days only",
            value=n_changed,
        ),
    ]
    variants["fold-internal (training-day history only)"] = with_age(
        SF.apply_status(raw_nd, internal)
    )
    for name, master in variants.items():
        use(master)
        met = evaluate(fold)
        tot, train, held = excluded(master, lo, hi)
        rows.append(
            dict(
                variant=name,
                fold=fold + 1,
                **met,
                excluded_total=tot,
                excluded_train=train,
                excluded_heldout=held
            )
        )
pd.DataFrame(rows).to_csv(OUT / "qc_timing_folds.csv", index=False)
pd.DataFrame(leak).to_csv(OUT / "qc_history_leak.csv", index=False)
obs = raw[raw[STAT].notna()].copy()
obs["later_same_day"] = obs.groupby(["room", "day"]).cumcount(ascending=False)
last_h = obs.groupby(["room", "day"]).hour.transform("max")
stats = {
    "morphometry hours": len(obs),
    "share of hours with later observations in the same day (day decision uses them)": float(
        (obs.later_same_day > 0).mean()
    ),
    "mean hours between an observation and the end of its day": float(
        (obs.day + pd.Timedelta(days=1) - obs.hour).dt.total_seconds().mean() / 3600
    ),
    "mean hours between an observation and the last observation of its day": float(
        (last_h - obs.hour).dt.total_seconds().mean() / 3600
    ),
}
pd.DataFrame({"metric": list(stats), "value": list(stats.values())}).to_csv(
    OUT / "qc_retrospective.csv", index=False
)
