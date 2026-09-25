"""Fold-restricted preprocessing and fold construction.

Every statistic used to normalise features is computed so that no information from
the held-out part of a fold reaches either the training rows or the held-out rows'
normalisation:

  * Morphometric scale reference (bbox height/area relative to the room's first
    observed week, morpho_hourly.py). Training rooms keep their own reference. In LORO
    the held-out room is normalised with the mean reference of the training rooms. In
    age-block folds the reference is computed without any hours inside the held-out
    age range.
  * Equipment-sensitive audio (level/energy) is standardised by median/IQR. In LORO the
    held-out room uses the pooled training rooms' median/IQR; in age-block folds each
    room's median/IQR excludes the held-out age range.
  * Camera-scale artifact hours flagged in the master table are excluded, and a causal
    (trailing 9-day median) check is applied on top.
  * Feature columns are kept if they have more than 50 non-missing training values in
    the fold.

fold() returns the windowed matrix, train/test masks and feature list for one fold;
fit_fold() fits models.hgb() on it and caches the fitted model and predictions.
"""
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from common import DATA, CACHE, AUDIO_DIR, ROOMS, HATCH, audio_class
import datasets as D
import models as M

# ---------------------------------------------------------------- inputs
MASTER = pd.read_csv(DATA / "master_hourly.csv", parse_dates=["hour", "date"])
MASTER["age_days"] = (MASTER.hour - HATCH).dt.total_seconds() / 86400.0

_au = []
for _r in ROOMS:
    _d = pd.read_csv(AUDIO_DIR / f"audio_hourly_{_r}_denoised.csv")
    _d["hour"] = pd.to_datetime(_d.time).dt.floor("h")
    _d["room"] = _r
    _au.append(_d)
AUDIO_RAW = pd.concat(_au, ignore_index=True).drop(columns=["time"])
EQUIP_COLS = [c for c in AUDIO_RAW.columns if audio_class(c) == "equip"]
AUDIO_RAW = AUDIO_RAW[["room", "hour"] + EQUIP_COLS]
AUDIO_RAW["age_days"] = (AUDIO_RAW.hour - HATCH).dt.total_seconds() / 86400.0

MORPH_RAW_COLS = ["bbox_h_mean", "bbox_h_p50", "bbox_h_p10", "bbox_h_p90", "bbox_h_std",
                  "bbox_h_iqr", "bbox_h_skew", "bbox_w_p50", "area_p50", "area_p90",
                  "aspect_p50", "aspect_iqr"]
MORPH_REL_COLS = ["bbox_h_mean", "bbox_h_p50", "bbox_h_p10", "bbox_h_p90", "bbox_h_std",
                  "bbox_h_iqr", "bbox_w_p50", "area_p50", "area_p90"]
MORPH_PASSTHRU_COLS = ["det_conf_mean", "n_det_sampled"]

# age-block edges: quintiles of age over the complete-case windowed table
_X0, _, _ = D.build("A", "fused", augment=True)
EDGES = np.quantile(_X0.age_days, np.linspace(0, 1, 6))
BLOCKS = [f"{EDGES[i]:.0f}-{EDGES[i+1]:.0f}d" for i in range(5)]
SEASON_CUT = 88.0          # summer = age < 88 d (build_master.py)


def _trailing_scale_flag():
    """Causal camera-scale check: day median vs trailing 9-day median, per room."""
    d = (MASTER.assign(day=MASTER.hour.dt.floor("D"))
               .groupby(["room", "day"])["morph__bbox_h_p50"].median()
               .rename("day_med").reset_index())
    d["trend"] = (d.groupby("room").day_med
                    .transform(lambda s: s.rolling(9, center=False, min_periods=3).median()))
    d["flag"] = (np.log(d.day_med / d.trend).abs() > np.log(1.35)).astype(int)
    return d[["room", "day", "flag"]]


TRAILING_FLAG = _trailing_scale_flag()

# per-room first-week reference as used in the master table (raw / relative is constant)
ROOM_REF = {}
for _r in ROOMS:
    _g = MASTER[MASTER.room == _r]
    ROOM_REF[_r] = ((_g.morph__bbox_h_p50 / _g.morph__bbox_h_p50_rel).median(),
                    (_g.morph__area_p50 / _g.morph__area_p50_rel).median())


def held_out_ages(protocol, fold_id):
    """(lo, hi) age range withheld by an age-block or season fold."""
    if protocol == "AGEBLOCK":
        return EDGES[fold_id], EDGES[fold_id + 1]
    return (0.0, SEASON_CUT) if fold_id == "summer" else (SEASON_CUT, 1e9)


def first_week_ref(df_room, lo=None, hi=None):
    """Median bbox height/area over the room's first 7 days of morphometry,
    optionally excluding hours with age in [lo, hi)."""
    df_room = df_room[df_room.n_det_sampled.notna()]
    if df_room.empty:
        return np.nan, np.nan
    cutoff = df_room.age_days.min() + 7
    w = df_room[df_room.age_days <= cutoff]
    if lo is not None:
        w = w[~((w.age_days >= lo) & (w.age_days < hi))]
    if w.empty:
        return np.nan, np.nan
    return w["bbox_h_p50"].median(), w["area_p50"].median()


def build_morph(protocol, fold_id):
    m = MASTER[["room", "hour", "age_days", "morph__n_det_sampled", "morph__scale_artifact"]
               + [f"morph__{c}" for c in MORPH_RAW_COLS]].copy()
    m.columns = ["room", "hour", "age_days", "n_det_sampled", "master_flag"] + MORPH_RAW_COLS
    m["day"] = m.hour.dt.floor("D")
    m = m.merge(TRAILING_FLAG.rename(columns={"flag": "trailing"}), on=["room", "day"], how="left")
    m["flag"] = (m.master_flag.fillna(0).astype(int) | m.trailing.fillna(0).astype(int)).astype(int).values

    ref = dict(ROOM_REF)
    if protocol == "LORO":
        train_rooms = [r for r in ROOMS if r != fold_id]
        ref[fold_id] = (np.mean([ROOM_REF[r][0] for r in train_rooms]),
                        np.mean([ROOM_REF[r][1] for r in train_rooms]))
    else:
        lo, hi = held_out_ages(protocol, fold_id)
        for r in ROOMS:
            g = m[m.room == r]
            first = g[g.n_det_sampled.notna()].age_days.min()
            if first + 7 >= lo and first < hi:        # reference window overlaps the held-out range
                ref[r] = first_week_ref(g, lo, hi)

    m["ref_h"] = m.room.map(lambda r: ref[r][0])
    m["ref_a"] = m.room.map(lambda r: ref[r][1])
    for c in MORPH_REL_COLS:
        base = "bbox_h" if c.startswith("bbox_h") else ("bbox_w" if c == "bbox_w_p50" else "area")
        denom = m.ref_h if base in ("bbox_h", "bbox_w") else m.ref_a
        m[c + "_rel"] = m[c] / denom

    blank_cols = MORPH_RAW_COLS + [c + "_rel" for c in MORPH_REL_COLS]
    m.loc[m.flag == 1, blank_cols] = np.nan
    m["scale_artifact"] = m.flag.where(m.master_flag.notna())
    out_cols = blank_cols + ["scale_artifact"]
    out = m[["room", "hour"] + out_cols].rename(columns={c: f"morph__{c}" for c in out_cols})
    passthru = MASTER[["room", "hour"] + [f"morph__{c}" for c in MORPH_PASSTHRU_COLS]]
    return out.merge(passthru, on=["room", "hour"], how="left")


def build_audio_equip(protocol, fold_id):
    au = AUDIO_RAW.copy()
    out = au[["room", "hour"]].copy()
    for c in EQUIP_COLS:
        vals = pd.Series(index=au.index, dtype=float)
        for r in ROOMS:
            rm = au.room == r
            if protocol == "LORO" and r == fold_id:
                calib = au[au.room != fold_id]
            elif protocol != "LORO":
                lo, hi = held_out_ages(protocol, fold_id)
                sub = au[rm]
                calib = sub[~((sub.age_days >= lo) & (sub.age_days < hi))]
            else:
                calib = au[rm]
            med = calib[c].median()
            q75, q25 = calib[c].quantile(0.75), calib[c].quantile(0.25)
            iqr = (q75 - q25) if (q75 - q25) != 0 else np.nan
            vals[rm] = (au.loc[rm, c] - med) / iqr
        out[f"audio_equip__{c}"] = vals
    return out


_assembled = {}


def assemble(protocol, fold_id):
    """Master table with morphometric and audio-level features normalised for this fold."""
    key = (protocol, fold_id)
    if key not in _assembled:
        m = MASTER.drop(columns=[c for c in MASTER.columns
                                  if c.startswith("morph__") or c.startswith("audio_equip__")])
        m = (m.merge(build_morph(protocol, fold_id), on=["room", "hour"], how="left")
              .merge(build_audio_equip(protocol, fold_id), on=["room", "hour"], how="left"))
        m["obs_morph"] = m["morph__bbox_h_p50"].notna().astype(int)
        _assembled[key] = m
    return _assembled[key]


OBS = ["obs_audio", "obs_yolo", "obs_morph", "obs_flow"]


def select_rows(m, rows):
    if rows == "complete":          # audio, tracking and morphometry all observed
        return m[(m.obs_audio == 1) & (m.obs_yolo == 1) & (m.obs_morph == 1)].copy()
    if rows == "any":               # at least one modality observed
        return m[(m[OBS] == 1).any(axis=1)].copy()
    return m[m.obs_morph == 1].copy()   # "morph": morphometry observed


def _held_out(protocol, fold_id, df):
    if protocol == "LORO":
        return df.room == fold_id
    if protocol == "AGEBLOCK":
        lo, hi = EDGES[fold_id], EDGES[fold_id + 1]
        return (df.age_days >= lo) & (df.age_days < hi if fold_id < 4 else df.age_days <= hi)
    return df.season == fold_id


def fold(protocol, fold_id, feature_set="fused", rows="complete"):
    """Windowed matrix, train mask, test mask (evaluation window) and feature list."""
    f = select_rows(assemble(protocol, fold_id), rows)
    train_rows = ~_held_out(protocol, fold_id, f)
    cand = D.block_cols(f, D.FEATURE_SETS[feature_set])
    cols = [c for c in cand if f.loc[train_rows, c].notna().sum() > 50]
    Xw = D.windowise(f, cols, D.WINDOWS_TRAIN)
    fcols = [f"{c}__m" for c in cols] + [f"{c}__s" for c in cols] + ["hour_sin", "hour_cos"] + OBS
    fcols = [c for c in fcols if c in Xw.columns]
    ho = _held_out(protocol, fold_id, Xw)
    tr, te = ~ho, ho & (Xw.window_h == D.WINDOWS_EVAL)
    fcols = [c for c in fcols if Xw.loc[tr, c].nunique(dropna=True) >= 2]   # drop constant columns
    return dict(Xw=Xw, tr=tr, te=te, fcols=fcols, cols=cols, f=f)


def fit_fold(protocol, fold_id, feature_set="fused"):
    """Fit models.hgb() on one fold (complete-case rows); cached on disk."""
    path = CACHE / f"{protocol}_{fold_id}_{feature_set}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    d = fold(protocol, fold_id, feature_set)
    Xtr, Xte = d["Xw"][d["tr"]], d["Xw"][d["te"]]
    mdl = M.hgb().fit(Xtr[d["fcols"]], Xtr.age_days)
    pred = mdl.predict(Xte[d["fcols"]])
    res = dict(model=mdl, fcols=d["fcols"], cols=d["cols"],
               pred=pd.DataFrame(dict(room=Xte.room.values, hour=Xte.hour.values,
                                      y=Xte.age_days.values, pred=pred)),
               met=M.metrics(Xte.age_days, pred), n_train=len(Xtr))
    path.write_bytes(pickle.dumps(res))
    return res


def block_of(c):
    for b in ("morph", "yolo", "flow", "audio_bio", "audio_equip"):
        if c.startswith(b + "__"):
            return b
    return "time/flags"
