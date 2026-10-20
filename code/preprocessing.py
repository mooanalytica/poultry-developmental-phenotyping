"""Fold-restricted preprocessing: references, purge and embargo, and fold construction without held-out leakage."""

import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets as D
import models as M
import numpy as np
import pandas as pd
import scale_filter
from common import AUDIO_DIR, CACHE, DATA, HATCH, ROOMS, audio_class

MASTER = scale_filter.load_master()  # past-only scale filter already applied (see scale_filter.py)
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

MORPH_RAW_COLS = [
    "bbox_h_mean",
    "bbox_h_p50",
    "bbox_h_p10",
    "bbox_h_p90",
    "bbox_h_std",
    "bbox_h_iqr",
    "bbox_h_skew",
    "bbox_w_p50",
    "area_p50",
    "area_p90",
    "aspect_p50",
    "aspect_iqr",
]
MORPH_REL_COLS = [
    "bbox_h_mean",
    "bbox_h_p50",
    "bbox_h_p10",
    "bbox_h_p90",
    "bbox_h_std",
    "bbox_h_iqr",
    "bbox_w_p50",
    "area_p50",
    "area_p90",
]
MORPH_PASSTHRU_COLS = ["det_conf_mean", "n_det_sampled"]

# age-block edges: quintiles of age over the complete-case windowed table
_X0, _, _ = D.build("A", "fused", augment=True)
EDGES = np.quantile(_X0.age_days, np.linspace(0, 1, 6))
BLOCKS = [f"{EDGES[i]:.0f}-{EDGES[i+1]:.0f}d" for i in range(5)]
PURGE_DAYS = max(D.WINDOWS_TRAIN) / 24.0  # 12 h, the longest rolling window
SEASON_CUT = 88.0  # summer = age < 88 d (build_master.py)


def _trailing_scale_flag():
    """Causal camera-scale check: day median vs trailing 9-day median, per room."""
    d = (
        MASTER.assign(day=MASTER.hour.dt.floor("D"))
        .groupby(["room", "day"])["morph__bbox_h_p50"]
        .median()
        .rename("day_med")
        .reset_index()
    )
    d["trend"] = d.groupby("room").day_med.transform(
        lambda s: s.rolling(9, center=False, min_periods=3).median()
    )
    d["flag"] = (np.log(d.day_med / d.trend).abs() > np.log(1.35)).astype(int)
    return d[["room", "day", "flag"]]


# the legacy row-based trailing check is only part of the retrospective (centred) reproduction;
# in the filtered modes the single past-only filter in scale_filter.py replaces both old checks
TRAILING_FLAG = _trailing_scale_flag() if scale_filter.LEGACY else _trailing_scale_flag().iloc[0:0]

# per-room first-week reference as used in the master table (raw / relative is constant)
ROOM_REF = {}
for _r in ROOMS:
    _g = MASTER[MASTER.room == _r]
    ROOM_REF[_r] = (
        (_g.morph__bbox_h_p50 / _g.morph__bbox_h_p50_rel).median(),
        (_g.morph__area_p50 / _g.morph__area_p50_rel).median(),
    )


def held_out_ages(protocol, fold_id):
    """(lo, hi) age range withheld by an age-block or season fold."""
    if protocol == "AGEBLOCK":
        return EDGES[fold_id], EDGES[fold_id + 1]
    return (0.0, SEASON_CUT) if fold_id == "summer" else (SEASON_CUT, 1e9)


def first_week_ref(df_room, lo=None, hi=None):
    """Median bbox height/area over the room's first 7 days of morphometry, optionally excluding hours with age in [lo, hi)."""
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


def fold_reference(protocol, fold_id, m):
    """Per-room body-size reference (bbox height, area) used to normalise this fold's morphometry. m: frame with room, age_days, n_det_sampled, bbox_h_p50, area_p50."""
    ref = dict(ROOM_REF)
    if protocol == "LORO":
        train_rooms = [r for r in ROOMS if r != fold_id]
        ref[fold_id] = (
            np.mean([ROOM_REF[r][0] for r in train_rooms]),
            np.mean([ROOM_REF[r][1] for r in train_rooms]),
        )
    else:
        lo, hi = held_out_ages(protocol, fold_id)
        for r in ROOMS:
            g = m[m.room == r]
            first = g[g.n_det_sampled.notna()].age_days.min()
            if first + 7 >= lo and first < hi:  # reference window overlaps the held-out range
                ref[r] = first_week_ref(g, lo, hi)
    return ref


def build_morph(protocol, fold_id):
    m = MASTER[
        ["room", "hour", "age_days", "morph__n_det_sampled", "morph__scale_artifact"]
        + [f"morph__{c}" for c in MORPH_RAW_COLS]
    ].copy()
    m.columns = ["room", "hour", "age_days", "n_det_sampled", "master_flag"] + MORPH_RAW_COLS
    m["day"] = m.hour.dt.floor("D")
    m = m.merge(TRAILING_FLAG.rename(columns={"flag": "trailing"}), on=["room", "day"], how="left")
    m["flag"] = (
        (m.master_flag.fillna(0).astype(int) | m.trailing.fillna(0).astype(int)).astype(int).values
    )

    ref = fold_reference(protocol, fold_id, m)

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


AUDIO_FW_DAYS = 7.0
DB_PREFIXES = (
    "band_fan_db",
    "band_mid_db",
    "band_voc_db",
    "band_high_db",
)  # logarithmic -> subtract; everything else -> divide


def audio_first_week_ref(protocol, fold_id, master=None, audio_raw=None):
    """Median of every audio feature over each room's own first AUDIO_FW_DAYS days of audio (hours t0 <= h < t0 + 7 d, t0 = the room's first hour with audio)."""
    master = MASTER if master is None else master
    audio_raw = AUDIO_RAW if audio_raw is None else audio_raw
    bio = [c for c in master.columns if c.startswith("audio_bio__")]
    out = {}
    for r in ROOMS:
        g = master[(master.room == r) & (master.obs_audio == 1)]
        t0 = g.hour.min()
        win = lambda d: d[(d.hour >= t0) & (d.hour < t0 + pd.Timedelta(days=AUDIO_FW_DAYS))]
        gb, ga = win(master[master.room == r]), win(audio_raw[audio_raw.room == r])
        if protocol != "LORO":
            lo, hi = held_out_ages(protocol, fold_id)
            gb = gb[~((gb.age_days >= lo) & (gb.age_days < hi))]
            ga = ga[~((ga.age_days >= lo) & (ga.age_days < hi))]
        out[r] = pd.concat(
            [gb[bio].median(), ga[EQUIP_COLS].median().rename(lambda c: f"audio_equip__{c}")]
        )
    return out


def apply_audio_first_week(m, protocol, fold_id, ref=None):
    """Replace audio_bio__* (raw) and audio_equip__* with values referenced to the room's own first week (ratio, or difference for dB)."""
    ref = audio_first_week_ref(protocol, fold_id) if ref is None else ref
    m = m.copy()
    for c in [c for c in m.columns if c.startswith(("audio_bio__", "audio_equip__"))]:
        base = np.array([ref[r].get(c, np.nan) for r in m.room.values], dtype=float)
        x = m[c].values.astype(float)
        stem = c.split("__", 1)[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            m[c] = x - base if stem.startswith(DB_PREFIXES) else x / base
    return m


def build_audio_equip(protocol, fold_id):
    if (
        AUDIO_REF == "first_week"
    ):  # V2: own-room first-week reference replaces the pooled cross-room median/IQR
        return None
    return _build_audio_equip_pooled(protocol, fold_id)


def _build_audio_equip_pooled(protocol, fold_id):
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
        m = MASTER.drop(
            columns=[
                c
                for c in MASTER.columns
                if c.startswith("morph__") or c.startswith("audio_equip__")
            ]
        )
        m = m.merge(build_morph(protocol, fold_id), on=["room", "hour"], how="left")
        if AUDIO_REF == "first_week":
            # V2: raw audio (equipment-level columns come from the raw audio table) referenced to each room's own first week
            raw_eq = AUDIO_RAW[["room", "hour"] + EQUIP_COLS].rename(
                columns={c: f"audio_equip__{c}" for c in EQUIP_COLS}
            )
            m = apply_audio_first_week(
                m.merge(raw_eq, on=["room", "hour"], how="left"), protocol, fold_id
            )
        else:
            m = m.merge(build_audio_equip(protocol, fold_id), on=["room", "hour"], how="left")
        m["obs_morph"] = m["morph__bbox_h_p50"].notna().astype(int)
        _assembled[key] = m
    return _assembled[key]


OBS = ["obs_audio", "obs_yolo", "obs_morph", "obs_flow"]


def select_rows(m, rows):
    if rows == "complete":  # audio, tracking and morphometry all observed
        return m[(m.obs_audio == 1) & (m.obs_yolo == 1) & (m.obs_morph == 1)].copy()
    if rows == "any":  # at least one modality observed
        return m[(m[OBS] == 1).any(axis=1)].copy()
    return m[m.obs_morph == 1].copy()  # "morph": morphometry observed


def _held_out(protocol, fold_id, df):
    if protocol == "LORO":
        return df.room == fold_id
    if protocol == "AGEBLOCK":
        lo, hi = EDGES[fold_id], EDGES[fold_id + 1]
        return (df.age_days >= lo) & (df.age_days < hi if fold_id < 4 else df.age_days <= hi)
    return df.season == fold_id


AUDIO_REF = os.environ.get("AUDIO_REF", "none")  # variant V2 (off by default): "first_week"
assert AUDIO_REF in ("none", "first_week")
BODY_ONLY = "morph_body_only"
BODY_ONLY_DROP = ("morph__det_conf_mean", "morph__n_det_sampled")
EMBARGO_H = 12.0  # symmetric embargo around each held-out block edge (primary; 24 h = sensitivity)


def purge_zone(protocol, fold_id, embargo_h=EMBARGO_H):
    """(z0, z1) in age days: the held-out interval widened by the embargo on both sides."""
    lo, hi = held_out_ages(protocol, fold_id)
    e = embargo_h / 24.0
    return lo - e, hi + e


def window_in_zone(Xw, f, z0, z1):
    """True where a window reads at least one observation with z0 <= age < z1."""
    out = np.zeros(len(Xw), bool)
    for room, g in f.groupby("room"):
        ages = np.sort(g.age_days.values)
        idx = np.where(Xw.room.values == room)[0]
        a, fa = Xw.age_days.values[idx], Xw.first_age_days.values[idx]
        lower = np.searchsorted(ages, np.maximum(fa, z0), side="left")
        upper = np.minimum(
            np.searchsorted(ages, z1, side="left"), np.searchsorted(ages, a, side="right")
        )
        out[idx] = upper > lower
    return out


def assert_purged(d, protocol, fold_id, embargo_h=EMBARGO_H, strict=True):
    """Re-derive every training window's observations independently and fail if any lies inside the purge zone."""
    f, Xw, tr = d["f"], d["Xw"], d["tr"]
    ho = _held_out(protocol, fold_id, Xw)
    assert not (tr & ho).any(), "a training window is anchored in the held-out set"
    if protocol == "LORO":
        assert not (Xw.loc[tr, "room"] == fold_id).any(), "held-out room present in training"
        return dict(checked=int(tr.sum()), violations=0)
    lo, hi = held_out_ages(protocol, fold_id)
    e = pd.Timedelta(hours=embargo_h)
    t0 = (HATCH + pd.Timedelta(days=lo) - e).to_datetime64()
    t1 = (HATCH + pd.Timedelta(days=min(hi, 1e5)) + e).to_datetime64()
    viol = checked = 0
    for room, g in f.groupby("room"):
        hrs = np.sort(g.hour.values)
        cs = np.concatenate([[0], np.cumsum((hrs >= t0) & (hrs < t1))])
        X = Xw[tr & (Xw.room == room)]
        if X.empty:
            continue
        h, W = X.hour.values, X.window_h.values
        i = np.searchsorted(hrs, h)
        first = np.where(
            W > 1, np.searchsorted(hrs, h - W.astype("timedelta64[h]"), side="right"), i
        )
        back = i - (W - 1)
        if D.SLOPE_MODE == "row":  # legacy slope also reads the row W-1 rows back
            first = np.where(
                (W > 1) & (back >= 0), np.minimum(first, np.clip(back, 0, None)), first
            )
        viol += int(((cs[i + 1] - cs[first]) > 0).sum())
        checked += len(X)
    if viol and not strict:
        return dict(checked=checked, violations=viol)
    if viol:
        raise AssertionError(
            f"{protocol} fold {fold_id}: {viol} of {checked} training windows contain an "
            f"observation inside the held-out interval or its {embargo_h:g} h embargo"
        )
    return dict(checked=checked, violations=0)


def fold(
    protocol,
    fold_id,
    feature_set="fused",
    rows="complete",
    purge="window",
    embargo_h=EMBARGO_H,
    check=None,
):
    """Windowed matrix, train and test masks (6 h evaluation windows) and feature list for one fold."""
    f = select_rows(assemble(protocol, fold_id), rows)
    train_rows = ~_held_out(protocol, fold_id, f)
    body_only = (
        feature_set == BODY_ONLY
    )  # V3: morphometry minus detector-health / flag / time-of-day columns
    cand = D.block_cols(
        f, D.FEATURE_SETS["morph_only"] if body_only else D.FEATURE_SETS[feature_set]
    )
    if body_only:
        cand = [c for c in cand if c not in BODY_ONLY_DROP]
    cols = [c for c in cand if f.loc[train_rows, c].notna().sum() > 50]
    Xw = D.windowise(f, cols, D.WINDOWS_TRAIN)
    fcols = (
        [f"{c}__m" for c in cols]
        + [f"{c}__s" for c in cols]
        + ([] if body_only else ["hour_sin", "hour_cos"] + OBS)
    )
    fcols = [c for c in fcols if c in Xw.columns]
    ho = _held_out(protocol, fold_id, Xw)
    tr, te = ~ho, ho & (Xw.window_h == D.WINDOWS_EVAL)
    info = dict(
        purge=purge,
        embargo_h=embargo_h,
        n_train_before_purge=int(tr.sum()),
        removed_total=0,
        removed_anchor_in_zone=0,
        removed_reach_only=0,
    )
    if protocol != "LORO" and purge != "none":
        z0, z1 = purge_zone(protocol, fold_id, embargo_h)
        in_zone = ((Xw.age_days >= z0) & (Xw.age_days < z1)).values
        if purge == "anchor":
            drop = in_zone
        elif purge == "window":
            drop = in_zone | window_in_zone(Xw, f, z0, z1)
        else:
            raise ValueError(purge)
        drop = pd.Series(drop, index=Xw.index) & tr
        info.update(
            removed_total=int(drop.sum()),
            removed_anchor_in_zone=int((drop & in_zone).sum()),
            removed_reach_only=int((drop & ~in_zone).sum()),
        )
        tr = tr & ~drop
    fcols = [c for c in fcols if Xw.loc[tr, c].nunique(dropna=True) >= 2]
    d = dict(Xw=Xw, tr=tr, te=te, fcols=fcols, cols=cols, f=f, purge_info=info)
    if check if check is not None else (purge == "window" or protocol == "LORO"):
        d["purge_check"] = assert_purged(d, protocol, fold_id, embargo_h)
    return d


import functools
import hashlib
import json
import os

import sklearn


@functools.lru_cache(maxsize=None)
def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _master_hash():
    return _file_sha256(str(DATA / scale_filter.MASTER_FILE))


def _audio_hash():
    return {r: _file_sha256(str(AUDIO_DIR / f"audio_hourly_{r}_denoised.csv")) for r in ROOMS}


CODE_FILES = ("datasets.py", "preprocessing.py", "models.py", "scale_filter.py", "common.py")


def _code_hash():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(_file_sha256(str(Path(__file__).resolve().parent / f)).encode())
    return h.hexdigest()


def cache_components(protocol, fold_id, feature_set, purge, embargo_h):
    """Everything a fitted fold depends on. fit_fold's cache file name is the hash of this dict."""
    return dict(
        protocol=protocol,
        fold_id=str(fold_id),
        feature_set=feature_set,
        master_sha256=_master_hash(),
        audio_sha256=_audio_hash(),
        scale_filter_mode=scale_filter.MODE,
        slope_mode=D.SLOPE_MODE,
        slope_min_frac=D.SLOPE_MIN_FRAC,
        slope_min_obs=D.SLOPE_MIN_OBS,
        windows_train=list(D.WINDOWS_TRAIN),
        window_eval=D.WINDOWS_EVAL,
        purge=purge if protocol != "LORO" else "n/a",
        embargo_h=embargo_h if protocol != "LORO" else 0.0,
        audio_ref=AUDIO_REF,
        hgb_params=M.hgb().get_params(),
        code_sha256=_code_hash(),
        versions=dict(sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__),
    )


def cache_key(protocol, fold_id, feature_set, purge="window", embargo_h=12.0):
    return hashlib.sha256(
        json.dumps(
            cache_components(protocol, fold_id, feature_set, purge, embargo_h),
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()


def fit_fold(protocol, fold_id, feature_set="fused", purge="window", embargo_h=EMBARGO_H):
    """Fit models.hgb() on one fold (complete-case rows); cached on disk."""
    key = cache_key(protocol, fold_id, feature_set, purge, embargo_h)
    path = CACHE / f"{protocol}_{fold_id}_{feature_set}_{key[:20]}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    d = fold(protocol, fold_id, feature_set, purge=purge, embargo_h=embargo_h)
    Xtr, Xte = d["Xw"][d["tr"]], d["Xw"][d["te"]]
    mdl = M.hgb().fit(Xtr[d["fcols"]], Xtr.age_days)
    pred = mdl.predict(Xte[d["fcols"]])
    res = dict(
        model=mdl,
        fcols=d["fcols"],
        cols=d["cols"],
        pred=pd.DataFrame(
            dict(room=Xte.room.values, hour=Xte.hour.values, y=Xte.age_days.values, pred=pred)
        ),
        met=M.metrics(Xte.age_days, pred),
        n_train=len(Xtr),
        purge_info=d["purge_info"],
        cache_key=key,
    )
    path.write_bytes(pickle.dumps(res))
    return res


def block_of(c):
    for b in ("morph", "yolo", "flow", "audio_bio", "audio_equip"):
        if c.startswith(b + "__"):
            return b
    return "time/flags"
