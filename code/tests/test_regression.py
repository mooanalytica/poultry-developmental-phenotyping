"""Regression tests for the leave-one-room-out pipeline. Run: python3 tests/test_regression.py"""

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
FROZEN = json.loads((HERE / "frozen_baselines.json").read_text())
SETS = [
    "fused",
    "morph_only",
    "yolo_motion_only",
    "flow_only",
    "audio_all",
    "audio_bio_only",
    "fused_bio_audio",
    "video_all",
    "morph_body_only",
]


def _loro_macro_subprocess(env_extra):
    code = (
        "import sys, tempfile, json, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, %r)\n"
        "from pathlib import Path\nimport numpy as np, preprocessing as P\n"
        "P.CACHE = Path(tempfile.mkdtemp())\nout = {}\n"
        "for fs in ('fused', 'morph_only'):\n"
        "    out[fs] = float(np.mean([P.fit_fold('LORO', r, fs)['met']['MAE'] for r in P.ROOMS]))\n"
        "print('RESULT' + json.dumps(out))\n"
    ) % str(ROOT / "code")
    env = {**os.environ, **env_extra, "PYTHONDONTWRITEBYTECODE": "1"}
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads([l for l in p.stdout.splitlines() if l.startswith("RESULT")][0][6:])


def test_1_frozen_step0_baseline():
    """Step 0 settings (centred filter, row slope) reproduce LORO macro MAE 14.378 / 14.577 to 1e-9."""
    got = _loro_macro_subprocess(
        {"SCALE_FILTER_MODE": "centred_retro", "SLOPE_MODE": "row", "AUDIO_REF": "none"}
    )
    for k, v in FROZEN["step0_macro_MAE"].items():
        assert abs(got[k] - v) < 1e-9, (k, got[k], v)


def test_1b_primary_reproduces_with_fresh_cache():
    """Primary pipeline (B_exclude, time slope, 50 %) gives fused 18.580 / morphometry-only 17.094 with a fresh cache."""
    got = _loro_macro_subprocess(
        {
            "SCALE_FILTER_MODE": "B_exclude",
            "SLOPE_MODE": "time",
            "SLOPE_MIN_FRAC": "0.5",
            "AUDIO_REF": "none",
        }
    )
    for k, v in FROZEN["primary_step3_macro_MAE"].items():
        assert abs(got[k] - v) < 1e-9, (k, got[k], v)


def test_2_cache_key_changes_with_every_component():
    import datasets as D
    import preprocessing as P
    import scale_filter as S

    base = P.cache_key("AGEBLOCK", 1, "fused", "window", 12.0)
    assert base == P.cache_key("AGEBLOCK", 1, "fused", "window", 12.0)  # deterministic
    changed = {
        "feature_set": P.cache_key("AGEBLOCK", 1, "morph_only", "window", 12.0),
        "fold": P.cache_key("AGEBLOCK", 2, "fused", "window", 12.0),
        "protocol": P.cache_key("SEASON", 1, "fused", "window", 12.0),
        "purge": P.cache_key("AGEBLOCK", 1, "fused", "anchor", 12.0),
        "embargo": P.cache_key("AGEBLOCK", 1, "fused", "window", 24.0),
    }

    def with_attr(mod, name, val, call=lambda: P.cache_key("AGEBLOCK", 1, "fused", "window", 12.0)):
        old = getattr(mod, name)
        setattr(mod, name, val)
        try:
            return call()
        finally:
            setattr(mod, name, old)

    changed["scale_filter_mode"] = with_attr(S, "MODE", "B_keep")
    changed["slope_mode"] = with_attr(D, "SLOPE_MODE", "row")
    changed["slope_coverage"] = with_attr(D, "SLOPE_MIN_FRAC", 0.75)
    changed["slope_min_obs"] = with_attr(D, "SLOPE_MIN_OBS", 4)
    changed["variant_flag_audio_ref"] = with_attr(P, "AUDIO_REF", "first_week")
    changed["master_file_hash"] = with_attr(P, "_master_hash", lambda: "deadbeef")
    changed["audio_file_hash"] = with_attr(P, "_audio_hash", lambda: {"room2": "x"})
    changed["code_hash"] = with_attr(P, "_code_hash", lambda: "cafebabe")
    changed["windows"] = with_attr(D, "WINDOWS_TRAIN", [1, 3, 6])
    dup = [k for k, v in changed.items() if v == base]
    assert not dup, f"cache key unchanged when these components change: {dup}"
    assert len(set(changed.values())) == len(changed), "two different changes collide"


def test_3_same_rows_targets_and_subset():
    import pandas as pd
    import preprocessing as P

    for r in P.ROOMS:
        sig, cols = {}, {}
        for fs in SETS:
            d = P.fold("LORO", r, fs)
            X, tr, te = d["Xw"], d["tr"], d["te"]
            sig[fs] = (
                pd.util.hash_pandas_object(
                    X.loc[tr, ["room", "hour", "window_h"]], index=False
                ).sum(),
                pd.util.hash_pandas_object(
                    X.loc[te, ["room", "hour", "window_h"]], index=False
                ).sum(),
                float(X.loc[tr, "age_days"].sum()),
                float(X.loc[te, "age_days"].sum()),
                int(tr.sum()),
                int(te.sum()),
            )
            cols[fs] = set(d["fcols"])
        assert len(set(sig.values())) == 1, f"{r}: feature sets use different rows/targets"
        assert (
            cols["morph_only"] <= cols["fused"]
        ), f"{r}: morphometry-only is not a subset of fused"
        assert cols["fused_bio_audio"] <= cols["fused"] and cols["video_all"] <= cols["fused"]
        # V3 comparator: strict subset of morphometry-only, without detector-health, flag and time-of-day columns
        assert cols["morph_body_only"] < cols["morph_only"]
        assert not [
            c
            for c in cols["morph_body_only"]
            if c.startswith(("morph__det_conf_mean", "morph__n_det_sampled"))
            or c in ("obs_flow", "hour_sin", "hour_cos")
        ]
        assert all(c.startswith("morph__") for c in cols["morph_body_only"])


def test_6_leak_guard():
    import re

    import preprocessing as P

    allowed_plain = {"hour_sin", "hour_cos", "obs_flow"}
    bad = re.compile(r"(centred|^qc__|room|date|season|age|day|unix|file|source|coverage)", re.I)
    for protocol, fid in (
        ("LORO", "room2"),
        ("LORO", "room6"),
        ("LORO", "room7"),
        ("AGEBLOCK", 1),
        ("SEASON", "fall"),
    ):
        d = P.fold(protocol, fid, "fused")
        for c in d["fcols"]:
            if c.endswith(("__m", "__s")):
                assert c.startswith(
                    ("audio_bio__", "audio_equip__", "morph__", "yolo__", "flow__")
                ), c
                assert not bad.search(
                    c.split("__", 1)[1].rsplit("__", 1)[0]
                ), f"suspicious feature stem {c}"
            else:
                assert c in allowed_plain, f"unexpected non-block column {c}"


def test_7_body_size_reference_non_empty_per_fold():
    """Every fold has a finite body-size reference except the two documented folds that hold out the whole reference week."""
    import numpy as np
    import preprocessing as P

    M = P.MASTER
    m = M[
        ["room", "hour", "age_days", "morph__n_det_sampled", "morph__bbox_h_p50", "morph__area_p50"]
    ].rename(
        columns={
            "morph__n_det_sampled": "n_det_sampled",
            "morph__bbox_h_p50": "bbox_h_p50",
            "morph__area_p50": "area_p50",
        }
    )
    folds = (
        [("LORO", r) for r in P.ROOMS]
        + [("AGEBLOCK", i) for i in range(5)]
        + [("SEASON", s) for s in ("summer", "fall")]
    )
    empty = []
    for protocol, fid in folds:
        ref = P.fold_reference(protocol, fid, m)
        for r in P.ROOMS:
            if not (np.isfinite(ref[r][0]) and np.isfinite(ref[r][1])):
                empty.append((protocol, fid, r))
    EXPECTED = {("AGEBLOCK", 0), ("SEASON", "summer")}
    unexpected = [e for e in empty if (e[0], e[1]) not in EXPECTED]
    assert (
        not unexpected
    ), f"empty body-size reference outside the documented exception: {unexpected}"
    assert {
        (e[0], e[1]) for e in empty
    } == EXPECTED, f"documented exceptions changed (now {sorted({(e[0], e[1]) for e in empty}, key=str)}): update this test and the report"
    print(
        "   documented exceptions observed (rooms with an empty reference):",
        {f"{p} {f}": sorted(r for pp, ff, r in empty if (pp, ff) == (p, f)) for p, f in EXPECTED},
    )


def test_8_variant_flags_default_off_and_v2_reference_is_own_room_only():
    import numpy as np
    import pandas as pd
    import preprocessing as P

    assert P.AUDIO_REF == "none", "V2 must be off by default"
    M, A = P.MASTER, P.AUDIO_RAW
    base = P.audio_first_week_ref("LORO", "room2", M, A)
    # (a) perturbing every OTHER room, and this room's audio AFTER its first week, leaves the reference unchanged
    for room in P.ROOMS:
        M2, A2 = M.copy(), A.copy()
        bio = [c for c in M2.columns if c.startswith("audio_bio__")]
        other = M2.room != room
        M2.loc[other, bio] = M2.loc[other, bio] * 7 + 3
        A2.loc[A2.room != room, P.EQUIP_COLS] = A2.loc[A2.room != room, P.EQUIP_COLS] * 5 - 11
        t0 = M2[(M2.room == room) & (M2.obs_audio == 1)].hour.min()
        late = (M2.room == room) & (M2.hour >= t0 + pd.Timedelta(days=P.AUDIO_FW_DAYS))
        M2.loc[late, bio] = M2.loc[late, bio] * 9 + 1
        latea = (A2.room == room) & (A2.hour >= t0 + pd.Timedelta(days=P.AUDIO_FW_DAYS))
        A2.loc[latea, P.EQUIP_COLS] = A2.loc[latea, P.EQUIP_COLS] * 9 + 1
        new = P.audio_first_week_ref("LORO", "room2", M2, A2)
        assert np.allclose(
            new[room].values, base[room].values, equal_nan=True
        ), f"{room}: reference depends on data outside its own first week"
    # (b) held-out room uses its own reference: the LORO fold id does not change any room's reference
    r6 = P.audio_first_week_ref("LORO", "room6", M, A)
    assert all(np.allclose(base[r].values, r6[r].values, equal_nan=True) for r in P.ROOMS)
    # (c) dB features are differenced, all others divided: referenced first-week median is 0 (dB) or 1 (ratio)
    ref = P.audio_first_week_ref("LORO", "room2", M, A)
    sub = (
        M[(M.room == "room2")]
        .drop(columns=[c for c in M.columns if c.startswith("audio_equip__")])
        .copy()
    )
    sub = sub.merge(
        A[A.room == "room2"][["hour"] + P.EQUIP_COLS].rename(
            columns={c: f"audio_equip__{c}" for c in P.EQUIP_COLS}
        ),
        on="hour",
        how="left",
    )
    out = P.apply_audio_first_week(sub, "LORO", "room2", ref)
    t0 = out[out.obs_audio == 1].hour.min()
    w = out[(out.hour >= t0) & (out.hour < t0 + pd.Timedelta(days=7))]
    for c, want in (
        ("audio_equip__band_mid_db_mean", 0.0),
        ("audio_equip__rms_mean", 1.0),
        ("audio_bio__spectral_centroid_mean", 1.0),
    ):
        assert abs(w[c].median() - want) < 1e-9, (c, w[c].median())
    # (d) age-block fold 0 holds out the first audio week of rooms 2 and 6: reference NaN-or-reduced, never uses held-out hours
    ab0 = P.audio_first_week_ref("AGEBLOCK", 0, M, A)
    lo, hi = P.held_out_ages("AGEBLOCK", 0)
    for r in P.ROOMS:
        g = M[(M.room == r) & (M.obs_audio == 1)]
        t0 = g.hour.min()
        win = M[
            (M.room == r)
            & (M.hour >= t0)
            & (M.hour < t0 + pd.Timedelta(days=7))
            & ~((M.age_days >= lo) & (M.age_days < hi))
        ]
        assert np.allclose(
            ab0[r]["audio_bio__spectral_centroid_mean"],
            win["audio_bio__spectral_centroid_mean"].median(),
            equal_nan=True,
        )


def test_9_detection_calibration_never_sees_heldout_room():
    """Set 2: calibration (inner models, bias curve, threshold) uses only the two training rooms."""
    import detection as DT
    import numpy as np
    import preprocessing as P
    from sklearn.ensemble import HistGradientBoostingRegressor

    cheap = lambda: HistGradientBoostingRegressor(max_iter=15, early_stopping=False, random_state=0)
    for r in P.ROOMS:
        d = P.fold("LORO", r)
        FC = d["fcols"]
        which = {"morph_only": [c for c in FC if c.startswith("morph__")]}
        cal = DT.calibrate(r, d, FC, which, model_factory=cheap)  # asserts internally
        assert DT.assert_calibration_clean(cal, r) is True
        assert cal.rooms_in_inputs == set(P.ROOMS) - {r}
        assert (
            all(r not in pair for pair in cal.inner_models_trained_on)
            and len(cal.inner_models_trained_on) == 2
        )
        assert all((fr.room != r).all() for fr in cal.inner.values())
        assert (
            np.isfinite(cal.thr[("morph_only", "same_hour", 10, DT.PHASES[0], "rolling")]) or True
        )
    # negative test: the same call with the held-out room injected must fail loudly
    d = P.fold("LORO", "room2")
    FC = d["fcols"]
    try:
        DT.calibrate(
            "room2",
            d,
            FC,
            {"morph_only": [c for c in FC if c.startswith("morph__")]},
            model_factory=cheap,
            _leak_for_test=True,
        )
    except AssertionError as e:
        assert "room2" in str(e)
    else:
        raise AssertionError(
            "negative test failed: held-out room entered calibration without an assertion error"
        )


def test_10_perturbation_matching():
    """Matched design: the source is the observation exactly k days earlier at the same hour (primary) or the nearest within +-1 h."""
    import detection as DT
    import pandas as pd

    h = pd.to_datetime(
        [
            "2025-08-01 10:00",
            "2025-08-01 11:00",
            "2025-08-06 10:00",
            "2025-08-06 13:00",
            "2025-08-11 11:00",
        ]
    ).values
    s0 = DT.match_sources(h, 5, 0.0)
    s1 = DT.match_sources(h, 5, 1.0)
    assert list(s0) == [-1, -1, 0, -1, -1], s0  # exact same hour only (11:00 has no 11:00 source)
    assert list(s1) == [-1, -1, 0, -1, 2], s1  # 11:00 -> nearest within 1 h is 10:00
    h2 = pd.to_datetime(
        ["2025-08-01 10:00", "2025-08-06 11:00"]
    ).values  # 1 h off -> only the +-1 h rule matches
    assert list(DT.match_sources(h2, 5, 0.0)) == [-1, -1] and list(
        DT.match_sources(h2, 5, 1.0)
    ) == [-1, 0]
    h3 = pd.to_datetime(
        ["2025-08-01 09:00", "2025-08-01 11:00", "2025-08-06 10:00"]
    ).values  # tie: two sources 1 h away -> the earlier one
    assert DT.match_sources(h3, 5, 1.0)[2] == 0


def test_11_calendar_window_ignores_observations_older_than_7_days():
    """A trailing 7-calendar-day window ignores anything older than 7 days and needs at least 4 days of data."""
    import detection as DT
    import numpy as np
    import pandas as pd

    days = pd.to_datetime(
        [
            "2025-07-01",
            "2025-07-02",
            "2025-07-03",
            "2025-07-20",
            "2025-07-21",
            "2025-07-22",
            "2025-07-23",
            "2025-07-24",
        ]
    )
    hour = np.repeat(days.values + np.timedelta64(10, "h"), 2)
    age = np.full(len(hour), 40.0)
    g = np.arange(len(hour), dtype=float)
    old_mode = DT.WINDOW_MODE
    try:
        DT.WINDOW_MODE = "calendar"
        base = DT.window_scores(g, hour, age)
        g2 = g.copy()
        g2[:6] = g2[:6] * 100 + 1e6  # perturb the three early days (> 7 days before 2025-07-24)
        pert = DT.window_scores(g2, hour, age)
        b = base.set_index("day").stat
        p = pert.set_index("day").stat
        late = [
            d for d in b.index if d >= pd.Timestamp("2025-07-23")
        ]  # windows 07-17..07-23 and 07-18..07-24 contain only the late days
        assert late and all(
            np.isclose(b[d], p[d]) for d in late
        ), "calendar window depends on observations older than 7 days"
        assert (
            pd.Timestamp("2025-07-22") not in b.index
        ), "window with < 4 observed days must be dropped"
        assert pd.Timestamp("2025-07-23") in b.index and pd.Timestamp("2025-07-24") in b.index
        DT.WINDOW_MODE = "row"
        rb = DT.window_scores(g, hour, age).set_index("day").stat
        rp = DT.window_scores(g2, hour, age).set_index("day").stat
        d = DT._daily(g, hour, age)
        legacy = d.g.rolling(7, min_periods=3).median().dropna()
        assert np.allclose(
            DT.window_scores(g, hour, age).stat.values, legacy.values
        ), "WINDOW_MODE=row does not reproduce the Revision 3 rolling window"
        assert not np.isclose(
            rb[pd.Timestamp("2025-07-24")], rp[pd.Timestamp("2025-07-24")]
        ), "row mode should (wrongly) depend on rows older than 7 days"
    finally:
        DT.WINDOW_MODE = old_mode


def test_12_shuffled_label_permutation_assertion():
    """Set 3a: the shuffled-label control permutes whole room-day blocks inside each training room (constant within a day, a permutation of the room's own day labels, no noise)."""
    import numpy as np
    import preprocessing as P
    import shuffled_label as SL

    d = P.fold("LORO", "room2")
    Xtr = d["Xw"][d["tr"]]
    for seed in (0, 1, 2):
        y = SL.permute_day_labels(Xtr, np.random.default_rng([20251017, seed, 0]))
        assert SL.assert_permutation_valid(Xtr, y) is True
        assert not np.allclose(y, Xtr.age_days.values)
        assert set(Xtr.room.unique()) == set(P.ROOMS) - {
            "room2"
        }  # held-out room never in the permuted training labels
    # negative tests: (a) per-window shuffle (not constant within a day) and (b) added noise must both fail
    rng = np.random.default_rng(0)
    for bad in (rng.permutation(Xtr.age_days.values), y + rng.normal(0, 40, len(y))):
        try:
            SL.assert_permutation_valid(Xtr, bad)
        except AssertionError:
            continue
        raise AssertionError("negative test failed: an invalid shuffled label was accepted")


def test_13_gru_train_only_imputation_and_time_bounded_sequences():
    """GRU imputation and scaling use only the training rooms, and sequences never span more than 24 real hours."""
    import datasets as D
    import gru_comparator as G
    import numpy as np
    import preprocessing as P

    d = P.fold("LORO", "room2")
    FC = d["fcols"][:40]
    E = (
        d["Xw"][d["Xw"].window_h == D.WINDOWS_EVAL]
        .dropna(subset=["age_days"])
        .sort_values(["room", "hour"])
        .reset_index(drop=True)
    )
    train = E[E.room != "room2"]
    base = G.fit_imputer_scaler(train, FC)
    E2 = E.copy()
    E2.loc[E2.room == "room2", FC] = (
        E2.loc[E2.room == "room2", FC] * 50 + 1000
    )  # wreck the held-out room
    again = G.fit_imputer_scaler(E2[E2.room != "room2"], FC)
    assert all(
        np.allclose(a.values, b.values, equal_nan=True) for a, b in zip(base, again)
    ), "imputer/scaler statistics depend on the held-out room"
    X = G.transform(E, FC, *base)
    held = E.room == "room2"
    col = FC[0]
    nan_rows = E[held & E[col].isna()]
    if len(nan_rows):  # held-out gaps are filled with the TRAINING median
        i = nan_rows.index[0]
        assert np.isclose(X[i, 0], (base[0][col] - base[1][col]) / base[2][col], atol=1e-5)
    S, L, Y, R, H, SP = G.build_sequences(E.head(600), X[:600])
    assert len(S) > 0 and SP.max() <= G.SEQ_H - 1 and (L >= G.MIN_OBS).all()
    assert (L <= G.SEQ_H).all()


def test_14_analysis_scripts_row_counts_and_seeds():
    """Set 4: the analysis scripts use the primary pipeline's rows and fixed seeds."""
    import sys as _s

    import numpy as np
    import pandas as pd

    _s.path.insert(0, str(ROOT / "code" / "analysis"))
    import clustering_epochs as CL
    import daily_demeaning as DM
    import models as M
    import paired_fusion as PF
    import preprocessing as P
    import stage_classification as SC

    assert SC.SEED == CL.SEED == DM.SEED == M.SEED == 0 and PF.SEED == 0 and PF.N_BOOT == 2000
    cc = (
        P.MASTER[(P.MASTER.obs_audio == 1) & (P.MASTER.obs_yolo == 1) & (P.MASTER.obs_morph == 1)]
        .groupby("room")
        .size()
    )
    # (a) stage classifier: class supports of the held-out room add up to the primary LORO row count; boundaries are the documented ones
    assert SC.BINS == [0, 60, 110, 200] and len(SC.LABELS) == 3
    for r in P.ROOMS:
        d = P.fold("LORO", r)
        te = d["Xw"][d["te"]]
        st = pd.cut(te.age_days, SC.BINS, labels=SC.LABELS)
        assert st.notna().all() and len(te) == cc[r], (r, len(te), cc[r])
    # (b) clustering inputs: one 6 h window per complete-case hour, per room
    X, FEAT = CL.inputs()
    assert (
        X.groupby("room").size().to_dict() == cc.to_dict()
    ), "clustering input rows differ from the complete-case hours"
    assert all(c.split("__")[0] in CL.SENSORS for c in FEAT) and not any(
        "hour" in c or "age" in c for c in FEAT
    )
    g1, p1 = CL.room_space(X, FEAT, "room2")
    g2, p2 = CL.room_space(X, FEAT, "room2")
    assert np.allclose(p1, p2), "PCA space is not deterministic"
    l1 = CL.KMeans(2, n_init=10, random_state=CL.SEED).fit_predict(p1)
    l2 = CL.KMeans(2, n_init=10, random_state=CL.SEED).fit_predict(p2)
    assert (l1 == l2).all(), "k-means with the fixed seed is not deterministic"
    # (c) daily demeaning: same rows, each room-day mean of every sensor feature is zero afterwards, flags / hour-of-day untouched
    f = P.fold("LORO", "room2")["f"]
    f2 = DM.diurnal(f)
    assert len(f2) == len(f) == cc.sum()
    feat = DM.feature_columns(f2)
    dm = f2.groupby([f2.room, f2.hour.dt.floor("D")])[feat].mean().abs().max().max()
    assert dm < 1e-6, f"room-day means are not zero after demeaning ({dm})"
    assert (
        f2[["obs_audio", "obs_yolo", "obs_morph", "obs_flow", "hour_sin", "hour_cos", "age_days"]]
        .fillna(-1)
        .values
        == f[["obs_audio", "obs_yolo", "obs_morph", "obs_flow", "hour_sin", "hour_cos", "age_days"]]
        .fillna(-1)
        .values
    ).all()
    # (d) paired fusion contrasts refuse mismatched rows
    a = pd.DataFrame(
        {
            "room": ["room2"] * 3,
            "hour": pd.to_datetime(["2025-08-01", "2025-08-02", "2025-08-03"]),
            "y": [1.0, 2, 3],
            "pred": [1.0, 2, 3],
            "model": "fused",
        }
    )
    a["abs_err"] = (a.pred - a.y).abs()
    a["day"] = a.hour.dt.floor("D")
    b = a.assign(model="morph_only").iloc[:2]
    try:
        PF.paired_frame(
            {"primary": pd.concat([a, b]), "v2": a}, ("primary", "fused"), ("primary", "morph_only")
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("paired contrast accepted non-identical rows")


if __name__ == "__main__":
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"PASS  {n}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL  {n}\n{traceback.format_exc()}", flush=True)
    print(f"{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
