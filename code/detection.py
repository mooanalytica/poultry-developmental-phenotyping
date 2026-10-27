"""Growth-gap detection by synthetic age substitution, calibrated inside each outer fold.
Usage: python3 code/detection.py [--out DIR] [--boot N]
"""

import os
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import models as M
import numpy as np
import pandas as pd
from common import RESULTS, ROOMS
from scipy.stats import rankdata

WIN = 7
WINDOW_MODE = os.environ.get(
    "WINDOW_MODE", "calendar"
)  # "calendar" (Set 2e) | "row" (Revision 3: row-based rolling windows)
assert WINDOW_MODE in ("calendar", "row")
MIN_PERIODS = max(1, WIN // 2)  # row mode: >= 3 rows
CAL_MIN_DAYS = 4  # calendar mode: a 7-day window / block needs >= 4 days with observations (pre-declared, not tuned)
MIN_BLOCK_DAYS = CAL_MIN_DAYS if WINDOW_MODE == "calendar" else max(1, WIN // 2)
MIN_WINDOWS = 10
TARGET_SPECIFICITY = 0.95  # pre-declared; not tuned
AGE_BIN = 10
PHASES = ["growth (<=70 d)", "plateau (>70 d)"]
KS_POS = [3, 5, 7, 10, 15, 20]
TABLE_KS = [5, 10, 15, 20]
RULES = {"same_hour": 0.0, "nearest_pm1h": 1.0}  # matching tolerance in hours
MODES = ("all_sensor", "morph_only")
SCOPES = ["all rooms"] + ROOMS


def phase_of(age):
    return np.where(np.asarray(age) <= 70, PHASES[0], PHASES[1])


def match_sources(hours, k_days, tol_h):
    """hours: sorted unique datetime64 anchors of one room."""
    hours = np.asarray(hours, dtype="datetime64[ns]")
    target = hours - np.timedelta64(int(k_days * 86400), "s")
    i = np.searchsorted(hours, target)
    cand = np.stack([np.clip(i - 1, 0, len(hours) - 1), np.clip(i, 0, len(hours) - 1)], 1)
    dt = np.abs((hours[cand] - target[:, None]).astype("timedelta64[s]").astype(float)) / 3600.0
    best = np.where(dt[:, 0] <= dt[:, 1], 0, 1)  # tie -> earlier
    ar = np.arange(len(hours))
    return np.where(dt[ar, best] <= tol_h + 1e-9, cand[ar, best], -1)


def _daily(gap_corrected, hour, age):
    df = pd.DataFrame(
        {"day": pd.to_datetime(hour).floor("D"), "g": gap_corrected, "phase": phase_of(age)}
    )
    return df.groupby(["day", "phase"]).g.median().reset_index().sort_values("day")


def window_scores(gap_corrected, hour, age):
    """Window statistic of the daily median gaps."""
    if WINDOW_MODE == "row":
        d = _daily(gap_corrected, hour, age)
        d["stat"] = d.g.rolling(WIN, min_periods=MIN_PERIODS).median()
        return d.dropna(subset=["stat"])[["day", "stat", "phase"]]
    df = pd.DataFrame({"day": pd.to_datetime(hour).floor("D"), "g": gap_corrected, "age": age})
    d = df.groupby("day").agg(g=("g", "median"), age=("age", "mean")).sort_index()
    roll = d.g.rolling(f"{WIN}D")
    d["stat"] = roll.median()
    d["n_days"] = roll.count()
    d = d[d.n_days >= CAL_MIN_DAYS].reset_index()
    d["phase"] = phase_of(d.age)
    return d[["day", "stat", "phase"]]


def block_scores(gap_corrected, hour, age):
    """Non-overlapping 7-day calendar blocks (aligned to the room's first day), median of the daily medians, kept if >= 3 days of data and entirely inside one regime."""
    d = _daily(gap_corrected, hour, age)
    d["block"] = ((d.day - d.day.min()).dt.days // WIN).astype(int)
    b = (
        d.groupby("block")
        .agg(
            stat=("g", "median"),
            n_days=("g", "size"),
            n_phases=("phase", "nunique"),
            phase=("phase", "last"),
            first_day=("day", "min"),
        )
        .reset_index()
    )
    return b[(b.n_days >= MIN_BLOCK_DAYS) & (b.n_phases == 1)][
        ["first_day", "stat", "phase"]
    ].rename(columns={"first_day": "day"})


def auc_fast(neg, pos):
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    r = rankdata(np.r_[neg, pos])
    n1, n0 = len(pos), len(neg)
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


class Calibration:
    """Everything fitted on the two training rooms of one outer fold, plus the provenance needed to prove it."""

    def __init__(self, heldout):
        self.heldout = heldout
        self.curve = None  # Series: age bin -> median unperturbed gap
        self.corr = None  # function (age, gap) -> bias-corrected gap
        self.inner_models_trained_on = []  # [(train_room, scored_room)]
        self.rooms_in_inputs = set()  # every room whose data entered any calibration quantity
        self.inner = {}  # (scored_room, mode, rule, k) -> matched unperturbed inner anchors
        self.thr = {}  # (mode, rule, k, phase, kind) -> threshold ; (..., "n") -> count
        self.inner_windows = {}  # (mode, rule, k, kind) -> inner negative window scores (DataFrame)


def assert_calibration_clean(cal, heldout=None):
    """No held-out data may enter calibration: not as training data of an inner model, not as a scored room, not in any frame."""
    heldout = heldout or cal.heldout
    bad = [x for x in cal.inner_models_trained_on if heldout in x]
    assert not bad, f"held-out room {heldout} used by inner models: {bad}"
    assert (
        heldout not in cal.rooms_in_inputs
    ), f"held-out room {heldout} present in calibration inputs {sorted(cal.rooms_in_inputs)}"
    assert set(cal.rooms_in_inputs) == set(ROOMS) - {
        heldout
    }, f"calibration inputs {sorted(cal.rooms_in_inputs)} are not exactly the two training rooms"
    for key, fr in cal.inner.items():
        assert key[0] != heldout, f"inner frame {key} belongs to the held-out room"
        assert (fr.room != heldout).all(), f"inner frame {key} contains held-out rows"
    return True


def eligible_rows(Xw, room, mask=None):
    """The room's evaluation windows (W = 6), sorted by hour, optionally restricted by a boolean mask aligned with Xw."""
    x = Xw[(Xw.room == room) & (Xw.window_h == 6)]
    if mask is not None:
        x = x[mask.reindex(x.index).fillna(False).values]
    return x.dropna(subset=["age_days"]).sort_values("hour").reset_index(drop=True)


def matched_unperturbed(rows, pred0, which, k, tol_h):
    """Anchors of one room that have a source observation k days earlier (rule tol_h) with usable source features, and their unperturbed gap."""
    src = match_sources(rows.hour.values, k, tol_h)
    has = src >= 0
    allnan = np.zeros(len(rows), bool)
    if has.any():
        allnan[has] = rows.iloc[src[has]][which].isna().all(axis=1).values
    used = has & ~allnan
    fr = rows.loc[used, ["room", "hour", "age_days"]].copy()
    fr["gap0"] = pred0[used] - rows.loc[used, "age_days"].values
    return fr, src, used


def calibrate(heldout, d, FC, which_by_mode, model_factory=None, _leak_for_test=False):
    """Fit bias correction and thresholds from the two training rooms of outer fold `heldout`. d = P.fold("LORO", heldout)."""
    model_factory = model_factory or M.hgb
    cal = Calibration(heldout)
    Xw, tr = d["Xw"], d["tr"]
    train_rooms = [r for r in ROOMS if r != heldout]
    rows_by = {}
    for r in train_rooms:
        cal.rooms_in_inputs.add(r)
        rows_by[r] = eligible_rows(Xw, r, tr)
    if _leak_for_test:  # negative test: deliberately let the held-out room into calibration
        cal.rooms_in_inputs.add(heldout)
        rows_by[heldout] = eligible_rows(Xw, heldout)
    pred0 = {}
    for a, b in ((train_rooms[0], train_rooms[1]), (train_rooms[1], train_rooms[0])):
        cal.inner_models_trained_on.append((a, b))
        Xa = Xw[(Xw.room == a) & tr]
        pred0[b] = model_factory().fit(Xa[FC], Xa.age_days).predict(rows_by[b][FC])
    allgap = pd.concat(
        [
            rows_by[b][["room", "age_days"]].assign(gap0=pred0[b] - rows_by[b].age_days.values)
            for b in train_rooms
        ]
    )
    allgap["bin"] = (allgap.age_days // AGE_BIN) * AGE_BIN
    cal.curve = allgap.groupby("bin").gap0.median()
    cal.corr = lambda age, gap: gap - np.interp(
        age, cal.curve.index.values + AGE_BIN / 2, cal.curve.values
    )
    for mode, which in which_by_mode.items():
        for rule, tol in RULES.items():
            for k in KS_POS:
                frames = {
                    b: matched_unperturbed(rows_by[b], pred0[b], which, k, tol)[0]
                    for b in train_rooms
                }
                for b in train_rooms:
                    cal.inner[(b, mode, rule, k)] = frames[b]
                for kind, fn in (("rolling", window_scores), ("block", block_scores)):
                    neg = []
                    for b, fr in frames.items():
                        if len(fr):
                            neg.append(
                                fn(
                                    cal.corr(fr.age_days.values, fr.gap0.values),
                                    fr.hour.values,
                                    fr.age_days.values,
                                ).assign(room=b)
                            )
                    neg = (
                        pd.concat(neg)
                        if neg
                        else pd.DataFrame(columns=["day", "stat", "phase", "room"])
                    )
                    cal.inner_windows[(mode, rule, k, kind)] = neg
                    for ph in PHASES:
                        s = -neg[neg.phase == ph].stat.values
                        cal.thr[(mode, rule, k, ph, kind)] = (
                            float(np.quantile(s, TARGET_SPECIFICITY)) if len(s) else np.nan
                        )
                        cal.thr[(mode, rule, k, ph, kind, "n")] = int(len(s))
    assert_calibration_clean(cal)
    return cal


def evaluate_fold(heldout, d, model, FC, which_by_mode, cal):
    """Apply the frozen calibration to the held-out room. Returns (rolling windows, blocks, anchor manifest)."""
    assert_calibration_clean(cal, heldout)
    rows = eligible_rows(d["Xw"], heldout, d["te"])
    assert (rows.room == heldout).all()
    pred0 = model.predict(rows[FC])
    wins, blks, man = [], [], []
    for mode, which in which_by_mode.items():
        for rule, tol in RULES.items():
            for k in KS_POS:
                fr, src, used = matched_unperturbed(rows, pred0, which, k, tol)
                shifted = rows.loc[used, FC].copy()
                shifted[which] = rows.iloc[src[used]][which].values
                age = rows.loc[used, "age_days"].values
                hr = rows.loc[used, "hour"].values
                neg_g = cal.corr(age, fr.gap0.values)
                pos_g = cal.corr(age, model.predict(shifted) - age)
                for fn, store in ((window_scores, wins), (block_scores, blks)):
                    for cls, g in (("neg", neg_g), ("pos", pos_g)):
                        w = fn(g, hr, age)
                        w["room"], w["mode"], w["rule"], w["k"], w["class"] = (
                            heldout,
                            mode,
                            rule,
                            k,
                            cls,
                        )
                        store.append(w)
                for ph in PHASES:
                    sel = phase_of(rows.age_days.values) == ph
                    n_used = int((sel & used).sum())
                    n_nosrc = int((sel & (src < 0)).sum())
                    man.append(
                        dict(
                            room=heldout,
                            mode=mode,
                            rule=rule,
                            k_days=k,
                            phase=ph,
                            n_eligible_anchors=int(sel.sum()),
                            n_used_anchors=n_used,
                            n_dropped_no_source=n_nosrc,
                            n_dropped_source_features_all_nan=int(sel.sum()) - n_used - n_nosrc,
                        )
                    )
    # unperturbed windows over ALL eligible anchors (the "0 d" point of the gap-recovery figure); written as a derived file only
    age_all = rows.age_days.values
    g0 = cal.corr(age_all, pred0 - age_all)
    w0 = window_scores(g0, rows.hour.values, age_all)
    w0["room"], w0["mode"], w0["rule"], w0["k"], w0["class"] = heldout, "both", "none", 0, "neg"
    evaluate_fold.k0 = getattr(evaluate_fold, "k0", []) + [w0]
    return pd.concat(wins, ignore_index=True), pd.concat(blks, ignore_index=True), pd.DataFrame(man)


def week_of(day):
    return pd.to_datetime(day).dt.to_period("W-SUN").astype(str)


def cell_stats(per_room, thr, scope):
    """per_room: {room: (neg scores, pos scores)}; thr: {room: threshold}. Returns AUC, sensitivity, achieved specificity."""
    rooms = ROOMS if scope == "all rooms" else [scope]
    neg = np.concatenate([per_room[r][0] for r in rooms])
    pos = np.concatenate([per_room[r][1] for r in rooms])
    tp = sum(int((per_room[r][1] > thr[r]).sum()) for r in rooms)
    tn = sum(int((per_room[r][0] <= thr[r]).sum()) for r in rooms)
    return (
        auc_fast(neg, pos),
        (tp / len(pos) if len(pos) else np.nan),
        (tn / len(neg) if len(neg) else np.nan),
    )


def bootstrap_cell(tests, inner, thr_point, rng, n_boot):
    """tests: {room: {week: (neg scores, pos scores)}}; inner: {heldout room: {inner room: {week: neg scores}}}."""
    out = {
        sc: {
            v: {"AUC": [], "sensitivity": [], "achieved_specificity": []}
            for v in ("frozen", "incl_cal")
        }
        for sc in SCOPES
    }
    weeks = {r: sorted(tests[r]) for r in ROOMS}
    for _ in range(n_boot):
        pr = {}
        for r in ROOMS:
            wk = rng.choice(weeks[r], size=len(weeks[r]), replace=True) if weeks[r] else []
            n = [tests[r][w][0] for w in wk]
            p = [tests[r][w][1] for w in wk]
            pr[r] = (
                np.concatenate(n) if n else np.array([]),
                np.concatenate(p) if p else np.array([]),
            )
        thr_b = {}
        for r in ROOMS:
            vals = []
            for wd in inner[r].values():
                ks = list(wd)
                if ks:
                    vals += [wd[w] for w in rng.choice(ks, size=len(ks), replace=True)]
            vals = np.concatenate(vals) if vals else np.array([])
            thr_b[r] = float(np.quantile(vals, TARGET_SPECIFICITY)) if len(vals) else thr_point[r]
        for variant, thr in (("frozen", thr_point), ("incl_cal", thr_b)):
            for sc in SCOPES:
                for key, v in zip(
                    ("AUC", "sensitivity", "achieved_specificity"), cell_stats(pr, thr, sc)
                ):
                    if not np.isnan(v):
                        out[sc][variant][key].append(v)
    return out


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else np.nan


def run_calibrated(out_dir, n_boot=2000, cache=None):
    import preprocessing as P

    P.CACHE = Path(cache) if cache else Path(tempfile.mkdtemp())
    np.random.seed(0)
    all_w, all_b, all_man, cals = [], [], [], {}
    for r in ROOMS:
        d = P.fold("LORO", r)
        res = P.fit_fold("LORO", r)
        FC = res["fcols"]
        which = {
            "morph_only": [c for c in FC if c.startswith("morph__")],
            "all_sensor": [
                c
                for c in FC
                if c.split("__")[0] in ("morph", "yolo", "flow", "audio_bio", "audio_equip")
            ],
        }
        cal = calibrate(r, d, FC, which)
        w, b, man = evaluate_fold(r, d, res["model"], FC, which, cal)
        all_w.append(w)
        all_b.append(b)
        all_man.append(man)
        cals[r] = cal
    W = pd.concat(all_w, ignore_index=True)
    B = pd.concat(all_b, ignore_index=True)
    MAN = pd.concat(all_man, ignore_index=True)
    W["week"] = week_of(W.day)
    W.to_csv(out_dir / "set2_window_scores.csv", index=False)
    pd.concat(evaluate_fold.k0, ignore_index=True).to_csv(
        out_dir / "set2_window_scores_k0.csv", index=False
    )
    rows = []
    rng = np.random.default_rng(0)
    for rule in RULES:
        for mode in MODES:
            for ph in PHASES:
                for k in KS_POS:
                    per, tests, inner, thr = {}, {}, {}, {}
                    for r in ROOMS:
                        sel = (
                            (W.room == r)
                            & (W["mode"] == mode)
                            & (W.rule == rule)
                            & (W.k == k)
                            & (W.phase == ph)
                        )
                        n = W[sel & (W["class"] == "neg")]
                        p = W[sel & (W["class"] == "pos")]
                        per[r] = (-n.stat.values, -p.stat.values)
                        thr[r] = cals[r].thr[(mode, rule, k, ph, "rolling")]
                        tests[r] = {
                            wk: (-n[n.week == wk].stat.values, -p[p.week == wk].stat.values)
                            for wk in sorted(set(n.week) | set(p.week))
                        }
                        iw = cals[r].inner_windows[(mode, rule, k, "rolling")]
                        iw = iw[iw.phase == ph].assign(week=lambda x: week_of(x.day))
                        inner[r] = {
                            ir: {wk: -g.stat.values for wk, g in q.groupby("week")}
                            for ir, q in iw.groupby("room")
                        }
                    nb = n_boot if (rule == "same_hour" and k in TABLE_KS) else min(n_boot, 500)
                    bt = bootstrap_cell(tests, inner, thr, rng, nb)
                    for sc in SCOPES:
                        rr = ROOMS if sc == "all rooms" else [sc]
                        a, s, sp = cell_stats(per, thr, sc)
                        npos = sum(len(per[r][1]) for r in rr)
                        nneg = sum(len(per[r][0]) for r in rr)
                        row = dict(
                            rule=rule,
                            mode=mode,
                            phase=ph,
                            k_days=k,
                            test_room=sc,
                            AUC=a,
                            sensitivity=s,
                            achieved_specificity=sp,
                            target_specificity=TARGET_SPECIFICITY,
                            n_pos_windows=npos,
                            n_neg_windows=nneg,
                            n_weeks=sum(len(tests[r]) for r in rr),
                            n_boot=nb,
                            threshold="/".join(f"{thr[r]:.3f}" for r in rr),
                            n_inner_neg_windows="/".join(
                                str(cals[r].thr[(mode, rule, k, ph, "rolling", "n")]) for r in rr
                            ),
                            min_required=MIN_WINDOWS,
                            estimable=bool(min(npos, nneg) >= MIN_WINDOWS),
                        )
                        for variant in ("frozen", "incl_cal"):
                            for name, val in (
                                ("AUC", a),
                                ("sensitivity", s),
                                ("achieved_specificity", sp),
                            ):
                                lo, hi = pct(bt[sc][variant][name], 2.5), pct(
                                    bt[sc][variant][name], 97.5
                                )
                                row[f"{name}_ci_lo_{variant}"], row[f"{name}_ci_hi_{variant}"] = (
                                    lo,
                                    hi,
                                )
                                row[f"{name}_ci_contains_point_{variant}"] = bool(
                                    np.isnan(val) or (lo - 1e-12 <= val <= hi + 1e-12)
                                )
                        rows.append(row)
    DET = pd.DataFrame(rows)
    DET["interval_scope_note"] = (
        "95% intervals are conditional on this cohort (3 rooms of one flock, one cycle) and on these rooms; they do not describe "
        "new farms. 'frozen' = test weeks resampled with the calibration fixed; 'incl_cal' = threshold also re-derived from resampled inner weeks."
    )
    DET.to_csv(out_dir / "set2_detection_by_room.csv", index=False)
    wc = (
        W.groupby(["room", "mode", "rule", "k", "phase", "class"])
        .size()
        .unstack("class")
        .reset_index()
        .rename(columns={"k": "k_days", "neg": "n_neg_windows", "pos": "n_pos_windows"})
    )
    MAN.merge(wc, on=["room", "mode", "rule", "k_days", "phase"], how="left").to_csv(
        out_dir / "set2_perturbation_manifest.csv", index=False
    )
    sb = []
    for rule in RULES:
        for mode in MODES:
            for ph in PHASES:
                for k in KS_POS:
                    per, thr = {}, {}
                    for r in ROOMS:
                        sel = (
                            (B.room == r)
                            & (B["mode"] == mode)
                            & (B.rule == rule)
                            & (B.k == k)
                            & (B.phase == ph)
                        )
                        n = B[sel & (B["class"] == "neg")]
                        p = B[sel & (B["class"] == "pos")]
                        per[r] = (-n.stat.values, -p.stat.values)
                        thr[r] = cals[r].thr[(mode, rule, k, ph, "block")]
                    for sc in SCOPES:
                        rr = ROOMS if sc == "all rooms" else [sc]
                        a, s, sp = cell_stats(per, thr, sc)
                        npos = sum(len(per[r][1]) for r in rr)
                        nneg = sum(len(per[r][0]) for r in rr)
                        sb.append(
                            dict(
                                rule=rule,
                                mode=mode,
                                phase=ph,
                                k_days=k,
                                test_room=sc,
                                n_blocks_pos=npos,
                                n_blocks_neg=nneg,
                                n_blocks_per_room="/".join(str(len(per[r][1])) for r in rr),
                                AUC=a,
                                sensitivity=s,
                                achieved_specificity=sp,
                                min_required=MIN_WINDOWS,
                                estimable=bool(min(npos, nneg) >= MIN_WINDOWS),
                                threshold="/".join(f"{thr[r]:.3f}" for r in rr),
                                n_inner_neg_blocks="/".join(
                                    str(cals[r].thr[(mode, rule, k, ph, "block", "n")]) for r in rr
                                ),
                            )
                        )
    pd.DataFrame(sb).to_csv(out_dir / "set2_sparse_blocks.csv", index=False)
    cols = [
        "mode",
        "phase",
        "k_days",
        "AUC",
        "AUC_ci_lo_frozen",
        "AUC_ci_hi_frozen",
        "sensitivity",
        "sensitivity_ci_lo_frozen",
        "sensitivity_ci_hi_frozen",
        "achieved_specificity",
        "achieved_specificity_ci_lo_frozen",
        "achieved_specificity_ci_hi_frozen",
        "AUC_ci_lo_incl_cal",
        "AUC_ci_hi_incl_cal",
        "sensitivity_ci_lo_incl_cal",
        "sensitivity_ci_hi_incl_cal",
        "n_pos_windows",
        "n_neg_windows",
        "n_weeks",
        "estimable",
        "interval_scope_note",
    ]
    DET[(DET.rule == "same_hour") & (DET.test_room == "all rooms") & DET.k_days.isin(TABLE_KS)][
        cols
    ].to_csv(out_dir / "set2_table8.csv", index=False)
    (
        W.groupby(["rule", "mode", "phase", "k", "class"])
        .stat.agg(["median", "size"])
        .reset_index()
        .rename(columns={"median": "median_corrected_gap", "size": "n_windows"})
        .to_csv(out_dir / "set2_gap_recovery_matched.csv", index=False)
    )
    flags = DET[[c for c in DET.columns if "contains_point" in c]].apply(lambda s: int((~s).sum()))
    print("intervals NOT containing their point estimate:", flags.to_dict())
    return DET


if __name__ == "__main__":
    out = RESULTS
    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
    run_calibrated(
        out, int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else 2000
    )
