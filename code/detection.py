"""Growth-gap detection validated with injected developmental delays.

A flock k days behind schedule is simulated by feeding the fused leave-one-room-out model
the sensor readings its held-out room produced k days earlier, labelled with today's age.
Two injection modes: morphometry features only, or every sensor feature. The model's
age-dependent bias is removed with a calibration curve estimated from the OTHER rooms'
unperturbed residuals, so no held-out-room information is used. Daily median gaps are
aggregated over 7-day windows.

Windows:
  rolling          one window ending on every day (consecutive windows overlap)
  non-overlapping  consecutive disjoint 7-day blocks; comparisons with fewer than 10
                   windows on either side are reported as not estimable
Operating point: for each held-out room, the 95%-specificity threshold is set ONLY on the
unperturbed (k = 0) windows of the two training rooms (95th percentile of their scores);
the held-out room's own windows never influence it. Sensitivity is the fraction of the
held-out room's injected windows above that threshold; realised specificity is the
fraction of its own unperturbed windows below it (it need not equal the nominal 95%).
Counts are pooled over the three held-out rooms for the all-rooms figures. AUC is a
threshold-free ranking statistic and is computed as before.
Uncertainty: 95% CIs from a block bootstrap over (room, Monday-Sunday week) units,
resampled within each room, 2,000 replicates; the training-room threshold is re-derived
from the resampled training-room windows inside every replicate.

Optional: `--out <dir>` writes every output to <dir> instead of results/.

Outputs (results/ or --out):
  detection_daily_gaps.csv, detection_performance.csv, detection_min_detectable.csv,
  detection_bootstrap.csv, gap_recovery.csv, gap_recovery_blocks.csv, detection_block_counts.csv
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common import RESULTS, ROOMS
import preprocessing as P

OUT_DIR = RESULTS
if "--out" in sys.argv:
    OUT_DIR = Path(sys.argv[sys.argv.index("--out") + 1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

KS = [0, 3, 5, 7, 10, 15, 20]
KPOS = [3, 5, 7, 10, 15, 20]
WIN = 7
MIN_WINDOWS = 10
N_BOOT = 2000
PHASES = ["growth (<=70 d)", "plateau (>70 d)"]


def shifted(room_df, k, which):
    """Copy whose `which` columns come from k days earlier (same room)."""
    g = room_df.sort_values("hour").set_index("hour")
    src = g[which].copy()
    src.index = src.index + pd.Timedelta(days=k)
    out = g.copy()
    out[which] = src.reindex(g.index, method="nearest", tolerance=pd.Timedelta("2h"))
    return out.reset_index()


def rolling_windows(daily):
    out = []
    for (room, mode, k), g in daily.groupby(["room", "mode", "k"]):
        g = g.sort_values("day").copy()
        g["stat"] = g.gap_corrected.rolling(WIN, min_periods=max(1, WIN // 2)).median()
        out.append(g)
    return pd.concat(out, ignore_index=True).dropna(subset=["stat"])


def block_windows(daily):
    out = []
    for (room, mode, k), g in daily.groupby(["room", "mode", "k"]):
        g = g.sort_values("day").reset_index(drop=True)
        g["block"] = np.arange(len(g)) // WIN
        b = (g.groupby("block")
               .agg(stat=("gap_corrected", "median"), n_days=("gap_corrected", "size"),
                    phase=("phase", "last"), n_phases=("phase", "nunique"),
                    first_day=("day", "min"), last_day=("day", "max")).reset_index())
        b = b[b.n_days >= max(1, WIN // 2)]
        b["room"], b["mode"], b["k"] = room, mode, k
        out.append(b)
    return pd.concat(out, ignore_index=True)


def loro_operating_point(neg_by_room, pos_by_room):
    """Leave-one-room-out operating point at nominal 95% specificity.

    For each held-out room r the threshold is the 95th percentile of the scores (-stat) of the
    unperturbed windows of the OTHER rooms only. It is then applied to r's own unperturbed
    windows (false positives / true negatives) and r's own injected windows (true positives /
    false negatives). r's windows never enter its own threshold."""
    per = {}
    for r in ROOMS:
        train_neg = [-neg_by_room[o] for o in ROOMS if o != r and len(neg_by_room[o])]
        train_neg = np.concatenate(train_neg) if train_neg else np.array([])
        rec = dict(tp=0, fn=0, tn=0, fp=0, n_train_neg=len(train_neg))
        if len(train_neg):
            thr = float(np.quantile(train_neg, 0.95))
            own_pos, own_neg = -pos_by_room[r], -neg_by_room[r]
            rec.update(tp=int((own_pos > thr).sum()), fn=int((own_pos <= thr).sum()),
                       tn=int((own_neg <= thr).sum()), fp=int((own_neg > thr).sum()))
        per[r] = rec
    pooled = {k: sum(per[r][k] for r in ROOMS) for k in ("tp", "fn", "tn", "fp")}
    return per, pooled


def rates(rec):
    sens = rec["tp"] / (rec["tp"] + rec["fn"]) if rec["tp"] + rec["fn"] else np.nan
    spec = rec["tn"] / (rec["tn"] + rec["fp"]) if rec["tn"] + rec["fp"] else np.nan
    return sens, spec


def detect(S):
    det = []
    neg_all = S[S.k == 0]
    for mode in ("morph_only", "all_sensor"):
        for phase in PHASES + ["all"]:
            neg = neg_all if phase == "all" else neg_all[neg_all.phase == phase]
            for k in KPOS:
                pos = S[(S.k == k) & (S["mode"] == mode)]
                if phase != "all":
                    pos = pos[pos.phase == phase]
                nb = {r: neg[neg.room == r].stat.values for r in ROOMS}
                pb = {r: pos[pos.room == r].stat.values for r in ROOMS}
                per, pooled = loro_operating_point(nb, pb)
                min_train_neg = min(per[r]["n_train_neg"] for r in ROOMS)
                row = dict(mode=mode, phase=phase, k_days=k, n_pos=len(pos), n_neg=len(neg))
                row["estimable"] = len(pos) >= MIN_WINDOWS and len(neg) >= MIN_WINDOWS
                if row["estimable"]:
                    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
                    row["AUC"] = roc_auc_score(y, np.r_[-neg.stat.values, -pos.stat.values])
                    row["sensitivity_at_95pct_specificity"], row["realised_specificity"] = rates(pooled)
                    for r in ROOMS:
                        row[f"sensitivity_{r}"], row[f"realised_specificity_{r}"] = rates(per[r])
                else:
                    row["AUC"] = row["sensitivity_at_95pct_specificity"] = row["realised_specificity"] = np.nan
                    for r in ROOMS:
                        row[f"sensitivity_{r}"] = row[f"realised_specificity_{r}"] = np.nan
                row["min_training_negatives"] = min_train_neg
                row["fpr_0.05_resolvable"] = bool(min_train_neg >= 20)
                det.append(row)
    return pd.DataFrame(det)


# ---------------------------------------------------------------- injection + calibration
gaps = []
for room in ROOMS:
    d = P.fold("LORO", room)
    res = P.fit_fold("LORO", room)
    FC = res["fcols"]
    MORPH_FC = [c for c in FC if c.startswith("morph__")]
    SENSOR_FC = [c for c in FC if c.split("__")[0] in ("morph", "yolo", "flow", "audio_bio", "audio_equip")]
    Xte, mdl = d["Xw"][d["te"]], res["model"]
    for mode, which in (("morph_only", MORPH_FC), ("all_sensor", SENSOR_FC)):
        for k in KS:
            if k == 0 and mode == "all_sensor":
                continue
            dd = shifted(Xte, k, which).dropna(subset=["age_days"])
            dd = dd.dropna(subset=which, how="all")
            if len(dd) < 50:
                continue
            dd = dd.copy(); dd["pred"] = mdl.predict(dd[FC])
            dd["gap"] = dd.pred - dd.age_days
            dd["k"], dd["mode"], dd["room"] = k, ("both" if k == 0 else mode), room
            gaps.append(dd[["room", "hour", "age_days", "pred", "gap", "k", "mode"]])
    print(f"  injected: {room}", flush=True)

G = pd.concat(gaps, ignore_index=True)
G["day"] = G.hour.dt.floor("D")
G["phase"] = np.where(G.age_days <= 70, "growth (<=70 d)", "plateau (>70 d)")
k0 = G[G.k == 0].copy()
k0["age_bin"] = (k0.age_days // 10) * 10
curves = {r: k0[k0.room != r].groupby("age_bin").gap.median() for r in sorted(G.room.unique())}
out = []
for room, g in G.groupby("room"):
    c = curves[room]
    g = g.copy(); g["gap_corrected"] = g.gap - np.interp(g.age_days, c.index.values + 5, c.values)
    out.append(g)
G = pd.concat(out, ignore_index=True)
daily = G.groupby(["room", "mode", "k", "day", "phase"]).gap_corrected.median().reset_index()
daily.to_csv(OUT_DIR / "detection_daily_gaps.csv", index=False)

# ---------------------------------------------------------------- AUC / sensitivity
ROLL, BLK = rolling_windows(daily), block_windows(daily)
det = pd.concat([detect(ROLL).assign(window="rolling"), detect(BLK).assign(window="non-overlapping")],
                ignore_index=True)
det["status"] = np.where(~det.estimable, "not estimable (<10 windows)",
                         np.where(~det["fpr_0.05_resolvable"], "estimable; <20 negatives, 5% FPR not resolvable",
                                  "estimable"))
det.to_csv(OUT_DIR / "detection_performance.csv", index=False)

md = []
for (wt, mode, phase), g in det.groupby(["window", "mode", "phase"]):
    ok = g[g.estimable & (g.sensitivity_at_95pct_specificity >= 0.80)]
    md.append(dict(window=wt, mode=mode, phase=phase, min_detectable_delay_days=ok.k_days.min() if len(ok) else np.nan,
                   n_estimable_delays=int(g.estimable.sum())))
pd.DataFrame(md).to_csv(OUT_DIR / "detection_min_detectable.csv", index=False)

rec = (BLK.groupby(["mode", "k", "phase"]).stat.agg(["median", "std", "count"]).reset_index())
rec.to_csv(OUT_DIR / "gap_recovery_blocks.csv", index=False)
cnt = []
for ph in PHASES:
    for k in KS:
        s = rec[(rec.phase == ph) & (rec.k == k)]
        n = int(s["count"].iloc[0])
        cnt.append(dict(phase=ph, k_days=k, n_independent_blocks=n, estimable=n >= MIN_WINDOWS))
pd.DataFrame(cnt).to_csv(OUT_DIR / "detection_block_counts.csv", index=False)

# ---------------------------------------------------------------- week-block bootstrap
ROLL["week"] = ROLL.day.dt.to_period("W-SUN").astype(str)
SCOPES = ["all rooms"] + ROOMS


def auc_only(neg, pos):
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    return roc_auc_score(y, np.r_[-neg, -pos])


def boot_cell(neg_df, pos_df, rng):
    """One bootstrap over (room, week) units; every replicate resamples all three rooms and
    re-derives each held-out room's threshold from the resampled training-room windows."""
    nb0 = {r: neg_df[neg_df.room == r].stat.values for r in ROOMS}
    pb0 = {r: pos_df[pos_df.room == r].stat.values for r in ROOMS}
    per0, pooled0 = loro_operating_point(nb0, pb0)
    point = {"all rooms": dict(auc=auc_only(np.concatenate(list(nb0.values())), np.concatenate(list(pb0.values()))),
                               sens_spec=rates(pooled0))}
    for r in ROOMS:
        point[r] = dict(auc=auc_only(nb0[r], pb0[r]) if len(nb0[r]) and len(pb0[r]) else np.nan,
                        sens_spec=rates(per0[r]))
    units = {r: sorted(set(neg_df[neg_df.room == r].week) | set(pos_df[pos_df.room == r].week)) for r in ROOMS}
    neg_by = {(r, w): g.stat.values for (r, w), g in neg_df.groupby(["room", "week"])}
    pos_by = {(r, w): g.stat.values for (r, w), g in pos_df.groupby(["room", "week"])}
    reps = {sc: dict(auc=[], sens=[], spec=[]) for sc in SCOPES}
    degen = {sc: 0 for sc in SCOPES}
    for _ in range(N_BOOT):
        nn = {r: [] for r in ROOMS}; pp = {r: [] for r in ROOMS}
        for r, wk in units.items():
            for w in (rng.choice(wk, size=len(wk), replace=True) if wk else []):
                if (r, w) in neg_by: nn[r].append(neg_by[(r, w)])
                if (r, w) in pos_by: pp[r].append(pos_by[(r, w)])
        nb = {r: np.concatenate(v) if v else np.array([]) for r, v in nn.items()}
        pb = {r: np.concatenate(v) if v else np.array([]) for r, v in pp.items()}
        per, pooled = loro_operating_point(nb, pb)
        an, ap = np.concatenate(list(nb.values())), np.concatenate(list(pb.values()))
        if len(an) and len(ap):
            reps["all rooms"]["auc"].append(auc_only(an, ap))
        else:
            degen["all rooms"] += 1
        s_, sp_ = rates(pooled)
        if not np.isnan(s_): reps["all rooms"]["sens"].append(s_)
        if not np.isnan(sp_): reps["all rooms"]["spec"].append(sp_)
        for r in ROOMS:
            if len(nb[r]) and len(pb[r]):
                reps[r]["auc"].append(auc_only(nb[r], pb[r]))
            else:
                degen[r] += 1
            s_, sp_ = rates(per[r])
            if not np.isnan(s_): reps[r]["sens"].append(s_)
            if not np.isnan(sp_): reps[r]["spec"].append(sp_)
    pct = lambda v, q: float(np.percentile(v, q)) if v else np.nan
    out = []
    for sc in SCOPES:
        rooms = ROOMS if sc == "all rooms" else [sc]
        out.append(dict(scope=sc, AUC=point[sc]["auc"], AUC_ci_lo=pct(reps[sc]["auc"], 2.5), AUC_ci_hi=pct(reps[sc]["auc"], 97.5),
                        sensitivity_95spec=point[sc]["sens_spec"][0],
                        sensitivity_ci_lo=pct(reps[sc]["sens"], 2.5), sensitivity_ci_hi=pct(reps[sc]["sens"], 97.5),
                        realised_specificity=point[sc]["sens_spec"][1],
                        specificity_ci_lo=pct(reps[sc]["spec"], 2.5), specificity_ci_hi=pct(reps[sc]["spec"], 97.5),
                        n_weeks=sum(len(units[r]) for r in rooms),
                        n_weeks_per_room="/".join(str(len(units[r])) for r in rooms),
                        n_windows_neg=sum(len(nb0[r]) for r in rooms), n_windows_pos=sum(len(pb0[r]) for r in rooms),
                        n_degenerate_replicates=degen[sc]))
    return out


rows = []
for mi, mode in enumerate(("all_sensor", "morph_only")):
    for pi, phase in enumerate(PHASES):
        neg_all = ROLL[(ROLL.k == 0) & (ROLL.phase == phase)]
        for k in [5, 10, 15, 20]:
            pos_all = ROLL[(ROLL.k == k) & (ROLL["mode"] == mode) & (ROLL.phase == phase)]
            rng = np.random.default_rng([0, mi, pi, k])
            for rec in boot_cell(neg_all, pos_all, rng):
                rows.append(dict(mode=mode, phase=phase, k_days=k, **rec))
Bt = pd.DataFrame(rows)
Bt["flags"] = Bt.apply(lambda x: "; ".join(f for f, on in (
    ("AUC CI lower bound <= 0.5", x.AUC_ci_lo <= 0.5),
    (f"degenerate replicates: {x.n_degenerate_replicates}/{N_BOOT}", x.n_degenerate_replicates > 0)) if on), axis=1)
Bt.to_csv(OUT_DIR / "detection_bootstrap.csv", index=False)

# ---------------------------------------------------------------- gap recovery (rolling) with CIs
rng = np.random.default_rng(0)
gr = []
for mode in ("both", "morph_only", "all_sensor"):
    for ph in PHASES:
        for k in KS:
            g = ROLL[(ROLL["mode"] == mode) & (ROLL.phase == ph) & (ROLL.k == k)]
            if g.empty:
                continue
            by = {r: {w: q.stat.values for w, q in gg.groupby("week")} for r, gg in g.groupby("room")}
            meds = np.empty(N_BOOT)
            for b in range(N_BOOT):
                vals = []
                for r, wk in by.items():
                    keys = list(wk)
                    for w in rng.choice(keys, size=len(keys), replace=True):
                        vals.append(wk[w])
                meds[b] = np.median(np.concatenate(vals))
            gr.append(dict(mode=mode, phase=ph, k_days=k, median_gap=float(g.stat.median()),
                           ci_lo=float(np.percentile(meds, 2.5)), ci_hi=float(np.percentile(meds, 97.5)),
                           n_weeks=sum(len(v) for v in by.values()),
                           n_weeks_per_room="/".join(str(len(by.get(r, {}))) for r in ROOMS),
                           n_windows=len(g)))
pd.DataFrame(gr).to_csv(OUT_DIR / "gap_recovery.csv", index=False)
print(det[det.window == "non-overlapping"][["mode", "phase", "k_days", "n_pos", "n_neg", "AUC", "status"]]
      .round(3).to_string(index=False))
