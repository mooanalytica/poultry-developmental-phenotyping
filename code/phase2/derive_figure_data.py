"""Row-level derived files for the figures: coverage, gap-recovery curves and MAE by phase.
Usage: python3 derive_figure_data.py RUN_DIR
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
RUN = Path(sys.argv[1])
OUT = RUN / "fig_data"
OUT.mkdir(parents=True, exist_ok=True)
ROOMS = ["room2", "room6", "room7"]
N_BOOT = 2000
import preprocessing as P
import scale_filter as S
from common import DATA, HATCH

M = P.MASTER.copy()
raw = pd.read_csv(
    DATA / S.MASTER_FILE, usecols=["room", "hour", "morph__bbox_h_p50"], parse_dates=["hour"]
).rename(columns={"morph__bbox_h_p50": "morph_raw"})
M = M.merge(raw, on=["room", "hour"], how="left")
M["day"] = M.hour.dt.floor("D")
M["morph_before_qc"] = M.morph_raw.notna().astype(int)
M["complete_case"] = ((M.obs_audio == 1) & (M.obs_yolo == 1) & (M.obs_morph == 1)).astype(int)
cov = (
    M.groupby(["room", "day"])
    .agg(
        audio=("obs_audio", "sum"),
        tracking=("obs_yolo", "sum"),
        optical_flow=("obs_flow", "sum"),
        morphometry_before_qc=("morph_before_qc", "sum"),
        morphometry_after_qc=("obs_morph", "sum"),
        complete_case=("complete_case", "sum"),
    )
    .reset_index()
)
cov["age_days"] = (cov.day - HATCH).dt.days
cov.to_csv(OUT / "coverage_daily.csv", index=False)
W = pd.read_csv(RUN / "detection" / "set2_window_scores.csv", parse_dates=["day"])
W0 = pd.read_csv(RUN / "detection" / "set2_window_scores_k0.csv", parse_dates=["day"])
W0["week"] = W0.day.dt.to_period("W-SUN").astype(str)
rng = np.random.default_rng(0)
out = []


def curve(df, label_mode, phase, k):
    by = {r: {w: g.stat.values for w, g in q.groupby("week")} for r, q in df.groupby("room")}
    if not by:
        return None
    meds = np.empty(N_BOOT)
    for b in range(N_BOOT):
        vals = []
        for r, wk in by.items():
            keys = list(wk)
            for w in rng.choice(keys, size=len(keys), replace=True):
                vals.append(wk[w])
        meds[b] = np.median(np.concatenate(vals))
    return dict(
        mode=label_mode,
        phase=phase,
        k_days=k,
        median_gap=float(df.stat.median()),
        ci_lo=float(np.percentile(meds, 2.5)),
        ci_hi=float(np.percentile(meds, 97.5)),
        n_windows=len(df),
        n_weeks=sum(len(v) for v in by.values()),
        n_weeks_per_room="/".join(str(len(by.get(r, {}))) for r in ROOMS),
    )


for ph in W.phase.unique():
    r0 = curve(W0[W0.phase == ph], "unperturbed (all eligible anchors)", ph, 0)
    if r0:
        out.append(r0)
    for mode in ("all_sensor", "morph_only"):
        for k in sorted(W.k.unique()):
            q = W[
                (W.rule == "same_hour")
                & (W["mode"] == mode)
                & (W.k == k)
                & (W["class"] == "pos")
                & (W.phase == ph)
            ]
            r = curve(q, mode, ph, int(k))
            if r:
                out.append(r)
pd.DataFrame(out).to_csv(OUT / "gap_recovery_curves.csv", index=False)
rows = []
for name, f in (
    ("primary", "core_primary/preds_loro_primary.csv"),
    ("V2", "core_v2/preds_loro_V2.csv"),
):
    Pp = pd.read_csv(RUN / f, parse_dates=["hour"])
    Pp["e"] = (Pp.pred - Pp.y).abs()
    Pp["day"] = Pp.hour.dt.floor("D")
    for model in Pp.model.unique():
        q = Pp[Pp.model == model]
        for ph, sel in (("growth (<=70 d)", q.y <= 70), ("plateau (>70 d)", q.y > 70)):
            by = {r: q[sel & (q.room == r)].groupby("day").e.agg(["sum", "count"]) for r in ROOMS}
            boot = {r: [] for r in ROOMS + ["macro"]}
            for _ in range(N_BOOT):
                vs = []
                for r in ROOMS:
                    t = by[r].iloc[rng.integers(0, len(by[r]), len(by[r]))]
                    v = t["sum"].sum() / t["count"].sum()
                    boot[r].append(v)
                    vs.append(v)
                boot["macro"].append(float(np.mean(vs)))
            for r in ROOMS + ["macro"]:
                point = (
                    float(np.mean([by[x]["sum"].sum() / by[x]["count"].sum() for x in ROOMS]))
                    if r == "macro"
                    else float(by[r]["sum"].sum() / by[r]["count"].sum())
                )
                rows.append(
                    dict(
                        source=name,
                        model=model,
                        phase=ph,
                        scope=r,
                        MAE=point,
                        ci_lo=float(np.percentile(boot[r], 2.5)),
                        ci_hi=float(np.percentile(boot[r], 97.5)),
                        n=int(
                            sum(by[x]["count"].sum() for x in ROOMS)
                            if r == "macro"
                            else by[r]["count"].sum()
                        ),
                        n_days=int(sum(len(by[x]) for x in ROOMS) if r == "macro" else len(by[r])),
                    )
                )
pd.DataFrame(rows).to_csv(OUT / "mae_by_phase.csv", index=False)
