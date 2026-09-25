"""All figures, drawn from the CSVs in results/ (no model is fitted here). 600 dpi PNG."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from common import RESULTS, FIGURES, ROOMS

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 600, "savefig.bbox": "tight"})
BANDS = ["<45d", "45-75d", "75-105d", "105-140d", ">140d"]
XT = ["<45", "45–75", "75–105", "105–140", ">140"]
BLOCKS = ["morph", "yolo", "flow", "audio_bio", "audio_equip", "time/flags"]
NAME = {"morph": "Morphometry", "yolo": "Spatial behaviour", "flow": "Optical flow",
        "audio_bio": "Bioacoustics", "audio_equip": "Acoustic level", "time/flags": "Time & data quality"}
COL = {"morph": "#1b4965", "yolo": "#3d7ea6", "flow": "#8bbdd9", "audio_bio": "#d1495b",
       "audio_equip": "#edae49", "time/flags": "#bcbcbc"}
SHORT = {"morph_only": "Morphometry", "yolo_motion_only": "Spatial behaviour", "flow_only": "Optical flow",
         "audio_bio_only": "Bioacoustics", "video_all": "All video", "fused": "Fused"}
ROOM_COL = {"room2": "#1b4965", "room6": "#d1495b", "room7": "#2a9d6f"}


def save(fig, name):
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print("->", name)


def fname(f):
    for p in ("morph__", "yolo__", "flow__", "audio_bio__", "audio_equip__"):
        f = f.replace(p, "")
    return f.replace("__m", " (level)").replace("__s", " (rate of change)").replace("_", " ")


# ---------------------------------------------------------------- attribution
A = pd.read_csv(RESULTS / "attribution_by_age_band.csv")
share = A.pivot(index="age_band", columns="block", values="share").reindex(index=BANDS, columns=BLOCKS)
absv = A.pivot(index="age_band", columns="block", values="mean_abs_shap_days").reindex(index=BANDS, columns=BLOCKS)
xs = np.arange(len(BANDS))

fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
a.stackplot(xs, [share[k].values for k in BLOCKS], colors=[COL[k] for k in BLOCKS],
            labels=[NAME[k] for k in BLOCKS])
a.set_xlim(0, len(BANDS) - 1); a.set_ylim(0, 1); a.set_xticks(xs); a.set_xticklabels(XT)
a.set_xlabel("Flock age (days)"); a.set_ylabel("Share of attribution")
a.legend(ncol=2, fontsize=8, loc="lower center"); a.grid(False)
body = share["morph"].values
b.plot(xs, body, "o-", color=COL["morph"], lw=2, ms=7, label="Morphometry")
b.plot(xs, (share["audio_bio"] + share["audio_equip"]).values, "s-", color=COL["audio_bio"], lw=2, ms=7,
       label="Bioacoustics + acoustic level")
b.plot(xs, share["yolo"].values, "^-", color=COL["yolo"], lw=2, ms=7, label="Spatial behaviour")
for x, v in zip(xs, body):
    b.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
b.set_xticks(xs); b.set_xticklabels(XT)
b.set_xlabel("Flock age (days)"); b.set_ylabel("Share of attribution")
b.legend(fontsize=8.5, loc="upper right")
fig.tight_layout()
save(fig, "attribution_trends.png")

fig, ax = plt.subplots(2, 1, figsize=(9, 7.4), sharex=True, gridspec_kw={"hspace": .15})
bot = np.zeros(len(BANDS))
for k in BLOCKS:
    v = share[k].values
    ax[0].bar(xs, v, bottom=bot, color=COL[k], edgecolor="white", width=.7, label=NAME[k])
    for i, (bo, vv) in enumerate(zip(bot, v)):
        if vv >= .07:
            ax[0].text(i, bo + vv / 2, f"{vv*100:.0f}%", ha="center", va="center", color="white",
                       fontsize=8, fontweight="bold")
    bot += v
    ax[1].plot(xs, absv[k].values, "o-", color=COL[k], lw=2, ms=5.5, label=NAME[k])
ax[0].set_ylim(0, 1); ax[0].set_ylabel("Share of attribution")
ax[0].set_title("(a) Relative share", loc="left", fontsize=10)
ax[0].legend(fontsize=8, bbox_to_anchor=(1.005, 1), loc="upper left")
ax[1].set_ylabel("Mean |SHAP| (days)"); ax[1].set_title("(b) Absolute magnitude", loc="left", fontsize=10)
ax[1].set_xticks(xs); ax[1].set_xticklabels(XT); ax[1].set_xlabel("Flock age (days)")
fig.suptitle("Modality attribution across development", x=.06, ha="left", y=.96, fontsize=11.5)
save(fig, "attribution_by_age_band.png")

top = pd.read_csv(RESULTS / "feature_attribution.csv").head(20)
fig, ax = plt.subplots(figsize=(9, 6.2))
ax.barh([fname(f) for f in top.feature], top.mean_abs_shap_mean_of_rooms, color=[COL[b] for b in top.block])
ax.invert_yaxis(); ax.tick_params(axis="y", labelsize=8.5); ax.set_xlabel("Mean |SHAP| (days)")
used = [k for k in BLOCKS if k in set(top.block)]
ax.legend([plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in used], [NAME[k] for k in used],
          fontsize=8.5, loc="lower right")
ax.set_title("Global feature attribution (top 20)", loc="left", fontsize=11.5)
save(fig, "feature_attribution_top20.png")

# ---------------------------------------------------------------- error by growth phase
T5 = pd.read_csv(RESULTS / "growth_phase_mae.csv")
fig, ax = plt.subplots(figsize=(9.2, 4.8))
w, xx = .37, np.arange(len(T5))
ax.bar(xx - w / 2, T5.growth_MAE, w, color="#1b4965", label="Growth phase (≤70 d)")
ax.bar(xx + w / 2, T5.plateau_MAE, w, color="#8bbdd9", label="Plateau (>70 d)")
for i, (g, p) in enumerate(zip(T5.growth_MAE, T5.plateau_MAE)):
    ax.text(i - w / 2, g + .3, f"{g:.1f}", ha="center", fontsize=8.5)
    ax.text(i + w / 2, p + .3, f"{p:.1f}", ha="center", fontsize=8.5)
ax.set_xticks(xx); ax.set_xticklabels([SHORT[f] for f in T5.feature_set])
ax.set_ylabel("Mean absolute error (days)"); ax.legend(loc="upper left")
ax.set_title("Error by growth phase and feature set", loc="left", fontsize=11.5)
save(fig, "error_by_growth_phase.png")

# ---------------------------------------------------------------- detection
Dp = pd.read_csv(RESULTS / "detection_performance.csv")
Dp = Dp[Dp.window == "non-overlapping"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, (phase, ttl) in zip(axes, (("growth (<=70 d)", "Growth phase (≤70 d)"),
                                   ("plateau (>70 d)", "Plateau (>70 d)"), ("all", "All ages"))):
    for mode, colr, lab in (("all_sensor", "#1b4965", "All sensors"), ("morph_only", "#edae49", "Morphometry only")):
        g = Dp[(Dp.phase == phase) & (Dp["mode"] == mode)].sort_values("k_days")
        e, ne = g[g.estimable], g[~g.estimable]
        ax.plot(e.k_days, e.sensitivity_at_95pct_specificity, "o-", color=colr, lw=2, label=lab)
        ax.scatter(ne.k_days, np.full(len(ne), -.05), marker="x", color=colr, s=40)
    ax.set_title(ttl, fontsize=10); ax.set_xlabel("Injected delay (days)"); ax.set_ylim(-.1, 1.05)
    if not len(Dp[(Dp.phase == phase) & Dp.estimable]):
        ax.text(.5, .5, "Not estimable\n(fewer than 10 windows)", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#777777")
axes[0].set_ylabel("Sensitivity at 95% specificity")
axes[2].legend(fontsize=8.5, loc="upper left")
fig.suptitle("Detection sensitivity by injected delay", x=.06, ha="left", fontsize=11.5)
save(fig, "detection_sensitivity.png")

KS = [0, 3, 5, 7, 10, 15, 20]
PH = ["growth (<=70 d)", "plateau (>70 d)"]
GR = pd.read_csv(RESULTS / "gap_recovery.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, ph, ttl in zip(axes, PH, ("(a) Growth phase (≤70 d)", "(b) Plateau (>70 d)")):
    ax.plot(KS, [-k for k in KS], "k--", lw=1.2, label="Perfect recovery")
    for mode, c, mk, lab in (("morph_only", "#1b4965", "o", "Morphometry shifted"),
                             ("all_sensor", "#d1495b", "s", "All sensors shifted")):
        d = pd.concat([GR[(GR["mode"] == "both") & (GR.phase == ph)],
                       GR[(GR["mode"] == mode) & (GR.phase == ph)]]).sort_values("k_days")
        ax.errorbar(d.k_days, d.median_gap, yerr=[d.median_gap - d.ci_lo, d.ci_hi - d.median_gap],
                    marker=mk, color=c, lw=1.8, ms=6, capsize=3, label=lab)
    ax.set_xlabel("Injected delay (days behind schedule)"); ax.set_title(ttl, fontsize=10)
axes[0].set_ylabel("Detected growth gap (days)")
axes[1].legend(fontsize=8.5, loc="lower left")
fig.tight_layout()
save(fig, "gap_recovery.png")

CNT = pd.read_csv(RESULTS / "detection_block_counts.csv")
REC = pd.read_csv(RESULTS / "gap_recovery_blocks.csv")
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))
w, xk = .38, np.arange(len(KS))
g = CNT[CNT.phase == PH[0]].set_index("k_days").reindex(KS)
p = CNT[CNT.phase == PH[1]].set_index("k_days").reindex(KS)
gb = a.bar(xk - w / 2, g.n_independent_blocks, w, color="#1b4965")
a.bar(xk + w / 2, p.n_independent_blocks, w, color="#8bbdd9")
for bar, est in zip(gb, g.estimable):
    if not est:
        bar.set_facecolor("#ffffff"); bar.set_edgecolor("#1b4965"); bar.set_hatch("///"); bar.set_linewidth(1.2)
for x, n in zip(xk - w / 2, g.n_independent_blocks):
    a.text(x, n + .6, f"{n}", ha="center", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
for x, n in zip(xk + w / 2, p.n_independent_blocks):
    a.text(x, n + .6, f"{n}", ha="center", fontsize=8)
a.axhline(10, ls="--", color="k", lw=1.2)
a.text(len(KS) - .5, 10.6, "minimum for estimation", ha="right", va="bottom", fontsize=8.5)
a.set_xticks(xk); a.set_xticklabels([str(k) for k in KS])
a.set_xlabel("Injected delay (days behind schedule)"); a.set_ylabel("Independent non-overlapping blocks")
a.set_title("(a) Available independent blocks by delay", fontsize=10)
a.legend([plt.Rectangle((0, 0), 1, 1, color="#1b4965"), plt.Rectangle((0, 0), 1, 1, color="#8bbdd9"),
          plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#1b4965", hatch="///")],
         ["Growth phase (≤70 d)", "Plateau (>70 d)", "Below minimum (not estimable)"],
         fontsize=8.2, loc="upper center", ncol=3, handlelength=1.4, columnspacing=1.0)
a.set_ylim(0, 52); a.set_yticks(range(0, 50, 10))
b.plot(KS, [-k for k in KS], "k--", lw=1.2, label="Perfect recovery")
for mode, c, mk, lab in (("morph_only", "#1b4965", "o", "Morphometry shifted"),
                         ("all_sensor", "#d1495b", "s", "All sensors shifted")):
    d = REC[(REC["mode"] == mode) & (REC.phase == PH[1])]
    d = pd.concat([REC[(REC["mode"] == "both") & (REC.phase == PH[1])].assign(mode=mode), d]).sort_values("k")
    d = d[d["count"] >= 10]
    b.errorbar(d.k, d["median"], yerr=d["std"] / np.sqrt(d["count"]), marker=mk, color=c,
               lw=1.8, ms=6, capsize=3, label=lab)
b.set_xlabel("Injected delay (days behind schedule)"); b.set_ylabel("Detected growth gap (days)")
b.set_title("(b) Plateau (>70 d): recovered gap vs injected delay", fontsize=10)
b.legend(fontsize=8.5, loc="lower left")
fig.tight_layout()
save(fig, "gap_recovery_sample_size.png")

# ---------------------------------------------------------------- predicted vs true age
AP = pd.read_csv(RESULTS / "ageblock_predictions.csv")
ABP = pd.read_csv(RESULTS / "ageblock_performance.csv")
ABP = ABP[ABP.block != "mean of 5 blocks"].reset_index(drop=True)
LP = pd.read_csv(RESULTS / "loro_predictions.csv")
LP = LP[LP.feature_set == "fused"]
per = {r: (mean_absolute_error(g.true_age, g.predicted_age), r2_score(g.true_age, g.predicted_age))
       for r, g in LP.groupby("room")}
lo_mae, lo_r2 = np.mean([v[0] for v in per.values()]), np.mean([v[1] for v in per.values()])
BT = {0: "backward extrapolation", 4: "forward extrapolation"}
BCOL = {"backward extrapolation": "#1b4965", "internal gap": "#8bbdd9", "forward extrapolation": "#d1495b"}
lim = [25, 185]
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.6))
for i, r in ABP.iterrows():
    te = AP[AP.block == r.block]
    lo, hi = te.true_age.min(), te.true_age.max()
    if i > 0:
        a.axvline(lo, color="#bbbbbb", lw=.8, zorder=0)
    a.text((lo + hi) / 2, lim[1] - 3, f"B{i + 1}\n{r.MAE:.1f} d", ha="center", va="top", fontsize=8.5,
           color="#333333", linespacing=1.1, bbox=dict(facecolor="white", edgecolor="none", pad=.6, alpha=.9), zorder=4)
    if i in BT:
        trn = AP[AP.block != r.block].true_age
        edge = float(trn.min() if i == 0 else trn.max())
        a.hlines(edge, lo, hi, colors="#555555", linestyles=(0, (3, 2)), lw=1.3,
                 label="Nearest training age" if i == 0 else None, zorder=3)
        a.text(lo + 1, edge + (-2 if i == 0 else 2), "nearest training age", fontsize=7.5, color="#555555",
               va="top" if i == 0 else "bottom", bbox=dict(facecolor="white", edgecolor="none", pad=.5, alpha=.8))
for bt, colr in BCOL.items():
    idx = [i for i in range(len(ABP)) if BT.get(i, "internal gap") == bt]
    d = AP[AP.block.isin(ABP.block.iloc[idx])]
    a.scatter(d.true_age, d.predicted_age, s=6, alpha=.35, color=colr, edgecolors="none", label=bt.capitalize())
a.plot(lim, lim, "k--", lw=1.2, label="Identity")
b.plot(lim, lim, "k--", lw=1.2)
for r, g in LP.groupby("room"):
    b.scatter(g.true_age, g.predicted_age, s=6, alpha=.25, color=ROOM_COL[r], edgecolors="none",
              label=r.replace("room", "Room "))
for ax in (a, b):
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal", adjustable="box"); ax.set_xlabel("True age (days)")
a.set_ylabel("Predicted age (days)")
a.set_title("(a) Leave-one-age-block-out", loc="left", fontsize=10)
b.set_title("(b) Leave-one-room-out", loc="left", fontsize=10)
a.text(.02, .80, f"MAE {ABP.MAE.mean():.2f} d\n(mean of 5 blocks)", transform=a.transAxes, fontsize=8.5, va="top")
b.text(.02, .80, f"MAE {lo_mae:.2f} d\nR² {lo_r2:.3f}\n(mean of 3 rooms)", transform=b.transAxes, fontsize=8.5, va="top")
a.legend(fontsize=7.8, loc="lower right", markerscale=2.5)
b.legend(fontsize=8, loc="lower right", markerscale=2.5)
fig.tight_layout()
save(fig, "predicted_vs_true_age.png")

fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharey=True)
for ax, r in zip(axes, ROOMS):
    d = LP[LP.room == r]
    ax.scatter(d.true_age, d.predicted_age, s=6, alpha=.25, color=ROOM_COL[r], edgecolors="none")
    ax.plot(lim, lim, "k--", lw=1.1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"Held-out {r.replace('room', 'Room ')} — MAE {per[r][0]:.1f} d", fontsize=10)
    ax.set_xlabel("True age (days)")
axes[0].set_ylabel("Predicted age (days)")
fig.tight_layout()
save(fig, "predicted_vs_true_by_room.png")

# ---------------------------------------------------------------- stage classification
CM = pd.read_csv(RESULTS / "stage_classifier_confusion.csv", index_col=0).to_numpy()
cmn = CM / CM.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.4, 4.8))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cmn[i, j]:.2f}\n({CM[i, j]:,})", ha="center", va="center", fontsize=10,
                color="white" if cmn[i, j] > .5 else "black")
lab = ["Brooding/early\n(<60 d)", "Growing\n(60-110 d)", "Mature\n(>110 d)"]
ax.set_xticks(range(3)); ax.set_xticklabels(lab, fontsize=9)
ax.set_yticks(range(3)); ax.set_yticklabels(lab, fontsize=9)
ax.set_xlabel("Predicted stage", fontsize=11); ax.set_ylabel("True stage", fontsize=11); ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.06)
cb.set_label("Row-normalised proportion", fontsize=10.5)
save(fig, "stage_classification.png")

# ---------------------------------------------------------------- negative controls
NC = pd.read_csv(RESULTS / "negative_controls.csv").set_index("test")


def mp(prefix):
    s = NC[NC.index.str.startswith(prefix)]
    return s.MAE.mean(), s.R2.mean()


F = [("Thermal only (real join)", *mp("thermal_diag_thermal_only_real"), "leak"),
     ("Fused + thermal (real join)", *mp("thermal_diag_fused_plus_thermal_real"), "leak"),
     ("Thermal only (values shuffled)", *mp("thermal_diag_thermal_only_shuffled"), "leak"),
     ("Fused, no thermal (reported model)", *mp("thermal_diag_fused_no_thermal"), "ok"),
     ("Fused + thermal (randomly assigned)", *mp("thermal_diag_fused_plus_thermal_random"), "ok"),
     ("Thermal only (randomly assigned)", *mp("thermal_diag_thermal_only_random"), "ctrl"),
     ("Hour-of-day only", *mp("clock_only"), "ctrl"), ("Shuffled age label", *mp("permuted_label"), "ctrl"),
     ("Sensor health only", *mp("sensor_health_only"), "ctrl"),
     ("Train summer -> test autumn", *mp("train_summer_test_fall"), "ctrl"),
     ("Train autumn -> test summer", *mp("train_fall_test_summer"), "ctrl")]
Dn = pd.DataFrame(F, columns=["name", "MAE", "R2", "kind"]).sort_values("MAE")
cmap = {"leak": "#c1121f", "ok": "#1b4965", "ctrl": "#9a9a9a"}
fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
ax[0].barh(Dn.name, Dn.MAE, color=[cmap[k] for k in Dn.kind]); ax[0].invert_yaxis()
for i, v in enumerate(Dn.MAE):
    ax[0].text(v + .7, i, f"{v:.1f}", va="center", fontsize=8.5)
ax[0].set_xlabel("Mean absolute error (days)"); ax[0].set_title("(a) Error", loc="left", fontsize=10)
R2c = Dn.R2.clip(lower=-1)
ax[1].barh(Dn.name, R2c, color=[cmap[k] for k in Dn.kind]); ax[1].invert_yaxis(); ax[1].set_yticklabels([])
for i, (v, vc) in enumerate(zip(Dn.R2, R2c)):
    ax[1].text(vc + (.03 if vc >= 0 else -.03), i, f"{v:.2f}" if v > -1 else f"{v:.1f}", va="center",
               ha="left" if vc >= 0 else "right", fontsize=8.5)
ax[1].axvline(0, color="k", lw=1); ax[1].set_xlim(-1.45, 1.25)
ax[1].set_xlabel("R² (clipped at −1)"); ax[1].set_title("(b) Variance explained", loc="left", fontsize=10)
ax[1].legend([plt.Rectangle((0, 0), 1, 1, color=cmap[k]) for k in ("ok", "leak", "ctrl")],
             ["Reported model", "Calendar leakage", "Negative controls"], fontsize=8.5, loc="lower right")
fig.suptitle("Negative controls", x=.06, ha="left", fontsize=11.5)
save(fig, "negative_controls.png")
