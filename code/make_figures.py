"""Build every figure from the results sheet and the registered derived files, in the shared style of code/plot_style.py.
Usage: python3 code/make_figures.py [--out DIR] [--sheet CSV]
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import plot_style as P
from plot_style import (
    BCOL,
    BLOCKS,
    COL,
    NAME,
    NEG_CMAP,
    POOLED_COL,
    ROOM_COL,
    ROOM_LABEL,
    ROOMS,
    fs,
    lw,
    ms,
    sa,
)

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "figures"))
ap.add_argument("--sheet", default=str(ROOT / "results/results_sheet.csv"))
ap.add_argument("--register", default=str(ROOT / "results/derived_files_register.csv"))
ap.add_argument("--only", default="")
A = ap.parse_args()
OUT = Path(A.out)
OUT.mkdir(parents=True, exist_ok=True)
CHECKS = ROOT / "results" / "figure_checks"
CHECKS.mkdir(parents=True, exist_ok=True)
SHEET = Path(A.sheet)
W1MM, W2MM = 90, 190
S = pd.read_csv(SHEET)
S["variant"] = S.variant.fillna("")
S["detail"] = S.detail.fillna("")
S["text_value"] = S.text_value.fillna("")
REG = pd.read_csv(A.register)


def sha(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def derived(rel_suffix, **kw):
    r = REG[REG.relative_path_in_run_dir.str.endswith(rel_suffix)].iloc[0]
    p = ROOT / "results" / r.relative_path_in_run_dir
    assert sha(p) == r.sha256, f"derived file {rel_suffix} does not match its registered hash"
    return pd.read_csv(p, **kw)


REGISTRY = []
CUR = [None]


def q(section, metric=None, variant=None, room=None, detail=None):
    d = S[S.section == section]
    for col, v in (
        ("metric", metric),
        ("variant", variant),
        ("room_or_fold", room),
        ("detail", detail),
    ):
        if v is not None:
            d = d[d[col] == v]
    assert len(d) >= 1, (section, metric, variant, room, detail)
    return d


def one(section, metric=None, variant=None, room=None, detail=None):
    d = q(section, metric, variant, room, detail)
    assert len(d) == 1, (section, metric, variant, room, detail, len(d))
    return d.iloc[0]


def numv(row, dec=2, field="value", thousands=False, sign=False):
    v = float(row[field])
    txt = f"{v:+,.{dec}f}" if sign else (f"{v:,.{dec}f}" if thousands else f"{v:.{dec}f}")
    REGISTRY.append(
        dict(
            figure=CUR[0],
            printed=txt,
            sheet_value=v,
            scale=1,
            decimals=dec,
            result_id=row.result_id,
            field=field,
        )
    )
    return txt


def num(section, metric, variant, room, dec=2, detail=None, field="value", thousands=False):
    return numv(one(section, metric, variant, room, detail), dec, field, thousands)


def reg_other(printed, value, dec, rid, field):
    REGISTRY.append(
        dict(
            figure=CUR[0],
            printed=printed,
            sheet_value=float(value),
            scale=1,
            decimals=dec,
            result_id=rid,
            field=field,
        )
    )


SIZES = []
CAPTIONS = {}


def plot_valid(Pd):
    """Drop windows without usable features from a scatter; printed numbers still come from the sheet."""
    flags = derived("valid_windows/valid_windows.csv", parse_dates=["hour"])
    m = Pd.merge(flags, on=["room", "hour"], how="left")
    assert m.valid.notna().all()
    return m[m.valid.astype(bool)].drop(columns="valid")


def finish(fig, name):
    meta_pdf = {"CreationDate": None, "Creator": None, "Producer": None, "Title": None}
    fig.savefig(OUT / f"{name}.svg", format="svg", metadata={"Date": None, "Creator": None})
    fig.savefig(OUT / f"{name}.pdf", format="pdf", metadata=meta_pdf)
    fig.savefig(
        OUT / f"{name}.png",
        format="png",
        dpi=600,
        pil_kwargs={"dpi": (600, 600), "optimize": False},
    )
    txt = [
        t.get_fontsize()
        for t in fig.findobj(matplotlib.text.Text)
        if t.get_visible() and t.get_text().strip()
    ]
    w, h = fig.get_size_inches()
    SIZES.append(
        dict(
            figure=name,
            width_mm=round(w * 25.4, 1),
            height_mm=round(h * 25.4, 1),
            column=(
                "single (90 mm)"
                if abs(w * 25.4 - 90) < 1
                else ("double (190 mm)" if abs(w * 25.4 - 190) < 1 else "custom")
            ),
            min_font_pt=min(txt),
            max_font_pt=max(txt),
            n_text_items=len(txt),
            png_pixels=f"{int(round(w * 600))} x {int(round(h * 600))}",
        )
    )
    plt.close(fig)


BANDS = ["<45d", "45-75d", "75-105d", "105-140d", ">140d"]
XT = ["<45", "45–75", "75–105", "105–140", ">140"]
LIM = [25, 185]


def fname(f):
    for p in ("morph__", "yolo__", "flow__", "audio_bio__", "audio_equip__"):
        f = f.replace(p, "")
    return f.replace("__m", " (level)").replace("__s", " (rate of change)").replace("_", " ")


def fig01():
    CUR[0] = "fig01"
    cov = derived("fig_data/coverage_daily.csv", parse_dates=["day"])
    fig, ax, g = P.new_figure(11, 4.8, W2MM)
    mods = [
        ("morphometry_after_qc", "Morphometry", COL["morph"]),
        ("tracking", "Spatial behaviour", COL["yolo"]),
        ("optical_flow", "Optical flow", COL["flow"]),
        ("audio", "Audio", COL["audio_bio"]),
    ]
    y = 0
    ticks = []
    labs = []
    for r in ROOMS:
        c = cov[cov.room == r].copy()
        c["age"] = (c.day - pd.Timestamp("2025-06-05")).dt.days
        for col, lab, colc in mods:
            present = np.sort(c.loc[c[col] > 0, "age"].unique())
            if len(present):
                ax.bar(
                    present,
                    height=0.8,
                    width=0.95,
                    bottom=y - 0.4,
                    color=colc,
                    align="center",
                    linewidth=0,
                )
            ticks.append(y)
            labs.append(f"{ROOM_LABEL[r]} — {lab}")
            y -= 1
        y -= 0.6
    ax.set_xlim(0, 185)
    ax.set_ylim(y + 0.4, 1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labs, fontsize=fs(8.5, g))
    ax.set_xlabel("Flock age (days)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    tot = {
        r: one(
            "S01_data_coverage", "hours from placement to the room's last observed hour", None, r
        ).value
        for r in ROOMS
    }
    CAPTIONS["fig01_coverage_timeline"] = (
        "Daily data coverage of the three rooms across the study. One tile marks a calendar day (flock age, days since placement on 5 June 2025) on which the modality contributed at least one observed hour; "
        "blank space is a day without any observation. Rows per room: morphometry (after the scale-artefact quality-control filter), spatial behaviour (tracking), optical flow and audio. Daily tiles preserve gaps. "
        f"Hours from placement to each room's last observed hour (sheet S01): room 2 {tot['room2']:,.0f}, room 6 {tot['room6']:,.0f}, room 7 {tot['room7']:,.0f}. The three rooms are replicates within one flock and one production cycle. No inferential interval is shown."
    )
    for r in ROOMS:
        reg_other(
            f"{tot[r]:,.0f}",
            tot[r],
            0,
            one(
                "S01_data_coverage",
                "hours from placement to the room's last observed hour",
                None,
                r,
            ).result_id,
            "value",
        )
    finish(fig, "fig01_coverage_timeline")


def fig04():
    CUR[0] = "fig04"
    Pd = derived("core_primary/preds_loro_primary.csv", parse_dates=["hour"])
    Pd = Pd[Pd.model == "fused"]
    Pd = plot_valid(Pd)
    fig, axes, g = P.new_figure(12, 4.3, W2MM, 1, 3, sharey=True)
    for ax, r in zip(axes, ROOMS):
        d = Pd[Pd.room == r]
        ax.scatter(
            d.y,
            d.pred,
            s=sa(6, g),
            alpha=0.25,
            color=ROOM_COL[r],
            edgecolors="none",
            rasterized=True,
        )
        ax.plot(LIM, LIM, "k--", lw=lw(1.1, g))
        ax.set_xlim(LIM)
        ax.set_ylim(LIM)
        m = num("S03_loro", "MAE", "primary fused", r, 1)
        ax.set_title(f"Held-out {ROOM_LABEL[r]} — MAE {m} d", fontsize=fs(10, g))
        ax.set_xlabel("True age (days)")
    axes[0].set_ylabel("Predicted age (days)")
    fig.tight_layout()
    CAPTIONS["fig04_loro_predicted_vs_true"] = (
        "Predicted against true flock age for the primary fused model under leave-one-room-out validation. Each point is one 6 h evaluation window of the held-out room (model trained on the other two rooms); the dashed line is perfect prediction. "
        "Panel titles give the held-out room's mean absolute error (MAE, days). The three rooms are replicates within one flock and one cycle; no interval is drawn on the scatter (intervals for these errors are in Fig 8 and Fig S1 and are conditional on this cohort and these rooms). Windows with no usable window features (at most half of the window-mean features present, about 17% of windows; they all receive one constant prediction) are not drawn, but they are included in every MAE, R² and n reported."
    )
    finish(fig, "fig04_loro_predicted_vs_true")


TIER2 = (
    "Gompertz growth-curve inversion",
    "Ordinal stage classifier",
    "GRU (24 h sequences, 5-seed mean)",
)


def fig05_models():
    CUR[0] = "fig05"
    d = S[
        (S.section == "S32_model_family")
        & (S.room_or_fold == "macro")
        & S.metric.isin(["MAE", "R2"])
    ]
    rows = d.pivot_table(index="variant", columns="metric", values="value", sort=False)
    ids = {
        v: (
            one("S32_model_family", "MAE", v, "macro").result_id,
            one("S32_model_family", "R2", v, "macro").result_id,
        )
        for v in rows.index
    }
    rows = rows.drop(index="HistGradientBoosting, no window augmentation").sort_values("MAE")
    col = [
        "#1b4965" if "(reported)" in v else ("#d1495b" if v in TIER2 else "#3d7ea6")
        for v in rows.index
    ]
    fig, axes, g = P.new_figure(14.4, 5, 228, 1, 2)
    a, b = axes  # 20% wider than 12 x 5: the extra width holds the legend
    a.barh(rows.index, rows.MAE, color=col)
    a.invert_yaxis()
    for i, (v, nm) in enumerate(zip(rows.MAE, rows.index)):
        a.text(v + 0.7, i, f"{v:.1f}", va="center", fontsize=fs(8, g))
        reg_other(f"{v:.1f}", v, 1, ids[nm][0], "value")
    a.set_xlabel("Mean absolute error (days)")
    a.tick_params(axis="y", labelsize=fs(8.5, g))
    a.set_title("(a) Error", fontsize=fs(9.5, g))
    R2c = rows.R2.clip(lower=-1)
    b.barh(rows.index, R2c, color=col)
    b.invert_yaxis()
    b.axvline(0, color="k", lw=lw(1, g))
    b.set_yticklabels([])
    for i, (v, vc, nm) in enumerate(zip(rows.R2, R2c, rows.index)):
        b.text(
            vc + (0.03 if vc >= 0 else -0.03),
            i,
            f"{v:.2f}" if v > -1 else f"{v:.1f}",
            va="center",
            ha="left" if vc >= 0 else "right",
            fontsize=fs(8, g),
        )
        reg_other(f"{v:.2f}" if v > -1 else f"{v:.1f}", v, 2 if v > -1 else 1, ids[nm][1], "value")
    b.set_xlim(-1.4, 1.15)
    b.set_xlabel("R$^2$ (clipped at −1)")
    b.set_title("(b) Variance explained", fontsize=fs(9.5, g))
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    fig.legend(
        [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#1b4965", "#3d7ea6", "#d1495b"]],
        ["Reported model", "Other regressor families", "Other model classes"],
        fontsize=fs(8.5, g),
        loc="center right",
        bbox_to_anchor=(0.995, 0.5),
    )
    gr = one("S32_model_family", "MAE", "GRU (24 h sequences, 5-seed mean)", "macro")
    gn = numv(gr, 0, "n", True)
    CAPTIONS["fig05_model_family_comparison"] = (
        "Comparison of model families under leave-one-room-out validation on the primary pipeline (same features, folds, seed and 4,198 complete-case 6 h windows for every model). "
        "(a) Mean absolute error and (b) R² (clipped at −1; unclipped value printed), each the mean of the three held-out rooms. Dark blue: the reported HistGradientBoosting model, which uses all window scales and handles missing values natively; mid blue: other regressor families "
        "(all on the 6 h evaluation windows only; median imputation, and standardisation where the model needs it, were fitted on the two training rooms and applied to the held-out room); red: other model classes — a Gompertz growth-curve inversion of relative body height "
        f"(scored only on windows with the size feature, n = {int(one('S32_model_family', 'n', 'Gompertz growth-curve inversion', 'room2').n) + int(one('S32_model_family', 'n', 'Gompertz growth-curve inversion', 'room6').n) + int(one('S32_model_family', 'n', 'Gompertz growth-curve inversion', 'room7').n):,}), "
        f"an ordinal three-stage classifier (predicted age = fixed stage centres) and a GRU over 24 h sequences (mean of 5 seeds, MAE range {gr.ci_lo:.1f}–{gr.ci_hi:.1f} d, scored on its own {gn} anchors rather than the 4,198 windows). "
        "Imputation and scaling of each model, and HGB without window augmentation (18.5 d, not drawn), are listed in the sheet (section S32). Hyperparameters are fixed for every model; none was tuned. Conditional on this cohort and these three rooms (n = 3 folds); no interval is drawn. MAE = mean absolute error."
    )
    finish(fig, "fig05_model_family_comparison")


def fig06():
    CUR[0] = "fig06"
    fig, (a, b), g = P.new_figure(12, 4.6, W2MM, 1, 2)
    xs = np.arange(len(BANDS))
    sh = {
        k: np.array(
            [
                one("S21_attribution", "share of total |SHAP|", f"primary | {k}", band).value
                for band in BANDS
            ]
        )
        for k in BLOCKS
    }
    a.stackplot(
        xs,
        [sh[k] for k in BLOCKS],
        colors=[COL[k] for k in BLOCKS],
        labels=[NAME[k] for k in BLOCKS],
    )
    a.set_xlim(0, len(BANDS) - 1)
    a.set_ylim(0, 1)
    a.set_xticks(xs)
    a.set_xticklabels(XT)
    a.set_xlabel("Flock age (days)")
    a.set_ylabel("Share of attribution")
    a.legend(ncol=2, fontsize=fs(8, g), loc="lower center")
    a.grid(False)
    b.plot(xs, sh["morph"], "o-", color=COL["morph"], lw=lw(2, g), ms=ms(7, g), label="Morphometry")
    b.plot(
        xs,
        sh["audio_bio"] + sh["audio_equip"],
        "s-",
        color=COL["audio_bio"],
        lw=lw(2, g),
        ms=ms(7, g),
        label="Bioacoustics + acoustic level",
    )
    b.plot(
        xs, sh["yolo"], "^-", color=COL["yolo"], lw=lw(2, g), ms=ms(7, g), label="Spatial behaviour"
    )
    for i, band in enumerate(BANDS):
        r = one("S21_attribution", "share of total |SHAP|", "primary | morph", band)
        b.annotate(
            numv(r, 2),
            (i, r.value),
            textcoords="offset points",
            xytext=(0, 8 * g + 2),
            ha="center",
            fontsize=fs(8, g),
        )
    b.set_xticks(xs)
    b.set_xticklabels(XT)
    b.set_xlabel("Flock age (days)")
    b.set_ylabel("Share of attribution")
    b.legend(fontsize=fs(8.5, g), loc="upper right")
    fig.tight_layout()
    CAPTIONS["fig06_attribution_by_modality"] = (
        "Attribution of the primary fused model's predictions to feature blocks across age bands. For every held-out 6 h window, the absolute SHAP values of all columns in a block were summed and divided by the total; shares are the mean over windows within each age band, "
        "then the mean over the three held-out rooms (leave-one-room-out). (a) Stacked shares of all six blocks. (b) The three largest groups; labels give the morphometry share. 'Attribution' denotes the share of total absolute SHAP value, not a causal effect. "
        "Conditional on this cohort and these three rooms (n = 3 folds); the range across rooms is in the sheet (section S21). SHAP = SHapley Additive exPlanations."
    )
    finish(fig, "fig06_attribution_by_modality")


def fig07():
    CUR[0] = "fig07"
    top = derived("attribution/feature_attribution.csv").head(20)
    fig, ax, g = P.new_figure(9, 6.2, W2MM)
    ax.barh(
        [fname(f) for f in top.feature],
        top.mean_abs_shap_mean_of_rooms,
        color=[COL[b] for b in top.block],
    )
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=fs(8.5, g))
    ax.set_xlabel("Mean |SHAP| (days)")
    for i, r in enumerate(top.itertuples()):
        reg_other(
            f"{r.mean_abs_shap_mean_of_rooms:.2f}",
            r.mean_abs_shap_mean_of_rooms,
            2,
            "derived:attribution/feature_attribution.csv",
            "mean_abs_shap_mean_of_rooms",
        )
    used = [k for k in BLOCKS if k in set(top.block)]
    ax.legend(
        [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in used],
        [NAME[k] for k in used],
        fontsize=fs(8.5, g),
        loc="lower right",
    )
    fig.tight_layout()
    CAPTIONS["fig07_feature_attribution"] = (
        "Twenty columns with the largest attribution in the primary fused model under leave-one-room-out validation. Bars give the mean absolute SHAP value (in days of predicted age) over the held-out 6 h windows, averaged over the three rooms. "
        "Colour marks the feature block; '(level)' is the window mean of the feature and '(rate of change)' its slope over the window. Names follow the feature dictionary. 'Attribution' denotes mean absolute SHAP value, not a causal effect. "
        "Conditional on this cohort and these three rooms (n = 3 folds). SHAP = SHapley Additive exPlanations."
    )
    finish(fig, "fig07_feature_attribution")


FSETS = [
    ("Morphometry", "morph_only"),
    ("Spatial behaviour", "yolo_motion_only"),
    ("Optical flow", "flow_only"),
    ("Bioacoustics", "audio_bio_only"),
    ("All video", "video_all"),
    ("Fused", "fused"),
]


def fig08():
    CUR[0] = "fig08"
    fig, ax, g = P.new_figure(9.2, 4.8, W2MM)
    w = 0.37
    xx = np.arange(len(FSETS))
    gr = [one("S31_phase_mae_feature_sets", "MAE growth (<=70 d)", nm, "macro") for nm, _ in FSETS]
    pl = [one("S31_phase_mae_feature_sets", "MAE plateau (>70 d)", nm, "macro") for nm, _ in FSETS]
    ax.bar(xx - w / 2, [r.value for r in gr], w, color=COL["morph"], label="Growth phase (≤70 d)")
    ax.bar(xx + w / 2, [r.value for r in pl], w, color=COL["flow"], label="Plateau (>70 d)")
    for i, (a_, b_) in enumerate(zip(gr, pl)):
        ax.text(i - w / 2, a_.value + 0.3, numv(a_, 1), ha="center", fontsize=fs(8.5, g))
        ax.text(i + w / 2, b_.value + 0.3, numv(b_, 1), ha="center", fontsize=fs(8.5, g))
    ax.set_xticks(xx)
    ax.set_xticklabels([n for n, _ in FSETS])
    ax.set_ylabel("Mean absolute error (days)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    ng, npl = numv(gr[-1], 0, "n", True), numv(pl[-1], 0, "n", True)
    CAPTIONS["fig08_error_by_growth_phase"] = (
        "Mean absolute error by developmental phase (growth, true age ≤ 70 d; plateau, > 70 d) for the six feature sets, under leave-one-room-out validation (primary pipeline). "
        f"Bars are the mean over the three held-out rooms of each room's MAE (days) in the phase; the fused model's phases contain {ng} (growth) and {npl} (plateau) 6 h windows. No interval is drawn on the bars; 95% room-day bootstrap intervals for the fused, morphometry-only, V3 and V2 models are in sheet section S27 and the paired differences in Fig S1. Conditional on this cohort and these rooms. MAE = mean absolute error."
    )
    finish(fig, "fig08_error_by_growth_phase")


def fig09():
    CUR[0] = "fig09"
    D = derived("body_size/body_size_daily.csv")
    fig, ax, g = P.new_figure(10, 4.8, W2MM)
    for r in ROOMS:
        d = D[D.room == r].sort_values("age_day")
        ax.plot(d.age_day, d.ratio, lw=lw(2, g), color=ROOM_COL[r], label=ROOM_LABEL[r])
    ax.axvline(70, color="#333", ls="--", lw=lw(1.5, g))
    ax.text(71.5, 0.98, "day 70", fontsize=fs(8.5, g), color="#333", va="bottom")
    ax.set_xlabel("Flock age (days)")
    ax.set_ylabel("Body height relative to first week (ratio)")
    ax.set_xlim(28, 185)
    ax.legend(fontsize=fs(9, g), loc="upper left")
    fig.tight_layout()
    pk = one(
        "S36_body_size",
        "peak ratio (maximum of the 7-day centred rolling median of the daily ratio)",
        None,
        "mean of rooms",
    )
    reg_other(f"{pk.value:.2f}", pk.value, 2, pk.result_id, "value")
    CAPTIONS["fig09_body_size_trajectory"] = (
        f"Body-size trajectory of the three rooms. Each line is the daily median of the relative median body height (after the causal scale-artefact filter) divided by the median of the room's first seven observed daily medians, so every room starts at 1. "
        f"The mean over rooms of the peak ratio (maximum of the 7-day centred rolling median) is {pk.value:.2f}. The dashed line marks day 70. Rooms differ in the size of the trajectory (the between-room difference is in sheet section S36), so the morphometric features are referenced to each room's own first week. "
        "Descriptive; conditional on this cohort and these three rooms."
    )
    finish(fig, "fig09_body_size_trajectory")


def fig10():
    CUR[0] = "figS1"
    fig, ax, g = P.new_figure(9.2, 4.0, W2MM)
    contrasts = list(dict.fromkeys(S[S.section == "S07_paired_fusion"].variant))
    y0 = np.arange(len(contrasts))[::-1]
    short = {
        "fused - morphometry-only": "Fused −\nmorphometry-only",
        "fused - V3 (body-size only)": "Fused − V3",
        "V2 fused - morphometry-only": "V2 fused −\nmorphometry-only",
        "V2 fused - V3 (body-size only)": "V2 fused − V3",
    }
    offs = (0.27, 0.09, -0.09, -0.27)
    scopes = ROOMS + ["macro"]
    cols = {**ROOM_COL, "macro": "k"}
    for yy, c in zip(y0, contrasts):
        for off, sc in zip(offs, scopes):
            r = one("S07_paired_fusion", "MAE difference (days)", c, sc)
            ax.errorbar(
                r.value,
                yy + off,
                xerr=[[r.value - r.ci_lo], [r.ci_hi - r.value]],
                fmt="o" if sc != "macro" else "D",
                color=cols[sc],
                ms=ms(6, g),
                capsize=3 * g + 1,
                lw=lw(1.8, g),
            )
    ax.axvline(0, color="k", ls="--", lw=lw(1.2, g))
    ax.set_yticks(y0)
    ax.set_yticklabels([short[c] for c in contrasts])
    ax.set_xlabel("MAE difference (days)\n(negative: first model has the lower error)")
    ax.grid(axis="y", visible=False)
    ax.legend(
        [
            Line2D([], [], color=cols[sc], marker="o" if sc != "macro" else "D", ls="", ms=ms(6, g))
            for sc in scopes
        ],
        [ROOM_LABEL.get(sc, "Mean of rooms") for sc in scopes],
        loc="lower right",
        fontsize=fs(8.5, g),
    )
    n = numv(one("S07_paired_fusion", "MAE difference (days)", contrasts[0], "macro"), 0, "n", True)
    ax.text(
        0.01, 0.99, f"n = {n} paired windows", transform=ax.transAxes, fontsize=fs(8.5, g), va="top"
    )
    fig.tight_layout()
    CAPTIONS["figS1_paired_fusion_differences"] = (
        "Paired differences in mean absolute error between models evaluated on identical held-out windows (leave-one-room-out): fused minus morphometry-only, fused minus V3 (strict body-size-only comparator), and the same for V2 (audio referenced to each room's own first week). "
        "Circles show the per-room difference and diamonds the mean of the three rooms; bars are 95% intervals from a room-day block bootstrap (2,000 resamples, days resampled within rooms). Negative values mean the first model has the lower error. "
        "The intervals are conditional on this cohort and these three rooms, not on new farms. n = number of paired 6 h windows. MAE = mean absolute error."
    )
    finish(fig, "figS1_paired_fusion_differences")


def fig11():
    CUR[0] = "fig11"
    fig, axes, g = P.new_figure(11, 4.4, W2MM, 1, 2, sharey=True)
    ks = [0, 3, 5, 7, 10, 15, 20]
    for ax, ph, ttl in zip(
        axes,
        ["growth (<=70 d)", "plateau (>70 d)"],
        ("(a) Growth phase (≤70 d)", "(b) Plateau (>70 d)"),
    ):
        ax.plot(ks, [-k for k in ks], "k--", lw=lw(1.2, g), label="Perfect recovery")
        for mode, c, mk, lab in (
            ("morph_only", COL["morph"], "o", "Morphometry shifted"),
            ("all_sensor", COL["audio_bio"], "s", "All sensors shifted"),
        ):
            rows = [
                one(
                    "S28_gap_recovery_curve",
                    "median corrected gap (days)",
                    "unperturbed (all eligible anchors) | k=0 d" if k == 0 else f"{mode} | k={k} d",
                    ph,
                )
                for k in ks
            ]
            y = np.array([r.value for r in rows])
            lo = np.array([r.ci_lo for r in rows])
            hi = np.array([r.ci_hi for r in rows])
            ax.errorbar(
                ks,
                y,
                yerr=[y - lo, hi - y],
                marker=mk,
                color=c,
                lw=lw(1.8, g),
                ms=ms(6, g),
                capsize=3 * g + 1,
                label=lab,
            )
            for r in rows:
                numv(r, 1)
        ax.set_xlabel("Injected delay (days behind schedule)")
        ax.set_title(ttl, fontsize=fs(10, g))
    axes[0].set_ylabel("Detected growth gap (days)")
    axes[1].legend(fontsize=fs(8.5, g), loc="lower left")
    fig.tight_layout()
    CAPTIONS["fig11_gap_recovery"] = (
        "Recovery of synthetic age substitution by the calibrated detector. For every held-out 6 h anchor with an observation k days earlier at the same hour of day, the model was fed the sensor readings from k days earlier (all sensor blocks, or morphometry only) while the label stayed today's age; "
        "the detected gap is predicted minus true age after a bias correction fitted on the two training rooms only (inner calibration). Points are the median gap of trailing 7-calendar-day windows (valid with at least 4 days of data), error bars are 95% intervals from a room-week block bootstrap (2,000 resamples), "
        "conditional on this cohort and these three rooms. The 0 d point is the unperturbed run on all eligible anchors; the dashed line is perfect recovery (gap = −k). Window counts and weeks per room are in sheet section S28. (a) growth phase (age ≤ 70 d); (b) plateau phase (> 70 d)."
    )
    finish(fig, "fig11_gap_recovery")


def fig12():
    CUR[0] = "fig12"
    AP = derived("core_primary/preds_ageblock_primary.csv", parse_dates=["hour"])
    AP = AP[AP.model == "fused"]
    APfull = AP
    AP = plot_valid(AP)
    LP = derived("core_primary/preds_loro_primary.csv", parse_dates=["hour"])
    LP = LP[LP.model == "fused"]
    LP = plot_valid(LP)
    blocks = list(dict.fromkeys(AP.block))
    BT = {0: "backward extrapolation", 4: "forward extrapolation"}
    fig, (a, b), g = P.new_figure(12, 5.6, W2MM, 1, 2)
    for i, blk in enumerate(blocks):
        te = APfull[APfull.block == blk]
        lo, hi = te.y.min(), te.y.max()
        if i > 0:
            a.axvline(lo, color="#bbbbbb", lw=lw(0.8, g), zorder=0)
        m = num("S05_ageblock", "MAE", "primary fused", blk, 1, detail="AGEBLOCK_12h")
        a.text(
            (lo + hi) / 2,
            LIM[1] - 3,
            f"B{i + 1}\n{m} d",
            ha="center",
            va="top",
            fontsize=fs(8.5, g),
            color="#333333",
            linespacing=1.1,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9),
            zorder=4,
        )
        if i in BT:
            trn = APfull[APfull.block != blk].y
            edge = float(trn.min() if i == 0 else trn.max())
            a.hlines(
                edge,
                lo,
                hi,
                colors=POOLED_COL,
                linestyles=(0, (3, 2)),
                lw=lw(1.3, g),
                label="Nearest training age" if i == 0 else None,
                zorder=3,
            )
            a.text(
                lo + 1,
                edge + (-2 if i == 0 else 2),
                "nearest training age",
                fontsize=fs(7.5, g),
                color=POOLED_COL,
                va="top" if i == 0 else "bottom",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.8),
            )
    for bt, colr in BCOL.items():
        idx = [i for i in range(len(blocks)) if BT.get(i, "internal gap") == bt]
        d = AP[AP.block.isin([blocks[i] for i in idx])]
        a.scatter(
            d.y,
            d.pred,
            s=sa(6, g),
            alpha=0.35,
            color=colr,
            edgecolors="none",
            label=bt.capitalize(),
            rasterized=True,
        )
    a.plot(LIM, LIM, "k--", lw=lw(1.2, g), label="Identity")
    b.plot(LIM, LIM, "k--", lw=lw(1.2, g))
    for r in ROOMS:
        d = LP[LP.room == r]
        b.scatter(
            d.y,
            d.pred,
            s=sa(6, g),
            alpha=0.25,
            color=ROOM_COL[r],
            edgecolors="none",
            label=ROOM_LABEL[r],
            rasterized=True,
        )
    for ax in (a, b):
        ax.set_xlim(LIM)
        ax.set_ylim(LIM)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("True age (days)")
    a.set_ylabel("Predicted age (days)")
    a.set_title("(a) Leave-one-age-block-out", loc="left", fontsize=fs(10, g))
    b.set_title("(b) Leave-one-room-out", loc="left", fontsize=fs(10, g))
    pooled = one("S05_ageblock", "MAE", "primary fused", "pooled fold-mean", "AGEBLOCK_12h")
    excl = one(
        "S05_ageblock", "MAE", "primary fused", "pooled fold-mean excl. block 1", "AGEBLOCK_12h"
    )
    macro = one("S03_loro", "MAE", "primary fused", "macro")
    r2m = one("S03_loro", "R2", "primary fused", "macro")
    a.text(
        0.02,
        0.80,
        f"MAE {numv(pooled, 2)} d\n(mean of 5 blocks)",
        transform=a.transAxes,
        fontsize=fs(8.5, g),
        va="top",
    )
    b.text(
        0.02,
        0.80,
        f"MAE {numv(macro, 2)} d\nR² {numv(r2m, 3)}\n(mean of 3 rooms)",
        transform=b.transAxes,
        fontsize=fs(8.5, g),
        va="top",
    )
    a.legend(fontsize=fs(7.8, g), loc="lower right", markerscale=2.5)
    b.legend(fontsize=fs(8, g), loc="lower right", markerscale=2.5)
    fig.tight_layout()
    CAPTIONS["fig12_ageblock_predicted_vs_true"] = (
        f"Predicted against true flock age for the primary fused model under (a) leave-one-age-block-out validation (12 h embargo; windows reading any observation inside the held-out block or its embargo were removed from training) and (b) leave-one-room-out validation. "
        "(a) The age range of the complete-case windows was divided into five blocks (quintiles of age, B1–B5); each block was held out in turn and the model, trained on the remaining blocks of all three rooms, predicted its 6 h windows. Colour marks backward extrapolation (B1), internal gaps (B2–B4) and forward extrapolation (B5); "
        f"labels give each block's MAE (days). The mean over blocks is {pooled.value:.2f} d ({excl.value:.2f} d without B1, whose relative-morphometry reference is empty because the block contains the whole reference week). (b) Points coloured by held-out room; the annotation is the mean over the three rooms. "
        "The dashed line is perfect prediction. Windows with no usable window features (about 17%; they all receive one constant prediction) are not drawn but are included in every MAE and R² reported. No interval is drawn: blocks and rooms are not replicates, and any interval would be conditional on this cohort and these rooms. MAE = mean absolute error."
    )
    finish(fig, "fig12_ageblock_predicted_vs_true")


LABS = ["Brooding/early (<60 d)", "Growing (60-110 d)", "Mature (>110 d)"]
TICK = ["Brooding/early\n(<60 d)", "Growing\n(60-110 d)", "Mature\n(>110 d)"]
CONF_W_MM = 113.8  # 6.4 in x 0.7: keeps the original text-to-matrix proportions at the 7 pt floor


def confusion_fig(key, title, mscope, fname_):
    CUR[0] = fname_[:5]
    fig, ax, g = P.new_figure(6.4, 4.8, CONF_W_MM)
    cm = np.array(
        [
            [one("S20_confusion", f"true {t} -> predicted {p}", None, key).value for p in LABS]
            for t in LABS
        ]
    )
    cmn = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(3):
        for j in range(3):
            r_ = one("S20_confusion", f"true {LABS[i]} -> predicted {LABS[j]}", None, key)
            ax.text(
                j,
                i,
                f"{cmn[i, j]:.2f}\n({numv(r_, 0, 'value', True)})",
                ha="center",
                va="center",
                fontsize=fs(10, g),
                color="white" if cmn[i, j] > 0.5 else "black",
            )
            reg_other(f"{cmn[i, j]:.2f}", cmn[i, j], 2, r_.result_id, "row-normalised value")
    lab = ["Brooding/early\n(<60 d)", "Growing\n(60-110 d)", "Mature\n(>110 d)"]
    ax.set_xticks(range(3))
    ax.set_xticklabels(lab, fontsize=fs(9, g))
    ax.set_yticks(range(3))
    ax.set_yticklabels(lab, fontsize=fs(9, g))
    ax.set_xlabel("Predicted stage", fontsize=fs(11, g))
    ax.set_ylabel("True stage", fontsize=fs(11, g))
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cb.set_label("Row-normalised proportion", fontsize=fs(10.5, g))
    cb.ax.tick_params(labelsize=fs(10, g))
    ax.set_title(title, fontsize=fs(10, g), color="black")
    fig.tight_layout()
    bal = numv(one("S19_classification", "balanced_accuracy", None, mscope), 3)
    f1 = numv(one("S19_classification", "macro_F1", None, mscope), 3)
    n = numv(one("S19_classification", "n", None, mscope), 0, "value", True)
    CAPTIONS[fname_] = (
        f"Three-stage classification of flock age under leave-one-room-out validation: {'all held-out windows pooled' if key == 'pooled' else 'held-out ' + title}. Stages are fixed age classes: brooding/early (0–60 d), growing (60–110 d) and mature (110–200 d). "
        "A cumulative-binary gradient-boosting classifier was trained on the 6 h windows of two rooms and applied to the third. Cells show the proportion of true-stage windows (rows sum to 1) and, in brackets, the number of windows. Balanced accuracy (mean of the three per-class recalls) is "
        f"{bal}, macro-F1 (unweighted mean of the three per-class F1 scores) is {f1}, and {n} windows were evaluated. No interval is shown: the rooms are the replicates, and any interval would be conditional on this cohort and these rooms. F1 = harmonic mean of precision and recall."
    )
    finish(fig, fname_)


def fig13_summary():
    CUR[0] = "figS2"
    scopes = [(r, ROOM_LABEL[r], r) for r in ROOMS] + [
        ("pooled", "All rooms", "pooled (all held-out windows)")
    ]
    fig, (ax, ax2), g = P.new_figure(11, 4.4, W2MM, 1, 2, gridspec_kw={"width_ratios": [1.0, 1.15]})
    x = np.arange(3)
    w = 0.2
    for k_, (key, title, mscope) in enumerate(scopes):
        vals = [one("S19_classification", f"recall_{l}", None, mscope) for l in LABS]
        ax.bar(
            x + (k_ - 1.5) * w,
            [v.value for v in vals],
            w,
            color=ROOM_COL.get(key, POOLED_COL),
            label=title,
        )
        for xi, v in zip(x + (k_ - 1.5) * w, vals):
            ax.text(
                xi,
                v.value + 0.02,
                numv(v, 2).lstrip("0"),
                ha="center",
                fontsize=fs(8, g),
                rotation=90,
                va="bottom",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(["<60 d", "60–110 d", ">110 d"])
    ax.set_ylim(0, 1.35)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Recall")
    ax.set_xlabel("True stage")
    ax.set_title("(a) Per-class recall", loc="left", fontsize=fs(10, g))
    ax.legend(loc="upper center", ncol=2, fontsize=fs(8.5, g))
    ax2.axis("off")
    ax2.set_title("(b) Supports and summary metrics", loc="left", fontsize=fs(10, g))
    hdr = ["", "n", "supports\n(<60 / 60–110 / >110 d)", "balanced\naccuracy", "macro-F1"]
    rows_ = []
    for key, title, mscope in scopes:
        sup = "/".join(
            numv(one("S19_classification", f"support_{l}", None, mscope), 0, "value", True)
            for l in LABS
        )
        rows_.append(
            [
                title,
                numv(one("S19_classification", "n", None, mscope), 0, "value", True),
                sup,
                numv(one("S19_classification", "balanced_accuracy", None, mscope), 3),
                numv(one("S19_classification", "macro_F1", None, mscope), 3),
            ]
        )
    tb = ax2.table(
        cellText=rows_,
        colLabels=hdr,
        loc="center",
        cellLoc="center",
        colWidths=[0.2, 0.11, 0.36, 0.17, 0.16],
    )
    tb.auto_set_font_size(False)
    tb.set_fontsize(fs(8.5, g))
    tb.scale(1, 2.2)
    for (r_, c_), cell in tb.get_celld().items():
        cell.set_linewidth(0.3)
        cell.set_edgecolor("#999999")
        cell.get_text().set_fontsize(fs(8.5, g))
        if r_ == 0:
            cell.set_facecolor("#f1f1f1")
    fig.tight_layout()
    CAPTIONS["figS2_recall_and_metrics"] = (
        "Per-class recall, class supports and summary metrics of the three-stage classifier (leave-one-room-out; confusion matrices in Figs 13a–d). (a) Recall of each true stage for each held-out room and for all held-out windows pooled (dark grey). "
        "(b) Number of evaluated windows, class supports (windows per true stage), balanced accuracy (mean of the three recalls) and macro-F1. Balanced accuracy and macro-F1 are given because the mature stage dominates the supports. "
        "No interval is shown: the three rooms are the replicates, and any interval would be conditional on this cohort and these rooms. F1 = harmonic mean of precision and recall."
    )
    finish(fig, "figS2_recall_and_metrics")


def fig14():
    CUR[0] = "fig14"
    fig, ax, g = P.new_figure(12.5, 5.2, W2MM, 1, 2)

    def thermal(prefix):
        d = S[(S.section == "S29_thermal_controls") & S.variant.str.startswith(prefix)]
        assert len(d), prefix
        out = {}
        for m in ("MAE", "R2"):
            dm = d[d.metric == m]
            out[m] = dm.value.mean()
            reg_other(
                f"{out[m]:.2f}" if m == "R2" else f"{out[m]:.1f}",
                out[m],
                2 if m == "R2" else 1,
                dm.iloc[0].result_id,
                f"mean over {len(dm)} tests with prefix {prefix}",
            )
        return out["MAE"], out["R2"]

    def fromsheet(mrow, rrow):
        reg_other(f"{mrow.value:.1f}", mrow.value, 1, mrow.result_id, "value")
        reg_other(f"{rrow.value:.2f}", rrow.value, 2, rrow.result_id, "value")
        return mrow.value, rrow.value

    F = [
        ("Thermal only (real join)", *thermal("thermal_diag_thermal_only_real"), "leak"),
        ("Fused + thermal (real join)", *thermal("thermal_diag_fused_plus_thermal_real"), "leak"),
        ("Thermal only (values shuffled)", *thermal("thermal_diag_thermal_only_shuffled"), "leak"),
        ("Fused, no thermal (reported model)", *thermal("thermal_diag_fused_no_thermal"), "ok"),
        (
            "Fused + thermal (randomly assigned)",
            *thermal("thermal_diag_fused_plus_thermal_random"),
            "ok",
        ),
        ("Thermal only (randomly assigned)", *thermal("thermal_diag_thermal_only_random"), "ctrl"),
        (
            "Hour-of-day only",
            *fromsheet(
                one("S26_negative_controls", "MAE", "clock-only (hour of day)", "macro"),
                one("S26_negative_controls", "R2", "clock-only (hour of day)", "macro"),
            ),
            "ctrl",
        ),
        (
            "Shuffled age label",
            *fromsheet(
                one(
                    "S14_shuffled_label",
                    "MAE (shuffled mean, 95% permutation range)",
                    None,
                    "macro",
                ),
                one(
                    "S14_shuffled_label", "R2 (shuffled mean, 95% permutation range)", None, "macro"
                ),
            ),
            "ctrl",
        ),
        (
            "Sensor health only",
            *fromsheet(
                one("S26_negative_controls", "MAE", "sensor-health-only", "macro"),
                one("S26_negative_controls", "R2", "sensor-health-only", "macro"),
            ),
            "ctrl",
        ),
        (
            "Train summer -> test autumn",
            *fromsheet(
                one("S06_season", "MAE", "primary fused", "train_summer_test_fall", "SEASON_12h"),
                one("S06_season", "R2", "primary fused", "train_summer_test_fall", "SEASON_12h"),
            ),
            "ctrl",
        ),
        (
            "Train autumn -> test summer",
            *fromsheet(
                one("S06_season", "MAE", "primary fused", "train_fall_test_summer", "SEASON_12h"),
                one("S06_season", "R2", "primary fused", "train_fall_test_summer", "SEASON_12h"),
            ),
            "ctrl",
        ),
    ]
    Dn = pd.DataFrame(F, columns=["name", "MAE", "R2", "kind"]).sort_values("MAE")
    a, b = ax
    a.barh(Dn.name, Dn.MAE, color=[NEG_CMAP[k] for k in Dn.kind])
    a.invert_yaxis()
    for i, v in enumerate(Dn.MAE):
        a.text(v + 0.7, i, f"{v:.1f}", va="center", fontsize=fs(8.5, g))
    a.set_xlabel("Mean absolute error (days)")
    a.set_title("(a) Error", loc="left", fontsize=fs(10, g))
    a.tick_params(axis="y", labelsize=fs(10, g))
    R2c = Dn.R2.clip(lower=-1)
    b.barh(Dn.name, R2c, color=[NEG_CMAP[k] for k in Dn.kind])
    b.invert_yaxis()
    b.set_yticklabels([])
    for i, (v, vc) in enumerate(zip(Dn.R2, R2c)):
        b.text(
            vc + (0.03 if vc >= 0 else -0.03),
            i,
            f"{v:.2f}" if v > -1 else f"{v:.1f}",
            va="center",
            ha="left" if vc >= 0 else "right",
            fontsize=fs(8.5, g),
        )
    b.axvline(0, color="k", lw=lw(1, g))
    b.set_xlim(-1.45, 1.25)
    b.set_xlabel("R² (clipped at −1)")
    b.set_title("(b) Variance explained", loc="left", fontsize=fs(10, g))
    b.legend(
        [plt.Rectangle((0, 0), 1, 1, color=NEG_CMAP[k]) for k in ("ok", "leak", "ctrl")],
        ["Reported model", "Calendar leakage", "Negative controls"],
        fontsize=fs(8.5, g),
        loc="upper left",
    )
    fig.tight_layout()
    CAPTIONS["fig14_negative_controls"] = (
        "Negative controls for leave-one-room-out age estimation (primary pipeline). Bars give (a) the mean absolute error and (b) R² (clipped at −1; unclipped value printed) of each test, averaged over the held-out rooms (and, for the thermal tests, over seeds). "
        "Red: thermal diagnostics that expose calendar leakage (a house-wide thermal trace joined to every room by timestamp predicts age through the calendar, not through the birds); blue: the reported model without thermal data and with a randomly assigned thermal trace; "
        "grey: negative controls — hour of day only, the shuffled-label control (age labels of the two training rooms permuted as whole room-day blocks, mean of 100 permutations), sensor health only, and seasonal extrapolation (summer = age < 88 d, 12 h embargo; the summer-held-out fold has no relative-morphometry features). "
        "Conditional on this cohort and these three rooms. MAE = mean absolute error."
    )
    finish(fig, "fig14_negative_controls")


def figC():
    CUR[0] = "fig10"
    cur = derived("clustering/set4_clustering_curves.csv")
    fig, ax, g = P.new_figure(10, 4.8, W2MM)
    td = {}
    for r in ROOMS:
        d = cur[cur.room == r]
        row = one("S23_clustering", "primary_transition_day", "primary", r)
        td[r] = row
        ax.plot(
            d.age_day,
            d.early_fraction_smoothed,
            lw=lw(2.4, g),
            color=ROOM_COL[r],
            label=f"{ROOM_LABEL[r]} — crosses 50% at day {numv(row, 0)}",
        )
        ax.plot(
            [row.value],
            [0.5],
            "o",
            ms=ms(8, g),
            color=ROOM_COL[r],
            markeredgecolor="white",
            markeredgewidth=lw(1.4, g),
            zorder=5,
        )
    ax.axhline(0.5, color="#777", ls=":", lw=lw(1.1, g))
    ax.axvline(70, color="#333", ls="--", lw=lw(1.5, g))
    ax.text(
        71.5, 0.92, "day-70 boundary from\nsupervised analyses", fontsize=fs(8.5, g), color="#333"
    )
    ax.set_xlabel("Flock age (days)")
    ax.set_ylabel("Proportion of hours in the\nearly-development cluster")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(25, 185)
    ax.legend(fontsize=fs(9, g), loc="upper right", bbox_to_anchor=(0.995, 0.80))
    fig.tight_layout()
    lo, hi = min(r.value for r in td.values()), max(r.value for r in td.values())
    CAPTIONS["fig10_clustering_crossover"] = (
        f"Unsupervised two-phase structure of the fused feature space. Each room was clustered separately (median imputation, within-room standardisation, 20 principal components, k-means with k = 2, seed 0) using only the sensor feature window means of the primary pipeline's complete-case 6 h windows; "
        f"age, date and time of day were withheld. Lines show the share of hours assigned to the cluster with the younger median age as a function of age (centred 7-day rolling mean of the daily share); dots mark the pre-registered 50% crossing (first age after day 35 at which the smoothed share falls below 0.5); "
        f"the crossings of the three rooms span {lo:.0f}–{hi:.0f} d. The dashed vertical line is the supervised day-70 boundary. The two-phase structure is clear in rooms 2 and 6 and weaker in room 7 (silhouette favours more than two clusters there; sheet S23). "
        "This is descriptive corroboration, not an interval estimate; conditional on this cohort and these three rooms."
    )
    finish(fig, "fig10_clustering_crossover")


FIGS = [
    ("fig01_coverage_timeline", fig01),
    ("fig04_loro_predicted_vs_true", fig04),
    ("fig05_model_family_comparison", fig05_models),
    ("fig06_attribution_by_modality", fig06),
    ("fig07_feature_attribution", fig07),
    ("fig08_error_by_growth_phase", fig08),
    ("fig09_body_size_trajectory", fig09),
    ("fig10_clustering_crossover", figC),
    ("fig11_gap_recovery", fig11),
    ("fig12_ageblock_predicted_vs_true", fig12),
    (
        "fig13a_confusion_room2",
        lambda: confusion_fig("room2", "Room 2", "room2", "fig13a_confusion_room2"),
    ),
    (
        "fig13b_confusion_room6",
        lambda: confusion_fig("room6", "Room 6", "room6", "fig13b_confusion_room6"),
    ),
    (
        "fig13c_confusion_room7",
        lambda: confusion_fig("room7", "Room 7", "room7", "fig13c_confusion_room7"),
    ),
    (
        "fig13d_confusion_all_rooms",
        lambda: confusion_fig(
            "pooled", "All rooms", "pooled (all held-out windows)", "fig13d_confusion_all_rooms"
        ),
    ),
    ("fig14_negative_controls", fig14),
    ("figS1_paired_fusion_differences", fig10),
    ("figS2_recall_and_metrics", fig13_summary),
]
if A.only:
    FIGS = [f for f in FIGS if f[0].startswith(tuple(A.only.split(",")))]
for name, fn in FIGS:
    fn()
if A.only:
    sys.exit(0)
chk = pd.DataFrame(REGISTRY)
EXEMPT = ("value/n*100", "row-normalised value", "min over rooms", "max over rooms")
chk["match"] = chk.apply(
    lambda r: abs(float(str(r.printed).replace(",", "").replace("+", "")) - r.sheet_value)
    <= 0.5 * 10 ** (-int(r.decimals)) + 1e-9
    or r.field in EXEMPT,
    axis=1,
)
look = S.set_index("result_id")


def relookup(r):
    if not re.match(r"^S\d", str(r.result_id)) or r.field not in look.columns:
        return True
    return abs(float(look.loc[r.result_id, r.field]) - r.sheet_value) < 1e-12


chk["sheet_lookup_equal"] = chk.apply(relookup, axis=1)
chk.to_csv(CHECKS / "figure_number_checks.csv", index=False)
pd.DataFrame(SIZES).to_csv(CHECKS / "figure_sizes_fonts.csv", index=False)
(OUT / "captions.md").write_text(
    "# Figure captions (generated by make_figures.py from results/results_sheet.csv)\n\n"
    + "\n\n".join(f"**{k}.** {v}" for k, v in CAPTIONS.items())
    + "\n"
)
rows = [
    dict(file=p.name, sha256=sha(p), bytes=p.stat().st_size)
    for p in sorted(OUT.glob("*"))
    if p.is_file() and p.suffix in (".svg", ".pdf", ".png", ".md")
]
pd.DataFrame(rows).to_csv(CHECKS / "figure_hashes.csv", index=False)
bad = chk[~(chk.match & chk.sheet_lookup_equal)]
print(f"{len(chk)} printed numbers checked, {len(bad)} mismatches; {len(FIGS)} figure files")
