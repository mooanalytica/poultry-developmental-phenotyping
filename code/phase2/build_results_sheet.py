"""Collect every reported number of a run into one long-format results sheet.
Usage: python3 build_results_sheet.py RUN_DIR OUT_CSV
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
RUN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
ROWS = []
COUNT = {}


def add(
    section,
    metric,
    room,
    value=np.nan,
    variant="",
    lo=np.nan,
    hi=np.nan,
    n=np.nan,
    detail="",
    source="",
    text="",
):
    COUNT[section] = COUNT.get(section, 0) + 1
    ROWS.append(
        dict(
            result_id=f"{section.split('_')[0]}-{COUNT[section]:04d}",
            section=section,
            metric=metric,
            variant=variant,
            room_or_fold=room,
            value=value,
            text_value=text,
            ci_lo=lo,
            ci_hi=hi,
            n=n,
            detail=detail,
            source_script=source,
        )
    )


def rd(stage, name):
    return pd.read_csv(RUN / stage / name)


ROOMS = ["room2", "room6", "room7"]
cov = rd("data", "coverage.csv")
src = "code/phase2/data_description.py"
for _, r in cov.iterrows():
    add(
        "S01_data_coverage",
        r.quantity,
        r.room,
        r["count"],
        n=r.denominator,
        detail=f"denominator: {r.denominator_definition}",
        source=src,
    )
for _, r in rd("data", "key_dates.csv").iterrows():
    add(
        "S02_key_dates",
        r["item"],
        r.room,
        r.age_days,
        text=str(r.value),
        detail=str(r.note) if pd.notna(r.note) else "",
        source=src,
    )
VAR = {
    ("primary", "fused"): "primary fused",
    ("primary", "morph_only"): "primary morphometry-only",
    ("primary", "morph_body_only"): "V3 body-size only",
    ("V2", "fused"): "V2 fused (audio first-week reference)",
}
src = "code/phase2/core_models.py"
for tag, f in (("primary", "core_primary/core_primary.csv"), ("V2", "core_v2/core_V2.csv")):
    c = pd.read_csv(RUN / f)
    L = c[c.analysis == "LORO"]
    nmap = L[L.metric == "n"].set_index(["model", "fold"]).value.to_dict()
    for _, r in L.iterrows():
        v = VAR.get((tag, r.model))
        if v is None:
            continue
        add(
            "S03_loro",
            r.metric,
            r.fold,
            r.value,
            variant=v,
            n=(
                nmap.get((r.model, r.fold), np.nan)
                if r.fold != "macro"
                else sum(nmap.get((r.model, x), 0) for x in ROOMS)
            ),
            source=src,
        )
    P_ = pd.read_csv(
        RUN
        / (
            "core_primary/preds_loro_primary.csv"
            if tag == "primary"
            else "core_v2/preds_loro_V2.csv"
        )
    )
    for model in P_.model.unique():
        if (tag, model) not in VAR:
            continue
        v = VAR[(tag, model)]
        q = P_[P_.model == model].assign(e=lambda d: (d.pred - d.y).abs())
        for ph, sel in (("growth (<=70 d)", q.y <= 70), ("plateau (>70 d)", q.y > 70)):
            per = {r: q[sel & (q.room == r)] for r in ROOMS}
            for r in ROOMS:
                add(
                    "S04_loro_growth_plateau",
                    f"MAE {ph}",
                    r,
                    per[r].e.mean(),
                    variant=v,
                    n=len(per[r]),
                    source="code/phase2/build_results_sheet.py (from per-window predictions)",
                )
            add(
                "S04_loro_growth_plateau",
                f"MAE {ph}",
                "macro",
                np.mean([per[r].e.mean() for r in ROOMS]),
                variant=v,
                n=sum(len(per[r]) for r in ROOMS),
                source="code/phase2/build_results_sheet.py (from per-window predictions)",
            )
    for _, r in c[c.analysis.str.startswith(("AGEBLOCK", "SEASON"))].iterrows():
        v = VAR.get((tag, r.model))
        if v is None:
            continue
        sec = "S05_ageblock" if r.analysis.startswith("AGEBLOCK") else "S06_season"
        add(sec, r.metric, r.fold, r.value, variant=v, detail=r.analysis, source=src)
for _, r in rd("paired", "paired_fusion.csv").iterrows():
    add(
        "S07_paired_fusion",
        "MAE difference (days)",
        r.heldout,
        r.MAE_diff,
        variant=r.contrast,
        lo=r.ci_lo,
        hi=r.ci_hi,
        n=r.n_pairs,
        detail=f"MAE_A={r.MAE_A:.6f}; MAE_B={r.MAE_B:.6f}; n_days={r.n_days}; 95% room-day bootstrap, conditional on this cohort and these rooms",
        source="code/analysis/paired_fusion.py",
    )
for _, r in rd("room_id", "room_id_before_after_V2.csv").iterrows():
    add(
        "S08_room_identification",
        "room identification accuracy",
        r.block,
        r.room_id_accuracy,
        variant=r.version,
        n=r.n_features,
        detail=f"chance={r.chance:.4f}; per-block accuracy {r.per_block_accuracy}",
        source="code/phase2/room_id.py",
    )
src = "code/detection.py"
D_ = rd("detection", "set2_detection_by_room.csv")
for _, r in D_.iterrows():
    base = dict(
        room=r.test_room,
        variant=f"{r['mode']} | {r.phase} | k={int(r.k_days)} d | rule={r.rule}",
        n=r.n_pos_windows,
        source=src,
    )
    det = f"n_neg_windows={int(r.n_neg_windows)}; n_weeks={int(r.n_weeks)}; threshold={r.threshold}; n_inner_neg={r.n_inner_neg_windows}; estimable={r.estimable}"
    add(
        "S09_detection",
        "AUC",
        base["room"],
        r.AUC,
        variant=base["variant"],
        lo=r.AUC_ci_lo_frozen,
        hi=r.AUC_ci_hi_frozen,
        n=base["n"],
        detail=det,
        source=src,
    )
    add(
        "S09_detection",
        "sensitivity",
        base["room"],
        r.sensitivity,
        variant=base["variant"],
        lo=r.sensitivity_ci_lo_frozen,
        hi=r.sensitivity_ci_hi_frozen,
        n=base["n"],
        detail=det,
        source=src,
    )
    add(
        "S09_detection",
        "achieved specificity",
        base["room"],
        r.achieved_specificity,
        variant=base["variant"],
        lo=r.achieved_specificity_ci_lo_frozen,
        hi=r.achieved_specificity_ci_hi_frozen,
        n=int(r.n_neg_windows),
        detail=det + f"; target={r.target_specificity}",
        source=src,
    )
    for m, lo, hi in (
        ("AUC", r.AUC_ci_lo_incl_cal, r.AUC_ci_hi_incl_cal),
        ("sensitivity", r.sensitivity_ci_lo_incl_cal, r.sensitivity_ci_hi_incl_cal),
        (
            "achieved specificity",
            r.achieved_specificity_ci_lo_incl_cal,
            r.achieved_specificity_ci_hi_incl_cal,
        ),
    ):
        add(
            "S10_detection_ci_incl_calibration",
            m,
            base["room"],
            r[
                {
                    "AUC": "AUC",
                    "sensitivity": "sensitivity",
                    "achieved specificity": "achieved_specificity",
                }[m]
            ],
            variant=base["variant"],
            lo=lo,
            hi=hi,
            n=base["n"],
            detail="interval also re-derives the threshold from resampled inner weeks",
            source=src,
        )
for _, r in rd("detection", "set2_perturbation_manifest.csv").iterrows():
    v = f"{r['mode']} | {r.phase} | k={int(r.k_days)} d | rule={r.rule}"
    for m in (
        "n_eligible_anchors",
        "n_used_anchors",
        "n_dropped_no_source",
        "n_dropped_source_features_all_nan",
        "n_neg_windows",
        "n_pos_windows",
    ):
        add("S11_perturbation_manifest", m, r.room, r[m], variant=v, source=src)
for _, r in rd("detection", "set2_sparse_blocks.csv").iterrows():
    v = f"{r['mode']} | {r.phase} | k={int(r.k_days)} d | rule={r.rule}"
    for m, col in (
        ("AUC", "AUC"),
        ("sensitivity", "sensitivity"),
        ("achieved specificity", "achieved_specificity"),
    ):
        add(
            "S12_sparse_blocks",
            m,
            r.test_room,
            r[col],
            variant=v,
            n=r.n_blocks_pos,
            detail=f"n_blocks_neg={int(r.n_blocks_neg)}; per room {r.n_blocks_per_room}; estimable={r.estimable}",
            source=src,
        )
for _, r in rd("detection", "set2_gap_recovery_matched.csv").iterrows():
    add(
        "S13_gap_recovery",
        "median corrected gap (days)",
        "all rooms",
        r.median_corrected_gap,
        variant=f"{r['mode']} | {r.phase} | k={int(r.k)} d | rule={r.rule} | {r['class']}",
        n=r.n_windows,
        source=src,
    )
for _, r in rd("shuffle_agg", "shuffled_label_summary.csv").iterrows():
    src_ = "code/shuffled_label.py + code/phase2/aggregate_shuffle.py"
    for m in ("MAE", "RMSE", "R2"):
        add(
            "S14_shuffled_label",
            f"{m} (shuffled mean, 95% permutation range)",
            r.scope,
            r[f"{m}_perm_mean"],
            variant="room-day block permutation, 100 permutations",
            lo=r[f"{m}_perm_lo"],
            hi=r[f"{m}_perm_hi"],
            n=r.n_permutations,
            detail=f"primary={r[f'{m}_primary']:.10f}",
            source=src_,
        )
        add(
            "S14_shuffled_label",
            f"{m} (primary model)",
            r.scope,
            r[f"{m}_primary"],
            variant="primary fused",
            source=src_,
        )
    add(
        "S14_shuffled_label", "p-value MAE", r.scope, r.p_value_MAE, n=r.n_permutations, source=src_
    )
    add("S14_shuffled_label", "p-value R2", r.scope, r.p_value_R2, n=r.n_permutations, source=src_)
g = rd("gru", "gru_stage.csv")
src = "code/phase2/gru_stage.py"
for _, r in g.iterrows():
    for m in ("MAE", "RMSE", "R2"):
        add(
            "S15_gru",
            m,
            r.heldout,
            r[m],
            variant=f"{r.model}" + (f" | seed {int(r.seed)}" if r.seed >= 0 else ""),
            n=r.n_eval,
            source=src,
        )
gr = g[g.model == "GRU time-bounded"]
for m in ("MAE", "RMSE", "R2"):
    per = gr.pivot_table(index="seed", columns="heldout", values=m)
    mac = per.mean(axis=1)
    add(
        "S15_gru",
        f"{m} macro, mean over 5 seeds",
        "macro",
        mac.mean(),
        variant="GRU time-bounded (headline)",
        lo=mac.min(),
        hi=mac.max(),
        n=5,
        detail="lo/hi = min and max over seeds; seed 0 was pre-declared",
        source=src,
    )
    for s, v in mac.items():
        add(
            "S15_gru",
            f"{m} macro",
            "macro",
            v,
            variant=f"GRU time-bounded | seed {int(s)}",
            source=src,
        )
for mname in (
    "HGB fused, same anchors",
    "HGB fused, all windows",
    "GRU Revision 3 construction (seed 0)",
):
    q = g[g.model == mname]
    for m in ("MAE", "RMSE", "R2"):
        add("S15_gru", f"{m} macro", "macro", q[m].mean(), variant=mname, source=src)
for _, r in rd("gru", "gru_legacy_spans.csv").iterrows():
    for m in (
        "n_candidate_sequences",
        "median_span_h",
        "p90_span_h",
        "max_span_h",
        "share_gt_24h",
        "share_gt_72h",
    ):
        add(
            "S16_gru_legacy_spans",
            m,
            r.room,
            r[m],
            variant="Revision 3 24-row sequences",
            source=src,
        )
src = "code/availability_regimes.py"
for _, r in rd("regimes", "set3c_regime_inventory.csv").iterrows():
    for m in ("hours", "anchors_total", "blocks_present"):
        add(
            "S17_regime_inventory",
            m,
            "all rooms",
            r[m],
            variant=r.regime,
            detail=f"rooms: {r.rooms_present}; age coverage {r.age_coverage_d} d",
            source=src,
        )
    for room in ROOMS:
        add("S17_regime_inventory", "hours", room, r[f"hours_{room}"], variant=r.regime, source=src)
        add(
            "S17_regime_inventory",
            "anchors",
            room,
            r[f"anchors_{room}"],
            variant=r.regime,
            source=src,
        )
for _, r in rd("regimes", "set3c_regime_results_by_room.csv").iterrows():
    for m in ("MAE", "RMSE", "R2"):
        add(
            "S18_regime_results",
            m,
            r.heldout,
            r[m],
            variant=f"{r.regime} | {r.model}",
            n=r.n_anchors,
            source=src,
        )
for _, r in rd("regimes", "set3c_regime_results_macro.csv").iterrows():
    for m in ("MAE_macro", "RMSE_macro", "R2_macro"):
        add(
            "S18_regime_results",
            m,
            "macro",
            r[m],
            variant=f"{r.regime} | {r.model}",
            n=r.n_anchors_scored,
            detail=f"rooms scored={int(r.rooms_scored)}",
            source=src,
        )
src = "code/analysis/stage_classification.py"
cl = rd("classification", "set4_classification_metrics.csv")
for _, r in cl.iterrows():
    for c in cl.columns:
        if c == "scope" or pd.isna(r[c]):
            continue
        add("S19_classification", c, r.scope, r[c], n=r.n if "n" in r else np.nan, source=src)
for room in ROOMS + ["pooled"]:
    cm = pd.read_csv(RUN / "classification" / f"set4_confusion_{room}_counts.csv", index_col=0)
    for t in cm.index:
        for p in cm.columns:
            add("S20_confusion", f"true {t} -> predicted {p}", room, cm.loc[t, p], source=src)


def attribution(stage, variant):
    a = pd.read_csv(RUN / stage / "attribution_by_age_band.csv")
    for _, r in a.iterrows():
        add(
            "S21_attribution",
            "share of total |SHAP|",
            r.age_band,
            r.share,
            variant=f"{variant} | {r.block}",
            lo=r.share_min_room,
            hi=r.share_max_room,
            n=r.n_rooms,
            detail=f"mean |SHAP| days={r.mean_abs_shap_days:.6f}; lo/hi = min/max over rooms",
            source="code/attribution.py",
        )


def demeaning(stage, variant):
    s = pd.read_csv(RUN / stage / "set4_demeaning_summary.csv")
    f = pd.read_csv(RUN / stage / "set4_demeaning_folds.csv")
    m = pd.read_csv(RUN / stage / "set4_demeaning_shap_share_macro.csv")
    for _, r in s.iterrows():
        for c in ("c0_drift_rho", "R2", "MAE", "RMSE"):
            add(
                "S22_daily_demeaning",
                c,
                "macro",
                r[c],
                variant=f"{variant} | {r.variant}",
                source="code/analysis/daily_demeaning.py",
            )
    for _, r in f.iterrows():
        for c in ("c0_drift_rho", "R2", "MAE", "RMSE"):
            add(
                "S22_daily_demeaning",
                c,
                r.room,
                r[c],
                variant=f"{variant} | {r.variant}",
                n=r.n,
                source="code/analysis/daily_demeaning.py",
            )
    for _, r in m.iterrows():
        add(
            "S22_daily_demeaning",
            "morphometry share of (morphometry + bioacoustics) |SHAP|",
            r.band,
            r.morph_share_of_morph_plus_bio,
            variant=f"{variant} | {r.variant}",
            detail="demeaning comparison definition",
            source="code/analysis/daily_demeaning.py",
        )
        add(
            "S22_daily_demeaning",
            "morphometry share of total |SHAP|",
            r.band,
            r.morph_share_of_total,
            variant=f"{variant} | {r.variant}",
            source="code/analysis/daily_demeaning.py",
        )


def clustering(stage, variant):
    c = pd.read_csv(RUN / stage / "set4_clustering_summary.csv")
    st = pd.read_csv(RUN / stage / "set4_clustering_seed_summary.csv")
    for _, r in c.iterrows():
        for m in (
            "n",
            "n_features",
            "k_best",
            "silhouette_kbest",
            "k2_boundary_day",
            "k2_balanced_accuracy",
            "k2_ARI_vs_day70",
            "k2_early_cluster_frac",
            "primary_transition_day",
            "silhouette_k2",
            "silhouette_k3",
            "silhouette_k4",
            "silhouette_k5",
            "silhouette_k6",
        ):
            add(
                "S23_clustering",
                m,
                r.room,
                r[m],
                variant=variant,
                n=r.n,
                source="code/analysis/clustering_epochs.py",
            )
    for _, r in st.iterrows():
        add(
            "S23_clustering",
            "transition day over 10 seeds (mean)",
            r.room,
            r.transition_day_mean,
            variant=variant,
            lo=r.transition_day_min,
            hi=r.transition_day_max,
            n=10,
            source="code/analysis/clustering_epochs.py",
        )


attribution("attribution", "primary")
demeaning("demeaning", "primary")
clustering("clustering", "primary")
attribution("attribution_step0", "Step 0 settings")
demeaning("demeaning_step0", "Step 0 settings")
clustering("clustering_step0", "Step 0 settings")
for _, r in rd("ref_step0", "ref_loro_step0.csv").iterrows():
    add(
        "S24_step0_reference",
        r.metric,
        r.fold,
        r.value,
        variant=f"Step 0 settings {r.model}",
        source="code/phase2/ref_loro.py",
    )
CUMN = {
    "step0": "Step 0 (centred filter, row slope, anchor purge)",
    "step1": "Step 1 (past-only filter, row slope, anchor purge)",
    "step2": "Step 2 (+ 12 h window purge, age-block and seasonal)",
}
for tag in ("step0", "step1", "step2"):
    for _, r in pd.read_csv(RUN / f"cum_{tag}" / f"cum_{tag}.csv").iterrows():
        add(
            "S25_cumulative_states",
            r.metric,
            r.fold,
            r.value,
            variant=f"{CUMN[tag]} | {r.model}",
            detail=r.analysis,
            source="code/phase2/cumulative_stage.py",
        )
for _, r in pd.read_csv(RUN / "controls" / "controls.csv").iterrows():
    add(
        "S26_negative_controls",
        r.metric,
        r.fold,
        r.value,
        variant=r.control,
        detail="LORO, primary pipeline",
        source="code/phase2/controls_stage.py",
    )
for _, r in rd("fig_data", "mae_by_phase.csv").iterrows():
    add(
        "S27_mae_by_phase_ci",
        f"MAE {r.phase}",
        r.scope,
        r.MAE,
        variant=f"{r.source} | {r.model}",
        lo=r.ci_lo,
        hi=r.ci_hi,
        n=r.n,
        detail=f"n_days={r.n_days}; 95% room-day bootstrap, conditional on this cohort and these rooms",
        source="code/phase2/derive_figure_data.py",
    )
for _, r in rd("fig_data", "gap_recovery_curves.csv").iterrows():
    add(
        "S28_gap_recovery_curve",
        "median corrected gap (days)",
        f"{r.phase}",
        r.median_gap,
        variant=f"{r['mode']} | k={int(r.k_days)} d",
        lo=r.ci_lo,
        hi=r.ci_hi,
        n=r.n_windows,
        detail=f"n_weeks={int(r.n_weeks)} ({r.n_weeks_per_room}); rolling 7-day calendar windows; same-hour matched anchors; 95% room-week bootstrap",
        source="code/phase2/derive_figure_data.py",
    )
for test, g in rd("controls_thermal", "controls_thermal.csv").groupby("test", sort=False):
    for m in ("MAE", "RMSE", "R2"):
        add(
            "S29_thermal_controls",
            m,
            "mean over folds and seeds",
            g[m].mean(),
            variant=test,
            n=len(g),
            detail="LORO; mean over held-out rooms and seeds, as figures.py",
            source="code/phase2/controls_thermal_stage.py",
        )
FSN = {
    "morph_only": "Morphometry",
    "yolo_motion_only": "Spatial behaviour",
    "flow_only": "Optical flow",
    "audio_bio_only": "Bioacoustics",
    "video_all": "All video",
    "fused": "Fused",
}
c_ = pd.read_csv(RUN / "core_primary/core_primary.csv")
Pp = pd.read_csv(RUN / "core_primary/preds_loro_primary.csv")
for fs, nm in FSN.items():
    for _, r in c_[
        (c_.analysis == "LORO") & (c_.model == fs) & c_.metric.isin(["MAE", "RMSE", "R2", "n"])
    ].iterrows():
        add(
            "S30_loro_feature_sets",
            r.metric,
            r.fold,
            r.value,
            variant=nm,
            source="code/phase2/core_models.py",
        )
    q = Pp[Pp.model == fs].assign(e=lambda d: (d.pred - d.y).abs())
    for ph, sel in (("growth (<=70 d)", q.y <= 70), ("plateau (>70 d)", q.y > 70)):
        per = {r: q[sel & (q.room == r)] for r in ROOMS}
        for r in ROOMS:
            add(
                "S31_phase_mae_feature_sets",
                f"MAE {ph}",
                r,
                per[r].e.mean(),
                variant=nm,
                n=len(per[r]),
                source="code/phase2/build_results_sheet.py (from per-window predictions)",
            )
        add(
            "S31_phase_mae_feature_sets",
            f"MAE {ph}",
            "macro",
            np.mean([per[r].e.mean() for r in ROOMS]),
            variant=nm,
            n=sum(len(per[r]) for r in ROOMS),
            source="code/phase2/build_results_sheet.py (from per-window predictions)",
        )
src = "code/phase2/model_family_stage.py"
mf = rd("model_family", "model_family.csv")
for model, q in mf.groupby("model", sort=False):
    det = f"imputation: {q.imputation.iloc[0]}; scaling: {q.scaling.iloc[0]}; LORO, 6 h evaluation windows, primary pipeline"
    for _, r in q.iterrows():
        for m in ("MAE", "RMSE", "R2", "n"):
            add("S32_model_family", m, r.fold, r[m], variant=model, n=r.n, detail=det, source=src)
    for m in ("MAE", "RMSE", "R2"):
        add(
            "S32_model_family",
            m,
            "macro",
            q[m].mean(),
            variant=model,
            n=q.n.sum(),
            detail=det + "; macro = mean over the three held-out rooms",
            source=src,
        )
gg = rd("gru", "gru_stage.csv")
gg = gg[gg.model == "GRU time-bounded"]
for m in ("MAE", "RMSE", "R2"):
    mac = gg.pivot_table(index="seed", columns="heldout", values=m).mean(axis=1)
    add(
        "S32_model_family",
        m,
        "macro",
        mac.mean(),
        variant="GRU (24 h sequences, 5-seed mean)",
        lo=mac.min(),
        hi=mac.max(),
        n=int(gg[gg.seed == 0].n_eval.sum()),
        detail="Set 3 time-bounded GRU (code/phase2/gru_stage.py line 16, code/gru_comparator.py lines 28-34 and 40-58): imputation = column medians and scaling = mean/SD of the imputed matrix, both fitted on the TRAINING rooms only; sequences of at most 24 real hours (SEQ_H = 24, at least 3 windows, span asserted); lo/hi = min and max over 5 seeds; scored on its own anchors, not the 4,198 windows",
        source="code/phase2/gru_stage.py",
    )
md_ = rd("model_family", "missing_data.csv")
for (subset, model), q in md_.groupby(["subset", "model"], sort=False):
    for _, r in q.iterrows():
        for m in ("MAE", "RMSE", "R2", "n"):
            add(
                "S33_missing_data",
                m,
                r.fold,
                r[m],
                variant=f"{model} | {subset}",
                n=r.n,
                source=src,
            )
    for m in ("MAE", "RMSE", "R2"):
        add(
            "S33_missing_data",
            m,
            "macro",
            q[m].mean(),
            variant=f"{model} | {subset}",
            n=q.n.sum(),
            detail="macro = mean over the three held-out rooms; n = windows summed over rooms",
            source=src,
        )
src = "code/phase2/et_fusion_stage.py"
em = rd("et_fusion", "et_fusion_metrics.csv")
for (lrn, fsn), q in em.groupby(["learner", "feature_set"], sort=False):
    det = (
        f"LORO, primary pipeline, 6 h evaluation windows; {q.n_features.iloc[0] if q.n_features.nunique() == 1 else 'columns vary by fold'} feature columns"
        + (
            "; median imputation fitted on training rooms only"
            if lrn == "ExtraTrees"
            else "; native NaN handling"
        )
    )
    for _, r in q.iterrows():
        for m in ("MAE", "RMSE", "R2", "n"):
            add(
                "S34_fusion_by_learner",
                m,
                r.fold,
                r[m],
                variant=f"{lrn} | {fsn}",
                n=r.n,
                detail=det,
                source=src,
            )
    for m in ("MAE", "RMSE", "R2"):
        add(
            "S34_fusion_by_learner",
            m,
            "macro",
            q[m].mean(),
            variant=f"{lrn} | {fsn}",
            n=q.n.sum(),
            detail=det + "; macro = mean over the three held-out rooms",
            source=src,
        )
for _, r in rd("et_fusion", "et_fusion_paired.csv").iterrows():
    add(
        "S35_fusion_paired_by_learner",
        "MAE difference (days)",
        r.heldout,
        r.MAE_diff,
        variant=r.contrast,
        lo=r.ci_lo,
        hi=r.ci_hi,
        n=r.n_pairs,
        detail=f"n_days={r.n_days}; 95% room-day block bootstrap (2,000 resamples, days within rooms, seed 0); negative = first model has the lower error; conditional on this cohort and these rooms",
        source=src,
    )
src = "code/phase2/body_size_stage.py"
for _, r in rd("body_size", "body_size_metrics.csv").iterrows():
    add("S36_body_size", r.metric, r.room, r.value, source=src)
src = "code/attribution.py"
for _, r in rd("attribution", "attribution_by_age_band.csv").iterrows():
    add(
        "S37_attribution_abs",
        "mean |SHAP| (days)",
        r.age_band,
        r.mean_abs_shap_days,
        variant=f"primary | {r.block}",
        n=r.n_rooms,
        detail="mean over held-out rooms of the per-room mean |SHAP|",
        source=src,
    )
src = "code/phase2/build_results_sheet.py (from code/analysis/clustering_epochs.py curves)"
cc = rd("clustering", "set4_clustering_curves.csv")
for room, g in cc.groupby("room"):
    s_ = g.sort_values("age_day").set_index("age_day").early_fraction_smoothed
    below = (s_[s_.index > 35] < 0.5).astype(int)
    day = next(
        (
            a
            for a in below.index
            if all(((a + k) in below.index and below[a + k] == 1) for k in range(3))
        ),
        np.nan,
    )
    add(
        "S38_clustering_persistence",
        "persistence criterion, 3 consecutive days (day)",
        room,
        day,
        detail="first age > 35 d from which the smoothed early-cluster share is below 0.5 for 3 consecutive days",
        source=src,
    )
src = "code/phase2/baselines_stage.py"
bl = rd("baselines", "baselines.csv")
for nm, q in bl.groupby("baseline", sort=False):
    for _, r in q.iterrows():
        for m in ("MAE", "RMSE", "R2", "n"):
            add(
                "S39_baselines",
                m,
                r.fold,
                r[m],
                variant=nm,
                n=r.n,
                detail="LORO, primary pipeline, 6 h evaluation windows; the isotonic baseline is scored on windows with the size feature",
                source=src,
            )
    for m in ("MAE", "RMSE", "R2"):
        add(
            "S39_baselines",
            m,
            "macro",
            q[m].mean(),
            variant=nm,
            n=q.n.sum(),
            detail="macro = mean over the three held-out rooms",
            source=src,
        )
src = "code/phase2/fold_manifest_stage.py"
fm = rd("fold_manifest", "fold_manifests.csv")
FM_METRICS = [
    "heldout_lo_age_d",
    "heldout_hi_age_d",
    "n_test",
    "n_train",
    "n_removed_by_embargo",
    "n_removed_by_QC_hours_heldout",
    "n_removed_by_QC_hours_train",
    "train_windows_reading_heldout_or_embargo_obs",
]
for _, r in fm.iterrows():
    fold = f"block {int(r.fold_id) + 1}" if r.protocol == "AGEBLOCK" else str(r.fold_id)
    for m in FM_METRICS:
        if pd.notna(r[m]):
            variant = f"{r.protocol} | {r.purge} purge, {r.embargo_h:g} h"
            add("S40_fold_manifest", m, fold, r[m], variant=variant, source=src)

src = "code/phase2/build_results_sheet.py (from core_primary/preds_ageblock_primary.csv)"
ap_ = pd.read_csv(RUN / "core_primary/preds_ageblock_primary.csv")
ap_ = ap_[ap_.model == "fused"]
order = list(dict.fromkeys(ap_.block))
n_cc = sum(
    r["value"]
    for r in ROWS
    if r["section"] == "S03_loro"
    and r["metric"] == "n"
    and r["variant"] == "primary fused"
    and r["room_or_fold"] in ROOMS
)
fm12 = fm[(fm.protocol == "AGEBLOCK") & (fm.purge == "window") & (fm.embargo_h == 12)]
assert len(order) == 5 and len(fm12) == 5
assert int(fm12.n_test.sum()) == len(ap_) == int(n_cc), "age-block windows do not sum to the n"


def block_metrics(d):
    e = d.pred - d.y
    sst = ((d.y - d.y.mean()) ** 2).sum()
    return dict(
        MAE=e.abs().mean(), RMSE=float(np.sqrt((e**2).mean())), R2=1 - (e**2).sum() / sst, n=len(d)
    )


per_block = [block_metrics(ap_[ap_.block == b]) for b in order]
detail = "12 h window purge, primary fused"
for i, bm in enumerate(per_block):
    for m in ("MAE", "RMSE", "R2", "n"):
        add(
            "S41_ageblock_aggregation",
            m,
            f"block {i + 1}",
            bm[m],
            variant="per block",
            n=bm["n"],
            detail=detail,
            source=src,
        )
for label, blocks in (("all 5 blocks", slice(0, 5)), ("excluding block 1", slice(1, 5))):
    pooled = block_metrics(ap_[ap_.block.isin(order[blocks])])
    n_ = sum(b["n"] for b in per_block[blocks])
    for m in ("MAE", "RMSE", "R2"):
        mean_ = float(np.mean([b[m] for b in per_block[blocks]]))
        add(
            "S41_ageblock_aggregation",
            f"{m}, mean of block values",
            "mean of blocks",
            mean_,
            variant=label,
            n=n_,
            detail="unweighted mean of the per-block metrics",
            source=src,
        )
        add(
            "S41_ageblock_aggregation",
            f"{m}, pooled over all held-out windows",
            "pooled",
            pooled[m],
            variant=label,
            n=pooled["n"],
            detail="one metric over all held-out windows of the blocks (R2 against their mean age)",
            source=src,
        )

src = "code/detection.py"
db = rd("detection", "set2_detection_by_room.csv")
for _, r in db.iterrows():
    v = f"{r['mode']} | {r.phase} | k={r.k_days} d | rule={r.rule}"
    note = f"room-week block bootstrap, {int(r.n_boot)} resamples, seed 0 (shared stream); target specificity {r.target_specificity}"
    for m, col in (
        ("positive windows", "n_pos_windows"),
        ("negative windows", "n_neg_windows"),
        ("room-week blocks", "n_weeks"),
        ("bootstrap resamples", "n_boot"),
    ):
        add("S42_detection_counts", m, r.test_room, r[col], variant=v, source=src)
    sp = r.achieved_specificity
    add(
        "S42_detection_counts",
        "false-positive rate",
        r.test_room,
        1 - sp,
        variant=v,
        n=r.n_neg_windows,
        detail="1 - achieved specificity on the held-out room's negative windows",
        source=src,
    )
    for tag, label in (("frozen", "fixed-calibration"), ("incl_cal", "calibration-inclusive")):
        add(
            "S42_detection_counts",
            f"false-positive rate, {label} interval",
            r.test_room,
            1 - sp,
            variant=v,
            lo=1 - r[f"achieved_specificity_ci_hi_{tag}"],
            hi=1 - r[f"achieved_specificity_ci_lo_{tag}"],
            n=r.n_neg_windows,
            detail=note,
            source=src,
        )
KS_TABLE = (5, 10, 15, 20)
wsc = rd("detection", "set2_window_scores.csv")
wsc = wsc[(wsc.rule == "same_hour") & (wsc["mode"] == "all_sensor") & wsc.k.isin(KS_TABLE)]
cnt = wsc.groupby(["phase", "k", "room", "week", "class"]).size().unstack("class", fill_value=0)
for (phase, k, room, week), r in cnt.iterrows():
    v = f"all_sensor | {phase} | k={k} d | rule=same_hour"
    add(
        "S43_detection_block_counts",
        "positive windows",
        f"{room} | {week}",
        r["pos"],
        variant=v,
        source=src,
    )
    add(
        "S43_detection_block_counts",
        "negative windows",
        f"{room} | {week}",
        r["neg"],
        variant=v,
        source=src,
    )
sb = rd("detection", "set2_sparse_blocks.csv")
sb = sb[(sb.rule == "same_hour") & sb.k_days.isin(KS_TABLE)]
for _, r in sb.iterrows():
    v = f"{r['mode']} | {r.phase} | k={r.k_days} d | rule={r.rule}"
    for m, col in (
        ("positive blocks", "n_blocks_pos"),
        ("negative blocks", "n_blocks_neg"),
        ("minimum blocks required (each class)", "min_required"),
        ("estimable", "estimable"),
    ):
        add(
            "S44_detection_nonoverlapping",
            m,
            r.test_room,
            float(r[col]),
            variant=v,
            detail=f"blocks per room (pos): {r.n_blocks_per_room}",
            source=src,
        )

src = "code/phase2/valid_windows_stage.py"
vw = rd("valid_windows", "valid_windows.csv")
defn = "complete-case anchors; a window is usable when more than half of its window-mean features are present, which needs at least 3 observed hours in the trailing 6 h"
for room, g in list(vw.groupby("room")) + [("all rooms", vw)]:
    n_obs = g.n_obs_hours
    for m, v in (
        ("evaluated windows", len(g)),
        ("windows with usable temporal features", int(g.valid.sum())),
        ("windows without usable temporal features", int((~g.valid).sum())),
        ("windows with 1 observed hour in the trailing 6 h", int((n_obs == 1).sum())),
        ("windows with 2 observed hours in the trailing 6 h", int((n_obs == 2).sum())),
        ("share of windows without usable temporal features", float((~g.valid).mean())),
    ):
        add("S45_window_validity", m, room, v, n=len(g), detail=defn, source=src)

src = "code/phase2/qc_timing_stage.py"
qf = rd("qc_timing", "qc_timing_folds.csv")
ORIG_QC = "original (QC built once on the full series)"
s41 = {
    r["room_or_fold"]: r["value"]
    for r in ROWS
    if r["section"] == "S41_ageblock_aggregation"
    and r["metric"] == "MAE"
    and r["variant"] == "per block"
}
for _, r in qf[qf.variant == ORIG_QC].iterrows():
    assert (
        abs(r.MAE - s41[f"block {int(r.fold)}"]) < 1e-9
    ), "original-QC rerun does not reproduce the primary age-block MAE"
for _, r in qf.iterrows():
    for m in ("MAE", "RMSE", "R2", "n", "excluded_total", "excluded_train", "excluded_heldout"):
        label = {
            "excluded_total": "QC-excluded complete-case hours",
            "excluded_train": "QC-excluded complete-case hours, training side",
            "excluded_heldout": "QC-excluded complete-case hours, withheld block",
        }.get(m, m)
        add(
            "S46_qc_timing",
            label,
            f"block {int(r.fold)}",
            r[m],
            variant=r.variant,
            n=r.n,
            detail="age-block evaluation, 12 h window purge, primary fused model",
            source=src,
        )
for v, g in qf.groupby("variant", sort=False):
    for m in ("MAE", "RMSE", "R2"):
        add(
            "S46_qc_timing",
            f"{m}, mean of block values",
            "mean of blocks",
            float(g[m].mean()),
            variant=v,
            n=int(g.n.sum()),
            detail="unweighted mean over the five blocks",
            source=src,
        )
    add(
        "S46_qc_timing",
        "QC-excluded complete-case hours (per fold, mean)",
        "mean of blocks",
        float(g.excluded_total.mean()),
        variant=v,
        source=src,
    )
for _, r in rd("qc_timing", "qc_history_leak.csv").iterrows():
    add(
        "S46_qc_timing",
        r.metric,
        f"block {int(r.fold)}",
        r.value,
        variant="age-block folds, 12 h embargo (+-0.5 d of days excluded)",
        source=src,
    )
for _, r in rd("qc_timing", "qc_retrospective.csv").iterrows():
    add(
        "S46_qc_timing", r.metric, "all rooms", r.value, variant="day-level QC decision", source=src
    )

src = "code/phase2/reconcile_stage.py"
for _, r in rd("reconcile", "reconcile.csv").iterrows():
    add("S47_reconciliation", r.metric, r.room, r.value, variant=r.fold, source=src)
src = "code/phase2/build_results_sheet.py (from the clustering curves)"
for tag, stage in (("primary", "clustering"), ("Step 0 settings", "clustering_step0")):
    curves = rd(stage, "set4_clustering_curves.csv")
    for room, g in curves.groupby("room"):
        series = g.sort_values("age_day").set_index("age_day").early_fraction_smoothed
        below = (series[series.index > 35] < 0.5).astype(int)
        crossing = next((a for a in below.index if below[a]), np.nan)
        persist = next(
            (
                a
                for a in below.index
                if all(((a + k) in below.index and below[a + k] == 1) for k in range(3))
            ),
            np.nan,
        )
        add(
            "S47_reconciliation",
            "50% crossing: first age > 35 d with the smoothed early-cluster share < 0.5 (day)",
            room,
            crossing,
            variant=tag,
            source=src,
        )
        add(
            "S47_reconciliation",
            "persistence: first age > 35 d from which the smoothed share stays < 0.5 for 3 consecutive days (day)",
            room,
            persist,
            variant=tag,
            source=src,
        )
        for day in range(60, 81):
            if day in series.index:
                add(
                    "S47_reconciliation",
                    f"smoothed early-cluster share at age {day} d",
                    room,
                    series[day],
                    variant=tag,
                    source=src,
                )
src = "code/phase2/build_results_sheet.py (S32, S30, S39)"
base = next(
    r["value"]
    for r in ROWS
    if r["section"] == "S39_baselines"
    and r["metric"] == "MAE"
    and r["room_or_fold"] == "macro"
    and r["variant"] == "median body height (isotonic)"
)
for r in list(ROWS):
    if (
        r["section"] in ("S32_model_family", "S30_loro_feature_sets")
        and r["metric"] == "MAE"
        and r["room_or_fold"] == "macro"
    ):
        add(
            "S47_reconciliation",
            "MAE advantage over the median-body-height baseline (days; positive = lower error)",
            "macro",
            base - r["value"],
            variant=r["variant"],
            detail=f"baseline MAE {base:.2f} d is scored on the windows that have the size feature",
            source=src,
        )

src = "code/phase2/build_results_sheet.py (primary predictions restricted to the windows flagged usable in valid_windows.csv)"
flags = vw[["room", "hour", "valid"]].assign(hour=lambda d: pd.to_datetime(d.hour))


def usable(preds):
    m = preds.assign(hour=pd.to_datetime(preds.hour)).merge(flags, on=["room", "hour"], how="left")
    assert m.valid.notna().all()
    return m[m.valid.astype(bool)]


ap_u = usable(ap_)
per_u = [block_metrics(ap_u[ap_u.block == b]) for b in order]
for i, bm in enumerate(per_u):
    for m in ("MAE", "RMSE", "R2", "n"):
        add(
            "S48_usable_window_metrics",
            m,
            f"block {i + 1}",
            bm[m],
            variant="age-block, usable windows",
            n=bm["n"],
            detail="12 h window purge, primary fused model",
            source=src,
        )
for m in ("MAE", "RMSE", "R2"):
    add(
        "S48_usable_window_metrics",
        f"{m}, mean of block values",
        "mean of blocks",
        float(np.mean([b[m] for b in per_u])),
        variant="age-block, usable windows",
        n=sum(b["n"] for b in per_u),
        detail="unweighted mean of the five per-block metrics",
        source=src,
    )
    add(
        "S48_usable_window_metrics",
        f"{m}, pooled over all usable windows",
        "pooled",
        block_metrics(ap_u)[m],
        variant="age-block, usable windows",
        n=len(ap_u),
        source=src,
    )
lo_ = pd.read_csv(RUN / "core_primary/preds_loro_primary.csv")
lo_u = usable(lo_[lo_.model == "fused"])
per_room = {r: block_metrics(lo_u[lo_u.room == r]) for r in ROOMS}
for r, bm in per_room.items():
    for m in ("MAE", "RMSE", "R2", "n"):
        add(
            "S48_usable_window_metrics",
            m,
            r,
            bm[m],
            variant="leave-one-room-out, usable windows",
            n=bm["n"],
            detail="primary fused model",
            source=src,
        )
for m in ("MAE", "RMSE", "R2"):
    add(
        "S48_usable_window_metrics",
        m,
        "macro",
        float(np.mean([b[m] for b in per_room.values()])),
        variant="leave-one-room-out, usable windows",
        n=sum(b["n"] for b in per_room.values()),
        detail="mean over the three held-out rooms",
        source=src,
    )

src = "code/phase2/build_results_sheet.py"
for mdl, label in (("morph_only", "morphometry-only"), ("video_all", "video-only")):
    pu = usable(lo_[lo_.model == mdl])
    pr = {r: block_metrics(pu[pu.room == r]) for r in ROOMS}
    for r, bm in pr.items():
        for m in ("MAE", "RMSE", "R2", "n"):
            add(
                "S49_matched_usable_windows",
                m,
                r,
                bm[m],
                variant=label,
                n=bm["n"],
                detail="LORO, windows with usable temporal features (the windows the median-body-height baseline is scored on)",
                source=src,
            )
    for m in ("MAE", "RMSE", "R2"):
        add(
            "S49_matched_usable_windows",
            m,
            "macro",
            float(np.mean([b[m] for b in pr.values()])),
            variant=label,
            n=sum(b["n"] for b in pr.values()),
            detail="mean over the three held-out rooms",
            source=src,
        )

S = pd.DataFrame(ROWS)
for c in ("value", "ci_lo", "ci_hi"):
    S[c] = S[c].astype(float)
S.to_csv(OUT, index=False, float_format="%.15g")
print("results sheet rows:", len(S), "sections:", S.section.nunique())
try:
    with pd.ExcelWriter(OUT.with_suffix(".xlsx"), engine="openpyxl") as xw:
        S.to_excel(xw, sheet_name="all_results", index=False)
        for sec, g in S.groupby("section", sort=True):
            g.drop(columns=["section"]).reset_index(drop=True).to_excel(
                xw, sheet_name=sec[:31], index=False
            )
except ImportError:  # fallback: stdlib writer (no spreadsheet package installed)
    from xlsx_min import write_xlsx

    write_xlsx(
        OUT.with_suffix(".xlsx"),
        {
            sec: g.drop(columns=["section"]).reset_index(drop=True)
            for sec, g in S.groupby("section", sort=True)
        },
    )
