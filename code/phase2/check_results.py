"""Consistency checks of the results sheet and tables against earlier outputs and the regression tests."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN, SHEET, TABLES = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "results" / "reference"
S = pd.read_csv(SHEET)
S["variant"] = S.variant.fillna("")
res = []
diffs = []


def rec(check, ok, detail):
    res.append(dict(check=check, result="PASS" if ok else "FAIL", detail=detail))
    print(("PASS " if ok else "FAIL ") + check + ": " + detail, flush=True)


def compare(name, pairs):
    """pairs: DataFrame with columns key, sheet, ref."""
    p = pairs.dropna(subset=["sheet", "ref"])
    d = (p.sheet - p.ref).abs()
    bad = p[d > 1e-10]
    for _, r in bad.iterrows():
        diffs.append(
            dict(
                section=name,
                key=r.key,
                sheet=r.sheet,
                set_output=r.ref,
                abs_diff=abs(r.sheet - r.ref),
            )
        )
    miss = len(pairs) - len(p)
    rec(
        f"match Set output: {name}",
        len(bad) == 0 and len(p) > 0,
        f"{len(p)} numbers compared, max |diff| = {d.max() if len(d) else float('nan'):.2e}, {len(bad)} differ by > 1e-10, {miss} not comparable",
    )


def get(section, metric, variant, room):
    d = S[
        (S.section == section)
        & (S.metric == metric)
        & (S.variant == variant)
        & (S.room_or_fold == room)
    ]
    return float(d.value.iloc[0]) if len(d) else np.nan


ROOMS = ["room2", "room6", "room7"]
st4 = pd.read_csv(APP / "step4/step4_all_runs_long.csv")
st4["v"] = np.where(
    st4.audio_ref == "first_week", "V2", np.where(st4.model == "morph_body_only", "V3", "primary")
)
VN = {
    ("primary", "fused"): "primary fused",
    ("primary", "morph_only"): "primary morphometry-only",
    ("V3", "morph_body_only"): "V3 body-size only",
    ("V2", "fused"): "V2 fused (audio first-week reference)",
}
rows = []
for (v, m), name in VN.items():
    q = st4[(st4.v == v) & (st4.model == m) & (st4.analysis == "LORO")]
    for _, r in q.iterrows():
        if r.metric in ("MAE", "RMSE", "R2", "n"):
            rows.append(
                dict(
                    key=f"{name}|{r.metric}|{r.fold}",
                    sheet=get("S03_loro", r.metric, name, r.fold),
                    ref=r.value,
                )
            )
compare("LORO (Step 4)", pd.DataFrame(rows))
for sec, pref in (("S05_ageblock", "AGEBLOCK"), ("S06_season", "SEASON")):
    rows = []
    for (v, m), name in VN.items():
        q = st4[(st4.v == v) & (st4.model == m) & st4.analysis.str.startswith(pref)]
        for _, r in q.iterrows():
            d = S[
                (S.section == sec)
                & (S.metric == r.metric)
                & (S.variant == name)
                & (S.room_or_fold == r.fold)
                & (S.detail == r.analysis)
            ]
            rows.append(
                dict(
                    key=f"{name}|{r.analysis}|{r.fold}|{r.metric}",
                    sheet=float(d.value.iloc[0]) if len(d) else np.nan,
                    ref=r.value,
                )
            )
    compare(f"{pref.lower()} (Step 4)", pd.DataFrame(rows))
pf = pd.read_csv(APP / "set4/set4_paired_fusion.csv")
NM = {"fused - morphometry-only": "fused - morphometry-only"}
rows = [
    dict(
        key=f"{r.contrast}|{r.heldout}|{k}",
        sheet=float(
            S[
                (S.section == "S07_paired_fusion")
                & (S.variant == r.contrast)
                & (S.room_or_fold == r.heldout)
            ][c].iloc[0]
        ),
        ref=r[k],
    )
    for _, r in pf.iterrows()
    for k, c in (("MAE_diff", "value"), ("ci_lo", "ci_lo"), ("ci_hi", "ci_hi"))
]
compare("paired fusion (Set 4)", pd.DataFrame(rows))
D_ = pd.read_csv(APP / "set2e/set2_detection_by_room.csv")
rows = []
for _, r in D_.iterrows():
    v = f"{r['mode']} | {r.phase} | k={int(r.k_days)} d | rule={r.rule}"
    for m, col, lo, hi in (
        ("AUC", "AUC", "AUC_ci_lo_frozen", "AUC_ci_hi_frozen"),
        ("sensitivity", "sensitivity", "sensitivity_ci_lo_frozen", "sensitivity_ci_hi_frozen"),
        (
            "achieved specificity",
            "achieved_specificity",
            "achieved_specificity_ci_lo_frozen",
            "achieved_specificity_ci_hi_frozen",
        ),
    ):
        x = S[
            (S.section == "S09_detection")
            & (S.metric == m)
            & (S.variant == v)
            & (S.room_or_fold == r.test_room)
        ].iloc[0]
        rows += [
            dict(key=f"{v}|{r.test_room}|{m}", sheet=x.value, ref=r[col]),
            dict(key=f"{v}|{r.test_room}|{m}|lo", sheet=x.ci_lo, ref=r[lo]),
            dict(key=f"{v}|{r.test_room}|{m}|hi", sheet=x.ci_hi, ref=r[hi]),
        ]
compare("detection by room (Set 2e)", pd.DataFrame(rows))
mf = pd.read_csv(APP / "set2e/set2_perturbation_manifest.csv")
rows = []
for _, r in mf.iterrows():
    v = f"{r['mode']} | {r.phase} | k={int(r.k_days)} d | rule={r.rule}"
    for m in (
        "n_eligible_anchors",
        "n_used_anchors",
        "n_dropped_no_source",
        "n_dropped_source_features_all_nan",
        "n_neg_windows",
        "n_pos_windows",
    ):
        x = S[
            (S.section == "S11_perturbation_manifest")
            & (S.metric == m)
            & (S.variant == v)
            & (S.room_or_fold == r.room)
        ]
        rows.append(
            dict(
                key=f"{v}|{r.room}|{m}",
                sheet=float(x.value.iloc[0]) if len(x) else np.nan,
                ref=r[m],
            )
        )
compare("perturbation manifest (Set 2e)", pd.DataFrame(rows))
sb = pd.read_csv(APP / "set2e/set2_sparse_blocks.csv")
rows = []
for _, r in sb.iterrows():
    v = f"{r['mode']} | {r.phase} | k={int(r.k_days)} d | rule={r.rule}"
    for m, col in (
        ("AUC", "AUC"),
        ("sensitivity", "sensitivity"),
        ("achieved specificity", "achieved_specificity"),
    ):
        x = S[
            (S.section == "S12_sparse_blocks")
            & (S.metric == m)
            & (S.variant == v)
            & (S.room_or_fold == r.test_room)
        ]
        rows.append(
            dict(
                key=f"{v}|{r.test_room}|{m}",
                sheet=float(x.value.iloc[0]) if len(x) else np.nan,
                ref=r[col],
            )
        )
compare("sparse blocks (Set 2e)", pd.DataFrame(rows))
sh = pd.read_csv(APP / "set3/shuffle/set3a_shuffled_label_summary.csv")
rows = []
for _, r in sh.iterrows():
    for m in ("MAE", "RMSE", "R2"):
        x = S[
            (S.section == "S14_shuffled_label")
            & (S.metric == f"{m} (shuffled mean, 95% permutation range)")
            & (S.room_or_fold == r.scope)
        ].iloc[0]
        rows += [
            dict(key=f"{r.scope}|{m}|mean", sheet=x.value, ref=r[f"{m}_perm_mean"]),
            dict(key=f"{r.scope}|{m}|lo", sheet=x.ci_lo, ref=r[f"{m}_perm_lo2.5"]),
            dict(key=f"{r.scope}|{m}|hi", sheet=x.ci_hi, ref=r[f"{m}_perm_hi97.5"]),
        ]
    rows += [
        dict(
            key=f"{r.scope}|p MAE",
            sheet=get("S14_shuffled_label", "p-value MAE", "", r.scope),
            ref=r.p_value_MAE,
        )
    ]
compare("shuffled-label control (Set 3a)", pd.DataFrame(rows))
sd = pd.read_csv(APP / "set3/gru/set3b_gru_seed_stability.csv")
rows = []
for _, r in sd.iterrows():
    rows.append(
        dict(
            key=f"{r.heldout}|seed{r.seed}|MAE",
            sheet=get("S15_gru", "MAE", f"GRU time-bounded | seed {int(r.seed)}", r.heldout),
            ref=r.MAE,
        )
    )
compare("GRU seeds 0-4 (Set 3b)", pd.DataFrame(rows))
g3 = pd.read_csv(APP / "set3/gru/set3b_gru_vs_hgb.csv")
rows = []
for _, r in g3[g3.heldout.isin(ROOMS) & g3.model.str.contains("HGB fused")].iterrows():
    nm = "HGB fused, same anchors" if "same anchors" in r.model else "HGB fused, all windows"
    rows.append(
        dict(key=f"{nm}|{r.heldout}", sheet=get("S15_gru", "MAE", nm, r.heldout), ref=r.MAE)
    )
compare("HGB comparators in the GRU table (Set 3b)", pd.DataFrame(rows))
rg = pd.read_csv(APP / "set3/availability/set3c_regime_results_by_room.csv")
rows = []
for _, r in rg.iterrows():
    for m in ("MAE", "RMSE", "R2"):
        rows.append(
            dict(
                key=f"{r.regime}|{r.model}|{r.heldout}|{m}",
                sheet=get("S18_regime_results", m, f"{r.regime} | {r.model}", r.heldout),
                ref=r[m],
            )
        )
compare("availability regimes (Set 3c)", pd.DataFrame(rows))
iv = pd.read_csv(APP / "set3/availability/set3c_regime_inventory.csv")
rows = [
    dict(
        key=f"{r.regime}|hours",
        sheet=get("S17_regime_inventory", "hours", r.regime, "all rooms"),
        ref=r.hours,
    )
    for _, r in iv.iterrows()
]
compare("regime inventory hours (Set 3c)", pd.DataFrame(rows))
cl = pd.read_csv(APP / "set4/classification/set4_classification_metrics.csv")
rows = []
for _, r in cl.iterrows():
    for c in cl.columns:
        if c != "scope" and pd.notna(r[c]):
            rows.append(
                dict(
                    key=f"{r.scope}|{c}", sheet=get("S19_classification", c, "", r.scope), ref=r[c]
                )
            )
compare("classification (Set 4)", pd.DataFrame(rows))
at = pd.read_csv(APP / "set4/attribution_primary/attribution_by_age_band.csv")
rows = [
    dict(
        key=f"{r.age_band}|{r.block}",
        sheet=get("S21_attribution", "share of total |SHAP|", f"primary | {r.block}", r.age_band),
        ref=r.share,
    )
    for _, r in at.iterrows()
]
compare("attribution (Set 4)", pd.DataFrame(rows))
dmf = pd.read_csv(APP / "set4/demeaning_primary/set4_demeaning_folds.csv")
rows = []
for _, r in dmf.iterrows():
    for c in ("R2", "MAE", "RMSE", "c0_drift_rho"):
        rows.append(
            dict(
                key=f"{r.variant}|{r.room}|{c}",
                sheet=get("S22_daily_demeaning", c, f"primary | {r.variant}", r.room),
                ref=r[c],
            )
        )
compare("daily demeaning (Set 4)", pd.DataFrame(rows))
cs = pd.read_csv(APP / "set4/clustering_primary/set4_clustering_summary.csv")
rows = []
for _, r in cs.iterrows():
    for c in (
        "n",
        "k_best",
        "silhouette_k2",
        "k2_boundary_day",
        "k2_ARI_vs_day70",
        "primary_transition_day",
    ):
        rows.append(
            dict(key=f"{r.room}|{c}", sheet=get("S23_clustering", c, "primary", r.room), ref=r[c])
        )
compare("clustering (Set 4)", pd.DataFrame(rows))
ri = pd.read_csv(APP / "step4/step4_room_id_before_after_V2.csv")
rows = [
    dict(
        key=f"{r.version}|{r.block}",
        sheet=get("S08_room_identification", "room identification accuracy", r.version, r.block),
        ref=r.room_id_accuracy,
    )
    for _, r in ri.iterrows()
]
compare("room identification (Step 4)", pd.DataFrame(rows))
s0 = pd.read_csv(APP / "step0_baseline.csv")
rows = [
    dict(
        key=f"{r.model}|{r.scope}|{r.metric}",
        sheet=get("S24_step0_reference", r.metric, f"Step 0 settings {r.model}", r.scope),
        ref=r.value,
    )
    for _, r in s0[(s0.analysis == "LORO")].iterrows()
]
compare("Step 0 LORO reference (step0_baseline.csv)", pd.DataFrame(rows))
s2 = pd.read_csv(APP / "step2_results_all.csv")
rows = []
CUMV = {
    "step0": (
        "Step 0 (centred filter, row slope, anchor purge)",
        "centred_retro",
        "anchor",
        12.0,
        "none",
    ),
    "step1": (
        "Step 1 (past-only filter, row slope, anchor purge)",
        "B_exclude",
        "anchor",
        12.0,
        "none",
    ),
    "step2": (
        "Step 2 (+ 12 h window purge, age-block and seasonal)",
        "B_exclude",
        "window",
        12.0,
        "window",
    ),
}
for tag, (nm, mode, pu, em, se) in CUMV.items():
    a = s2[
        (s2["mode"] == mode) & (s2.analysis == "AGEBLOCK") & (s2.purge == pu) & (s2.embargo_h == em)
    ]
    pooled = a[a.fold == "pooled fold-mean"].iloc[0]
    blocks = a[a.fold != "pooled fold-mean"]
    g = lambda m, fold, an: float(
        S[
            (S.section == "S25_cumulative_states")
            & (S.variant == f"{nm} | fused")
            & (S.detail == an)
            & (S.room_or_fold == fold)
            & (S.metric == m)
        ].value.iloc[0]
    )
    rows += [
        dict(
            key=f"{tag}|pooled MAE",
            sheet=g("MAE", "pooled fold-mean", "AGEBLOCK_12h"),
            ref=pooled.MAE,
        ),
        dict(
            key=f"{tag}|pooled MAE excl block 1",
            sheet=g("MAE", "pooled fold-mean excl. block 1", "AGEBLOCK_12h"),
            ref=blocks.MAE.iloc[1:].mean(),
        ),
    ]
    se_ = (
        s2[
            (s2["mode"] == mode)
            & (s2.analysis == "SEASON")
            & (s2.purge == se)
            & (s2.embargo_h == em)
        ]
        .set_index("fold")
        .MAE
    )
    rows += [
        dict(key=f"{tag}|season {k}", sheet=g("MAE", k, "SEASON_12h"), ref=se_[k])
        for k in ("train_summer_test_fall", "train_fall_test_summer")
    ]
    if tag == "step0":
        rows += [
            dict(
                key=f"step0|LORO {m_}|{mod}|{sc}",
                sheet=float(
                    S[
                        (S.section == "S25_cumulative_states")
                        & (S.variant == f"{nm} | {mod}")
                        & (S.detail == "LORO")
                        & (S.room_or_fold == sc)
                        & (S.metric == m_)
                    ].value.iloc[0]
                ),
                ref=float(
                    s0[
                        (s0.analysis == "LORO")
                        & (s0.model == mod)
                        & (s0.scope == sc)
                        & (s0.metric == m_)
                    ].value.iloc[0]
                ),
            )
            for mod in ("fused", "morph_only")
            for sc in ROOMS + ["macro"]
            for m_ in ("MAE", "RMSE", "R2")
        ]
compare("cumulative states (Steps 0-2 re-run) vs earlier Set outputs", pd.DataFrame(rows))
old_sheet = APP / "phase2/results_sheet_phase3_baseline.csv"
if old_sheet.exists():
    O = pd.read_csv(old_sheet)
    O["variant"] = O.variant.fillna("")
    NEW_SECTIONS = (
        "S26_negative_controls",
        "S27_mae_by_phase_ci",
        "S28_gap_recovery_curve",
        "S29_thermal_controls",
        "S30_loro_feature_sets",
        "S31_phase_mae_feature_sets",
        "S32_model_family",
        "S33_missing_data",
        "S34_fusion_by_learner",
        "S35_fusion_paired_by_learner",
        "S36_body_size",
        "S37_attribution_abs",
        "S38_clustering_persistence",
        "S39_baselines",
    )
    N = S[~S.section.isin(NEW_SECTIONS)].reset_index(drop=True)
    same = (
        O.shape == N.shape
        and list(O.columns) == list(N.columns)
        and all(bool(((O[k] == N[k]) | (O[k].isna() & N[k].isna())).all()) for k in O.columns)
    )
    rec(
        "only the new rows change the sheet (all earlier rows byte-identical in content)",
        same,
        f"earlier sheet {O.shape}, new sheet without S26-S39 {N.shape}; new rows = {int(S.section.isin(NEW_SECTIONS).sum())} (sections S26-S39)",
    )
iv_ = S.dropna(subset=["ci_lo", "ci_hi", "value"])
bad = iv_[(iv_.value < iv_.ci_lo - 1e-9) | (iv_.value > iv_.ci_hi + 1e-9)]
rec(
    "every interval contains its point estimate",
    len(bad) == 0,
    f"{len(iv_)} rows with an interval, {len(bad)} violations"
    + (
        ""
        if len(bad) == 0
        else ": "
        + "; ".join(f"{r.result_id} {r.metric} {r.room_or_fold}" for _, r in bad.head(8).iterrows())
    ),
)
n_loro = {r: get("S03_loro", "n", "primary fused", r) for r in ROOMS}
okn = all(
    get("S03_loro", "n", v, r) == n_loro[r]
    for v in (
        "primary morphometry-only",
        "V3 body-size only",
        "V2 fused (audio first-week reference)",
    )
    for r in ROOMS
)
rec("same n across fused / morphometry-only / V3 / V2 (LORO)", okn, str(n_loro))
cc_ = {
    r: S[
        (S.section == "S01_data_coverage")
        & (S.metric.str.startswith("complete-case hours, primary"))
        & (S.room_or_fold == r)
        & S.detail.str.contains("placement")
    ].value.iloc[0]
    for r in ROOMS
}
rec(
    "LORO n == complete-case hours in the coverage table",
    all(cc_[r] == n_loro[r] for r in ROOMS),
    f"coverage {cc_}",
)
cn = {r: get("S23_clustering", "n", "primary", r) for r in ROOMS}
rec("clustering n == LORO n", cn == n_loro, str(cn))
cn = {
    r: S[
        (S.section == "S22_daily_demeaning")
        & (S.metric == "R2")
        & (S.room_or_fold == r)
        & (S.variant == "primary | diurnal")
    ].n.iloc[0]
    for r in ROOMS
}
rec("daily-demeaning n == LORO n", cn == n_loro, str(cn))
sup = {
    r: sum(
        S[
            (S.section == "S19_classification")
            & (S.room_or_fold == r)
            & (S.metric.str.startswith("support_"))
        ].value
    )
    for r in ROOMS
}
rec("classification supports sum to LORO n", sup == n_loro, str(sup))
el = {
    r: S[
        (S.section == "S11_perturbation_manifest")
        & (S.metric == "n_eligible_anchors")
        & (S.room_or_fold == r)
        & (S.variant.str.startswith("all_sensor"))
        & S.variant.str.contains("k=5 d")
        & S.variant.str.contains("rule=same_hour")
    ].value.sum()
    for r in ROOMS
}
rec("detection eligible anchors (both regimes) == LORO n", el == n_loro, str(el))
sh_n = S[(S.section == "S14_shuffled_label") & S.metric.str.startswith("MAE (shuffled")].n.unique()
rec("shuffled-label: 100 permutations", list(sh_n) == [100], str(sh_n))
rec(
    "GRU: 5 seeds",
    S[(S.section == "S15_gru") & (S.variant == "GRU time-bounded (headline)")].n.min() == 5,
    "",
)
mt = (ROOT / "code/make_tables.py").read_text()
rec(
    "make_tables.py reads only the results sheet",
    len(re.findall(r"read_csv\(", mt)) == 1 and "SHEET" in mt,
    f"{len(re.findall('read_csv[(]', mt))} read_csv call(s)",
)
t = subprocess.run(
    [sys.executable, str(ROOT / "code/tests/test_regression.py")],
    capture_output=True,
    text=True,
    cwd=ROOT,
)
last = [l for l in t.stdout.splitlines() if "tests passed" in l]
rec("regression tests on the clean run", t.returncode == 0, last[-1] if last else t.stdout[-300:])
pd.DataFrame(res).to_csv(SHEET.parent / "phase2_checks.csv", index=False)
pd.DataFrame(diffs, columns=["section", "key", "sheet", "set_output", "abs_diff"]).to_csv(
    SHEET.parent / "phase2_set_mismatches.csv", index=False
)
(SHEET.parent / "phase2_checks.md").write_text(
    "| check | result | detail |\n|---|---|---|\n"
    + "\n".join(f"| {r['check']} | {r['result']} | {r['detail']} |" for r in res)
    + "\n"
)
sys.exit(0 if all(r["result"] == "PASS" for r in res) else 1)
