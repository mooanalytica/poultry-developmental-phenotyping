"""Build the manuscript tables from results/results_sheet.csv only.
Usage: python3 code/make_tables.py [SHEET_CSV] [OUT_DIR]
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHEET = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "results_sheet.csv"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
S = pd.read_csv(SHEET, keep_default_na=True)
S["variant"] = S.variant.fillna("")
S["detail"] = S.detail.fillna("")
S["text_value"] = S.text_value.fillna("")
ROOMS = ["room2", "room6", "room7"]
LIST = []
try:
    import docx as _docx
except ImportError:
    _docx = None


def f(x, d=2):
    return "" if pd.isna(x) else f"{x:.{d}f}"


def ci(v, lo, hi, d=2):
    return (
        ""
        if pd.isna(v)
        else (f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]" if not pd.isna(lo) else f"{v:.{d}f}")
    )


def esc(t):
    return re.sub(r"([_&%#$])", r"\\\1", str(t))


def save(name, df, caption):
    df = df.copy()
    df.to_csv(OUT / f"{name}.csv", index=False)
    cols = "l" * len(df.columns)
    tex = [
        "% " + caption,
        "\\begin{tabular}{" + cols + "}",
        "\\hline",
        " & ".join(esc(c) for c in df.columns) + " \\\\",
        "\\hline",
    ]
    tex += [
        " & ".join(esc(v) for v in r) + " \\\\"
        for r in df.fillna("").astype(str).itertuples(index=False)
    ] + ["\\hline", "\\end{tabular}"]
    (OUT / f"{name}.tex").write_text("\n".join(tex) + "\n")
    md = [
        "**" + caption + "**",
        "",
        "| " + " | ".join(df.columns) + " |",
        "|" + "|".join("---" for _ in df.columns) + "|",
    ] + ["| " + " | ".join(r) + " |" for r in df.fillna("").astype(str).itertuples(index=False)]
    (OUT / f"{name}.md").write_text("\n".join(md) + "\n")
    LIST.append((name, caption, len(df)))
    if _docx is not None:
        doc = _docx.Document()
        doc.add_paragraph(caption)
        tb = doc.add_table(rows=1, cols=len(df.columns))
        tb.style = "Table Grid"
        for j, c_ in enumerate(df.columns):
            tb.rows[0].cells[j].text = str(c_)
        for r_ in df.fillna("").astype(str).itertuples(index=False):
            cells = tb.add_row().cells
            for j, v_ in enumerate(r_):
                cells[j].text = v_
        doc.save(OUT / f"{name}.docx")


def sel(section, metric=None, variant=None, room=None, contains=None):
    d = S[S.section == section]
    if metric is not None:
        d = d[d.metric == metric]
    if variant is not None:
        d = d[d.variant == variant]
    if room is not None:
        d = d[d.room_or_fold == room]
    if contains is not None:
        d = d[d.variant.str.contains(contains, regex=False)]
    return d


def val(section, metric, variant, room):
    d = sel(section, metric, variant, room)
    return float(d.value.iloc[0]) if len(d) else np.nan


c = S[S.section == "S01_data_coverage"].copy()
t = pd.DataFrame(
    {
        "room": c.room_or_fold,
        "quantity": c.metric,
        "denominator": c.detail.str.replace("denominator: ", "", regex=False),
        "count": c.value,
        "denominator_n": c.n,
        "percent": (100 * c.value / c.n),
    }
)
t["count"] = t["count"].map(lambda x: f"{x:,.0f}")
t["denominator_n"] = t["denominator_n"].map(lambda x: f"{x:,.0f}")
t["percent"] = t["percent"].map(lambda x: f(x, 1))
save(
    "table01_data_coverage",
    t,
    "Table 1. Coverage per room and modality, with the denominator stated for every row",
)
k = S[S.section == "S02_key_dates"]
save(
    "table01b_key_dates",
    pd.DataFrame(
        {
            "item": k.metric,
            "room": k.room_or_fold,
            "date_or_range": k.text_value,
            "age_days": k.value.map(lambda x: f(x, 2)),
            "note": k.detail,
        }
    ),
    "Table 1b. Key dates and reference periods",
)
V = [
    "primary fused",
    "primary morphometry-only",
    "V3 body-size only",
    "V2 fused (audio first-week reference)",
]
rows = []
for v in V:
    rows.append(
        {
            "model": v,
            **{f"MAE {r}": f(val("S03_loro", "MAE", v, r)) for r in ROOMS},
            "MAE macro": f(val("S03_loro", "MAE", v, "macro")),
            "RMSE macro": f(val("S03_loro", "RMSE", v, "macro")),
            "R2 macro": f(val("S03_loro", "R2", v, "macro"), 3),
            **{f"R2 {r}": f(val("S03_loro", "R2", v, r), 3) for r in ROOMS},
            "n (windows, all rooms)": f"{sum(val('S03_loro', 'n', v, r) for r in ROOMS):,.0f}",
        }
    )
save(
    "table02_loro",
    pd.DataFrame(rows),
    "Table 2. Leave-one-room-out performance (MAE and RMSE in days); V2 and V3 are labelled extensions",
)
rows = []
for v in V:
    row = {"model": v}
    for ph in ("growth (<=70 d)", "plateau (>70 d)"):
        for r in ROOMS + ["macro"]:
            row[f"MAE {ph} {r}"] = f(val("S04_loro_growth_plateau", f"MAE {ph}", v, r))
    rows.append(row)
save(
    "table03_growth_plateau",
    pd.DataFrame(rows),
    "Table 3. MAE by growth (<=70 d) and plateau (>70 d) phase, leave-one-room-out",
)
rows = []
for v in ["primary fused", "V3 body-size only", "V2 fused (audio first-week reference)"]:
    for an in ("AGEBLOCK_12h", "AGEBLOCK_24h"):
        d = S[(S.section == "S05_ageblock") & (S.variant == v) & (S.detail == an)]
        if d.empty:
            continue
        row = {"model": v, "embargo": an.split("_")[1]}
        for fold in d[d.metric == "MAE"].room_or_fold.unique():
            row[fold] = f(d[(d.metric == "MAE") & (d.room_or_fold == fold)].value.iloc[0])
        row["pooled R2"] = f(
            d[(d.metric == "R2") & (d.room_or_fold == "pooled fold-mean")].value.iloc[0], 3
        )
        rows.append(row)
save(
    "table04_ageblock",
    pd.DataFrame(rows).fillna(""),
    "Table 4. Age-block MAE (days) per block (block edges = quintiles of age of the complete-case windows); pooled fold-mean also given without block 1 (its relative-morphometry reference is empty)",
)
rows = []
for v in ["primary fused", "V3 body-size only", "V2 fused (audio first-week reference)"]:
    for an in ("SEASON_12h", "SEASON_24h"):
        d = S[(S.section == "S06_season") & (S.variant == v) & (S.detail == an)]
        for fold in d.room_or_fold.unique():
            g = d[d.room_or_fold == fold]
            rows.append(
                {
                    "model": v,
                    "embargo": an.split("_")[1],
                    "fold": fold,
                    "MAE": f(g[g.metric == "MAE"].value.iloc[0]),
                    "R2": f(g[g.metric == "R2"].value.iloc[0], 3),
                    "n_test": f"{g[g.metric == 'n'].value.iloc[0]:,.0f}",
                }
            )
save(
    "table05_season",
    pd.DataFrame(rows),
    "Table 5. Seasonal extrapolation (the held-out season is the test set; the summer-held-out fold has no relative morphometry features)",
)
p = S[S.section == "S07_paired_fusion"]
rows = [
    {
        "contrast": c_,
        **{
            r: ci(
                g[g.room_or_fold == r].value.iloc[0],
                g[g.room_or_fold == r].ci_lo.iloc[0],
                g[g.room_or_fold == r].ci_hi.iloc[0],
            )
            for r in ROOMS + ["macro"]
        },
        "n pairs (all rooms)": f"{g[g.room_or_fold != 'macro'].n.sum():,.0f}",
    }
    for c_, g in p.groupby("variant", sort=False)
]
save(
    "table06_paired_fusion",
    pd.DataFrame(rows),
    "Table 6. Paired MAE differences (days, first model minus second; same rows) with 95% room-day bootstrap intervals; conditional on this cohort and these rooms",
)
d = S[
    (S.section == "S09_detection")
    & S.variant.str.contains("rule=same_hour")
    & (S.room_or_fold == "all rooms")
    & S.variant.str.contains(r"k=(?:5|10|15|20) d", regex=True)
].copy()
d[["mode", "phase", "k", "rule"]] = d.variant.str.split(" \\| ", expand=True)
rows = []
for (mode, phase, kk), g in d.groupby(["mode", "phase", "k"], sort=False):
    gi = {m: g[g.metric == m].iloc[0] for m in ("AUC", "sensitivity", "achieved specificity")}
    rows.append(
        {
            "mode": mode,
            "regime": phase,
            "offset": kk.replace("k=", ""),
            "AUC [95% CI]": ci(gi["AUC"].value, gi["AUC"].ci_lo, gi["AUC"].ci_hi, 3),
            "sensitivity [95% CI]": ci(
                gi["sensitivity"].value, gi["sensitivity"].ci_lo, gi["sensitivity"].ci_hi, 3
            ),
            "achieved specificity [95% CI]": ci(
                gi["achieved specificity"].value,
                gi["achieved specificity"].ci_lo,
                gi["achieved specificity"].ci_hi,
                3,
            ),
            "positive windows": f"{gi['AUC'].n:,.0f}",
            "negative windows": re.search(r"n_neg_windows=(\d+)", gi["AUC"].detail).group(1),
            "weeks": re.search(r"n_weeks=(\d+)", gi["AUC"].detail).group(1),
        }
    )
save(
    "table08_detection",
    pd.DataFrame(rows),
    "Table 8. Detection of injected delays on the calibrated pipeline (in-fold calibration, target specificity 0.95; matched anchors, same-hour rule; trailing 7-day windows with >= 4 days of data). Intervals: 95% room-week bootstrap with the calibration held fixed; conditional on this cohort and these rooms, not new farms",
)
d = S[
    (S.section == "S09_detection")
    & S.variant.str.contains("rule=same_hour")
    & S.variant.str.contains("all_sensor")
    & S.variant.str.contains(r"k=(?:5|10|15|20) d", regex=True)
    & S.room_or_fold.isin(ROOMS)
].copy()
d[["mode", "phase", "k", "rule"]] = d.variant.str.split(" \\| ", expand=True)
rows = []
for (phase, kk), g in d.groupby(["phase", "k"], sort=False):
    row = {"regime": phase, "offset": kk.replace("k=", "")}
    for r in ROOMS:
        for m, lab in (("achieved specificity", "specificity"), ("sensitivity", "sensitivity")):
            x = g[(g.room_or_fold == r) & (g.metric == m)].iloc[0]
            row[f"{lab} {r}"] = f(x.value, 3)
        row[f"windows {r} (pos)"] = (
            f"{g[(g.room_or_fold == r) & (g.metric == 'AUC')].n.iloc[0]:,.0f}"
        )
    rows.append(row)
save(
    "table08b_per_room_specificity",
    pd.DataFrame(rows),
    "Table 8b. Achieved specificity and sensitivity per held-out room (all-sensor, same-hour rule; target specificity 0.95)",
)
m = S[
    (S.section == "S11_perturbation_manifest")
    & S.variant.str.contains("all_sensor")
    & S.variant.str.contains(r"k=(?:5|10|15|20) d", regex=True)
].copy()
m[["mode", "phase", "k", "rule"]] = m.variant.str.split(" \\| ", expand=True)
g = m.groupby(["rule", "phase", "k", "metric"]).value.sum().unstack("metric").reset_index()
save(
    "table08c_perturbation_manifest",
    g[
        [
            "rule",
            "phase",
            "k",
            "n_eligible_anchors",
            "n_used_anchors",
            "n_dropped_no_source",
            "n_dropped_source_features_all_nan",
            "n_pos_windows",
            "n_neg_windows",
        ]
    ].astype(
        {
            c: int
            for c in (
                "n_eligible_anchors",
                "n_used_anchors",
                "n_dropped_no_source",
                "n_dropped_source_features_all_nan",
                "n_pos_windows",
                "n_neg_windows",
            )
        }
    ),
    "Table 8c. Perturbation manifest summed over rooms (all-sensor): eligible, used and dropped anchors and window counts, both matching rules",
)
b = S[
    (S.section == "S12_sparse_blocks")
    & S.variant.str.contains("all_sensor")
    & (S.room_or_fold == "all rooms")
    & S.variant.str.contains(r"k=(?:5|10|15|20) d", regex=True)
].copy()
rows = []
for var, gg in b.groupby("variant", sort=False):
    x = gg.set_index("metric")
    rows.append(
        {
            "setting": var,
            "blocks (pos)": f"{x.n.iloc[0]:,.0f}",
            "detail": x.detail.iloc[0],
            "AUC": f(x.loc["AUC", "value"], 3),
            "sensitivity": f(x.loc["sensitivity", "value"], 3),
            "achieved specificity": f(x.loc["achieved specificity", "value"], 3),
            "estimable": (
                "yes" if "estimable=True" in x.detail.iloc[0] else "not estimable (<10 blocks)"
            ),
        }
    )
save(
    "table08d_sparse_blocks",
    pd.DataFrame(rows),
    "Table 8d. Non-overlapping 7-day blocks (4-of-7-day rule): block counts, AUC and sensitivity; growth cells are not estimable",
)
s = S[S.section == "S14_shuffled_label"]
rows = []
for r in ROOMS + ["macro"]:
    row = {"scope": r}
    for mm in ("MAE", "R2"):
        a = s[
            (s.room_or_fold == r) & (s.metric == f"{mm} (shuffled mean, 95% permutation range)")
        ].iloc[0]
        row[f"{mm} shuffled mean [2.5-97.5%]"] = ci(
            a.value, a.ci_lo, a.ci_hi, 3 if mm == "R2" else 2
        )
        row[f"{mm} primary"] = f(
            s[(s.room_or_fold == r) & (s.metric == f"{mm} (primary model)")].value.iloc[0],
            3 if mm == "R2" else 2,
        )
    row["p (MAE)"] = f(s[(s.room_or_fold == r) & (s.metric == "p-value MAE")].value.iloc[0], 4)
    row["p (R2)"] = f(s[(s.room_or_fold == r) & (s.metric == "p-value R2")].value.iloc[0], 4)
    row["permutations"] = f"{a.n:,.0f}"
    rows.append(row)
save(
    "table09_shuffled_label",
    pd.DataFrame(rows),
    "Table 9. Shuffled-label control: age labels of the training rooms permuted as whole room-day blocks (100 permutations), evaluated against the true ages of the held-out room",
)
g = S[S.section == "S15_gru"]
rows = []
for v in [
    "GRU time-bounded (headline)",
    "HGB fused, same anchors",
    "HGB fused, all windows",
    "GRU Revision 3 construction (seed 0)",
]:
    row = {"model": v}
    for m in ("MAE", "RMSE", "R2"):
        key = f"{m} macro, mean over 5 seeds" if v.startswith("GRU time") else f"{m} macro"
        x = g[(g.variant == v) & (g.metric == key)].iloc[0]
        row[f"{m} macro"] = (
            ci(x.value, x.ci_lo, x.ci_hi, 3 if m == "R2" else 2)
            if v.startswith("GRU time")
            else f(x.value, 3 if m == "R2" else 2)
        )
    rows.append(row)
save(
    "table10_gru_vs_hgb",
    pd.DataFrame(rows),
    "Table 10. GRU (time-bounded 24 h sequences, training-room imputation) against HGB fused; GRU headline = mean over 5 seeds with the range in brackets (seed 0 pre-declared)",
)
rows = []
for sd in range(5):
    row = {"seed": sd}
    for r in ROOMS + ["macro"]:
        d_ = g[
            (g.metric == ("MAE macro" if r == "macro" else "MAE"))
            & (g.room_or_fold == r)
            & (g.variant == f"GRU time-bounded | seed {sd}")
        ]
        row[f"MAE {r}"] = f(d_.value.iloc[0])
    rows.append(row)
save("table10b_gru_seeds", pd.DataFrame(rows), "Table 10b. Time-bounded GRU, MAE (days) per seed")
sp = S[S.section == "S16_gru_legacy_spans"]
save(
    "table10c_gru_legacy_spans",
    sp.pivot_table(index="metric", columns="room_or_fold", values="value", sort=False)
    .round(3)
    .reset_index(),
    "Table 10c. Revision 3 GRU sequences: real-time span of '24-row' sequences (hours)",
)
inv = S[S.section == "S17_regime_inventory"]
res = S[S.section == "S18_regime_results"]
rows = []
for reg in (
    inv[inv.metric == "hours"]
    .query("room_or_fold == 'all rooms'")
    .sort_values("value", ascending=False)
    .variant
):
    h = inv[
        (inv.variant == reg) & (inv.metric == "hours") & (inv.room_or_fold == "all rooms")
    ].iloc[0]
    a = (
        inv[(inv.variant == reg) & (inv.metric == "anchors") & inv.room_or_fold.isin(ROOMS)]
        .set_index("room_or_fold")
        .value
    )
    scored = res[res.variant.str.startswith(reg + " | ")].room_or_fold.isin(ROOMS).any()
    row = {
        "regime (blocks present)": reg,
        "hours": f"{h.value:,.0f}",
        "rooms and age coverage": h.detail,
        **{f"anchors {r}": f"{a.get(r, 0):,.0f}" for r in ROOMS},
        "scored (>= 30 anchors in a held-out room)": "yes" if scored else "no: unscored",
    }
    for mod in (
        "A: primary complete-case fused",
        "B: fused trained on all hours",
        "C: regime-specific (present blocks only)",
    ):
        x = res[(res.variant == f"{reg} | {mod}") & (res.metric == "MAE_macro")]
        row[f"MAE macro {mod[0]}"] = f(x.value.iloc[0]) if len(x) else ""
    rows.append(row)
save(
    "table11_availability_regimes",
    pd.DataFrame(rows),
    "Table 11. Modality-availability regimes, named by the blocks present; models A (primary complete-case model applied unchanged), B (trained on all hours), C (regime-specific). Unscored regimes have fewer than 30 anchors per held-out room",
)
c = S[S.section == "S19_classification"]
rows = []
for sc in ROOMS + ["pooled (all held-out windows)", "macro (mean of rooms)"]:
    x = c[c.room_or_fold == sc].set_index("metric").value
    rows.append(
        {
            "scope": sc,
            **{
                k: f(x.get(k), 3)
                for k in ("accuracy", "balanced_accuracy", "macro_F1", "adjacent_accuracy", "qwk")
            },
            **{k.replace("_", " "): f(x.get(k), 3) for k in x.index if k.startswith("recall_")},
            **{
                k.replace("_", " "): f"{x.get(k):,.0f}" for k in x.index if k.startswith("support_")
            },
            "n": f"{x.get('n'):,.0f}",
        }
    )
save(
    "table12_classification",
    pd.DataFrame(rows),
    "Table 12. Stage classification (classes (0,60], (60,110], (110,200] d): per-class recall, supports, balanced accuracy and macro-F1 (leave-one-room-out)",
)
cm = S[S.section == "S20_confusion"]
rows = [
    {"scope": sc, "true -> predicted": m.replace("true ", ""), "windows": f"{v:,.0f}"}
    for sc, m, v in zip(cm.room_or_fold, cm.metric, cm.value)
]
save(
    "table12b_confusion",
    pd.DataFrame(rows),
    "Table 12b. Confusion-matrix counts per held-out room and pooled",
)
a = S[(S.section == "S21_attribution") & S.variant.str.startswith("primary")].copy()
a["block"] = a.variant.str.replace("primary | ", "", regex=False)
t = (
    a.pivot_table(index="room_or_fold", columns="block", values="value", sort=False)
    .round(3)
    .reset_index()
    .rename(columns={"room_or_fold": "age band"})
)
save(
    "table13_attribution",
    t,
    "Table 13. Share of total |SHAP| by block and age band (mean of the three rooms)",
)
dm = S[(S.section == "S22_daily_demeaning") & S.variant.str.startswith("primary")]
rows = []
for var in dm.variant.unique():
    x = dm[(dm.variant == var) & (dm.room_or_fold == "macro")].set_index("metric").value
    rows.append(
        {
            "variant": var.replace("primary | ", ""),
            "R2": f(x.get("R2"), 3),
            "MAE": f(x.get("MAE")),
            "C0 (median |rho| daily-mean feature vs age)": f(x.get("c0_drift_rho"), 3),
            **{
                f"morph share of (morph + bioacoustics) |SHAP|, {b}": f(
                    dm[
                        (dm.variant == var)
                        & (dm.room_or_fold == b)
                        & (dm.metric == "morphometry share of (morphometry + bioacoustics) |SHAP|")
                    ].value.iloc[0],
                    3,
                )
                for b in ("<45 d", ">140 d")
            },
        }
    )
save(
    "table14_daily_demeaning",
    pd.DataFrame(rows),
    "Table 14. Same-day room demeaning test. NOTE: the shares here are the morphometry share of (morphometry + bioacoustics) |SHAP| only (demeaning comparison definition), not the share of total used in Table 13",
)
cl = S[(S.section == "S23_clustering") & (S.variant == "primary")]
t = (
    cl.pivot_table(index="metric", columns="room_or_fold", values="value", sort=False)
    .round(3)
    .reset_index()
)
save(
    "table15_clustering",
    t,
    "Table 15. Unsupervised clustering per room (primary pipeline): crossover/transition day, k=2 boundary, silhouette, ARI vs day 70",
)
rows = []
for (sec, met), _ in S[
    S.variant.isin(["Step 0 settings", "primary"])
    & S.section.isin(["S21_attribution", "S23_clustering"])
].groupby(["section", "metric"]):
    pass


def pair(section, metric, room, vprefix_old, vprefix_new, label):
    o = S[
        (S.section == section)
        & (S.metric == metric)
        & (S.room_or_fold == room)
        & (S.variant == vprefix_old)
    ]
    n = S[
        (S.section == section)
        & (S.metric == metric)
        & (S.room_or_fold == room)
        & (S.variant == vprefix_new)
    ]
    return {
        "analysis": label,
        "scope": room,
        "old (Step 0 settings)": f(o.value.iloc[0], 3) if len(o) else "",
        "new (primary)": f(n.value.iloc[0], 3) if len(n) else "",
    }


rows += [
    pair(
        "S21_attribution",
        "share of total |SHAP|",
        b,
        f"Step 0 settings | {blk}",
        f"primary | {blk}",
        f"attribution share of total, {blk}",
    )
    for blk in ("morph", "audio_bio")
    for b in ("<45d", ">140d")
]
rows += [
    pair(
        "S23_clustering",
        "primary_transition_day",
        r,
        "Step 0 settings",
        "primary",
        "clustering transition day",
    )
    for r in ROOMS
]
rows += [
    pair(
        "S23_clustering",
        "k2_ARI_vs_day70",
        r,
        "Step 0 settings",
        "primary",
        "clustering ARI vs day 70",
    )
    for r in ROOMS
]
rows += [
    pair(
        "S22_daily_demeaning",
        m_,
        "macro",
        "Step 0 settings | diurnal",
        "primary | diurnal",
        f"daily demeaning {m_}",
    )
    for m_ in ("R2", "MAE")
]
rows += [
    pair(
        "S22_daily_demeaning",
        "morphometry share of (morphometry + bioacoustics) |SHAP|",
        b,
        "Step 0 settings | diurnal",
        "primary | diurnal",
        "demeaning: morph share of (morph + bioacoustics)",
    )
    for b in ("<45 d", ">140 d")
]
rows += [
    pair(
        "S24_step0_reference",
        "MAE",
        "macro",
        "Step 0 settings fused",
        "primary fused",
        "LORO fused MAE (Step 0 vs primary)",
    )
]
rows[-1]["new (primary)"] = f(val("S03_loro", "MAE", "primary fused", "macro"), 3)
rows += [
    dict(
        analysis="LORO morphometry-only MAE (Step 0 vs primary)",
        scope="macro",
        **{
            "old (Step 0 settings)": f(
                val("S24_step0_reference", "MAE", "Step 0 settings morph_only", "macro"), 3
            ),
            "new (primary)": f(val("S03_loro", "MAE", "primary morphometry-only", "macro"), 3),
        },
    )
]
save(
    "table16_old_vs_new",
    pd.DataFrame(rows),
    "Table 16. Earlier (Step 0 settings, same scripts) against the primary pipeline",
)
r_ = S[S.section == "S08_room_identification"]
save(
    "table17_room_identification",
    pd.DataFrame(
        {
            "version": r_.variant,
            "audio block": r_.room_or_fold,
            "features": r_.n.map(lambda x: f"{x:.0f}"),
            "room-identification accuracy": r_.value.map(lambda x: f(x, 3)),
            "detail": r_.detail,
        }
    ),
    "Table 17. Room identification from audio features (age-block cross-validation; chance 0.333), before and after the V2 first-week reference",
)
cu = S[S.section == "S25_cumulative_states"]


def cval(step, model, analysis, fold, metric):
    d_ = cu[
        (cu.variant == f"{step} | {model}")
        & (cu.detail == analysis)
        & (cu.room_or_fold == fold)
        & (cu.metric == metric)
    ]
    return float(d_.value.iloc[0]) if len(d_) else np.nan


rows = []
for step in [
    "Step 0 (centred filter, row slope, anchor purge)",
    "Step 1 (past-only filter, row slope, anchor purge)",
    "Step 2 (+ 12 h window purge, age-block and seasonal)",
]:
    for model in ("fused", "morph_only"):
        row = {
            "stage": step,
            "model": model,
            **{f"LORO MAE {r}": f(cval(step, model, "LORO", r, "MAE")) for r in ROOMS},
            "LORO MAE macro": f(cval(step, model, "LORO", "macro", "MAE")),
            "LORO R2 macro": f(cval(step, model, "LORO", "macro", "R2"), 3),
        }
        if model == "fused":
            row.update(
                {
                    "age-block pooled MAE": f(
                        cval(step, model, "AGEBLOCK_12h", "pooled fold-mean", "MAE")
                    ),
                    "age-block excl. block 1": f(
                        cval(step, model, "AGEBLOCK_12h", "pooled fold-mean excl. block 1", "MAE")
                    ),
                    "season summer->fall MAE": f(
                        cval(step, model, "SEASON_12h", "train_summer_test_fall", "MAE")
                    ),
                    "season fall->summer MAE": f(
                        cval(step, model, "SEASON_12h", "train_fall_test_summer", "MAE")
                    ),
                }
            )
        rows.append(row)


def from_core(label, v, model_label, fused=True):
    row = {
        "stage": label,
        "model": model_label,
        **{f"LORO MAE {r}": f(val("S03_loro", "MAE", v, r)) for r in ROOMS},
        "LORO MAE macro": f(val("S03_loro", "MAE", v, "macro")),
        "LORO R2 macro": f(val("S03_loro", "R2", v, "macro"), 3),
    }
    a = (
        S[
            (S.section == "S05_ageblock")
            & (S.variant == v)
            & (S.detail == "AGEBLOCK_12h")
            & (S.metric == "MAE")
        ]
        .set_index("room_or_fold")
        .value
    )
    s_ = (
        S[
            (S.section == "S06_season")
            & (S.variant == v)
            & (S.detail == "SEASON_12h")
            & (S.metric == "MAE")
        ]
        .set_index("room_or_fold")
        .value
    )
    row.update(
        {
            "age-block pooled MAE": f(a.get("pooled fold-mean")),
            "age-block excl. block 1": f(a.get("pooled fold-mean excl. block 1")),
            "season summer->fall MAE": f(s_.get("train_summer_test_fall")),
            "season fall->summer MAE": f(s_.get("train_fall_test_summer")),
        }
    )
    return row


rows += [
    from_core("Step 3 = primary (time-based slope, calendar-bounded)", "primary fused", "fused"),
    from_core(
        "Step 3 = primary (time-based slope, calendar-bounded)",
        "primary morphometry-only",
        "morph_only",
    ),
    from_core(
        "V2 (audio first-week reference), labelled extension",
        "V2 fused (audio first-week reference)",
        "fused",
    ),
    from_core("V3 (body-size-only comparator), labelled extension", "V3 body-size only", "V3"),
]
save(
    "table18_cumulative",
    pd.DataFrame(rows).fillna(""),
    "Table 18. Cumulative effect of each change on leave-one-room-out MAE (days), age-block (12 h purge) and seasonal MAE; Steps 0-2 use the row-based slope, Step 3 the time-based slope",
)
pd.DataFrame([{"table": n, "caption": c_, "rows": r} for n, c_, r in LIST]).to_csv(
    OUT / "_table_list.csv", index=False
)
