"""Run every analysis from an empty cache and write the results sheet, tables and manifest.
Usage: python3 code/run_all.py [--run-dir DIR] [--sheet CSV] [--tables DIR] [--jobs N] [--no-tests] [--resume]
"""

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", default="results/run")
ap.add_argument("--sheet", default="results/results_sheet.csv")
ap.add_argument("--tables", default="results/tables")
ap.add_argument("--jobs", type=int, default=4)
ap.add_argument("--no-tests", action="store_true")
ap.add_argument(
    "--resume",
    action="store_true",
    help="skip stages that already finished in this run directory (debugging the sheet/tables/checks only; a clean run does not use it)",
)
A = ap.parse_args()
RUN = (ROOT / A.run_dir).resolve()
SHEET = (ROOT / A.sheet).resolve()
TABLES = (ROOT / A.tables).resolve()
PY = sys.executable
PRIMARY = {"SCALE_FILTER_MODE": "B_exclude", "SLOPE_MODE": "time", "SLOPE_MIN_FRAC": "0.5"}
STEP0 = {"SCALE_FILTER_MODE": "centred_retro", "SLOPE_MODE": "row", "AUDIO_REF": "none"}


def out(name):
    return str(RUN / name)


def S(name, cmd, env=None, deps=(), single_thread=False):
    return dict(name=name, cmd=[PY] + cmd, env=env or {}, deps=list(deps), single=single_thread)


STAGES = [
    S(
        "core_primary",
        ["code/phase2/core_models.py"],
        {
            **PRIMARY,
            "PHASE2_OUT": out("core_primary"),
            "STEP4_TAG": "primary",
            "AUDIO_REF": "none",
            "SETS": "fused,morph_only,morph_body_only,yolo_motion_only,flow_only,audio_bio_only,video_all",
            "AB_SETS": "fused,morph_body_only",
        },
    ),
    S(
        "core_v2",
        ["code/phase2/core_models.py"],
        {
            **PRIMARY,
            "PHASE2_OUT": out("core_v2"),
            "STEP4_TAG": "V2",
            "AUDIO_REF": "first_week",
            "SETS": "fused",
            "AB_SETS": "fused",
        },
    ),
    S(
        "hashes_primary",
        ["code/phase2/feature_hashes.py"],
        {
            **PRIMARY,
            "PHASE2_OUT": out("hashes_primary"),
            "HASH_LABEL": "primary",
            "AUDIO_REF": "none",
            "HASH_SETS": "fused,morph_only,morph_body_only",
        },
    ),
    S(
        "hashes_v2",
        ["code/phase2/feature_hashes.py"],
        {
            **PRIMARY,
            "PHASE2_OUT": out("hashes_v2"),
            "HASH_LABEL": "V2",
            "AUDIO_REF": "first_week",
            "HASH_SETS": "fused",
        },
    ),
    S("room_id", ["code/phase2/room_id.py"], {**PRIMARY, "PHASE2_OUT": out("room_id")}),
    S(
        "paired",
        [
            "code/analysis/paired_fusion.py",
            "--primary",
            out("core_primary/preds_loro_primary.csv"),
            "--v2",
            out("core_v2/preds_loro_V2.csv"),
            "--out",
            out("paired/paired_fusion.csv"),
        ],
        PRIMARY,
        deps=["core_primary", "core_v2"],
    ),
    S("detection", ["code/detection.py", "--out", out("detection"), "--boot", "2000"], PRIMARY),
    *[
        S(
            f"shuffle_{r}",
            ["code/shuffled_label.py", "--room", r, "--out", out("shuffle"), "--n", "100"],
            PRIMARY,
        )
        for r in ("room2", "room6", "room7")
    ],
    S(
        "shuffle_agg",
        [
            "code/phase2/aggregate_shuffle.py",
            out("shuffle"),
            out("core_primary/core_primary.csv"),
            out("shuffle_agg"),
        ],
        PRIMARY,
        deps=["shuffle_room2", "shuffle_room6", "shuffle_room7", "core_primary"],
    ),
    S(
        "gru",
        ["code/phase2/gru_stage.py"],
        {**PRIMARY, "PHASE2_OUT": out("gru")},
        single_thread=True,
    ),
    S("regimes", ["code/availability_regimes.py", "--out", out("regimes")], PRIMARY),
    S(
        "classification",
        ["code/analysis/stage_classification.py", "--out", out("classification")],
        PRIMARY,
    ),
    S("attribution", ["code/attribution.py", "--out", out("attribution")], PRIMARY),
    S(
        "demeaning",
        ["code/analysis/daily_demeaning.py", "--out", out("demeaning"), "--raw"],
        PRIMARY,
    ),
    S("clustering", ["code/analysis/clustering_epochs.py", "--out", out("clustering")], PRIMARY),
    S(
        "ref_step0",
        ["code/phase2/ref_loro.py"],
        {**STEP0, "PHASE2_OUT": out("ref_step0"), "REF_TAG": "step0"},
    ),
    S("attribution_step0", ["code/attribution.py", "--out", out("attribution_step0")], STEP0),
    S(
        "demeaning_step0",
        ["code/analysis/daily_demeaning.py", "--out", out("demeaning_step0")],
        STEP0,
    ),
    S(
        "clustering_step0",
        ["code/analysis/clustering_epochs.py", "--out", out("clustering_step0")],
        STEP0,
    ),
    S("data", ["code/phase2/data_description.py"], {**PRIMARY, "PHASE2_OUT": out("data")}),
    S("controls", ["code/phase2/controls_stage.py"], {**PRIMARY, "PHASE2_OUT": out("controls")}),
    S(
        "controls_thermal",
        ["code/phase2/controls_thermal_stage.py"],
        {**PRIMARY, "PHASE2_OUT": out("controls_thermal")},
    ),
    S(
        "model_family",
        ["code/phase2/model_family_stage.py"],
        {**PRIMARY, "PHASE2_OUT": out("model_family"), "AUDIO_REF": "none"},
    ),
    S(
        "et_fusion",
        ["code/phase2/et_fusion_stage.py"],
        {**PRIMARY, "PHASE2_OUT": out("et_fusion"), "AUDIO_REF": "none"},
    ),
    S(
        "valid_windows",
        ["code/phase2/valid_windows_stage.py"],
        {**PRIMARY, "PHASE2_OUT": out("valid_windows"), "AUDIO_REF": "none"},
    ),
    S("body_size", ["code/phase2/body_size_stage.py"], {**PRIMARY, "PHASE2_OUT": out("body_size")}),
    S(
        "baselines",
        ["code/phase2/baselines_stage.py"],
        {**PRIMARY, "PHASE2_OUT": out("baselines"), "AUDIO_REF": "none"},
    ),
    S(
        "cum_step0",
        ["code/phase2/cumulative_stage.py"],
        {
            "SCALE_FILTER_MODE": "centred_retro",
            "SLOPE_MODE": "row",
            "AUDIO_REF": "none",
            "PHASE2_OUT": out("cum_step0"),
            "CUM_TAG": "step0",
        },
    ),
    S(
        "cum_step1",
        ["code/phase2/cumulative_stage.py"],
        {
            "SCALE_FILTER_MODE": "B_exclude",
            "SLOPE_MODE": "row",
            "AUDIO_REF": "none",
            "PHASE2_OUT": out("cum_step1"),
            "CUM_TAG": "step1",
        },
    ),
    S(
        "cum_step2",
        ["code/phase2/cumulative_stage.py"],
        {
            "SCALE_FILTER_MODE": "B_exclude",
            "SLOPE_MODE": "row",
            "AUDIO_REF": "none",
            "PHASE2_OUT": out("cum_step2"),
            "CUM_TAG": "step2",
        },
    ),
]
STAGES += [
    S(
        "fig_data",
        ["code/phase2/derive_figure_data.py", str(RUN)],
        PRIMARY,
        deps=["core_primary", "core_v2", "detection"],
    )
]
ANALYSIS_STAGES = [s["name"] for s in STAGES]
STAGES += [
    S("sheet", ["code/phase2/build_results_sheet.py", str(RUN), str(SHEET)], deps=ANALYSIS_STAGES),
    S("tables", ["code/make_tables.py", str(SHEET), str(TABLES)], deps=["sheet"]),
    S(
        "workbook",
        [
            "code/phase2/build_final_workbook.py",
            str(SHEET),
            str(SHEET.parent / "results_sheet_final.xlsx"),
        ],
        deps=["sheet"],
    ),
]
if not A.no_tests:
    STAGES += [
        S(
            "checks",
            ["code/phase2/check_results.py", str(RUN), str(SHEET), str(TABLES)],
            deps=["tables"],
        )
    ]
CACHE = ROOT / ".cache"
if (not A.resume) and CACHE.exists() and any(CACHE.iterdir()):
    dest = ROOT / ".cache_prephase2"
    if dest.exists():
        dest = ROOT / f".cache_prephase2_{dt.datetime.now():%Y%m%d_%H%M%S}"
    shutil.move(str(CACHE), str(dest))
    print(f"moved existing cache to {dest.name}")
CACHE.mkdir(exist_ok=True)
(RUN / "logs").mkdir(parents=True, exist_ok=True)
runtimes = []


def run_stage(st):
    t0 = time.time()
    (RUN / st["name"]).mkdir(parents=True, exist_ok=True)
    if (
        A.resume
        and (RUN / st["name"] / ".done").exists()
        and st["name"] not in ("sheet", "tables", "checks")
    ):
        print(f"[skipped, finished earlier] {st['name']}", flush=True)
        return 0
    (CACHE / RUN.name / st["name"]).mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "POULTRY_CACHE": str(CACHE / RUN.name / st["name"]),
        **st["env"],
    }
    if st["single"]:
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with open(RUN / "logs" / f"{st['name']}.log", "w") as lg:
        rc = subprocess.run(
            st["cmd"], cwd=ROOT, env=env, stdout=lg, stderr=subprocess.STDOUT
        ).returncode
    rec = dict(
        stage=st["name"],
        seconds=round(time.time() - t0, 1),
        status="ok" if rc == 0 else f"FAILED ({rc})",
    )
    runtimes.append(rec)
    if rc == 0:
        (RUN / st["name"] / ".done").write_text(rec["status"])
    print(f"[{rec['status']}] {st['name']}  {rec['seconds']:.0f}s", flush=True)
    return rc


done, failed, failed_names, pending = set(), False, [], {s["name"]: s for s in STAGES}
T0 = time.time()
with cf.ThreadPoolExecutor(A.jobs) as ex:
    running = {}
    while (pending or running) and not failed:
        for n in [n for n, s in pending.items() if all(d in done for d in s["deps"])]:
            if len(running) >= A.jobs:
                break
            running[ex.submit(run_stage, pending.pop(n))] = n
        if not running:
            break
        fin, _ = cf.wait(list(running), return_when=cf.FIRST_COMPLETED)
        for fu in fin:
            n = running.pop(fu)
            if fu.result() == 0:
                done.add(n)
            else:
                failed = True
                failed_names.append(n)
total = time.time() - T0
import csv

with open(RUN / "stage_runtimes.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["stage", "seconds", "status"])
    w.writeheader()
    [w.writerow(r) for r in sorted(runtimes, key=lambda r: r["stage"])]
    w.writerow(
        dict(
            stage="TOTAL (wall clock)", seconds=round(total, 1), status="FAILED" if failed else "ok"
        )
    )
if failed and failed_names != ["checks"]:
    sys.exit(
        f"stage(s) failed: {failed_names}: see " + str(RUN / "logs") + " - not fixing silently"
    )


def sha(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


code_files = sorted(
    list((ROOT / "code").glob("*.py"))
    + list((ROOT / "code/phase2").glob("*.py"))
    + list((ROOT / "code/analysis").glob("*.py"))
)
ch = hashlib.sha256()
[ch.update(sha(f).encode()) for f in code_files]
lines = [
    f"date: {dt.datetime.now().isoformat(timespec='seconds')}",
    f"run directory: {RUN.name}",
    f"wall-clock runtime: {total/60:.1f} min",
    f"python: {sys.version.split()[0]} on {platform.platform()}",
    f"master data (used): data/master_hourly_oct15.csv sha256={sha(ROOT / 'data/master_hourly_oct15.csv')}",
    f"master data (original, untouched): data/master_hourly.csv sha256={sha(ROOT / 'data/master_hourly.csv')}",
]
lines += [
    f"audio input {p.name} sha256={sha(p)}" for p in sorted((ROOT / "data/audio").glob("*.csv"))
]
lines += [f"code hash (all code/*.py, code/phase2, code/analysis): sha256={ch.hexdigest()}"] + [
    f"  {f.relative_to(ROOT)} {sha(f)[:16]}" for f in code_files
]
lines += [
    "seeds: PYTHONHASHSEED=0; models.hgb random_state=0; numpy seed 0; torch seed per GRU seed 0-4; shuffled-label seeds default_rng([20251017, permutation, fold]); bootstrap default_rng(0)"
]
lines += ["", "feature-matrix hashes (SHA-256 of the windowed feature matrix per fold):"] + [
    l.rstrip()
    for st in ("hashes_primary", "hashes_v2")
    for f in sorted((RUN / st).glob("feature_hashes_*.txt"))
    for l in f.read_text().splitlines()
]
lines += ["", "stage runtimes (s):"] + [
    f"  {r['stage']}: {r['seconds']}" for r in sorted(runtimes, key=lambda r: r["stage"])
]
lines += ["", "package versions:"] + [
    f"  {d.metadata['Name']}=={d.version}"
    for d in sorted(md.distributions(), key=lambda d: (d.metadata["Name"] or "").lower())
]
lines += [f"results sheet sha256: {sha(SHEET)}"]
(SHEET.parent / "results_sheet_manifest.txt").write_text("\n".join(lines) + "\n")
DERIVED = [
    "core_primary/preds_loro_primary.csv",
    "core_primary/preds_ageblock_primary.csv",
    "core_v2/preds_loro_V2.csv",
    "detection/set2_window_scores.csv",
    "detection/set2_window_scores_k0.csv",
    "attribution/feature_attribution.csv",
    "clustering/set4_clustering_curves.csv",
    "fig_data/coverage_daily.csv",
    "fig_data/gap_recovery_curves.csv",
    "fig_data/mae_by_phase.csv",
    "shuffle_agg/shuffled_label_permutations.csv",
    "controls/controls.csv",
    "model_family/model_family.csv",
    "model_family/missing_data.csv",
    "et_fusion/et_fusion_preds.csv",
    "et_fusion/et_fusion_metrics.csv",
    "et_fusion/et_fusion_paired.csv",
    "valid_windows/valid_windows.csv",
    "body_size/body_size_daily.csv",
    "body_size/body_size_metrics.csv",
    "baselines/baselines.csv",
    "controls_thermal/controls_thermal.csv",
]
import csv as _csv

with open(SHEET.parent / "derived_files_register.csv", "w", newline="") as fh:
    w = _csv.writer(fh)
    w.writerow(["relative_path_in_run_dir", "sha256", "bytes", "produced_by_stage"])
    for rel in DERIVED:
        w.writerow(
            [f"{RUN.name}/{rel}", sha(RUN / rel), (RUN / rel).stat().st_size, rel.split("/")[0]]
        )
with open(SHEET.parent / "results_sheet_manifest.txt", "a") as fh:
    fh.write(
        "\nregistered derived files (read by make_figures.py):\n"
        + "\n".join(f"  {RUN.name}/{rel} sha256={sha(RUN / rel)}" for rel in DERIVED)
        + "\n"
    )
shutil.copy(RUN / "data/feature_dictionary.csv", SHEET.parent / "feature_dictionary.csv")
shutil.copy(RUN / "data/coverage.csv", SHEET.parent / "data_coverage.csv")
shutil.copy(RUN / "data/key_dates.csv", SHEET.parent / "key_dates.csv")
print(f"done in {total/60:.1f} min; sheet {SHEET} sha256 {sha(SHEET)}")
if failed:
    sys.exit("consistency checks FAILED: see results/phase2_checks.md")
