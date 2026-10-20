"""Daily-demeaning (diurnal) calendar-confound test: subtract each room-day mean from every sensor feature and refit."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import datasets as D
import models as M
import numpy as np
import pandas as pd
import preprocessing as P
import shap
from common import ROOMS
from scipy.stats import spearmanr

BANDS = [0, 45, 75, 105, 140, 200]
BANDL = ["<45 d", "45-75 d", "75-105 d", "105-140 d", ">140 d"]
SEED = M.SEED


def feature_columns(f):
    return D.block_cols(f, list(D.BLOCKS.keys()))


def diurnal(f):
    """Subtract each room's same-day mean from every sensor feature (room x calendar day groups)."""
    f = f.copy()
    feat = feature_columns(f)
    f[feat] = f[feat] - f.groupby([f.room, f.hour.dt.floor("D")])[feat].transform("mean")
    return f


def drift_rho(f):
    feat = feature_columns(f)
    rhos = []
    for _, g in f.groupby("room"):
        d = g.groupby(g.age_days.round())[feat].mean()
        for c in feat:
            s = d[c].dropna()
            if len(s) >= 20 and s.nunique() > 1:
                rhos.append(abs(spearmanr(s.index, s.values)[0]))
    return float(np.nanmedian(rhos))


def fold_fit(room, f):
    train_rows = f.room != room
    cand = D.block_cols(f, D.FEATURE_SETS["fused"])
    cols = [c for c in cand if f.loc[train_rows, c].notna().sum() > 50]
    Xw = D.windowise(f, cols, D.WINDOWS_TRAIN)
    fcols = [f"{c}__m" for c in cols] + [f"{c}__s" for c in cols] + ["hour_sin", "hour_cos"] + P.OBS
    fcols = [c for c in fcols if c in Xw.columns]
    tr = Xw.room != room
    te = (Xw.room == room) & (Xw.window_h == D.WINDOWS_EVAL)
    fcols = [c for c in fcols if Xw.loc[tr, c].nunique(dropna=True) >= 2]
    mdl = M.hgb().fit(Xw.loc[tr, fcols], Xw.loc[tr, "age_days"])
    Xte = Xw[te]
    return mdl, Xte, fcols


def run(out, with_raw=False):
    variants = {"diurnal": diurnal}
    if with_raw:
        variants = {"raw (no transform)": lambda f: f, **variants}
    rows, shap_rows = [], []
    for vname, tfm in variants.items():
        for room in ROOMS:
            f = tfm(P.fold("LORO", room)["f"])
            c0 = drift_rho(f)
            mdl, Xte, fcols = fold_fit(room, f)
            met = M.metrics(Xte.age_days, mdl.predict(Xte[fcols]))
            rows.append(
                dict(
                    variant=vname,
                    room=room,
                    c0_drift_rho=c0,
                    MAE=met["MAE"],
                    RMSE=met["RMSE"],
                    R2=met["R2"],
                    n=met["n"],
                )
            )
            blk = np.array([P.block_of(c) for c in fcols])
            sv = np.abs(shap.TreeExplainer(mdl).shap_values(Xte[fcols]))
            band = pd.cut(Xte.age_days, BANDS, labels=BANDL).astype(str).to_numpy()
            rec = pd.DataFrame(
                {"variant": vname, "room": room, "band": band, "total": sv.sum(axis=1)}
            )
            for b in ("morph", "audio_bio"):
                rec[b] = sv[:, blk == b].sum(axis=1)
            shap_rows.append(rec)
    R = pd.DataFrame(rows)
    R.to_csv(out / "set4_demeaning_folds.csv", index=False)
    summ = R.groupby("variant")[["c0_drift_rho", "R2", "MAE", "RMSE"]].mean().round(4)
    summ.to_csv(out / "set4_demeaning_summary.csv")
    SH = pd.concat(shap_rows, ignore_index=True)
    g = SH.groupby(["variant", "room", "band"])[["morph", "audio_bio", "total"]].sum().reset_index()
    g["morph_share_of_morph_plus_bio"] = g.morph / (g.morph + g.audio_bio)
    g["morph_share_of_total"] = g.morph / g.total
    g["audio_bio_share_of_total"] = g.audio_bio / g.total
    g.to_csv(out / "set4_demeaning_shap_by_room_band.csv", index=False)
    macro = (
        g.groupby(["variant", "band"])[
            ["morph_share_of_morph_plus_bio", "morph_share_of_total", "audio_bio_share_of_total"]
        ]
        .mean()
        .reset_index()
    )
    macro["band"] = pd.Categorical(macro.band, BANDL)
    macro = macro.sort_values(["variant", "band"])
    macro.to_csv(out / "set4_demeaning_shap_share_macro.csv", index=False)
    return R, summ, macro


if __name__ == "__main__":
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    R, summ, macro = run(out, "--raw" in sys.argv)
    pd.set_option("display.width", 250)
    print(summ.to_string())
    print(macro.round(3).to_string(index=False))
