"""Modality-availability regimes: score the primary model and regime-specific models on hours with different blocks observed."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import datasets as D
import models as M
import numpy as np
import pandas as pd
import preprocessing as P
from common import ROOMS

BLOCKNAME = {
    "obs_audio": "audio",
    "obs_yolo": "tracking",
    "obs_morph": "morphometry",
    "obs_flow": "optical flow",
}
BLOCKCOLS = {
    "obs_audio": ("audio_bio__", "audio_equip__"),
    "obs_yolo": ("yolo__",),
    "obs_morph": ("morph__",),
    "obs_flow": ("flow__",),
}
FLAGS = ["obs_audio", "obs_yolo", "obs_morph", "obs_flow"]
MIN_ANCHORS, MIN_TRAIN = 30, 200


def regime_name(pat):
    present = [BLOCKNAME[f] for f, v in zip(FLAGS, pat) if v]
    absent = [BLOCKNAME[f] for f, v in zip(FLAGS, pat) if not v]
    return " + ".join(present) + (
        f"  (absent: {', '.join(absent)})" if absent else "  (all four blocks)"
    )


if __name__ == "__main__":
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    M_ = P.MASTER
    pat_hours = (
        M_.groupby(FLAGS)
        .agg(hours=("hour", "size"), age_min=("age_days", "min"), age_max=("age_days", "max"))
        .reset_index()
    )
    pat_hours = pat_hours[pat_hours[FLAGS].sum(axis=1) > 0]
    per_room_h = M_.groupby(FLAGS + ["room"]).size().unstack("room", fill_value=0).reset_index()
    inv = pat_hours.merge(per_room_h, on=FLAGS).rename(columns={r: f"hours_{r}" for r in ROOMS})
    folds = {r: P.fold("LORO", r, "fused", "any") for r in ROOMS}
    anchors = folds["room2"]["Xw"]
    anchors = anchors[(anchors.window_h == D.WINDOWS_EVAL) & anchors.age_days.notna()]
    ac = (
        anchors.groupby(FLAGS + ["room"])
        .size()
        .unstack("room", fill_value=0)
        .reset_index()
        .rename(columns={r: f"anchors_{r}" for r in ROOMS})
    )
    inv = inv.merge(ac, on=FLAGS, how="left").fillna(0)
    inv["regime"] = [regime_name(tuple(int(x) for x in r)) for r in inv[FLAGS].values]
    inv["blocks_present"] = inv[FLAGS].sum(axis=1)
    inv["hours_total_share"] = inv.hours / inv.hours.sum()
    inv["anchors_total"] = inv[[f"anchors_{r}" for r in ROOMS]].sum(axis=1)
    inv["rooms_present"] = [
        ", ".join(r for r in ROOMS if x[f"hours_{r}"] > 0) for _, x in inv.iterrows()
    ]
    inv["age_coverage_d"] = [f"{a:.0f}-{b:.0f}" for a, b in zip(inv.age_min, inv.age_max)]
    res = []
    for room in ROOMS:
        dA = P.fold("LORO", room, "fused", "complete")
        mA = P.fit_fold("LORO", room)["model"]
        FCA = dA["fcols"]
        dB = folds[room]
        Xw, tr, te, FCB = dB["Xw"], dB["tr"], dB["te"], dB["fcols"]
        mB = M.hgb().fit(Xw.loc[tr, FCB], Xw.loc[tr, "age_days"])
        Etr = Xw[tr]
        Ete = Xw[te & Xw.age_days.notna()]
        for _, x in inv.iterrows():
            pat = tuple(int(x[f]) for f in FLAGS)
            sel = np.all([(Ete[f] == v).values for f, v in zip(FLAGS, pat)], axis=0)
            q = Ete[sel]
            if len(q) < MIN_ANCHORS:
                continue
            row = dict(regime=x.regime, heldout=room, n_anchors=len(q))
            for name, model, cols in (
                ("A: primary complete-case fused", mA, FCA),
                ("B: fused trained on all hours", mB, FCB),
            ):
                pa = model.predict(q.reindex(columns=cols))
                res.append(
                    {**row, "model": name, **M.metrics(q.age_days, pa), "n_train_windows": np.nan}
                )
            blocks = [f for f, v in zip(FLAGS, pat) if v]
            cols = [
                c
                for c in FCB
                if c.endswith(("__m", "__s"))
                and any(c.startswith(p) for f in blocks for p in BLOCKCOLS[f])
            ] + ["hour_sin", "hour_cos"]
            trs = Etr[np.all([(Etr[f] == 1).values for f in blocks], axis=0)]
            if len(trs) >= MIN_TRAIN:
                mC = M.hgb().fit(trs[cols], trs.age_days)
                res.append(
                    {
                        **row,
                        "model": "C: regime-specific (present blocks only)",
                        **M.metrics(q.age_days, mC.predict(q[cols])),
                        "n_train_windows": len(trs),
                        "n_features": len(cols),
                    }
                )
            else:
                res.append(
                    {
                        **row,
                        "model": "C: regime-specific (present blocks only)",
                        "MAE": np.nan,
                        "RMSE": np.nan,
                        "R2": np.nan,
                        "n_train_windows": len(trs),
                    }
                )
    R = pd.DataFrame(res)
    macro = (
        R.groupby(["regime", "model"])
        .agg(
            MAE_macro=("MAE", "mean"),
            RMSE_macro=("RMSE", "mean"),
            R2_macro=("R2", "mean"),
            rooms_scored=("heldout", "nunique"),
            n_anchors_scored=("n_anchors", "sum"),
        )
        .reset_index()
    )
    macro.to_csv(out / "set3c_regime_results_macro.csv", index=False)
    R.to_csv(out / "set3c_regime_results_by_room.csv", index=False)
    train = {
        "A: primary complete-case fused": "trained on training-room hours with audio + tracking + morphometry (all window sizes), primary pipeline; applied unchanged, absent blocks NaN",
        "B: fused trained on all hours": "trained on all training-room hours with at least one block, all window sizes, native NaN handling",
        "C: regime-specific (present blocks only)": "trained on training-room anchors where the regime's blocks are observed, using only those blocks' features + hour of day",
    }
    inv.sort_values("hours", ascending=False).drop(columns=["age_min", "age_max"]).to_csv(
        out / "set3c_regime_inventory.csv", index=False
    )
    pd.DataFrame([{"model": k, "how_trained": v} for k, v in train.items()]).to_csv(
        out / "set3c_regime_training.csv", index=False
    )
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    print(
        inv.sort_values("hours", ascending=False)[
            ["regime", "hours", "rooms_present", "age_coverage_d", "anchors_total"]
        ].to_string(index=False)
    )
    print(macro.round(2).to_string(index=False))
