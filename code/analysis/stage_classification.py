"""Three-stage ordinal classifier (cumulative-binary HGB) under leave-one-room-out, with full reporting."""

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
from common import ROOMS
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

SEED = M.SEED
BINS = [0, 60, 110, 200]
LABELS = ["Brooding/early (<60 d)", "Growing (60-110 d)", "Mature (>110 d)"]
CENTRES = np.array([30, 85, 155.0])


def classify_fold(room, d=None):
    d = d or P.fold("LORO", room)
    Xw, FC = d["Xw"], d["fcols"]
    E2 = Xw[Xw.window_h == D.WINDOWS_EVAL].copy()
    E2["stage"] = pd.cut(E2.age_days, BINS, labels=LABELS)
    Xtr, Xte = E2[E2.room != room], E2[E2.room == room]
    ytr, yte = Xtr.stage.cat.codes, Xte.stage.cat.codes
    cum = []
    for k in range(len(LABELS) - 1):
        m = HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.07, max_depth=6, min_samples_leaf=40, random_state=SEED
        )
        m.fit(Xtr[FC], (ytr > k).astype(float))
        cum.append(np.clip(m.predict(Xte[FC]), 0, 1))
    Pr = np.zeros((len(Xte), 3))
    Pr[:, 0] = 1 - cum[0]
    Pr[:, 1] = np.clip(cum[0] - cum[1], 0, 1)
    Pr[:, 2] = cum[1]
    return yte.values, Pr.argmax(1), Xte


def summarise(y, p):
    rec = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    return dict(
        accuracy=accuracy_score(y, p),
        balanced_accuracy=balanced_accuracy_score(y, p),
        macro_F1=f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0),
        adjacent_accuracy=float((np.abs(p - y) <= 1).mean()),
        qwk=cohen_kappa_score(y, p, weights="quadratic"),
        n=len(y),
        **{f"recall_{LABELS[i]}": rec[1][i] for i in range(3)},
        **{f"precision_{LABELS[i]}": rec[0][i] for i in range(3)},
        **{f"support_{LABELS[i]}": int(rec[3][i]) for i in range(3)},
    )


if __name__ == "__main__":
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    rows, cms, ys, ps = [], [], [], []
    for room in ROOMS:
        y, p, Xte = classify_fold(room)
        ys.append(y)
        ps.append(p)
        rows.append(dict(scope=room, **summarise(y, p)))
        cm = confusion_matrix(y, p, labels=[0, 1, 2])
        cms.append(cm)
        for i, lab in enumerate(LABELS):
            for j, lab2 in enumerate(LABELS):
                pass
        pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(
            out / f"set4_confusion_{room}_counts.csv"
        )
        pd.DataFrame(
            cm / np.maximum(cm.sum(1, keepdims=True), 1), index=LABELS, columns=LABELS
        ).to_csv(out / f"set4_confusion_{room}_rownorm.csv")
    y, p = np.concatenate(ys), np.concatenate(ps)
    rows.append(dict(scope="pooled (all held-out windows)", **summarise(y, p)))
    num = [c for c in rows[0] if c != "scope"]
    rows.append(
        dict(
            scope="macro (mean of rooms)",
            **{
                c: float(np.mean([r[c] for r in rows[:3]]))
                for c in num
                if not c.startswith("support") and c != "n"
            },
            n=sum(r["n"] for r in rows[:3]),
        )
    )
    S = pd.DataFrame(rows)
    S.to_csv(out / "set4_classification_metrics.csv", index=False)
    cm = sum(cms)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(out / "set4_confusion_pooled_counts.csv")
    pd.DataFrame(cm / cm.sum(1, keepdims=True), index=LABELS, columns=LABELS).to_csv(
        out / "set4_confusion_pooled_rownorm.csv"
    )
    pd.set_option("display.width", 250)
    print(S.round(3).T.to_string())
