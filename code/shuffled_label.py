"""Shuffled-label negative control that permutes whole room-day label blocks in the training rooms."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from common import ROOMS


def permute_day_labels(Xtr, rng):
    """Whole room-day block permutation of the age labels. Returns a label vector aligned with Xtr (index order)."""
    y = np.empty(len(Xtr), dtype=float)
    day = pd.to_datetime(Xtr.hour).dt.floor("D").values
    room = Xtr.room.values
    for r in np.unique(room):
        m = room == r
        days = np.unique(day[m])
        lab = (
            pd.Series(Xtr.age_days.values[m]).groupby(day[m]).mean().reindex(days).values
        )  # one label per day
        donor = rng.permutation(len(days))  # day i receives the label of day donor[i]
        new = dict(zip(days, lab[donor]))
        y[m] = [new[d] for d in day[m]]
    return y


def assert_permutation_valid(Xtr, y):
    """Always-run checks: labels are constant within a room-day, only reorder the room's day labels, and are not the true labels."""
    day = pd.to_datetime(Xtr.hour).dt.floor("D")
    df = pd.DataFrame(
        {"room": Xtr.room.values, "day": day.values, "y": y, "true": Xtr.age_days.values}
    )
    assert (
        df.groupby(["room", "day"]).y.nunique() == 1
    ).all(), "shuffled label is not constant within a room-day block"
    for r, g in df.groupby("room"):
        true_day = g.groupby("day").true.mean().sort_values().values
        perm_day = g.groupby("day").y.first().sort_values().values
        assert np.allclose(
            true_day, perm_day
        ), f"{r}: shuffled labels are not a permutation of the room's own day labels"
    assert not np.allclose(df.y.values, df.true.values), "permutation left the labels unchanged"
    return True


if __name__ == "__main__":
    import models as M
    import preprocessing as P

    room = sys.argv[sys.argv.index("--room") + 1]
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 100
    d = P.fold("LORO", room)
    FC = d["fcols"]
    fi = ROOMS.index(room)
    Xtr, Xte = d["Xw"][d["tr"]], d["Xw"][d["te"]]
    rows = []
    for p in range(n):
        rng = np.random.default_rng([20251017, p, fi])
        y = permute_day_labels(Xtr, rng)
        assert_permutation_valid(Xtr, y)
        pred = M.hgb().fit(Xtr[FC], y).predict(Xte[FC])
        met = M.metrics(Xte.age_days, pred)
        rows.append(
            dict(
                heldout=room,
                permutation=p,
                seed=f"20251017,{p},{fi}",
                MAE=met["MAE"],
                RMSE=met["RMSE"],
                R2=met["R2"],
                n_test=met["n"],
                n_train=len(Xtr),
                pred_mean=float(pred.mean()),
            )
        )
    pd.DataFrame(rows).to_csv(out / f"shuffle_{room}.csv", index=False)
