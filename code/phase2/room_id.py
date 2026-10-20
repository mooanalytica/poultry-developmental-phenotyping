"""Room-identification accuracy of the audio block before and after the V2 first-week referencing."""

import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import preprocessing as P
from sklearn.ensemble import HistGradientBoostingClassifier

M = P.MASTER.drop(columns=[c for c in P.MASTER.columns if c.startswith("audio_equip__")])
raw_eq = P.AUDIO_RAW[["room", "hour"] + P.EQUIP_COLS].rename(
    columns={c: f"audio_equip__{c}" for c in P.EQUIP_COLS}
)
before = M.merge(
    P._build_audio_equip_pooled("LORO", "no_heldout_room"), on=["room", "hour"], how="left"
)
after = P.apply_audio_first_week(M.merge(raw_eq, on=["room", "hour"], how="left"), "LORO", "room2")
cc = (M.obs_audio == 1) & (M.obs_yolo == 1) & (M.obs_morph == 1)
edges = np.quantile(M.loc[cc, "age_days"], np.linspace(0, 1, 6))
blk = np.clip(np.searchsorted(edges, M.age_days, side="right") - 1, 0, 4)
rows = []
for name, frame in (
    ("before_V2 (primary: bio raw, equip own-room median/IQR)", before),
    ("after_V2 (first-week reference)", after),
):
    for sub, pref in (
        ("audio_bio", ("audio_bio__",)),
        ("audio_equip", ("audio_equip__",)),
        ("audio_all", ("audio_bio__", "audio_equip__")),
    ):
        cols = [c for c in frame.columns if c.startswith(pref)]
        X = frame.loc[cc, cols].replace([np.inf, -np.inf], np.nan)
        y = frame.loc[cc, "room"]
        b = blk[cc.values]
        acc = []
        for k in range(5):
            tr, te = b != k, b == k
            clf = HistGradientBoostingClassifier(random_state=0, max_iter=100).fit(X[tr], y[tr])
            acc.append(float((clf.predict(X[te]) == y[te]).mean()))
        rows.append(
            dict(
                version=name,
                block=sub,
                n_features=len(cols),
                room_id_accuracy=float(np.mean(acc)),
                per_block_accuracy=";".join(f"{a:.3f}" for a in acc),
                chance=1 / 3,
            )
        )
R = pd.DataFrame(rows)
R.to_csv(__import__("os").environ["PHASE2_OUT"] + "/room_id_before_after_V2.csv", index=False)
print(R.round(3).to_string(index=False))
