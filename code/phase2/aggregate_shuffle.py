"""Phase 2: aggregate the 100-permutation shuffled-label runs (code/shuffled_label.py, one process per held-out room) against the primary fused LORO result.
Usage: python3 aggregate_shuffle.py SHUFFLE_DIR CORE_PRIMARY_CSV OUT_DIR"""

import sys
from pathlib import Path

import pandas as pd

sd, core, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
S = pd.concat([pd.read_csv(sd / f"shuffle_{r}.csv") for r in ("room2", "room6", "room7")])
S.to_csv(out / "shuffled_label_permutations.csv", index=False)
c = pd.read_csv(core)
c = c[
    (c.analysis == "LORO") & (c.model == "fused") & c.metric.isin(["MAE", "RMSE", "R2"])
].pivot_table(index="fold", columns="metric", values="value")
piv = {
    m: S.pivot_table(index="permutation", columns="heldout", values=m)
    for m in ("MAE", "RMSE", "R2")
}
for m in piv:
    piv[m]["macro"] = piv[m][["room2", "room6", "room7"]].mean(axis=1)
rows = []
for sc in ("room2", "room6", "room7", "macro"):
    r = dict(scope=sc, n_permutations=len(piv["MAE"]))
    for m in ("MAE", "RMSE", "R2"):
        v = piv[m][sc]
        r.update(
            {
                f"{m}_perm_mean": v.mean(),
                f"{m}_perm_lo": v.quantile(0.025),
                f"{m}_perm_hi": v.quantile(0.975),
                f"{m}_primary": c.loc[sc, m],
            }
        )
    r["p_value_MAE"] = (1 + (piv["MAE"][sc] <= c.loc[sc, "MAE"]).sum()) / (1 + len(piv["MAE"]))
    r["p_value_R2"] = (1 + (piv["R2"][sc] >= c.loc[sc, "R2"]).sum()) / (1 + len(piv["R2"]))
    rows.append(r)
pd.DataFrame(rows).to_csv(out / "shuffled_label_summary.csv", index=False)
print(pd.DataFrame(rows).round(3).T.to_string())
