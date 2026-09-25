"""Alternative models on the same leave-one-room-out folds and features.

Tier 1: other regressor families. Gradient boosting (reported) uses all window scales;
every other Tier-1 model uses the 6 h evaluation window only, because kernel and
neighbour methods do not scale to the augmented set.
Tier 2: Gompertz growth-curve inversion on relative body height; an ordinal stage
classifier (<60 / 60-110 / >110 d, cumulative-binary decomposition with two boosted
regressors); and a GRU over 24 h sequences. For the GRU, missing values are median-filled
over all rooms before sequencing.

Outputs (results/): model_comparison_folds.csv, model_comparison.csv,
stage_classifier_metrics.csv, stage_classifier_confusion.csv
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from scipy.optimize import curve_fit
from common import RESULTS, ROOMS
import datasets as D
import models as M
import preprocessing as P

SEED = M.SEED


def pipe(m):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), m)


TIER1 = {
    "Gradient boosting (reported)": ("hgb_aug", None),
    "Gradient boosting (no augmentation)": ("hgb_plain", None),
    "Random forest": ("plain", RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                                     n_jobs=-1, random_state=SEED)),
    "Extremely randomised trees": ("plain", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3,
                                                                n_jobs=-1, random_state=SEED)),
    "Single decision tree": ("plain", DecisionTreeRegressor(min_samples_leaf=20, random_state=SEED)),
    "Support vector regression (RBF)": ("scaled", SVR(C=100, gamma="scale", epsilon=2.0)),
    "k-nearest neighbours (k=15)": ("scaled", KNeighborsRegressor(n_neighbors=15, weights="distance")),
    "Elastic net": ("scaled", ElasticNetCV(l1_ratio=[.2, .5, .9], cv=3, random_state=SEED, max_iter=5000)),
    "Ridge regression": ("scaled", RidgeCV(alphas=np.logspace(-3, 3, 25))),
    "Neural network (MLP)": ("scaled", MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=600,
                                                    early_stopping=True, random_state=SEED)),
}

rows, ord_rows, cms, seq_rows = [], [], [], []
FOLDS = {r: P.fold("LORO", r) for r in ROOMS}

for label, (kind, mdl) in TIER1.items():
    for room in ROOMS:
        d = FOLDS[room]; Xw, FC = d["Xw"], d["fcols"]
        E = Xw[Xw.window_h == D.WINDOWS_EVAL]
        if kind == "hgb_aug":
            rows.append(dict(tier="1", model=label, fold=f"holdout_{room}", **P.fit_fold("LORO", room)["met"]))
            continue
        Xtr, Xte = E[E.room != room], E[E.room == room]
        if kind.startswith("hgb"):
            m = M.hgb()
        elif kind == "plain":
            m = make_pipeline(SimpleImputer(strategy="median"), clone(mdl))
        else:
            m = pipe(clone(mdl))
        m.fit(Xtr[FC], Xtr.age_days)
        rows.append(dict(tier="1", model=label, fold=f"holdout_{room}",
                         **M.metrics(Xte.age_days, m.predict(Xte[FC]))))
    print(f"  {label}", flush=True)


def gompertz(t, A, b, c):
    return A * np.exp(-b * np.exp(-c * t))


SIZE = "morph__bbox_h_p50_rel__m"
for room in ROOMS:
    Xw = FOLDS[room]["Xw"]; E = Xw[Xw.window_h == D.WINDOWS_EVAL]
    Xtr, Xte = E[E.room != room].dropna(subset=[SIZE]), E[E.room == room].dropna(subset=[SIZE])
    dtr = Xtr.groupby(Xtr.age_days.round()).agg(size=(SIZE, "median")).reset_index()
    dtr.columns = ["age", "size"]
    popt, _ = curve_fit(gompertz, dtr.age, dtr["size"], p0=[2.3, 1.2, 0.03], maxfev=20000)
    A, b, c = popt
    y = Xte[SIZE].clip(1e-3, A * 0.999)
    inv = np.clip(-np.log(np.log(A / y) / b) / c, 20, 200)
    rows.append(dict(tier="2", model="Gompertz growth-curve inversion", fold=f"holdout_{room}",
                     **M.metrics(Xte.age_days, inv)))
print("  Gompertz", flush=True)

BINS = [0, 60, 110, 200]
LABELS = ["Brooding/early (<60 d)", "Growing (60-110 d)", "Mature (>110 d)"]
for room in ROOMS:
    d = FOLDS[room]; Xw, FC = d["Xw"], d["fcols"]
    E2 = Xw[Xw.window_h == D.WINDOWS_EVAL].copy()
    E2["stage"] = pd.cut(E2.age_days, BINS, labels=LABELS)
    Xtr, Xte = E2[E2.room != room], E2[E2.room == room]
    ytr, yte = Xtr.stage.cat.codes, Xte.stage.cat.codes
    Pr = np.zeros((len(Xte), len(LABELS)))
    cum = []
    for k in range(len(LABELS) - 1):
        m = HistGradientBoostingRegressor(max_iter=250, learning_rate=.07, max_depth=6,
                                          min_samples_leaf=40, random_state=SEED)
        m.fit(Xtr[FC], (ytr > k).astype(float))
        cum.append(np.clip(m.predict(Xte[FC]), 0, 1))
    Pr[:, 0] = 1 - cum[0]
    Pr[:, 1] = np.clip(cum[0] - cum[1], 0, 1)
    Pr[:, 2] = cum[1]
    pred = Pr.argmax(1)
    row = dict(tier="2", model="Ordinal stage classifier", fold=f"holdout_{room}",
               accuracy=accuracy_score(yte, pred), adjacent_accuracy=float((np.abs(pred - yte) <= 1).mean()),
               qwk=cohen_kappa_score(yte, pred, weights="quadratic"))
    row.update(M.metrics(Xte.age_days, np.array([30, 85, 155.0])[pred]))
    ord_rows.append(row)
    cms.append(confusion_matrix(yte, pred, labels=[0, 1, 2]))
print("  ordinal", flush=True)

import torch, torch.nn as nn
torch.manual_seed(SEED)
np.random.seed(0)
SEQ, STRIDE = 24, 3


def sequences(df, feat):
    xs, ys, rooms = [], [], []
    for room, g in df.groupby("room"):
        g = g.sort_values("hour")
        V = g[feat].values.astype("float32"); A_ = g.age_days.values.astype("float32"); H = g.hour.values
        for i in range(0, len(g) - SEQ, STRIDE):
            span = (H[i + SEQ - 1] - H[i]).astype("timedelta64[h]").astype(int)
            if span > SEQ * 3:
                continue
            xs.append(V[i:i + SEQ]); ys.append(A_[i + SEQ - 1]); rooms.append(room)
    return np.array(xs), np.array(ys, dtype="float32"), np.array(rooms)


class GRUReg(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.GRU(d, 64, num_layers=1, batch_first=True)
        self.h = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.g(x)
        return self.h(o[:, -1]).squeeze(-1)


for room in ROOMS:
    d = FOLDS[room]; FC = d["fcols"]
    Eseq = d["Xw"][d["Xw"].window_h == D.WINDOWS_EVAL].copy()
    Eseq[FC] = Eseq[FC].fillna(Eseq[FC].median())
    Xs, Ys, Rs = sequences(Eseq, FC)
    tr, te = Rs != room, Rs == room
    mu = Xs[tr].reshape(-1, Xs.shape[2]).mean(0); sd = Xs[tr].reshape(-1, Xs.shape[2]).std(0) + 1e-6
    xtr = torch.tensor((Xs[tr] - mu) / sd); xte = torch.tensor((Xs[te] - mu) / sd)
    ytr = torch.tensor(Ys[tr]); yte = Ys[te]
    net = GRUReg(Xs.shape[2]); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.L1Loss(); n = len(xtr); idx = np.arange(n)
    for ep in range(40):
        net.train(); np.random.shuffle(idx)
        for i in range(0, n, 128):
            b = idx[i:i + 128]
            opt.zero_grad(); l = lossf(net(xtr[b]), ytr[b]); l.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p = net(xte).numpy()
    seq_rows.append(dict(tier="2", model="GRU sequence model (24 h)", fold=f"holdout_{room}",
                         n_sequences=int(len(Xs)), **M.metrics(yte, p)))
    print(f"  GRU {room}", flush=True)

O = pd.DataFrame(ord_rows)
ALL = pd.concat([pd.DataFrame(rows), O.drop(columns=["accuracy", "adjacent_accuracy", "qwk"]),
                 pd.DataFrame(seq_rows)], ignore_index=True)
ALL.to_csv(RESULTS / "model_comparison_folds.csv", index=False)
summ = ALL.groupby(["tier", "model"])[["MAE", "RMSE", "R2"]].mean().reset_index().sort_values("MAE")
summ.to_csv(RESULTS / "model_comparison.csv", index=False)
O[["accuracy", "adjacent_accuracy", "qwk"]].mean().to_frame("mean_of_rooms").to_csv(RESULTS / "stage_classifier_metrics.csv")
pd.DataFrame(np.sum(cms, axis=0), index=LABELS, columns=LABELS).to_csv(RESULTS / "stage_classifier_confusion.csv")
print(summ.round(3).to_string(index=False))
