"""Model-family comparison on the primary pipeline and HGB versus ExtraTrees under missing data."""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

np.random.seed(0)
import datasets as D
import models as M
import preprocessing as P
from scipy.optimize import curve_fit
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

OUT = Path(os.environ["PHASE2_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
SEED = M.SEED
ROOMS = P.ROOMS
TIER1 = {
    "HistGradientBoosting (reported)": ("hgb_aug", None, "none (native NaN handling)", "none"),
    "HistGradientBoosting, no window augmentation": (
        "hgb_plain",
        None,
        "none (native NaN handling)",
        "none",
    ),
    "ExtraTrees": (
        "plain",
        ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=1, random_state=SEED),
        "median, fitted on training rooms",
        "none",
    ),
    "Random forest": (
        "plain",
        RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=1, random_state=SEED),
        "median, fitted on training rooms",
        "none",
    ),
    "Single decision tree": (
        "plain",
        DecisionTreeRegressor(min_samples_leaf=20, random_state=SEED),
        "median, fitted on training rooms",
        "none",
    ),
    "Elastic net": (
        "scaled",
        ElasticNetCV(l1_ratio=[0.2, 0.5, 0.9], cv=3, random_state=SEED, max_iter=5000),
        "median, fitted on training rooms",
        "standardised, fitted on training rooms",
    ),
    "Ridge regression": (
        "scaled",
        RidgeCV(alphas=np.logspace(-3, 3, 25)),
        "median, fitted on training rooms",
        "standardised, fitted on training rooms",
    ),
    "k-nearest neighbours (k=15)": (
        "scaled",
        KNeighborsRegressor(n_neighbors=15, weights="distance"),
        "median, fitted on training rooms",
        "standardised, fitted on training rooms",
    ),
    "Support vector regression (RBF)": (
        "scaled",
        SVR(C=100, gamma="scale", epsilon=2.0),
        "median, fitted on training rooms",
        "standardised, fitted on training rooms",
    ),
    "Neural network (MLP)": (
        "scaled",
        MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=600, early_stopping=True, random_state=SEED
        ),
        "median, fitted on training rooms",
        "standardised, fitted on training rooms",
    ),
}
rows = []


def put(model, room, met, imp, sc, **kw):
    rows.append(
        dict(
            model=model,
            fold=room,
            MAE=met["MAE"],
            RMSE=met["RMSE"],
            R2=met["R2"],
            n=met["n"],
            imputation=imp,
            scaling=sc,
            **kw
        )
    )


def pipe(kind, mdl):
    return (
        make_pipeline(SimpleImputer(strategy="median"), clone(mdl))
        if kind == "plain"
        else make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clone(mdl))
    )


FOLDS = {r: P.fold("LORO", r) for r in ROOMS}
for label, (kind, mdl, imp, sc) in TIER1.items():
    for room in ROOMS:
        d = FOLDS[room]
        Xw, FC = d["Xw"], d["fcols"]
        E = Xw[Xw.window_h == D.WINDOWS_EVAL]
        if kind == "hgb_aug":
            put(label, room, P.fit_fold("LORO", room)["met"], imp, sc)
            continue
        Xtr, Xte = E[E.room != room], E[E.room == room]
        m = M.hgb() if kind == "hgb_plain" else pipe(kind, mdl)
        m.fit(Xtr[FC], Xtr.age_days)
        put(label, room, M.metrics(Xte.age_days, m.predict(Xte[FC])), imp, sc)


# Gompertz growth-curve inversion on relative body height (windows without the size feature are not scored: n is smaller)
def gompertz(t, A, b, c):
    return A * np.exp(-b * np.exp(-c * t))


SIZE = "morph__bbox_h_p50_rel__m"
for room in ROOMS:
    Xw = FOLDS[room]["Xw"]
    E = Xw[Xw.window_h == D.WINDOWS_EVAL]
    Xtr, Xte = E[E.room != room].dropna(subset=[SIZE]), E[E.room == room].dropna(subset=[SIZE])
    dtr = Xtr.groupby(Xtr.age_days.round()).agg(size=(SIZE, "median")).reset_index()
    dtr.columns = ["age", "size"]
    A_, b_, c_ = curve_fit(gompertz, dtr.age, dtr["size"], p0=[2.3, 1.2, 0.03], maxfev=20000)[0]
    y = Xte[SIZE].clip(1e-3, A_ * 0.999)
    inv = np.clip(-np.log(np.log(A_ / y) / b_) / c_, 20, 200)
    put(
        "Gompertz growth-curve inversion",
        room,
        M.metrics(Xte.age_days, inv),
        "windows without the size feature are not scored",
        "none",
    )
# Ordinal stage classifier (cumulative-binary decomposition, two HGB regressors; predicted age = fixed stage centres 30 / 85 / 155 d)
BINS = [0, 60, 110, 200]
LABELS = ["B", "G", "M"]
for room in ROOMS:
    d = FOLDS[room]
    Xw, FC = d["Xw"], d["fcols"]
    E2 = Xw[Xw.window_h == D.WINDOWS_EVAL].copy()
    E2["stage"] = pd.cut(E2.age_days, BINS, labels=LABELS)
    Xtr, Xte = E2[E2.room != room], E2[E2.room == room]
    ytr, yte = Xtr.stage.cat.codes, Xte.stage.cat.codes
    Pr = np.zeros((len(Xte), 3))
    cum = []
    for k in range(2):
        m = HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.07, max_depth=6, min_samples_leaf=40, random_state=SEED
        ).fit(Xtr[FC], (ytr > k).astype(float))
        cum.append(np.clip(m.predict(Xte[FC]), 0, 1))
    Pr[:, 0] = 1 - cum[0]
    Pr[:, 1] = np.clip(cum[0] - cum[1], 0, 1)
    Pr[:, 2] = cum[1]
    pred = Pr.argmax(1)
    put(
        "Ordinal stage classifier",
        room,
        M.metrics(Xte.age_days, np.array([30, 85, 155.0])[pred]),
        "none (native NaN handling)",
        "none",
        accuracy=accuracy_score(yte, pred),
    )
R1 = pd.DataFrame(rows)
R1.to_csv(OUT / "model_family.csv", index=False)
ET = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=1, random_state=SEED)
mr = []


def mput(subset, model, room, y, p):
    mr.append(dict(subset=subset, model=model, fold=room, **M.metrics(y, p)))


for room in ROOMS:
    # (a) complete-case: HGB (reported, augmented) and ExtraTrees (6 h windows, median imputation)
    d = FOLDS[room]
    Xw, FC = d["Xw"], d["fcols"]
    E = Xw[Xw.window_h == D.WINDOWS_EVAL]
    Xtr, Xte = E[E.room != room], E[E.room == room]
    r_ = P.fit_fold("LORO", room)
    mput(
        "(a) complete-case windows",
        "HistGradientBoosting",
        room,
        Xte.age_days,
        r_["pred"].pred.values if np.allclose(r_["pred"].y.values, Xte.age_days.values) else None,
    )
    et = make_pipeline(SimpleImputer(strategy="median"), clone(ET)).fit(Xtr[FC], Xtr.age_days)
    mput("(a) complete-case windows", "ExtraTrees", room, Xte.age_days, et.predict(Xte[FC]))
    # (b) all eligible windows (at least one modality observed in the anchor hour): train and test on every eligible window; (c) the test windows whose anchor hour has no video block
    d2 = P.fold("LORO", room, "fused", "any")
    Xw2, FC2 = d2["Xw"], d2["fcols"]
    tr2, te2 = d2["tr"], d2["te"]
    Xtr2, Xte2 = Xw2[tr2], Xw2[te2]
    h = M.hgb().fit(Xtr2[FC2], Xtr2.age_days)
    ph = h.predict(Xte2[FC2])
    E2 = Xw2[Xw2.window_h == D.WINDOWS_EVAL]
    Etr2, Ete2 = E2[E2.room != room], E2[E2.room == room]
    et2 = make_pipeline(SimpleImputer(strategy="median"), clone(ET)).fit(Etr2[FC2], Etr2.age_days)
    pe = et2.predict(Ete2[FC2])
    assert (Ete2.hour.values == Xte2.hour.values).all()
    novid = ((Xte2.obs_yolo == 0) & (Xte2.obs_morph == 0) & (Xte2.obs_flow == 0)).values
    mput("(b) all eligible windows", "HistGradientBoosting", room, Xte2.age_days, ph)
    mput("(b) all eligible windows", "ExtraTrees", room, Ete2.age_days, pe)
    mput(
        "(c) hours without video only",
        "HistGradientBoosting",
        room,
        Xte2.age_days[novid],
        ph[novid],
    )
    mput("(c) hours without video only", "ExtraTrees", room, Ete2.age_days[novid], pe[novid])
pd.DataFrame(mr).to_csv(OUT / "missing_data.csv", index=False)
