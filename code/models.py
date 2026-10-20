"""Model definition, metrics and validation folds (leave-one-room-out and leave-one-age-block-out)."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0


def hgb():
    """Fixed configuration used for every feature set (no per-model tuning)."""
    return HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.06,
        max_depth=6,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=SEED,
    )


def ridge():
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))
    )


def metrics(y, p):
    return dict(
        MAE=mean_absolute_error(y, p),
        RMSE=float(np.sqrt(mean_squared_error(y, p))),
        R2=r2_score(y, p),
        n=len(y),
    )


def folds(X, protocol):
    if protocol == "LORO":
        for r in sorted(X.room.unique()):
            yield f"holdout_{r}", X.room != r, X.room == r
    else:
        edges = np.quantile(X.age_days, np.linspace(0, 1, 6))
        for i in range(5):
            lo, hi = edges[i], edges[i + 1]
            te = (X.age_days >= lo) & (X.age_days < hi if i < 4 else X.age_days <= hi)
            yield f"age_{lo:.0f}-{hi:.0f}d", ~te, te
