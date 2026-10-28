"""Models + purged, embargoed, expanding-window walk-forward.

The model predicts THREE outcomes of the trade plan: stop first / time-out / target first.
(A binary "target vs not" label lumps time-outs, which usually end near zero, with stops,
which lose a full unit of risk; that makes a sound trade look bad.)

Every fitted object (imputer, scaler, model, calibrator, outcome payoffs) is fit inside the
fold on past data only. A training row t is usable for a fold starting at bar F only if its
label is fully resolved before F minus an embargo: exit_idx[t] < F - embargo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EMBARGO = 6
CLASSES = ("stop", "expiry", "target")
CLS_MAP = {c: i for i, c in enumerate(CLASSES)}


def make_model(kind: str):
    if kind == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.05, max_iter=3000))
    if kind == "gbm":  # LightGBM-style histogram GBM (sklearn), deliberately shallow
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=250,
                                              min_samples_leaf=300, l2_regularization=5.0,
                                              random_state=0)
    raise ValueError(kind)


class BaseRate:
    def fit(self, X, y):
        self.p = np.bincount(y.astype(int), minlength=3) / len(y)
        return self

    def predict_proba(self, X):
        return np.tile(self.p, (len(X), 1))


class Calibrated:
    """Fit on the first 80% of the training window; vector-scale (multinomial Platt) on the
    last 20%, with a purge between the two parts."""

    def __init__(self, kind):
        self.kind = kind

    def fit(self, X, y, exit_idx, idx):
        y = y.astype(int)
        if self.kind == "base":
            self.model, self.cal = BaseRate().fit(X, y), None
            return self
        cut = int(len(X) * 0.8)
        fit_mask = exit_idx[:cut] < idx[cut] - EMBARGO
        self.model = make_model(self.kind).fit(X[:cut][fit_mask], y[:cut][fit_mask])
        lp = np.log(np.clip(self.model.predict_proba(X[cut:]), 1e-4, 1))
        self.cal = LogisticRegression(C=100.0, max_iter=2000).fit(lp, y[cut:])
        return self

    def predict(self, X):
        p = self.model.predict_proba(X)
        if self.cal is None:
            return p
        return self.cal.predict_proba(np.log(np.clip(p, 1e-4, 1)))


def quarterly_folds(first: str, end: pd.Timestamp, split_at: pd.Timestamp):
    starts = list(pd.date_range(pd.Timestamp(first, tz="UTC"), end, freq="QS"))
    edges = sorted(set(starts + [split_at, end]))
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def payoffs(data: pd.DataFrame) -> np.ndarray:
    """Average realised result of each outcome, in units of risk (R = return / stop distance)."""
    R = data["gross_ret"] / (data["sl_mult"] * data["sigma"])
    return np.array([R[data["outcome"] == c].mean() for c in CLASSES])


def walk_forward(data: pd.DataFrame, feats: list[str], kind: str, folds, keep_models=False):
    """Returns OOS DataFrame [p_stop, p_expiry, p_target, R_stop, R_expiry, R_target]."""
    cols = [f"p_{c}" for c in CLASSES] + [f"R_{c}" for c in CLASSES]
    out = pd.DataFrame(np.nan, index=data.index, columns=cols)
    X_all, ex, idx = data[feats].to_numpy(), data["exit_idx"].to_numpy(), data.index.to_numpy()
    y_all = data["outcome"].map(CLS_MAP).to_numpy()
    models = []
    for a, b in folds:
        te = ((data["open_time"] >= a) & (data["open_time"] < b)).to_numpy()
        if not te.any():
            continue
        tr = ex < idx[te][0] - EMBARGO
        m = Calibrated(kind).fit(X_all[tr], y_all[tr], ex[tr], idx[tr])
        out.loc[te, cols[:3]] = m.predict(X_all[te])
        out.loc[te, cols[3:]] = payoffs(data[tr])
        if keep_models:
            models.append((a, b, m, tr))
    return (out, models) if keep_models else out
