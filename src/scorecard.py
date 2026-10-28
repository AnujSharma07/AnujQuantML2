"""Points scorecard: a sparse, integer, human-readable model (in the spirit of Ustun & Rudin's
SLIM / RiskSLIM, built here with a simple L1 + rounding heuristic instead of a MIP solver).

Inside each walk-forward fold, using training data only:
 1. Each feature becomes two yes/no rules: "in the top third" / "in the bottom third"
    (thresholds = training terciles).
 2. An L1 logistic regression on 'target first' vs 'stop first' (resolved trades) picks the rules.
 3. Keep at most MAX_RULES rules, scale so the strongest is worth 3 points, round to integers.
 4. Translate each total score into an outcome mix using the training trades with that score,
    shrunk toward the overall mix when there are few of them (empirical-Bayes style).
A trader can then check the whole model by hand: add up the points, read the row of the table.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .features import FEATURE_DOCS
from .models import CLASSES, EMBARGO, payoffs

MAX_RULES, MAX_POINTS, SHRINK_N = 5, 3, 150
SKIP = {"hour_sin", "hour_cos"}   # not meaningful as a rule a person would act on


class Scorecard:
    def fit(self, X: pd.DataFrame, outcome: pd.Series):
        feats = [f for f in X.columns if f not in SKIP]
        self.lo = X[feats].quantile(1 / 3)
        self.hi = X[feats].quantile(2 / 3)
        B = self._rules(X, feats)
        res = outcome.isin(["target", "stop"]).to_numpy()
        y = (outcome[res] == "target").astype(int).to_numpy()
        lr = None
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        for C in (0.002, 0.004, 0.008, 0.016, 0.03, 0.06):   # widen until we have some rules
            lr = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=500).fit(B[res], y)
            if (np.abs(lr.coef_[0]) > 1e-6).sum() >= 3:
                break
        coef = pd.Series(lr.coef_[0], index=B.columns)
        coef = coef[coef.abs() > 1e-6].reindex(coef.abs().sort_values(ascending=False).index).dropna()[:MAX_RULES]
        pts = (coef / coef.abs().max() * MAX_POINTS).round().astype(int) if len(coef) else coef
        self.points = pts[pts != 0]
        # score -> outcome mix, shrunk to the overall mix
        s = self.score(X)
        overall = outcome.value_counts(normalize=True).reindex(CLASSES).fillna(0)
        tab = {}
        for v in range(-MAX_POINTS * MAX_RULES, MAX_POINTS * MAX_RULES + 1):
            m = s == v
            n = int(m.sum())
            mix = outcome[m].value_counts(normalize=True).reindex(CLASSES).fillna(0) if n else overall
            w = n / (n + SHRINK_N)
            tab[v] = (w * mix + (1 - w) * overall).to_numpy()
        self.table = tab
        self.counts = s.value_counts().to_dict()
        return self

    def _rules(self, X, feats=None):
        feats = feats or list(self.lo.index)
        cols = {}
        for f in feats:
            cols[f"{f}|high"] = (X[f] > self.hi[f]).astype(float)
            cols[f"{f}|low"] = (X[f] < self.lo[f]).astype(float)
        return pd.DataFrame(cols, index=X.index).fillna(0.0)

    def fired(self, X: pd.DataFrame) -> pd.DataFrame:
        B = self._rules(X)
        return B[self.points.index] if len(self.points) else B.iloc[:, :0]

    def score(self, X):
        F = self.fired(X)
        return (F * self.points).sum(axis=1).astype(int) if len(self.points) else pd.Series(0, index=X.index)

    def predict(self, X):
        s = self.score(X)
        return np.vstack([self.table[int(np.clip(v, -15, 15))] for v in s])

    def describe(self):
        rows = []
        for rule, p in self.points.items():
            f, side = rule.split("|")
            rows.append({"rule": rule, "points": int(p), "text": FEATURE_DOCS[f][2 if side == "high" else 3]})
        return rows


def walk_forward_score(data: pd.DataFrame, feats: list[str], folds):
    cols = [f"p_{c}" for c in CLASSES] + [f"R_{c}" for c in CLASSES]
    out = pd.DataFrame(np.nan, index=data.index, columns=cols)
    out["score"] = np.nan
    ex, idx = data["exit_idx"].to_numpy(), data.index.to_numpy()
    fired_rules, models = {}, []
    for a, b in folds:
        te = ((data["open_time"] >= a) & (data["open_time"] < b)).to_numpy()
        if not te.any():
            continue
        tr = ex < idx[te][0] - EMBARGO
        sc = Scorecard().fit(data.loc[tr, feats], data.loc[tr, "outcome"])
        Xte = data.loc[te, feats]
        out.loc[te, cols[:3]] = sc.predict(Xte)
        out.loc[te, cols[3:6]] = payoffs(data[tr])
        out.loc[te, "score"] = sc.score(Xte).to_numpy()
        F = sc.fired(Xte)
        desc = {d["rule"]: d for d in sc.describe()}
        for i, row in zip(data.index[te], F.to_numpy()):
            fired_rules[i] = [desc[r] for r, on in zip(F.columns, row) if on]
        models.append((a, b, sc, tr))
    return out, fired_rules, models
