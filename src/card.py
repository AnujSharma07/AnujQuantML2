"""Turns model numbers into what a new trader sees. Thresholds below are fixed in advance,
not tuned on results (tuning them would be another source of overfitting)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_DOCS

# Edge = expected result per trade after costs, in units of risk (R). 1R = the stop distance.
ON_R, OFF_R = 0.05, 0.02            # signal turns on at +0.05R, turns off below +0.02R (hysteresis)
TIERS = [(0.20, "High"), (0.10, "Medium"), (0.05, "Low")]

EDUCATION = [
    "A stop-loss is the price where you accept you were wrong and exit, so one bad trade stays small.",
    "\u201cNo signal\u201d is an answer too: most of the time, the best trade is no trade.",
    "Risking 1% of your account per trade means ten losses in a row still leave you with about 90%.",
    "A 55% win rate still means 45 losers out of 100. Expect losing streaks.",
    "The target is 1.5\u00d7 the stop distance, so you can be wrong more often than right and still break even.",
    "Volatility tells you how far price usually moves, not which way.",
    "Never move your stop further away after you enter. That is how small losses become big ones.",
]


def tier_of(edge: float) -> str:
    for th, name in TIERS:
        if edge >= th:
            return name
    return "None"


def hysteresis(edge: pd.Series) -> pd.Series:
    on, out = False, []
    for e in edge.to_numpy():
        if not np.isfinite(e):
            on = False
            out.append(False); continue
        on = e >= ON_R if not on else e >= OFF_R
        out.append(on)
    return pd.Series(out, index=edge.index)


def headline(state_on: bool, tier: str, regime: str) -> str:
    if regime == "stormy":
        return "Market is stormy \u2014 sit this out"
    if not state_on:
        return "No clear edge \u2014 stay out"
    return {"High": "Good setup to buy for the next 24h",
            "Medium": "Leaning up for the next 24h",
            "Low": "Leaning slightly up for the next 24h",
            "None": "Leaning slightly up for the next 24h"}[tier]


def reasons(contrib: dict[str, float], zvals: dict[str, float], k=3):
    """contrib: feature -> signed contribution to log-odds of 'target first' vs 'stop first'."""
    items = sorted(((f, c) for f, c in contrib.items() if not f.startswith("hour")),
                   key=lambda x: -abs(x[1]))[:k]
    out = []
    for f, c in items:
        hi_txt, lo_txt = FEATURE_DOCS[f][2], FEATURE_DOCS[f][3]
        txt = hi_txt if zvals.get(f, 0) >= 0 else lo_txt
        out.append({"text": txt, "direction": "supports buying" if c > 0 else "argues against buying"})
    return out


def similar_setups(Z_hist: np.ndarray, outcomes_hist: np.ndarray, z_now: np.ndarray, k=20):
    d = np.sqrt(np.nansum((Z_hist - z_now) ** 2, axis=1))
    nn = np.argsort(d)[:k]
    o = outcomes_hist[nn]
    return {"k": int(len(nn)), "target": int((o == "target").sum()),
            "stop": int((o == "stop").sum()), "expiry": int((o == "expiry").sum())}
