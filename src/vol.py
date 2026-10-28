"""Volatility forecasts for the next 24h (6 bars of 4h).

* EWMA: parameter-free and causal. Used for the trade barriers, so labels are fixed before
  any model is fit (a fitted vol model would make labels depend on the train/test split).
* HAR-RV (Corsi 2009) in logs: fitted on training data only, used for the trader's
  "expected 24h range" and for the calm / normal / stormy regime tag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

H = 6  # bars per 24h


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def ewma_sigma_24h(close: pd.Series, halflife_bars: int = 18) -> pd.Series:
    """sigma of the next 24h log return, using data up to and including bar t."""
    r2 = log_returns(close) ** 2
    var_bar = r2.ewm(halflife=halflife_bars, min_periods=60).mean()
    return np.sqrt(var_bar * H)


def har_components(close: pd.Series) -> pd.DataFrame:
    r2 = log_returns(close) ** 2
    rv_d = r2.rolling(H).sum()                 # last 24h
    rv_w = r2.rolling(7 * H).sum() / 7         # avg daily RV, last 7d
    rv_m = r2.rolling(30 * H).sum() / 30       # avg daily RV, last 30d
    fwd = r2.shift(-1).rolling(H).sum().shift(-(H - 1))  # RV over bars t+1..t+6 (TARGET only)
    eps = 1e-10
    return pd.DataFrame({"l_d": np.log(rv_d + eps), "l_w": np.log(rv_w + eps),
                         "l_m": np.log(rv_m + eps), "l_fwd": np.log(fwd + eps)})


class HAR:
    def fit(self, comp: pd.DataFrame):
        d = comp.dropna()
        X = np.c_[np.ones(len(d)), d[["l_d", "l_w", "l_m"]].values]
        self.beta, *_ = np.linalg.lstsq(X, d["l_fwd"].values, rcond=None)
        resid = d["l_fwd"].values - X @ self.beta
        self.s2 = resid.var()
        return self

    def predict_sigma(self, comp: pd.DataFrame) -> pd.Series:
        X = np.c_[np.ones(len(comp)), comp[["l_d", "l_w", "l_m"]].values]
        mu = X @ self.beta
        return pd.Series(np.sqrt(np.exp(mu + 0.5 * self.s2)), index=comp.index)


def qlike(realised_var, forecast_var):
    m = np.isfinite(realised_var) & np.isfinite(forecast_var) & (forecast_var > 0) & (realised_var > 0)
    rv, fv = realised_var[m], forecast_var[m]
    return float(np.mean(rv / fv - np.log(rv / fv) - 1))
