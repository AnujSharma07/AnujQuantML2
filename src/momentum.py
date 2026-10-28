"""'Big picture' layer: time-series momentum (trend) for holding BTC, with optional volatility scaling.

Pre-specified from the literature, not tuned:
  * Signal: for each lookback N in {7, 14, 28, 56, 112} days, +1 if the trailing N-day return is positive, else 0.
    Position = average of the five signals (0, 0.2, ... 1). Averaging horizons avoids picking one N
    (Liu & Tsyvinski 2021 find BTC momentum at 1-4 weeks; Moskowitz, Ooi & Pedersen 2012 for TSMOM generally).
  * Volatility scaling (Moreira & Muir 2017): multiply by min(1, 50% / realised 30-day annualised vol).
  * Decide at the daily close (00:00 UTC) on data up to that close; trade at the next day's open.
  * Costs: 0.13% per side on the notional actually traded; no leverage, no shorting (spot beginner).
Single-lookback variants are reported only as a robustness check, never used to choose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import FEE, SLIP

LOOKBACKS = (7, 14, 28, 56, 112)
VOL_TARGET = 0.50


def daily_from_4h(df4: pd.DataFrame) -> pd.DataFrame:
    g = df4.set_index("open_time").resample("1D")
    d = pd.DataFrame({"open": g["open"].first(), "close": g["close"].last(), "n": g["open"].count()}).dropna()
    return d


def positions(d: pd.DataFrame, lookbacks=LOOKBACKS, vol_scale=True) -> pd.Series:
    c = d["close"]
    sig = sum((c / c.shift(n) - 1 > 0).astype(float) for n in lookbacks) / len(lookbacks)
    if vol_scale:
        vol = np.log(c).diff().rolling(30).std() * np.sqrt(365)
        sig = sig * np.minimum(1.0, VOL_TARGET / vol)
    warm = c.shift(max(lookbacks)).notna()
    return sig.where(warm)        # weight decided at close of day t, held from open of t+1


def backtest(d: pd.DataFrame, w: pd.Series, start, end, cost=FEE + SLIP) -> pd.Series:
    """Daily returns of holding weight w[t] from open[t+1] to open[t+2]; costs on |change in weight|."""
    o = d["open"]
    r_open = o.shift(-1) / o - 1                      # open[t] -> open[t+1]
    held = w.shift(1).fillna(0.0)                     # weight decided at close t-1 is held during day t
    turn = held.diff().abs().fillna(held.abs())
    ret = held * r_open - turn * cost
    m = (ret.index >= start) & (ret.index < end) & r_open.notna()
    return ret[m]


def stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    yrs = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    sd = r.std()
    return {"total_return": float(eq.iloc[-1] - 1), "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1),
            "sharpe": float(r.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0,
            "max_dd": float((eq / eq.cummax() - 1).min()), "vol": float(sd * np.sqrt(365))}


def study(df4: pd.DataFrame, periods: dict) -> dict:
    d = daily_from_4h(df4)
    out = {"periods": {k: [str(a), str(b)] for k, (a, b) in periods.items()}, "variants": {}}
    variants = {"buy_hold": None,
                "trend_ensemble": (LOOKBACKS, False),
                "trend_ensemble_volscaled": (LOOKBACKS, True)}
    for n in LOOKBACKS:
        variants[f"trend_{n}d"] = ((n,), False)
    for name, spec in variants.items():
        w = pd.Series(1.0, index=d.index) if spec is None else positions(d, *spec)
        out["variants"][name] = {}
        for per, (a, b) in periods.items():
            r = backtest(d, w, a, b)
            s = stats(r)
            s["avg_weight"] = float(w.shift(1)[(w.index >= a) & (w.index < b)].fillna(0).mean())
            out["variants"][name][per] = s
    return out, d


def sharpe_diff_bootstrap(r_a: pd.Series, r_b: pd.Series, n_boot=2000, block=20, seed=0) -> dict:
    """Stationary block bootstrap of Sharpe(a) - Sharpe(b) on aligned daily returns (keeps autocorrelation)."""
    x = pd.concat([r_a, r_b], axis=1).dropna().to_numpy()
    n = len(x)
    rng = np.random.default_rng(seed)
    sr = lambda m: m.mean(0) / m.std(0) * np.sqrt(365)
    obs = sr(x)
    diffs = []
    for _ in range(n_boot):
        idx = []
        while len(idx) < n:
            s = rng.integers(0, n)
            L = rng.geometric(1 / block)
            idx.extend(((s + np.arange(L)) % n).tolist())
        b = x[idx[:n]]
        v = sr(b)
        diffs.append(v[0] - v[1])
    diffs = np.array(diffs)
    return {"sharpe_a": float(obs[0]), "sharpe_b": float(obs[1]), "diff": float(obs[0] - obs[1]),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_value_diff_le_0": float((diffs <= 0).mean())}


def weekly(w: pd.Series) -> pd.Series:
    """Beginner cadence: decide only at the Sunday close, hold the whole week."""
    return w.where(w.index.dayofweek == 6).ffill()


def full_study(df4: pd.DataFrame, end) -> dict:
    """Everything the report needs: all variants x all periods, bootstrap vs buy-and-hold, Deflated Sharpe."""
    from .backtest import deflated_sharpe
    T = lambda s: pd.Timestamp(s, tz="UTC")
    P = {"early_2018_2021": (T("2018-06-01"), T("2022-01-01")), "dev_2022_2025": (T("2022-01-01"), T("2025-06-15")),
         "test_2025_2026": (T("2025-06-15"), end), "full_2018_2026": (T("2018-06-01"), end)}
    d = daily_from_4h(df4)
    W = {"buy_hold": pd.Series(1.0, index=d.index),
         "trend_daily": positions(d, LOOKBACKS, False),
         "trend_weekly": weekly(positions(d, LOOKBACKS, False)),
         "trend_daily_volscaled": positions(d, LOOKBACKS, True)}
    for n in LOOKBACKS:
        W[f"trend_{n}d_only"] = positions(d, (n,), False)
    out = {"periods": {k: [str(a.date()), str(b.date())] for k, (a, b) in P.items()}, "variants": {}, "bootstrap_vs_buy_hold": {}}
    for name, w in W.items():
        out["variants"][name] = {}
        for per, (a, b) in P.items():
            s = stats(backtest(d, w, a, b))
            held = w.shift(1).fillna(0)
            s["avg_weight"] = float(held[(held.index >= a) & (held.index < b)].mean())
            out["variants"][name][per] = s
    for name in ("trend_daily", "trend_weekly"):
        out["bootstrap_vs_buy_hold"][name] = {per: sharpe_diff_bootstrap(backtest(d, W[name], a, b), backtest(d, W["buy_hold"], a, b))
                                               for per, (a, b) in P.items()}
    a, b = P["full_2018_2026"]
    trials = [backtest(d, w, a, b) for k, w in W.items() if k != "buy_hold"]
    trials = [float(r.mean() / r.std()) for r in trials]
    out["deflated_sharpe_full"] = {k: deflated_sharpe(backtest(d, W[k], a, b), trials) for k in ("trend_daily", "trend_weekly")}
    held = W["trend_weekly"].shift(1).fillna(0)
    out["weekly_changes_per_year"] = float(((held.diff().abs() > 0) & (held.index >= a)).sum() / ((b - a).days / 365.25))
    return out, d, W


def gauge_at(W_weekly: pd.Series, d: pd.DataFrame, t) -> dict:
    """What the trend gauge says at time t: the last Sunday decision at or before t."""
    day = pd.Timestamp(t).floor("D")
    past = W_weekly[W_weekly.index < day].dropna()   # decision made at a close strictly before this day
    if past.empty:
        return {}
    dec_day = past.index[-1]
    while dec_day.dayofweek != 6 and dec_day > past.index[0]:
        dec_day -= pd.Timedelta(days=1)
    c = d["close"]
    checks = {n: bool(c.loc[:dec_day].iloc[-1] > c.loc[:dec_day].iloc[-1 - n]) for n in LOOKBACKS}
    return {"weight": float(past.iloc[-1]), "checks": {f"{n}": v for n, v in checks.items()},
            "n_up": int(sum(checks.values())), "decided_on": str(dec_day.date())}
