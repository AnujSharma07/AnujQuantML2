"""Trade simulator for a beginner on spot: long-only, one position at a time, 100% of the
account per trade (no leverage), fees + slippage on both sides, marked to market every bar."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

FEE = 0.0010   # Binance spot taker, per side
SLIP = 0.0003  # per side


def risk_weight(stop_dist, c_side=FEE + SLIP, risk=0.01):
    """Fraction of the account to put in one trade so that a normal stop-out, costs included, loses about `risk`
    of the account: min(1/(1+c), risk/(d + 2c)). The same formula drives the card and the sized backtest."""
    return np.minimum(1.0 / (1.0 + c_side), risk / (stop_dist + 2 * c_side))


def simulate(df: pd.DataFrame, lab: pd.DataFrame, signal: pd.Series, start, end, cost_mult=1.0, sized=False):
    """Return (bar_returns Series indexed by open_time, trades DataFrame).
    sized=False: the whole account goes into each trade (clean per-trade arithmetic).
    sized=True : each trade uses risk_weight(), so a normal stop-out costs about 1% of the account."""
    c_side = (FEE + SLIP) * cost_mult
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    times = df["open_time"]
    in_rng = ((times >= start) & (times < end)).to_numpy()
    rets = np.zeros(len(df))
    trades = []
    sig = signal.reindex(df.index).fillna(False).to_numpy()
    t = int(np.argmax(in_rng))
    last = len(in_rng) - 1 - int(np.argmax(in_rng[::-1]))
    while t <= last:
        if sig[t] and np.isfinite(lab["exit_idx"].iat[t]):
            k = int(lab["exit_idx"].iat[t])
            if k > last:
                break
            e, x = lab["entry_px"].iat[t], lab["entry_px"].iat[t] * (1 + lab["gross_ret"].iat[t])
            d_stop = 1 - lab["sl_px"].iat[t] / e
            w = float(risk_weight(d_stop, c_side)) if sized else 1.0
            prev = e
            for j in range(t + 1, k + 1):
                px = x if j == k else c[j]
                rets[j] += w * (px / prev - 1)
                prev = px
            rets[t + 1] -= w * c_side
            rets[k] -= w * c_side
            trades.append({"decision_time": times.iat[t], "entry_time": times.iat[t + 1], "exit_time": times.iat[k],
                           "outcome": lab["outcome"].iat[t], "gross": lab["gross_ret"].iat[t],
                           "net": (1 + lab["gross_ret"].iat[t]) * (1 - c_side) ** 2 - 1,
                           "sigma": (lab["tp_px"].iat[t] / e - 1) / 1.5, "weight": w})
            t = k  # decide again at the close of the exit bar
        else:
            t += 1
    r = pd.Series(rets, index=times)[in_rng]
    return r, pd.DataFrame(trades)


def buy_and_hold(df, start, end, cost_mult=1.0):
    m = ((df["open_time"] >= start) & (df["open_time"] < end)).to_numpy()
    c = df["close"][m].to_numpy()
    r = np.r_[c[0] / df["open"][m].iloc[0] - 1, c[1:] / c[:-1] - 1]
    r[0] -= (FEE + SLIP) * cost_mult
    return pd.Series(r, index=df["open_time"][m])


def metrics(bar_rets: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    daily = (1 + bar_rets).groupby(bar_rets.index.floor("D")).prod() - 1
    eq = (1 + daily).cumprod()
    years = max((daily.index[-1] - daily.index[0]).days / 365.25, 1e-9)
    sd = daily.std()
    out = {
        "total_return": float(eq.iloc[-1] - 1),
        "cagr": float(eq.iloc[-1] ** (1 / years) - 1),
        "sharpe": float(daily.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0,
        "max_dd": float((eq / eq.cummax() - 1).min()),
        "exposure": float((bar_rets != 0).mean()),
    }
    if trades is not None:
        n = len(trades)
        out.update({
            "n_trades": n,
            "trades_per_month": n / (years * 12),
            "win_rate": float((trades["outcome"] == "target").mean()) if n else np.nan,
            "stop_rate": float((trades["outcome"] == "stop").mean()) if n else np.nan,
            "avg_net_trade": float(trades["net"].mean()) if n else np.nan,
            "avg_R": float((trades["net"] / trades["sigma"]).mean()) if n else np.nan,
        })
    return out


def deflated_sharpe(daily_rets: pd.Series, trial_sharpes: list[float]) -> dict:
    """Bailey & Lopez de Prado (2014). Sharpe values here are per-day (not annualised)."""
    r = daily_rets.dropna()
    T = len(r)
    if T < 30 or r.std() == 0:
        return {"n_trials": len(trial_sharpes), "sr_daily": 0.0, "sr0_daily": None, "dsr": None,
                "note": "no trades, nothing to deflate"}
    sr = r.mean() / r.std()
    N = max(len(trial_sharpes), 1)
    var_sr = np.var(trial_sharpes, ddof=1) if N > 1 else 0.0
    g = 0.5772156649
    sr0 = np.sqrt(var_sr) * ((1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * np.e))) if N > 1 else 0.0
    skew, kurt = r.skew(), r.kurt() + 3
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return {"n_trials": N, "sr_daily": float(sr), "sr0_daily": float(sr0), "dsr": float(norm.cdf(z))}
