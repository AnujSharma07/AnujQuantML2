"""Trend-layer tests. Run: python tests/test_momentum.py"""
import pathlib, sys, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from src.momentum import backtest, positions, weekly


def make(n=400, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": np.r_[100, c[:-1]], "close": c}, index=idx)


def test_weight_uses_only_past_closes():
    d = make()
    full = positions(d)
    for t in (150, 250, 399):
        part = positions(d.iloc[: t + 1])
        assert np.isclose(full.iloc[t], part.iloc[t]) or (np.isnan(full.iloc[t]) and np.isnan(part.iloc[t]))


def test_trade_happens_the_next_day():
    # weight decided at close of day t must earn open[t+1] -> open[t+2], never day t itself
    d = make()
    w = pd.Series(0.0, index=d.index); w.iloc[200] = 1.0
    r = backtest(d, w, d.index[0], d.index[-1], cost=0.0)
    o = d["open"]
    assert np.isclose(r.iloc[200], 0.0)
    assert np.isclose(r.iloc[201], o.iloc[202] / o.iloc[201] - 1)


def test_costs_charged_on_changes_only():
    d = make()
    w = pd.Series(1.0, index=d.index)
    r0 = backtest(d, w, d.index[0], d.index[-1], cost=0.0)
    r1 = backtest(d, w, d.index[0], d.index[-1], cost=0.01)
    assert np.isclose((r0 - r1).sum(), 0.01)          # one entry, never changed again


def test_weekly_changes_only_on_sundays():
    d = make()
    wk = weekly(positions(d)).dropna()
    ch = wk.diff().fillna(0) != 0
    assert (wk.index[ch].dayofweek == 6).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
