"""Hand-checked tests for the simulator, costs, purging, scorecard and helpers.
Every expected number here is worked out by hand in the comments.
Run: python tests/test_backtest_and_models.py   (or pytest)"""
import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from src import backtest as bt
from src.labels import triple_barrier

C = bt.FEE + bt.SLIP   # cost per side = 0.13%


def toy(prices):
    """prices: list of (open, high, low, close) 4h bars."""
    t = pd.date_range("2024-01-01", periods=len(prices), freq="4h", tz="UTC")
    o, h, l, c = map(list, zip(*prices))
    return pd.DataFrame({"open_time": t, "open": o, "high": h, "low": l, "close": c, "volume": 1.0, "taker_buy_base": 0.5})


def test_target_trade_by_hand():
    # sigma = 2%: target = 100*1.03 = 103, stop = 98. Bar 2 high 104 -> target hit on bar 2.
    df = toy([(100, 100, 100, 100), (100, 101, 99.5, 101), (101, 104, 100.5, 103.5)] + [(103, 103, 103, 103)] * 6)
    sig = pd.Series(0.02, index=df.index)
    lab = triple_barrier(df, sig)
    assert lab.loc[0, "outcome"] == "target" and lab.loc[0, "exit_idx"] == 2
    assert np.isclose(lab.loc[0, "gross_ret"], 0.03)
    s = pd.Series(False, index=df.index); s[0] = True
    r, tr = bt.simulate(df, lab, s, df.open_time.iat[0], df.open_time.iat[-1] + pd.Timedelta(hours=4))
    # net = 1.03 * (1 - 0.0013)^2 - 1 = 2.7323%
    assert np.isclose(tr["net"].iat[0], 1.03 * (1 - C) ** 2 - 1)
    # bar-by-bar marked-to-market returns must add up to the same trade (single position, additive per bar)
    assert np.isclose((1 + r).prod() - 1, 1.03 * (1 - C) ** 2 - 1, atol=2e-4)


def test_gap_through_stop_exits_at_open():
    # stop = 98; bar 2 opens at 95 (gap) -> exit at 95, not 98
    df = toy([(100, 100, 100, 100), (100, 100.5, 99, 99.5), (95, 96, 94, 95.5)] + [(95, 95, 95, 95)] * 6)
    lab = triple_barrier(df, pd.Series(0.02, index=df.index))
    assert lab.loc[0, "outcome"] == "stop" and np.isclose(lab.loc[0, "gross_ret"], -0.05)


def test_same_bar_touching_both_counts_as_stop():
    df = toy([(100, 100, 100, 100), (100, 104, 97, 100)] + [(100, 100, 100, 100)] * 6)
    lab = triple_barrier(df, pd.Series(0.02, index=df.index))
    assert lab.loc[0, "outcome"] == "stop"


def test_one_position_at_a_time():
    rng = np.random.default_rng(0)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    df = toy([(a, max(a, b) * 1.005, min(a, b) * 0.995, b) for a, b in zip(np.r_[100, p[:-1]], p)])
    lab = triple_barrier(df, pd.Series(0.02, index=df.index))
    s = pd.Series(True, index=df.index)
    _, tr = bt.simulate(df, lab, s, df.open_time.iat[0], df.open_time.iat[-1])
    # each new entry must come strictly after the previous exit
    assert (tr["entry_time"].iloc[1:].values > tr["exit_time"].iloc[:-1].values).all()


def test_buy_and_hold_by_hand():
    df = toy([(100, 100, 100, 100), (100, 110, 100, 110), (110, 121, 110, 121)])
    r = bt.buy_and_hold(df, df.open_time.iat[0], df.open_time.iat[-1] + pd.Timedelta(hours=4))
    # 100 -> 121 is +21%, minus one entry cost
    assert np.isclose((1 + r).prod(), 1.21 * (1 - C))


def test_purging_in_walk_forward():
    from src.models import EMBARGO
    from src.pipeline import run
    o = run(synthetic=True, verbose=False)
    data = o["data"]
    first_test_bar = data.index[data["open_time"] >= o["test_start"]][0]
    usable = data["exit_idx"] < first_test_bar - EMBARGO
    # the last usable training row must finish at least EMBARGO bars before the test starts
    assert data.loc[usable, "exit_idx"].max() <= first_test_bar - EMBARGO - 1


def test_scorecard_is_small_and_integer():
    from src.data import make_synthetic
    from src.features import FEATURES, build_features
    from src.scorecard import MAX_POINTS, MAX_RULES, Scorecard
    from src.vol import ewma_sigma_24h
    kl, fu = make_synthetic(edge=0.12, end="2022-06-01")
    sig = ewma_sigma_24h(kl.close)
    lab = triple_barrier(kl, sig)
    X = build_features(kl, fu)
    m = lab["outcome"].notna() & X["dist_hi_30d"].notna()
    sc = Scorecard().fit(X.loc[m, FEATURES], lab.loc[m, "outcome"])
    assert 1 <= len(sc.points) <= MAX_RULES
    assert all(abs(int(v)) <= MAX_POINTS and float(v).is_integer() for v in sc.points)
    assert any(r.startswith("taker") for r in sc.points.index)   # finds the planted edge
    P = sc.predict(X.loc[m, FEATURES].iloc[:50])
    assert np.allclose(P.sum(axis=1), 1)


def test_out_of_10_always_sums_to_10():
    from src.pipeline import out_of_10
    for p in ([0.31, 0.47, 0.22], [0.05, 0.9, 0.05], [1 / 3, 1 / 3, 1 / 3]):
        n = out_of_10(p)
        assert sum(n) == 10 and min(n) >= 0


def test_risk_sizing_caps_stop_loss_at_one_percent():
    from src.backtest import risk_weight
    from src.pipeline import _dollars
    # stop 2.6% away, costs 0.13% per side: $1,000 account
    d = _dollars(0.026, 0.039)
    w = float(risk_weight(0.026))
    assert np.isclose(d["size_for_1pct_risk_on_1000"], round(1000 * w, 2))
    assert np.isclose(d["loss_if_stop"], 10.0, atol=0.01)            # 1% of the account, fees included
    assert np.isclose(d["gain_if_target"], 1000 * w * (0.039 - 2 * C), atol=0.01)
    assert risk_weight(0.001) <= 1 / (1 + C)                          # never more than the whole account


def test_sized_backtest_scales_trade():
    df = toy([(100, 100, 100, 100), (100, 101, 99.5, 101), (101, 104, 100.5, 103.5)] + [(103, 103, 103, 103)] * 6)
    lab = triple_barrier(df, pd.Series(0.02, index=df.index))
    s = pd.Series(False, index=df.index); s[0] = True
    end = df.open_time.iat[-1] + pd.Timedelta(hours=4)
    _, tr = bt.simulate(df, lab, s, df.open_time.iat[0], end, sized=True)
    w = float(bt.risk_weight(0.02))
    assert np.isclose(tr["weight"].iat[0], w)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
