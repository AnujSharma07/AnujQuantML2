"""No-look-ahead tests. Run: python -m pytest -q   (or: python tests/test_no_lookahead.py)"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src.data import make_synthetic
from src.features import build_features
from src.labels import triple_barrier
from src.vol import ewma_sigma_24h

KL, FU = make_synthetic(start="2023-01-01", end="2024-06-01", seed=1)


def test_features_do_not_use_future():
    """Feature row t must be identical whether or not bars after t exist."""
    full = build_features(KL, FU)
    for t in (500, 1500, len(KL) - 50):
        cut = KL.iloc[: t + 1]
        fu_cut = FU[FU.funding_time <= cut.open_time.iat[-1] + pd.Timedelta(hours=4)]
        part = build_features(cut, fu_cut)
        a, b = full.iloc[t], part.iloc[t]
        assert np.allclose(a.values, b.values, equal_nan=True), (t, (a - b).abs().idxmax())


def test_sigma_does_not_use_future():
    full = ewma_sigma_24h(KL.close)
    part = ewma_sigma_24h(KL.close.iloc[:1001])
    assert np.isclose(full.iat[1000], part.iat[1000])


def test_labels_only_use_bars_t_plus_1_to_t_plus_6():
    sig = ewma_sigma_24h(KL.close)
    base = triple_barrier(KL, sig)
    t = 800
    shocked = KL.copy()
    for col in ("open", "high", "low", "close"):   # wreck the past and the far future
        shocked.loc[:t, col] *= 3.0
        shocked.loc[t + 7:, col] *= 0.2
    lab2 = triple_barrier(shocked, sig)            # sigma_t is known at t, so keep it
    assert base.loc[t, "outcome"] == lab2.loc[t, "outcome"]
    assert np.isclose(base.loc[t, "gross_ret"], lab2.loc[t, "gross_ret"])


def test_entry_is_next_open():
    sig = ewma_sigma_24h(KL.close)
    lab = triple_barrier(KL, sig)
    assert np.isclose(lab.loc[900, "entry_px"], KL.loc[901, "open"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
