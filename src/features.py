"""Twenty causal features. Row t uses only bars <= t and funding settled at or before the close of t."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import BAR

# name -> (what it measures, timestamp rule, text if HIGH, text if LOW)
FEATURE_DOCS = {
    "ret_1":       ("log return, last 4h", "close[t]/close[t-1]", "Price jumped in the last 4 hours", "Price dropped in the last 4 hours"),
    "ret_6":       ("log return, last 24h", "close[t]/close[t-6]", "Price is up over the last day", "Price is down over the last day"),
    "ret_24":      ("log return, last 4 days", "close[t]/close[t-24]", "Price is up over the last few days", "Price is down over the last few days"),
    "ret_42":      ("log return, last 7 days", "close[t]/close[t-42]", "Price has risen over the past week", "Price has fallen over the past week"),
    "park_24":     ("Parkinson (high-low) volatility, 4 days", "bars t-23..t", "The market has been unusually jumpy", "The market has been unusually quiet"),
    "vol_ratio":   ("short / long volatility (2d vs 14d)", "bars t-83..t", "Volatility is picking up versus the last two weeks", "Volatility has been shrinking; big moves often follow quiet spells"),
    "atr_pct":     ("average true range, % of price", "bars t-13..t", "Typical candles are large right now", "Typical candles are small right now"),
    "dist_hi_7d":  ("distance to 7-day high, in ATRs", "bars t-41..t", "Price is pressing against its 7-day high", "Price is well below its 7-day high"),
    "dist_lo_7d":  ("distance to 7-day low, in ATRs", "bars t-41..t", "Price is well above its 7-day low", "Price is sitting near its 7-day low"),
    "dist_hi_30d": ("distance to 30-day high, in ATRs", "bars t-179..t", "Price is close to its 30-day high", "Price is far below its 30-day high"),
    "dist_lo_30d": ("distance to 30-day low, in ATRs", "bars t-179..t", "Price is far above its 30-day low", "Price is close to its 30-day low"),
    "ema_slope":   ("slope of the 8-day average, vol-scaled", "EMA(50) at t vs t-6", "The 8-day trend is pointing up", "The 8-day trend is pointing down"),
    "px_vs_ema":   ("price vs 8-day average, vol-scaled", "close[t], EMA(50)[t]", "Price is stretched above its 8-day average", "Price is stretched below its 8-day average"),
    "vol_z":       ("volume vs its 30-day norm (z-score)", "bars t-179..t", "Trading volume is unusually high", "Trading volume is unusually low"),
    "taker":       ("share of volume from aggressive buyers, 24h", "bars t-5..t", "Aggressive buyers have dominated the last day", "Aggressive sellers have dominated the last day"),
    "funding":     ("last settled perp funding rate", "settled <= close of t", "Funding is high: leveraged longs are crowded", "Funding is low or negative: shorts are crowded"),
    "funding_3d":  ("average funding, last 3 days", "9 settlements <= close of t", "Leveraged longs have been paying up for days", "Leveraged traders have leaned short for days"),
    "hour_sin":    ("time of day (sine)", "bar open time", "Time-of-day pattern", "Time-of-day pattern"),
    "hour_cos":    ("time of day (cosine)", "bar open time", "Time-of-day pattern", "Time-of-day pattern"),
    "weekend":     ("weekend flag", "bar open time", "It is the weekend (thinner market)", "It is a weekday"),
}
FEATURES = list(FEATURE_DOCS)


def build_features(df: pd.DataFrame, funding: pd.DataFrame | None) -> pd.DataFrame:
    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    r = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for k in (1, 6, 24, 42):
        f[f"ret_{k}"] = np.log(c / c.shift(k))
    hl = np.log(h / l) ** 2
    f["park_24"] = np.sqrt(hl.rolling(24).mean() / (4 * np.log(2)))
    f["vol_ratio"] = r.rolling(12).std() / r.rolling(84).std()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    f["atr_pct"] = atr / c
    for name, w in (("7d", 42), ("30d", 180)):
        f[f"dist_hi_{name}"] = (c - h.rolling(w).max()) / atr
        f[f"dist_lo_{name}"] = (c - l.rolling(w).min()) / atr
    ema = c.ewm(span=50, adjust=False, min_periods=50).mean()
    rv = r.rolling(42).std()
    f["ema_slope"] = np.log(ema / ema.shift(6)) / rv
    f["px_vs_ema"] = np.log(c / ema) / rv
    lv = np.log(v + 1)
    f["vol_z"] = (lv - lv.rolling(180).mean()) / lv.rolling(180).std()
    f["taker"] = (df["taker_buy_base"].rolling(6).sum() / v.rolling(6).sum()) - 0.5
    if funding is not None and len(funding):
        close_time = df["open_time"] + BAR
        fu = funding.sort_values("funding_time").copy()
        fu["funding_3d"] = fu["funding_rate"].rolling(9, min_periods=3).mean()
        m = pd.merge_asof(pd.DataFrame({"close_time": close_time}), fu,
                          left_on="close_time", right_on="funding_time", direction="backward")
        f["funding"] = m["funding_rate"].to_numpy()
        f["funding_3d"] = m["funding_3d"].to_numpy()
    hr = df["open_time"].dt.hour
    f["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    f["weekend"] = (df["open_time"].dt.dayofweek >= 5).astype(float)
    return f.replace([np.inf, -np.inf], np.nan)
