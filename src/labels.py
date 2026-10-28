"""Triple-barrier labels: "if I buy at the next open, does the target hit before the stop?"

Decision at the CLOSE of bar t. Entry at the OPEN of bar t+1 (no trading on a price we
used to decide). Barriers: target = entry*(1+tp*sigma_t), stop = entry*(1-sl*sigma_t),
time limit = `horizon` bars. If a single bar touches both, we count the stop (we cannot
know the intrabar order, so we assume the worse case).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier(df: pd.DataFrame, sigma: pd.Series, tp_mult=1.5, sl_mult=1.0, horizon=6) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    s = sigma.to_numpy()
    n = len(df)
    out = {k: np.full(n, np.nan) for k in ("label", "gross_ret", "exit_idx", "entry_px", "tp_px", "sl_px")}
    outcome = np.full(n, None, dtype=object)
    for t in range(n - horizon - 1 + 1):
        if t + horizon >= n or not np.isfinite(s[t]):
            continue
        e = o[t + 1]
        tp, sl = e * (1 + tp_mult * s[t]), e * (1 - sl_mult * s[t])
        res, px, k_exit = "expiry", c[t + horizon], t + horizon
        for k in range(t + 1, t + horizon + 1):
            if l[k] <= sl:                       # stop first (conservative)
                res, px, k_exit = "stop", min(o[k], sl) if k > t + 1 else sl, k
                break
            if h[k] >= tp:
                res, px, k_exit = "target", tp, k
                break
        out["label"][t] = 1.0 if res == "target" else 0.0
        out["gross_ret"][t] = px / e - 1
        out["exit_idx"][t] = k_exit
        out["entry_px"][t], out["tp_px"][t], out["sl_px"][t] = e, tp, sl
        outcome[t] = res
    res = pd.DataFrame(out, index=df.index)
    res["outcome"] = outcome
    return res
