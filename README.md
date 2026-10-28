# Bitcoin assistant for a first-time trader

Roostoo Quant Engineering Intern assessment. Anuj Pavan Sharma, IIT Guwahati.

The plan is `plan.pdf`, with an editable copy in `plan.docx`. Every table and chart is in `outputs/results_appendix.pdf`. To see what the trader sees, open `outputs/dashboard.html` in any browser. It works offline.

## What it does

The assistant answers two questions in plain words.

1. **This week: how much of my Bitcoin budget should I hold?** Every Sunday it runs five trend checks: is BTC higher than it was 1, 2, 4, 8 and 16 weeks ago? Each yes means holding a fifth of the budget.
2. **Today: if I buy now with a stop and a target, what usually happens in the next 24 hours?** This is the machine-learning part. It uses triple-barrier labels and a multinomial logistic regression with a purged walk-forward. The answer is shown as "out of 10 trades like this", with the trade written in dollars for a $1,000 account and the stop sized to cost about 1%.

## Results (Binance BTCUSDT 4h, Jan 2018 to Jun 2026, after 0.10% fee + 0.03% slippage per side)

**Trend gauge, checked on Sundays only**

| Period | Buy and hold | Trend gauge | Time in Bitcoin |
|---|---|---|---|
| 2018 to 2021 (never used for any decision) | +517%, worst fall −63% | +749%, worst fall −40% | 57% |
| 2022 to mid-2025 | +128%, worst fall −67% | +109%, worst fall −34% | 53% |
| Test year, Jun 2025 to Jun 2026 | -38%, worst fall −51% | -15%, worst fall −25% | 43% |
| All of 2018 to 2026 | +778%, worst fall −77% | +1,408%, worst fall −47% | 53% |

The worst fall is smaller in every period and with every single lookback. The higher Sharpe ratio (1.06 vs 0.75) is not statistically significant: the bootstrap range is -0.09 to +0.70. I added this layer after seeing the test year, so it is also reported on 2018 to 2021, it has no tuned parameters, and every variant is in the appendix.

**Short-term ML trade check**

- **2022 to mid-2025:** 182 trades, averaging +0.10 times the amount risked, and it beat all 500 random-entry versions.
- **With 1% risk sizing:** +17% over the same period, with a worst fall of −7%.
- **Release test: failed.** The 95% lower bound on average profit is -0.02, and at double costs it averages -0.004.
- **Test year:** 5 trades, -3% with 1% sizing.

So the dashboard shows this part in **research mode**: the odds and the risk are visible, but it never says buy.

## Run it

```bash
pip install -r requirements.txt
python run_all.py --csv data/btc_4h_data_2018_to_2025.csv   # about 2 minutes, writes outputs/
python run_all.py --synthetic    # fake market with no edge: should find nothing
python run_all.py --control      # fake market with a planted edge: should find it
for t in tests/*.py; do python $t; done                      # 18 tests
```

## How it was checked

- **Data:** every 4h candle was rebuilt from the 1h file, and 18,473 of 18,474 match within 0.2%. Two corrupted rows (BTC at $108,000 on 6 July 2018) were removed. Gaps are not filled.
- **No look-ahead:** features, labels and trend weights are tested against deleting future candles. Trades enter at the next open.
- **Overlapping trades:** purging plus a one-day embargo in the walk-forward.
- **Fills:** if one candle touches both levels, it counts as a stop (7 of 18,396 cases). A gap through the stop exits at the open.
- **Costs and sizing:** costs are charged on every trade and checked again at double. The same 1%-risk sizing formula drives the card and the sized backtest.
- **Is it luck?** A random-entry benchmark, a circular-shift test on the ranking, the Deflated Sharpe Ratio, a block bootstrap against buy and hold, a weekly-block release gate, and results by year, by regime, and with the stop and target moved ±25%.

## Files

```
data/                 Binance BTCUSDT 4h and 1h candles, Jan 2018 to Jun 2026
src/                  cleaning, labels, features, volatility, models, scorecard, backtest, trend gauge, checks, report, dashboard
tests/                18 tests
notebook.ipynb        the whole study, run end to end, outputs saved
outputs/              real-data results.json, results_appendix.pdf, dashboard.html, figures, test-year trades
outputs_placebo/      the same pipeline on a fake market with no edge
outputs_control/      the same pipeline on a fake market with a small planted edge
plan.pdf, plan.docx   the 3-page plan
```
