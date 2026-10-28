"""Rebuild everything:  python run_all.py            (real Binance data, downloaded + cached in ./data)
                         python run_all.py --synthetic (placebo: a random walk with no edge)
                         python run_all.py --control   (positive control: synthetic market with a small planted edge)
                         python run_all.py --refresh   (re-download the latest bars)
Outputs go to ./outputs: results.json, results_appendix.pdf, dashboard.html, figures, test-year trades."""
import argparse
import json
from pathlib import Path

from src.pipeline import run
from src.analysis import run_all_checks
from src.momentum import full_study, gauge_at
from src.report import dashboard, figures, results_pdf

ap = argparse.ArgumentParser()
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--control", action="store_true")
ap.add_argument("--refresh", action="store_true")
ap.add_argument("--csv", default=None, help="Binance kline CSV (4h), e.g. data/btc_4h_data_2018_to_2025.csv")
args = ap.parse_args()

name = "outputs_control" if args.control else "outputs_placebo" if args.synthetic else "outputs"
out = Path(__file__).parent / name
out.mkdir(exist_ok=True)
o = run(synthetic=args.synthetic or args.control, refresh=args.refresh, control_edge=0.12 if args.control else 0.0, csv=args.csv)
if args.csv:
    h1 = Path(args.csv).with_name(Path(args.csv).name.replace("4h", "1h"))
    o["results"]["luck_checks"] = run_all_checks(o, str(h1) if h1.exists() else None)
    trend, d_daily, W = full_study(o["df"], o["end"])
    o["results"]["trend"] = trend
    o["trend_daily"], o["trend_W"] = d_daily, W
    gate = o["results"]["luck_checks"]["release_gate"]["passed"]
    o["results"]["short_term_mode"] = "live" if gate else "research"
    for c in o["cards"]:
        c["trend"] = gauge_at(W["trend_weekly"], d_daily, c["time"])
        c["short_term_mode"] = o["results"]["short_term_mode"]
        if not gate and c.get("data_ok", True) and c["trade"]:
            c["headline"] = "Model leans up, but it has not passed the release test \u2014 no buy signal"
(out / "results.json").write_text(json.dumps(o["results"], indent=1, default=str))
o["trades_test"].to_csv(out / "trades_test_year.csv", index=False)
files = figures(o, out / "figures")
results_pdf(o, files, out / "results_appendix.pdf")
dashboard(o, out / "dashboard.html")
R = o["results"]
print(f"\nChosen model: {R['chosen_model']} | edge validated on dev: {R['edge_validated']}")
for k in (f"{R['chosen_model']}|all_signals|test", "always_take|-|test", "buy_hold|-|test"):
    x = R["backtests"][k]
    print(f"  {k:28s} trades={x.get('n_trades', '-')!s:>5}  CAGR={x['cagr']:+.1%}  Sharpe={x['sharpe']:+.2f}  MaxDD={x['max_dd']:.1%}")
print(f"Wrote {out}/")
