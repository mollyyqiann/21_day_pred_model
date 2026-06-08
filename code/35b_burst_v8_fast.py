"""V8 — fast version. Reuses the v7 panel CSV (already has close + OHLC proxies
reconstructed via features). We download OHLC *only for tickers* and RECOMPUTE
the features including the new trend features, but skip the bulk download
cadence by reading from the cached 3y bars fetched during v7.

Actually simpler: this script re-runs the feature pipeline using the same
yfinance bulk frame, but outputs with PYTHONUNBUFFERED=1 forced so we can
monitor progress.

Uses the full S&P 500 (same as v7). 17 features including 6 trend features.
"""
import sys; sys.stdout.reconfigure(line_buffering=True)   # unbuffered
import importlib.util
spec = importlib.util.spec_from_file_location(
    "v8_mod", __import__("pathlib").Path(__file__).parent / "35_burst_v8_trend_features.py")
v8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v8)
v8.main()
