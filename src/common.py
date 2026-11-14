"""Shared data loader for the whole repo.

One rule, repeated in every module that touches market data: real Yahoo
Finance data when it is available, a clearly labelled simulated fallback
when it is not. Nothing in this repo ever presents simulated numbers as
real-market results — `load_data()` returns the source string alongside
the prices, and `run.py` prints it at the top of the output.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "results")
CACHE = os.path.join(DATA_DIR, "prices.csv")

TICKERS = ["SPY", "AAPL", "MSFT", "TLT"]
RISK_FREE_ANNUAL = 0.04  # assumption, stated wherever it is used


def load_data(tickers=TICKERS, period="5y"):
    """Download adjusted daily closes; fall back to simulated prices.

    Returns (prices_df, source). Real downloads are cached in
    data/prices.csv so re-runs don't hammer the API; simulated data is
    never cached (a later run with working internet must get real data).
    """
    if os.path.exists(CACHE):
        df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        if all(t in df.columns for t in tickers):
            return df[tickers].dropna(), "REAL (Yahoo Finance, cached)"
    try:
        import yfinance as yf

        raw = yf.download(tickers, period=period, interval="1d",
                          auto_adjust=True, progress=False)
        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) \
            else raw[["Close"]].rename(columns={"Close": tickers[0]})
        closes = closes.dropna()
        closes.index = pd.to_datetime(closes.index).tz_localize(None)
        if len(closes) < 500 or closes.isna().any().any():
            raise ValueError("download returned too little usable data")
    except Exception as exc:  # noqa: BLE001 - any download failure -> fallback
        print(f"Yahoo Finance download failed ({exc}); "
              "using SIMULATED data instead.")
        return _simulate(tickers), "SIMULATED (Yahoo Finance unreachable)"
    os.makedirs(DATA_DIR, exist_ok=True)
    closes.to_csv(CACHE)
    return closes, "REAL (Yahoo Finance)"


def _simulate(tickers, n=1255, seed=42):
    """Correlated GBMs: one common market factor plus idiosyncratic noise,
    so the assets are correlated (like real stocks) but not cointegrated.
    Fixed seed -> reproducible. ONLY used when Yahoo is unreachable."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    dt = 1 / 252
    market = rng.standard_normal(n)
    out = {}
    for i, t in enumerate(tickers):
        start = {"SPY": 450.0, "AAPL": 180.0, "MSFT": 330.0,
                 "TLT": 90.0}.get(t, 100.0)
        vol = {"SPY": 0.16, "AAPL": 0.28, "MSFT": 0.26,
               "TLT": 0.14}.get(t, 0.20)
        shocks = 0.6 * market + 0.8 * rng.standard_normal(n)
        out[t] = start * np.exp(np.cumsum(
            (0.07 - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * shocks))
    return pd.DataFrame(out, index=dates)[tickers]


def daily_returns(prices):
    """Simple daily returns, in percent-free decimal form."""
    return prices.pct_change().dropna()


def annualize(return_series):
    """Annualised mean return and volatility from a daily return series."""
    mu = float(return_series.mean() * 252)
    sigma = float(return_series.std() * np.sqrt(252))
    return mu, sigma
