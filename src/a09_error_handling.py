"""A09 — Errors, exceptions and defensive coding (CFA L1 PSM: writing
reliable analysis code).

CFA idea: real code meets bad data — a ticker that no longer exists, a
download that returns an empty table, a division by a volatility of
zero. The professional habit is to *expect* failure: catch it, say what
went wrong, and keep going with a sensible fallback instead of crashing
at 2 a.m. during a batch run.

Worked examples: (1) fetch a delisted ticker and handle the failure;
(2) a safe Sharpe ratio that refuses to divide by zero; (3) a data
validator that rejects an empty or all-NaN price table before any math
touches it; (4) logging instead of print for audit trails.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("cfa_python.a09")


def safe_sharpe(returns, risk_free_daily=0.0):
    """Sharpe ratio that returns None (with a warning) instead of
    crashing when volatility is zero — e.g. a flat price series."""
    returns = np.asarray(returns, dtype=float)
    excess = returns - risk_free_daily
    vol = excess.std(ddof=1)
    if vol == 0 or np.isnan(vol) or len(returns) < 2:
        logger.warning("safe_sharpe: zero/undefined volatility, "
                       "returning None instead of dividing by zero")
        return None
    return float(excess.mean() / vol * np.sqrt(252))


def validate_prices(prices):
    """Refuse to analyse a price table that is empty, all-NaN, or has
    non-positive prices. Raise a clear error naming the problem."""
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("validate_prices: price table is empty")
    if prices.isna().all().all():
        raise ValueError("validate_prices: price table is all-NaN")
    if (prices <= 0).any().any():
        bad = prices.columns[(prices <= 0).any()].tolist()
        raise ValueError(f"validate_prices: non-positive prices in {bad}")
    return True


def fetch_price_or_none(ticker):
    """Try to fetch one ticker's latest close; return None on any
    failure instead of raising. The caller decides what to do next."""
    try:
        import yfinance as yf

        data = yf.download(ticker, period="5d", interval="1d",
                           auto_adjust=True, progress=False)
        if data is None or len(data) == 0:
            raise ValueError("empty download")
        closes = data["Close"]
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
        return float(closes.iloc[-1])
    except Exception as exc:  # noqa: BLE001 - the point is to catch all
        logger.warning("fetch_price_or_none: %s failed (%s)", ticker, exc)
        return None


def run(prices, source):
    print("A09  Errors & defensive coding")
    # (1) A ticker that does not exist: handled, not fatal.
    bogus = fetch_price_or_none("NOTAREALTICKER123")
    print(f"  Bogus ticker fetch -> {bogus} (None = handled gracefully)")
    real = fetch_price_or_none("SPY")
    print(f"  Real ticker fetch -> SPY ${real:.2f}" if real else
          "  Real ticker fetch -> None (offline?)")

    # (2) Safe Sharpe on real data, and on a degenerate flat series.
    rets = prices["SPY"].pct_change().dropna()
    sharpe = safe_sharpe(rets)
    flat_sharpe = safe_sharpe(np.zeros(100))
    print(f"  SPY Sharpe (5y): {sharpe:.2f}" if sharpe is not None
          else "  SPY Sharpe: None")
    print(f"  Sharpe of a flat series: {flat_sharpe} "
          f"(no crash, no infinity)")

    # (3) Validator accepts good data and rejects bad data.
    validate_prices(prices)
    print("  validate_prices(real data): passed")
    try:
        validate_prices(prices.copy() * -1)
    except ValueError as exc:
        print(f"  validate_prices(negative prices): raised ValueError "
              f"as designed -> '{exc}'")

    # (4) .get() with a default beats KeyError on a missing dict key.
    info = {"ticker": "SPY", "currency": "USD"}
    print(f"  Missing key with .get(): exchange = "
          f"{info.get('exchange', 'UNKNOWN')} (no crash)")
    return {"bogus_fetch": bogus, "spy_fetch": real,
            "spy_sharpe": sharpe, "flat_sharpe": flat_sharpe}
