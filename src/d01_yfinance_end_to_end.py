"""D01 — yfinance end to end (CFA L2 PSM Unit 2: "Data Collection,
Wrangling and Feature Engineering in Machine Learning").

CFA idea: before any model, you need data — prices, corporate
financials, and the market's own view of a stock (its options chain).
The official L2 module pulls company financial statements from the
SimFin platform, which needs a free API key and a separate download.
The free, no-key practical source is yfinance: one Ticker object
gives you prices, dividends, splits, financial statements and the
live options chain. The technique is identical — download, inspect,
keep what you need — only the vendor changes. (If you do get a SimFin
key, its `sf.load()` returns the same shaped tables; the wrangling in
D02/D03 applies unchanged.)

Worked example: pull everything yfinance knows about AAPL in one
place — 1 year of prices, every dividend and split, the headline
fundamentals, the last 5 annual financial statements, and the nearest
options expiry. All numbers below are live Yahoo Finance data.
"""
import numpy as np
import pandas as pd


def _safe(label, fn):
    """Run one download; return None (and say so) instead of crashing
    the whole module when one Yahoo endpoint is down."""
    try:
        out = fn()
    except Exception as exc:  # noqa: BLE001 - one bad endpoint != dead run
        print(f"  {label}: unavailable ({exc})")
        return None
    return out


def run(prices, source):
    import yfinance as yf

    ticker = yf.Ticker("AAPL")
    print("D01  yfinance end to end — everything about one ticker (AAPL)")

    # ---- 1. Prices, dividends, splits --------------------------------
    hist = _safe("history (1y daily)",
                 lambda: ticker.history(period="1y", auto_adjust=True))
    divs = _safe("dividends", lambda: ticker.dividends)
    splits = _safe("splits", lambda: ticker.splits)
    n_days = len(hist) if hist is not None else 0
    n_div = len(divs) if divs is not None else 0
    n_split = len(splits) if splits is not None else 0
    last_div = float(divs.iloc[-1]) if n_div else np.nan
    print(f"  Prices: {n_days} trading days in the last year; "
          f"last close ${float(hist['Close'].iloc[-1]):.2f} "
          f"({hist.index[-1].date()})" if n_days else "  Prices: none")
    print(f"  Dividends: {n_div} payments on record, latest "
          f"${last_div:.2f}/share; splits: {n_split} on record "
          f"(latest: {splits.index[-1].date()} {float(splits.iloc[-1]):.0f}-for-1)"
          if n_split else "  Dividends/splits: see above")

    # ---- 2. Headline fundamentals (info dict) ------------------------
    info = _safe("info", lambda: ticker.info) or {}
    sector = info.get("sector", "n/a")
    mkt_cap = info.get("marketCap", np.nan)
    pe = info.get("trailingPE", np.nan)
    fwd_pe = info.get("forwardPE", np.nan)
    beta = info.get("beta", np.nan)
    # Dividend yield: yfinance's dividendYield is already in PERCENT in
    # this version (0.3 means 0.3%), and older versions returned a
    # decimal — so compute it from dividendRate / price when possible,
    # which is unambiguous either way.
    div_rate = info.get("dividendRate")
    last_close = float(hist["Close"].iloc[-1]) if n_days else np.nan
    if div_rate and last_close == last_close and last_close > 0:
        div_yield = float(div_rate) / last_close
    else:
        raw_yield = info.get("dividendYield", np.nan)
        div_yield = raw_yield / 100.0 if raw_yield == raw_yield \
            and raw_yield > 0.05 else raw_yield
    print(f"  Info: sector {sector}; market cap "
          f"${mkt_cap / 1e12:.2f}tn; trailing P/E {pe:.1f}, forward P/E "
          f"{fwd_pe:.1f}; beta {beta:.2f}; dividend yield "
          f"{div_yield * 100 if div_yield == div_yield else float('nan'):.2f}%")

    # ---- 3. Financial statements (annual, last 5 fiscal years) -------
    fin = _safe("income statement", lambda: ticker.financials)
    bal = _safe("balance sheet", lambda: ticker.balance_sheet)
    cfs = _safe("cash-flow statement", lambda: ticker.cashflow)
    rev = np.nan
    if fin is not None and "Total Revenue" in fin.index:
        rev = float(fin.loc["Total Revenue"].iloc[0])
        years = [c.year for c in fin.columns]
        print(f"  Income statement: {fin.shape[0]} rows x {len(years)} "
              f"fiscal years ({min(years)}-{max(years)}); latest revenue "
              f"${rev / 1e9:.1f}bn")
    if bal is not None:
        print(f"  Balance sheet: {bal.shape[0]} rows; latest total assets "
              f"${float(bal.loc['Total Assets'].iloc[0]) / 1e9:.1f}bn"
              if "Total Assets" in bal.index else
              f"  Balance sheet: {bal.shape[0]} rows")
    if cfs is not None:
        print(f"  Cash-flow statement: {cfs.shape[0]} rows")

    # ---- 4. Options chain (the market's view, right now) -------------
    expiries = _safe("option expiries", lambda: ticker.options)
    n_exp = len(expiries) if expiries else 0
    n_calls = n_puts = 0
    atm_call = np.nan
    if n_exp:
        chain = _safe("nearest-expiry chain",
                      lambda: ticker.option_chain(expiries[0]))
        if chain is not None:
            n_calls, n_puts = len(chain.calls), len(chain.puts)
            spot = float(hist["Close"].iloc[-1]) if n_days else np.nan
            atm = chain.calls.iloc[
                (chain.calls["strike"] - spot).abs().argsort()[:1]]
            atm_call = float(atm["lastPrice"].iloc[0]) \
                if len(atm) and pd.notna(atm["lastPrice"].iloc[0]) else np.nan
    print(f"  Options: {n_exp} expiries listed; nearest ({expiries[0] if n_exp else 'n/a'}): "
          f"{n_calls} calls, {n_puts} puts; ATM call last price "
          f"${atm_call:.2f}" if n_exp else "  Options: none listed")

    print("  Lesson: one Ticker object replaces five downloads — and "
          "every table above is a pandas DataFrame, so D02's wrangling "
          "applies to all of it")
    print("  How to read these fundamentals: trailing P/E ~40 means "
          "investors pay $40 for $1 of past earnings — high vs history "
          "means high growth expectations (or overvaluation); beta ~1.2 "
          "means the stock typically moves 20% more than the market; "
          "a dividend yield near 0.5% means income is NOT the reason "
          "to hold this stock — growth is. The options chain's ATM "
          "call price is the market's price for one month of upside "
          "exposure (see D05/C04 for what drives it)")
    return {"yfin_price_days": int(n_days),
            "yfin_dividends": int(n_div),
            "yfin_splits": int(n_split),
            "yfin_market_cap_tn": float(mkt_cap / 1e12)
            if mkt_cap == mkt_cap else np.nan,
            "yfin_trailing_pe": float(pe) if pe == pe else np.nan,
            "yfin_latest_revenue_bn": float(rev / 1e9)
            if rev == rev else np.nan,
            "yfin_option_expiries": int(n_exp),
            "yfin_nearest_calls": int(n_calls),
            "yfin_nearest_puts": int(n_puts)}
