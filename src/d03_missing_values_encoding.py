"""D03 — Missing values and one-hot encoding (CFA L2 PSM Unit 2:
"Data Collection, Wrangling and Feature Engineering in Machine
Learning").

CFA idea: real data is gappy. Financial statements genuinely have
missing cells (a line item a company stops reporting), and any
categorical column (sector, exchange, credit rating) has to become
numbers before a model can eat it. The three questions, in order:
(1) where are the gaps, (2) what does each fix cost, (3) how do
categories become columns without leaking the answer.

Worked example: (a) count the real missing cells in AAPL/MSFT annual
statements and compare drop vs median-fill; (b) build a small
ticker info table (sector, market cap, trailing P/E from yfinance)
and one-hot encode the sector — with drop_first=True, so the model
does not get a redundant "dummy variable trap" column.
"""
import numpy as np
import pandas as pd


def _statements(ticker):
    """Same fetch as D02, reused here so this module stands alone."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    fin = t.financials
    return None if fin is None or fin.empty else fin


def run(prices, source):
    print("D03  Missing values + one-hot encoding (real gaps, real fixes)")

    # ---- (a) Where are the gaps? Real statement tables ---------------
    missing_total = 0
    for ticker in ["AAPL", "MSFT"]:
        fin = _statements(ticker)
        if fin is None:
            print(f"  {ticker}: statements unavailable today")
            continue
        n_cells = int(fin.size)
        n_missing = int(fin.isna().sum().sum())
        missing_total += n_missing
        per_row = fin.isna().sum(axis=1)
        worst = per_row[per_row > 0]
        print(f"  {ticker} income statement: {n_missing} of {n_cells} "
              f"cells missing ({n_missing / n_cells:.1%}); "
              f"{len(worst)} line items affected, worst: "
              f"{worst.index[0] if len(worst) else 'none'} "
              f"({int(worst.iloc[0]) if len(worst) else 0} gaps)")

        # What each fix costs: dropping rows vs filling with the median.
        numeric = fin.apply(pd.to_numeric, errors="coerce")
        dropped = numeric.dropna()
        filled = numeric.fillna(numeric.median(numeric_only=True))
        print(f"    dropna(): {numeric.shape[0]} -> {dropped.shape[0]} "
              f"rows kept; median-fill: {filled.isna().sum().sum()} "
              f"missing cells left (median of the same line item — "
              f"never fill a price with a mean from another company)")

    # ---- (b) Categorical -> numbers: one-hot encode the sector -------
    import yfinance as yf

    rows = []
    for ticker in ["SPY", "AAPL", "MSFT", "TLT"]:
        try:
            info = yf.Ticker(ticker).info
        except Exception:  # noqa: BLE001 - info is a bonus, not the lesson
            info = {}
        rows.append({
            "ticker": ticker,
            "sector": info.get("sector", "Unknown") or "Unknown",
            "market_cap_bn": (info.get("marketCap") or np.nan) / 1e9,
            "trailing_pe": info.get("trailingPE", np.nan),
        })
    info_df = pd.DataFrame(rows)
    print("  Ticker info table:")
    print(info_df.to_string(index=False))
    print(f"  Sectors present: {sorted(info_df['sector'].unique())} — "
          f"a model cannot eat the word 'Technology', so one-hot encode:")
    encoded = pd.get_dummies(info_df, columns=["sector"],
                             prefix="sector", drop_first=True,
                             dtype=float)
    print(encoded.to_string(index=False))
    print("  drop_first=True drops one sector column: with an intercept "
          "in the regression, all sectors + intercept would be perfectly "
          "collinear (the dummy variable trap)")

    return {"stmt_missing_cells": int(missing_total),
            "info_rows": int(len(info_df)),
            "encoded_cols": int(encoded.shape[1])}
