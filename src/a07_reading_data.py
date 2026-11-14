"""A07 — Reading financial data: CSV files and the Yahoo Finance API
(CFA L1 PSM: "obtain real-world financial data using Pandas DataReader
and Yahoo Finance API").

CFA idea: real analysis starts by getting data into a DataFrame. The two
workhorse sources are (1) a CSV file on disk and (2) an API — here the
Yahoo Finance API via yfinance, the same source this whole repo uses.

Worked example: read the bundled monthly price sample from CSV, then
compare it with the daily data the loader pulled from Yahoo.
"""
import os

import pandas as pd

SAMPLE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "sample_prices.csv")


def run(prices, source):
    sample = pd.read_csv(SAMPLE, comment="#", index_col=0, parse_dates=True)
    print("A07  Reading financial data")
    print(f"  CSV sample: {SAMPLE.split('/')[-1]} — {sample.shape[0]} "
          f"monthly rows x {sample.shape[1]} cols, "
          f"{sample.index[0].date()} to {sample.index[-1].date()}")
    print("  Latest monthly closes (CSV): " + ", ".join(
        f"{t} ${sample[t].iloc[-1]:.2f}" for t in sample.columns))
    print(f"  Daily Yahoo data: {prices.shape[0]} rows x "
          f"{prices.shape[1]} cols, {prices.index[0].date()} to "
          f"{prices.index[-1].date()}")
    print(f"  Data source for this run: {source}")
    return {"sample_rows": int(sample.shape[0]),
            "sample_last_spy": float(sample["SPY"].iloc[-1]),
            "daily_rows": int(prices.shape[0])}
