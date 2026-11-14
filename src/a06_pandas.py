"""A06 — pandas Series and DataFrames (CFA L1 PSM).

CFA idea: a Series is a labelled column, a DataFrame is a table of them.
Almost all finance data work is DataFrame work: describe it, slice it by
date, resample it (daily -> monthly), and compute rolling statistics.

Worked example: the four-asset price table — summary stats, monthly
returns, annual returns by calendar year, and the latest 60-day
volatility of each asset.
"""
import numpy as np


def run(prices, source):
    desc = prices.describe().loc[["mean", "min", "max"]]
    monthly = prices.resample("M").last().pct_change().dropna()
    annual = prices.resample("A").last().pct_change().dropna()
    roll_vol = prices.pct_change().rolling(60).std().iloc[-1] * np.sqrt(252)

    print("A06  pandas Series & DataFrames")
    print(f"  Price table: {prices.shape[0]} rows x {prices.shape[1]} cols, "
          f"{prices.index[0].date()} to {prices.index[-1].date()}")
    print("  Mean price by asset: " + ", ".join(
        f"{t} ${desc.loc['mean', t]:.2f}" for t in prices.columns))
    print("  Annual SPY returns by year: " + ", ".join(
        f"{d.year} {v:+.1%}" for d, v in annual["SPY"].items()))
    print("  Latest 60-day annualised vol: " + ", ".join(
        f"{t} {roll_vol[t]:.1%}" for t in prices.columns))
    print(f"  Monthly returns table: {monthly.shape[0]} months, "
          f"SPY best {monthly['SPY'].max():+.1%}, worst {monthly['SPY'].min():+.1%}")
    return {"spy_mean_price": float(desc.loc["mean", "SPY"]),
            "spy_best_month": float(monthly["SPY"].max()),
            "spy_worst_month": float(monthly["SPY"].min()),
            "spy_roll60_vol": float(roll_vol["SPY"])}
