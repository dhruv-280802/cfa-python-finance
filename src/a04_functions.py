"""A04 — Functions and modules (CFA L1 PSM).

CFA idea: a function packages a calculation you will reuse — name it
well, give it inputs, return an output. A module is a file of functions
you import elsewhere (this whole repo is built that way: run.py imports
every module in src/).

Worked example: two functions every analyst writes in their first week —
holding-period return and CAGR — applied to each asset in the sample.
"""
import numpy as np


def holding_period_return(start_price, end_price):
    """Total return from start to end, as a decimal: (P1 - P0) / P0."""
    return end_price / start_price - 1


def cagr(start_price, end_price, years):
    """Compound annual growth rate: the steady yearly rate that turns
    start_price into end_price over `years` years."""
    return (end_price / start_price) ** (1 / years) - 1


def run(prices, source):
    n_years = (prices.index[-1] - prices.index[0]).days / 365.25
    results = {}
    print("A04  Functions")
    for t in prices.columns:
        hpr = holding_period_return(float(prices[t].iloc[0]),
                                    float(prices[t].iloc[-1]))
        g = cagr(float(prices[t].iloc[0]), float(prices[t].iloc[-1]),
                 n_years)
        results[t] = {"hpr": hpr, "cagr": g}
        print(f"  {t}: holding-period return {hpr:+.1%}, CAGR {g:+.2%} "
              f"over {n_years:.1f} years")
    print("  (These two functions are re-used by later modules — "
          "that is the point of writing functions.)")
    return {f"{t}_hpr": v["hpr"] for t, v in results.items()} | \
           {f"{t}_cagr": v["cagr"] for t, v in results.items()}
