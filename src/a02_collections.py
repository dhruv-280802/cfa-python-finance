"""A02 — Lists, dicts, sets (CFA L1 PSM: Python Programming Fundamentals).

CFA idea: a list keeps order, a dict maps names to values, a set keeps
unique items. In finance you use lists for return series, dicts for
ticker -> price lookups, and sets for "which sectors do I hold?".

Worked example: SPY's monthly returns as a list — sort them, slice the
best and worst, and use a dict to look up each ticker's latest price.
"""
import numpy as np


def run(prices, source):
    monthly = prices["SPY"].resample("M").last().pct_change().dropna()
    rets = monthly.tolist()                    # list of floats
    best = max(rets)
    worst = min(rets)
    avg = sum(rets) / len(rets)
    positive = [r for r in rets if r > 0]      # list comprehension
    latest = {t: float(prices[t].iloc[-1]) for t in prices.columns}  # dict
    sectors = {"SPY": "broad market", "AAPL": "technology",
               "MSFT": "technology", "TLT": "bonds"}
    unique_sectors = set(sectors.values())     # set: duplicates removed

    print("A02  Lists, dicts, sets")
    print(f"  {len(rets)} monthly SPY returns; average {avg:.2%}, "
          f"best {best:.2%}, worst {worst:.2%}")
    print(f"  Positive months: {len(positive)} of {len(rets)}")
    print(f"  Latest prices (dict lookup): "
          + ", ".join(f"{k} ${v:.2f}" for k, v in latest.items()))
    print(f"  Sectors held: {sorted(unique_sectors)} "
          f"(a set drops the duplicate 'technology')")
    return {"n_monthly": len(rets), "avg_monthly_return": avg,
            "best_month": best, "worst_month": worst,
            "positive_months": len(positive)}
